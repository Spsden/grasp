//! SQLite + sqlite-vec storage backend for MenteDB.
//!
//! This crate replaces the bespoke page/WAL/HNSW/CSR storage engine with a
//! single SQLite database file:
//!
//! * `memories` stores canonical memory rows with scalar indexes.
//! * `memory_tags` stores tag memberships for deterministic filtering.
//! * `memory_vec` stores a rebuildable sqlite-vec projection.
//! * `edges` stores graph relationships as durable rows.
//! * `memory_operations` stores the write audit trail.
//! * `retrieval_traces` and `retrieval_trace_hits` store optional recall traces.
//! * `schema_meta` stores schema version and projection metadata.
//!
//! All access goes through a single `parking_lot::Mutex<Connection>`. SQLite's
//! own WAL handles crash recovery; the custom WAL/page-manager/buffer-pool are
//! gone. This is the substrate the facade will eventually talk to instead of
//! `StorageEngine` + `IndexManager` + `GraphManager`.

use std::collections::{HashMap, HashSet};
use std::os::raw::c_char;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, SpaceId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode, MenteError};
use parking_lot::{Mutex, RwLock};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::json;
use uuid::Uuid;

const SCHEMA_VERSION: usize = 3;

/// Tunable retrieval parameters.
///
/// These defaults match the original hybrid-search behavior, but keeping them
/// in a config struct makes ranking experiments visible and repeatable.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// How many candidates to fetch before final filtering.
    pub fetch_multiplier: usize,
    /// RRF smoothing constant for vector and text rank fusion.
    pub rrf_k: f32,
    /// Weight applied to the fused rank score.
    pub rrf_weight: f32,
    /// Weight applied to stored salience.
    pub salience_weight: f32,
    /// Weight applied to the recency prior.
    pub recency_weight: f32,
    /// Fixed recency prior until a richer temporal prior is added.
    pub recency_prior: f32,
    /// RRF smoothing constant for multi-query recall.
    pub multi_query_rrf_k: f32,
    /// Maximum number of retrieval traces kept when tracing is enabled.
    pub trace_retention_limit: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            fetch_multiplier: 4,
            rrf_k: 60.0,
            rrf_weight: 0.7,
            salience_weight: 0.05,
            recency_weight: 0.02,
            recency_prior: 0.5,
            multi_query_rrf_k: 60.0,
            trace_retention_limit: 200,
        }
    }
}

/// A stored retrieval trace header.
#[derive(Debug, Clone)]
pub struct RetrievalTrace {
    pub trace_id: String,
    pub query_text: Option<String>,
    pub query_embedding_dim: usize,
    pub k: usize,
    pub fetch_k: usize,
    pub tags: Vec<String>,
    pub tags_or: bool,
    pub time_range: Option<(Timestamp, Timestamp)>,
    pub candidate_count: usize,
    pub result_count: usize,
    pub created_at: Timestamp,
}

/// One final ranked hit from a stored retrieval trace.
#[derive(Debug, Clone)]
pub struct RetrievalTraceHit {
    pub trace_id: String,
    pub rank: usize,
    pub memory_id: MemoryId,
    pub score: f32,
    pub vector_rank: Option<usize>,
    pub bm25_rank: Option<usize>,
    pub salience: Option<f32>,
}

/// A write-side audit record.
#[derive(Debug, Clone)]
pub struct MemoryOperation {
    pub operation_id: String,
    pub operation_type: String,
    pub memory_id: Option<MemoryId>,
    pub source: Option<MemoryId>,
    pub target: Option<MemoryId>,
    pub payload_json: String,
    pub created_at: Timestamp,
}

/// Provenance for why a memory exists.
#[derive(Debug, Clone)]
pub struct MemorySource {
    pub source_id: String,
    pub memory_id: MemoryId,
    pub source_type: String,
    pub conversation_id: Option<String>,
    pub turn_id: Option<String>,
    pub actor_id: Option<String>,
    pub observed_at: Option<Timestamp>,
    pub extractor: Option<String>,
    pub extractor_hash: Option<String>,
    pub prompt_hash: Option<String>,
    pub payload_json: String,
    pub created_at: Timestamp,
}

impl MemorySource {
    /// Create a source record with a generated source id.
    pub fn new(memory_id: MemoryId, source_type: impl Into<String>) -> Self {
        let now = now_us();
        Self {
            source_id: Uuid::new_v4().to_string(),
            memory_id,
            source_type: source_type.into(),
            conversation_id: None,
            turn_id: None,
            actor_id: None,
            observed_at: None,
            extractor: None,
            extractor_hash: None,
            prompt_hash: None,
            payload_json: "{}".to_string(),
            created_at: now,
        }
    }
}

/// Canonical entity row used for deterministic graph and recall filters.
#[derive(Debug, Clone)]
pub struct EntityRecord {
    pub entity_id: String,
    pub entity_type: String,
    pub canonical: String,
    pub attributes_json: String,
    pub confidence: f32,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl EntityRecord {
    /// Create an entity with a generated entity id.
    pub fn new(entity_type: impl Into<String>, canonical: impl Into<String>) -> Self {
        let now = now_us();
        Self {
            entity_id: Uuid::new_v4().to_string(),
            entity_type: entity_type.into(),
            canonical: canonical.into(),
            attributes_json: "{}".to_string(),
            confidence: 1.0,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Alias for a canonical entity.
#[derive(Debug, Clone)]
pub struct EntityAlias {
    pub entity_id: String,
    pub alias: String,
    pub source: Option<String>,
    pub confidence: f32,
}

/// Link between a memory and an entity.
#[derive(Debug, Clone)]
pub struct MemoryEntityLink {
    pub memory_id: MemoryId,
    pub entity_id: String,
    pub role: Option<String>,
    pub confidence: f32,
    pub evidence: Option<String>,
}

/// A durable conversation/session container.
#[derive(Debug, Clone)]
pub struct ConversationRecord {
    pub conversation_id: String,
    pub title: Option<String>,
    pub metadata_json: String,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl ConversationRecord {
    pub fn new(conversation_id: impl Into<String>) -> Self {
        let now = now_us();
        Self {
            conversation_id: conversation_id.into(),
            title: None,
            metadata_json: "{}".to_string(),
            created_at: now,
            updated_at: now,
        }
    }
}

/// One observed event in a conversation timeline.
#[derive(Debug, Clone)]
pub struct ConversationEvent {
    pub event_id: String,
    pub conversation_id: String,
    pub turn_id: Option<String>,
    pub event_type: String,
    pub actor_id: Option<String>,
    pub content: Option<String>,
    pub payload_json: String,
    pub observed_at: Timestamp,
    pub created_at: Timestamp,
}

impl ConversationEvent {
    pub fn new(conversation_id: impl Into<String>, event_type: impl Into<String>) -> Self {
        let now = now_us();
        Self {
            event_id: Uuid::new_v4().to_string(),
            conversation_id: conversation_id.into(),
            turn_id: None,
            event_type: event_type.into(),
            actor_id: None,
            content: None,
            payload_json: "{}".to_string(),
            observed_at: now,
            created_at: now,
        }
    }
}

/// A versioned extraction attempt over raw memory or conversation data.
#[derive(Debug, Clone)]
pub struct ExtractionRun {
    pub run_id: String,
    pub source_memory_id: Option<MemoryId>,
    pub conversation_id: Option<String>,
    pub extractor: String,
    pub extractor_version: String,
    pub model: Option<String>,
    pub prompt_hash: Option<String>,
    pub config_hash: Option<String>,
    pub status: String,
    pub error: Option<String>,
    pub output_json: String,
    pub started_at: Timestamp,
    pub completed_at: Option<Timestamp>,
}

impl ExtractionRun {
    pub fn new(extractor: impl Into<String>, extractor_version: impl Into<String>) -> Self {
        Self {
            run_id: Uuid::new_v4().to_string(),
            source_memory_id: None,
            conversation_id: None,
            extractor: extractor.into(),
            extractor_version: extractor_version.into(),
            model: None,
            prompt_hash: None,
            config_hash: None,
            status: "pending".to_string(),
            error: None,
            output_json: "{}".to_string(),
            started_at: now_us(),
            completed_at: None,
        }
    }
}

/// An atomic derived claim, backed by source evidence.
#[derive(Debug, Clone)]
pub struct ClaimRecord {
    pub claim_id: String,
    pub claim_text: String,
    pub claim_type: String,
    pub subject_entity_id: Option<String>,
    pub predicate: Option<String>,
    pub object_entity_id: Option<String>,
    pub confidence: f32,
    pub status: String,
    pub valid_from: Option<Timestamp>,
    pub valid_until: Option<Timestamp>,
    pub attributes_json: String,
    pub source_run_id: Option<String>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl ClaimRecord {
    pub fn new(claim_text: impl Into<String>, claim_type: impl Into<String>) -> Self {
        let now = now_us();
        Self {
            claim_id: Uuid::new_v4().to_string(),
            claim_text: claim_text.into(),
            claim_type: claim_type.into(),
            subject_entity_id: None,
            predicate: None,
            object_entity_id: None,
            confidence: 1.0,
            status: "active".to_string(),
            valid_from: None,
            valid_until: None,
            attributes_json: "{}".to_string(),
            source_run_id: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Entity participation in a claim.
#[derive(Debug, Clone)]
pub struct ClaimEntityLink {
    pub claim_id: String,
    pub entity_id: String,
    pub role: String,
    pub confidence: f32,
}

/// Evidence span linking a claim back to canonical memory/source rows.
#[derive(Debug, Clone)]
pub struct ClaimEvidence {
    pub evidence_id: String,
    pub claim_id: String,
    pub memory_id: MemoryId,
    pub source_id: Option<String>,
    pub evidence_text: Option<String>,
    pub span_start: Option<i64>,
    pub span_end: Option<i64>,
    pub confidence: f32,
    pub created_at: Timestamp,
}

impl ClaimEvidence {
    pub fn new(claim_id: impl Into<String>, memory_id: MemoryId) -> Self {
        Self {
            evidence_id: Uuid::new_v4().to_string(),
            claim_id: claim_id.into(),
            memory_id,
            source_id: None,
            evidence_text: None,
            span_start: None,
            span_end: None,
            confidence: 1.0,
            created_at: now_us(),
        }
    }
}

/// A typed relationship between two canonical entities.
#[derive(Debug, Clone)]
pub struct EntityRelationship {
    pub relationship_id: String,
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub relation_type: String,
    pub confidence: f32,
    pub status: String,
    pub valid_from: Option<Timestamp>,
    pub valid_until: Option<Timestamp>,
    pub attributes_json: String,
    pub source_run_id: Option<String>,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

impl EntityRelationship {
    pub fn new(
        source_entity_id: impl Into<String>,
        target_entity_id: impl Into<String>,
        relation_type: impl Into<String>,
    ) -> Self {
        let now = now_us();
        Self {
            relationship_id: Uuid::new_v4().to_string(),
            source_entity_id: source_entity_id.into(),
            target_entity_id: target_entity_id.into(),
            relation_type: relation_type.into(),
            confidence: 1.0,
            status: "active".to_string(),
            valid_from: None,
            valid_until: None,
            attributes_json: "{}".to_string(),
            source_run_id: None,
            created_at: now,
            updated_at: now,
        }
    }
}

/// Evidence span linking an entity relationship back to memory/source rows.
#[derive(Debug, Clone)]
pub struct RelationshipEvidence {
    pub evidence_id: String,
    pub relationship_id: String,
    pub memory_id: MemoryId,
    pub source_id: Option<String>,
    pub evidence_text: Option<String>,
    pub span_start: Option<i64>,
    pub span_end: Option<i64>,
    pub confidence: f32,
    pub created_at: Timestamp,
}

impl RelationshipEvidence {
    pub fn new(relationship_id: impl Into<String>, memory_id: MemoryId) -> Self {
        Self {
            evidence_id: Uuid::new_v4().to_string(),
            relationship_id: relationship_id.into(),
            memory_id,
            source_id: None,
            evidence_text: None,
            span_start: None,
            span_end: None,
            confidence: 1.0,
            created_at: now_us(),
        }
    }
}

/// Map any error into `MenteError::Storage` with a human-readable message.
/// We cannot `impl From<rusqlite::Error> for MenteError` here (both are foreign
/// types, blocked by the orphan rule), so callers use `.map_err(store_err)?`.
fn store_err<E: std::fmt::Display>(e: E) -> MenteError {
    MenteError::Storage(e.to_string())
}

/// Register the `vec0` virtual table module as a SQLite auto-extension.
///
/// `sqlite-vec` 0.1.x ships only the raw C init symbol (`sqlite3_vec_init`);
/// there is no `load()` helper. We wire it through SQLite's
/// `sqlite3_auto_extension`, which is process-global and applies to every
/// connection opened *after* this call. Guarded by `Once` so repeated opens are
/// no-ops. Must run before any `Connection::open*` in this backend.
fn ensure_vec0_registered() {
    use rusqlite::ffi::{sqlite3, sqlite3_api_routines, sqlite3_auto_extension};
    use std::sync::Once;
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the documented entry point of the
        // sqlite-vec extension and has the `sqlite3_loadext_entry` signature
        // SQLite expects. The transmute mirrors the crate's own test.
        unsafe {
            type SqliteExtensionInit = unsafe extern "C" fn(
                db: *mut sqlite3,
                pz_err_msg: *mut *const c_char,
                api: *const sqlite3_api_routines,
            ) -> i32;
            let init = std::mem::transmute::<*const (), SqliteExtensionInit>(
                sqlite_vec::sqlite3_vec_init as *const (),
            );
            sqlite3_auto_extension(Some(init));
        }
    });
}

/// Serialize an `f32` embedding into the little-endian byte blob `sqlite-vec`
/// expects for `vec0` inserts and MATCH queries.
fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice::<f32, u8>(v).to_vec()
}

/// Inverse of [`embedding_to_blob`].
fn blob_to_embedding(b: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice::<u8, f32>(b).to_vec()
}

/// Return a unit-length copy of `v` (L2-normalized). We store the normalized
/// form in the `vec0` table so that L2 distance there is monotonically
/// equivalent to cosine distance, and we can recover cosine similarity from the
/// raw L2 distance without depending on the `distance_metric=` vec0 option.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Dot product of two equal-length slices. For unit vectors this is cosine
/// similarity, which is how [`Backend::vector_search_filtered`] ranks.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Convert L2 squared-distance on unit vectors into cosine similarity in
/// `[0, 1]`. Derivation: for unit vectors `a`, `b`, `|a-b|^2 = 2 - 2·(a·b)`,
/// and `a·b` is cosine similarity, so `sim = 1 - dist^2 / 2`. Clamped because
/// floating point can drift slightly outside the theoretical range.
fn l2_distance_to_similarity(l2_distance: f64) -> f32 {
    let sim = 1.0 - (l2_distance * l2_distance) / 2.0;
    sim.clamp(0.0, 1.0) as f32
}

// --- enum <-> string helpers (SQLite stores these as TEXT) -----------------

fn memory_type_str(t: MemoryType) -> &'static str {
    match t {
        MemoryType::Episodic => "Episodic",
        MemoryType::Semantic => "Semantic",
        MemoryType::Procedural => "Procedural",
        MemoryType::AntiPattern => "AntiPattern",
        MemoryType::Reasoning => "Reasoning",
        MemoryType::Correction => "Correction",
    }
}

fn parse_memory_type(s: &str) -> MemoryType {
    match s {
        "Semantic" => MemoryType::Semantic,
        "Procedural" => MemoryType::Procedural,
        "AntiPattern" => MemoryType::AntiPattern,
        "Reasoning" => MemoryType::Reasoning,
        "Correction" => MemoryType::Correction,
        _ => MemoryType::Episodic,
    }
}

/// Which side of a memory node to read edges from.
#[derive(Clone, Copy, Debug)]
pub enum Direction {
    /// Edges where the node is the source.
    Outgoing,
    /// Edges where the node is the target.
    Incoming,
    /// Both directions.
    Both,
}

fn edge_type_str(t: EdgeType) -> &'static str {
    match t {
        EdgeType::Caused => "Caused",
        EdgeType::Before => "Before",
        EdgeType::Related => "Related",
        EdgeType::Contradicts => "Contradicts",
        EdgeType::Supports => "Supports",
        EdgeType::Supersedes => "Supersedes",
        EdgeType::Derived => "Derived",
        EdgeType::PartOf => "PartOf",
    }
}

fn parse_edge_type(s: &str) -> EdgeType {
    match s {
        "Caused" => EdgeType::Caused,
        "Before" => EdgeType::Before,
        "Related" => EdgeType::Related,
        "Contradicts" => EdgeType::Contradicts,
        "Supports" => EdgeType::Supports,
        "Supersedes" => EdgeType::Supersedes,
        "Derived" => EdgeType::Derived,
        "PartOf" => EdgeType::PartOf,
        _ => EdgeType::Related,
    }
}

/// Parse a stored TEXT id back into its newtype. The id newtypes wrap `Uuid`
/// and implement `FromStr`; on a malformed value we surface a `Storage` error.
fn parse_id<I: std::str::FromStr>(s: &str) -> Result<I, MenteError>
where
    I::Err: std::fmt::Display,
{
    s.parse::<I>().map_err(store_err)
}

fn parse_optional_memory_id(value: Option<String>) -> Option<MemoryId> {
    value.and_then(|s| parse_id::<MemoryId>(&s).ok())
}

fn now_us() -> Timestamp {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as Timestamp
}

fn usize_to_i64(value: usize) -> Result<i64, MenteError> {
    i64::try_from(value).map_err(store_err)
}

fn tags_json(tags: Option<&[&str]>) -> Result<String, MenteError> {
    let tags: Vec<&str> = tags.unwrap_or_default().to_vec();
    serde_json::to_string(&tags).map_err(store_err)
}

fn parse_tags_json(value: &str) -> Vec<String> {
    serde_json::from_str(value).unwrap_or_default()
}

fn retrieval_config_json(config: &RetrievalConfig) -> Result<String, MenteError> {
    serde_json::to_string(&json!({
        "fetch_multiplier": config.fetch_multiplier,
        "rrf_k": config.rrf_k,
        "rrf_weight": config.rrf_weight,
        "salience_weight": config.salience_weight,
        "recency_weight": config.recency_weight,
        "recency_prior": config.recency_prior,
        "multi_query_rrf_k": config.multi_query_rrf_k,
        "trace_retention_limit": config.trace_retention_limit,
    }))
    .map_err(store_err)
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// SQLite-backed implementation of MenteDB's storage + vector index.
///
/// Holds a single connection guarded by a mutex. WAL mode is enabled on file
/// opens so readers don't block the (single) writer. The connection keeps the
/// `sqlite-vec` extension loaded for the lifetime of the backend.
pub struct Backend {
    conn: Mutex<Connection>,
    retrieval_config: RwLock<RetrievalConfig>,
    trace_retrieval: AtomicBool,
    /// The dimension the `memory_vec` vec0 table was created with, or `0` when
    /// no vector index exists yet (deferred until an embedder is configured).
    /// Atomic because [`Backend::ensure_vector_index`] mutates it through
    /// `&self` once the facade learns the embedding dimension.
    embedding_dim: AtomicUsize,
}

impl Backend {
    /// Open (or create) a file-backed database at `path`.
    pub fn open(path: &Path, embedding_dim: usize) -> Result<Self, MenteError> {
        Self::open_with_retrieval_config(path, embedding_dim, RetrievalConfig::default())
    }

    /// Open with explicit retrieval configuration.
    pub fn open_with_retrieval_config(
        path: &Path,
        embedding_dim: usize,
        retrieval_config: RetrievalConfig,
    ) -> Result<Self, MenteError> {
        // Ensure the parent directory exists so a fresh path works.
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent).map_err(store_err)?;
        }
        ensure_vec0_registered();
        let mut conn = Connection::open(path).map_err(store_err)?;
        // WAL: concurrent readers don't block the writer, and crash recovery is
        // handled by SQLite instead of the old custom WAL.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(store_err)?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(store_err)?;
        let effective = Self::init(&mut conn, embedding_dim)?;
        Ok(Self {
            conn: Mutex::new(conn),
            retrieval_config: RwLock::new(retrieval_config),
            trace_retrieval: AtomicBool::new(false),
            embedding_dim: AtomicUsize::new(effective),
        })
    }

    /// Open an ephemeral in-memory database (used by tests and quick spikes).
    pub fn open_in_memory(embedding_dim: usize) -> Result<Self, MenteError> {
        Self::open_in_memory_with_retrieval_config(embedding_dim, RetrievalConfig::default())
    }

    /// Open an ephemeral database with explicit retrieval configuration.
    pub fn open_in_memory_with_retrieval_config(
        embedding_dim: usize,
        retrieval_config: RetrievalConfig,
    ) -> Result<Self, MenteError> {
        ensure_vec0_registered();
        let mut conn = Connection::open_in_memory().map_err(store_err)?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(store_err)?;
        let effective = Self::init(&mut conn, embedding_dim)?;
        Ok(Self {
            conn: Mutex::new(conn),
            retrieval_config: RwLock::new(retrieval_config),
            trace_retrieval: AtomicBool::new(false),
            embedding_dim: AtomicUsize::new(effective),
        })
    }

    /// The dimension the vector index was created with (0 = deferred / absent).
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim.load(Ordering::Relaxed)
    }

    /// Current retrieval tuning used by hybrid and multi-query search.
    pub fn retrieval_config(&self) -> RetrievalConfig {
        self.retrieval_config.read().clone()
    }

    /// Replace retrieval tuning for future searches.
    pub fn set_retrieval_config(&self, config: RetrievalConfig) {
        *self.retrieval_config.write() = config;
    }

    /// Enable or disable persisted retrieval traces.
    pub fn set_retrieval_tracing(&self, enabled: bool) {
        self.trace_retrieval.store(enabled, Ordering::Relaxed);
    }

    /// Whether retrieval traces are persisted for future searches.
    pub fn retrieval_tracing_enabled(&self) -> bool {
        self.trace_retrieval.load(Ordering::Relaxed)
    }

    /// Create (or recreate) the vec0 vector index at `dim` and backfill it from
    /// the embeddings already stored in `memories.embedding`. Called by the
    /// facade once an embedder is configured. No-op if the index already exists
    /// at the same dimension.
    ///
    /// Detailed behavior:
    /// - Early exit if `dim == 0` (deferred, no index) or if `dim` equals the
    ///   current in-memory `embedding_dim`.
    /// - Executes inside a DB transaction: drops any existing `memory_vec` and
    ///   recreates it with `CREATE VIRTUAL TABLE memory_vec USING
    ///   vec0(embedding float[<dim>])` so vec0 is configured for the requested
    ///   dimensionality.
    /// - Backfills by scanning `memories` for non-null `embedding` blobs:
    ///   - rows with a stored vector length != `dim` are skipped;
    ///   - matching vectors are normalized (so cosine similarity becomes
    ///     equivalent to L2 on the stored vectors), converted to a BLOB, and
    ///     inserted into `memory_vec` as `(rowid, embedding)`.
    /// - Updates `schema_meta.embedding_dim`, records a `vector_index_rebuild`
    ///   operation with counts (`backfilled`, `skipped_dimension_mismatch`),
    ///   commits the transaction, and updates the in-memory `embedding_dim`.
    /// - Note: the original embeddings remain in `memories.embedding`; only a
    ///   normalized copy is stored in `memory_vec` for KNN.
    pub fn ensure_vector_index(&self, dim: usize) -> Result<(), MenteError> {
        if dim == 0 || dim == self.embedding_dim.load(Ordering::Relaxed) {
            return Ok(());
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        // Drop any existing vec0 table (possibly at a different dim) and
        // recreate at the requested dimension.
        let _ = tx.execute("DROP TABLE IF EXISTS memory_vec", []);
        tx.execute(
            &format!("CREATE VIRTUAL TABLE memory_vec USING vec0(embedding float[{dim}])"),
            [],
        )
        .map_err(store_err)?;

        // Backfill from stored embeddings (normalized).
        let pairs: Vec<(i64, Vec<u8>)> = {
            let mut select = tx
                .prepare("SELECT rowid, embedding FROM memories WHERE embedding IS NOT NULL")
                .map_err(store_err)?;
            let rows = select
                .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
                .map_err(store_err)?;
            let mut pairs = Vec::new();
            for row in rows {
                pairs.push(row.map_err(store_err)?);
            }
            pairs
        };
        let mut backfilled = 0usize;
        let mut skipped = 0usize;
        for (rowid, bytes) in pairs {
            let emb = blob_to_embedding(&bytes);
            if emb.len() != dim {
                skipped += 1;
                continue;
            }
            let norm = normalize(&emb);
            let blob = embedding_to_blob(&norm);
            tx.execute(
                "INSERT INTO memory_vec (rowid, embedding) VALUES (?1, ?2)",
                params![rowid, blob.as_slice()],
            )
            .map_err(store_err)?;
            backfilled += 1;
        }
        tx.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('embedding_dim', ?1)",
            params![dim.to_string()],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            &tx,
            "vector_index_rebuild",
            None,
            None,
            None,
            json!({
                "embedding_dim": dim,
                "backfilled": backfilled,
                "skipped_dimension_mismatch": skipped,
            }),
        )?;
        tx.commit().map_err(store_err)?;

        self.embedding_dim.store(dim, Ordering::Relaxed);
        Ok(())
    }

    /// Create all tables/indexes. `hint_dim` is used only on first creation; a
    /// stored `embedding_dim` in `schema_meta` wins on reopen so the vec0
    /// table survives across launches. Returns the effective dimension (0 means
    /// "deferred, no vector index yet").
    fn init(conn: &mut Connection, hint_dim: usize) -> Result<usize, MenteError> {
        // The vec0 module was registered process-globally by
        // `ensure_vec0_registered()` before this connection was opened, so the
        // `memory_vec` virtual table below will resolve automatically.
        let tx = conn.transaction().map_err(store_err)?;

        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id           TEXT PRIMARY KEY,
                agent_id     TEXT NOT NULL,
                space_id     TEXT NOT NULL,
                memory_type  TEXT NOT NULL,
                content      TEXT NOT NULL,
                embedding    BLOB,
                created_at   INTEGER NOT NULL,
                accessed_at  INTEGER NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                salience     REAL    NOT NULL DEFAULT 1.0,
                confidence   REAL    NOT NULL DEFAULT 1.0,
                attributes   TEXT    NOT NULL DEFAULT '{}',
                valid_from   INTEGER,
                valid_until  INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_mem_agent     ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_mem_space     ON memories(space_id);
            CREATE INDEX IF NOT EXISTS idx_mem_type      ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_mem_salience  ON memories(salience DESC);
            CREATE INDEX IF NOT EXISTS idx_mem_created   ON memories(created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT NOT NULL,
                tag       TEXT NOT NULL,
                PRIMARY KEY (memory_id, tag)
            );
            CREATE INDEX IF NOT EXISTS idx_tag ON memory_tags(tag);

            -- Knowledge graph edges (replaces the CSR/CSC graph). Traversal is
            -- done via the recursive CTE in [`Backend::subgraph`].
            CREATE TABLE IF NOT EXISTS edges (
                source     TEXT NOT NULL,
                target     TEXT NOT NULL,
                edge_type  TEXT NOT NULL,
                weight     REAL NOT NULL,
                created_at INTEGER NOT NULL,
                valid_from INTEGER,
                valid_until INTEGER,
                label      TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_edge_source ON edges(source, edge_type);
            CREATE INDEX IF NOT EXISTS idx_edge_target ON edges(target, edge_type);

            -- Full-text mirror of `memories.content` (replaces Bm25Index).
            -- External-content FTS5 keyed by the implicit `memories.rowid`;
            -- the triggers keep it in sync with upserts/deletes on `memories`.
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                content='memories',
                content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
            END;

            CREATE TABLE IF NOT EXISTS memory_operations (
                operation_id   TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                memory_id      TEXT,
                source         TEXT,
                target         TEXT,
                payload_json   TEXT NOT NULL,
                created_at     INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_operations_created
                ON memory_operations(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_operations_memory
                ON memory_operations(memory_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_operations_edge
                ON memory_operations(source, target, created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_sources (
                source_id       TEXT PRIMARY KEY,
                memory_id       TEXT NOT NULL,
                source_type     TEXT NOT NULL,
                conversation_id TEXT,
                turn_id         TEXT,
                actor_id        TEXT,
                observed_at     INTEGER,
                extractor       TEXT,
                extractor_hash  TEXT,
                prompt_hash     TEXT,
                payload_json    TEXT NOT NULL DEFAULT '{}',
                created_at      INTEGER NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_memory_sources_memory
                ON memory_sources(memory_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_sources_type
                ON memory_sources(source_type, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_memory_sources_turn
                ON memory_sources(conversation_id, turn_id);

            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title           TEXT,
                metadata_json   TEXT NOT NULL DEFAULT '{}',
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_conversations_updated
                ON conversations(updated_at DESC);

            CREATE TABLE IF NOT EXISTS conversation_events (
                event_id        TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                turn_id         TEXT,
                event_type      TEXT NOT NULL,
                actor_id        TEXT,
                content         TEXT,
                payload_json    TEXT NOT NULL DEFAULT '{}',
                observed_at     INTEGER NOT NULL,
                created_at      INTEGER NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_conversation_events_conversation
                ON conversation_events(conversation_id, observed_at, created_at);
            CREATE INDEX IF NOT EXISTS idx_conversation_events_turn
                ON conversation_events(conversation_id, turn_id);
            CREATE INDEX IF NOT EXISTS idx_conversation_events_type
                ON conversation_events(event_type, observed_at DESC);

            CREATE TABLE IF NOT EXISTS extraction_runs (
                run_id           TEXT PRIMARY KEY,
                source_memory_id TEXT,
                conversation_id  TEXT,
                extractor        TEXT NOT NULL,
                extractor_version TEXT NOT NULL,
                model            TEXT,
                prompt_hash      TEXT,
                config_hash      TEXT,
                status           TEXT NOT NULL,
                error            TEXT,
                output_json      TEXT NOT NULL DEFAULT '{}',
                started_at       INTEGER NOT NULL,
                completed_at     INTEGER,
                FOREIGN KEY(source_memory_id) REFERENCES memories(id) ON DELETE SET NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                    ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_extraction_runs_memory
                ON extraction_runs(source_memory_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_extraction_runs_conversation
                ON extraction_runs(conversation_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_extraction_runs_status
                ON extraction_runs(status, started_at DESC);

            CREATE TABLE IF NOT EXISTS entities (
                entity_id       TEXT PRIMARY KEY,
                entity_type     TEXT NOT NULL,
                canonical       TEXT NOT NULL,
                attributes_json TEXT NOT NULL DEFAULT '{}',
                confidence      REAL NOT NULL DEFAULT 1.0,
                created_at      INTEGER NOT NULL,
                updated_at      INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entities_type_canonical
                ON entities(entity_type, canonical);
            CREATE INDEX IF NOT EXISTS idx_entities_canonical
                ON entities(canonical);

            CREATE TABLE IF NOT EXISTS entity_aliases (
                entity_id  TEXT NOT NULL,
                alias      TEXT NOT NULL,
                source     TEXT,
                confidence REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(entity_id, alias),
                FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias
                ON entity_aliases(alias);

            CREATE TABLE IF NOT EXISTS memory_entities (
                memory_id  TEXT NOT NULL,
                entity_id  TEXT NOT NULL,
                role       TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 1.0,
                evidence   TEXT,
                PRIMARY KEY(memory_id, entity_id, role),
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_memory_entities_entity
                ON memory_entities(entity_id);
            CREATE INDEX IF NOT EXISTS idx_memory_entities_memory
                ON memory_entities(memory_id);

            CREATE TABLE IF NOT EXISTS claims (
                claim_id          TEXT PRIMARY KEY,
                claim_text        TEXT NOT NULL,
                claim_type        TEXT NOT NULL,
                subject_entity_id TEXT,
                predicate         TEXT,
                object_entity_id  TEXT,
                confidence        REAL NOT NULL DEFAULT 1.0,
                status            TEXT NOT NULL DEFAULT 'active',
                valid_from        INTEGER,
                valid_until       INTEGER,
                attributes_json   TEXT NOT NULL DEFAULT '{}',
                source_run_id     TEXT,
                created_at        INTEGER NOT NULL,
                updated_at        INTEGER NOT NULL,
                FOREIGN KEY(subject_entity_id) REFERENCES entities(entity_id)
                    ON DELETE SET NULL,
                FOREIGN KEY(object_entity_id) REFERENCES entities(entity_id)
                    ON DELETE SET NULL,
                FOREIGN KEY(source_run_id) REFERENCES extraction_runs(run_id)
                    ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claims_status
                ON claims(status, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_claims_subject
                ON claims(subject_entity_id, predicate);
            CREATE INDEX IF NOT EXISTS idx_claims_object
                ON claims(object_entity_id);
            CREATE INDEX IF NOT EXISTS idx_claims_valid
                ON claims(valid_from, valid_until);

            CREATE TABLE IF NOT EXISTS claim_entities (
                claim_id   TEXT NOT NULL,
                entity_id  TEXT NOT NULL,
                role       TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                PRIMARY KEY(claim_id, entity_id, role),
                FOREIGN KEY(claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE,
                FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_claim_entities_entity
                ON claim_entities(entity_id, role);

            CREATE TABLE IF NOT EXISTS claim_evidence (
                evidence_id   TEXT PRIMARY KEY,
                claim_id      TEXT NOT NULL,
                memory_id     TEXT NOT NULL,
                source_id     TEXT,
                evidence_text TEXT,
                span_start    INTEGER,
                span_end      INTEGER,
                confidence    REAL NOT NULL DEFAULT 1.0,
                created_at    INTEGER NOT NULL,
                FOREIGN KEY(claim_id) REFERENCES claims(claim_id) ON DELETE CASCADE,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY(source_id) REFERENCES memory_sources(source_id)
                    ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claim_evidence_claim
                ON claim_evidence(claim_id, confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_claim_evidence_memory
                ON claim_evidence(memory_id);

            CREATE TABLE IF NOT EXISTS entity_relationships (
                relationship_id  TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation_type    TEXT NOT NULL,
                confidence       REAL NOT NULL DEFAULT 1.0,
                status           TEXT NOT NULL DEFAULT 'active',
                valid_from       INTEGER,
                valid_until      INTEGER,
                attributes_json  TEXT NOT NULL DEFAULT '{}',
                source_run_id    TEXT,
                created_at       INTEGER NOT NULL,
                updated_at       INTEGER NOT NULL,
                FOREIGN KEY(source_entity_id) REFERENCES entities(entity_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(target_entity_id) REFERENCES entities(entity_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(source_run_id) REFERENCES extraction_runs(run_id)
                    ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_entity_relationships_source
                ON entity_relationships(source_entity_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_entity_relationships_target
                ON entity_relationships(target_entity_id, relation_type);
            CREATE INDEX IF NOT EXISTS idx_entity_relationships_status
                ON entity_relationships(status, updated_at DESC);

            CREATE TABLE IF NOT EXISTS relationship_evidence (
                evidence_id     TEXT PRIMARY KEY,
                relationship_id TEXT NOT NULL,
                memory_id       TEXT NOT NULL,
                source_id       TEXT,
                evidence_text   TEXT,
                span_start      INTEGER,
                span_end        INTEGER,
                confidence      REAL NOT NULL DEFAULT 1.0,
                created_at      INTEGER NOT NULL,
                FOREIGN KEY(relationship_id)
                    REFERENCES entity_relationships(relationship_id)
                    ON DELETE CASCADE,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY(source_id) REFERENCES memory_sources(source_id)
                    ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_relationship_evidence_relationship
                ON relationship_evidence(relationship_id, confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_relationship_evidence_memory
                ON relationship_evidence(memory_id);

            CREATE TABLE IF NOT EXISTS retrieval_traces (
                trace_id            TEXT PRIMARY KEY,
                query_text          TEXT,
                query_embedding_dim INTEGER NOT NULL,
                k                   INTEGER NOT NULL,
                fetch_k             INTEGER NOT NULL,
                tags_json           TEXT NOT NULL,
                tags_or             INTEGER NOT NULL,
                time_start          INTEGER,
                time_end            INTEGER,
                candidate_count     INTEGER NOT NULL,
                result_count        INTEGER NOT NULL,
                config_json         TEXT NOT NULL,
                created_at          INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_traces_created
                ON retrieval_traces(created_at DESC);

            CREATE TABLE IF NOT EXISTS retrieval_trace_hits (
                trace_id    TEXT NOT NULL,
                rank        INTEGER NOT NULL,
                memory_id   TEXT NOT NULL,
                score       REAL NOT NULL,
                vector_rank INTEGER,
                bm25_rank   INTEGER,
                salience    REAL,
                PRIMARY KEY(trace_id, rank),
                FOREIGN KEY(trace_id) REFERENCES retrieval_traces(trace_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_retrieval_trace_hits_memory
                ON retrieval_trace_hits(memory_id);
            "#,
        )
        .map_err(store_err)?;

        // Populate FTS from any pre-existing rows (no-op on a fresh open).
        // FTS5's 'rebuild' command rescans the external content table.
        let _ = tx.execute(
            "INSERT INTO memories_fts(memories_fts) VALUES('rebuild')",
            [],
        );

        // Resolve the effective vector dimension: a previously-stored dim wins
        // (so the vec0 table survives reopen), otherwise fall back to the hint.
        // `0` means "deferred", no embedder configured yet, so no vec0 table
        // is created until [`Backend::ensure_vector_index`] is called.
        let stored: Option<String> = tx
            .query_row(
                "SELECT value FROM schema_meta WHERE key = 'embedding_dim'",
                [],
                |r| r.get(0),
            )
            .optional()
            .map_err(store_err)?;
        let effective = stored
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&d| d > 0)
            .unwrap_or(hint_dim);
        if effective > 0 {
            tx.execute(
                &format!(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(embedding float[{effective}])"
                ),
                [],
            )
            .map_err(store_err)?;
        }
        tx.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', ?1)",
            params![SCHEMA_VERSION.to_string()],
        )
        .map_err(store_err)?;
        tx.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('embedding_dim', ?1)",
            params![effective.to_string()],
        )
        .map_err(store_err)?;

        tx.commit().map_err(store_err)?;
        Ok(effective)
    }

    fn record_operation_on(
        tx: &rusqlite::Transaction<'_>,
        operation_type: &str,
        memory_id: Option<MemoryId>,
        source: Option<MemoryId>,
        target: Option<MemoryId>,
        payload: serde_json::Value,
    ) -> Result<(), MenteError> {
        let operation_id = Uuid::new_v4().to_string();
        let memory_id = memory_id.map(|id| id.to_string());
        let source = source.map(|id| id.to_string());
        let target = target.map(|id| id.to_string());
        let payload_json = serde_json::to_string(&payload).map_err(store_err)?;
        tx.execute(
            r#"
            INSERT INTO memory_operations (
                operation_id, operation_type, memory_id, source, target,
                payload_json, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                operation_id,
                operation_type,
                memory_id.as_deref(),
                source.as_deref(),
                target.as_deref(),
                payload_json,
                now_us() as i64,
            ],
        )
        .map_err(store_err)?;
        Ok(())
    }

    fn insert_memory_source_on(
        tx: &rusqlite::Transaction<'_>,
        source: &MemorySource,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO memory_sources (
                source_id, memory_id, source_type, conversation_id, turn_id,
                actor_id, observed_at, extractor, extractor_hash, prompt_hash,
                payload_json, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            ON CONFLICT(source_id) DO UPDATE SET
                memory_id       = excluded.memory_id,
                source_type     = excluded.source_type,
                conversation_id = excluded.conversation_id,
                turn_id         = excluded.turn_id,
                actor_id        = excluded.actor_id,
                observed_at     = excluded.observed_at,
                extractor       = excluded.extractor,
                extractor_hash  = excluded.extractor_hash,
                prompt_hash     = excluded.prompt_hash,
                payload_json    = excluded.payload_json,
                created_at      = excluded.created_at
            "#,
            params![
                source.source_id,
                source.memory_id.to_string(),
                source.source_type,
                source.conversation_id.as_deref(),
                source.turn_id.as_deref(),
                source.actor_id.as_deref(),
                source.observed_at.map(|t| t as i64),
                source.extractor.as_deref(),
                source.extractor_hash.as_deref(),
                source.prompt_hash.as_deref(),
                source.payload_json,
                source.created_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "memory_source_upsert",
            Some(source.memory_id),
            None,
            None,
            json!({
                "source_id": source.source_id,
                "source_type": source.source_type,
                "conversation_id": source.conversation_id,
                "turn_id": source.turn_id,
                "actor_id": source.actor_id,
                "observed_at": source.observed_at,
                "extractor": source.extractor,
                "extractor_hash": source.extractor_hash,
                "prompt_hash": source.prompt_hash,
            }),
        )?;
        Ok(())
    }

    fn upsert_entity_on(
        tx: &rusqlite::Transaction<'_>,
        entity: &EntityRecord,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO entities (
                entity_id, entity_type, canonical, attributes_json,
                confidence, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            ON CONFLICT(entity_id) DO UPDATE SET
                entity_type     = excluded.entity_type,
                canonical       = excluded.canonical,
                attributes_json = excluded.attributes_json,
                confidence      = excluded.confidence,
                updated_at      = excluded.updated_at
            "#,
            params![
                entity.entity_id,
                entity.entity_type,
                entity.canonical,
                entity.attributes_json,
                entity.confidence as f64,
                entity.created_at as i64,
                entity.updated_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "entity_upsert",
            None,
            None,
            None,
            json!({
                "entity_id": entity.entity_id,
                "entity_type": entity.entity_type,
                "canonical": entity.canonical,
                "confidence": entity.confidence,
            }),
        )?;
        Ok(())
    }

    fn add_entity_alias_on(
        tx: &rusqlite::Transaction<'_>,
        alias: &EntityAlias,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO entity_aliases (entity_id, alias, source, confidence)
            VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(entity_id, alias) DO UPDATE SET
                source = excluded.source,
                confidence = excluded.confidence
            "#,
            params![
                alias.entity_id,
                alias.alias,
                alias.source.as_deref(),
                alias.confidence as f64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "entity_alias_upsert",
            None,
            None,
            None,
            json!({
                "entity_id": alias.entity_id,
                "alias": alias.alias,
                "source": alias.source,
                "confidence": alias.confidence,
            }),
        )?;
        Ok(())
    }

    fn link_memory_entity_on(
        tx: &rusqlite::Transaction<'_>,
        link: &MemoryEntityLink,
    ) -> Result<(), MenteError> {
        let role = link.role.as_deref().unwrap_or("");
        tx.execute(
            r#"
            INSERT INTO memory_entities (
                memory_id, entity_id, role, confidence, evidence
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(memory_id, entity_id, role) DO UPDATE SET
                confidence = excluded.confidence,
                evidence = excluded.evidence
            "#,
            params![
                link.memory_id.to_string(),
                link.entity_id,
                role,
                link.confidence as f64,
                link.evidence.as_deref(),
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "memory_entity_link_upsert",
            Some(link.memory_id),
            None,
            None,
            json!({
                "entity_id": link.entity_id,
                "role": link.role,
                "confidence": link.confidence,
                "evidence": link.evidence,
            }),
        )?;
        Ok(())
    }

    fn upsert_conversation_on(
        tx: &rusqlite::Transaction<'_>,
        conversation: &ConversationRecord,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO conversations (
                conversation_id, title, metadata_json, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(conversation_id) DO UPDATE SET
                title = excluded.title,
                metadata_json = excluded.metadata_json,
                updated_at = excluded.updated_at
            "#,
            params![
                conversation.conversation_id,
                conversation.title.as_deref(),
                conversation.metadata_json,
                conversation.created_at as i64,
                conversation.updated_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "conversation_upsert",
            None,
            None,
            None,
            json!({
                "conversation_id": conversation.conversation_id,
                "title": conversation.title,
            }),
        )?;
        Ok(())
    }

    fn insert_conversation_event_on(
        tx: &rusqlite::Transaction<'_>,
        event: &ConversationEvent,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO conversation_events (
                event_id, conversation_id, turn_id, event_type, actor_id,
                content, payload_json, observed_at, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(event_id) DO UPDATE SET
                conversation_id = excluded.conversation_id,
                turn_id = excluded.turn_id,
                event_type = excluded.event_type,
                actor_id = excluded.actor_id,
                content = excluded.content,
                payload_json = excluded.payload_json,
                observed_at = excluded.observed_at
            "#,
            params![
                event.event_id,
                event.conversation_id,
                event.turn_id.as_deref(),
                event.event_type,
                event.actor_id.as_deref(),
                event.content.as_deref(),
                event.payload_json,
                event.observed_at as i64,
                event.created_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "conversation_event_upsert",
            None,
            None,
            None,
            json!({
                "event_id": event.event_id,
                "conversation_id": event.conversation_id,
                "turn_id": event.turn_id,
                "event_type": event.event_type,
                "actor_id": event.actor_id,
                "observed_at": event.observed_at,
            }),
        )?;
        Ok(())
    }

    fn upsert_extraction_run_on(
        tx: &rusqlite::Transaction<'_>,
        run: &ExtractionRun,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO extraction_runs (
                run_id, source_memory_id, conversation_id, extractor,
                extractor_version, model, prompt_hash, config_hash, status,
                error, output_json, started_at, completed_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            ON CONFLICT(run_id) DO UPDATE SET
                source_memory_id = excluded.source_memory_id,
                conversation_id = excluded.conversation_id,
                extractor = excluded.extractor,
                extractor_version = excluded.extractor_version,
                model = excluded.model,
                prompt_hash = excluded.prompt_hash,
                config_hash = excluded.config_hash,
                status = excluded.status,
                error = excluded.error,
                output_json = excluded.output_json,
                completed_at = excluded.completed_at
            "#,
            params![
                run.run_id,
                run.source_memory_id.map(|id| id.to_string()),
                run.conversation_id.as_deref(),
                run.extractor,
                run.extractor_version,
                run.model.as_deref(),
                run.prompt_hash.as_deref(),
                run.config_hash.as_deref(),
                run.status,
                run.error.as_deref(),
                run.output_json,
                run.started_at as i64,
                run.completed_at.map(|t| t as i64),
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "extraction_run_upsert",
            run.source_memory_id,
            None,
            None,
            json!({
                "run_id": run.run_id,
                "conversation_id": run.conversation_id,
                "extractor": run.extractor,
                "extractor_version": run.extractor_version,
                "model": run.model,
                "prompt_hash": run.prompt_hash,
                "config_hash": run.config_hash,
                "status": run.status,
            }),
        )?;
        Ok(())
    }

    fn upsert_claim_on(
        tx: &rusqlite::Transaction<'_>,
        claim: &ClaimRecord,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO claims (
                claim_id, claim_text, claim_type, subject_entity_id, predicate,
                object_entity_id, confidence, status, valid_from, valid_until,
                attributes_json, source_run_id, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
            ON CONFLICT(claim_id) DO UPDATE SET
                claim_text = excluded.claim_text,
                claim_type = excluded.claim_type,
                subject_entity_id = excluded.subject_entity_id,
                predicate = excluded.predicate,
                object_entity_id = excluded.object_entity_id,
                confidence = excluded.confidence,
                status = excluded.status,
                valid_from = excluded.valid_from,
                valid_until = excluded.valid_until,
                attributes_json = excluded.attributes_json,
                source_run_id = excluded.source_run_id,
                updated_at = excluded.updated_at
            "#,
            params![
                claim.claim_id,
                claim.claim_text,
                claim.claim_type,
                claim.subject_entity_id.as_deref(),
                claim.predicate.as_deref(),
                claim.object_entity_id.as_deref(),
                claim.confidence as f64,
                claim.status,
                claim.valid_from.map(|t| t as i64),
                claim.valid_until.map(|t| t as i64),
                claim.attributes_json,
                claim.source_run_id.as_deref(),
                claim.created_at as i64,
                claim.updated_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "claim_upsert",
            None,
            None,
            None,
            json!({
                "claim_id": claim.claim_id,
                "claim_type": claim.claim_type,
                "subject_entity_id": claim.subject_entity_id,
                "predicate": claim.predicate,
                "object_entity_id": claim.object_entity_id,
                "confidence": claim.confidence,
                "status": claim.status,
                "source_run_id": claim.source_run_id,
            }),
        )?;
        Ok(())
    }

    fn link_claim_entity_on(
        tx: &rusqlite::Transaction<'_>,
        link: &ClaimEntityLink,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO claim_entities (claim_id, entity_id, role, confidence)
            VALUES (?1, ?2, ?3, ?4)
            ON CONFLICT(claim_id, entity_id, role) DO UPDATE SET
                confidence = excluded.confidence
            "#,
            params![
                link.claim_id,
                link.entity_id,
                link.role,
                link.confidence as f64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "claim_entity_link_upsert",
            None,
            None,
            None,
            json!({
                "claim_id": link.claim_id,
                "entity_id": link.entity_id,
                "role": link.role,
                "confidence": link.confidence,
            }),
        )?;
        Ok(())
    }

    fn add_claim_evidence_on(
        tx: &rusqlite::Transaction<'_>,
        evidence: &ClaimEvidence,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO claim_evidence (
                evidence_id, claim_id, memory_id, source_id, evidence_text,
                span_start, span_end, confidence, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(evidence_id) DO UPDATE SET
                claim_id = excluded.claim_id,
                memory_id = excluded.memory_id,
                source_id = excluded.source_id,
                evidence_text = excluded.evidence_text,
                span_start = excluded.span_start,
                span_end = excluded.span_end,
                confidence = excluded.confidence
            "#,
            params![
                evidence.evidence_id,
                evidence.claim_id,
                evidence.memory_id.to_string(),
                evidence.source_id.as_deref(),
                evidence.evidence_text.as_deref(),
                evidence.span_start,
                evidence.span_end,
                evidence.confidence as f64,
                evidence.created_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "claim_evidence_upsert",
            Some(evidence.memory_id),
            None,
            None,
            json!({
                "evidence_id": evidence.evidence_id,
                "claim_id": evidence.claim_id,
                "source_id": evidence.source_id,
                "confidence": evidence.confidence,
            }),
        )?;
        Ok(())
    }

    fn upsert_entity_relationship_on(
        tx: &rusqlite::Transaction<'_>,
        relationship: &EntityRelationship,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO entity_relationships (
                relationship_id, source_entity_id, target_entity_id,
                relation_type, confidence, status, valid_from, valid_until,
                attributes_json, source_run_id, created_at, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            ON CONFLICT(relationship_id) DO UPDATE SET
                source_entity_id = excluded.source_entity_id,
                target_entity_id = excluded.target_entity_id,
                relation_type = excluded.relation_type,
                confidence = excluded.confidence,
                status = excluded.status,
                valid_from = excluded.valid_from,
                valid_until = excluded.valid_until,
                attributes_json = excluded.attributes_json,
                source_run_id = excluded.source_run_id,
                updated_at = excluded.updated_at
            "#,
            params![
                relationship.relationship_id,
                relationship.source_entity_id,
                relationship.target_entity_id,
                relationship.relation_type,
                relationship.confidence as f64,
                relationship.status,
                relationship.valid_from.map(|t| t as i64),
                relationship.valid_until.map(|t| t as i64),
                relationship.attributes_json,
                relationship.source_run_id.as_deref(),
                relationship.created_at as i64,
                relationship.updated_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "entity_relationship_upsert",
            None,
            None,
            None,
            json!({
                "relationship_id": relationship.relationship_id,
                "source_entity_id": relationship.source_entity_id,
                "target_entity_id": relationship.target_entity_id,
                "relation_type": relationship.relation_type,
                "confidence": relationship.confidence,
                "status": relationship.status,
                "source_run_id": relationship.source_run_id,
            }),
        )?;
        Ok(())
    }

    fn add_relationship_evidence_on(
        tx: &rusqlite::Transaction<'_>,
        evidence: &RelationshipEvidence,
    ) -> Result<(), MenteError> {
        tx.execute(
            r#"
            INSERT INTO relationship_evidence (
                evidence_id, relationship_id, memory_id, source_id, evidence_text,
                span_start, span_end, confidence, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            ON CONFLICT(evidence_id) DO UPDATE SET
                relationship_id = excluded.relationship_id,
                memory_id = excluded.memory_id,
                source_id = excluded.source_id,
                evidence_text = excluded.evidence_text,
                span_start = excluded.span_start,
                span_end = excluded.span_end,
                confidence = excluded.confidence
            "#,
            params![
                evidence.evidence_id,
                evidence.relationship_id,
                evidence.memory_id.to_string(),
                evidence.source_id.as_deref(),
                evidence.evidence_text.as_deref(),
                evidence.span_start,
                evidence.span_end,
                evidence.confidence as f64,
                evidence.created_at as i64,
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            tx,
            "relationship_evidence_upsert",
            Some(evidence.memory_id),
            None,
            None,
            json!({
                "evidence_id": evidence.evidence_id,
                "relationship_id": evidence.relationship_id,
                "source_id": evidence.source_id,
                "confidence": evidence.confidence,
            }),
        )?;
        Ok(())
    }

    fn store_memory_on(
        tx: &rusqlite::Transaction<'_>,
        node: &MemoryNode,
        dim: usize,
    ) -> Result<(), MenteError> {
        let attrs_json = serde_json::to_string(&node.attributes).map_err(store_err)?;
        let emb_blob = if node.embedding.is_empty() {
            None
        } else {
            Some(embedding_to_blob(&node.embedding))
        };

        let id_str = node.id.to_string();

        // Upsert the memory row. `ON CONFLICT(id) DO UPDATE` preserves the
        // implicit rowid, which keeps the vec0 key stable across re-stores.
        tx.execute(
            r#"
            INSERT INTO memories (
                id, agent_id, space_id, memory_type, content, embedding,
                created_at, accessed_at, access_count, salience, confidence,
                attributes, valid_from, valid_until
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14
            )
            ON CONFLICT(id) DO UPDATE SET
                agent_id     = excluded.agent_id,
                space_id     = excluded.space_id,
                memory_type  = excluded.memory_type,
                content      = excluded.content,
                embedding    = excluded.embedding,
                accessed_at  = excluded.accessed_at,
                access_count = excluded.access_count,
                salience     = excluded.salience,
                confidence   = excluded.confidence,
                attributes   = excluded.attributes,
                valid_from   = excluded.valid_from,
                valid_until  = excluded.valid_until
            "#,
            params![
                id_str,
                node.agent_id.to_string(),
                node.space_id.to_string(),
                memory_type_str(node.memory_type),
                node.content,
                emb_blob.as_deref(),
                node.created_at as i64,
                node.accessed_at as i64,
                node.access_count as i64,
                node.salience as f64,
                node.confidence as f64,
                attrs_json,
                node.valid_from.map(|t| t as i64),
                node.valid_until.map(|t| t as i64),
            ],
        )
        .map_err(store_err)?;

        // Refresh the derived vector row. vec0 `REPLACE` semantics are version
        // dependent, so delete-then-insert is the portable choice.
        let rowid: i64 = tx
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![id_str],
                |r| r.get(0),
            )
            .map_err(store_err)?;
        let _ = tx.execute("DELETE FROM memory_vec WHERE rowid = ?1", params![rowid]);
        if dim > 0 && !node.embedding.is_empty() {
            let norm = normalize(&node.embedding);
            let norm_blob = embedding_to_blob(&norm);
            tx.execute(
                "INSERT INTO memory_vec (rowid, embedding) VALUES (?1, ?2)",
                params![rowid, norm_blob.as_slice()],
            )
            .map_err(store_err)?;
        }

        // Refresh tags (delete + reinsert is simplest and correct).
        tx.execute(
            "DELETE FROM memory_tags WHERE memory_id = ?1",
            params![id_str],
        )
        .map_err(store_err)?;
        for tag in &node.tags {
            tx.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?1, ?2)",
                params![id_str, tag],
            )
            .map_err(store_err)?;
        }

        Self::record_operation_on(
            tx,
            "memory_upsert",
            Some(node.id),
            None,
            None,
            json!({
                "agent_id": node.agent_id.to_string(),
                "space_id": node.space_id.to_string(),
                "memory_type": memory_type_str(node.memory_type),
                "content_len": node.content.len(),
                "embedding_dim": node.embedding.len(),
                "tag_count": node.tags.len(),
                "valid_from": node.valid_from,
                "valid_until": node.valid_until,
            }),
        )?;

        Ok(())
    }

    /// Persist (or update) a memory and refresh its vector + tag rows.
    ///
    /// The original embedding is stored verbatim in `memories.embedding`
    /// (returned unchanged by [`Self::get_memory`]); a normalized copy goes
    /// into the `vec0` table so KNN behaves as cosine similarity.
    pub fn store_memory(&self, node: &MemoryNode) -> Result<(), MenteError> {
        let dim = self.embedding_dim.load(Ordering::Relaxed);
        if !node.embedding.is_empty() && dim > 0 && node.embedding.len() != dim {
            return Err(MenteError::EmbeddingDimensionMismatch {
                got: node.embedding.len(),
                expected: dim,
            });
        }

        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::store_memory_on(&tx, node, dim)?;
        tx.commit().map_err(store_err)
    }

    /// Persist a memory and its provenance in one transaction.
    pub fn store_memory_with_source(
        &self,
        node: &MemoryNode,
        source: &MemorySource,
    ) -> Result<(), MenteError> {
        if source.memory_id != node.id {
            return Err(MenteError::Storage(format!(
                "memory source {} points at {}, expected {}",
                source.source_id, source.memory_id, node.id
            )));
        }
        let dim = self.embedding_dim.load(Ordering::Relaxed);
        if !node.embedding.is_empty() && dim > 0 && node.embedding.len() != dim {
            return Err(MenteError::EmbeddingDimensionMismatch {
                got: node.embedding.len(),
                expected: dim,
            });
        }

        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::store_memory_on(&tx, node, dim)?;
        Self::insert_memory_source_on(&tx, source)?;
        tx.commit().map_err(store_err)
    }

    /// Add or update provenance for an existing memory.
    pub fn add_memory_source(&self, source: &MemorySource) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::insert_memory_source_on(&tx, source)?;
        tx.commit().map_err(store_err)
    }

    /// List provenance records for a memory, newest first.
    pub fn memory_sources(&self, id: MemoryId) -> Result<Vec<MemorySource>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT source_id, memory_id, source_type, conversation_id, turn_id,
                       actor_id, observed_at, extractor, extractor_hash, prompt_hash,
                       payload_json, created_at
                FROM memory_sources
                WHERE memory_id = ?1
                ORDER BY created_at DESC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![id.to_string()], |row| {
                Ok(MemorySource {
                    source_id: row.get(0)?,
                    memory_id: parse_id::<MemoryId>(&row.get::<_, String>(1)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    source_type: row.get(2)?,
                    conversation_id: row.get(3)?,
                    turn_id: row.get(4)?,
                    actor_id: row.get(5)?,
                    observed_at: row.get::<_, Option<i64>>(6)?.map(|t| t as Timestamp),
                    extractor: row.get(7)?,
                    extractor_hash: row.get(8)?,
                    prompt_hash: row.get(9)?,
                    payload_json: row.get(10)?,
                    created_at: row.get::<_, i64>(11)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Add or update a conversation container.
    pub fn upsert_conversation(&self, conversation: &ConversationRecord) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_conversation_on(&tx, conversation)?;
        tx.commit().map_err(store_err)
    }

    /// Add or update an event in the conversation timeline.
    pub fn add_conversation_event(&self, event: &ConversationEvent) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        tx.execute(
            r#"
            INSERT OR IGNORE INTO conversations (
                conversation_id, metadata_json, created_at, updated_at
            ) VALUES (?1, '{}', ?2, ?2)
            "#,
            params![event.conversation_id, event.created_at as i64],
        )
        .map_err(store_err)?;
        tx.execute(
            "UPDATE conversations SET updated_at = MAX(updated_at, ?2) WHERE conversation_id = ?1",
            params![event.conversation_id, event.observed_at as i64],
        )
        .map_err(store_err)?;
        Self::insert_conversation_event_on(&tx, event)?;
        tx.commit().map_err(store_err)
    }

    /// Conversation events ordered by observed time.
    pub fn conversation_events(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> Result<Vec<ConversationEvent>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT event_id, conversation_id, turn_id, event_type, actor_id,
                       content, payload_json, observed_at, created_at
                FROM conversation_events
                WHERE conversation_id = ?1
                ORDER BY observed_at ASC, created_at ASC
                LIMIT ?2
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![conversation_id, usize_to_i64(limit)?], |row| {
                Ok(ConversationEvent {
                    event_id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    turn_id: row.get(2)?,
                    event_type: row.get(3)?,
                    actor_id: row.get(4)?,
                    content: row.get(5)?,
                    payload_json: row.get(6)?,
                    observed_at: row.get::<_, i64>(7)? as Timestamp,
                    created_at: row.get::<_, i64>(8)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Add or update one extraction run.
    pub fn upsert_extraction_run(&self, run: &ExtractionRun) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_extraction_run_on(&tx, run)?;
        tx.commit().map_err(store_err)
    }

    /// Recent extraction runs, newest first.
    pub fn recent_extraction_runs(&self, limit: usize) -> Result<Vec<ExtractionRun>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT run_id, source_memory_id, conversation_id, extractor,
                       extractor_version, model, prompt_hash, config_hash,
                       status, error, output_json, started_at, completed_at
                FROM extraction_runs
                ORDER BY started_at DESC
                LIMIT ?1
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![usize_to_i64(limit)?], |row| {
                Ok(ExtractionRun {
                    run_id: row.get(0)?,
                    source_memory_id: parse_optional_memory_id(row.get(1)?),
                    conversation_id: row.get(2)?,
                    extractor: row.get(3)?,
                    extractor_version: row.get(4)?,
                    model: row.get(5)?,
                    prompt_hash: row.get(6)?,
                    config_hash: row.get(7)?,
                    status: row.get(8)?,
                    error: row.get(9)?,
                    output_json: row.get(10)?,
                    started_at: row.get::<_, i64>(11)? as Timestamp,
                    completed_at: row.get::<_, Option<i64>>(12)?.map(|t| t as Timestamp),
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Extraction runs for one source memory, newest first.
    pub fn extraction_runs_for_memory(
        &self,
        memory_id: MemoryId,
    ) -> Result<Vec<ExtractionRun>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT run_id, source_memory_id, conversation_id, extractor,
                       extractor_version, model, prompt_hash, config_hash,
                       status, error, output_json, started_at, completed_at
                FROM extraction_runs
                WHERE source_memory_id = ?1
                ORDER BY started_at DESC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![memory_id.to_string()], |row| {
                Ok(ExtractionRun {
                    run_id: row.get(0)?,
                    source_memory_id: parse_optional_memory_id(row.get(1)?),
                    conversation_id: row.get(2)?,
                    extractor: row.get(3)?,
                    extractor_version: row.get(4)?,
                    model: row.get(5)?,
                    prompt_hash: row.get(6)?,
                    config_hash: row.get(7)?,
                    status: row.get(8)?,
                    error: row.get(9)?,
                    output_json: row.get(10)?,
                    started_at: row.get::<_, i64>(11)? as Timestamp,
                    completed_at: row.get::<_, Option<i64>>(12)?.map(|t| t as Timestamp),
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Add or update a canonical entity.
    pub fn upsert_entity(&self, entity: &EntityRecord) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_entity_on(&tx, entity)?;
        tx.commit().map_err(store_err)
    }

    /// Load an entity by id.
    pub fn get_entity(&self, entity_id: &str) -> Result<Option<EntityRecord>, MenteError> {
        let conn = self.conn.lock();
        conn.query_row(
            r#"
            SELECT entity_id, entity_type, canonical, attributes_json,
                   confidence, created_at, updated_at
            FROM entities
            WHERE entity_id = ?1
            "#,
            params![entity_id],
            |row| {
                Ok(EntityRecord {
                    entity_id: row.get(0)?,
                    entity_type: row.get(1)?,
                    canonical: row.get(2)?,
                    attributes_json: row.get(3)?,
                    confidence: row.get::<_, f64>(4)? as f32,
                    created_at: row.get::<_, i64>(5)? as Timestamp,
                    updated_at: row.get::<_, i64>(6)? as Timestamp,
                })
            },
        )
        .optional()
        .map_err(store_err)
    }

    /// Resolve an entity by exact `(entity_type, canonical)` identity.
    ///
    /// The facade normalizes canonical names before writing. Keeping lookup
    /// here avoids duplicate canonical entities while preserving aliases as
    /// separate lookup rows.
    pub fn entity_by_canonical(
        &self,
        entity_type: &str,
        canonical: &str,
    ) -> Result<Option<EntityRecord>, MenteError> {
        let conn = self.conn.lock();
        conn.query_row(
            r#"
            SELECT entity_id, entity_type, canonical, attributes_json,
                   confidence, created_at, updated_at
            FROM entities
            WHERE entity_type = ?1 AND canonical = ?2
            ORDER BY updated_at DESC
            LIMIT 1
            "#,
            params![entity_type, canonical],
            |row| {
                Ok(EntityRecord {
                    entity_id: row.get(0)?,
                    entity_type: row.get(1)?,
                    canonical: row.get(2)?,
                    attributes_json: row.get(3)?,
                    confidence: row.get::<_, f64>(4)? as f32,
                    created_at: row.get::<_, i64>(5)? as Timestamp,
                    updated_at: row.get::<_, i64>(6)? as Timestamp,
                })
            },
        )
        .optional()
        .map_err(store_err)
    }

    /// List canonical entities, sorted by update time.
    pub fn list_entities(&self, limit: usize) -> Result<Vec<EntityRecord>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT entity_id, entity_type, canonical, attributes_json,
                       confidence, created_at, updated_at
                FROM entities
                ORDER BY updated_at DESC
                LIMIT ?1
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![usize_to_i64(limit)?], |row| {
                Ok(EntityRecord {
                    entity_id: row.get(0)?,
                    entity_type: row.get(1)?,
                    canonical: row.get(2)?,
                    attributes_json: row.get(3)?,
                    confidence: row.get::<_, f64>(4)? as f32,
                    created_at: row.get::<_, i64>(5)? as Timestamp,
                    updated_at: row.get::<_, i64>(6)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Add or update an alias for an entity.
    pub fn add_entity_alias(&self, alias: &EntityAlias) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::add_entity_alias_on(&tx, alias)?;
        tx.commit().map_err(store_err)
    }

    /// List aliases for an entity.
    pub fn entity_aliases(&self, entity_id: &str) -> Result<Vec<EntityAlias>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT entity_id, alias, source, confidence
                FROM entity_aliases
                WHERE entity_id = ?1
                ORDER BY alias ASC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![entity_id], |row| {
                Ok(EntityAlias {
                    entity_id: row.get(0)?,
                    alias: row.get(1)?,
                    source: row.get(2)?,
                    confidence: row.get::<_, f64>(3)? as f32,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Resolve entities by alias.
    pub fn entities_by_alias(&self, alias: &str) -> Result<Vec<EntityRecord>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT e.entity_id, e.entity_type, e.canonical, e.attributes_json,
                       e.confidence, e.created_at, e.updated_at
                FROM entity_aliases a
                JOIN entities e ON e.entity_id = a.entity_id
                WHERE a.alias = ?1
                ORDER BY a.confidence DESC, e.updated_at DESC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![alias], |row| {
                Ok(EntityRecord {
                    entity_id: row.get(0)?,
                    entity_type: row.get(1)?,
                    canonical: row.get(2)?,
                    attributes_json: row.get(3)?,
                    confidence: row.get::<_, f64>(4)? as f32,
                    created_at: row.get::<_, i64>(5)? as Timestamp,
                    updated_at: row.get::<_, i64>(6)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Link a memory to a canonical entity.
    pub fn link_memory_entity(&self, link: &MemoryEntityLink) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::link_memory_entity_on(&tx, link)?;
        tx.commit().map_err(store_err)
    }

    /// Batch upsert canonical entities, aliases, and links for one or more
    /// memories in a single transaction.
    pub fn upsert_entity_bundle(
        &self,
        entities: &[EntityRecord],
        aliases: &[EntityAlias],
        links: &[MemoryEntityLink],
    ) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        for entity in entities {
            Self::upsert_entity_on(&tx, entity)?;
        }
        for alias in aliases {
            Self::add_entity_alias_on(&tx, alias)?;
        }
        for link in links {
            Self::link_memory_entity_on(&tx, link)?;
        }
        tx.commit().map_err(store_err)
    }

    /// Replace all entity links for a memory, then upsert canonical entities
    /// and aliases used by the new links.
    pub fn replace_memory_entity_bundle(
        &self,
        memory_id: MemoryId,
        entities: &[EntityRecord],
        aliases: &[EntityAlias],
        links: &[MemoryEntityLink],
    ) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        tx.execute(
            "DELETE FROM memory_entities WHERE memory_id = ?1",
            params![memory_id.to_string()],
        )
        .map_err(store_err)?;
        for entity in entities {
            Self::upsert_entity_on(&tx, entity)?;
        }
        for alias in aliases {
            Self::add_entity_alias_on(&tx, alias)?;
        }
        for link in links {
            Self::link_memory_entity_on(&tx, link)?;
        }
        Self::record_operation_on(
            &tx,
            "memory_entity_links_replace",
            Some(memory_id),
            None,
            None,
            serde_json::json!({
                "entity_count": entities.len(),
                "alias_count": aliases.len(),
                "link_count": links.len(),
            }),
        )?;
        tx.commit().map_err(store_err)
    }

    /// Entity links for one memory.
    pub fn memory_entity_links(&self, id: MemoryId) -> Result<Vec<MemoryEntityLink>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT memory_id, entity_id, role, confidence, evidence
                FROM memory_entities
                WHERE memory_id = ?1
                ORDER BY confidence DESC, entity_id ASC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![id.to_string()], |row| {
                let role: String = row.get(2)?;
                Ok(MemoryEntityLink {
                    memory_id: parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    entity_id: row.get(1)?,
                    role: if role.is_empty() { None } else { Some(role) },
                    confidence: row.get::<_, f64>(3)? as f32,
                    evidence: row.get(4)?,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Memory links for one entity.
    pub fn memories_for_entity(
        &self,
        entity_id: &str,
    ) -> Result<Vec<MemoryEntityLink>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT memory_id, entity_id, role, confidence, evidence
                FROM memory_entities
                WHERE entity_id = ?1
                ORDER BY confidence DESC, memory_id ASC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![entity_id], |row| {
                let role: String = row.get(2)?;
                Ok(MemoryEntityLink {
                    memory_id: parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    entity_id: row.get(1)?,
                    role: if role.is_empty() { None } else { Some(role) },
                    confidence: row.get::<_, f64>(3)? as f32,
                    evidence: row.get(4)?,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Add or update a derived claim.
    pub fn upsert_claim(&self, claim: &ClaimRecord) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_claim_on(&tx, claim)?;
        tx.commit().map_err(store_err)
    }

    /// Link a claim to an entity with a semantic role.
    pub fn link_claim_entity(&self, link: &ClaimEntityLink) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::link_claim_entity_on(&tx, link)?;
        tx.commit().map_err(store_err)
    }

    /// Add evidence for a claim.
    pub fn add_claim_evidence(&self, evidence: &ClaimEvidence) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::add_claim_evidence_on(&tx, evidence)?;
        tx.commit().map_err(store_err)
    }

    /// Persist a claim and all of its local links in one transaction.
    pub fn upsert_claim_bundle(
        &self,
        claim: &ClaimRecord,
        entities: &[ClaimEntityLink],
        evidence: &[ClaimEvidence],
    ) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_claim_on(&tx, claim)?;
        for link in entities {
            Self::link_claim_entity_on(&tx, link)?;
        }
        for item in evidence {
            Self::add_claim_evidence_on(&tx, item)?;
        }
        tx.commit().map_err(store_err)
    }

    /// Claims linked to an entity, newest first.
    pub fn claims_for_entity(&self, entity_id: &str) -> Result<Vec<ClaimRecord>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT c.claim_id, c.claim_text, c.claim_type,
                       c.subject_entity_id, c.predicate, c.object_entity_id,
                       c.confidence, c.status, c.valid_from, c.valid_until,
                       c.attributes_json, c.source_run_id, c.created_at, c.updated_at
                FROM claims c
                JOIN claim_entities ce ON ce.claim_id = c.claim_id
                WHERE ce.entity_id = ?1
                ORDER BY c.updated_at DESC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![entity_id], |row| {
                Ok(ClaimRecord {
                    claim_id: row.get(0)?,
                    claim_text: row.get(1)?,
                    claim_type: row.get(2)?,
                    subject_entity_id: row.get(3)?,
                    predicate: row.get(4)?,
                    object_entity_id: row.get(5)?,
                    confidence: row.get::<_, f64>(6)? as f32,
                    status: row.get(7)?,
                    valid_from: row.get::<_, Option<i64>>(8)?.map(|t| t as Timestamp),
                    valid_until: row.get::<_, Option<i64>>(9)?.map(|t| t as Timestamp),
                    attributes_json: row.get(10)?,
                    source_run_id: row.get(11)?,
                    created_at: row.get::<_, i64>(12)? as Timestamp,
                    updated_at: row.get::<_, i64>(13)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Claims with evidence in a memory, newest first.
    pub fn claims_for_memory(&self, memory_id: MemoryId) -> Result<Vec<ClaimRecord>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT DISTINCT c.claim_id, c.claim_text, c.claim_type,
                       c.subject_entity_id, c.predicate, c.object_entity_id,
                       c.confidence, c.status, c.valid_from, c.valid_until,
                       c.attributes_json, c.source_run_id, c.created_at, c.updated_at
                FROM claims c
                JOIN claim_evidence ev ON ev.claim_id = c.claim_id
                WHERE ev.memory_id = ?1
                ORDER BY c.updated_at DESC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![memory_id.to_string()], |row| {
                Ok(ClaimRecord {
                    claim_id: row.get(0)?,
                    claim_text: row.get(1)?,
                    claim_type: row.get(2)?,
                    subject_entity_id: row.get(3)?,
                    predicate: row.get(4)?,
                    object_entity_id: row.get(5)?,
                    confidence: row.get::<_, f64>(6)? as f32,
                    status: row.get(7)?,
                    valid_from: row.get::<_, Option<i64>>(8)?.map(|t| t as Timestamp),
                    valid_until: row.get::<_, Option<i64>>(9)?.map(|t| t as Timestamp),
                    attributes_json: row.get(10)?,
                    source_run_id: row.get(11)?,
                    created_at: row.get::<_, i64>(12)? as Timestamp,
                    updated_at: row.get::<_, i64>(13)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Evidence rows for one claim.
    pub fn claim_evidence(&self, claim_id: &str) -> Result<Vec<ClaimEvidence>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT evidence_id, claim_id, memory_id, source_id, evidence_text,
                       span_start, span_end, confidence, created_at
                FROM claim_evidence
                WHERE claim_id = ?1
                ORDER BY confidence DESC, created_at ASC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![claim_id], |row| {
                Ok(ClaimEvidence {
                    evidence_id: row.get(0)?,
                    claim_id: row.get(1)?,
                    memory_id: parse_id::<MemoryId>(&row.get::<_, String>(2)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    source_id: row.get(3)?,
                    evidence_text: row.get(4)?,
                    span_start: row.get(5)?,
                    span_end: row.get(6)?,
                    confidence: row.get::<_, f64>(7)? as f32,
                    created_at: row.get::<_, i64>(8)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Add or update a typed relationship between two entities.
    pub fn upsert_entity_relationship(
        &self,
        relationship: &EntityRelationship,
    ) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_entity_relationship_on(&tx, relationship)?;
        tx.commit().map_err(store_err)
    }

    /// Add evidence for an entity relationship.
    pub fn add_relationship_evidence(
        &self,
        evidence: &RelationshipEvidence,
    ) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::add_relationship_evidence_on(&tx, evidence)?;
        tx.commit().map_err(store_err)
    }

    /// Persist a relationship and evidence in one transaction.
    pub fn upsert_relationship_bundle(
        &self,
        relationship: &EntityRelationship,
        evidence: &[RelationshipEvidence],
    ) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_entity_relationship_on(&tx, relationship)?;
        for item in evidence {
            Self::add_relationship_evidence_on(&tx, item)?;
        }
        tx.commit().map_err(store_err)
    }

    /// Relationships where the entity participates as source or target.
    pub fn relationships_for_entity(
        &self,
        entity_id: &str,
    ) -> Result<Vec<EntityRelationship>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT relationship_id, source_entity_id, target_entity_id,
                       relation_type, confidence, status, valid_from, valid_until,
                       attributes_json, source_run_id, created_at, updated_at
                FROM entity_relationships
                WHERE source_entity_id = ?1 OR target_entity_id = ?1
                ORDER BY updated_at DESC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![entity_id], |row| {
                Ok(EntityRelationship {
                    relationship_id: row.get(0)?,
                    source_entity_id: row.get(1)?,
                    target_entity_id: row.get(2)?,
                    relation_type: row.get(3)?,
                    confidence: row.get::<_, f64>(4)? as f32,
                    status: row.get(5)?,
                    valid_from: row.get::<_, Option<i64>>(6)?.map(|t| t as Timestamp),
                    valid_until: row.get::<_, Option<i64>>(7)?.map(|t| t as Timestamp),
                    attributes_json: row.get(8)?,
                    source_run_id: row.get(9)?,
                    created_at: row.get::<_, i64>(10)? as Timestamp,
                    updated_at: row.get::<_, i64>(11)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Evidence rows for one relationship.
    pub fn relationship_evidence(
        &self,
        relationship_id: &str,
    ) -> Result<Vec<RelationshipEvidence>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT evidence_id, relationship_id, memory_id, source_id,
                       evidence_text, span_start, span_end, confidence, created_at
                FROM relationship_evidence
                WHERE relationship_id = ?1
                ORDER BY confidence DESC, created_at ASC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![relationship_id], |row| {
                Ok(RelationshipEvidence {
                    evidence_id: row.get(0)?,
                    relationship_id: row.get(1)?,
                    memory_id: parse_id::<MemoryId>(&row.get::<_, String>(2)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    source_id: row.get(3)?,
                    evidence_text: row.get(4)?,
                    span_start: row.get(5)?,
                    span_end: row.get(6)?,
                    confidence: row.get::<_, f64>(7)? as f32,
                    created_at: row.get::<_, i64>(8)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Persist one extraction run and all derived artifacts in one transaction.
    pub fn persist_extraction_artifacts(
        &self,
        run: &ExtractionRun,
        claims: &[ClaimRecord],
        claim_entities: &[ClaimEntityLink],
        claim_evidence: &[ClaimEvidence],
        relationships: &[EntityRelationship],
        relationship_evidence: &[RelationshipEvidence],
    ) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        Self::upsert_extraction_run_on(&tx, run)?;
        for claim in claims {
            Self::upsert_claim_on(&tx, claim)?;
        }
        for link in claim_entities {
            Self::link_claim_entity_on(&tx, link)?;
        }
        for evidence in claim_evidence {
            Self::add_claim_evidence_on(&tx, evidence)?;
        }
        for relationship in relationships {
            Self::upsert_entity_relationship_on(&tx, relationship)?;
        }
        for evidence in relationship_evidence {
            Self::add_relationship_evidence_on(&tx, evidence)?;
        }
        Self::record_operation_on(
            &tx,
            "extraction_artifacts_persist",
            run.source_memory_id,
            None,
            None,
            json!({
                "run_id": run.run_id,
                "claim_count": claims.len(),
                "claim_entity_count": claim_entities.len(),
                "claim_evidence_count": claim_evidence.len(),
                "relationship_count": relationships.len(),
                "relationship_evidence_count": relationship_evidence.len(),
            }),
        )?;
        tx.commit().map_err(store_err)
    }

    /// Load a memory by id, including its tags. Returns `None` if absent.
    pub fn get_memory(&self, id: MemoryId) -> Result<Option<MemoryNode>, MenteError> {
        let conn = self.conn.lock();
        let id_str = id.to_string();
        let node: Option<MemoryNode> = conn
            .query_row(
                r#"
                SELECT id, agent_id, space_id, memory_type, content, embedding,
                       created_at, accessed_at, access_count, salience, confidence,
                       attributes, valid_from, valid_until
                FROM memories WHERE id = ?1
                "#,
                params![id_str],
                |row| {
                    let emb_blob: Option<Vec<u8>> = row.get(5)?;
                    Ok(MemoryNode {
                        id: parse_id(&row.get::<_, String>(0)?).unwrap_or_else(|_| MemoryId::nil()),
                        agent_id: parse_id(&row.get::<_, String>(1)?)
                            .unwrap_or_else(|_| AgentId::nil()),
                        space_id: parse_id(&row.get::<_, String>(2)?)
                            .unwrap_or_else(|_| SpaceId::nil()),
                        memory_type: parse_memory_type(&row.get::<_, String>(3)?),
                        content: row.get(4)?,
                        embedding: emb_blob
                            .as_deref()
                            .map(blob_to_embedding)
                            .unwrap_or_default(),
                        created_at: row.get::<_, i64>(6)? as u64,
                        accessed_at: row.get::<_, i64>(7)? as u64,
                        access_count: row.get::<_, i64>(8)? as u32,
                        salience: row.get::<_, f64>(9)? as f32,
                        confidence: row.get::<_, f64>(10)? as f32,
                        attributes: serde_json::from_str(&row.get::<_, String>(11)?)
                            .unwrap_or_default(),
                        valid_from: row.get::<_, Option<i64>>(12)?.map(|t| t as u64),
                        valid_until: row.get::<_, Option<i64>>(13)?.map(|t| t as u64),
                        tags: Vec::new(),
                    })
                },
            )
            .optional()
            .map_err(store_err)?;

        let mut node = match node {
            Some(n) => n,
            None => return Ok(None),
        };

        // Hydrate tags.
        let mut stmt = conn
            .prepare("SELECT tag FROM memory_tags WHERE memory_id = ?1 ORDER BY tag")
            .map_err(store_err)?;
        let tags: Vec<String> = stmt
            .query_map(params![id_str], |r| r.get::<_, String>(0))
            .map_err(store_err)?
            .filter_map(Result::ok)
            .collect();
        node.tags = tags;
        Ok(Some(node))
    }

    /// Delete a memory, its vector row, and its tags. Returns `true` if a row
    /// was removed.
    pub fn delete_memory(&self, id: MemoryId) -> Result<bool, MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        let id_str = id.to_string();
        let rowid: Option<i64> = tx
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![id_str],
                |r| r.get(0),
            )
            .optional()
            .map_err(store_err)?;
        let Some(rowid) = rowid else {
            return Ok(false);
        };
        let _ = tx.execute("DELETE FROM memory_vec WHERE rowid = ?1", params![rowid]);
        let tag_count = tx
            .execute(
                "DELETE FROM memory_tags WHERE memory_id = ?1",
                params![id_str],
            )
            .map_err(store_err)?;
        let edge_count = tx
            .execute(
                "DELETE FROM edges WHERE source = ?1 OR target = ?1",
                params![id_str],
            )
            .map_err(store_err)?;
        tx.execute("DELETE FROM memories WHERE id = ?1", params![id_str])
            .map_err(store_err)?;
        Self::record_operation_on(
            &tx,
            "memory_delete",
            Some(id),
            None,
            None,
            json!({
                "rowid": rowid,
                "deleted_tags": tag_count,
                "deleted_edges": edge_count,
            }),
        )?;
        tx.commit().map_err(store_err)?;
        Ok(true)
    }

    /// Total number of stored memories.
    pub fn count(&self) -> Result<usize, MenteError> {
        let conn = self.conn.lock();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .map_err(store_err)?;
        Ok(n as usize)
    }

    /// Brute-force exact KNN over the embedding index.
    ///
    /// Returns `(MemoryId, cosine_similarity)` pairs, most similar first. The
    /// query vector is normalized internally; `similarity` is in `[0, 1]`.
    pub fn knn(&self, query: &[f32], k: usize) -> Result<Vec<(MemoryId, f32)>, MenteError> {
        if k == 0 || query.is_empty() || self.embedding_dim.load(Ordering::Relaxed) == 0 {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock();
        let norm = normalize(query);
        let blob = embedding_to_blob(&norm);

        // Two steps so the join can't perturb the KNN plan:
        //   1. rowid + distance from vec0
        //   2. map rowid -> memory id
        let mut stmt = conn
            .prepare(
                "SELECT rowid, distance FROM memory_vec
                 WHERE embedding MATCH ?1
                 ORDER BY distance
                 LIMIT ?2",
            )
            .map_err(store_err)?;
        let hits: Vec<(i64, f64)> = stmt
            .query_map(params![blob.as_slice(), k as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })
            .map_err(store_err)?
            .filter_map(Result::ok)
            .collect();

        let mut out = Vec::with_capacity(hits.len());
        for (rowid, dist) in hits {
            let id_str: Option<String> = conn
                .query_row(
                    "SELECT id FROM memories WHERE rowid = ?1",
                    params![rowid],
                    |r| r.get::<_, String>(0),
                )
                .optional()
                .map_err(store_err)?;
            if let Some(id_str) = id_str
                && let Ok(id) = parse_id(&id_str)
            {
                out.push((id, l2_distance_to_similarity(dist)));
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Edges (knowledge graph), replaces the CSR/CSC graph engine.
    // -----------------------------------------------------------------------

    /// Insert a typed, weighted, optionally temporally-bounded edge between two
    /// memories. Duplicate edges are permitted (the original CSR did too).
    pub fn add_edge(&self, edge: &MemoryEdge) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        tx.execute(
            r#"INSERT INTO edges
                 (source, target, edge_type, weight, created_at, valid_from, valid_until, label)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)"#,
            params![
                edge.source.to_string(),
                edge.target.to_string(),
                edge_type_str(edge.edge_type),
                edge.weight as f64,
                edge.created_at as i64,
                edge.valid_from.map(|t| t as i64),
                edge.valid_until.map(|t| t as i64),
                edge.label.as_deref(),
            ],
        )
        .map_err(store_err)?;
        Self::record_operation_on(
            &tx,
            "edge_insert",
            None,
            Some(edge.source),
            Some(edge.target),
            json!({
                "edge_type": edge_type_str(edge.edge_type),
                "weight": edge.weight,
                "valid_from": edge.valid_from,
                "valid_until": edge.valid_until,
                "label": edge.label.as_deref(),
            }),
        )?;
        tx.commit().map_err(store_err)?;
        Ok(())
    }

    /// Load every edge in the graph. Used at open time to hydrate the in-memory
    /// `GraphManager` from the SQLite source of truth.
    pub fn all_edges(&self) -> Result<Vec<MemoryEdge>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT source, target, edge_type, weight, created_at, valid_from, valid_until, label FROM edges",
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok(MemoryEdge {
                    source: parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    target: parse_id::<MemoryId>(&row.get::<_, String>(1)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    edge_type: parse_edge_type(&row.get::<_, String>(2)?),
                    weight: row.get::<_, f64>(3)? as f32,
                    created_at: row.get::<_, i64>(4)? as u64,
                    valid_from: row.get::<_, Option<i64>>(5)?.map(|t| t as u64),
                    valid_until: row.get::<_, Option<i64>>(6)?.map(|t| t as u64),
                    label: row.get::<_, Option<String>>(7)?,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Load every edge touching `id` from the chosen direction(s).
    pub fn edges_for(&self, id: MemoryId, dir: Direction) -> Result<Vec<MemoryEdge>, MenteError> {
        let conn = self.conn.lock();
        let id_str = id.to_string();
        let sql = match dir {
            Direction::Outgoing => {
                "SELECT source, target, edge_type, weight, created_at, valid_from, valid_until, label FROM edges WHERE source = ?1"
            }
            Direction::Incoming => {
                "SELECT source, target, edge_type, weight, created_at, valid_from, valid_until, label FROM edges WHERE target = ?1"
            }
            Direction::Both => {
                "SELECT source, target, edge_type, weight, created_at, valid_from, valid_until, label FROM edges WHERE source = ?1 OR target = ?1"
            }
        };
        let mut stmt = conn.prepare(sql).map_err(store_err)?;
        let rows = stmt
            .query_map(params![id_str], |row| {
                Ok(MemoryEdge {
                    source: parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    target: parse_id::<MemoryId>(&row.get::<_, String>(1)?)
                        .unwrap_or_else(|_| MemoryId::nil()),
                    edge_type: parse_edge_type(&row.get::<_, String>(2)?),
                    weight: row.get::<_, f64>(3)? as f32,
                    created_at: row.get::<_, i64>(4)? as u64,
                    valid_from: row.get::<_, Option<i64>>(5)?.map(|t| t as u64),
                    valid_until: row.get::<_, Option<i64>>(6)?.map(|t| t as u64),
                    label: row.get::<_, Option<String>>(7)?,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Distinct neighbor ids of `id` via edges of the given types (None = all
    /// types) in the chosen direction. Drives belief propagation and
    /// contradiction traversal.
    pub fn neighbors(
        &self,
        id: MemoryId,
        edge_types: Option<&[EdgeType]>,
        dir: Direction,
    ) -> Result<Vec<MemoryId>, MenteError> {
        let edges = self.edges_for(id, dir)?;
        let type_ok = |e: &MemoryEdge| match edge_types {
            None => true,
            Some(ts) => ts.contains(&e.edge_type),
        };
        let mut out: Vec<MemoryId> = Vec::new();
        for e in edges.into_iter().filter(|e| type_ok(e)) {
            let other = if e.source == id { e.target } else { e.source };
            if !out.contains(&other) {
                out.push(other);
            }
        }
        Ok(out)
    }

    /// Traverse outwards from `roots` up to `max_depth` hops following outgoing
    /// edges of the given types (None = all). Implemented as a SQLite recursive
    /// CTE so traversal runs entirely inside the database instead of an
    /// application-side BFS over a CSR. Returns the set of reachable ids
    /// (including the roots).
    pub fn subgraph(
        &self,
        roots: &[MemoryId],
        max_depth: usize,
        edge_types: Option<&[EdgeType]>,
    ) -> Result<HashSet<MemoryId>, MenteError> {
        if roots.is_empty() {
            return Ok(HashSet::new());
        }
        let conn = self.conn.lock();

        // Base case: one `SELECT ?, 0` per root, UNION ALL'd together.
        let base: String = (0..roots.len())
            .map(|_| "SELECT ?, 0")
            .collect::<Vec<_>>()
            .join(" UNION ALL ");

        // Optional edge-type filter on the recursive arm.
        let type_filter = match edge_types {
            Some(ts) if !ts.is_empty() => {
                let ins: String = ts.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                format!(" AND e.edge_type IN ({ins})")
            }
            _ => String::new(),
        };

        let sql = format!(
            "WITH RECURSIVE reach(id, depth) AS (
                {base}
                UNION ALL
                SELECT e.target, r.depth + 1
                FROM reach r JOIN edges e ON e.source = r.id
                WHERE r.depth < ? {type_filter}
             )
             SELECT DISTINCT id FROM reach"
        );

        // Bind parameters in order: roots, depth, then edge-type labels.
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for r in roots {
            params_vec.push(Box::new(r.to_string()));
        }
        params_vec.push(Box::new(max_depth as i64));
        if let Some(ts) = edge_types {
            for t in ts {
                params_vec.push(Box::new(edge_type_str(*t).to_string()));
            }
        }

        let mut stmt = conn.prepare(&sql).map_err(store_err)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_vec.into_iter()), |row| {
                Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                    .unwrap_or_else(|_| MemoryId::nil()))
            })
            .map_err(store_err)?;
        let mut out = HashSet::new();
        for row in rows {
            out.insert(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Remove every edge that touches `id` (used when forgetting a memory).
    pub fn delete_edges_for(&self, id: MemoryId) -> Result<(), MenteError> {
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        let s = id.to_string();
        let deleted = tx
            .execute(
                "DELETE FROM edges WHERE source = ?1 OR target = ?1",
                params![s],
            )
            .map_err(store_err)?;
        Self::record_operation_on(
            &tx,
            "edge_delete_for_memory",
            Some(id),
            None,
            None,
            json!({ "deleted_edges": deleted }),
        )?;
        tx.commit().map_err(store_err)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bulk helpers, replace StorageEngine::scan_all_memories / store_memory_batch
    // -----------------------------------------------------------------------

    /// Every stored memory id (used on open instead of rebuilding a page map).
    pub fn all_memory_ids(&self) -> Result<Vec<MemoryId>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT id FROM memories").map_err(store_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                    .unwrap_or_else(|_| MemoryId::nil()))
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Every stored memory (with tags). Replaces the common facade pattern of
    /// `page_map.values().map(|pid| storage.load_memory(pid))`.
    pub fn all_memories(&self) -> Result<Vec<MemoryNode>, MenteError> {
        let ids = self.all_memory_ids()?;
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(node) = self.get_memory(id)? {
                out.push(node);
            }
        }
        Ok(out)
    }

    /// Validate and persist many nodes in one SQLite transaction. Dimensions
    /// are checked up front so a bad row aborts before any write.
    pub fn store_memory_batch(&self, nodes: &[MemoryNode]) -> Result<(), MenteError> {
        let dim = self.embedding_dim.load(Ordering::Relaxed);
        // Validate up front so a bad row aborts before any write. Only enforce
        // when a vector index exists (dim > 0); a deferred index accepts any
        // length until `ensure_vector_index` is called.
        for n in nodes {
            if dim > 0 && !n.embedding.is_empty() && n.embedding.len() != dim {
                return Err(MenteError::EmbeddingDimensionMismatch {
                    got: n.embedding.len(),
                    expected: dim,
                });
            }
        }
        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        for n in nodes {
            Self::store_memory_on(&tx, n, dim)?;
        }
        tx.commit().map_err(store_err)
    }

    /// Recent write-side audit rows, newest first.
    pub fn recent_operations(&self, limit: usize) -> Result<Vec<MemoryOperation>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT operation_id, operation_type, memory_id, source, target,
                       payload_json, created_at
                FROM memory_operations
                ORDER BY created_at DESC
                LIMIT ?1
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![usize_to_i64(limit)?], |row| {
                Ok(MemoryOperation {
                    operation_id: row.get(0)?,
                    operation_type: row.get(1)?,
                    memory_id: parse_optional_memory_id(row.get(2)?),
                    source: parse_optional_memory_id(row.get(3)?),
                    target: parse_optional_memory_id(row.get(4)?),
                    payload_json: row.get(5)?,
                    created_at: row.get::<_, i64>(6)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Recent persisted retrieval traces, newest first.
    pub fn recent_retrieval_traces(&self, limit: usize) -> Result<Vec<RetrievalTrace>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT trace_id, query_text, query_embedding_dim, k, fetch_k,
                       tags_json, tags_or, time_start, time_end,
                       candidate_count, result_count, created_at
                FROM retrieval_traces
                ORDER BY created_at DESC
                LIMIT ?1
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![usize_to_i64(limit)?], |row| {
                let time_start: Option<i64> = row.get(7)?;
                let time_end: Option<i64> = row.get(8)?;
                let time_range = match (time_start, time_end) {
                    (Some(start), Some(end)) => Some((start as Timestamp, end as Timestamp)),
                    _ => None,
                };
                Ok(RetrievalTrace {
                    trace_id: row.get(0)?,
                    query_text: row.get(1)?,
                    query_embedding_dim: row.get::<_, i64>(2)? as usize,
                    k: row.get::<_, i64>(3)? as usize,
                    fetch_k: row.get::<_, i64>(4)? as usize,
                    tags: parse_tags_json(&row.get::<_, String>(5)?),
                    tags_or: row.get::<_, i64>(6)? != 0,
                    time_range,
                    candidate_count: row.get::<_, i64>(9)? as usize,
                    result_count: row.get::<_, i64>(10)? as usize,
                    created_at: row.get::<_, i64>(11)? as Timestamp,
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Final ranked hits for a persisted retrieval trace.
    pub fn retrieval_trace_hits(
        &self,
        trace_id: &str,
    ) -> Result<Vec<RetrievalTraceHit>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                r#"
                SELECT trace_id, rank, memory_id, score, vector_rank, bm25_rank, salience
                FROM retrieval_trace_hits
                WHERE trace_id = ?1
                ORDER BY rank ASC
                "#,
            )
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![trace_id], |row| {
                let memory_id = parse_id::<MemoryId>(&row.get::<_, String>(2)?)
                    .unwrap_or_else(|_| MemoryId::nil());
                Ok(RetrievalTraceHit {
                    trace_id: row.get(0)?,
                    rank: row.get::<_, i64>(1)? as usize,
                    memory_id,
                    score: row.get::<_, f64>(3)? as f32,
                    vector_rank: row.get::<_, Option<i64>>(4)?.map(|rank| rank as usize),
                    bm25_rank: row.get::<_, Option<i64>>(5)?.map(|rank| rank as usize),
                    salience: row.get::<_, Option<f64>>(6)?.map(|score| score as f32),
                })
            })
            .map_err(store_err)?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row.map_err(store_err)?);
        }
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_retrieval_trace(
        &self,
        query_embedding_dim: usize,
        query_text: Option<&str>,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
        fetch_k: usize,
        candidate_count: usize,
        scored: &[(MemoryId, f32)],
        vector_hits: &[MemoryId],
        bm25_hits: &[MemoryId],
        config: &RetrievalConfig,
    ) -> Result<(), MenteError> {
        let trace_id = Uuid::new_v4().to_string();
        let created_at = now_us();
        let config_json = retrieval_config_json(config)?;
        let tags_json = tags_json(tags)?;
        let vector_ranks: HashMap<MemoryId, usize> = vector_hits
            .iter()
            .enumerate()
            .map(|(rank, id)| (*id, rank + 1))
            .collect();
        let bm25_ranks: HashMap<MemoryId, usize> = bm25_hits
            .iter()
            .enumerate()
            .map(|(rank, id)| (*id, rank + 1))
            .collect();

        let mut conn = self.conn.lock();
        let tx = conn.transaction().map_err(store_err)?;
        tx.execute(
            r#"
            INSERT INTO retrieval_traces (
                trace_id, query_text, query_embedding_dim, k, fetch_k,
                tags_json, tags_or, time_start, time_end, candidate_count,
                result_count, config_json, created_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
            "#,
            params![
                trace_id,
                query_text,
                usize_to_i64(query_embedding_dim)?,
                usize_to_i64(k)?,
                usize_to_i64(fetch_k)?,
                tags_json,
                if tags_or { 1_i64 } else { 0_i64 },
                time_range.map(|(start, _)| start as i64),
                time_range.map(|(_, end)| end as i64),
                usize_to_i64(candidate_count)?,
                usize_to_i64(scored.len())?,
                config_json,
                created_at as i64,
            ],
        )
        .map_err(store_err)?;

        for (rank, (id, score)) in scored.iter().enumerate() {
            let salience: Option<f64> = tx
                .query_row(
                    "SELECT salience FROM memories WHERE id = ?1",
                    params![id.to_string()],
                    |row| row.get(0),
                )
                .optional()
                .map_err(store_err)?;
            tx.execute(
                r#"
                INSERT INTO retrieval_trace_hits (
                    trace_id, rank, memory_id, score, vector_rank, bm25_rank, salience
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                "#,
                params![
                    trace_id,
                    usize_to_i64(rank + 1)?,
                    id.to_string(),
                    *score as f64,
                    vector_ranks.get(id).map(|rank| *rank as i64),
                    bm25_ranks.get(id).map(|rank| *rank as i64),
                    salience,
                ],
            )
            .map_err(store_err)?;
        }

        Self::purge_old_traces_on(&tx, config.trace_retention_limit)?;
        tx.commit().map_err(store_err)
    }

    fn purge_old_traces_on(tx: &rusqlite::Transaction<'_>, keep: usize) -> Result<(), MenteError> {
        let stale: Vec<String> = {
            let mut stmt = tx
                .prepare(
                    r#"
                    SELECT trace_id FROM retrieval_traces
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?1
                    "#,
                )
                .map_err(store_err)?;
            let rows = stmt
                .query_map(params![usize_to_i64(keep)?], |row| row.get::<_, String>(0))
                .map_err(store_err)?;
            let mut stale = Vec::new();
            for row in rows {
                stale.push(row.map_err(store_err)?);
            }
            stale
        };
        for trace_id in stale {
            tx.execute(
                "DELETE FROM retrieval_trace_hits WHERE trace_id = ?1",
                params![trace_id],
            )
            .map_err(store_err)?;
            tx.execute(
                "DELETE FROM retrieval_traces WHERE trace_id = ?1",
                params![trace_id],
            )
            .map_err(store_err)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Hybrid search, replaces IndexManager (RRF over vec0 + FTS5 BM25).
    // -----------------------------------------------------------------------

    /// Vector + optional BM25 hybrid search (no tag filter). Mirrors
    /// `IndexManager::hybrid_search`.
    pub fn hybrid_search(
        &self,
        query_embedding: &[f32],
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.hybrid_search_with_query_mode(query_embedding, None, tags, false, time_range, k)
    }

    /// Hybrid search with an optional text query for BM25 matching.
    pub fn hybrid_search_with_query(
        &self,
        query_embedding: &[f32],
        query_text: Option<&str>,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.hybrid_search_with_query_mode(query_embedding, query_text, tags, false, time_range, k)
    }

    /// Hybrid search with configurable tag mode (AND vs OR). Mirrors
    /// `IndexManager::hybrid_search_with_query_mode`.
    ///
    /// Algorithm: take an over-fetched vector candidate set, optionally take
    /// BM25 candidates from FTS5, merge via Reciprocal Rank Fusion, drop
    /// anything outside the tag/time filter, then boost by salience and the
    /// configured recency prior before truncating to `k`.
    pub fn hybrid_search_with_query_mode(
        &self,
        query_embedding: &[f32],
        query_text: Option<&str>,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
        k: usize,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        if k == 0 || query_embedding.is_empty() {
            return Ok(Vec::new());
        }
        let config = self.retrieval_config();
        let fetch_multiplier = config.fetch_multiplier.max(1);
        let fetch_k = k.saturating_mul(fetch_multiplier).max(k);

        // 1) Candidate filter set from tags + time window.
        let candidate_set = self.candidate_set(tags, tags_or, time_range)?;
        // Filters requested but nothing matches, so there are no results.
        if matches!(candidate_set.as_ref().map(|s| s.len()), Some(0)) {
            if self.retrieval_tracing_enabled()
                && let Err(e) = self.record_retrieval_trace(
                    query_embedding.len(),
                    query_text,
                    tags,
                    tags_or,
                    time_range,
                    k,
                    fetch_k,
                    0,
                    &[],
                    &[],
                    &[],
                    &config,
                )
            {
                tracing::warn!("failed to record empty retrieval trace: {e}");
            }
            return Ok(Vec::new());
        }

        // 2) Vector candidates: brute-force over the candidate set when filters
        //    are present (so restrictive filters still return k results),
        //    otherwise a plain vec0 KNN.
        let vector_hits: Vec<MemoryId> = match &candidate_set {
            Some(cs) => self.vector_search_filtered(query_embedding, cs, fetch_k)?,
            None => self
                .knn(query_embedding, fetch_k)?
                .into_iter()
                .map(|(id, _)| id)
                .collect(),
        };

        // 3) BM25 candidates from FTS5 (when a text query is provided).
        let bm25_hits: Vec<MemoryId> = match query_text {
            Some(qt) if !qt.is_empty() => self.fts_search(qt, fetch_k)?,
            _ => Vec::new(),
        };

        if vector_hits.is_empty() && bm25_hits.is_empty() {
            if self.retrieval_tracing_enabled()
                && let Err(e) = self.record_retrieval_trace(
                    query_embedding.len(),
                    query_text,
                    tags,
                    tags_or,
                    time_range,
                    k,
                    fetch_k,
                    0,
                    &[],
                    &vector_hits,
                    &bm25_hits,
                    &config,
                )
            {
                tracing::warn!("failed to record empty retrieval trace: {e}");
            }
            return Ok(Vec::new());
        }

        // 4) Reciprocal Rank Fusion.
        let mut rrf: HashMap<MemoryId, f32> = HashMap::new();
        for (rank, id) in vector_hits.iter().enumerate() {
            *rrf.entry(*id).or_insert(0.0) += 1.0 / (config.rrf_k + rank as f32);
        }
        for (rank, id) in bm25_hits.iter().enumerate() {
            *rrf.entry(*id).or_insert(0.0) += 1.0 / (config.rrf_k + rank as f32);
        }
        let candidate_count = rrf.len();

        // 5) Apply the candidate filter (drops bm25 hits that fall outside it)
        //    and add the salience + recency boost.
        let mut scored: Vec<(MemoryId, f32)> = Vec::with_capacity(rrf.len());
        for (id, rrf_score) in rrf {
            if let Some(cs) = &candidate_set
                && !cs.contains(&id)
            {
                continue;
            }
            let salience = self.salience_of(id)?.unwrap_or(0.5);
            let combined = rrf_score * config.rrf_weight
                + salience * config.salience_weight
                + config.recency_prior * config.recency_weight;
            scored.push((id, combined));
        }

        // 6) Rank and truncate.
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        if self.retrieval_tracing_enabled()
            && let Err(e) = self.record_retrieval_trace(
                query_embedding.len(),
                query_text,
                tags,
                tags_or,
                time_range,
                k,
                fetch_k,
                candidate_count,
                &scored,
                &vector_hits,
                &bm25_hits,
                &config,
            )
        {
            tracing::warn!("failed to record retrieval trace: {e}");
        }
        Ok(scored)
    }

    // --- hybrid search helpers ---------------------------------------------

    /// Build the candidate id set imposed by tag + time filters. Returns
    /// `None` when no filters were requested, `Some(empty)` when filters were
    /// requested but matched nothing.
    fn candidate_set(
        &self,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Option<HashSet<MemoryId>>> {
        match (tags, time_range) {
            (None, None) => Ok(None),
            (Some(t), None) => Ok(Some(self.ids_matching_tags(t, tags_or)?)),
            (None, Some((start, end))) => Ok(Some(self.ids_matching_time(start, end)?)),
            (Some(t), Some((start, end))) => {
                let tag_set = self.ids_matching_tags(t, tags_or)?;
                let time_set = self.ids_matching_time(start, end)?;
                Ok(Some(tag_set.intersection(&time_set).copied().collect()))
            }
        }
    }

    /// Memory ids carrying the requested tags. `or = true` means union (any
    /// tag), `or = false` means intersection (all tags).
    fn ids_matching_tags(&self, tags: &[&str], or: bool) -> MenteResult<HashSet<MemoryId>> {
        if tags.is_empty() {
            return Ok(HashSet::new());
        }
        let conn = self.conn.lock();
        let placeholders = tags.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = if or {
            format!("SELECT DISTINCT memory_id FROM memory_tags WHERE tag IN ({placeholders})")
        } else {
            format!(
                "SELECT memory_id FROM memory_tags WHERE tag IN ({placeholders})
                 GROUP BY memory_id HAVING COUNT(DISTINCT tag) = ?"
            )
        };
        let mut params_vec: Vec<Box<dyn rusqlite::ToSql>> = tags
            .iter()
            .map(|t| Box::new((*t).to_string()) as Box<dyn rusqlite::ToSql>)
            .collect();
        if !or {
            params_vec.push(Box::new(tags.len() as i64));
        }
        let mut stmt = conn.prepare(&sql).map_err(store_err)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_vec.into_iter()), |row| {
                Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                    .unwrap_or_else(|_| MemoryId::nil()))
            })
            .map_err(store_err)?;
        let mut out = HashSet::new();
        for row in rows {
            out.insert(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Memory ids whose `created_at` falls within `[start, end]` inclusive.
    fn ids_matching_time(
        &self,
        start: Timestamp,
        end: Timestamp,
    ) -> MenteResult<HashSet<MemoryId>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare("SELECT id FROM memories WHERE created_at >= ?1 AND created_at <= ?2")
            .map_err(store_err)?;
        let rows = stmt
            .query_map(params![start as i64, end as i64], |row| {
                Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                    .unwrap_or_else(|_| MemoryId::nil()))
            })
            .map_err(store_err)?;
        let mut out = HashSet::new();
        for row in rows {
            out.insert(row.map_err(store_err)?);
        }
        Ok(out)
    }

    /// Brute-force cosine ranking over a candidate set, returning the top `k`
    /// ids most similar to `query`. Used when tag/time filters are present so
    /// that restrictive filters don't starve the global vec0 KNN.
    fn vector_search_filtered(
        &self,
        query: &[f32],
        candidates: &HashSet<MemoryId>,
        k: usize,
    ) -> MenteResult<Vec<MemoryId>> {
        if candidates.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock();
        let placeholders = candidates.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = format!(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND id IN ({placeholders})"
        );
        let params_vec: Vec<Box<dyn rusqlite::ToSql>> = candidates
            .iter()
            .map(|id| Box::new(id.to_string()) as Box<dyn rusqlite::ToSql>)
            .collect();
        let mut stmt = conn.prepare(&sql).map_err(store_err)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params_vec.into_iter()), |row| {
                let id = parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                    .unwrap_or_else(|_| MemoryId::nil());
                let emb: Vec<u8> = row.get(1)?;
                Ok((id, blob_to_embedding(&emb)))
            })
            .map_err(store_err)?;

        let qnorm = normalize(query);
        let mut scored: Vec<(MemoryId, f32)> = Vec::new();
        for row in rows {
            let (id, emb) = row.map_err(store_err)?;
            scored.push((id, dot(&qnorm, &normalize(&emb))));
        }
        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored.into_iter().map(|(id, _)| id).collect())
    }

    /// FTS5 BM25 search over `memories.content`. Returns ids best-matching the
    /// query text (most relevant first). A malformed MATCH query yields an
    /// empty result rather than failing the whole hybrid search.
    fn fts_search(&self, query_text: &str, k: usize) -> MenteResult<Vec<MemoryId>> {
        let conn = self.conn.lock();
        let mut stmt = conn
            .prepare(
                "SELECT m.id FROM memories_fts f
                 JOIN memories m ON m.rowid = f.rowid
                 WHERE memories_fts MATCH ?1
                 ORDER BY bm25(memories_fts)
                 LIMIT ?2",
            )
            .map_err(store_err)?;
        let rows_result = stmt.query_map(params![query_text, k as i64], |row| {
            Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?).unwrap_or_else(|_| MemoryId::nil()))
        });
        let mut out = Vec::new();
        match rows_result {
            Ok(rows) => {
                for row in rows {
                    match row {
                        Ok(id) => out.push(id),
                        // MATCH syntax error or bad token, treat as no matches.
                        Err(_) => {
                            out.clear();
                            break;
                        }
                    }
                }
            }
            Err(_) => return Ok(Vec::new()),
        }
        Ok(out)
    }

    /// Read one memory's salience, or `None` if it does not exist.
    fn salience_of(&self, id: MemoryId) -> MenteResult<Option<f32>> {
        let conn = self.conn.lock();
        let value: Option<f64> = conn
            .query_row(
                "SELECT salience FROM memories WHERE id = ?1",
                params![id.to_string()],
                |row| row.get(0),
            )
            .optional()
            .map_err(store_err)?;
        Ok(value.map(|v| v as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::types::{AgentId, MemoryId};

    fn make_node(id: MemoryId, content: &str, embedding: Vec<f32>) -> MemoryNode {
        let mut n = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            content.to_string(),
            embedding,
        );
        n.id = id;
        n.tags = vec!["pref".to_string(), "ui".to_string()];
        n.salience = 0.8;
        n
    }

    #[test]
    fn store_and_get_roundtrips() {
        let db = Backend::open_in_memory(3).unwrap();
        let id = MemoryId::new();
        let node = make_node(id, "User prefers dark mode", vec![0.1, 0.2, 0.3]);
        db.store_memory(&node).unwrap();

        let loaded = db.get_memory(id).unwrap().expect("memory should exist");
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.content, "User prefers dark mode");
        assert_eq!(loaded.embedding, vec![0.1, 0.2, 0.3]); // original preserved
        assert_eq!(loaded.memory_type, MemoryType::Semantic);
        assert_eq!(loaded.salience, 0.8);
        assert_eq!(loaded.tags, vec!["pref".to_string(), "ui".to_string()]);
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn store_upserts_on_same_id() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        db.store_memory(&make_node(id, "first", vec![1.0, 0.0]))
            .unwrap();
        db.store_memory(&make_node(id, "second", vec![0.0, 1.0]))
            .unwrap();
        assert_eq!(db.count().unwrap(), 1);
        let loaded = db.get_memory(id).unwrap().unwrap();
        assert_eq!(loaded.content, "second");
        assert_eq!(loaded.embedding, vec![0.0, 1.0]);
    }

    #[test]
    fn delete_removes_memory_and_vector() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        db.store_memory(&make_node(id, "gone", vec![1.0, 0.0]))
            .unwrap();
        assert!(db.delete_memory(id).unwrap());
        assert!(db.get_memory(id).unwrap().is_none());
        assert!(db.knn(&[1.0, 0.0], 5).unwrap().is_empty());
    }

    #[test]
    fn knn_returns_nearest_first() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        // Three points on the unit circle.
        db.store_memory(&make_node(a, "east", vec![1.0, 0.0]))
            .unwrap();
        db.store_memory(&make_node(b, "north", vec![0.0, 1.0]))
            .unwrap();
        db.store_memory(&make_node(c, "sw", vec![-0.7, -0.7]))
            .unwrap();

        // Query near "east": a must rank first, c last.
        let hits = db.knn(&[0.9, 0.1], 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, a, "east should be nearest");
        assert!(
            hits[0].1 > 0.95,
            "east similarity should be ~1, got {}",
            hits[0].1
        );
        assert_eq!(hits[2].0, c, "sw should be farthest");
        assert!(hits[2].1 < hits[0].1);
    }

    #[test]
    fn rejects_mismatched_embedding_dimension() {
        let db = Backend::open_in_memory(3).unwrap();
        let id = MemoryId::new();
        let res = db.store_memory(&make_node(id, "bad", vec![0.1, 0.2]));
        assert!(matches!(
            res,
            Err(MenteError::EmbeddingDimensionMismatch {
                got: 2,
                expected: 3
            })
        ));
    }

    #[test]
    fn open_file_backed_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mem.sqlite");
        let id = MemoryId::new();
        {
            let db = Backend::open(&path, 2).unwrap();
            db.store_memory(&make_node(id, "persisted", vec![1.0, 0.0]))
                .unwrap();
        }
        let db = Backend::open(&path, 2).unwrap();
        let loaded = db.get_memory(id).unwrap().expect("should survive reopen");
        assert_eq!(loaded.content, "persisted");
    }

    // --- edges + graph traversal ------------------------------------------

    fn edge(src: MemoryId, tgt: MemoryId, et: EdgeType, w: f32) -> MemoryEdge {
        MemoryEdge {
            source: src,
            target: tgt,
            edge_type: et,
            weight: w,
            created_at: 1000,
            valid_from: None,
            valid_until: None,
            label: None,
        }
    }

    #[test]
    fn edges_roundtrip_and_directions() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        db.add_edge(&edge(a, b, EdgeType::Related, 0.5)).unwrap();
        db.add_edge(&edge(c, a, EdgeType::Supports, 0.9)).unwrap();

        let out = db.edges_for(a, Direction::Outgoing).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].target, b);

        let inc = db.edges_for(a, Direction::Incoming).unwrap();
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0].source, c);

        let both = db.edges_for(a, Direction::Both).unwrap();
        assert_eq!(both.len(), 2);
    }

    #[test]
    fn neighbors_filtered_by_edge_type() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        db.add_edge(&edge(a, b, EdgeType::Related, 0.5)).unwrap();
        db.add_edge(&edge(a, c, EdgeType::Contradicts, 0.9))
            .unwrap();

        let related = db
            .neighbors(a, Some(&[EdgeType::Related]), Direction::Outgoing)
            .unwrap();
        assert_eq!(related, vec![b]);

        let contradicts = db
            .neighbors(a, Some(&[EdgeType::Contradicts]), Direction::Outgoing)
            .unwrap();
        assert_eq!(contradicts, vec![c]);

        let all = db.neighbors(a, None, Direction::Outgoing).unwrap();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn subgraph_traverses_outgoing_edges() {
        let db = Backend::open_in_memory(2).unwrap();
        // Chain a -> b -> c, plus a -> d (unrelated type to filter out).
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        let d = MemoryId::new();
        db.add_edge(&edge(a, b, EdgeType::Caused, 1.0)).unwrap();
        db.add_edge(&edge(b, c, EdgeType::Caused, 1.0)).unwrap();
        db.add_edge(&edge(a, d, EdgeType::Related, 0.1)).unwrap();

        // Follow Caused only, from a, depth 5: reaches {a, b, c}, excludes d.
        let reach = db.subgraph(&[a], 5, Some(&[EdgeType::Caused])).unwrap();
        assert!(reach.contains(&a));
        assert!(reach.contains(&b));
        assert!(reach.contains(&c));
        assert!(!reach.contains(&d), "Related edge to d must be excluded");

        // Depth cap: depth 0 reaches only the root.
        let depth0 = db.subgraph(&[a], 0, None).unwrap();
        assert_eq!(depth0.len(), 1);
        assert!(depth0.contains(&a));
    }

    #[test]
    fn delete_edges_for_removes_both_directions() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        db.add_edge(&edge(a, b, EdgeType::Related, 0.5)).unwrap();
        db.delete_edges_for(a).unwrap();
        assert!(db.edges_for(a, Direction::Both).unwrap().is_empty());
        assert!(db.edges_for(b, Direction::Both).unwrap().is_empty());
    }

    // --- batch + scan ------------------------------------------------------

    #[test]
    fn batch_store_and_all_ids() {
        let db = Backend::open_in_memory(2).unwrap();
        let ids: Vec<MemoryId> = (0..3).map(|_| MemoryId::new()).collect();
        let nodes: Vec<MemoryNode> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| make_node(*id, &format!("n{i}"), vec![i as f32, 1.0]))
            .collect();
        db.store_memory_batch(&nodes).unwrap();
        let mut all = db.all_memory_ids().unwrap();
        all.sort();
        let mut expected = ids.clone();
        expected.sort();
        assert_eq!(all, expected);
        assert_eq!(db.count().unwrap(), 3);
    }

    #[test]
    fn batch_rejects_bad_dimension_before_writing() {
        let db = Backend::open_in_memory(2).unwrap();
        let good = make_node(MemoryId::new(), "ok", vec![1.0, 1.0]);
        let bad = make_node(MemoryId::new(), "bad", vec![1.0, 1.0, 1.0]);
        let res = db.store_memory_batch(&[good, bad]);
        assert!(res.is_err());
        // Nothing should have been written.
        assert_eq!(db.count().unwrap(), 0);
    }

    // --- hybrid search -----------------------------------------------------

    #[test]
    fn hybrid_vector_only_ranks_nearest_first() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        db.store_memory(&make_node(a, "east text", vec![1.0, 0.0]))
            .unwrap();
        db.store_memory(&make_node(b, "north text", vec![0.0, 1.0]))
            .unwrap();

        let res = db.hybrid_search(&[0.95, 0.05], None, None, 2).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0, a, "east should rank first");
    }

    #[test]
    fn hybrid_with_bm25_finds_by_keyword() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        // `a` is vector-far but keyword-matches "postgres"; `b` is vector-close
        // but has no matching keyword. RRF must still surface `a`.
        db.store_memory(&make_node(a, "postgres database migration", vec![0.0, 1.0]))
            .unwrap();
        db.store_memory(&make_node(b, "unrelated content", vec![1.0, 0.0]))
            .unwrap();

        let res = db
            .hybrid_search_with_query(&[1.0, 0.0], Some("postgres"), None, None, 2)
            .unwrap();
        assert!(
            res.iter().any(|(id, _)| *id == a),
            "bm25 match should surface a despite low vector similarity"
        );
    }

    #[test]
    fn write_operations_are_recorded() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        db.store_memory(&make_node(a, "alpha", vec![1.0, 0.0]))
            .unwrap();
        db.add_edge(&edge(a, b, EdgeType::Related, 0.5)).unwrap();
        db.delete_memory(a).unwrap();

        let operations = db.recent_operations(10).unwrap();
        assert!(
            operations
                .iter()
                .any(|op| op.operation_type == "memory_upsert")
        );
        assert!(
            operations
                .iter()
                .any(|op| op.operation_type == "edge_insert")
        );
        assert!(
            operations
                .iter()
                .any(|op| op.operation_type == "memory_delete")
        );
    }

    #[test]
    fn memory_sources_roundtrip_with_store() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        let node = make_node(id, "source tracked", vec![1.0, 0.0]);
        let mut source = MemorySource::new(id, "conversation_turn");
        source.conversation_id = Some("conv-1".to_string());
        source.turn_id = Some("turn-7".to_string());
        source.actor_id = Some("user".to_string());
        source.payload_json = json!({"raw": "source tracked"}).to_string();

        db.store_memory_with_source(&node, &source).unwrap();

        let sources = db.memory_sources(id).unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].source_type, "conversation_turn");
        assert_eq!(sources[0].conversation_id.as_deref(), Some("conv-1"));

        let operations = db.recent_operations(10).unwrap();
        assert!(
            operations
                .iter()
                .any(|op| op.operation_type == "memory_source_upsert")
        );
    }

    #[test]
    fn entities_aliases_and_memory_links_roundtrip() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        db.store_memory(&make_node(id, "Synapse uses MCP", vec![1.0, 0.0]))
            .unwrap();

        let mut entity = EntityRecord::new("project", "Synapse");
        entity.confidence = 0.95;
        let entity_id = entity.entity_id.clone();
        db.upsert_entity(&entity).unwrap();
        db.add_entity_alias(&EntityAlias {
            entity_id: entity_id.clone(),
            alias: "synapse".to_string(),
            source: Some("test".to_string()),
            confidence: 0.9,
        })
        .unwrap();
        db.link_memory_entity(&MemoryEntityLink {
            memory_id: id,
            entity_id: entity_id.clone(),
            role: Some("subject".to_string()),
            confidence: 0.88,
            evidence: Some("Synapse".to_string()),
        })
        .unwrap();

        let by_alias = db.entities_by_alias("synapse").unwrap();
        assert_eq!(by_alias.len(), 1);
        assert_eq!(by_alias[0].canonical, "Synapse");

        let links = db.memory_entity_links(id).unwrap();
        assert_eq!(links.len(), 1);
        assert_eq!(links[0].entity_id, entity_id);
        assert_eq!(links[0].role.as_deref(), Some("subject"));
    }

    #[test]
    fn conversation_events_roundtrip() {
        let db = Backend::open_in_memory(2).unwrap();
        let mut conversation = ConversationRecord::new("conv-1");
        conversation.title = Some("Synapse planning".to_string());
        conversation.metadata_json = json!({"source": "test"}).to_string();
        db.upsert_conversation(&conversation).unwrap();

        let mut join = ConversationEvent::new("conv-1", "participant_joined");
        join.actor_id = Some("pratap".to_string());
        join.payload_json = json!({"role": "owner"}).to_string();
        join.observed_at = 10;
        let mut message = ConversationEvent::new("conv-1", "user_message");
        message.turn_id = Some("turn-1".to_string());
        message.actor_id = Some("pratap".to_string());
        message.content = Some("Synapse should remember Flutter preferences".to_string());
        message.observed_at = 20;

        db.add_conversation_event(&message).unwrap();
        db.add_conversation_event(&join).unwrap();

        let events = db.conversation_events("conv-1", 10).unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_type, "participant_joined");
        assert_eq!(events[1].turn_id.as_deref(), Some("turn-1"));

        let operations = db.recent_operations(10).unwrap();
        assert!(
            operations
                .iter()
                .any(|op| op.operation_type == "conversation_event_upsert")
        );
    }

    #[test]
    fn extraction_runs_roundtrip_with_source_memory() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        db.store_memory(&make_node(id, "raw episode", vec![1.0, 0.0]))
            .unwrap();

        let mut run = ExtractionRun::new("claim_extractor", "v1");
        run.source_memory_id = Some(id);
        run.extractor_version = "2026-07-06".to_string();
        run.model = Some("local-test-model".to_string());
        run.prompt_hash = Some("prompt-sha".to_string());
        run.config_hash = Some("config-sha".to_string());
        run.status = "completed".to_string();
        run.output_json = json!({"claims": 1}).to_string();
        run.completed_at = Some(run.started_at + 1);

        db.upsert_extraction_run(&run).unwrap();

        let by_memory = db.extraction_runs_for_memory(id).unwrap();
        assert_eq!(by_memory.len(), 1);
        assert_eq!(by_memory[0].run_id, run.run_id);
        assert_eq!(by_memory[0].status, "completed");

        let recent = db.recent_extraction_runs(10).unwrap();
        assert_eq!(recent[0].extractor, "claim_extractor");
    }

    #[test]
    fn claims_relationships_and_evidence_roundtrip() {
        let db = Backend::open_in_memory(2).unwrap();
        let memory_id = MemoryId::new();
        db.store_memory(&make_node(
            memory_id,
            "Pratap uses Flutter for Synapse.",
            vec![1.0, 0.0],
        ))
        .unwrap();

        let person = EntityRecord::new("person", "Pratap");
        let person_id = person.entity_id.clone();
        let project = EntityRecord::new("project", "Synapse");
        let project_id = project.entity_id.clone();
        let tech = EntityRecord::new("technology", "Flutter");
        let tech_id = tech.entity_id.clone();
        db.upsert_entity_bundle(&[person, project, tech], &[], &[])
            .unwrap();

        let mut run = ExtractionRun::new("claim_extractor", "v1");
        run.source_memory_id = Some(memory_id);
        run.status = "completed".to_string();
        db.upsert_extraction_run(&run).unwrap();

        let mut claim = ClaimRecord::new("Pratap uses Flutter for Synapse", "fact");
        claim.subject_entity_id = Some(person_id.clone());
        claim.predicate = Some("uses".to_string());
        claim.object_entity_id = Some(tech_id.clone());
        claim.source_run_id = Some(run.run_id.clone());
        claim.confidence = 0.92;
        let claim_id = claim.claim_id.clone();

        let mut evidence = ClaimEvidence::new(claim_id.clone(), memory_id);
        evidence.evidence_text = Some("Pratap uses Flutter for Synapse.".to_string());
        evidence.span_start = Some(0);
        evidence.span_end = Some(34);
        evidence.confidence = 0.9;

        db.upsert_claim_bundle(
            &claim,
            &[
                ClaimEntityLink {
                    claim_id: claim_id.clone(),
                    entity_id: person_id.clone(),
                    role: "subject".to_string(),
                    confidence: 0.92,
                },
                ClaimEntityLink {
                    claim_id: claim_id.clone(),
                    entity_id: tech_id.clone(),
                    role: "object".to_string(),
                    confidence: 0.92,
                },
            ],
            &[evidence],
        )
        .unwrap();

        let by_entity = db.claims_for_entity(&person_id).unwrap();
        assert_eq!(by_entity.len(), 1);
        assert_eq!(by_entity[0].claim_id, claim_id);

        let by_memory = db.claims_for_memory(memory_id).unwrap();
        assert_eq!(by_memory.len(), 1);
        assert_eq!(by_memory[0].predicate.as_deref(), Some("uses"));

        let evidence_rows = db.claim_evidence(&claim_id).unwrap();
        assert_eq!(evidence_rows.len(), 1);
        assert_eq!(evidence_rows[0].span_start, Some(0));

        let mut relationship = EntityRelationship::new(project_id.clone(), tech_id.clone(), "uses");
        relationship.source_run_id = Some(run.run_id.clone());
        relationship.confidence = 0.91;
        let relationship_id = relationship.relationship_id.clone();
        let mut relationship_evidence =
            RelationshipEvidence::new(relationship_id.clone(), memory_id);
        relationship_evidence.evidence_text = Some("Synapse uses Flutter".to_string());

        db.upsert_relationship_bundle(&relationship, &[relationship_evidence])
            .unwrap();

        let project_relationships = db.relationships_for_entity(&project_id).unwrap();
        assert_eq!(project_relationships.len(), 1);
        assert_eq!(project_relationships[0].relation_type, "uses");

        let rel_evidence = db.relationship_evidence(&relationship_id).unwrap();
        assert_eq!(rel_evidence.len(), 1);

        let operations = db.recent_operations(20).unwrap();
        assert!(
            operations
                .iter()
                .any(|op| op.operation_type == "claim_upsert")
        );
        assert!(
            operations
                .iter()
                .any(|op| op.operation_type == "entity_relationship_upsert")
        );
    }

    #[test]
    fn retrieval_tracing_records_rank_inputs() {
        let db = Backend::open_in_memory(2).unwrap();
        db.set_retrieval_tracing(true);
        let a = MemoryId::new();
        let b = MemoryId::new();
        db.store_memory(&make_node(a, "postgres database migration", vec![0.0, 1.0]))
            .unwrap();
        db.store_memory(&make_node(b, "unrelated content", vec![1.0, 0.0]))
            .unwrap();

        let res = db
            .hybrid_search_with_query(&[1.0, 0.0], Some("postgres"), None, None, 2)
            .unwrap();
        assert!(!res.is_empty());

        let traces = db.recent_retrieval_traces(1).unwrap();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].query_text.as_deref(), Some("postgres"));
        assert_eq!(traces[0].result_count, res.len());

        let hits = db.retrieval_trace_hits(&traces[0].trace_id).unwrap();
        assert_eq!(hits.len(), res.len());
        assert!(hits.iter().any(|hit| hit.bm25_rank.is_some()));
    }

    #[test]
    fn hybrid_tag_filter_restricts_results() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let mut na = make_node(a, "alpha", vec![1.0, 0.0]);
        na.tags = vec!["x".into()];
        let mut nb = make_node(b, "beta", vec![0.9, 0.1]);
        nb.tags = vec!["y".into()];
        db.store_memory(&na).unwrap();
        db.store_memory(&nb).unwrap();

        let res = db
            .hybrid_search(&[1.0, 0.0], Some(&["x"]), None, 10)
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, a);
    }

    #[test]
    fn hybrid_tag_and_mode_requires_all_tags() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let mut na = make_node(a, "both", vec![1.0, 0.0]);
        na.tags = vec!["x".into(), "y".into()];
        let mut nb = make_node(b, "one", vec![0.9, 0.1]);
        nb.tags = vec!["x".into()];
        db.store_memory(&na).unwrap();
        db.store_memory(&nb).unwrap();

        // AND mode: only `a` has both x and y.
        let res = db
            .hybrid_search_with_query_mode(&[1.0, 0.0], None, Some(&["x", "y"]), false, None, 10)
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, a);

        // OR mode: both have at least x.
        let res_or = db
            .hybrid_search_with_query_mode(&[1.0, 0.0], None, Some(&["x", "y"]), true, None, 10)
            .unwrap();
        assert_eq!(res_or.len(), 2);
    }

    #[test]
    fn hybrid_time_filter_restricts_results() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let mut na = make_node(a, "old", vec![1.0, 0.0]);
        na.created_at = 100;
        let mut nb = make_node(b, "new", vec![0.9, 0.1]);
        nb.created_at = 500;
        db.store_memory(&na).unwrap();
        db.store_memory(&nb).unwrap();

        let res = db
            .hybrid_search(&[1.0, 0.0], None, Some((400, 600)), 10)
            .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, b);
    }

    // --- deferred vector index --------------------------------------------

    #[test]
    fn deferred_index_backfills_on_ensure_vector_index() {
        // Open with dim 0 (no embedder configured yet): embeddings are still
        // stored, but there is no vector index so KNN returns nothing.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mem.sqlite");
        let id = MemoryId::new();

        {
            let db = Backend::open(&path, 0).unwrap();
            assert_eq!(db.embedding_dim(), 0);
            db.store_memory(&make_node(id, "deferred", vec![1.0, 0.0]))
                .unwrap();
            // No vec0 yet, KNN is unavailable.
            assert!(db.knn(&[1.0, 0.0], 5).unwrap().is_empty());
            // Configuring the embedder builds the index and backfills.
            db.ensure_vector_index(2).unwrap();
            assert_eq!(db.embedding_dim(), 2);
            let hits = db.knn(&[1.0, 0.0], 5).unwrap();
            assert_eq!(hits.len(), 1);
            assert_eq!(hits[0].0, id);
        }

        // On reopen the stored dim wins, so the index is ready immediately.
        let db = Backend::open(&path, 0).unwrap();
        assert_eq!(db.embedding_dim(), 2);
        let hits = db.knn(&[1.0, 0.0], 5).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, id);
    }
}
