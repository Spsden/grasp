//! SQLite + sqlite-vec storage backend for MenteDB.
//!
//! This crate replaces the bespoke page/WAL/HNSW/CSR storage engine with a
//! single SQLite database file:
//!
//!   * `memories`    — one row per `MemoryNode`, with ordinary B-tree indexes
//!                     on the columns used for filtering (`agent_id`, `space_id`,
//!                     `memory_type`, `salience`, `created_at`).
//!   * `memory_tags` — a junction table replacing the roaring-bitmap tag index
//!                     (`WHERE tag IN (...)`).
//!   * `memory_vec`  — a `sqlite-vec` `vec0` virtual table for brute-force
//!                     exact KNN over embeddings. It is a *derived* index: the
//!                     source-of-truth bytes live in `memories.embedding`, and
//!                     the vec0 table is keyed by the implicit `memories.rowid`
//!                     so it can always be rebuilt from the source rows.
//!   * `schema_meta` — key/value row holding the schema version and the
//!                     embedding dimension the vec0 table was created with.
//!
//! All access goes through a single `parking_lot::Mutex<Connection>`. SQLite's
//! own WAL handles crash recovery; the custom WAL/page-manager/buffer-pool are
//! gone. This is the substrate the facade will eventually talk to instead of
//! `StorageEngine` + `IndexManager` + `GraphManager`.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};

use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{AgentId, MemoryId, SpaceId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode, MenteError};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};

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
    use rusqlite::ffi::sqlite3_auto_extension;
    use std::sync::Once;
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the documented entry point of the
        // sqlite-vec extension and has the `sqlite3_loadext_entry` signature
        // SQLite expects. The transmute mirrors the crate's own test.
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
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
    /// The dimension the `memory_vec` vec0 table was created with, or `0` when
    /// no vector index exists yet (deferred until an embedder is configured).
    /// Atomic because [`Backend::ensure_vector_index`] mutates it through
    /// `&self` once the facade learns the embedding dimension.
    embedding_dim: AtomicUsize,
}

impl Backend {
    /// Open (or create) a file-backed database at `path`.
    pub fn open(path: &Path, embedding_dim: usize) -> Result<Self, MenteError> {
        // Ensure the parent directory exists so a fresh path works.
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(store_err)?;
            }
        }
        ensure_vec0_registered();
        let mut conn = Connection::open(path).map_err(store_err)?;
        // WAL: concurrent readers don't block the writer, and crash recovery is
        // handled by SQLite instead of the old custom WAL.
        conn.pragma_update(None, "journal_mode", "WAL").map_err(store_err)?;
        let effective = Self::init(&mut conn, embedding_dim)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedding_dim: AtomicUsize::new(effective),
        })
    }

    /// Open an ephemeral in-memory database (used by tests and quick spikes).
    pub fn open_in_memory(embedding_dim: usize) -> Result<Self, MenteError> {
        ensure_vec0_registered();
        let mut conn = Connection::open_in_memory().map_err(store_err)?;
        let effective = Self::init(&mut conn, embedding_dim)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedding_dim: AtomicUsize::new(effective),
        })
    }

    /// The dimension the vector index was created with (0 = deferred / absent).
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim.load(Ordering::Relaxed)
    }

    /// Create (or recreate) the vec0 vector index at `dim` and backfill it from
    /// the embeddings already stored in `memories.embedding`. Called by the
    /// facade once an embedder is configured. No-op if the index already exists
    /// at the same dimension.
    pub fn ensure_vector_index(&self, dim: usize) -> Result<(), MenteError> {
        if dim == 0 || dim == self.embedding_dim.load(Ordering::Relaxed) {
            return Ok(());
        }
        let conn = self.conn.lock();
        // Drop any existing vec0 table (possibly at a different dim) and
        // recreate at the requested dimension.
        let _ = conn.execute("DROP TABLE IF EXISTS memory_vec", []);
        conn.execute(
            &format!("CREATE VIRTUAL TABLE memory_vec USING vec0(embedding float[{dim}])"),
            [],
        )
        .map_err(store_err)?;

        // Backfill from stored embeddings (normalized).
        let mut select = conn
            .prepare("SELECT rowid, embedding FROM memories WHERE embedding IS NOT NULL")
            .map_err(store_err)?;
        let rows = select
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .map_err(store_err)?;
        let mut pairs: Vec<(i64, Vec<u8>)> = Vec::new();
        for row in rows {
            pairs.push(row.map_err(store_err)?);
        }
        drop(select);
        for (rowid, bytes) in pairs {
            let emb = blob_to_embedding(&bytes);
            if emb.len() != dim {
                continue; // skip rows that don't match the new dim
            }
            let norm = normalize(&emb);
            let blob = embedding_to_blob(&norm);
            conn.execute(
                "INSERT INTO memory_vec (rowid, embedding) VALUES (?1, ?2)",
                params![rowid, blob.as_slice()],
            )
            .map_err(store_err)?;
        }
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('embedding_dim', ?1)",
            params![dim.to_string()],
        )
        .map_err(store_err)?;

        self.embedding_dim.store(dim, Ordering::Relaxed);
        Ok(())
    }

    /// Create all tables/indexes. `hint_dim` is used only on first creation; a
    /// stored `embedding_dim` in `schema_meta` wins on reopen so the vec0
    /// table survives across launches. Returns the effective dimension (0 means
    /// "deferred — no vector index yet").
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
            "#,
        )
        .map_err(store_err)?;

        // Populate FTS from any pre-existing rows (no-op on a fresh open).
        // FTS5's 'rebuild' command rescans the external content table.
        let _ = tx.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')", []);

        // Resolve the effective vector dimension: a previously-stored dim wins
        // (so the vec0 table survives reopen), otherwise fall back to the hint.
        // `0` means "deferred" — no embedder configured yet, so no vec0 table
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
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('embedding_dim', ?1)",
            params![effective.to_string()],
        )
        .map_err(store_err)?;

        tx.commit().map_err(store_err)?;
        Ok(effective)
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

        let conn = self.conn.lock();
        let attrs_json = serde_json::to_string(&node.attributes).map_err(store_err)?;
        let emb_blob = if node.embedding.is_empty() {
            None
        } else {
            Some(embedding_to_blob(&node.embedding))
        };

        let id_str = node.id.to_string();

        // Upsert the memory row. `ON CONFLICT(id) DO UPDATE` preserves the
        // implicit rowid, which keeps the vec0 key stable across re-stores.
        conn.execute(
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
        let rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![id_str],
                |r| r.get(0),
            )
            .map_err(store_err)?;
        let _ = conn.execute(
            "DELETE FROM memory_vec WHERE rowid = ?1",
            params![rowid],
        );
        if dim > 0 && !node.embedding.is_empty() {
            // Store a normalized copy so L2 KNN approximates cosine ranking.
            let norm = normalize(&node.embedding);
            let norm_blob = embedding_to_blob(&norm);
            let _ = &emb_blob; // original bytes already stored in memories.embedding
            conn.execute(
                "INSERT INTO memory_vec (rowid, embedding) VALUES (?1, ?2)",
                params![rowid, norm_blob.as_slice()],
            )
            .map_err(store_err)?;
        }

        // Refresh tags (delete + reinsert is simplest and correct).
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id = ?1",
            params![id_str],
        )
        .map_err(store_err)?;
        for tag in &node.tags {
            conn.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?1, ?2)",
                params![id_str, tag],
            )
            .map_err(store_err)?;
        }

        Ok(())
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
                        agent_id: parse_id(&row.get::<_, String>(1)?).unwrap_or_else(|_| AgentId::nil()),
                        space_id: parse_id(&row.get::<_, String>(2)?).unwrap_or_else(|_| SpaceId::nil()),
                        memory_type: parse_memory_type(&row.get::<_, String>(3)?),
                        content: row.get(4)?,
                        embedding: emb_blob.as_deref().map(blob_to_embedding).unwrap_or_default(),
                        created_at: row.get::<_, i64>(6)? as u64,
                        accessed_at: row.get::<_, i64>(7)? as u64,
                        access_count: row.get::<_, i64>(8)? as u32,
                        salience: row.get::<_, f64>(9)? as f32,
                        confidence: row.get::<_, f64>(10)? as f32,
                        attributes: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
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
        let conn = self.conn.lock();
        let id_str = id.to_string();
        let rowid: Option<i64> = conn
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
        conn.execute("DELETE FROM memories WHERE id = ?1", params![id_str])
            .map_err(store_err)?;
        let _ = conn.execute("DELETE FROM memory_vec WHERE rowid = ?1", params![rowid]);
        conn.execute("DELETE FROM memory_tags WHERE memory_id = ?1", params![id_str])
            .map_err(store_err)?;
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
            if let Some(id_str) = id_str {
                if let Ok(id) = parse_id(&id_str) {
                    out.push((id, l2_distance_to_similarity(dist)));
                }
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Edges (knowledge graph) — replaces the CSR/CSC graph engine.
    // -----------------------------------------------------------------------

    /// Insert a typed, weighted, optionally temporally-bounded edge between two
    /// memories. Duplicate edges are permitted (the original CSR did too).
    pub fn add_edge(&self, edge: &MemoryEdge) -> Result<(), MenteError> {
        let conn = self.conn.lock();
        conn.execute(
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
                edge.label,
            ],
        )
        .map_err(store_err)?;
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
            Direction::Outgoing => "SELECT source, target, edge_type, weight, created_at, valid_from, valid_until, label FROM edges WHERE source = ?1",
            Direction::Incoming => "SELECT source, target, edge_type, weight, created_at, valid_from, valid_until, label FROM edges WHERE target = ?1",
            Direction::Both => "SELECT source, target, edge_type, weight, created_at, valid_from, valid_until, label FROM edges WHERE source = ?1 OR target = ?1",
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
                Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?).unwrap_or_else(|_| MemoryId::nil()))
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
        let conn = self.conn.lock();
        let s = id.to_string();
        conn.execute("DELETE FROM edges WHERE source = ?1 OR target = ?1", params![s])
            .map_err(store_err)?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Bulk helpers — replace StorageEngine::scan_all_memories / store_memory_batch
    // -----------------------------------------------------------------------

    /// Every stored memory id (used on open instead of rebuilding a page map).
    pub fn all_memory_ids(&self) -> Result<Vec<MemoryId>, MenteError> {
        let conn = self.conn.lock();
        let mut stmt = conn.prepare("SELECT id FROM memories").map_err(store_err)?;
        let rows = stmt
            .query_map([], |row| {
                Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?).unwrap_or_else(|_| MemoryId::nil()))
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

    /// Validate and persist many nodes. Dimensions are checked up front so a
    /// bad row aborts before any write. Each node goes through [`store_memory`]
    /// (its own implicit transaction); for true single-transaction batching we
    /// would fold the per-node SQL under one `tx`, but per-item cost is
    /// dominated by SQLite I/O anyway.
    ///
    /// [`store_memory`]: Backend::store_memory
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
        for n in nodes {
            self.store_memory(n)?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Hybrid search — replaces IndexManager (RRF over vec0 + FTS5 BM25).
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
    /// Algorithm (faithful port of the IndexManager fusion): take the top
    /// `k*4` vector candidates (vec0 KNN, or brute-force cosine over the
    /// filter candidate set when tag/time filters are present), take the top
    /// `k*4` BM25 candidates from FTS5, merge via Reciprocal Rank Fusion
    /// (rrf_k = 60), drop anything outside the candidate set, then boost by
    /// salience (×0.05) and a fixed recency term (×0.02 × 0.5) before
    /// truncating to `k`.
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
        let fetch_k = k * 4;
        let rrf_k: f32 = 60.0;

        // 1) Candidate filter set from tags + time window.
        let candidate_set = self.candidate_set(tags, tags_or, time_range)?;
        // Filters requested but nothing matches → no results.
        if matches!(candidate_set.as_ref().map(|s| s.len()), Some(0)) {
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
            return Ok(Vec::new());
        }

        // 4) Reciprocal Rank Fusion.
        let mut rrf: HashMap<MemoryId, f32> = HashMap::new();
        for (rank, id) in vector_hits.iter().enumerate() {
            *rrf.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }
        for (rank, id) in bm25_hits.iter().enumerate() {
            *rrf.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
        }

        // 5) Apply the candidate filter (drops bm25 hits that fall outside it)
        //    and add the salience + recency boost.
        let mut scored: Vec<(MemoryId, f32)> = Vec::with_capacity(rrf.len());
        for (id, rrf_score) in rrf {
            if let Some(cs) = &candidate_set {
                if !cs.contains(&id) {
                    continue;
                }
            }
            let salience = self.salience_of(id)?.unwrap_or(0.5);
            let recency = 0.5f32;
            let combined = rrf_score * 0.7 + salience * 0.05 + recency * 0.02;
            scored.push((id, combined));
        }

        // 6) Rank and truncate.
        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
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

    /// Memory ids carrying the requested tags. `or = true` → union (any tag),
    /// `or = false` → intersection (all tags).
    fn ids_matching_tags(&self, tags: &[&str], or: bool) -> MenteResult<HashSet<MemoryId>> {
        if tags.is_empty() {
            return Ok(HashSet::new());
        }
        let conn = self.conn.lock();
        let placeholders = tags.iter().map(|_| "?").collect::<Vec<_>>().join(",");
        let sql = if or {
            format!(
                "SELECT DISTINCT memory_id FROM memory_tags WHERE tag IN ({placeholders})"
            )
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
    fn ids_matching_time(&self, start: Timestamp, end: Timestamp) -> MenteResult<HashSet<MemoryId>> {
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
        scored.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
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
            Ok(parse_id::<MemoryId>(&row.get::<_, String>(0)?)
                .unwrap_or_else(|_| MemoryId::nil()))
        });
        let mut out = Vec::new();
        match rows_result {
            Ok(rows) => {
                for row in rows {
                    match row {
                        Ok(id) => out.push(id),
                        // MATCH syntax error / bad token → treat as no matches.
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
        let mut n = MemoryNode::new(AgentId::new(), MemoryType::Semantic, content.to_string(), embedding);
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
        db.store_memory(&make_node(id, "first", vec![1.0, 0.0])).unwrap();
        db.store_memory(&make_node(id, "second", vec![0.0, 1.0])).unwrap();
        assert_eq!(db.count().unwrap(), 1);
        let loaded = db.get_memory(id).unwrap().unwrap();
        assert_eq!(loaded.content, "second");
        assert_eq!(loaded.embedding, vec![0.0, 1.0]);
    }

    #[test]
    fn delete_removes_memory_and_vector() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        db.store_memory(&make_node(id, "gone", vec![1.0, 0.0])).unwrap();
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
        db.store_memory(&make_node(a, "east",  vec![1.0, 0.0])).unwrap();
        db.store_memory(&make_node(b, "north", vec![0.0, 1.0])).unwrap();
        db.store_memory(&make_node(c, "sw",    vec![-0.7, -0.7])).unwrap();

        // Query near "east": a must rank first, c last.
        let hits = db.knn(&[0.9, 0.1], 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, a, "east should be nearest");
        assert!(hits[0].1 > 0.95, "east similarity should be ~1, got {}", hits[0].1);
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
            Err(MenteError::EmbeddingDimensionMismatch { got: 2, expected: 3 })
        ));
    }

    #[test]
    fn open_file_backed_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mem.sqlite");
        let id = MemoryId::new();
        {
            let db = Backend::open(&path, 2).unwrap();
            db.store_memory(&make_node(id, "persisted", vec![1.0, 0.0])).unwrap();
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
        db.add_edge(&edge(a, c, EdgeType::Contradicts, 0.9)).unwrap();

        let related = db.neighbors(a, Some(&[EdgeType::Related]), Direction::Outgoing).unwrap();
        assert_eq!(related, vec![b]);

        let contradicts = db.neighbors(a, Some(&[EdgeType::Contradicts]), Direction::Outgoing).unwrap();
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
        db.store_memory(&make_node(a, "east text", vec![1.0, 0.0])).unwrap();
        db.store_memory(&make_node(b, "north text", vec![0.0, 1.0])).unwrap();

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
        db.store_memory(&make_node(a, "postgres database migration", vec![0.0, 1.0])).unwrap();
        db.store_memory(&make_node(b, "unrelated content", vec![1.0, 0.0])).unwrap();

        let res =
            db.hybrid_search_with_query(&[1.0, 0.0], Some("postgres"), None, None, 2).unwrap();
        assert!(
            res.iter().any(|(id, _)| *id == a),
            "bm25 match should surface a despite low vector similarity"
        );
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

        let res = db.hybrid_search(&[1.0, 0.0], Some(&["x"]), None, 10).unwrap();
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
        let res =
            db.hybrid_search_with_query_mode(&[1.0, 0.0], None, Some(&["x", "y"]), false, None, 10)
                .unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, a);

        // OR mode: both have at least x.
        let res_or =
            db.hybrid_search_with_query_mode(&[1.0, 0.0], None, Some(&["x", "y"]), true, None, 10)
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

        let res = db.hybrid_search(&[1.0, 0.0], None, Some((400, 600)), 10).unwrap();
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
            db.store_memory(&make_node(id, "deferred", vec![1.0, 0.0])).unwrap();
            // No vec0 yet → KNN unavailable.
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
