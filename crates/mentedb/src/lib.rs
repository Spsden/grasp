//! # MenteDB: The Mind Database for AI Agents
//!
//! MenteDB is a purpose-built database engine for AI agent memory.
//! It's a cognition preparation engine that pre-digests knowledge
//! for single-pass transformer consumption.
//!
//! ## Core Concepts
//!
//! - **MemoryNode**: The atomic unit of knowledge (embeddings, graph, temporal, attributes)
//! - **MemoryEdge**: Typed, weighted relationships between memories
//! - **MemoryTier**: Cognitive inspired storage hierarchy (working, episodic, semantic, procedural, archival)
//! - **Context Assembly**: Token budget aware context building that respects attention patterns
//! - **MQL**: Mente Query Language for memory retrieval and manipulation
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use mentedb::prelude::*;
//! use mentedb::MenteDb;
//! use std::path::Path;
//!
//! let mut db = MenteDb::open(Path::new("./my-agent-memory")).unwrap();
//! // store, recall, relate, forget...
//! db.close().unwrap();
//! ```
//!
//! ## Feature Highlights
//!
//! - **Unified `process_turn`** pipeline: single call handles context retrieval,
//!   pain signals, episodic storage, write inference, action detection, sentiment,
//!   phantom tracking, trajectory, speculative caching, fact extraction, and
//!   auto-maintenance (decay / archival / consolidation)
//! - Seven cognitive features: interference detection, pain signals, phantom tracking,
//!   speculative caching, stream monitoring, trajectory tracking, write inference
//! - SQLite + sqlite-vec storage with hybrid search (vector + BM25 + tags + temporal + salience)
//! - Durable edge table with an in-memory graph mirror for belief propagation
//! - Token budget aware context assembly with attention curve optimization
//! - MQL query language with vector, tag, temporal, and graph traversal support
//! - SQLite WAL crash recovery plus explicit write and retrieval inspection tables
//!
//! ## Repository
//!
//! Source code: <https://github.com/nambok/mentedb>

use std::path::{Path, PathBuf};

use entity_extraction::{ExtractedEntity, RuleBasedEntityExtractor};
use mentedb_cognitive::EntityResolver;
use mentedb_cognitive::interference::{InterferenceDetector, InterferencePair};
use mentedb_cognitive::llm::EntityMergeGroup;
use mentedb_cognitive::pain::{PainRegistry, PainSignal};
use mentedb_cognitive::phantom::{PhantomConfig, PhantomMemory, PhantomTracker};
use mentedb_cognitive::speculative::{CacheEntry, CacheStats, SpeculativeCache};
use mentedb_cognitive::stream::{CognitionStream, StreamAlert, StreamConfig};
use mentedb_cognitive::trajectory::{TrajectoryNode, TrajectoryTracker};
use mentedb_cognitive::write_inference::{
    InferredAction, WriteInferenceConfig, WriteInferenceEngine,
};
use mentedb_consolidation::archival::{ArchivalConfig, ArchivalDecision, ArchivalPipeline};
use mentedb_consolidation::compression::{CompressedMemory, MemoryCompressor};
use mentedb_consolidation::consolidation::{ConsolidationCandidate, ConsolidationEngine};
use mentedb_consolidation::decay::{DecayConfig, DecayEngine};
use mentedb_context::{AssemblyConfig, ContextAssembler, ContextWindow, ScoredMemory};
use mentedb_core::edge::EdgeType;
use mentedb_core::error::MenteResult;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::{MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode, MenteError};
use mentedb_embedding::provider::EmbeddingProvider;
use mentedb_graph::GraphManager;
use mentedb_query::{Mql, QueryPlan};
use mentedb_sqlite::Backend;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

// Re-export sub-crates for direct access.

/// Engine version, derived from Cargo.toml at compile time.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Cognitive pipeline: speculative caching, trajectory tracking, inference.
pub use mentedb_cognitive as cognitive;
/// Consolidation, decay, and memory lifecycle management.
pub use mentedb_consolidation as consolidation;
/// Context assembly engine.
pub use mentedb_context as context;
/// Core types: MemoryNode, MemoryEdge, errors, config.
pub use mentedb_core as core;
/// Knowledge graph engine.
pub use mentedb_graph as graph;
/// Index structures for vector, tag, temporal, and salience search.
pub use mentedb_index as index;
/// MQL parser and query planner.
pub use mentedb_query as query;
/// SQLite storage backend and inspection types.
pub use mentedb_sqlite as sqlite;
pub use mentedb_sqlite::{
    ClaimEntityLink, ClaimEvidence, ClaimRecord, ConversationEvent, ConversationRecord,
    EntityAlias, EntityRecord, EntityRelationship, ExtractionRun, MemoryEntityLink,
    MemoryOperation, MemorySource, RelationshipEvidence, RetrievalConfig, RetrievalTrace,
    RetrievalTraceHit,
};

/// Renderer-neutral graph projection DTOs for app clients.
pub mod entity_extraction;
/// Validated extraction job boundary for derived claims and relationships.
pub mod extraction_jobs;
pub mod graph_projection;
/// Unified process_turn orchestration.
pub mod process_turn;
/// Bounded sleep maintenance for app background workers.
pub mod sleep;

/// Sleeptime enrichment pipeline (requires `enrichment` feature).
#[cfg(feature = "enrichment")]
pub mod enrichment;

pub use entity_extraction::{EntityExtractionConfig, ExtractedEntity as ExtractedMemoryEntity};
pub use extraction_jobs::{ValidatedExtractionBatch, validate_extraction_batch};
pub use graph_projection::{
    GraphProjection, GraphProjectionConfig, GraphProjectionEdge, GraphProjectionNode,
};
pub use sleep::{
    SleepMaintenanceConfig, SleepMaintenanceIssue, SleepMaintenanceLease, SleepMaintenanceResult,
    SleepMaintenanceStage,
};

/// Commonly used types, re-exported for convenience.
pub mod prelude {
    pub use mentedb_core::edge::EdgeType;
    pub use mentedb_core::error::MenteResult;
    pub use mentedb_core::memory::MemoryType;
    pub use mentedb_core::types::*;
    pub use mentedb_core::{MemoryEdge, MemoryNode, MemoryTier, MenteError};
    pub use mentedb_sqlite::{
        ClaimEntityLink, ClaimEvidence, ClaimRecord, ConversationEvent, ConversationRecord,
        EntityAlias, EntityRecord, EntityRelationship, ExtractionRun, MemoryEntityLink,
        MemoryOperation, MemorySource, RelationshipEvidence, RetrievalConfig, RetrievalTrace,
        RetrievalTraceHit,
    };

    pub use crate::MenteDb;
    pub use crate::entity_extraction::{
        EntityExtractionConfig, ExtractedEntity as ExtractedMemoryEntity,
    };
    pub use crate::extraction_jobs::{ValidatedExtractionBatch, validate_extraction_batch};
}

use std::collections::HashMap;

/// Configuration for sleeptime enrichment pipeline.
///
/// Enrichment runs BETWEEN conversations, never in the hot path.
/// The engine tracks state and provides candidates; callers invoke
/// the async LLM pipeline when ready.
#[derive(Debug, Clone)]
pub struct EnrichmentConfig {
    /// Whether enrichment is enabled. Default: false (opt-in).
    pub enabled: bool,
    /// Run enrichment after this many process_turn calls. Default: 50.
    pub trigger_interval: u64,
    /// Minimum confidence for extracted memories to be stored. Default: 0.6.
    pub min_confidence: f32,
    /// Maximum confidence for enrichment-generated memories. Default: 0.7.
    pub max_enrichment_confidence: f32,
    /// Whether to generate a user model summary. Default: false.
    pub enable_user_model: bool,
    /// Embedding similarity threshold to merge entities. Default: 0.7.
    pub entity_merge_threshold: f32,
    /// Embedding similarity below which entities are kept separate. Default: 0.4.
    pub entity_separate_threshold: f32,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            trigger_interval: 50,
            min_confidence: 0.6,
            max_enrichment_confidence: 0.7,
            enable_user_model: false,
            entity_merge_threshold: 0.7,
            entity_separate_threshold: 0.4,
        }
    }
}

/// Result of running the enrichment pipeline.
#[derive(Debug, Clone, Default)]
pub struct EnrichmentResult {
    /// Number of new memories stored from extraction.
    pub memories_stored: usize,
    /// Number of entity nodes created or updated.
    pub entities_processed: usize,
    /// Number of edges created (Derived, Related, PartOf).
    pub edges_created: usize,
    /// Number of memories skipped as duplicates.
    pub duplicates_skipped: usize,
    /// Number of contradictions detected.
    pub contradictions_found: usize,
    /// Turn ID at which enrichment was completed.
    pub completed_at_turn: u64,
    /// Number of entity links created (Related edges between same-name entities).
    pub entities_linked: usize,
    /// Number of entity pairs left ambiguous (below merge threshold).
    pub entities_ambiguous: usize,
}

/// Result of a single entity linking run.
#[derive(Debug, Clone, Default)]
pub struct EntityLinkResult {
    /// Number of entity pairs linked with Related edges.
    pub linked: usize,
    /// Number of entity pairs tagged as ambiguous (MaybeRelated).
    pub ambiguous: usize,
    /// Number of edges created.
    pub edges_created: usize,
}

/// A confirmed entity resolution from an external resolver (LLM).
///
/// Used to feed LLM entity resolution results back into the engine
/// so it can create graph edges and update the EntityResolver cache.
#[derive(Debug, Clone)]
pub struct EntityLinkResolution {
    /// The canonical entity name decided by the resolver.
    pub canonical: String,
    /// All aliases that map to this canonical name.
    pub aliases: Vec<String>,
    /// Confidence in this resolution (0.0 to 1.0).
    pub confidence: f32,
}

/// A pair of entity names that the LLM confirmed are DIFFERENT entities.
#[derive(Debug, Clone)]
pub struct EntitySeparation {
    pub name_a: String,
    pub name_b: String,
}

/// Configuration for the cognitive engine subsystems.
#[derive(Debug, Clone)]
pub struct CognitiveConfig {
    /// Whether write inference (auto-edges, contradiction detection) is enabled on store.
    pub write_inference: bool,
    /// Whether salience decay is applied during retrieval.
    pub decay_on_recall: bool,
    /// Whether pain tracking is enabled.
    pub pain_tracking: bool,
    /// Whether interference detection is available.
    pub interference_detection: bool,
    /// Whether phantom tracking is enabled.
    pub phantom_tracking: bool,
    /// Whether speculative caching is enabled.
    pub speculative_cache: bool,
    /// Whether archival evaluation is available.
    pub archival_evaluation: bool,
    /// Configuration for the write inference engine.
    pub inference_config: WriteInferenceConfig,
    /// Configuration for the decay engine.
    pub decay_config: DecayConfig,
    /// Configuration for phantom tracking.
    pub phantom_config: PhantomConfig,
    /// Configuration for the archival pipeline.
    pub archival_config: ArchivalConfig,
    /// Configuration for the cognition stream.
    pub stream_config: StreamConfig,
    /// Configuration for sleeptime enrichment.
    pub enrichment_config: EnrichmentConfig,
    /// Configuration for deterministic entity surface linking and entity-aware recall.
    pub entity_extraction_config: EntityExtractionConfig,
    /// Similarity threshold for interference detection.
    pub interference_threshold: f32,
    /// Maximum trajectory turns to track.
    pub trajectory_max_turns: usize,
    /// Maximum speculative cache entries.
    pub speculative_cache_size: usize,
    /// Maximum pain signals to retain.
    pub pain_max_warnings: usize,
}

impl Default for CognitiveConfig {
    fn default() -> Self {
        Self {
            write_inference: true,
            decay_on_recall: true,
            pain_tracking: true,
            interference_detection: true,
            phantom_tracking: true,
            speculative_cache: true,
            archival_evaluation: true,
            inference_config: WriteInferenceConfig::default(),
            decay_config: DecayConfig::default(),
            phantom_config: PhantomConfig::default(),
            archival_config: ArchivalConfig::default(),
            stream_config: StreamConfig::default(),
            enrichment_config: EnrichmentConfig::default(),
            entity_extraction_config: EntityExtractionConfig::default(),
            interference_threshold: 0.8,
            trajectory_max_turns: 100,
            speculative_cache_size: 10,
            pain_max_warnings: 5,
        }
    }
}

/// The unified database facade for MenteDB.
///
/// `MenteDb` coordinates storage, indexing, graph relationships, query parsing,
/// context assembly, and cognitive subsystems into a single coherent API.
///
/// All internal state is protected by fine-grained locks, so every public method
/// takes `&self`. This allows `Arc<MenteDb>` to be shared across threads without
/// an external `RwLock`.
pub struct MenteDb {
    /// SQLite + sqlite-vec storage/index backend. Replaces StorageEngine +
    /// IndexManager + the MemoryId->PageId map. The single source of truth for
    /// memories, vectors, tags, and edges.
    db: Backend,
    /// In-memory knowledge graph (CSR/CSC), hydrated from `db` on open and
    /// kept write-through on `relate`. Kept in-memory because the cognitive
    /// engines (belief propagation, contradiction traversal, supersede
    /// filtering) read it on every recall.
    graph: GraphManager,
    /// Expected embedding dimension (0 = no embedder configured yet).
    embedding_dim: usize,
    /// Database directory path for persistence.
    path: PathBuf,
    /// Optional embedding provider for auto-embedding on store and search.
    embedder: Option<Box<dyn EmbeddingProvider>>,
    /// Cognitive engine configuration.
    cognitive_config: CognitiveConfig,
    /// Write inference engine for auto-edge creation and contradiction detection.
    write_inference: WriteInferenceEngine,
    /// Decay engine for salience management.
    decay: DecayEngine,
    /// Consolidation engine for memory merging.
    consolidation: ConsolidationEngine,
    /// Pain registry for tracking recurring failures.
    pain: RwLock<PainRegistry>,
    /// Trajectory tracker for conversation patterns.
    trajectory: RwLock<TrajectoryTracker>,
    /// Cognition stream for token-level monitoring.
    stream: CognitionStream,
    /// Phantom tracker for detecting referenced-but-missing knowledge.
    phantom: RwLock<PhantomTracker>,
    /// Speculative cache for pre-fetching likely-needed memories.
    speculative: RwLock<SpeculativeCache>,
    /// Interference detector for finding confusable memories.
    interference: InterferenceDetector,
    /// Entity resolver for canonical name resolution.
    entity_resolver: RwLock<EntityResolver>,
    /// Memory compressor for content summarization.
    compressor: MemoryCompressor,
    /// Archival pipeline for lifecycle evaluation.
    archival: ArchivalPipeline,
    /// Turn ID of the last completed enrichment cycle.
    last_enrichment_turn: RwLock<u64>,
    /// Whether enrichment is currently pending (set by maintenance trigger).
    enrichment_pending: RwLock<bool>,
}

impl MenteDb {
    /// Opens (or creates) a MenteDB instance at the given path.
    pub fn open(path: &Path) -> MenteResult<Self> {
        Self::open_with_config(path, CognitiveConfig::default())
    }

    /// Opens a MenteDB instance with custom cognitive configuration.
    pub fn open_with_config(path: &Path, cognitive_config: CognitiveConfig) -> MenteResult<Self> {
        info!("Opening MenteDB at {}", path.display());
        std::fs::create_dir_all(path)?;
        // All durable state lives in one SQLite file. Open with dim 0
        // (deferred); the vec0 index is (re)created when an embedder is
        // attached via set_embedder / open_with_embedder.
        let db = Backend::open(&path.join("memory.sqlite"), 0)?;

        // Hydrate the in-memory graph from the SQLite edge store (source of
        // truth) and register every memory as a graph node.
        let graph = GraphManager::new();
        let memory_ids = db.all_memory_ids()?;
        for id in &memory_ids {
            graph.add_memory(*id);
        }
        for edge in db.all_edges()? {
            // add_relationship persists nothing here; SQLite already holds the
            // edge. We mirror it into the CSR for the graph algorithms.
            let _ = graph.add_relationship(&edge);
        }
        if !memory_ids.is_empty() {
            info!(
                memories = memory_ids.len(),
                "hydrated graph from sqlite store"
            );
        }

        let write_inference =
            WriteInferenceEngine::with_config(cognitive_config.inference_config.clone());
        let decay = DecayEngine::new(cognitive_config.decay_config.clone());
        let consolidation = ConsolidationEngine::new();
        let pain = RwLock::new(PainRegistry::new(cognitive_config.pain_max_warnings));
        let trajectory = RwLock::new(TrajectoryTracker::new(
            cognitive_config.trajectory_max_turns,
        ));
        let stream = CognitionStream::with_config(cognitive_config.stream_config.clone());
        let phantom = RwLock::new(PhantomTracker::new(cognitive_config.phantom_config.clone()));
        let speculative = RwLock::new(SpeculativeCache::new(
            cognitive_config.speculative_cache_size,
            0.5,
            0.4,
        ));
        let interference = InterferenceDetector::new(cognitive_config.interference_threshold);
        let entity_resolver = RwLock::new(EntityResolver::new());
        let compressor = MemoryCompressor::new();
        let archival = ArchivalPipeline::new(cognitive_config.archival_config.clone());

        // Load persisted state for subsystems that support it.
        let cognitive_dir = path.join("cognitive");
        if cognitive_dir.exists() {
            let _ = trajectory
                .write()
                .transitions
                .load(&cognitive_dir.join("transitions.json"));
            let _ = speculative
                .write()
                .load(&cognitive_dir.join("speculative.json"));
            let _ = entity_resolver
                .write()
                .load(&cognitive_dir.join("entities.json"));
        }

        Ok(Self {
            db,
            graph,
            embedding_dim: 0,
            path: path.to_path_buf(),
            embedder: None,
            cognitive_config,
            write_inference,
            decay,
            consolidation,
            pain,
            trajectory,
            stream,
            phantom,
            speculative,
            interference,
            entity_resolver,
            compressor,
            archival,
            last_enrichment_turn: RwLock::new(0),
            enrichment_pending: RwLock::new(false),
        })
    }

    /// Opens a MenteDB instance with a configured embedding provider.
    pub fn open_with_embedder(
        path: &Path,
        embedder: Box<dyn EmbeddingProvider>,
    ) -> MenteResult<Self> {
        let mut db = Self::open(path)?;
        let dim = embedder.dimensions();
        db.embedding_dim = dim;
        db.db.ensure_vector_index(dim)?;
        db.embedder = Some(embedder);
        Ok(db)
    }

    /// Opens a MenteDB instance with both embedder and cognitive config.
    pub fn open_with_embedder_and_config(
        path: &Path,
        embedder: Box<dyn EmbeddingProvider>,
        cognitive_config: CognitiveConfig,
    ) -> MenteResult<Self> {
        let mut db = Self::open_with_config(path, cognitive_config)?;
        db.set_embedder(embedder);
        Ok(db)
    }

    /// Set the embedding provider after construction and (re)build the vec0
    /// vector index at the provider's dimension, backfilling any memories
    /// already stored.
    pub fn set_embedder(&mut self, embedder: Box<dyn EmbeddingProvider>) {
        self.embedding_dim = embedder.dimensions();
        let dim = self.embedding_dim;
        self.embedder = Some(embedder);
        if let Err(e) = self.db.ensure_vector_index(dim) {
            warn!("Failed to (re)build vector index at dim {dim}: {e}");
        }
    }

    /// Generate an embedding for the given text using the configured provider.
    /// Returns None if no provider is configured.
    pub fn embed_text(&self, text: &str) -> MenteResult<Option<Vec<f32>>> {
        match &self.embedder {
            Some(e) => Ok(Some(e.embed(text)?)),
            None => Ok(None),
        }
    }

    /// Stores a memory node into the database.
    ///
    /// The node is persisted to storage, added to all indexes, and registered
    /// in the graph for relationship traversal.
    ///
    /// When cognitive features are enabled (the default), write inference
    /// automatically runs to:
    /// - Detect contradictions with existing memories
    /// - Create relationship edges (Related, Supersedes, Contradicts)
    /// - Invalidate superseded memories
    /// - Propagate confidence changes through the graph
    pub fn store(&self, node: MemoryNode) -> MenteResult<()> {
        let id = node.id;
        debug!("Storing memory {}", id);

        // Auto-provision the vector index from the first embedding seen. This
        // covers callers that embed themselves without configuring an
        // EmbeddingProvider (tests, bulk import, direct API use).
        if !node.embedding.is_empty() && self.db.embedding_dim() == 0 {
            self.db.ensure_vector_index(node.embedding.len())?;
        }

        // Validate embedding dimension when configured.
        let dim = self.db.embedding_dim();
        if dim > 0 && !node.embedding.is_empty() && node.embedding.len() != dim {
            return Err(MenteError::EmbeddingDimensionMismatch {
                got: node.embedding.len(),
                expected: dim,
            });
        }

        // Persist (upsert) the memory, its vector row, and its tags. SQLite
        // triggers keep the FTS5 mirror in sync automatically.
        self.db.store_memory(&node)?;
        self.graph.add_memory(id);
        if let Err(e) = self.index_memory_entities(&node) {
            warn!(memory_id = %id, error = %e, "failed to index memory entities");
        }

        // Run write inference to auto-create edges and detect contradictions.
        if self.cognitive_config.write_inference {
            self.run_write_inference(&node);
        }

        Ok(())
    }

    /// Stores a memory with explicit provenance in the same SQLite transaction.
    pub fn store_with_source(&self, node: MemoryNode, source: MemorySource) -> MenteResult<()> {
        let id = node.id;
        debug!("Storing memory {} with provenance", id);

        if !node.embedding.is_empty() && self.db.embedding_dim() == 0 {
            self.db.ensure_vector_index(node.embedding.len())?;
        }

        let dim = self.db.embedding_dim();
        if dim > 0 && !node.embedding.is_empty() && node.embedding.len() != dim {
            return Err(MenteError::EmbeddingDimensionMismatch {
                got: node.embedding.len(),
                expected: dim,
            });
        }

        self.db.store_memory_with_source(&node, &source)?;
        self.graph.add_memory(id);
        if let Err(e) = self.index_memory_entities(&node) {
            warn!(memory_id = %id, error = %e, "failed to index memory entities");
        }

        if self.cognitive_config.write_inference {
            self.run_write_inference(&node);
        }

        Ok(())
    }

    /// Store multiple memories in a single SQLite transaction.
    pub fn store_batch(&self, nodes: Vec<MemoryNode>) -> MenteResult<Vec<MemoryId>> {
        // Auto-provision the vector index from the first embedding seen.
        if let Some(first_dim) = nodes.iter().find_map(|n| {
            if n.embedding.is_empty() {
                None
            } else {
                Some(n.embedding.len())
            }
        }) && self.db.embedding_dim() == 0
        {
            self.db.ensure_vector_index(first_dim)?;
        }

        let dim = self.db.embedding_dim();
        // Validate all embeddings upfront
        for node in &nodes {
            if dim > 0 && !node.embedding.is_empty() && node.embedding.len() != dim {
                return Err(MenteError::EmbeddingDimensionMismatch {
                    got: node.embedding.len(),
                    expected: dim,
                });
            }
        }

        self.db.store_memory_batch(&nodes)?;

        let mut ids = Vec::with_capacity(nodes.len());
        for node in &nodes {
            self.graph.add_memory(node.id);
            if let Err(e) = self.index_memory_entities(node) {
                warn!(memory_id = %node.id, error = %e, "failed to index memory entities");
            }
            ids.push(node.id);
        }

        Ok(ids)
    }

    fn index_memory_entities(&self, node: &MemoryNode) -> MenteResult<()> {
        let config = self.cognitive_config.entity_extraction_config.clone();
        if !config.enabled {
            return Ok(());
        }
        let extractor = RuleBasedEntityExtractor::new(config);
        let extracted = extractor.extract_memory(node);
        self.persist_extracted_entities(node.id, &extracted)
    }

    fn persist_extracted_entities(
        &self,
        memory_id: MemoryId,
        extracted: &[ExtractedEntity],
    ) -> MenteResult<()> {
        if extracted.is_empty() {
            return self
                .db
                .replace_memory_entity_bundle(memory_id, &[], &[], &[]);
        }

        let mut records = Vec::with_capacity(extracted.len());
        let mut aliases = Vec::new();
        let mut links = Vec::with_capacity(extracted.len());
        let now = current_time_us();

        for mention in extracted {
            let mut entity = match self
                .db
                .entity_by_canonical(&mention.entity_type, &mention.canonical)?
            {
                Some(mut existing) => {
                    existing.confidence = existing.confidence.max(mention.confidence);
                    existing.updated_at = now;
                    existing
                }
                None => {
                    let mut created =
                        EntityRecord::new(mention.entity_type.clone(), mention.canonical.clone());
                    created.confidence = mention.confidence;
                    created
                }
            };
            entity.attributes_json = "{}".to_string();
            let entity_id = entity.entity_id.clone();

            for alias in &mention.aliases {
                aliases.push(EntityAlias {
                    entity_id: entity_id.clone(),
                    alias: alias.clone(),
                    source: Some("surface_linker".to_string()),
                    confidence: mention.confidence,
                });
            }

            links.push(MemoryEntityLink {
                memory_id,
                entity_id,
                role: mention
                    .role
                    .clone()
                    .or_else(|| Some("mentioned".to_string())),
                confidence: mention.confidence,
                evidence: mention.evidence.clone(),
            });
            records.push(entity);
        }

        self.db
            .replace_memory_entity_bundle(memory_id, &records, &aliases, &links)?;
        let mut phantom = self.phantom.write();
        for record in &records {
            phantom.register_entity(&record.canonical);
        }
        Ok(())
    }

    /// Recalls memories using an MQL query string.
    ///
    /// Parses the query, builds an execution plan, runs it against the
    /// appropriate indexes/graph, and assembles the results into a
    /// token-budget-aware context window.
    pub fn recall(&self, query: &str) -> MenteResult<ContextWindow> {
        debug!("Recalling with query: {}", query);
        let plan = Mql::parse(query)?;

        let scored = self.execute_plan(&plan)?;
        let config = AssemblyConfig::default();
        let window = ContextAssembler::assemble(scored, vec![], &config);
        Ok(window)
    }

    /// Shortcut for vector similarity search.
    ///
    /// Returns the top-k most similar memory IDs with their scores.
    /// Memories that have been superseded, contradicted, or temporally
    /// invalidated are automatically excluded from results.
    pub fn recall_similar(&self, embedding: &[f32], k: usize) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_similar_filtered(embedding, k, None, None)
    }

    /// Vector similarity search with optional tag and time range filters.
    pub fn recall_similar_filtered(
        &self,
        embedding: &[f32],
        k: usize,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.recall_similar_filtered_at(embedding, k, now, tags, time_range)
    }

    /// Vector similarity search at a specific point in time.
    ///
    /// Only returns memories that were temporally valid at the given timestamp.
    /// Superseded/contradicted memories are excluded unless the edge itself
    /// was not yet valid at that time.
    pub fn recall_similar_at(
        &self,
        embedding: &[f32],
        k: usize,
        at: Timestamp,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_similar_filtered_at(embedding, k, at, None, None)
    }

    /// Vector similarity search at a specific point in time with optional filters.
    ///
    /// Only returns memories that were temporally valid at the given timestamp.
    /// Superseded/contradicted memories are excluded unless the edge itself
    /// was not yet valid at that time. Optionally filters by tags and time range.
    pub fn recall_similar_filtered_at(
        &self,
        embedding: &[f32],
        k: usize,
        at: Timestamp,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_hybrid_at(embedding, None, k, at, tags, time_range)
    }

    /// Hybrid search combining vector similarity and BM25 keyword matching.
    ///
    /// When `query_text` is provided, BM25 results are fused with vector
    /// results via Reciprocal Rank Fusion (RRF) for better recall on
    /// exact entity names, dates, and specific terms.
    pub fn recall_hybrid_at(
        &self,
        embedding: &[f32],
        query_text: Option<&str>,
        k: usize,
        at: Timestamp,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_hybrid_at_mode(embedding, query_text, k, at, tags, false, time_range)
    }

    /// Hybrid recall with configurable tag mode (AND vs OR).
    #[allow(clippy::too_many_arguments)]
    pub fn recall_hybrid_at_mode(
        &self,
        embedding: &[f32],
        query_text: Option<&str>,
        k: usize,
        at: Timestamp,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        debug!(
            "Recall hybrid, k={}, at={}, bm25={}, tags_or={}",
            k,
            at,
            query_text.is_some(),
            tags_or
        );
        // Over-fetch to account for filtered-out results
        let results = self.db.hybrid_search_with_query_mode(
            embedding,
            query_text,
            tags,
            tags_or,
            time_range,
            k * 3,
        )?;
        let mut scored: HashMap<MemoryId, f32> = results.into_iter().collect();
        for (id, boost) in self.entity_recall_candidates(query_text, tags, tags_or, time_range)? {
            scored
                .entry(id)
                .and_modify(|score| *score += boost)
                .or_insert(boost);
        }
        let mut results: Vec<(MemoryId, f32)> = scored.into_iter().collect();
        results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let graph = self.graph.graph();
        let filtered: Vec<(MemoryId, f32)> = results
            .into_iter()
            .filter(|(id, _)| {
                let incoming = graph.incoming(*id);
                let has_active_supersede = incoming.iter().any(|(_, e)| {
                    (e.edge_type == EdgeType::Supersedes || e.edge_type == EdgeType::Contradicts)
                        && e.is_valid_at(at)
                });
                !has_active_supersede
            })
            .filter(|(id, _)| {
                // Temporal validity: exclude memories not valid at `at`.
                match self.db.get_memory(*id) {
                    Ok(Some(node)) => node.is_valid_at(at),
                    _ => true,
                }
            })
            .take(k)
            .collect();
        Ok(filtered)
    }

    fn entity_recall_candidates(
        &self,
        query_text: Option<&str>,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        let Some(query) = query_text else {
            return Ok(Vec::new());
        };
        let config = self.cognitive_config.entity_extraction_config.clone();
        if !config.enabled || !config.recall_enabled || query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let extractor = RuleBasedEntityExtractor::new(config.clone());
        let mentions = extractor.extract_query(query);
        let mut lookup_aliases: HashMap<String, f32> = query_aliases(query, &config)
            .into_iter()
            .map(|alias| (alias, 1.0))
            .collect();
        for mention in mentions {
            for alias in mention
                .aliases
                .iter()
                .cloned()
                .chain([mention.canonical.clone(), mention.canonical.to_lowercase()])
            {
                lookup_aliases
                    .entry(alias)
                    .and_modify(|confidence| *confidence = confidence.max(mention.confidence))
                    .or_insert(mention.confidence);
            }
        }
        if lookup_aliases.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored: HashMap<MemoryId, f32> = HashMap::new();
        for (alias, alias_confidence) in lookup_aliases {
            let entities = self.db.entities_by_alias(&alias)?;
            for entity in entities {
                let links = self.db.memories_for_entity(&entity.entity_id)?;
                for link in links.into_iter().take(config.recall_fetch_limit) {
                    let Some(memory) = self.db.get_memory(link.memory_id)? else {
                        continue;
                    };
                    if !memory_matches_tags(&memory, tags, tags_or)
                        || !memory_matches_time_range(&memory, time_range)
                    {
                        continue;
                    }
                    let boost = config.recall_boost * alias_confidence * link.confidence.max(0.0);
                    scored
                        .entry(link.memory_id)
                        .and_modify(|score| *score = score.max(boost))
                        .or_insert(boost);
                }
            }
        }

        Ok(scored.into_iter().collect())
    }

    /// Multi-query search with Reciprocal Rank Fusion (RRF).
    ///
    /// Runs multiple vector searches (one per embedding) and merges results
    /// using RRF: score = Σ 1/(k + rank_i). This improves recall by matching
    /// on different semantic aspects of a query.
    /// When `query_texts` is provided, each search also runs BM25 matching.
    pub fn recall_similar_multi(
        &self,
        embeddings: &[Vec<f32>],
        k: usize,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_hybrid_multi(embeddings, None, k, tags, time_range)
    }

    /// Multi-query hybrid search with BM25 + vector fusion.
    ///
    /// Each query text is searched via both BM25 and vector, then all results
    /// are merged via RRF.
    pub fn recall_hybrid_multi(
        &self,
        embeddings: &[Vec<f32>],
        query_texts: Option<&[String]>,
        k: usize,
        tags: Option<&[&str]>,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.recall_hybrid_multi_mode(embeddings, query_texts, k, tags, false, time_range)
    }

    /// Multi-query hybrid search with configurable tag mode.
    pub fn recall_hybrid_multi_mode(
        &self,
        embeddings: &[Vec<f32>],
        query_texts: Option<&[String]>,
        k: usize,
        tags: Option<&[&str]>,
        tags_or: bool,
        time_range: Option<(Timestamp, Timestamp)>,
    ) -> MenteResult<Vec<(MemoryId, f32)>> {
        use std::collections::HashMap;

        let rrf_k = self.db.retrieval_config().multi_query_rrf_k;
        let mut rrf_scores: HashMap<MemoryId, f32> = HashMap::new();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for (i, emb) in embeddings.iter().enumerate() {
            let qt = query_texts.and_then(|texts| texts.get(i).map(|s| s.as_str()));
            let results = self.recall_hybrid_at_mode(emb, qt, k, now, tags, tags_or, time_range)?;
            for (rank, (id, _score)) in results.iter().enumerate() {
                *rrf_scores.entry(*id).or_insert(0.0) += 1.0 / (rrf_k + rank as f32);
            }
        }

        let mut merged: Vec<(MemoryId, f32)> = rrf_scores.into_iter().collect();
        merged.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);
        Ok(merged)
    }

    /// Invalidate a memory by setting its valid_until timestamp.
    ///
    /// The memory remains in storage for historical queries but is excluded
    /// from current recall results.
    pub fn invalidate_memory(&self, id: MemoryId, at: Timestamp) -> MenteResult<()> {
        debug!("Invalidating memory {} at {}", id, at);
        let mut node = self
            .db
            .get_memory(id)?
            .ok_or(MenteError::MemoryNotFound(id))?;
        node.invalidate(at);
        self.db.store_memory(&node)?;
        Ok(())
    }

    /// Adds a typed, weighted edge between two memories in the graph.
    pub fn relate(&self, edge: MemoryEdge) -> MenteResult<()> {
        debug!("Relating {} -> {}", edge.source, edge.target);
        // SQLite is the source of truth; mirror into the in-memory graph for
        // the cognitive algorithms.
        self.db.add_edge(&edge)?;
        self.graph.add_relationship(&edge)?;
        Ok(())
    }

    /// Retrieves a single memory by its ID.
    pub fn get_memory(&self, id: MemoryId) -> MenteResult<MemoryNode> {
        self.db
            .get_memory(id)?
            .ok_or(MenteError::MemoryNotFound(id))
    }

    /// Returns all memory IDs currently stored in the database.
    pub fn memory_ids(&self) -> Vec<MemoryId> {
        self.db.all_memory_ids().unwrap_or_default()
    }

    /// Returns the number of memories currently stored.
    pub fn memory_count(&self) -> usize {
        self.db.count().unwrap_or(0)
    }

    /// Add or update provenance for an existing memory.
    pub fn add_memory_source(&self, source: MemorySource) -> MenteResult<()> {
        self.db.add_memory_source(&source)
    }

    /// Provenance records for a memory, newest first.
    pub fn memory_sources(&self, id: MemoryId) -> MenteResult<Vec<MemorySource>> {
        self.db.memory_sources(id)
    }

    /// Add or update a conversation container.
    pub fn upsert_conversation(&self, conversation: ConversationRecord) -> MenteResult<()> {
        self.db.upsert_conversation(&conversation)
    }

    /// Add or update one observed event in a conversation timeline.
    pub fn add_conversation_event(&self, event: ConversationEvent) -> MenteResult<()> {
        self.db.add_conversation_event(&event)
    }

    /// Conversation events ordered by observed time.
    pub fn conversation_events(
        &self,
        conversation_id: &str,
        limit: usize,
    ) -> MenteResult<Vec<ConversationEvent>> {
        self.db.conversation_events(conversation_id, limit)
    }

    /// Add or update one extraction run.
    pub fn upsert_extraction_run(&self, run: ExtractionRun) -> MenteResult<()> {
        self.db.upsert_extraction_run(&run)
    }

    /// Extraction runs for one source memory, newest first.
    pub fn extraction_runs_for_memory(
        &self,
        memory_id: MemoryId,
    ) -> MenteResult<Vec<ExtractionRun>> {
        self.db.extraction_runs_for_memory(memory_id)
    }

    /// Recent extraction runs, newest first.
    pub fn recent_extraction_runs(&self, limit: usize) -> MenteResult<Vec<ExtractionRun>> {
        self.db.recent_extraction_runs(limit)
    }

    /// Validate and persist derived extraction artifacts in one transaction.
    pub fn store_validated_extraction(&self, batch: ValidatedExtractionBatch) -> MenteResult<()> {
        validate_extraction_batch(&batch)?;
        self.db.persist_extraction_artifacts(
            &batch.run,
            &batch.claims,
            &batch.claim_entities,
            &batch.claim_evidence,
            &batch.relationships,
            &batch.relationship_evidence,
        )
    }

    /// Add or update a canonical entity.
    pub fn upsert_entity(&self, entity: EntityRecord) -> MenteResult<()> {
        self.db.upsert_entity(&entity)
    }

    /// Load an entity by id.
    pub fn get_entity(&self, entity_id: &str) -> MenteResult<Option<EntityRecord>> {
        self.db.get_entity(entity_id)
    }

    /// List canonical entities, newest updated first.
    pub fn list_entities(&self, limit: usize) -> MenteResult<Vec<EntityRecord>> {
        self.db.list_entities(limit)
    }

    /// Add or update an alias in the durable entity table.
    pub fn upsert_entity_alias(&self, alias: EntityAlias) -> MenteResult<()> {
        self.db.add_entity_alias(&alias)
    }

    /// List aliases for an entity.
    pub fn entity_aliases(&self, entity_id: &str) -> MenteResult<Vec<EntityAlias>> {
        self.db.entity_aliases(entity_id)
    }

    /// Resolve canonical entities by alias.
    pub fn entities_by_alias(&self, alias: &str) -> MenteResult<Vec<EntityRecord>> {
        self.db.entities_by_alias(alias)
    }

    /// Link a memory to a canonical entity.
    pub fn link_memory_entity(&self, link: MemoryEntityLink) -> MenteResult<()> {
        self.db.link_memory_entity(&link)
    }

    /// Entity links for one memory.
    pub fn memory_entity_links(&self, id: MemoryId) -> MenteResult<Vec<MemoryEntityLink>> {
        self.db.memory_entity_links(id)
    }

    /// Memory links for one entity.
    pub fn memories_for_entity(&self, entity_id: &str) -> MenteResult<Vec<MemoryEntityLink>> {
        self.db.memories_for_entity(entity_id)
    }

    /// Add or update a derived claim.
    pub fn upsert_claim(&self, claim: ClaimRecord) -> MenteResult<()> {
        self.db.upsert_claim(&claim)
    }

    /// Claims linked to an entity, newest first.
    pub fn claims_for_entity(&self, entity_id: &str) -> MenteResult<Vec<ClaimRecord>> {
        self.db.claims_for_entity(entity_id)
    }

    /// Claims backed by evidence in a memory.
    pub fn claims_for_memory(&self, memory_id: MemoryId) -> MenteResult<Vec<ClaimRecord>> {
        self.db.claims_for_memory(memory_id)
    }

    /// Evidence rows for one claim.
    pub fn claim_evidence(&self, claim_id: &str) -> MenteResult<Vec<ClaimEvidence>> {
        self.db.claim_evidence(claim_id)
    }

    /// Relationships where an entity participates as source or target.
    pub fn relationships_for_entity(
        &self,
        entity_id: &str,
    ) -> MenteResult<Vec<EntityRelationship>> {
        self.db.relationships_for_entity(entity_id)
    }

    /// Evidence rows for one entity relationship.
    pub fn relationship_evidence(
        &self,
        relationship_id: &str,
    ) -> MenteResult<Vec<RelationshipEvidence>> {
        self.db.relationship_evidence(relationship_id)
    }

    /// Current retrieval tuning for hybrid and multi-query recall.
    pub fn retrieval_config(&self) -> RetrievalConfig {
        self.db.retrieval_config()
    }

    /// Replace retrieval tuning for future recall calls.
    pub fn set_retrieval_config(&self, config: RetrievalConfig) {
        self.db.set_retrieval_config(config);
    }

    /// Enable or disable persisted retrieval traces.
    pub fn set_retrieval_tracing(&self, enabled: bool) {
        self.db.set_retrieval_tracing(enabled);
    }

    /// Whether persisted retrieval tracing is enabled.
    pub fn retrieval_tracing_enabled(&self) -> bool {
        self.db.retrieval_tracing_enabled()
    }

    /// Recent write-side audit rows, newest first.
    pub fn recent_memory_operations(&self, limit: usize) -> MenteResult<Vec<MemoryOperation>> {
        self.db.recent_operations(limit)
    }

    /// Write-side audit rows that mention a memory, newest first.
    pub fn memory_operations_for(
        &self,
        id: MemoryId,
        limit: usize,
    ) -> MenteResult<Vec<MemoryOperation>> {
        self.db.operations_for_memory(id, limit)
    }

    /// Recent retrieval trace headers, newest first.
    pub fn recent_retrieval_traces(&self, limit: usize) -> MenteResult<Vec<RetrievalTrace>> {
        self.db.recent_retrieval_traces(limit)
    }

    /// Final ranked hits for a persisted retrieval trace.
    pub fn retrieval_trace_hits(&self, trace_id: &str) -> MenteResult<Vec<RetrievalTraceHit>> {
        self.db.retrieval_trace_hits(trace_id)
    }

    /// Removes a memory from storage, its vector row, tags, and edges, and
    /// drops it from the in-memory graph.
    pub fn forget(&self, id: MemoryId) -> MenteResult<()> {
        debug!("Forgetting memory {}", id);
        let _ = self.db.delete_memory(id)?;
        self.graph.remove_memory(id);
        Ok(())
    }

    /// Returns a reference to the underlying graph manager.
    pub fn graph(&self) -> &GraphManager {
        &self.graph
    }

    /// Returns a mutable reference to the underlying graph manager.
    #[deprecated(note = "GraphManager now uses interior mutability; use graph() instead")]
    pub fn graph_mut(&mut self) -> &mut GraphManager {
        &mut self.graph
    }

    /// Returns a reference to the cognitive configuration.
    pub fn cognitive_config(&self) -> &CognitiveConfig {
        &self.cognitive_config
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Write Inference
    // -----------------------------------------------------------------------

    /// Run write inference on a newly stored memory.
    ///
    /// Finds semantically similar existing memories, runs the inference engine
    /// to detect contradictions and relationships, then applies the actions
    /// (creating edges, invalidating superseded memories, etc.).
    fn run_write_inference(&self, new_memory: &MemoryNode) {
        // Find candidate memories to compare against via vector search.
        // We load a small set of the most similar memories.
        let candidates = if !new_memory.embedding.is_empty() {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            self.recall_hybrid_at(&new_memory.embedding, None, 20, now, None, None)
                .unwrap_or_default()
        } else {
            vec![]
        };

        if candidates.is_empty() {
            return;
        }

        // Load the actual MemoryNode data for each candidate.
        let existing: Vec<MemoryNode> = candidates
            .iter()
            .filter(|(id, _)| *id != new_memory.id)
            .filter_map(|(id, _)| self.db.get_memory(*id).ok().flatten())
            .collect();

        if existing.is_empty() {
            return;
        }

        let actions = self
            .write_inference
            .infer_on_write(new_memory, &existing, &[]);

        let action_count = actions.len();
        for action in actions {
            if let Err(e) = self.apply_inferred_action(action) {
                warn!("Failed to apply inferred action: {}", e);
            }
        }
        if action_count > 0 {
            debug!(
                "Write inference for {} produced {} actions",
                new_memory.id, action_count
            );
        }
    }

    /// Apply a single inferred action from the write inference engine.
    fn apply_inferred_action(&self, action: InferredAction) -> MenteResult<()> {
        match action {
            InferredAction::CreateEdge {
                source,
                target,
                edge_type,
                weight,
            } => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let edge = MemoryEdge {
                    source,
                    target,
                    edge_type,
                    weight,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: None,
                };
                debug!(
                    "Auto-creating {:?} edge {} -> {}",
                    edge_type, source, target
                );
                self.relate(edge)?;
            }
            InferredAction::InvalidateMemory {
                memory,
                superseded_by,
                valid_until,
            } => {
                debug!(
                    "Invalidating memory {} (superseded by {})",
                    memory, superseded_by
                );
                self.invalidate_memory(memory, valid_until)?;
                // Also persist the Supersedes edge (db + in-memory graph).
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let edge = MemoryEdge {
                    source: superseded_by,
                    target: memory,
                    edge_type: EdgeType::Supersedes,
                    weight: 1.0,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: None,
                };
                self.relate(edge)?;
            }
            InferredAction::MarkObsolete {
                memory,
                superseded_by,
            } => {
                debug!(
                    "Marking {} obsolete (superseded by {})",
                    memory, superseded_by
                );
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                self.invalidate_memory(memory, now)?;
                let edge = MemoryEdge {
                    source: superseded_by,
                    target: memory,
                    edge_type: EdgeType::Supersedes,
                    weight: 1.0,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: None,
                };
                self.relate(edge)?;
            }
            InferredAction::FlagContradiction {
                existing,
                new,
                reason,
            } => {
                debug!(
                    "Contradiction detected: {} vs {}: {}",
                    existing, new, reason
                );
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_micros() as u64;
                let edge = MemoryEdge {
                    source: new,
                    target: existing,
                    edge_type: EdgeType::Contradicts,
                    weight: 1.0,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: Some(reason),
                };
                self.relate(edge)?;
            }
            InferredAction::UpdateConfidence {
                memory,
                new_confidence,
            } => {
                debug!("Updating confidence for {} to {}", memory, new_confidence);
                if let Ok(mut node) = self.get_memory(memory) {
                    node.confidence = new_confidence;
                    self.db.store_memory(&node)?;
                }
            }
            InferredAction::PropagateBeliefChange { root, delta } => {
                debug!("Propagating belief change from {} (delta={})", root, delta);
                if let Ok(node) = self.get_memory(root) {
                    let new_confidence = (node.confidence + delta).clamp(0.0, 1.0);
                    let affected = self.graph.propagate_belief_change(root, new_confidence);
                    for (affected_id, new_conf) in affected {
                        if let Ok(mut affected_node) = self.get_memory(affected_id) {
                            affected_node.confidence = new_conf;
                            let _ = self.db.store_memory(&affected_node);
                        }
                    }
                }
            }
            InferredAction::UpdateContent {
                memory,
                new_content,
                reason,
            } => {
                debug!("Updating content of {}: {}", memory, reason);
                if let Ok(mut node) = self.get_memory(memory) {
                    node.content = new_content;
                    // The db upsert refreshes the FTS5 mirror (via trigger) and
                    // the vec0 row, so no separate index call is needed.
                    self.db.store_memory(&node)?;
                }
            }
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Salience Decay
    // -----------------------------------------------------------------------

    /// Apply salience decay to a batch of memories in-place.
    ///
    /// Call this during retrieval to ensure scores reflect temporal relevance,
    /// or periodically to maintain salience accuracy across the database.
    pub fn apply_decay(&self, memories: &mut [MemoryNode]) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.decay.apply_decay_batch(memories, now);
    }

    /// Compute the decayed salience for a single memory at the current time.
    pub fn compute_decayed_salience(&self, memory: &MemoryNode) -> f32 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.decay.compute_decay(
            memory.salience,
            memory.created_at,
            memory.accessed_at,
            memory.access_count,
            now,
        )
    }

    /// Apply decay globally: recompute salience for all memories and persist.
    ///
    /// This is an expensive operation intended for periodic maintenance.
    /// For real-time use, prefer `apply_decay` on retrieved memories.
    pub fn apply_decay_global(&self) -> MenteResult<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let ids = self.db.all_memory_ids()?;

        let mut updated = 0;
        for id in ids {
            if let Ok(Some(mut node)) = self.db.get_memory(id) {
                let new_salience = self.decay.compute_decay(
                    node.salience,
                    node.created_at,
                    node.accessed_at,
                    node.access_count,
                    now,
                );
                if (new_salience - node.salience).abs() > 0.001 {
                    node.salience = new_salience;
                    self.db.store_memory(&node)?;
                    updated += 1;
                }
            }
        }
        if updated > 0 {
            info!("Decay pass updated {} memories", updated);
        }
        Ok(updated)
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Consolidation
    // -----------------------------------------------------------------------

    /// Find groups of similar memories that are candidates for consolidation.
    ///
    /// Returns clusters of memories that share high semantic similarity and
    /// could be merged into unified knowledge.
    pub fn find_consolidation_candidates(
        &self,
        min_cluster_size: usize,
        similarity_threshold: f32,
    ) -> MenteResult<Vec<ConsolidationCandidate>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Load all memories eligible for consolidation.
        let eligible: Vec<MemoryNode> = self
            .db
            .all_memories()?
            .into_iter()
            .filter(|node| ConsolidationEngine::should_consolidate(node, now))
            .collect();

        if eligible.is_empty() {
            return Ok(vec![]);
        }

        Ok(self
            .consolidation
            .find_candidates(&eligible, min_cluster_size, similarity_threshold))
    }

    /// Consolidate a cluster of memories into a single merged memory.
    ///
    /// The source memories are invalidated (not deleted) and a new consolidated
    /// semantic memory is stored with Derived edges back to the sources.
    pub fn consolidate_cluster(&self, memory_ids: &[MemoryId]) -> MenteResult<MemoryId> {
        let cluster: Vec<MemoryNode> = memory_ids
            .iter()
            .filter_map(|id| self.db.get_memory(*id).ok().flatten())
            .collect();

        if cluster.len() < 2 {
            return Err(MenteError::Query(
                "consolidation requires at least 2 memories".into(),
            ));
        }

        let result = self.consolidation.consolidate(&cluster);

        // Create the consolidated memory node.
        let agent_id = cluster[0].agent_id;
        let mut consolidated = MemoryNode::new(
            agent_id,
            result.new_type,
            result.summary,
            result.combined_embedding,
        );
        consolidated.confidence = result.combined_confidence;

        let consolidated_id = consolidated.id;
        self.store(consolidated)?;

        // Invalidate source memories and create Derived edges.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        for source_id in &result.source_memories {
            let _ = self.invalidate_memory(*source_id, now);
            let edge = MemoryEdge {
                source: consolidated_id,
                target: *source_id,
                edge_type: EdgeType::Derived,
                weight: 1.0,
                created_at: now,
                valid_from: None,
                valid_until: None,
                label: None,
            };
            let _ = self.relate(edge);
        }

        info!(
            "Consolidated {} memories into {}",
            result.source_memories.len(),
            consolidated_id
        );
        Ok(consolidated_id)
    }

    /// Flushes all data and closes the database.
    pub fn close(&self) -> MenteResult<()> {
        info!("Closing MenteDB");
        self.flush()?;
        // The SQLite connection is closed when MenteDb is dropped; nothing
        // explicit to do here now that the bespoke WAL/page engine is gone.
        Ok(())
    }

    /// Rebuild all indexes by scanning every memory in storage.
    ///
    /// SQLite keeps the FTS5 mirror (via triggers) and the vec0 index (via
    /// store-time updates) in sync automatically, so this is largely a no-op:
    /// we just re-affirm the vector index at the configured dimension.
    /// Returns the number of memories.
    pub fn rebuild_indexes(&self) -> MenteResult<usize> {
        info!("Rebuilding indexes...");
        let total = self.db.count()?;
        let dim = if self.embedding_dim > 0 {
            self.embedding_dim
        } else {
            self.db.embedding_dim()
        };
        if dim > 0 {
            // Force a rebuild by dropping to 0 first would lose data; instead
            // rely on ensure_vector_index being idempotent when dim matches,
            // and only act if no index exists yet.
            let _ = self.db.ensure_vector_index(dim);
        }
        info!(
            indexed = total,
            total, "index rebuild complete (sqlite indexes are self-maintaining)"
        );
        Ok(total)
    }

    /// Persist cognitive subsystem state to disk without closing.
    ///
    /// Memories, vectors, tags, and edges are already durable in SQLite (WAL)
    /// the moment they are written, so this only flushes the cognitive
    /// subsystems that still keep their own JSON state (trajectory,
    /// speculative cache, entity resolver).
    pub fn flush(&self) -> MenteResult<()> {
        debug!("Flushing cognitive subsystem state");
        // Persist cognitive subsystem state.
        let cognitive_dir = self.path.join("cognitive");
        if std::fs::create_dir_all(&cognitive_dir).is_ok() {
            let _ = self
                .trajectory
                .write()
                .transitions
                .save(&cognitive_dir.join("transitions.json"), 1);
            let _ = self
                .speculative
                .write()
                .save(&cognitive_dir.join("speculative.json"), 0);
            let _ = self
                .entity_resolver
                .write()
                .save(&cognitive_dir.join("entities.json"));
        }
        Ok(())
    }

    /// Executes a query plan against the indexes and graph, returning scored memories.
    fn execute_plan(&self, plan: &QueryPlan) -> MenteResult<Vec<ScoredMemory>> {
        match plan {
            QueryPlan::VectorSearch { query, k, .. } => {
                let hits = self.db.hybrid_search(query, None, None, *k)?;
                self.load_scored_memories(&hits)
            }
            QueryPlan::TagScan { tags, limit, .. } => {
                let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                let k = limit.unwrap_or(10);
                // Use a zero-vector for tag-only search; salience+bitmap still apply.
                let hits = self.db.hybrid_search(&[], Some(&tag_refs), None, k)?;
                self.load_scored_memories(&hits)
            }
            QueryPlan::TemporalScan { start, end, .. } => {
                let hits = self
                    .db
                    .hybrid_search(&[], None, Some((*start, *end)), 100)?;
                self.load_scored_memories(&hits)
            }
            QueryPlan::GraphTraversal { start, depth, .. } => {
                let (ids, _edges) = self.graph.get_context_subgraph(*start, *depth);
                let scored: Vec<ScoredMemory> = ids
                    .iter()
                    .filter_map(|id| {
                        self.db
                            .get_memory(*id)
                            .ok()
                            .flatten()
                            .map(|node| ScoredMemory {
                                memory: node,
                                score: 1.0,
                            })
                    })
                    .collect();
                Ok(scored)
            }
            QueryPlan::PointLookup { id } => {
                let node = self
                    .db
                    .get_memory(*id)?
                    .ok_or(MenteError::MemoryNotFound(*id))?;
                Ok(vec![ScoredMemory {
                    memory: node,
                    score: 1.0,
                }])
            }
            _ => Ok(vec![]),
        }
    }

    /// Loads MemoryNodes from storage and pairs them with their search scores.
    ///
    /// When decay is enabled, salience is recomputed and factored into the
    /// final score to prioritize temporally relevant memories.
    fn load_scored_memories(&self, hits: &[(MemoryId, f32)]) -> MenteResult<Vec<ScoredMemory>> {
        let now = if self.cognitive_config.decay_on_recall {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64
        } else {
            0
        };

        let mut scored = Vec::with_capacity(hits.len());
        for &(id, score) in hits {
            if let Ok(Some(node)) = self.db.get_memory(id) {
                let final_score = if self.cognitive_config.decay_on_recall {
                    let decayed_salience = self.decay.compute_decay(
                        node.salience,
                        node.created_at,
                        node.accessed_at,
                        node.access_count,
                        now,
                    );
                    // Blend search similarity with decayed salience.
                    // 70% similarity, 30% salience, keeps search relevance
                    // primary but rewards recently active memories.
                    score * 0.7 + decayed_salience * 0.3
                } else {
                    score
                };
                scored.push(ScoredMemory {
                    memory: node,
                    score: final_score,
                });
            }
        }
        // Re-sort by blended score when decay is applied.
        if self.cognitive_config.decay_on_recall {
            scored.sort_unstable_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        Ok(scored)
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Pain Registry
    // -----------------------------------------------------------------------

    /// Record a pain signal, a recurring failure or frustration pattern.
    ///
    /// Pain signals are tracked by keywords and surfaced as warnings when
    /// similar contexts arise in future queries.
    pub fn record_pain(&self, signal: PainSignal) {
        if self.cognitive_config.pain_tracking {
            self.pain.write().record_pain(signal);
        }
    }

    /// Get pain warnings relevant to the given context keywords.
    ///
    /// Returns formatted warning text if any pain signals match the keywords.
    /// Use this before answering to warn about past failures.
    pub fn get_pain_warnings(&self, context_keywords: &[String]) -> Vec<PainSignal> {
        if !self.cognitive_config.pain_tracking {
            return vec![];
        }
        let registry = self.pain.read();
        registry
            .get_pain_for_context(context_keywords)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Format pain warnings as a human-readable string.
    pub fn format_pain_warnings(&self, signals: &[&PainSignal]) -> String {
        self.pain.read().format_pain_warnings(signals)
    }

    /// Decay all pain signals to reduce intensity over time.
    pub fn decay_pain(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.pain.write().decay_all(now);
    }

    /// Get all recorded pain signals.
    pub fn all_pain_signals(&self) -> Vec<PainSignal> {
        self.pain.read().all_signals().to_vec()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Trajectory Tracking
    // -----------------------------------------------------------------------

    /// Record a conversation turn in the trajectory tracker.
    ///
    /// Tracks the evolution of topics, decisions, and open questions across
    /// a conversation. Used for resume context and topic prediction.
    pub fn record_trajectory_turn(&self, turn: TrajectoryNode) {
        self.trajectory.write().record_turn(turn);
    }

    /// Get a resume context string summarizing the conversation so far.
    ///
    /// Returns None if no trajectory has been recorded.
    pub fn get_resume_context(&self) -> Option<String> {
        self.trajectory.read().get_resume_context()
    }

    /// Predict the next likely topics based on conversation trajectory.
    ///
    /// Returns up to 3 predicted topic strings based on transition patterns.
    pub fn predict_next_topics(&self) -> Vec<String> {
        self.trajectory.read().predict_next_topics()
    }

    /// Get the full trajectory of recorded turns.
    pub fn get_trajectory(&self) -> Vec<TrajectoryNode> {
        self.trajectory.read().get_trajectory().to_vec()
    }

    /// Reinforce a transition that led to a speculative cache hit.
    pub fn reinforce_transition(&self, hit_topic: &str) {
        self.trajectory.write().reinforce_transition(hit_topic);
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Cognition Stream
    // -----------------------------------------------------------------------

    /// Feed a token to the cognition stream for real-time monitoring.
    ///
    /// Tokens are buffered and analyzed for contradictions with known facts
    /// when `check_stream_alerts()` is called.
    pub fn feed_stream_token(&self, token: &str) {
        self.stream.feed_token(token);
    }

    /// Check for stream alerts against known facts.
    ///
    /// Compares the buffered token stream against the provided known facts
    /// to detect contradictions, corrections, and reinforcements.
    pub fn check_stream_alerts(&self, known_facts: &[(MemoryId, String)]) -> Vec<StreamAlert> {
        self.stream.check_alerts(known_facts)
    }

    /// Drain the token buffer, returning accumulated text.
    pub fn drain_stream_buffer(&self) -> String {
        self.stream.drain_buffer()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Phantom Tracking
    // -----------------------------------------------------------------------

    /// Detect phantom memories, entities referenced in content but not stored.
    ///
    /// Scans content for entity mentions that don't exist in the known entities
    /// list, flagging them as knowledge gaps that should be filled.
    pub fn detect_phantoms(
        &self,
        content: &str,
        known_entities: &[String],
        turn_id: u64,
    ) -> Vec<PhantomMemory> {
        if !self.cognitive_config.phantom_tracking {
            return vec![];
        }
        self.phantom
            .write()
            .detect_gaps(content, known_entities, turn_id)
    }

    /// Resolve a phantom memory (mark it as no longer a gap).
    pub fn resolve_phantom(&self, phantom_id: MemoryId) {
        self.phantom.write().resolve(phantom_id.into());
    }

    /// Get all active (unresolved) phantom memories, sorted by priority.
    pub fn get_active_phantoms(&self) -> Vec<PhantomMemory> {
        self.phantom
            .read()
            .get_active_phantoms()
            .into_iter()
            .cloned()
            .collect()
    }

    /// Format phantom warnings as a human-readable string.
    pub fn format_phantom_warnings(&self) -> String {
        self.phantom.read().format_phantom_warnings()
    }

    /// Register an entity so the phantom tracker knows it exists.
    pub fn register_entity(&self, entity: &str) {
        self.phantom.write().register_entity(entity);
    }

    /// Register multiple entities at once.
    pub fn register_entities(&self, entities: &[&str]) {
        self.phantom.write().register_entities(entities);
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Speculative Cache
    // -----------------------------------------------------------------------

    /// Try to hit the speculative cache for a query.
    ///
    /// If a previous prediction matches the current query (by keyword overlap
    /// or embedding similarity), returns the pre-assembled context.
    pub fn try_speculative_hit(
        &self,
        query: &str,
        query_embedding: Option<&[f32]>,
    ) -> Option<CacheEntry> {
        if !self.cognitive_config.speculative_cache {
            return None;
        }
        self.speculative.write().try_hit(query, query_embedding)
    }

    /// Pre-assemble speculative cache entries for predicted topics.
    ///
    /// The builder function should return `(context_text, memory_ids, optional_embedding)`
    /// for each topic prediction.
    pub fn pre_assemble_speculative<F>(&self, predictions: Vec<String>, builder: F)
    where
        F: Fn(&str) -> Option<(String, Vec<MemoryId>, Option<Vec<f32>>)>,
    {
        if self.cognitive_config.speculative_cache {
            self.speculative.write().pre_assemble(predictions, builder);
        }
    }

    /// Evict stale entries from the speculative cache.
    pub fn evict_stale_speculative(&self, max_age_us: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.speculative.write().evict_stale(max_age_us, now);
    }

    /// Get speculative cache statistics.
    pub fn speculative_cache_stats(&self) -> CacheStats {
        self.speculative.read().stats()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Interference Detection
    // -----------------------------------------------------------------------

    /// Detect interference between a set of memories.
    ///
    /// Returns pairs of memories that are similar enough to cause confusion,
    /// along with disambiguation hints. Use this during context assembly to
    /// add disambiguation notes or separate confusable memories.
    pub fn detect_interference(&self, memories: &[MemoryNode]) -> Vec<InterferencePair> {
        if !self.cognitive_config.interference_detection {
            return vec![];
        }
        self.interference.detect_interference(memories)
    }

    /// Generate a disambiguation hint for two confusable memories.
    pub fn generate_disambiguation(&self, a: &MemoryNode, b: &MemoryNode) -> String {
        self.interference.generate_disambiguation(a, b)
    }

    /// Arrange memory IDs to maximize separation between interfering pairs.
    pub fn arrange_with_separation(
        memories: Vec<MemoryId>,
        pairs: &[InterferencePair],
    ) -> Vec<MemoryId> {
        InterferenceDetector::arrange_with_separation(memories, pairs)
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Entity Resolution
    // -----------------------------------------------------------------------

    /// Resolve an entity name to its canonical form.
    ///
    /// Uses cached aliases and rule-based matching (no LLM).
    pub fn resolve_entity(&self, name: &str) -> mentedb_cognitive::ResolvedEntity {
        self.entity_resolver.read().resolve(name)
    }

    /// Add an alias mapping for entity resolution.
    pub fn add_entity_alias(&self, alias: &str, canonical: &str, confidence: f32) {
        self.entity_resolver
            .write()
            .add_alias(alias, canonical, confidence);
    }

    /// Get the canonical name for an entity, if known.
    pub fn get_canonical_entity(&self, name: &str) -> Option<String> {
        self.entity_resolver.read().get_canonical(name).cloned()
    }

    /// List all known entities in the resolver.
    pub fn known_entities(&self) -> Vec<String> {
        self.entity_resolver.read().known_entities()
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Memory Compression
    // -----------------------------------------------------------------------

    /// Compress a memory's content, extracting key facts and removing filler.
    ///
    /// Returns a compressed representation with the original ID, compressed text,
    /// compression ratio, and extracted key facts.
    pub fn compress_memory(&self, memory: &MemoryNode) -> CompressedMemory {
        self.compressor.compress(memory)
    }

    /// Compress a batch of memories.
    pub fn compress_memories(&self, memories: &[MemoryNode]) -> Vec<CompressedMemory> {
        self.compressor.compress_batch(memories)
    }

    /// Estimate token count for a text string.
    pub fn estimate_tokens(text: &str) -> usize {
        MemoryCompressor::estimate_tokens(text)
    }

    // -----------------------------------------------------------------------
    // Cognitive Engine: Archival Evaluation
    // -----------------------------------------------------------------------

    /// Evaluate whether a memory should be kept, archived, or deleted.
    ///
    /// Uses age, salience, and access patterns to make lifecycle decisions.
    pub fn evaluate_archival(&self, memory: &MemoryNode) -> ArchivalDecision {
        if !self.cognitive_config.archival_evaluation {
            return ArchivalDecision::Keep;
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.archival.evaluate(memory, now)
    }

    /// Evaluate archival decisions for a batch of memories.
    pub fn evaluate_archival_batch(
        &self,
        memories: &[MemoryNode],
    ) -> Vec<(MemoryId, ArchivalDecision)> {
        if !self.cognitive_config.archival_evaluation {
            return memories
                .iter()
                .map(|m| (m.id, ArchivalDecision::Keep))
                .collect();
        }
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.archival.evaluate_batch(memories, now)
    }

    /// Run archival evaluation on all memories in the database.
    ///
    /// Returns decisions for each memory. Does NOT apply them, call
    /// `invalidate_memory` or `forget` to act on the decisions.
    pub fn evaluate_archival_global(&self) -> MenteResult<Vec<(MemoryId, ArchivalDecision)>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        let memories: Vec<MemoryNode> = self.db.all_memories()?;
        Ok(self.archival.evaluate_batch(&memories, now))
    }

    // -----------------------------------------------------------------------
    // Sleeptime Enrichment Pipeline
    // -----------------------------------------------------------------------

    /// Check whether enrichment is pending (triggered by turn count or manual).
    pub fn needs_enrichment(&self) -> bool {
        if !self.cognitive_config.enrichment_config.enabled {
            return false;
        }
        *self.enrichment_pending.read()
    }

    /// Get the turn ID when enrichment last completed.
    pub fn last_enrichment_turn(&self) -> u64 {
        *self.last_enrichment_turn.read()
    }

    /// Manually trigger enrichment on the next check.
    pub fn request_enrichment(&self) {
        *self.enrichment_pending.write() = true;
    }

    /// Get episodic memories that haven't been enriched yet.
    ///
    /// Returns all Episodic memories created after the last enrichment turn,
    /// sorted by creation time. These are the candidates for LLM extraction.
    pub fn enrichment_candidates(&self) -> Vec<MemoryNode> {
        let last_turn = *self.last_enrichment_turn.read();
        let mut candidates: Vec<MemoryNode> = self
            .db
            .all_memories()
            .unwrap_or_default()
            .into_iter()
            .filter(|m| {
                m.memory_type == mentedb_core::memory::MemoryType::Episodic
                    && !m.tags.contains(&"source:enrichment".to_string())
                    && m.created_at > last_turn
            })
            .collect();
        candidates.sort_by_key(|m| m.created_at);
        candidates
    }

    /// Store enrichment results: extracted memories with provenance tracking.
    ///
    /// Each stored memory gets:
    /// - `source:enrichment` tag for identification
    /// - Confidence capped at `max_enrichment_confidence`
    /// - `Derived` edges back to source episodic memories
    ///
    /// Returns (memories_stored, edges_created).
    pub fn store_enrichment_memories(
        &self,
        memories: Vec<MemoryNode>,
        source_ids: &[MemoryId],
    ) -> MenteResult<(usize, usize)> {
        let max_conf = self
            .cognitive_config
            .enrichment_config
            .max_enrichment_confidence;
        let mut stored = 0usize;
        let mut edges = 0usize;

        for mut mem in memories {
            // Tag as enrichment-generated
            if !mem.tags.contains(&"source:enrichment".to_string()) {
                mem.tags.push("source:enrichment".to_string());
            }
            // Cap confidence
            if mem.confidence > max_conf {
                mem.confidence = max_conf;
            }

            let mem_id = mem.id;
            self.store(mem)?;
            stored += 1;

            // Create Derived edges back to source episodics
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64;
            for src_id in source_ids {
                let edge = MemoryEdge {
                    source: mem_id,
                    target: *src_id,
                    edge_type: EdgeType::Derived,
                    weight: 0.8,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: Some("enrichment".to_string()),
                };
                if self.relate(edge).is_ok() {
                    edges += 1;
                }
            }
        }

        debug!(stored, edges, "enrichment memories stored");
        Ok((stored, edges))
    }

    /// Mark enrichment as complete for the given turn.
    pub fn mark_enrichment_complete(&self, turn_id: u64) {
        *self.last_enrichment_turn.write() = turn_id;
        *self.enrichment_pending.write() = false;
        debug!(turn_id, "enrichment cycle complete");
    }

    /// Get the enrichment configuration.
    pub fn enrichment_config(&self) -> &EnrichmentConfig {
        &self.cognitive_config.enrichment_config
    }

    /// Get all unique entity names from stored entity memories.
    ///
    /// Returns deduplicated, normalized entity names extracted from
    /// `entity:{name}` tags across all stored memories.
    pub fn all_entity_names(&self) -> Vec<String> {
        let mut names = std::collections::HashSet::new();
        for mem in self.db.all_memories().unwrap_or_default() {
            for tag in &mem.tags {
                if let Some(name) = tag.strip_prefix("entity:") {
                    names.insert(name.to_lowercase().trim().to_string());
                }
            }
        }
        let mut sorted: Vec<String> = names.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Get entity names that the EntityResolver hasn't resolved yet.
    ///
    /// These are the entities that need LLM resolution. The EntityResolver
    /// cache handles known entities for free.
    pub fn unresolved_entity_names(&self) -> Vec<String> {
        let all_names = self.all_entity_names();
        self.entity_resolver.read().unresolved_names(&all_names)
    }

    /// Get entity names with their memory content for LLM context.
    ///
    /// Returns (name, content) pairs for entities that need resolution.
    /// The content helps the LLM disambiguate (e.g., "Python" near
    /// "web framework" vs "Python" near "Monty Python").
    pub fn entity_names_with_context(&self) -> Vec<(String, Option<String>)> {
        let mut entity_contexts: HashMap<String, String> = HashMap::new();

        for mem in self.db.all_memories().unwrap_or_default() {
            for tag in &mem.tags {
                if let Some(name) = tag.strip_prefix("entity:") {
                    let normalized = name.to_lowercase().trim().to_string();
                    // One entity tag per memory (enrichment creates separate memories per entity)
                    entity_contexts
                        .entry(normalized)
                        .and_modify(|existing| {
                            // Append content from multiple mentions, cap at ~500 chars
                            if existing.len() < 300 {
                                existing.push_str(" | ");
                                let remaining = 500usize.saturating_sub(existing.len());
                                existing.push_str(&mem.content[..mem.content.len().min(remaining)]);
                            }
                        })
                        .or_insert_with(|| mem.content[..mem.content.len().min(300)].to_string());
                    break;
                }
            }
        }

        entity_contexts
            .into_iter()
            .map(|(name, ctx)| (name, Some(ctx)))
            .collect()
    }

    /// Apply LLM entity resolution results: create graph edges and update cache.
    ///
    /// Takes merge groups from the LLM (via `CognitiveLlmService.resolve_entities()`)
    /// and confirmed-different pairs. Creates `entity_link:` edges between entity
    /// memories that belong to the same group, learns aliases in the EntityResolver,
    /// and negative-caches confirmed-different pairs.
    pub fn apply_entity_link_resolutions(
        &self,
        merge_groups: &[EntityLinkResolution],
        separations: &[EntitySeparation],
    ) -> MenteResult<EntityLinkResult> {
        let mut result = EntityLinkResult::default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Build a map: normalized entity name → list of memory IDs
        let entity_memory_map = self.build_entity_memory_map();

        let mut resolver = self.entity_resolver.write();

        for group in merge_groups {
            // Learn the group in the resolver cache
            let mut aliases: Vec<String> = group.aliases.clone();
            aliases.retain(|a| a.to_lowercase() != group.canonical.to_lowercase());
            resolver.learn_group(&EntityMergeGroup {
                canonical: group.canonical.clone(),
                aliases,
                confidence: group.confidence,
            });

            // Collect all memory IDs for this merge group
            let mut group_memory_ids: Vec<MemoryId> = Vec::new();

            // Add memories for the canonical name
            let canonical_norm = group.canonical.to_lowercase();
            if let Some(ids) = entity_memory_map.get(&canonical_norm) {
                group_memory_ids.extend(ids);
            }

            // Add memories for each alias
            for alias in &group.aliases {
                let alias_norm = alias.to_lowercase();
                if let Some(ids) = entity_memory_map.get(&alias_norm) {
                    group_memory_ids.extend(ids);
                }
            }

            group_memory_ids.sort();
            group_memory_ids.dedup();

            // Create edges between all pairs in the group
            let label = format!("entity_link:{}", canonical_norm);
            for i in 0..group_memory_ids.len() {
                for j in (i + 1)..group_memory_ids.len() {
                    let a_id = group_memory_ids[i];
                    let b_id = group_memory_ids[j];

                    // Check for existing edge
                    let graph = self.graph.read_graph();
                    let already_linked = graph.outgoing(a_id).iter().any(|(tid, e)| {
                        *tid == b_id
                            && e.edge_type == EdgeType::Related
                            && e.label
                                .as_ref()
                                .is_some_and(|l| l.starts_with("entity_link:"))
                    });
                    drop(graph);

                    if already_linked {
                        continue;
                    }

                    let edge = MemoryEdge {
                        source: a_id,
                        target: b_id,
                        edge_type: EdgeType::Related,
                        weight: group.confidence,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: Some(label.clone()),
                    };
                    if self.relate(edge).is_ok() {
                        result.edges_created += 1;
                    }
                    result.linked += 1;
                }
            }

            debug!(
                canonical = group.canonical,
                aliases = ?group.aliases,
                memories = group_memory_ids.len(),
                "entity resolution: merged group"
            );
        }

        // Process negative cache entries
        for sep in separations {
            resolver.mark_different(&sep.name_a, &sep.name_b);
            debug!(
                a = sep.name_a,
                b = sep.name_b,
                "entity resolution: confirmed different"
            );
        }

        // Persist resolver state
        let cognitive_dir = self.path.join("cognitive");
        if cognitive_dir.exists() || std::fs::create_dir_all(&cognitive_dir).is_ok() {
            let _ = resolver.save(&cognitive_dir.join("entities.json"));
        }

        debug!(
            linked = result.linked,
            edges = result.edges_created,
            groups = merge_groups.len(),
            separations = separations.len(),
            "entity link resolutions applied"
        );
        Ok(result)
    }

    /// Link entities using only the sync EntityResolver (cache + rules, no LLM).
    ///
    /// This is the fast path, links entities that are already known to be
    /// the same from previous LLM resolutions. For full LLM-powered resolution,
    /// use `unresolved_entity_names()` + `apply_entity_link_resolutions()`.
    pub fn link_entities(&self) -> MenteResult<EntityLinkResult> {
        let entity_memory_map = self.build_entity_memory_map();
        let resolver = self.entity_resolver.read();

        // Group entity names by their resolved canonical name
        let mut canonical_groups: HashMap<String, Vec<String>> = HashMap::new();
        for entity_name in entity_memory_map.keys() {
            let resolved = resolver.resolve(entity_name);
            if resolved.source != mentedb_cognitive::ResolutionSource::Identity {
                canonical_groups
                    .entry(resolved.canonical.clone())
                    .or_default()
                    .push(entity_name.clone());
            }
        }

        drop(resolver);

        let mut result = EntityLinkResult::default();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        for (canonical, names) in &canonical_groups {
            // Collect all memory IDs across all aliases in this group
            let mut group_memory_ids: Vec<MemoryId> = Vec::new();
            for name in names {
                if let Some(ids) = entity_memory_map.get(name) {
                    group_memory_ids.extend(ids);
                }
            }
            // Also include the canonical name itself
            if let Some(ids) = entity_memory_map.get(canonical) {
                group_memory_ids.extend(ids);
            }
            group_memory_ids.sort();
            group_memory_ids.dedup();

            if group_memory_ids.len() < 2 {
                continue;
            }

            let label = format!("entity_link:{}", canonical);
            for i in 0..group_memory_ids.len() {
                for j in (i + 1)..group_memory_ids.len() {
                    let a_id = group_memory_ids[i];
                    let b_id = group_memory_ids[j];

                    let graph = self.graph.read_graph();
                    let already_linked = graph.outgoing(a_id).iter().any(|(tid, e)| {
                        *tid == b_id
                            && e.edge_type == EdgeType::Related
                            && e.label
                                .as_ref()
                                .is_some_and(|l| l.starts_with("entity_link:"))
                    });
                    drop(graph);

                    if already_linked {
                        continue;
                    }

                    let edge = MemoryEdge {
                        source: a_id,
                        target: b_id,
                        edge_type: EdgeType::Related,
                        weight: 1.0,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: Some(label.clone()),
                    };
                    if self.relate(edge).is_ok() {
                        result.edges_created += 1;
                    }
                    result.linked += 1;
                }
            }
        }

        debug!(
            linked = result.linked,
            edges = result.edges_created,
            groups = canonical_groups.len(),
            "sync entity linking complete"
        );
        Ok(result)
    }

    /// Build a map of normalized entity name → list of MemoryIds.
    fn build_entity_memory_map(&self) -> HashMap<String, Vec<MemoryId>> {
        let mut map: HashMap<String, Vec<MemoryId>> = HashMap::new();
        for mem in self.db.all_memories().unwrap_or_default() {
            for tag in &mem.tags {
                if let Some(name) = tag.strip_prefix("entity:") {
                    let normalized = name.to_lowercase().trim().to_string();
                    // One entity tag per memory (enrichment creates separate memories per entity)
                    map.entry(normalized).or_default().push(mem.id);
                    break;
                }
            }
        }
        map
    }

    /// Get all entity memory nodes (memories tagged with `entity:{name}`).
    pub fn entity_memories(&self) -> Vec<MemoryNode> {
        self.db
            .all_memories()
            .unwrap_or_default()
            .into_iter()
            .filter(|m| m.tags.iter().any(|t| t.starts_with("entity:")))
            .collect()
    }

    /// Get entity categories with their member entities for community detection.
    ///
    /// Returns a map of category → list of (entity_name, context_snippet).
    /// Categories come from `entity_type:` tags on entity memories.
    pub fn entity_communities(&self) -> HashMap<String, Vec<(String, String)>> {
        let mut categories: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for mem in self.db.all_memories().unwrap_or_default() {
            // Skip non-entity memories and existing community summaries
            if mem.tags.iter().any(|t| t == "community_summary") {
                continue;
            }

            let entity_name = mem
                .tags
                .iter()
                .find_map(|t| t.strip_prefix("entity:"))
                .map(|n| n.to_string());

            if let Some(name) = entity_name {
                let entity_type = mem
                    .tags
                    .iter()
                    .find_map(|t| t.strip_prefix("entity_type:"))
                    .unwrap_or("general")
                    .to_lowercase();

                let context: String = mem.content.chars().take(200).collect();
                categories
                    .entry(entity_type)
                    .or_default()
                    .push((name, context));
            }
        }

        // Only return categories with 2+ entities (meaningful clusters)
        categories.retain(|_, members| members.len() >= 2);
        categories
    }

    /// Store a community summary memory with edges to member entities.
    ///
    /// Creates a `community_summary` tagged memory and `Derived` edges
    /// from the summary to each member entity in the category.
    pub fn store_community_summary(
        &self,
        category: &str,
        summary: &str,
        member_names: &[String],
    ) -> MenteResult<MemoryId> {
        if category.is_empty() {
            return Err(MenteError::Storage(
                "community category cannot be empty".into(),
            ));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;

        // Check if a community summary already exists for this category
        let community_tag = format!("community:{}", category);
        let mut existing_id = None;
        for mem in self.db.all_memories().unwrap_or_default() {
            if mem.tags.iter().any(|t| t == &community_tag) {
                // Update existing summary content
                let mut updated = mem.clone();
                updated.content = summary.to_string();
                if let Some(ref embedder) = self.embedder {
                    updated.embedding = embedder
                        .embed(summary)
                        .unwrap_or_else(|_| updated.embedding.clone());
                }
                self.db.store_memory(&updated)?;
                existing_id = Some(updated.id);
                break;
            }
        }

        let node_id = if let Some(id) = existing_id {
            id
        } else {
            // Create new community summary
            let embedding = self
                .embedder
                .as_ref()
                .and_then(|e| e.embed(summary).ok())
                .unwrap_or_default();

            let mut node = MemoryNode::new(
                mentedb_core::types::AgentId::new(),
                MemoryType::Semantic,
                summary.to_string(),
                embedding,
            );
            node.tags = vec![
                "community_summary".to_string(),
                community_tag,
                "source:enrichment".to_string(),
            ];
            node.confidence = 0.7;
            let id = node.id;
            self.store(node)?;
            id
        };

        // (Re)create Derived edges from summary to member entity memories.
        // On update this refreshes edges to reflect current membership.
        let entity_map = self.build_entity_memory_map();
        for name in member_names {
            let normalized = name.to_lowercase();
            if let Some(member_ids) = entity_map.get(&normalized) {
                for member_id in member_ids {
                    self.relate(MemoryEdge {
                        source: node_id,
                        target: *member_id,
                        edge_type: EdgeType::Derived,
                        weight: 0.8,
                        created_at: now,
                        valid_from: None,
                        valid_until: None,
                        label: Some(format!("community_member:{}", category)),
                    })?;
                }
            }
        }

        Ok(node_id)
    }

    /// Get existing community summaries.
    pub fn community_summaries(&self) -> Vec<MemoryNode> {
        self.db
            .all_memories()
            .unwrap_or_default()
            .into_iter()
            .filter(|m| m.tags.iter().any(|t| t == "community_summary"))
            .collect()
    }

    /// Collect all semantic/procedural facts for user profile generation.
    ///
    /// Returns high-confidence memories suitable for profile building.
    pub fn profile_facts(&self) -> Vec<String> {
        let mut facts = Vec::new();

        for mem in self.db.all_memories().unwrap_or_default() {
            // Only semantic and procedural memories with decent confidence
            if mem.confidence < 0.5 {
                continue;
            }
            match mem.memory_type {
                MemoryType::Semantic | MemoryType::Procedural => {
                    // Skip community summaries and entity nodes
                    if mem
                        .tags
                        .iter()
                        .any(|t| t == "community_summary" || t.starts_with("entity:"))
                    {
                        continue;
                    }
                    facts.push(mem.content.chars().take(300).collect());
                }
                _ => {}
            }
        }

        // Cap at 100 most relevant facts to fit in LLM context
        facts.truncate(100);
        facts
    }

    /// Store or update the user profile as an always-scoped memory.
    ///
    /// There is exactly one user profile memory (tagged `user_profile`).
    /// If one already exists, it's replaced entirely.
    pub fn store_user_profile(&self, profile: &str) -> MenteResult<MemoryId> {
        // Find existing profile
        for mem in self.db.all_memories().unwrap_or_default() {
            if mem.tags.iter().any(|t| t == "user_profile") {
                // Update in place
                let mut updated = mem.clone();
                updated.content = profile.to_string();
                if let Some(ref embedder) = self.embedder {
                    updated.embedding = embedder
                        .embed(profile)
                        .unwrap_or_else(|_| updated.embedding.clone());
                }
                self.db.store_memory(&updated)?;
                if let Err(e) = self.index_memory_entities(&updated) {
                    warn!(
                        memory_id = %updated.id,
                        error = %e,
                        "failed to index user profile entities"
                    );
                }
                return Ok(updated.id);
            }
        }

        // Create new profile
        let embedding = self
            .embedder
            .as_ref()
            .and_then(|e| e.embed(profile).ok())
            .unwrap_or_default();

        let mut node = MemoryNode::new(
            mentedb_core::types::AgentId::new(),
            MemoryType::Semantic,
            profile.to_string(),
            embedding,
        );
        node.tags = vec![
            "user_profile".to_string(),
            "scope:always".to_string(),
            "source:enrichment".to_string(),
        ];
        node.confidence = 0.8;
        let node_id = node.id;
        self.store(node)?;

        Ok(node_id)
    }

    /// Get the current user profile, if one exists.
    pub fn user_profile(&self) -> Option<MemoryNode> {
        self.db
            .all_memories()
            .unwrap_or_default()
            .into_iter()
            .find(|m| m.tags.iter().any(|t| t == "user_profile"))
    }
}

fn current_time_us() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as Timestamp
}

fn memory_matches_tags(node: &MemoryNode, tags: Option<&[&str]>, tags_or: bool) -> bool {
    let Some(tags) = tags else {
        return true;
    };
    if tags.is_empty() {
        return false;
    }
    if tags_or {
        tags.iter()
            .any(|tag| node.tags.iter().any(|memory_tag| memory_tag == tag))
    } else {
        tags.iter()
            .all(|tag| node.tags.iter().any(|memory_tag| memory_tag == tag))
    }
}

fn memory_matches_time_range(
    node: &MemoryNode,
    time_range: Option<(Timestamp, Timestamp)>,
) -> bool {
    match time_range {
        Some((start, end)) => node.created_at >= start && node.created_at <= end,
        None => true,
    }
}

fn query_aliases(query: &str, config: &EntityExtractionConfig) -> Vec<String> {
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|raw| {
            raw.trim_matches(|ch: char| {
                matches!(
                    ch,
                    ',' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '?' | '!'
                )
            })
            .trim_end_matches('.')
            .to_lowercase()
        })
        .filter(|token| {
            token.len() >= config.min_phrase_chars
                && token.chars().any(|ch| ch.is_ascii_alphabetic())
        })
        .collect();

    let mut aliases = Vec::new();
    for start in 0..tokens.len() {
        let max_end = (start + config.max_alias_words).min(tokens.len());
        for end in (start + 1)..=max_end {
            let alias = tokens[start..end].join(" ");
            if alias.len() <= config.max_phrase_chars
                && !aliases.iter().any(|existing| existing == &alias)
            {
                aliases.push(alias);
            }
        }
    }
    aliases
}
