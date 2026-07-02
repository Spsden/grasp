//! Bounded sleeptime maintenance for background workers.

use std::fs::{File, OpenOptions};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use fs2::FileExt;
use mentedb_consolidation::archival::ArchivalDecision;
use mentedb_consolidation::consolidation::{ConsolidationCandidate, ConsolidationEngine};
use mentedb_core::types::{MemoryId, Timestamp};
use serde::{Deserialize, Serialize};

use crate::{MemoryNode, MenteDb, MenteResult};

/// Configuration for one bounded sleep maintenance run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepMaintenanceConfig {
    /// Maximum memories loaded for decay, archival, and consolidation.
    pub max_memories: usize,
    /// Apply salience decay to loaded memories.
    pub apply_decay: bool,
    /// Minimum salience delta before rewriting a memory.
    pub decay_write_epsilon: f32,
    /// Evaluate lifecycle decisions for loaded memories.
    pub evaluate_archival: bool,
    /// Apply delete recommendations. Archive recommendations are reported only.
    pub apply_archival_deletes: bool,
    /// Run similarity-based consolidation over loaded memories.
    pub run_consolidation: bool,
    /// Maximum consolidation clusters applied in one run.
    pub max_consolidation_clusters: usize,
    /// Minimum cluster size for consolidation candidates.
    pub consolidation_min_cluster_size: usize,
    /// Similarity threshold for consolidation candidates.
    pub consolidation_similarity_threshold: f32,
    /// Link entities already known by the sync entity resolver.
    pub link_entities: bool,
}

impl Default for SleepMaintenanceConfig {
    fn default() -> Self {
        Self {
            max_memories: 1_000,
            apply_decay: true,
            decay_write_epsilon: 0.001,
            evaluate_archival: true,
            apply_archival_deletes: false,
            run_consolidation: true,
            max_consolidation_clusters: 4,
            consolidation_min_cluster_size: 2,
            consolidation_similarity_threshold: 0.85,
            link_entities: true,
        }
    }
}

/// Stage that produced a non-fatal maintenance issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SleepMaintenanceStage {
    /// Salience decay.
    Decay,
    /// Archival evaluation or application.
    Archival,
    /// Consolidation.
    Consolidation,
    /// Entity linking.
    EntityLinking,
}

/// Non-fatal issue recorded during maintenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepMaintenanceIssue {
    /// Stage where the issue occurred.
    pub stage: SleepMaintenanceStage,
    /// Memory involved, when the issue is memory-specific.
    pub memory_id: Option<MemoryId>,
    /// Human-readable error message.
    pub message: String,
}

/// Summary of a bounded sleep maintenance run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SleepMaintenanceResult {
    /// Memories loaded under the configured budget.
    pub processed_memories: usize,
    /// Memories whose salience was rewritten.
    pub decay_updated: usize,
    /// Memories evaluated by the archival pipeline.
    pub archival_evaluated: usize,
    /// Keep decisions.
    pub archival_keep: usize,
    /// Archive recommendations. Physical archive storage is left for later.
    pub archival_archive: usize,
    /// Delete recommendations.
    pub archival_delete: usize,
    /// Consolidation recommendations from the archival pipeline.
    pub archival_consolidate: usize,
    /// Delete recommendations actually applied.
    pub archival_delete_applied: usize,
    /// Consolidation candidate clusters found.
    pub consolidation_candidates: usize,
    /// Consolidation clusters applied.
    pub consolidated: usize,
    /// IDs of memories created by consolidation.
    pub consolidated_memory_ids: Vec<MemoryId>,
    /// Entity pairs linked by the sync resolver.
    pub entity_pairs_linked: usize,
    /// Graph edges created by entity linking.
    pub entity_edges_created: usize,
    /// Entity pairs left ambiguous by linking.
    pub entity_pairs_ambiguous: usize,
    /// Whether LLM enrichment is pending after this run.
    pub enrichment_pending: bool,
    /// Candidate memories available for LLM enrichment.
    pub enrichment_candidates: usize,
    /// Non-fatal maintenance issues.
    pub issues: Vec<SleepMaintenanceIssue>,
}

/// Cross-process lease that prevents overlapping sleep maintenance runs.
#[derive(Debug)]
pub struct SleepMaintenanceLease {
    file: File,
    path: PathBuf,
}

impl SleepMaintenanceLease {
    /// Path to the lock file backing this lease.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for SleepMaintenanceLease {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

impl MenteDb {
    /// Try to run sleep maintenance while holding the database lease.
    ///
    /// Returns `Ok(None)` when another process already owns the lease.
    pub fn try_run_sleep_maintenance(
        &self,
        config: SleepMaintenanceConfig,
    ) -> MenteResult<Option<SleepMaintenanceResult>> {
        let Some(_lease) = self.try_acquire_sleep_maintenance_lease()? else {
            return Ok(None);
        };
        self.run_sleep_maintenance(config).map(Some)
    }

    /// Run bounded sleep maintenance without acquiring the process lease.
    ///
    /// Use `try_run_sleep_maintenance` from background workers. This direct
    /// method is useful when the caller already coordinates execution.
    pub fn run_sleep_maintenance(
        &self,
        config: SleepMaintenanceConfig,
    ) -> MenteResult<SleepMaintenanceResult> {
        let loaded = self.load_sleep_maintenance_memories(config.max_memories);
        let mut result = SleepMaintenanceResult {
            processed_memories: loaded.len(),
            ..SleepMaintenanceResult::default()
        };

        if config.apply_decay {
            self.apply_sleep_decay(&loaded, &config, &mut result)?;
        }

        let loaded = self.load_sleep_maintenance_memories(config.max_memories);
        if config.evaluate_archival {
            self.evaluate_sleep_archival(&loaded, &config, &mut result)?;
        }

        if config.run_consolidation {
            self.run_sleep_consolidation(&loaded, &config, &mut result);
        }

        if config.link_entities {
            let entity_result = self.link_entities()?;
            result.entity_pairs_linked = entity_result.linked;
            result.entity_edges_created = entity_result.edges_created;
            result.entity_pairs_ambiguous = entity_result.ambiguous;
        }

        result.enrichment_pending = self.needs_enrichment();
        result.enrichment_candidates = if result.enrichment_pending {
            self.enrichment_candidates().len()
        } else {
            0
        };

        Ok(result)
    }

    /// Try to acquire the sleep maintenance lease for this database.
    ///
    /// Returns `Ok(None)` when another process is already running maintenance.
    pub fn try_acquire_sleep_maintenance_lease(
        &self,
    ) -> MenteResult<Option<SleepMaintenanceLease>> {
        let lock_dir = self.path.join("locks");
        std::fs::create_dir_all(&lock_dir)?;
        let lock_path = lock_dir.join("sleep-maintenance.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;

        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(SleepMaintenanceLease {
                file,
                path: lock_path,
            })),
            Err(err) if err.kind() == ErrorKind::WouldBlock => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    fn load_sleep_maintenance_memories(
        &self,
        max_memories: usize,
    ) -> Vec<(MemoryId, MemoryNode)> {
        if max_memories == 0 {
            return Vec::new();
        }

        let mut entries: Vec<(MemoryId, MemoryNode)> = self
            .db
            .all_memories()
            .unwrap_or_default()
            .into_iter()
            .map(|node| (node.id, node))
            .collect();
        entries.sort_by_key(|(memory_id, _)| *memory_id);
        entries.truncate(max_memories);
        entries
    }

    fn apply_sleep_decay(
        &self,
        loaded: &[(MemoryId, MemoryNode)],
        config: &SleepMaintenanceConfig,
        result: &mut SleepMaintenanceResult,
    ) -> MenteResult<()> {
        let now = current_timestamp_us();
        for (memory_id, node) in loaded {
            let new_salience = self.decay.compute_decay(
                node.salience,
                node.created_at,
                node.accessed_at,
                node.access_count,
                now,
            );
            if (new_salience - node.salience).abs() <= config.decay_write_epsilon {
                continue;
            }

            let mut updated = node.clone();
            updated.salience = new_salience;
            // The db upsert refreshes FTS5 (trigger) and vec0 automatically.
            self.db.store_memory(&updated)?;
            let _ = memory_id;
            result.decay_updated += 1;
        }
        Ok(())
    }

    fn evaluate_sleep_archival(
        &self,
        loaded: &[(MemoryId, MemoryNode)],
        config: &SleepMaintenanceConfig,
        result: &mut SleepMaintenanceResult,
    ) -> MenteResult<()> {
        let memories: Vec<MemoryNode> = loaded.iter().map(|(_, node)| node.clone()).collect();
        let decisions = self.evaluate_archival_batch(&memories);
        result.archival_evaluated = decisions.len();

        for (memory_id, decision) in decisions {
            match decision {
                ArchivalDecision::Keep => {
                    result.archival_keep += 1;
                }
                ArchivalDecision::Archive => {
                    result.archival_archive += 1;
                }
                ArchivalDecision::Delete => {
                    result.archival_delete += 1;
                    if config.apply_archival_deletes {
                        self.forget(memory_id)?;
                        result.archival_delete_applied += 1;
                    }
                }
                ArchivalDecision::Consolidate(_) => {
                    result.archival_consolidate += 1;
                }
            }
        }

        Ok(())
    }

    fn run_sleep_consolidation(
        &self,
        loaded: &[(MemoryId, MemoryNode)],
        config: &SleepMaintenanceConfig,
        result: &mut SleepMaintenanceResult,
    ) {
        let candidates = self.find_sleep_consolidation_candidates(loaded, config);
        result.consolidation_candidates = candidates.len();

        for candidate in candidates
            .into_iter()
            .take(config.max_consolidation_clusters)
        {
            match self.consolidate_cluster(&candidate.memories) {
                Ok(memory_id) => {
                    result.consolidated += 1;
                    result.consolidated_memory_ids.push(memory_id);
                }
                Err(err) => result.issues.push(SleepMaintenanceIssue {
                    stage: SleepMaintenanceStage::Consolidation,
                    memory_id: None,
                    message: err.to_string(),
                }),
            }
        }
    }

    fn find_sleep_consolidation_candidates(
        &self,
        loaded: &[(MemoryId, MemoryNode)],
        config: &SleepMaintenanceConfig,
    ) -> Vec<ConsolidationCandidate> {
        let now = current_timestamp_us();
        let eligible: Vec<MemoryNode> = loaded
            .iter()
            .map(|(_, node)| node)
            .filter(|node| ConsolidationEngine::should_consolidate(node, now))
            .cloned()
            .collect();

        if eligible.is_empty() {
            return Vec::new();
        }

        self.consolidation.find_candidates(
            &eligible,
            config.consolidation_min_cluster_size,
            config.consolidation_similarity_threshold,
        )
    }
}

fn current_timestamp_us() -> Timestamp {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
