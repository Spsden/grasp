//! Backward-compatible storage facade implemented on top of SQLite.

use std::path::Path;

use mentedb_core::MemoryNode;
use mentedb_core::error::{MenteError, MenteResult};
use mentedb_core::types::MemoryId;
use mentedb_sqlite::Backend;

/// Stable storage handle returned by this compatibility layer.
///
/// The old crate returned page identifiers from a bespoke page file. The
/// SQLite backend stores memories by their canonical `MemoryId`, so the storage
/// handle is now the memory id itself.
pub type StorageId = MemoryId;

/// Compatibility wrapper for callers that still import `mentedb_storage`.
pub struct StorageEngine {
    backend: Backend,
}

impl StorageEngine {
    /// Open a SQLite database under the provided directory.
    pub fn open(path: &Path) -> MenteResult<Self> {
        std::fs::create_dir_all(path)?;
        let db_path = path.join("mentedb.sqlite");
        Ok(Self {
            backend: Backend::open(&db_path, 0)?,
        })
    }

    /// Access the real SQLite backend for advanced inspection.
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Store or update a memory and return its canonical id.
    pub fn store_memory(&self, node: &MemoryNode) -> MenteResult<StorageId> {
        if !node.embedding.is_empty() && self.backend.embedding_dim() == 0 {
            self.backend.ensure_vector_index(node.embedding.len())?;
        }
        self.backend.store_memory(node)?;
        Ok(node.id)
    }

    /// Store many memories in one SQLite transaction.
    pub fn store_memory_batch(&self, nodes: &[MemoryNode]) -> MenteResult<Vec<StorageId>> {
        if let Some(first_dim) = nodes.iter().find_map(|node| {
            if node.embedding.is_empty() {
                None
            } else {
                Some(node.embedding.len())
            }
        }) && self.backend.embedding_dim() == 0
        {
            self.backend.ensure_vector_index(first_dim)?;
        }
        self.backend.store_memory_batch(nodes)?;
        Ok(nodes.iter().map(|node| node.id).collect())
    }

    /// Load a memory by storage id.
    pub fn load_memory(&self, id: StorageId) -> MenteResult<MemoryNode> {
        self.backend
            .get_memory(id)?
            .ok_or(MenteError::MemoryNotFound(id))
    }

    /// Delete a memory and its derived rows.
    pub fn delete_memory(&self, id: StorageId) -> MenteResult<bool> {
        self.backend.delete_memory(id)
    }

    /// Scan all memories, returning `(memory_id, storage_id)` pairs.
    pub fn scan_all_memories(&self) -> Vec<(MemoryId, StorageId)> {
        self.backend
            .all_memory_ids()
            .unwrap_or_default()
            .into_iter()
            .map(|id| (id, id))
            .collect()
    }

    /// Search vectors via the SQLite backend.
    pub fn search_vectors(&self, query: &[f32], k: usize) -> MenteResult<Vec<(MemoryId, f32)>> {
        self.backend.knn(query, k)
    }

    /// SQLite owns checkpointing internally in WAL mode.
    pub fn checkpoint(&self) -> MenteResult<()> {
        Ok(())
    }

    /// Drop closes SQLite resources. This method is retained for compatibility.
    pub fn close(&self) -> MenteResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    #[test]
    fn stores_loads_and_scans_memories() {
        let dir = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(dir.path()).unwrap();
        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Episodic,
            "The user prefers Rust over Go".to_string(),
            vec![0.1, 0.2, 0.3, 0.4],
        );

        let id = engine.store_memory(&node).unwrap();
        let loaded = engine.load_memory(id).unwrap();

        assert_eq!(node.id, loaded.id);
        assert_eq!(node.content, loaded.content);
        assert_eq!(engine.scan_all_memories(), vec![(node.id, node.id)]);
    }

    #[test]
    fn persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let node = MemoryNode::new(
            AgentId::new(),
            MemoryType::Semantic,
            "persisted memory".to_string(),
            vec![1.0, 0.0],
        );
        let id = node.id;

        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            engine.store_memory(&node).unwrap();
            engine.close().unwrap();
        }
        {
            let engine = StorageEngine::open(dir.path()).unwrap();
            let loaded = engine.load_memory(id).unwrap();
            assert_eq!(loaded.content, "persisted memory");
        }
    }
}
