//! Integration tests for the SQLite-backed compatibility storage facade.

use mentedb_core::MemoryNode;
use mentedb_core::memory::MemoryType;
use mentedb_core::types::AgentId;
use mentedb_storage::StorageEngine;

#[test]
fn test_store_and_load_memory() {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();

    let node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Episodic,
        "The user prefers Rust over Go".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
    );

    let storage_id = engine.store_memory(&node).unwrap();
    let loaded = engine.load_memory(storage_id).unwrap();

    assert_eq!(storage_id, node.id);
    assert_eq!(node.id, loaded.id);
    assert_eq!(node.content, loaded.content);
    assert_eq!(node.embedding, loaded.embedding);
    assert_eq!(node.memory_type, loaded.memory_type);
    assert_eq!(node.agent_id, loaded.agent_id);
}

#[test]
fn test_multiple_memories() {
    let dir = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(dir.path()).unwrap();

    let nodes: Vec<MemoryNode> = (0..10)
        .map(|i| {
            MemoryNode::new(
                AgentId::new(),
                MemoryType::Semantic,
                format!("memory #{i}"),
                vec![i as f32; 4],
            )
        })
        .collect();

    let storage_ids = engine.store_memory_batch(&nodes).unwrap();

    for (node, storage_id) in nodes.iter().zip(storage_ids.iter()) {
        let loaded = engine.load_memory(*storage_id).unwrap();
        assert_eq!(node.id, loaded.id);
        assert_eq!(node.content, loaded.content);
        assert_eq!(node.embedding, loaded.embedding);
    }
}

#[test]
fn test_persist_across_reopen() {
    let dir = tempfile::tempdir().unwrap();

    let node = MemoryNode::new(
        AgentId::new(),
        MemoryType::Procedural,
        "persisted memory".to_string(),
        vec![3.14, 2.72],
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
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.content, "persisted memory");
    }
}

#[test]
fn test_checkpoint_and_reload() {
    let dir = tempfile::tempdir().unwrap();

    let node = MemoryNode::new(
        AgentId::new(),
        MemoryType::AntiPattern,
        "do not use global state".to_string(),
        vec![0.0, 1.0],
    );
    let id = node.id;

    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        engine.store_memory(&node).unwrap();
        engine.checkpoint().unwrap();
        engine.close().unwrap();
    }
    {
        let engine = StorageEngine::open(dir.path()).unwrap();
        let loaded = engine.load_memory(id).unwrap();
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.content, "do not use global state");
    }
}
