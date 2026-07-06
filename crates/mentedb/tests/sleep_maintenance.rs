use mentedb::prelude::*;
use mentedb::{CognitiveConfig, MenteDb, SleepMaintenanceConfig};
use mentedb_core::types::AgentId;

fn open_sleep_test_db() -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = CognitiveConfig {
        write_inference: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();
    (db, dir)
}

fn memory(agent: AgentId, content: &str, salience: f32, accessed_at: u64) -> MemoryNode {
    let mut node = MemoryNode::new(
        agent,
        MemoryType::Episodic,
        content.to_string(),
        vec![salience, 1.0 - salience],
    );
    node.salience = salience;
    node.created_at = accessed_at;
    node.accessed_at = accessed_at;
    node
}

#[test]
fn sleep_maintenance_applies_bounded_decay() {
    let (db, _dir) = open_sleep_test_db();
    let agent = AgentId::new();
    let old = memory(agent, "Old preference", 1.0, 0);
    let old_id = old.id;

    db.store(old).unwrap();

    let result = db
        .run_sleep_maintenance(SleepMaintenanceConfig {
            max_memories: 1,
            run_consolidation: false,
            link_entities: false,
            ..SleepMaintenanceConfig::default()
        })
        .unwrap();

    assert_eq!(result.processed_memories, 1);
    assert_eq!(result.decay_updated, 1);
    assert_eq!(result.archival_evaluated, 1);

    let reloaded = db.get_memory(old_id).unwrap();
    assert!(reloaded.salience < 1.0);
}

#[test]
fn sleep_maintenance_reports_archival_without_deleting_by_default() {
    let (db, _dir) = open_sleep_test_db();
    let agent = AgentId::new();
    let old = memory(agent, "Very old low-salience note", 0.01, 0);
    let old_id = old.id;

    db.store(old).unwrap();

    let result = db
        .run_sleep_maintenance(SleepMaintenanceConfig {
            max_memories: 1,
            apply_decay: false,
            run_consolidation: false,
            link_entities: false,
            ..SleepMaintenanceConfig::default()
        })
        .unwrap();

    assert_eq!(result.archival_delete, 1);
    assert_eq!(result.archival_delete_applied, 0);
    assert!(db.get_memory(old_id).is_ok());
}

#[test]
fn sleep_maintenance_lease_prevents_overlapping_runs() {
    let (db, _dir) = open_sleep_test_db();

    let first = db.try_acquire_sleep_maintenance_lease().unwrap();
    assert!(first.is_some());

    let second = db.try_acquire_sleep_maintenance_lease().unwrap();
    assert!(second.is_none());

    drop(first);

    let third = db.try_acquire_sleep_maintenance_lease().unwrap();
    assert!(third.is_some());
}
