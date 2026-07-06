use mentedb::prelude::*;
use mentedb::{CognitiveConfig, GraphProjectionConfig, MenteDb};
use mentedb_core::types::AgentId;

fn memory(agent: AgentId, content: &str, salience: f32, created_at: u64) -> MemoryNode {
    let mut node = MemoryNode::new(
        agent,
        MemoryType::Episodic,
        content.to_string(),
        vec![salience, 1.0 - salience],
    );
    node.salience = salience;
    node.created_at = created_at;
    node.accessed_at = created_at;
    node
}

fn edge(source: MemoryId, target: MemoryId, edge_type: EdgeType) -> MemoryEdge {
    MemoryEdge {
        source,
        target,
        edge_type,
        weight: 0.9,
        created_at: 42,
        valid_from: None,
        valid_until: None,
        label: Some("test".to_string()),
    }
}

fn open_projection_test_db() -> (MenteDb, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let config = CognitiveConfig {
        write_inference: false,
        ..Default::default()
    };
    let db = MenteDb::open_with_config(dir.path(), config).unwrap();
    (db, dir)
}

#[test]
fn graph_projection_sorts_limits_and_filters_edges() {
    let (db, _dir) = open_projection_test_db();
    let agent = AgentId::new();

    let high = memory(agent, "High salience preference", 0.9, 3);
    let mid = memory(agent, "Mid salience project", 0.5, 2);
    let low = memory(agent, "Low salience note", 0.1, 1);
    let high_id = high.id;
    let mid_id = mid.id;
    let low_id = low.id;

    db.store(high).unwrap();
    db.store(mid).unwrap();
    db.store(low).unwrap();
    db.relate(edge(high_id, mid_id, EdgeType::Related)).unwrap();
    db.relate(edge(mid_id, low_id, EdgeType::Related)).unwrap();

    let projection = db
        .graph_projection(GraphProjectionConfig {
            limit: 2,
            ..GraphProjectionConfig::default()
        })
        .unwrap();

    assert_eq!(projection.available_nodes, 3);
    assert!(projection.truncated);
    assert_eq!(projection.nodes.len(), 2);
    assert_eq!(projection.nodes[0].id, high_id);
    assert_eq!(projection.nodes[1].id, mid_id);
    assert_eq!(projection.edges.len(), 1);
    assert_eq!(projection.edges[0].source, high_id);
    assert_eq!(projection.edges[0].target, mid_id);
}

#[test]
fn centered_graph_projection_returns_reachable_subgraph() {
    let (db, _dir) = open_projection_test_db();
    let agent = AgentId::new();

    let root = memory(agent, "Root", 0.6, 1);
    let child = memory(agent, "Child", 0.5, 2);
    let grandchild = memory(agent, "Grandchild", 0.4, 3);
    let unrelated = memory(agent, "Unrelated", 1.0, 4);
    let root_id = root.id;
    let child_id = child.id;
    let grandchild_id = grandchild.id;
    let unrelated_id = unrelated.id;

    db.store(root).unwrap();
    db.store(child).unwrap();
    db.store(grandchild).unwrap();
    db.store(unrelated).unwrap();
    db.relate(edge(root_id, child_id, EdgeType::Caused))
        .unwrap();
    db.relate(edge(child_id, grandchild_id, EdgeType::Before))
        .unwrap();

    let projection = db
        .graph_projection(GraphProjectionConfig {
            center: Some(root_id),
            depth: 2,
            limit: 10,
            ..GraphProjectionConfig::default()
        })
        .unwrap();

    let node_ids: Vec<MemoryId> = projection.nodes.iter().map(|node| node.id).collect();
    assert!(node_ids.contains(&root_id));
    assert!(node_ids.contains(&child_id));
    assert!(node_ids.contains(&grandchild_id));
    assert!(!node_ids.contains(&unrelated_id));
    assert_eq!(projection.edges.len(), 2);
}

#[test]
fn graph_projection_hides_invalidated_memories_by_default() {
    let (db, _dir) = open_projection_test_db();
    let agent = AgentId::new();

    let mut active = memory(agent, "Active", 0.4, 1);
    let mut invalidated = memory(agent, "Invalidated", 0.9, 2);
    invalidated.invalidate(3);
    active.tags.push("entity:active".to_string());

    let active_id = active.id;
    let invalidated_id = invalidated.id;
    db.store(active).unwrap();
    db.store(invalidated).unwrap();

    let default_projection = db
        .graph_projection(GraphProjectionConfig::default())
        .unwrap();
    let default_ids: Vec<MemoryId> = default_projection
        .nodes
        .iter()
        .map(|node| node.id)
        .collect();
    assert!(default_ids.contains(&active_id));
    assert!(!default_ids.contains(&invalidated_id));

    let history_projection = db
        .graph_projection(GraphProjectionConfig {
            include_invalidated: true,
            ..GraphProjectionConfig::default()
        })
        .unwrap();
    let history_ids: Vec<MemoryId> = history_projection
        .nodes
        .iter()
        .map(|node| node.id)
        .collect();
    assert!(history_ids.contains(&invalidated_id));
}
