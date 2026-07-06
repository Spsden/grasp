use std::collections::HashSet;

use mentedb_consolidation::archival::ArchivalDecision;
use mentedb_core::edge::EdgeType;
use mentedb_core::types::{MemoryId, Timestamp};
use mentedb_core::{MemoryEdge, MemoryNode};
use mentedb_sqlite::{
    CorrectionPolicyMutation, ForgetPolicyMutation, LifecyclePolicyMutation, MemoryLifecycleEvent,
    PolicyAction, PolicyRun,
};
use serde_json::json;

use crate::{MenteDb, MenteResult, current_time_us};

/// Whether a forget operation should hide data or physically remove it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForgetMode {
    /// Invalidate memories and derived rows while preserving inspectability.
    Soft,
    /// Delete affected rows and rebuildable projections.
    Hard,
}

impl ForgetMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Soft => "soft",
            Self::Hard => "hard",
        }
    }
}

/// Scope used by the forget policy resolver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForgetScope {
    Memory(MemoryId),
    Entity(String),
    Alias(String),
    Conversation(String),
    Tag(String),
}

impl ForgetScope {
    fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Memory(id) => json!({ "type": "memory", "memory_id": id.to_string() }),
            Self::Entity(id) => json!({ "type": "entity", "entity_id": id }),
            Self::Alias(alias) => json!({ "type": "alias", "alias": alias }),
            Self::Conversation(id) => json!({ "type": "conversation", "conversation_id": id }),
            Self::Tag(tag) => json!({ "type": "tag", "tag": tag }),
        }
    }
}

/// Request to apply a correction against prior memories or derived rows.
#[derive(Debug, Clone)]
pub struct CorrectionRequest {
    pub new_memory_id: MemoryId,
    pub old_memory_id: Option<MemoryId>,
    pub claim_ids: Vec<String>,
    pub relationship_ids: Vec<String>,
    pub reason: Option<String>,
}

impl CorrectionRequest {
    pub fn new(new_memory_id: MemoryId) -> Self {
        Self {
            new_memory_id,
            old_memory_id: None,
            claim_ids: Vec::new(),
            relationship_ids: Vec::new(),
            reason: None,
        }
    }
}

/// IDs affected by one policy plan.
#[derive(Debug, Clone, Default)]
pub struct PolicyAffected {
    pub memory_ids: Vec<MemoryId>,
    pub claim_ids: Vec<String>,
    pub relationship_ids: Vec<String>,
    pub entity_ids: Vec<String>,
}

impl PolicyAffected {
    fn to_json(&self) -> serde_json::Value {
        json!({
            "memory_ids": ids_json(&self.memory_ids),
            "claim_ids": self.claim_ids,
            "relationship_ids": self.relationship_ids,
            "entity_ids": self.entity_ids,
        })
    }
}

/// Previewable policy plan.
#[derive(Debug, Clone)]
pub struct PolicyPlan {
    pub run: PolicyRun,
    pub actions: Vec<PolicyAction>,
    pub affected: PolicyAffected,
}

/// Result of applying a policy plan.
#[derive(Debug, Clone)]
pub struct PolicyReport {
    pub run_id: String,
    pub actions: usize,
    pub affected: PolicyAffected,
}

impl MenteDb {
    /// Preview a correction plan without mutating storage.
    pub fn preview_correction_policy(&self, request: CorrectionRequest) -> MenteResult<PolicyPlan> {
        self.build_correction_plan(request, false)
    }

    /// Apply a correction plan.
    pub fn apply_correction_policy(&self, request: CorrectionRequest) -> MenteResult<PolicyReport> {
        let plan = self.build_correction_plan(request.clone(), true)?;
        let now = plan.run.completed_at.unwrap_or_else(current_time_us);
        let memory_invalidations: Vec<(MemoryId, Timestamp)> = plan
            .affected
            .memory_ids
            .iter()
            .map(|id| (*id, now))
            .collect();
        let edges = request
            .old_memory_id
            .map(|old| {
                vec![MemoryEdge {
                    source: request.new_memory_id,
                    target: old,
                    edge_type: EdgeType::Supersedes,
                    weight: 1.0,
                    created_at: now,
                    valid_from: None,
                    valid_until: None,
                    label: request.reason.clone(),
                }]
            })
            .unwrap_or_default();
        let lifecycle_events = correction_lifecycle_events(&plan, request.new_memory_id, now);

        self.db.apply_correction_policy(CorrectionPolicyMutation {
            run: &plan.run,
            actions: &plan.actions,
            memory_invalidations: &memory_invalidations,
            corrected_claim_ids: &plan.affected.claim_ids,
            corrected_relationship_ids: &plan.affected.relationship_ids,
            edges: &edges,
            lifecycle_events: &lifecycle_events,
            applied_at: now,
        })?;
        for edge in &edges {
            let _ = self.graph.add_relationship(edge);
        }

        Ok(PolicyReport {
            run_id: plan.run.run_id.clone(),
            actions: plan.actions.len(),
            affected: plan.affected,
        })
    }

    /// Preview a forget plan without mutating storage.
    pub fn preview_forget_policy(
        &self,
        scope: ForgetScope,
        mode: ForgetMode,
    ) -> MenteResult<PolicyPlan> {
        self.build_forget_plan(scope, mode, false)
    }

    /// Apply a forget plan.
    pub fn apply_forget_policy(
        &self,
        scope: ForgetScope,
        mode: ForgetMode,
    ) -> MenteResult<PolicyReport> {
        let plan = self.build_forget_plan(scope, mode, true)?;
        let now = plan.run.completed_at.unwrap_or_else(current_time_us);
        let lifecycle_events = forget_lifecycle_events(&plan, mode, now);
        self.db.apply_forget_policy(ForgetPolicyMutation {
            run: &plan.run,
            actions: &plan.actions,
            memory_ids: &plan.affected.memory_ids,
            claim_ids: &plan.affected.claim_ids,
            relationship_ids: &plan.affected.relationship_ids,
            entity_ids: &plan.affected.entity_ids,
            lifecycle_events: &lifecycle_events,
            hard_delete: mode == ForgetMode::Hard,
            applied_at: now,
        })?;
        if mode == ForgetMode::Hard {
            for memory_id in &plan.affected.memory_ids {
                self.graph.remove_memory(*memory_id);
            }
        }

        Ok(PolicyReport {
            run_id: plan.run.run_id.clone(),
            actions: plan.actions.len(),
            affected: plan.affected,
        })
    }

    /// Apply salience decay as a durable policy run.
    pub fn apply_decay_policy_global(&self) -> MenteResult<PolicyReport> {
        let now = current_time_us();
        let mut run = PolicyRun::new("decay", "apply");
        run.status = "applied".to_string();
        run.completed_at = Some(now);

        let mut updates = Vec::new();
        let mut actions = Vec::new();
        let mut events = Vec::new();
        for mut node in self.db.all_memories()? {
            let new_salience = self.decay.compute_decay(
                node.salience,
                node.created_at,
                node.accessed_at,
                node.access_count,
                now,
            );
            if (new_salience - node.salience).abs() <= 0.001 {
                continue;
            }
            let previous_salience = node.salience;
            node.salience = new_salience;
            actions.push(applied_action(
                &run.run_id,
                "decay_memory",
                "memory",
                Some(node.id.to_string()),
                json!({ "salience": previous_salience }),
                json!({ "salience": new_salience }),
            ));
            events.push(lifecycle_event(
                node.id,
                "decayed",
                Some("salience_decay_applied"),
                Some("decay"),
                json!({
                    "previous_salience": previous_salience,
                    "new_salience": new_salience,
                    "applied_at": now,
                }),
            ));
            updates.push(node);
        }

        run.scope_json = json!({ "type": "global" }).to_string();
        run.result_json = json!({ "memory_count": updates.len() }).to_string();
        let affected = PolicyAffected {
            memory_ids: updates.iter().map(|node| node.id).collect(),
            ..Default::default()
        };
        self.db.apply_lifecycle_policy(LifecyclePolicyMutation {
            run: &run,
            actions: &actions,
            memory_updates: &updates,
            lifecycle_events: &events,
        })?;

        Ok(PolicyReport {
            run_id: run.run_id,
            actions: actions.len(),
            affected,
        })
    }

    /// Apply archive decisions as a durable policy run.
    pub fn apply_archive_policy_global(&self) -> MenteResult<PolicyReport> {
        let now = current_time_us();
        let mut run = PolicyRun::new("archive", "apply");
        run.status = "applied".to_string();
        run.completed_at = Some(now);

        let mut updates = Vec::new();
        let mut actions = Vec::new();
        let mut events = Vec::new();
        for mut node in self.db.all_memories()? {
            if memory_is_archived(&node) {
                continue;
            }
            match self.archival.evaluate(&node, now) {
                ArchivalDecision::Archive | ArchivalDecision::Delete => {
                    let previous_tags = node.tags.clone();
                    let previous_salience = node.salience;
                    node.tags.push("archived".to_string());
                    node.salience = node.salience.min(0.02);
                    actions.push(applied_action(
                        &run.run_id,
                        "archive_memory",
                        "memory",
                        Some(node.id.to_string()),
                        json!({ "tags": previous_tags, "salience": previous_salience }),
                        json!({ "tags": node.tags, "salience": node.salience }),
                    ));
                    events.push(lifecycle_event(
                        node.id,
                        "archived",
                        Some("archival_policy_applied"),
                        Some("archive"),
                        json!({ "applied_at": now }),
                    ));
                    updates.push(node);
                }
                ArchivalDecision::Keep | ArchivalDecision::Consolidate(_) => {}
            }
        }

        run.scope_json = json!({ "type": "global" }).to_string();
        run.result_json = json!({ "memory_count": updates.len() }).to_string();
        let affected = PolicyAffected {
            memory_ids: updates.iter().map(|node| node.id).collect(),
            ..Default::default()
        };
        self.db.apply_lifecycle_policy(LifecyclePolicyMutation {
            run: &run,
            actions: &actions,
            memory_updates: &updates,
            lifecycle_events: &events,
        })?;

        Ok(PolicyReport {
            run_id: run.run_id,
            actions: actions.len(),
            affected,
        })
    }

    /// Recent policy runs, newest first.
    pub fn recent_policy_runs(&self, limit: usize) -> MenteResult<Vec<PolicyRun>> {
        self.db.recent_policy_runs(limit)
    }

    /// Actions for one policy run.
    pub fn policy_actions_for_run(&self, run_id: &str) -> MenteResult<Vec<PolicyAction>> {
        self.db.policy_actions_for_run(run_id)
    }

    fn build_correction_plan(
        &self,
        request: CorrectionRequest,
        apply: bool,
    ) -> MenteResult<PolicyPlan> {
        let new_memory = self.get_memory(request.new_memory_id)?;
        let mut affected = PolicyAffected::default();
        let mut claim_ids: HashSet<String> = request.claim_ids.iter().cloned().collect();
        let mut relationship_ids: HashSet<String> =
            request.relationship_ids.iter().cloned().collect();

        if let Some(old_memory_id) = request.old_memory_id {
            affected.memory_ids.push(old_memory_id);
            for claim in self.db.claims_for_memory(old_memory_id)? {
                if claim.status == "active" {
                    claim_ids.insert(claim.claim_id);
                }
            }
            for relationship in self.db.relationships_for_memory(old_memory_id)? {
                if relationship.status == "active" {
                    relationship_ids.insert(relationship.relationship_id);
                }
            }
        }

        if claim_ids.is_empty() && relationship_ids.is_empty() {
            self.collect_correction_candidates(
                request.new_memory_id,
                &mut claim_ids,
                &mut relationship_ids,
            )?;
        }

        affected.claim_ids = sorted_strings(claim_ids);
        affected.relationship_ids = sorted_strings(relationship_ids);
        affected.memory_ids.sort_unstable();
        affected.memory_ids.dedup();

        let now = current_time_us();
        let mut run = PolicyRun::new("correction", if apply { "apply" } else { "preview" });
        run.status = if apply { "applied" } else { "planned" }.to_string();
        run.scope_json = json!({
            "new_memory_id": request.new_memory_id.to_string(),
            "old_memory_id": request.old_memory_id.map(|id| id.to_string()),
            "reason": request.reason,
        })
        .to_string();
        run.result_json = affected.to_json().to_string();
        if apply {
            run.completed_at = Some(now);
        }

        let mut actions = Vec::new();
        for memory_id in &affected.memory_ids {
            actions.push(policy_action(
                &run.run_id,
                if apply {
                    "invalidate_memory"
                } else {
                    "would_invalidate_memory"
                },
                "memory",
                Some(memory_id.to_string()),
                json!({}),
                json!({ "valid_until": now }),
                apply,
            ));
        }
        for claim_id in &affected.claim_ids {
            actions.push(policy_action(
                &run.run_id,
                if apply {
                    "correct_claim"
                } else {
                    "would_correct_claim"
                },
                "claim",
                Some(claim_id.clone()),
                json!({ "status": "active" }),
                json!({ "status": "corrected", "valid_until": now }),
                apply,
            ));
        }
        for relationship_id in &affected.relationship_ids {
            actions.push(policy_action(
                &run.run_id,
                if apply {
                    "correct_relationship"
                } else {
                    "would_correct_relationship"
                },
                "relationship",
                Some(relationship_id.clone()),
                json!({ "status": "active" }),
                json!({ "status": "corrected", "valid_until": now }),
                apply,
            ));
        }
        if actions.is_empty() {
            actions.push(policy_action(
                &run.run_id,
                "no_correction_candidates",
                "memory",
                Some(new_memory.id.to_string()),
                json!({}),
                json!({}),
                apply,
            ));
        }

        Ok(PolicyPlan {
            run,
            actions,
            affected,
        })
    }

    fn collect_correction_candidates(
        &self,
        new_memory_id: MemoryId,
        claim_ids: &mut HashSet<String>,
        relationship_ids: &mut HashSet<String>,
    ) -> MenteResult<()> {
        let entity_links = self.db.memory_entity_links(new_memory_id)?;
        for link in entity_links {
            for claim in self.db.claims_for_entity(&link.entity_id)? {
                if claim.status != "active" {
                    continue;
                }
                let evidence = self.db.claim_evidence(&claim.claim_id)?;
                if evidence.iter().any(|ev| ev.memory_id == new_memory_id) {
                    continue;
                }
                claim_ids.insert(claim.claim_id);
            }
            for relationship in self.db.relationships_for_entity(&link.entity_id)? {
                if relationship.status != "active" {
                    continue;
                }
                let evidence = self
                    .db
                    .relationship_evidence(&relationship.relationship_id)?;
                if evidence.iter().any(|ev| ev.memory_id == new_memory_id) {
                    continue;
                }
                relationship_ids.insert(relationship.relationship_id);
            }
        }
        Ok(())
    }

    fn build_forget_plan(
        &self,
        scope: ForgetScope,
        mode: ForgetMode,
        apply: bool,
    ) -> MenteResult<PolicyPlan> {
        let mut affected = PolicyAffected::default();
        self.resolve_forget_scope(&scope, &mut affected)?;
        affected.memory_ids.sort_unstable();
        affected.memory_ids.dedup();
        affected.claim_ids.sort();
        affected.claim_ids.dedup();
        affected.relationship_ids.sort();
        affected.relationship_ids.dedup();
        affected.entity_ids.sort();
        affected.entity_ids.dedup();

        let mut run = PolicyRun::new("forget", if apply { mode.as_str() } else { "preview" });
        run.status = if apply { "applied" } else { "planned" }.to_string();
        run.scope_json = json!({
            "scope": scope.to_json(),
            "mode": mode.as_str(),
        })
        .to_string();
        run.result_json = affected.to_json().to_string();
        if apply {
            run.completed_at = Some(current_time_us());
        }

        let mut actions = Vec::new();
        let action_prefix = if apply { "forget" } else { "would_forget" };
        for memory_id in &affected.memory_ids {
            actions.push(policy_action(
                &run.run_id,
                format!("{action_prefix}_memory"),
                "memory",
                Some(memory_id.to_string()),
                json!({}),
                json!({ "mode": mode.as_str() }),
                apply,
            ));
        }
        for claim_id in &affected.claim_ids {
            actions.push(policy_action(
                &run.run_id,
                format!("{action_prefix}_claim"),
                "claim",
                Some(claim_id.clone()),
                json!({}),
                json!({ "mode": mode.as_str() }),
                apply,
            ));
        }
        for relationship_id in &affected.relationship_ids {
            actions.push(policy_action(
                &run.run_id,
                format!("{action_prefix}_relationship"),
                "relationship",
                Some(relationship_id.clone()),
                json!({}),
                json!({ "mode": mode.as_str() }),
                apply,
            ));
        }
        for entity_id in &affected.entity_ids {
            actions.push(policy_action(
                &run.run_id,
                format!("{action_prefix}_entity"),
                "entity",
                Some(entity_id.clone()),
                json!({}),
                json!({ "mode": mode.as_str() }),
                apply,
            ));
        }

        Ok(PolicyPlan {
            run,
            actions,
            affected,
        })
    }

    fn resolve_forget_scope(
        &self,
        scope: &ForgetScope,
        affected: &mut PolicyAffected,
    ) -> MenteResult<()> {
        match scope {
            ForgetScope::Memory(memory_id) => {
                self.collect_memory_forget_closure(*memory_id, affected)?;
            }
            ForgetScope::Entity(entity_id) => {
                affected.entity_ids.push(entity_id.clone());
                self.collect_entity_forget_closure(entity_id, affected)?;
            }
            ForgetScope::Alias(alias) => {
                for entity in self.db.entities_by_alias(alias)? {
                    affected.entity_ids.push(entity.entity_id.clone());
                    self.collect_entity_forget_closure(&entity.entity_id, affected)?;
                }
            }
            ForgetScope::Conversation(conversation_id) => {
                for memory_id in self.db.memory_ids_for_conversation(conversation_id)? {
                    self.collect_memory_forget_closure(memory_id, affected)?;
                }
            }
            ForgetScope::Tag(tag) => {
                for memory_id in self.db.memory_ids_for_tag(tag)? {
                    self.collect_memory_forget_closure(memory_id, affected)?;
                }
            }
        }
        Ok(())
    }

    fn collect_entity_forget_closure(
        &self,
        entity_id: &str,
        affected: &mut PolicyAffected,
    ) -> MenteResult<()> {
        for link in self.db.memories_for_entity(entity_id)? {
            self.collect_memory_forget_closure(link.memory_id, affected)?;
        }
        for claim in self.db.claims_for_entity(entity_id)? {
            affected.claim_ids.push(claim.claim_id.clone());
            for evidence in self.db.claim_evidence(&claim.claim_id)? {
                affected.memory_ids.push(evidence.memory_id);
            }
        }
        for relationship in self.db.relationships_for_entity(entity_id)? {
            affected
                .relationship_ids
                .push(relationship.relationship_id.clone());
            for evidence in self
                .db
                .relationship_evidence(&relationship.relationship_id)?
            {
                affected.memory_ids.push(evidence.memory_id);
            }
        }
        Ok(())
    }

    fn collect_memory_forget_closure(
        &self,
        memory_id: MemoryId,
        affected: &mut PolicyAffected,
    ) -> MenteResult<()> {
        affected.memory_ids.push(memory_id);
        for claim in self.db.claims_for_memory(memory_id)? {
            affected.claim_ids.push(claim.claim_id);
        }
        for relationship in self.db.relationships_for_memory(memory_id)? {
            affected.relationship_ids.push(relationship.relationship_id);
        }
        Ok(())
    }
}

pub(crate) fn memory_is_archived(memory: &MemoryNode) -> bool {
    memory.tags.iter().any(|tag| tag == "archived")
}

fn correction_lifecycle_events(
    plan: &PolicyPlan,
    new_memory_id: MemoryId,
    now: Timestamp,
) -> Vec<MemoryLifecycleEvent> {
    let mut events = Vec::new();
    for memory_id in &plan.affected.memory_ids {
        events.push(lifecycle_event(
            *memory_id,
            "corrected",
            Some("correction_policy_applied"),
            Some("correction"),
            json!({
                "run_id": plan.run.run_id,
                "corrected_by": new_memory_id.to_string(),
                "valid_until": now,
            }),
        ));
    }
    events.push(lifecycle_event(
        new_memory_id,
        "correction_applied",
        Some("correction_policy_applied"),
        Some("correction"),
        json!({
            "run_id": plan.run.run_id,
            "corrected_memory_count": plan.affected.memory_ids.len(),
            "corrected_claim_count": plan.affected.claim_ids.len(),
            "corrected_relationship_count": plan.affected.relationship_ids.len(),
        }),
    ));
    events
}

fn forget_lifecycle_events(
    plan: &PolicyPlan,
    mode: ForgetMode,
    now: Timestamp,
) -> Vec<MemoryLifecycleEvent> {
    plan.affected
        .memory_ids
        .iter()
        .map(|memory_id| {
            lifecycle_event(
                *memory_id,
                "forgotten",
                Some("forget_policy_applied"),
                Some("forget"),
                json!({
                    "run_id": plan.run.run_id,
                    "mode": mode.as_str(),
                    "applied_at": now,
                }),
            )
        })
        .collect()
}

fn lifecycle_event(
    memory_id: MemoryId,
    event_type: &str,
    reason: Option<&str>,
    policy: Option<&str>,
    payload: serde_json::Value,
) -> MemoryLifecycleEvent {
    let mut event = MemoryLifecycleEvent::new(memory_id, event_type);
    event.reason = reason.map(ToString::to_string);
    event.policy = policy.map(ToString::to_string);
    event.payload_json = payload.to_string();
    event
}

fn policy_action(
    run_id: &str,
    action_type: impl Into<String>,
    target_type: impl Into<String>,
    target_id: Option<String>,
    before: serde_json::Value,
    after: serde_json::Value,
    applied: bool,
) -> PolicyAction {
    let mut action = PolicyAction::new(run_id, action_type, target_type, target_id);
    action.before_json = before.to_string();
    action.after_json = after.to_string();
    action.status = if applied { "applied" } else { "planned" }.to_string();
    action
}

fn applied_action(
    run_id: &str,
    action_type: impl Into<String>,
    target_type: impl Into<String>,
    target_id: Option<String>,
    before: serde_json::Value,
    after: serde_json::Value,
) -> PolicyAction {
    policy_action(
        run_id,
        action_type,
        target_type,
        target_id,
        before,
        after,
        true,
    )
}

fn sorted_strings(values: HashSet<String>) -> Vec<String> {
    let mut values: Vec<String> = values.into_iter().collect();
    values.sort();
    values
}

fn ids_json(ids: &[MemoryId]) -> Vec<String> {
    ids.iter().map(ToString::to_string).collect()
}
