#[cfg(feature = "extraction")]
use std::collections::HashMap;
use std::collections::HashSet;

use mentedb_core::MenteError;
#[cfg(feature = "extraction")]
use mentedb_sqlite::MemoryLifecycleEvent;
use mentedb_sqlite::{
    ClaimEntityLink, ClaimEvidence, ClaimRecord, EntityAlias, EntityRecord, EntityRelationship,
    ExtractionRun, MemoryEntityLink, RelationshipEvidence,
};
#[cfg(feature = "extraction")]
use serde_json::{Value, json};

use crate::MenteResult;

/// Validated derived knowledge from one extraction run.
///
/// The model/provider layer should produce this shape only after preserving the
/// raw episode. Grasp stores the run metadata and all derived artifacts
/// atomically so the derived graph can be inspected or rebuilt later.
#[derive(Debug, Clone)]
pub struct ValidatedExtractionBatch {
    pub run: ExtractionRun,
    pub entities: Vec<EntityRecord>,
    pub entity_aliases: Vec<EntityAlias>,
    pub memory_entities: Vec<MemoryEntityLink>,
    pub claims: Vec<ClaimRecord>,
    pub claim_entities: Vec<ClaimEntityLink>,
    pub claim_evidence: Vec<ClaimEvidence>,
    pub relationships: Vec<EntityRelationship>,
    pub relationship_evidence: Vec<RelationshipEvidence>,
}

/// Validate derived extraction artifacts before writing them to SQLite.
pub fn validate_extraction_batch(batch: &ValidatedExtractionBatch) -> MenteResult<()> {
    validate_run(&batch.run)?;
    let claim_ids: HashSet<&str> = batch
        .claims
        .iter()
        .map(|claim| claim.claim_id.as_str())
        .collect();
    let entity_ids: HashSet<&str> = batch
        .entities
        .iter()
        .map(|entity| entity.entity_id.as_str())
        .collect();
    let relationship_ids: HashSet<&str> = batch
        .relationships
        .iter()
        .map(|relationship| relationship.relationship_id.as_str())
        .collect();

    for entity in &batch.entities {
        validate_entity(entity)?;
    }
    for alias in &batch.entity_aliases {
        require_known(&entity_ids, &alias.entity_id, "entity alias", "entity_id")?;
        require_non_empty(&alias.alias, "entity alias alias")?;
        validate_confidence(alias.confidence, "entity alias confidence")?;
    }
    for link in &batch.memory_entities {
        require_known(
            &entity_ids,
            &link.entity_id,
            "memory entity link",
            "entity_id",
        )?;
        if let Some(role) = &link.role {
            require_non_empty(role, "memory entity link role")?;
        }
        validate_confidence(link.confidence, "memory entity link confidence")?;
    }
    for claim in &batch.claims {
        validate_claim(claim, &entity_ids)?;
    }
    for link in &batch.claim_entities {
        require_known(&claim_ids, &link.claim_id, "claim entity link", "claim_id")?;
        require_known(
            &entity_ids,
            &link.entity_id,
            "claim entity link",
            "entity_id",
        )?;
        require_non_empty(&link.role, "claim entity link role")?;
        validate_confidence(link.confidence, "claim entity link confidence")?;
    }
    for evidence in &batch.claim_evidence {
        require_known(&claim_ids, &evidence.claim_id, "claim evidence", "claim_id")?;
        validate_confidence(evidence.confidence, "claim evidence confidence")?;
        validate_span(
            evidence.span_start,
            evidence.span_end,
            "claim evidence span",
        )?;
    }
    for relationship in &batch.relationships {
        validate_relationship(relationship, &entity_ids)?;
    }
    for evidence in &batch.relationship_evidence {
        require_known(
            &relationship_ids,
            &evidence.relationship_id,
            "relationship evidence",
            "relationship_id",
        )?;
        validate_confidence(evidence.confidence, "relationship evidence confidence")?;
        validate_span(
            evidence.span_start,
            evidence.span_end,
            "relationship evidence span",
        )?;
    }

    Ok(())
}

fn validate_run(run: &ExtractionRun) -> MenteResult<()> {
    require_non_empty(&run.run_id, "extraction run id")?;
    require_non_empty(&run.extractor, "extraction run extractor")?;
    require_non_empty(&run.extractor_version, "extraction run extractor_version")?;
    require_non_empty(&run.status, "extraction run status")?;
    validate_json_object(&run.output_json, "extraction run output_json")
}

fn validate_entity(entity: &EntityRecord) -> MenteResult<()> {
    require_non_empty(&entity.entity_id, "entity id")?;
    require_non_empty(&entity.entity_type, "entity type")?;
    require_non_empty(&entity.canonical, "entity canonical")?;
    validate_confidence(entity.confidence, "entity confidence")?;
    validate_json_object(&entity.attributes_json, "entity attributes_json")
}

fn validate_claim(claim: &ClaimRecord, entity_ids: &HashSet<&str>) -> MenteResult<()> {
    require_non_empty(&claim.claim_id, "claim id")?;
    require_non_empty(&claim.claim_text, "claim text")?;
    require_non_empty(&claim.claim_type, "claim type")?;
    require_non_empty(&claim.status, "claim status")?;
    if let Some(entity_id) = &claim.subject_entity_id {
        require_known(entity_ids, entity_id, "claim", "subject_entity_id")?;
    }
    if let Some(entity_id) = &claim.object_entity_id {
        require_known(entity_ids, entity_id, "claim", "object_entity_id")?;
    }
    validate_confidence(claim.confidence, "claim confidence")?;
    validate_time_bounds(claim.valid_from, claim.valid_until, "claim validity")?;
    validate_json_object(&claim.attributes_json, "claim attributes_json")
}

fn validate_relationship(
    relationship: &EntityRelationship,
    entity_ids: &HashSet<&str>,
) -> MenteResult<()> {
    require_non_empty(&relationship.relationship_id, "relationship id")?;
    require_known(
        entity_ids,
        &relationship.source_entity_id,
        "relationship",
        "source_entity_id",
    )?;
    require_known(
        entity_ids,
        &relationship.target_entity_id,
        "relationship",
        "target_entity_id",
    )?;
    require_non_empty(&relationship.relation_type, "relationship relation_type")?;
    require_non_empty(&relationship.status, "relationship status")?;
    validate_confidence(relationship.confidence, "relationship confidence")?;
    validate_time_bounds(
        relationship.valid_from,
        relationship.valid_until,
        "relationship validity",
    )?;
    validate_json_object(
        &relationship.attributes_json,
        "relationship attributes_json",
    )
}

fn require_non_empty(value: &str, field: &str) -> MenteResult<()> {
    if value.trim().is_empty() {
        return Err(MenteError::InvalidInput(format!(
            "{field} must not be empty"
        )));
    }
    Ok(())
}

fn require_known(known: &HashSet<&str>, value: &str, record: &str, field: &str) -> MenteResult<()> {
    if !known.contains(value) {
        return Err(MenteError::InvalidInput(format!(
            "{record} references unknown {field}: {value}"
        )));
    }
    Ok(())
}

fn validate_confidence(value: f32, field: &str) -> MenteResult<()> {
    if !(0.0..=1.0).contains(&value) || !value.is_finite() {
        return Err(MenteError::InvalidInput(format!(
            "{field} must be finite and between 0 and 1"
        )));
    }
    Ok(())
}

fn validate_span(start: Option<i64>, end: Option<i64>, field: &str) -> MenteResult<()> {
    if let (Some(start), Some(end)) = (start, end)
        && start > end
    {
        return Err(MenteError::InvalidInput(format!(
            "{field} start must be <= end"
        )));
    }
    Ok(())
}

fn validate_time_bounds(
    valid_from: Option<u64>,
    valid_until: Option<u64>,
    field: &str,
) -> MenteResult<()> {
    if let (Some(start), Some(end)) = (valid_from, valid_until)
        && start > end
    {
        return Err(MenteError::InvalidInput(format!(
            "{field} start must be <= end"
        )));
    }
    Ok(())
}

fn validate_json_object(value: &str, field: &str) -> MenteResult<()> {
    let parsed: serde_json::Value = serde_json::from_str(value)
        .map_err(|err| MenteError::Serialization(format!("{field} must be valid JSON: {err}")))?;
    if !parsed.is_object() {
        return Err(MenteError::InvalidInput(format!(
            "{field} must be a JSON object"
        )));
    }
    Ok(())
}

/// Options for converting an LLM extraction response into durable rows.
#[cfg(feature = "extraction")]
#[derive(Debug, Clone)]
pub struct ExtractionStoreOptions {
    pub extractor: String,
    pub extractor_version: String,
    pub model: Option<String>,
    pub prompt_hash: Option<String>,
    pub config_hash: Option<String>,
    pub conversation_id: Option<String>,
}

#[cfg(feature = "extraction")]
impl Default for ExtractionStoreOptions {
    fn default() -> Self {
        Self {
            extractor: "llm_extraction".to_string(),
            extractor_version: "v1".to_string(),
            model: None,
            prompt_hash: None,
            config_hash: None,
            conversation_id: None,
        }
    }
}

/// Counts for rows persisted from one extraction response.
#[cfg(feature = "extraction")]
#[derive(Debug, Clone, Default)]
pub struct ExtractionStoreReport {
    pub run_id: String,
    pub entities: usize,
    pub aliases: usize,
    pub memory_links: usize,
    pub claims: usize,
    pub claim_entity_links: usize,
    pub claim_evidence: usize,
    pub relationships: usize,
    pub relationship_evidence: usize,
}

#[cfg(feature = "extraction")]
impl crate::MenteDb {
    /// Persist a validated LLM extraction result against an existing source memory.
    pub fn store_extraction_result(
        &self,
        source_memory_id: mentedb_core::types::MemoryId,
        result: &mentedb_extraction::schema::ExtractionResult,
        options: ExtractionStoreOptions,
    ) -> MenteResult<ExtractionStoreReport> {
        let source = self.get_memory(source_memory_id)?;
        let now = current_time_us();
        let mut run = ExtractionRun::new(options.extractor.clone(), options.extractor_version);
        run.source_memory_id = Some(source_memory_id);
        run.conversation_id = options.conversation_id;
        run.model = options.model;
        run.prompt_hash = options.prompt_hash;
        run.config_hash = options.config_hash;
        run.status = "completed".to_string();
        run.completed_at = Some(now);

        let mut entity_state = EntityBuildState::new(self, &options.extractor, now);
        let empty_attributes: HashMap<String, String> = HashMap::new();
        let mut memory_entities = Vec::new();
        let mut memory_link_keys = HashSet::new();

        for extracted in &result.entities {
            let confidence = 1.0;
            let entity_id = entity_state.ensure_entity(
                &extracted.name,
                &extracted.entity_type,
                &extracted.attributes,
                confidence,
            )?;
            let role = extracted
                .attributes
                .get("relationship")
                .cloned()
                .unwrap_or_else(|| "mentioned".to_string());
            push_memory_entity_link(
                &mut memory_entities,
                &mut memory_link_keys,
                source_memory_id,
                entity_id,
                Some(role),
                confidence,
                Some(extracted.name.clone()),
            );
        }

        let mut claims = Vec::new();
        let mut claim_entities = Vec::new();
        let mut claim_evidence = Vec::new();

        for (index, extracted) in result.memories.iter().enumerate() {
            require_non_empty(&extracted.content, "extracted memory content")?;
            require_non_empty(&extracted.memory_type, "extracted memory type")?;
            let confidence =
                validate_model_confidence(extracted.confidence, "extracted memory confidence")?;
            let mut claim =
                ClaimRecord::new(extracted.content.trim(), extracted.memory_type.trim());
            claim.confidence = confidence;
            claim.source_run_id = Some(run.run_id.clone());
            claim.attributes_json = json!({
                "source": "llm_memory",
                "source_index": index,
                "tags": extracted.tags,
                "context": extracted.context,
                "reasoning": extracted.reasoning,
                "entities": extracted.entities,
            })
            .to_string();

            let mut linked_entities = Vec::new();
            for (entity_index, entity_name) in extracted.entities.iter().enumerate() {
                if entity_name.trim().is_empty() {
                    continue;
                }
                let entity_id = entity_state.ensure_entity(
                    entity_name,
                    "entity",
                    &empty_attributes,
                    confidence,
                )?;
                let role = match entity_index {
                    0 => "subject",
                    1 => "object",
                    _ => "mentioned",
                }
                .to_string();
                claim_entities.push(ClaimEntityLink {
                    claim_id: claim.claim_id.clone(),
                    entity_id: entity_id.clone(),
                    role: role.clone(),
                    confidence,
                });
                push_memory_entity_link(
                    &mut memory_entities,
                    &mut memory_link_keys,
                    source_memory_id,
                    entity_id.clone(),
                    Some(role),
                    confidence,
                    Some(entity_name.clone()),
                );
                linked_entities.push(entity_id);
            }

            claim.subject_entity_id = linked_entities.first().cloned();
            claim.object_entity_id = linked_entities.get(1).cloned();

            let mut evidence = ClaimEvidence::new(claim.claim_id.clone(), source_memory_id);
            evidence.evidence_text = Some(extracted.content.clone());
            evidence.confidence = confidence;
            if let Some((start, end)) = find_span(&source.content, &extracted.content) {
                evidence.span_start = Some(start);
                evidence.span_end = Some(end);
            }

            claim_evidence.push(evidence);
            claims.push(claim);
        }

        let mut relationships = Vec::new();
        let mut relationship_evidence = Vec::new();
        for extracted in &result.relationships {
            require_non_empty(&extracted.source, "relationship source")?;
            require_non_empty(&extracted.target, "relationship target")?;
            require_non_empty(&extracted.relation_type, "relationship relation_type")?;
            let confidence =
                validate_model_confidence(extracted.confidence, "relationship confidence")?;
            let source_entity_id = entity_state.ensure_entity(
                &extracted.source,
                "entity",
                &empty_attributes,
                confidence,
            )?;
            let target_entity_id = entity_state.ensure_entity(
                &extracted.target,
                "entity",
                &empty_attributes,
                confidence,
            )?;

            push_memory_entity_link(
                &mut memory_entities,
                &mut memory_link_keys,
                source_memory_id,
                source_entity_id.clone(),
                Some("relationship_source".to_string()),
                confidence,
                Some(extracted.source.clone()),
            );
            push_memory_entity_link(
                &mut memory_entities,
                &mut memory_link_keys,
                source_memory_id,
                target_entity_id.clone(),
                Some("relationship_target".to_string()),
                confidence,
                Some(extracted.target.clone()),
            );

            let mut relationship = EntityRelationship::new(
                source_entity_id,
                target_entity_id,
                extracted.relation_type.trim(),
            );
            relationship.confidence = confidence;
            relationship.source_run_id = Some(run.run_id.clone());
            relationship.attributes_json = relationship_attributes_json(extracted)?;

            let mut evidence =
                RelationshipEvidence::new(relationship.relationship_id.clone(), source_memory_id);
            evidence.confidence = confidence;
            let evidence_text = if extracted.evidence.trim().is_empty() {
                format!(
                    "{} {} {}",
                    extracted.source.trim(),
                    extracted.relation_type.trim(),
                    extracted.target.trim()
                )
            } else {
                extracted.evidence.clone()
            };
            evidence.evidence_text = Some(evidence_text.clone());
            if let Some((start, end)) = find_span(&source.content, &evidence_text) {
                evidence.span_start = Some(start);
                evidence.span_end = Some(end);
            }

            relationship_evidence.push(evidence);
            relationships.push(relationship);
        }

        let (entities, aliases) = entity_state.into_parts();
        let report = ExtractionStoreReport {
            run_id: run.run_id.clone(),
            entities: entities.len(),
            aliases: aliases.len(),
            memory_links: memory_entities.len(),
            claims: claims.len(),
            claim_entity_links: claim_entities.len(),
            claim_evidence: claim_evidence.len(),
            relationships: relationships.len(),
            relationship_evidence: relationship_evidence.len(),
        };

        run.output_json = json!({
            "memories": result.memories.len(),
            "entities": result.entities.len(),
            "relationships": result.relationships.len(),
            "persisted": {
                "entities": report.entities,
                "aliases": report.aliases,
                "memory_links": report.memory_links,
                "claims": report.claims,
                "claim_entity_links": report.claim_entity_links,
                "claim_evidence": report.claim_evidence,
                "relationships": report.relationships,
                "relationship_evidence": report.relationship_evidence,
            }
        })
        .to_string();

        self.store_validated_extraction(ValidatedExtractionBatch {
            run,
            entities,
            entity_aliases: aliases,
            memory_entities,
            claims,
            claim_entities,
            claim_evidence,
            relationships,
            relationship_evidence,
        })?;

        let mut event = MemoryLifecycleEvent::new(source_memory_id, "extracted");
        event.reason = Some("llm_extraction_completed".to_string());
        event.policy = Some(options.extractor);
        event.payload_json = json!({
            "run_id": report.run_id,
            "entities": report.entities,
            "claims": report.claims,
            "relationships": report.relationships,
        })
        .to_string();
        self.record_lifecycle_event(event)?;

        Ok(report)
    }
}

#[cfg(feature = "extraction")]
struct EntityBuildState<'a> {
    db: &'a crate::MenteDb,
    entities_by_key: HashMap<String, EntityRecord>,
    aliases: Vec<EntityAlias>,
    alias_keys: HashSet<String>,
    alias_source: &'a str,
    now: u64,
}

#[cfg(feature = "extraction")]
impl<'a> EntityBuildState<'a> {
    fn new(db: &'a crate::MenteDb, alias_source: &'a str, now: u64) -> Self {
        Self {
            db,
            entities_by_key: HashMap::new(),
            aliases: Vec::new(),
            alias_keys: HashSet::new(),
            alias_source,
            now,
        }
    }

    fn ensure_entity(
        &mut self,
        name: &str,
        entity_type: &str,
        attributes: &HashMap<String, String>,
        confidence: f32,
    ) -> MenteResult<String> {
        let canonical = name.trim();
        require_non_empty(canonical, "entity canonical")?;
        let entity_type = if entity_type.trim().is_empty() {
            "entity"
        } else {
            entity_type.trim()
        };
        let confidence = validate_model_confidence(confidence, "entity confidence")?;
        let key = normalize_entity_key(canonical);

        let entity = match self.entities_by_key.get_mut(&key) {
            Some(existing) => {
                existing.confidence = existing.confidence.max(confidence);
                existing.updated_at = self.now;
                merge_entity_attributes(existing, attributes)?;
                existing.clone()
            }
            None => {
                let mut entity = self
                    .db
                    .db
                    .entity_by_canonical(entity_type, canonical)?
                    .unwrap_or_else(|| EntityRecord::new(entity_type, canonical));
                entity.confidence = entity.confidence.max(confidence);
                entity.updated_at = self.now;
                merge_entity_attributes(&mut entity, attributes)?;
                self.entities_by_key.insert(key, entity.clone());
                entity
            }
        };

        push_entity_alias(
            &mut self.aliases,
            &mut self.alias_keys,
            &entity.entity_id,
            canonical,
            self.alias_source,
            confidence,
        );
        let lower = canonical.to_lowercase();
        if lower != canonical {
            push_entity_alias(
                &mut self.aliases,
                &mut self.alias_keys,
                &entity.entity_id,
                &lower,
                self.alias_source,
                confidence,
            );
        }

        Ok(entity.entity_id)
    }

    fn into_parts(self) -> (Vec<EntityRecord>, Vec<EntityAlias>) {
        (self.entities_by_key.into_values().collect(), self.aliases)
    }
}

#[cfg(feature = "extraction")]
fn merge_entity_attributes(
    entity: &mut EntityRecord,
    attributes: &HashMap<String, String>,
) -> MenteResult<()> {
    if attributes.is_empty() {
        return Ok(());
    }
    let mut object = match serde_json::from_str::<Value>(&entity.attributes_json) {
        Ok(Value::Object(object)) => object,
        Ok(_) => {
            return Err(MenteError::InvalidInput(
                "entity attributes_json must be a JSON object".to_string(),
            ));
        }
        Err(err) => {
            return Err(MenteError::Serialization(format!(
                "entity attributes_json must be valid JSON: {err}"
            )));
        }
    };
    for (key, value) in attributes {
        object.insert(key.clone(), Value::String(value.clone()));
    }
    entity.attributes_json = Value::Object(object).to_string();
    Ok(())
}

#[cfg(feature = "extraction")]
fn push_entity_alias(
    aliases: &mut Vec<EntityAlias>,
    alias_keys: &mut HashSet<String>,
    entity_id: &str,
    alias: &str,
    source: &str,
    confidence: f32,
) {
    let alias = alias.trim();
    if alias.is_empty() {
        return;
    }
    let key = format!("{}:{}", entity_id, alias.to_lowercase());
    if alias_keys.insert(key) {
        aliases.push(EntityAlias {
            entity_id: entity_id.to_string(),
            alias: alias.to_string(),
            source: Some(source.to_string()),
            confidence,
        });
    }
}

#[cfg(feature = "extraction")]
fn push_memory_entity_link(
    links: &mut Vec<MemoryEntityLink>,
    link_keys: &mut HashSet<String>,
    memory_id: mentedb_core::types::MemoryId,
    entity_id: String,
    role: Option<String>,
    confidence: f32,
    evidence: Option<String>,
) {
    let role_key = role.clone().unwrap_or_default();
    let key = format!("{memory_id}:{entity_id}:{role_key}");
    if link_keys.insert(key) {
        links.push(MemoryEntityLink {
            memory_id,
            entity_id,
            role,
            confidence,
            evidence,
        });
    }
}

#[cfg(feature = "extraction")]
fn relationship_attributes_json(
    relationship: &mentedb_extraction::schema::ExtractedRelationship,
) -> MenteResult<String> {
    let mut object = serde_json::Map::new();
    object.insert(
        "source_label".to_string(),
        Value::String(relationship.source.clone()),
    );
    object.insert(
        "target_label".to_string(),
        Value::String(relationship.target.clone()),
    );
    for (key, value) in &relationship.attributes {
        object.insert(key.clone(), Value::String(value.clone()));
    }
    let value = Value::Object(object);
    validate_json_object(&value.to_string(), "relationship attributes_json")?;
    Ok(value.to_string())
}

#[cfg(feature = "extraction")]
fn validate_model_confidence(value: f32, field: &str) -> MenteResult<f32> {
    validate_confidence(value, field)?;
    Ok(value)
}

#[cfg(feature = "extraction")]
fn normalize_entity_key(name: &str) -> String {
    name.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

#[cfg(feature = "extraction")]
fn find_span(haystack: &str, needle: &str) -> Option<(i64, i64)> {
    if needle.is_empty() {
        return None;
    }
    haystack
        .find(needle)
        .map(|start| (start as i64, (start + needle.len()) as i64))
}

#[cfg(feature = "extraction")]
fn current_time_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::types::MemoryId;
    use mentedb_sqlite::{ClaimEvidence, ClaimRecord, ExtractionRun};

    #[test]
    fn validates_minimal_batch() {
        let memory_id = MemoryId::new();
        let run = ExtractionRun::new("claim_extractor", "v1");
        let claim = ClaimRecord::new("Pratap uses Flutter", "fact");
        let claim_id = claim.claim_id.clone();
        let mut evidence = ClaimEvidence::new(claim_id.clone(), memory_id);
        evidence.span_start = Some(0);
        evidence.span_end = Some(20);

        validate_extraction_batch(&ValidatedExtractionBatch {
            run,
            entities: Vec::new(),
            entity_aliases: Vec::new(),
            memory_entities: Vec::new(),
            claims: vec![claim],
            claim_entities: Vec::new(),
            claim_evidence: vec![evidence],
            relationships: Vec::new(),
            relationship_evidence: Vec::new(),
        })
        .unwrap();
    }

    #[test]
    fn rejects_evidence_for_missing_claim() {
        let run = ExtractionRun::new("claim_extractor", "v1");
        let evidence = ClaimEvidence::new("missing", MemoryId::new());
        let err = validate_extraction_batch(&ValidatedExtractionBatch {
            run,
            entities: Vec::new(),
            entity_aliases: Vec::new(),
            memory_entities: Vec::new(),
            claims: Vec::new(),
            claim_entities: Vec::new(),
            claim_evidence: vec![evidence],
            relationships: Vec::new(),
            relationship_evidence: Vec::new(),
        })
        .unwrap_err();
        assert!(err.to_string().contains("unknown claim_id"));
    }
}
