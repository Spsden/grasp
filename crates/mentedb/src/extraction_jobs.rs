use std::collections::HashSet;

use mentedb_core::MenteError;
use mentedb_sqlite::{
    ClaimEntityLink, ClaimEvidence, ClaimRecord, EntityRelationship, ExtractionRun,
    RelationshipEvidence,
};

use crate::MenteResult;

/// Validated derived knowledge from one extraction run.
///
/// The model/provider layer should produce this shape only after preserving the
/// raw episode. Grasp stores the run metadata and all derived artifacts
/// atomically so the derived graph can be inspected or rebuilt later.
#[derive(Debug, Clone)]
pub struct ValidatedExtractionBatch {
    pub run: ExtractionRun,
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
    let relationship_ids: HashSet<&str> = batch
        .relationships
        .iter()
        .map(|relationship| relationship.relationship_id.as_str())
        .collect();

    for claim in &batch.claims {
        validate_claim(claim)?;
    }
    for link in &batch.claim_entities {
        require_known(&claim_ids, &link.claim_id, "claim entity link", "claim_id")?;
        require_non_empty(&link.entity_id, "claim entity link entity_id")?;
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
        validate_relationship(relationship)?;
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

fn validate_claim(claim: &ClaimRecord) -> MenteResult<()> {
    require_non_empty(&claim.claim_id, "claim id")?;
    require_non_empty(&claim.claim_text, "claim text")?;
    require_non_empty(&claim.claim_type, "claim type")?;
    require_non_empty(&claim.status, "claim status")?;
    validate_confidence(claim.confidence, "claim confidence")?;
    validate_time_bounds(claim.valid_from, claim.valid_until, "claim validity")?;
    validate_json_object(&claim.attributes_json, "claim attributes_json")
}

fn validate_relationship(relationship: &EntityRelationship) -> MenteResult<()> {
    require_non_empty(&relationship.relationship_id, "relationship id")?;
    require_non_empty(
        &relationship.source_entity_id,
        "relationship source_entity_id",
    )?;
    require_non_empty(
        &relationship.target_entity_id,
        "relationship target_entity_id",
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
