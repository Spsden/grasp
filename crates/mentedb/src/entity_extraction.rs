use std::collections::HashMap;

use mentedb_core::MemoryNode;

/// Tunable settings for deterministic entity surface linking.
///
/// This is deliberately not a semantic natural-language extractor. It only
/// indexes explicit entity tags, structured fields, and exact aliases already
/// present in durable storage. LLM extraction should own semantic entities,
/// claims, relationships, corrections, and temporal interpretation.
#[derive(Debug, Clone)]
pub struct EntityExtractionConfig {
    /// Whether write-time surface linking is enabled.
    pub enabled: bool,
    /// Whether entity-linked memories are included as an additive recall signal.
    pub recall_enabled: bool,
    /// Maximum explicit entities linked for a single memory.
    pub max_entities_per_memory: usize,
    /// Maximum aliases written for a single explicit entity.
    pub max_aliases_per_entity: usize,
    /// Minimum explicit field value length accepted.
    pub min_phrase_chars: usize,
    /// Maximum explicit field value length accepted.
    pub max_phrase_chars: usize,
    /// Maximum words used when resolving query text against existing aliases.
    pub max_alias_words: usize,
    /// Confidence for entities explicitly tagged with `entity:`.
    pub tag_confidence: f32,
    /// Confidence for structured fields such as `Name: Pratap`.
    pub structured_confidence: f32,
    /// Minimum confidence required before writing an entity link.
    pub min_confidence: f32,
    /// Score added during recall for each linked entity match.
    pub recall_boost: f32,
    /// Maximum linked memories considered per resolved entity during recall.
    pub recall_fetch_limit: usize,
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            recall_enabled: true,
            max_entities_per_memory: 16,
            max_aliases_per_entity: 8,
            min_phrase_chars: 2,
            max_phrase_chars: 80,
            max_alias_words: 5,
            tag_confidence: 0.98,
            structured_confidence: 0.95,
            min_confidence: 0.55,
            recall_boost: 1.0,
            recall_fetch_limit: 32,
        }
    }
}

/// An explicit entity mention found in structured input or tags.
#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedEntity {
    /// Canonical display name.
    pub canonical: String,
    /// Stable type name, for example `person`, `project`, or `technology`.
    pub entity_type: String,
    /// Alternate spellings or normalized lookup keys.
    pub aliases: Vec<String>,
    /// Mention role within the source memory.
    pub role: Option<String>,
    /// Linker confidence in this explicit mention.
    pub confidence: f32,
    /// Small evidence span from the source text or tag.
    pub evidence: Option<String>,
}

/// Deterministic surface linker used by the hot path and tests.
#[derive(Debug, Clone)]
pub struct RuleBasedEntityExtractor {
    config: EntityExtractionConfig,
}

impl RuleBasedEntityExtractor {
    pub fn new(config: EntityExtractionConfig) -> Self {
        Self { config }
    }

    pub fn extract_memory(&self, memory: &MemoryNode) -> Vec<ExtractedEntity> {
        self.extract_text(&memory.content, &memory.tags)
    }

    pub fn extract_query(&self, _query: &str) -> Vec<ExtractedEntity> {
        Vec::new()
    }

    pub fn extract_text(&self, text: &str, tags: &[String]) -> Vec<ExtractedEntity> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut entities = EntityAccumulator::new(&self.config);
        self.extract_tag_entities(tags, &mut entities);
        self.extract_structured_fields(text, &mut entities);
        entities.finish()
    }

    fn extract_tag_entities(&self, tags: &[String], entities: &mut EntityAccumulator<'_>) {
        let tagged_type = tags
            .iter()
            .find_map(|tag| tag.strip_prefix("entity_type:"))
            .map(normalize_type);

        for tag in tags {
            let Some(raw) = tag.strip_prefix("entity:") else {
                continue;
            };
            self.add_explicit(
                raw,
                tagged_type.as_deref().unwrap_or("concept"),
                Some("mentioned"),
                self.config.tag_confidence,
                Some(raw.trim().to_string()),
                entities,
            );
        }
    }

    fn extract_structured_fields(&self, text: &str, entities: &mut EntityAccumulator<'_>) {
        for line in text.lines() {
            let clean = trim_context_prefix(line).trim();

            if let Some(value) = field_value(clean, "Name") {
                let value = first_structured_value(value);
                self.add_explicit(
                    value,
                    "person",
                    Some("subject"),
                    self.config.structured_confidence,
                    Some(value.trim().trim_end_matches('.').to_string()),
                    entities,
                );
            }

            if let Some(project) = clean
                .strip_prefix("Project ")
                .map(take_until_boundary)
                .filter(|value| !value.trim().is_empty())
            {
                self.add_explicit(
                    project,
                    "project",
                    Some("subject"),
                    self.config.structured_confidence,
                    Some(project.trim().to_string()),
                    entities,
                );
            }

            for key in ["Location", "Base location"] {
                if let Some(value) = field_value(clean, key) {
                    for part in split_structured_list(value) {
                        if part.to_lowercase().starts_with("near ") {
                            continue;
                        }
                        self.add_explicit(
                            &part,
                            "place",
                            Some("location"),
                            self.config.structured_confidence,
                            Some(part.clone()),
                            entities,
                        );
                    }
                }
            }

            for key in ["Organization", "Company", "Employer"] {
                if let Some(value) = field_value(clean, key) {
                    for part in split_structured_list(value) {
                        self.add_explicit(
                            &part,
                            "organization",
                            Some("affiliation"),
                            self.config.structured_confidence,
                            Some(part.clone()),
                            entities,
                        );
                    }
                }
            }

            for key in ["Technology", "Technologies", "Stack", "Core stack"] {
                if let Some(value) = field_value(clean, key) {
                    for part in split_structured_list(value) {
                        self.add_explicit(
                            &part,
                            "technology",
                            Some("tool"),
                            self.config.structured_confidence,
                            Some(part.clone()),
                            entities,
                        );
                    }
                }
            }
        }
    }

    fn add_explicit(
        &self,
        raw: &str,
        entity_type: &str,
        role: Option<&str>,
        confidence: f32,
        evidence: Option<String>,
        entities: &mut EntityAccumulator<'_>,
    ) {
        let Some(canonical) = canonicalize_phrase(raw) else {
            return;
        };
        entities.add(ExtractedEntity {
            canonical: canonical.clone(),
            entity_type: normalize_type(entity_type),
            aliases: aliases_for(raw, &canonical, self.config.max_aliases_per_entity),
            role: role.map(ToOwned::to_owned),
            confidence,
            evidence,
        });
    }
}

struct EntityAccumulator<'a> {
    config: &'a EntityExtractionConfig,
    by_key: HashMap<String, ExtractedEntity>,
}

impl<'a> EntityAccumulator<'a> {
    fn new(config: &'a EntityExtractionConfig) -> Self {
        Self {
            config,
            by_key: HashMap::new(),
        }
    }

    fn add(&mut self, entity: ExtractedEntity) {
        if entity.confidence < self.config.min_confidence {
            return;
        }
        if entity.canonical.len() < self.config.min_phrase_chars
            || entity.canonical.len() > self.config.max_phrase_chars
        {
            return;
        }

        let key = format!(
            "{}:{}",
            entity.entity_type.to_lowercase(),
            entity.canonical.to_lowercase()
        );
        match self.by_key.get_mut(&key) {
            Some(existing) => {
                if entity.confidence > existing.confidence {
                    existing.confidence = entity.confidence;
                    existing.role = entity.role.clone().or_else(|| existing.role.clone());
                    if existing.evidence.is_none() {
                        existing.evidence = entity.evidence.clone();
                    }
                }
                for alias in entity.aliases {
                    if existing.aliases.len() < self.config.max_aliases_per_entity
                        && !existing.aliases.iter().any(|current| current == &alias)
                    {
                        existing.aliases.push(alias);
                    }
                }
            }
            None => {
                if self.by_key.len() < self.config.max_entities_per_memory {
                    self.by_key.insert(key, entity);
                }
            }
        }
    }

    fn finish(self) -> Vec<ExtractedEntity> {
        let limit = self.config.max_entities_per_memory;
        let mut out: Vec<ExtractedEntity> = self.by_key.into_values().collect();
        out.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.canonical.cmp(&b.canonical))
        });
        out.truncate(limit);
        out
    }
}

fn field_value<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    let (head, value) = line.split_once(':')?;
    if head.trim().eq_ignore_ascii_case(field) {
        Some(value.trim())
    } else {
        None
    }
}

fn first_structured_value(value: &str) -> &str {
    value
        .split_once(". ")
        .map(|(head, _)| head)
        .unwrap_or(value)
        .trim_end_matches('.')
        .trim()
}

fn split_structured_list(value: &str) -> Vec<String> {
    let normalized = value
        .replace(" with ", ",")
        .replace(" and ", ",")
        .replace(" plus ", ",");
    normalized
        .split(',')
        .filter_map(|part| {
            let cleaned = part
                .trim()
                .trim_start_matches("and ")
                .trim_start_matches("with ")
                .trim_end_matches('.')
                .trim();
            if cleaned.is_empty() {
                None
            } else {
                Some(cleaned.to_string())
            }
        })
        .collect()
}

fn take_until_boundary(raw: &str) -> &str {
    raw.split_once(" is ")
        .map(|(head, _)| head)
        .or_else(|| raw.split_once(" uses ").map(|(head, _)| head))
        .or_else(|| raw.split_once(" includes ").map(|(head, _)| head))
        .unwrap_or(raw)
}

fn trim_section_prefix(line: &str) -> &str {
    let trimmed = line.trim();
    if let Some(rest) = trimmed.strip_prefix('[')
        && let Some((_, after)) = rest.split_once(']')
    {
        return after.trim();
    }
    trimmed
}

fn trim_context_prefix(line: &str) -> &str {
    let without_section = trim_section_prefix(line);
    for prefix in ["User:", "Assistant:", "System:"] {
        if let Some(rest) = without_section.strip_prefix(prefix) {
            return rest.trim();
        }
    }
    without_section
}

fn canonicalize_phrase(raw: &str) -> Option<String> {
    let cleaned = raw
        .trim()
        .trim_matches(|ch: char| matches!(ch, ',' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']'))
        .trim_end_matches('.')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let cleaned = cleaned
        .strip_prefix("Project ")
        .or_else(|| cleaned.strip_prefix("Name "))
        .unwrap_or(&cleaned)
        .to_string();
    if cleaned.is_empty() || !cleaned.chars().any(|ch| ch.is_ascii_alphabetic()) {
        return None;
    }

    if cleaned.chars().any(|ch| ch.is_ascii_uppercase()) {
        return Some(cleaned);
    }

    Some(
        cleaned
            .split_whitespace()
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    Some(first) => {
                        let mut out = first.to_ascii_uppercase().to_string();
                        out.push_str(chars.as_str());
                        out
                    }
                    None => String::new(),
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
    )
}

fn aliases_for(raw: &str, canonical: &str, max_aliases: usize) -> Vec<String> {
    let mut aliases = Vec::new();
    for alias in [
        raw.trim().trim_end_matches('.').to_string(),
        canonical.to_string(),
        canonical.to_lowercase(),
    ] {
        if aliases.len() >= max_aliases {
            break;
        }
        if !alias.is_empty() && !aliases.iter().any(|current| current == &alias) {
            aliases.push(alias);
        }
    }
    aliases
}

fn normalize_type(raw: &str) -> String {
    raw.trim().to_lowercase().replace(' ', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::memory::MemoryType;
    use mentedb_core::types::AgentId;

    #[test]
    fn links_only_explicit_structured_entities() {
        let extractor = RuleBasedEntityExtractor::new(EntityExtractionConfig::default());
        let mut node = MemoryNode::new(
            AgentId::nil(),
            MemoryType::Semantic,
            "Name: Pratap.\nProject Synapse is a Flutter app with Google Workspace OAuth."
                .to_string(),
            vec![1.0, 0.0],
        );
        node.tags = vec!["profile".to_string(), "projects".to_string()];

        let entities = extractor.extract_memory(&node);
        assert!(
            entities
                .iter()
                .any(|entity| entity.canonical == "Pratap" && entity.entity_type == "person")
        );
        assert!(
            entities
                .iter()
                .any(|entity| entity.canonical == "Synapse" && entity.entity_type == "project")
        );
        assert!(!entities.iter().any(|entity| {
            entity.canonical == "Google Workspace OAuth" && entity.entity_type == "technology"
        }));
    }

    #[test]
    fn links_explicit_tags_and_structured_stack_fields() {
        let extractor = RuleBasedEntityExtractor::new(EntityExtractionConfig::default());
        let entities = extractor.extract_text(
            "Core stack: Flutter with Dart, TypeScript, Supabase, and PostgreSQL.",
            &[
                "entity:synapse".to_string(),
                "entity_type:project".to_string(),
            ],
        );
        assert!(
            entities
                .iter()
                .any(|entity| entity.canonical == "Synapse" && entity.entity_type == "project")
        );
        assert!(
            entities
                .iter()
                .any(|entity| entity.canonical == "Flutter" && entity.entity_type == "technology")
        );
        assert!(
            entities.iter().any(
                |entity| entity.canonical == "PostgreSQL" && entity.entity_type == "technology"
            )
        );
    }
}
