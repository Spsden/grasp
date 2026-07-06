//! Inspect a Grasp memory store without writing new memories.
//!
//! Run with:
//! cargo run -p mentedb --example inspect_memory -- --db-dir ./grasp-data summary

use std::collections::HashSet;
use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::str::FromStr;

use mentedb::core::types::MemoryId;
use mentedb::prelude::*;
use mentedb::{
    ClaimRecord, EntityRecord, EntityRelationship, MemoryEntityLink, MemoryLifecycleEvent,
    MemoryOperation, MemorySource, MenteDb, RetrievalTrace, RetrievalTraceHit,
};

const DEFAULT_DB_DIR: &str = "./grasp-data";
const DEFAULT_LIMIT: usize = 10;
const DEFAULT_TRACE_HITS: usize = 5;

#[derive(Debug)]
struct InspectArgs {
    db_dir: PathBuf,
    command: Command,
}

#[derive(Debug)]
enum Command {
    Help,
    Summary { limit: usize },
    Operations { limit: usize },
    Traces { limit: usize, hits: usize },
    Entities { limit: usize },
    Entity { query: String, limit: usize },
    Memory { id: MemoryId },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        eprintln!();
        print_usage();
        process::exit(1);
    }
}

fn run() -> MenteResult<()> {
    let args = parse_args(env::args().skip(1).collect())?;
    if matches!(args.command, Command::Help) {
        print_usage();
        return Ok(());
    }

    let sqlite_path = args.db_dir.join("memory.sqlite");
    if !sqlite_path.exists() {
        eprintln!(
            "warning: {} does not exist yet, opening will create an empty store",
            sqlite_path.display()
        );
    }

    let db = MenteDb::open(&args.db_dir)?;
    match args.command {
        Command::Help => unreachable!(),
        Command::Summary { limit } => print_summary(&db, &args.db_dir, limit)?,
        Command::Operations { limit } => print_operations(&db, limit)?,
        Command::Traces { limit, hits } => print_traces(&db, limit, hits)?,
        Command::Entities { limit } => print_entities(&db, limit)?,
        Command::Entity { query, limit } => print_entity(&db, &query, limit)?,
        Command::Memory { id } => print_memory(&db, id)?,
    }
    Ok(())
}

fn parse_args(raw: Vec<String>) -> MenteResult<InspectArgs> {
    let mut db_dir = PathBuf::from(DEFAULT_DB_DIR);
    let mut command_parts = Vec::new();
    let mut iter = raw.into_iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--db-dir" | "--path" => {
                let value = iter
                    .next()
                    .ok_or_else(|| invalid_input("--db-dir requires a path"))?;
                db_dir = PathBuf::from(value);
            }
            "-h" | "--help" => {
                return Ok(InspectArgs {
                    db_dir,
                    command: Command::Help,
                });
            }
            _ => {
                command_parts.push(arg);
                command_parts.extend(iter);
                break;
            }
        }
    }

    Ok(InspectArgs {
        db_dir,
        command: parse_command(command_parts)?,
    })
}

fn parse_command(mut parts: Vec<String>) -> MenteResult<Command> {
    if parts.is_empty() {
        return Ok(Command::Summary {
            limit: DEFAULT_LIMIT,
        });
    }

    let command = parts.remove(0);
    match command.as_str() {
        "help" => Ok(Command::Help),
        "summary" => {
            let limit = take_usize_option(&mut parts, "--limit", DEFAULT_LIMIT)?;
            reject_extra(&parts)?;
            Ok(Command::Summary { limit })
        }
        "operations" | "ops" => {
            let limit = take_usize_option(&mut parts, "--limit", DEFAULT_LIMIT)?;
            reject_extra(&parts)?;
            Ok(Command::Operations { limit })
        }
        "traces" => {
            let limit = take_usize_option(&mut parts, "--limit", DEFAULT_LIMIT)?;
            let hits = take_usize_option(&mut parts, "--hits", DEFAULT_TRACE_HITS)?;
            reject_extra(&parts)?;
            Ok(Command::Traces { limit, hits })
        }
        "entities" => {
            let limit = take_usize_option(&mut parts, "--limit", DEFAULT_LIMIT)?;
            reject_extra(&parts)?;
            Ok(Command::Entities { limit })
        }
        "entity" => {
            let limit = take_usize_option(&mut parts, "--limit", DEFAULT_LIMIT)?;
            let query = take_one_positional(&parts, "entity requires an id, alias, or canonical")?;
            Ok(Command::Entity { query, limit })
        }
        "memory" => {
            let raw_id = take_one_positional(&parts, "memory requires a memory id")?;
            let id = MemoryId::from_str(&raw_id)
                .map_err(|err| invalid_input(format!("invalid memory id {raw_id}: {err}")))?;
            Ok(Command::Memory { id })
        }
        _ => Err(invalid_input(format!("unknown command {command}"))),
    }
}

fn take_usize_option(parts: &mut Vec<String>, flag: &str, default: usize) -> MenteResult<usize> {
    let Some(pos) = parts.iter().position(|part| part == flag) else {
        return Ok(default);
    };
    parts.remove(pos);
    if pos >= parts.len() {
        return Err(invalid_input(format!("{flag} requires a value")));
    }
    let raw = parts.remove(pos);
    raw.parse::<usize>()
        .map_err(|err| invalid_input(format!("invalid value for {flag}: {err}")))
}

fn take_one_positional(parts: &[String], message: &str) -> MenteResult<String> {
    if parts.len() == 1 {
        Ok(parts[0].clone())
    } else {
        Err(invalid_input(message))
    }
}

fn reject_extra(parts: &[String]) -> MenteResult<()> {
    if parts.is_empty() {
        Ok(())
    } else {
        Err(invalid_input(format!(
            "unexpected arguments: {}",
            parts.join(" ")
        )))
    }
}

fn invalid_input(message: impl Into<String>) -> MenteError {
    MenteError::InvalidInput(message.into())
}

fn print_usage() {
    println!(
        "usage:
  cargo run -p mentedb --example inspect_memory -- --db-dir <dir> summary [--limit N]
  cargo run -p mentedb --example inspect_memory -- --db-dir <dir> operations [--limit N]
  cargo run -p mentedb --example inspect_memory -- --db-dir <dir> traces [--limit N] [--hits N]
  cargo run -p mentedb --example inspect_memory -- --db-dir <dir> entities [--limit N]
  cargo run -p mentedb --example inspect_memory -- --db-dir <dir> entity <id-or-alias> [--limit N]
  cargo run -p mentedb --example inspect_memory -- --db-dir <dir> memory <memory-id>"
    );
}

fn print_summary(db: &MenteDb, db_dir: &Path, limit: usize) -> MenteResult<()> {
    let entities = db.list_entities(limit)?;
    let traces = db.recent_retrieval_traces(limit)?;
    let operations = db.recent_memory_operations(limit)?;
    let extraction_runs = db.recent_extraction_runs(limit)?;
    let lifecycle_events = db.recent_lifecycle_events(limit)?;

    println!("store: {}", db_dir.display());
    println!("memories: {}", db.memory_count());
    println!("sample_limit: {limit}");
    println!("entities_sampled: {}", entities.len());
    println!("retrieval_traces_sampled: {}", traces.len());
    println!("operations_sampled: {}", operations.len());
    println!("extraction_runs_sampled: {}", extraction_runs.len());
    println!("lifecycle_events_sampled: {}", lifecycle_events.len());
    Ok(())
}

fn print_operations(db: &MenteDb, limit: usize) -> MenteResult<()> {
    for op in db.recent_memory_operations(limit)? {
        print_operation(&op);
    }
    Ok(())
}

fn print_traces(db: &MenteDb, limit: usize, hit_limit: usize) -> MenteResult<()> {
    for trace in db.recent_retrieval_traces(limit)? {
        print_trace(&trace);
        if hit_limit == 0 {
            continue;
        }
        for hit in db
            .retrieval_trace_hits(&trace.trace_id)?
            .into_iter()
            .take(hit_limit)
        {
            print_trace_hit(db, &hit);
        }
    }
    Ok(())
}

fn print_entities(db: &MenteDb, limit: usize) -> MenteResult<()> {
    for entity in db.list_entities(limit)? {
        let aliases = db.entity_aliases(&entity.entity_id)?;
        let memory_count = db.memories_for_entity(&entity.entity_id)?.len();
        let claim_count = db.claims_for_entity(&entity.entity_id)?.len();
        let relationship_count = db.relationships_for_entity(&entity.entity_id)?.len();
        println!(
            "{} [{}] type={} confidence={:.3} memories={} claims={} relationships={} aliases={}",
            entity.entity_id,
            entity.canonical,
            entity.entity_type,
            entity.confidence,
            memory_count,
            claim_count,
            relationship_count,
            aliases
                .iter()
                .map(|alias| alias.alias.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}

fn print_entity(db: &MenteDb, query: &str, limit: usize) -> MenteResult<()> {
    let mut seen = HashSet::new();
    let mut matches = Vec::new();

    if let Some(entity) = db.get_entity(query)? {
        seen.insert(entity.entity_id.clone());
        matches.push(entity);
    }
    for entity in db.entities_by_alias(query)? {
        if seen.insert(entity.entity_id.clone()) {
            matches.push(entity);
        }
    }
    for entity in db.list_entities(1000)? {
        if entity.canonical.eq_ignore_ascii_case(query) && seen.insert(entity.entity_id.clone()) {
            matches.push(entity);
        }
    }

    if matches.is_empty() {
        println!("no entity matched {query}");
        return Ok(());
    }

    for entity in matches {
        print_entity_detail(db, &entity, limit)?;
    }
    Ok(())
}

fn print_entity_detail(db: &MenteDb, entity: &EntityRecord, limit: usize) -> MenteResult<()> {
    println!(
        "entity {} canonical={} type={} confidence={:.3}",
        entity.entity_id, entity.canonical, entity.entity_type, entity.confidence
    );
    println!("  attributes: {}", compact(&entity.attributes_json, 180));

    let aliases = db.entity_aliases(&entity.entity_id)?;
    for alias in aliases.iter().take(limit) {
        println!(
            "  alias {} source={} confidence={:.3}",
            alias.alias,
            alias.source.as_deref().unwrap_or(""),
            alias.confidence
        );
    }

    for link in db
        .memories_for_entity(&entity.entity_id)?
        .into_iter()
        .take(limit)
    {
        print_memory_link("  memory", &link);
    }

    for claim in db
        .claims_for_entity(&entity.entity_id)?
        .into_iter()
        .take(limit)
    {
        print_claim("  claim", &claim);
    }

    for relationship in db
        .relationships_for_entity(&entity.entity_id)?
        .into_iter()
        .take(limit)
    {
        print_relationship("  relationship", &relationship);
    }
    Ok(())
}

fn print_memory(db: &MenteDb, id: MemoryId) -> MenteResult<()> {
    let node = db.get_memory(id)?;
    println!("memory {}", node.id);
    println!("  type: {:?}", node.memory_type);
    println!("  agent_id: {}", node.agent_id);
    println!("  space_id: {}", node.space_id);
    println!("  embedding_dim: {}", node.embedding.len());
    println!("  salience: {:.3}", node.salience);
    println!("  confidence: {:.3}", node.confidence);
    println!("  created_at: {}", node.created_at);
    println!("  accessed_at: {}", node.accessed_at);
    println!("  access_count: {}", node.access_count);
    println!("  valid_from: {}", optional_ts(node.valid_from));
    println!("  valid_until: {}", optional_ts(node.valid_until));
    println!("  tags: {}", node.tags.join(", "));
    println!("  content: {}", compact(&node.content, 240));

    for source in db.memory_sources(id)? {
        print_source("  source", &source);
    }
    for link in db.memory_entity_links(id)? {
        print_memory_link("  entity", &link);
    }
    for claim in db.claims_for_memory(id)? {
        print_claim("  claim", &claim);
    }
    for event in db.lifecycle_events_for_memory(id, 20)? {
        print_lifecycle_event("  lifecycle", &event);
    }
    for op in db.memory_operations_for(id, 20)? {
        print_operation_with_prefix("  op", &op);
    }
    Ok(())
}

fn print_trace(trace: &RetrievalTrace) {
    println!(
        "trace {} query={} k={} fetch_k={} candidates={} results={} created_at={}",
        trace.trace_id,
        trace.query_text.as_deref().unwrap_or(""),
        trace.k,
        trace.fetch_k,
        trace.candidate_count,
        trace.result_count,
        trace.created_at
    );
    println!("  stage: {}", compact(&trace.stage_json, 220));
    println!("  config: {}", compact(&trace.config_json, 220));
}

fn print_trace_hit(db: &MenteDb, hit: &RetrievalTraceHit) {
    let content = db
        .get_memory(hit.memory_id)
        .map(|node| compact(&node.content, 120))
        .unwrap_or_else(|_| "<missing memory>".to_string());
    println!(
        "  hit rank={} memory={} score={:.4} source={} vector_rank={} bm25_rank={} salience={} content={}",
        hit.rank,
        hit.memory_id,
        hit.score,
        hit.source,
        optional_usize(hit.vector_rank),
        optional_usize(hit.bm25_rank),
        optional_f32(hit.salience),
        content
    );
    println!("    why: {}", compact(&hit.explanation_json, 220));
}

fn print_operation(op: &MemoryOperation) {
    print_operation_with_prefix("operation", op);
}

fn print_operation_with_prefix(prefix: &str, op: &MemoryOperation) {
    println!(
        "{} {} type={} memory={} source={} target={} created_at={} payload={}",
        prefix,
        op.operation_id,
        op.operation_type,
        op.memory_id.map(|id| id.to_string()).unwrap_or_default(),
        op.source.map(|id| id.to_string()).unwrap_or_default(),
        op.target.map(|id| id.to_string()).unwrap_or_default(),
        op.created_at,
        compact(&op.payload_json, 180)
    );
}

fn print_source(prefix: &str, source: &MemorySource) {
    println!(
        "{} {} type={} conversation={} turn={} actor={} extractor={} created_at={} payload={}",
        prefix,
        source.source_id,
        source.source_type,
        source.conversation_id.as_deref().unwrap_or(""),
        source.turn_id.as_deref().unwrap_or(""),
        source.actor_id.as_deref().unwrap_or(""),
        source.extractor.as_deref().unwrap_or(""),
        source.created_at,
        compact(&source.payload_json, 160)
    );
}

fn print_lifecycle_event(prefix: &str, event: &MemoryLifecycleEvent) {
    println!(
        "{} {} memory={} type={} reason={} policy={} created_at={} payload={}",
        prefix,
        event.event_id,
        event.memory_id,
        event.event_type,
        event.reason.as_deref().unwrap_or(""),
        event.policy.as_deref().unwrap_or(""),
        event.created_at,
        compact(&event.payload_json, 160)
    );
}

fn print_memory_link(prefix: &str, link: &MemoryEntityLink) {
    println!(
        "{} memory={} entity={} role={} confidence={:.3} evidence={}",
        prefix,
        link.memory_id,
        link.entity_id,
        link.role.as_deref().unwrap_or(""),
        link.confidence,
        link.evidence.as_deref().unwrap_or("")
    );
}

fn print_claim(prefix: &str, claim: &ClaimRecord) {
    println!(
        "{} {} type={} status={} confidence={:.3} predicate={} text={}",
        prefix,
        claim.claim_id,
        claim.claim_type,
        claim.status,
        claim.confidence,
        claim.predicate.as_deref().unwrap_or(""),
        compact(&claim.claim_text, 180)
    );
}

fn print_relationship(prefix: &str, relationship: &EntityRelationship) {
    println!(
        "{} {} {} {} type={} status={} confidence={:.3}",
        prefix,
        relationship.relationship_id,
        relationship.source_entity_id,
        relationship.target_entity_id,
        relationship.relation_type,
        relationship.status,
        relationship.confidence
    );
}

fn compact(text: &str, max_chars: usize) -> String {
    let one_line = text.replace('\n', " ");
    if one_line.chars().count() <= max_chars {
        return one_line;
    }
    let take = max_chars.saturating_sub(3);
    let mut out = one_line.chars().take(take).collect::<String>();
    out.push_str("...");
    out
}

fn optional_ts(value: Option<u64>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_usize(value: Option<usize>) -> String {
    value.map(|value| value.to_string()).unwrap_or_default()
}

fn optional_f32(value: Option<f32>) -> String {
    value.map(|value| format!("{value:.3}")).unwrap_or_default()
}
