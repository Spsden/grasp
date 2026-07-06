//! Rust version of the Pratap Synapse memory drift benchmark.
//!
//! Run with:
//! cargo run -p mentedb --example pratap_memory_drift -- --turn-mode mimic-flutter

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use mentedb::prelude::*;
use mentedb::process_turn::{ProcessTurnInput, ProcessTurnResult};
use mentedb_context::{DeltaTracker, ScoredMemory};
use mentedb_embedding::hash_provider::HashEmbeddingProvider;
use mentedb_embedding::http_provider::{HttpEmbeddingConfig, HttpEmbeddingProvider};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEFAULT_FIXTURE: &str = "benchmarks/fixtures/pratap_synapse_memory.json";
const DEFAULT_DB_DIR: &str = "benchmarks/test-db-rust";
const DEFAULT_RESULTS_DIR: &str = "benchmarks/results";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TurnMode {
    MimicFlutter,
    SingleProcess,
}

#[derive(Debug)]
struct Args {
    fixture: PathBuf,
    db_dir: PathBuf,
    output: Option<PathBuf>,
    dry_run: bool,
    turn_mode: TurnMode,
    embedding_provider: String,
    embedding_api_key_env: String,
    embedding_model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    project_context: String,
    memory_bank: Vec<MemoryBankSection>,
    turns: Vec<TurnFixture>,
}

#[derive(Debug, Deserialize)]
struct MemoryBankSection {
    section: String,
    #[serde(default)]
    tags: Vec<String>,
    items: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct TurnFixture {
    id: String,
    phase: Option<String>,
    title: Option<String>,
    question: String,
    #[serde(default)]
    required_context_terms: Vec<String>,
}

#[derive(Debug, Serialize)]
struct TurnRecord {
    kind: &'static str,
    turn_index: usize,
    turn_id: String,
    phase: Option<String>,
    title: Option<String>,
    question: String,
    context_count: usize,
    context_total: usize,
    context_matched: usize,
    context_pass: bool,
    missing_context_terms: Vec<String>,
    context_snippets: Vec<String>,
    pre_stored_ids: Vec<String>,
    post_stored_ids: Vec<String>,
    post_episodic_id: Option<String>,
    post_facts_extracted: usize,
    post_edges_created: u32,
    post_enrichment_pending: bool,
}

#[derive(Debug)]
struct Coverage {
    total: usize,
    matched: usize,
    missing: Vec<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    load_env_file(Path::new(".env"))?;
    init_tracing();

    let args = parse_args()?;
    let fixture = load_fixture(&args.fixture)?;
    if args.dry_run {
        return dry_run(&fixture);
    }

    let output = args
        .output
        .clone()
        .unwrap_or_else(|| default_output_path(&fixture.id));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut db = MenteDb::open(&args.db_dir)?;
    set_embedding_provider(&mut db, &args)?;
    let mut delta_tracker = DeltaTracker::new();

    let seeded_ids = seed_memory_bank(&db, &fixture)?;
    println!(
        "Seeded {} memory bank entries into {}",
        seeded_ids.len(),
        args.db_dir.display()
    );
    println!("Turn mode: {}", format_turn_mode(args.turn_mode));
    println!(
        "Embedding: {} / {}",
        args.embedding_provider,
        args.embedding_model.as_deref().unwrap_or("default")
    );
    println!("Output: {}", output.display());

    let mut writer = BufWriter::new(File::create(&output)?);
    write_json_line(
        &mut writer,
        &json!({
            "kind": "run_start",
            "fixture_id": fixture.id,
            "db_dir": args.db_dir,
            "turn_mode": format_turn_mode(args.turn_mode),
            "seeded_memory_ids": seeded_ids,
            "llm_enabled": false,
            "runner": "rust"
        }),
    )?;

    let mut records = Vec::new();
    for (idx, turn) in fixture.turns.iter().enumerate() {
        let turn_index = idx + 1;
        let pre_result = match args.turn_mode {
            TurnMode::MimicFlutter => process_turn(
                &db,
                &mut delta_tracker,
                &turn.question,
                None,
                turn_index as u64,
                &fixture.project_context,
            )?,
            TurnMode::SingleProcess => search_context(&db, &turn.question, 8)?,
        };

        let retrieved_context = context_text(&pre_result.context);
        let coverage = compute_coverage(&retrieved_context, &turn.required_context_terms);
        let answer = synthetic_answer(&turn.id);

        let post_result = process_turn(
            &db,
            &mut delta_tracker,
            &turn.question,
            Some(answer),
            turn_index as u64,
            &fixture.project_context,
        )?;

        let record = TurnRecord {
            kind: "turn",
            turn_index,
            turn_id: turn.id.clone(),
            phase: turn.phase.clone(),
            title: turn.title.clone(),
            question: turn.question.clone(),
            context_count: pre_result.context.len(),
            context_total: coverage.total,
            context_matched: coverage.matched,
            context_pass: coverage.missing.is_empty(),
            missing_context_terms: coverage.missing,
            context_snippets: pre_result
                .context
                .iter()
                .take(3)
                .map(|item| snippet(&item.memory.content, 220))
                .collect(),
            pre_stored_ids: pre_result
                .stored_ids
                .iter()
                .map(ToString::to_string)
                .collect(),
            post_stored_ids: post_result
                .stored_ids
                .iter()
                .map(ToString::to_string)
                .collect(),
            post_episodic_id: post_result.episodic_id.map(|id| id.to_string()),
            post_facts_extracted: post_result.facts_extracted,
            post_edges_created: post_result.edges_created,
            post_enrichment_pending: post_result.enrichment_pending,
        };

        print_turn_summary(&record);
        write_json_line(&mut writer, &record)?;
        records.push(record);
    }

    let passed = records.iter().filter(|record| record.context_pass).count();
    write_json_line(
        &mut writer,
        &json!({
            "kind": "summary",
            "fixture_id": fixture.id,
            "turns": records.len(),
            "context_passed": passed,
            "context_failed": records.len() - passed,
            "context_pass_rate": if records.is_empty() { 0.0 } else { passed as f64 / records.len() as f64 }
        }),
    )?;
    writer.flush()?;

    println!();
    println!(
        "Context pass rate: {}/{} ({:.1}%)",
        passed,
        records.len(),
        if records.is_empty() {
            0.0
        } else {
            passed as f64 / records.len() as f64 * 100.0
        }
    );
    println!("DB kept at: {}", args.db_dir.display());

    db.close()?;
    if passed == records.len() {
        Ok(())
    } else {
        std::process::exit(1);
    }
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut values = env::args().skip(1).peekable();
    let mut args = Args {
        fixture: PathBuf::from(DEFAULT_FIXTURE),
        db_dir: PathBuf::from(DEFAULT_DB_DIR),
        output: None,
        dry_run: false,
        turn_mode: TurnMode::MimicFlutter,
        embedding_provider: env::var("MENTEDB_EMBEDDING_PROVIDER")
            .unwrap_or_else(|_| "hash".to_string()),
        embedding_api_key_env: "OPENAI_API_KEY".to_string(),
        embedding_model: env::var("MENTEDB_EMBEDDING_MODEL")
            .ok()
            .filter(|value| !value.is_empty()),
    };

    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--fixture" => args.fixture = PathBuf::from(next_value(&mut values, "--fixture")?),
            "--db-dir" => args.db_dir = PathBuf::from(next_value(&mut values, "--db-dir")?),
            "--output" => args.output = Some(PathBuf::from(next_value(&mut values, "--output")?)),
            "--dry-run" => args.dry_run = true,
            "--turn-mode" => {
                args.turn_mode = match next_value(&mut values, "--turn-mode")?.as_str() {
                    "mimic-flutter" => TurnMode::MimicFlutter,
                    "single-process" => TurnMode::SingleProcess,
                    other => return Err(format!("unknown --turn-mode: {other}").into()),
                };
            }
            "--embedding-provider" => {
                args.embedding_provider = next_value(&mut values, "--embedding-provider")?;
            }
            "--embedding-api-key-env" => {
                args.embedding_api_key_env = next_value(&mut values, "--embedding-api-key-env")?;
            }
            "--embedding-model" => {
                let model = next_value(&mut values, "--embedding-model")?;
                args.embedding_model = if model.is_empty() { None } else { Some(model) };
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }

    Ok(args)
}

fn next_value(
    values: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value").into())
}

fn print_help() {
    println!("Run the Rust Pratap/Synapse memory drift benchmark.");
    println!();
    println!("Options:");
    println!("  --fixture <path>                 Defaults to {DEFAULT_FIXTURE}");
    println!("  --db-dir <path>                  Defaults to {DEFAULT_DB_DIR}");
    println!(
        "  --output <path>                  Defaults to benchmarks/results/<fixture>-rust-<timestamp>.jsonl"
    );
    println!("  --turn-mode <mimic-flutter|single-process>");
    println!("  --embedding-provider <hash|ollama|openai|cohere|voyage>");
    println!("  --embedding-model <model>");
    println!("  --embedding-api-key-env <name>   Defaults to OPENAI_API_KEY");
    println!("  --dry-run");
}

fn load_env_file(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Ok(());
    }

    for raw_line in fs::read_to_string(path)?.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        if env::var_os(key.trim()).is_none() {
            let value = value.trim().trim_matches('"').trim_matches('\'');
            unsafe {
                env::set_var(key.trim(), value);
            }
        }
    }

    Ok(())
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            env::var("RUST_LOG")
                .unwrap_or_else(|_| "mentedb=info,mentedb_cognitive=info".to_string()),
        )
        .try_init();
}

fn load_fixture(path: &Path) -> Result<Fixture, Box<dyn std::error::Error>> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn dry_run(fixture: &Fixture) -> Result<(), Box<dyn std::error::Error>> {
    println!("Fixture: {}", fixture.id);
    println!("Project context: {}", fixture.project_context);
    println!("Memory entries: {}", count_memory_items(fixture));
    println!("Turns: {}", fixture.turns.len());
    for (idx, turn) in fixture.turns.iter().enumerate() {
        println!(
            "{:02}. {} -> required context: {}",
            idx + 1,
            turn.id,
            turn.required_context_terms.join(", ")
        );
    }
    Ok(())
}

fn set_embedding_provider(db: &mut MenteDb, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    match args.embedding_provider.as_str() {
        "hash" => db.set_embedder(Box::new(HashEmbeddingProvider::new(384))),
        "ollama" => {
            let model = args
                .embedding_model
                .as_deref()
                .unwrap_or("nomic-embed-text");
            db.set_embedder(Box::new(HttpEmbeddingProvider::new(
                HttpEmbeddingConfig::ollama(model),
            )));
        }
        "openai" => {
            let key = env::var(&args.embedding_api_key_env)
                .map_err(|_| format!("{} is not set", args.embedding_api_key_env))?;
            let model = args
                .embedding_model
                .as_deref()
                .unwrap_or("text-embedding-3-small");
            db.set_embedder(Box::new(HttpEmbeddingProvider::new(
                HttpEmbeddingConfig::openai(key, model),
            )));
        }
        "cohere" => {
            let key = env::var(&args.embedding_api_key_env)
                .map_err(|_| format!("{} is not set", args.embedding_api_key_env))?;
            let model = args
                .embedding_model
                .as_deref()
                .unwrap_or("embed-english-v3.0");
            db.set_embedder(Box::new(HttpEmbeddingProvider::new(
                HttpEmbeddingConfig::cohere(key, model),
            )));
        }
        "voyage" => {
            let key = env::var(&args.embedding_api_key_env)
                .map_err(|_| format!("{} is not set", args.embedding_api_key_env))?;
            let model = args.embedding_model.as_deref().unwrap_or("voyage-2");
            db.set_embedder(Box::new(HttpEmbeddingProvider::new(
                HttpEmbeddingConfig::voyage(key, model),
            )));
        }
        other => return Err(format!("unknown embedding provider: {other}").into()),
    }
    Ok(())
}

fn seed_memory_bank(
    db: &MenteDb,
    fixture: &Fixture,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut stored_ids = Vec::new();
    let agent_id = AgentId::nil();
    let fixture_tag = format!("fixture:{}", fixture.id);

    for section in &fixture.memory_bank {
        let section_tag = format!("section:{}", slug(&section.section));
        let mut base_tags = vec![fixture_tag.clone(), section_tag];
        base_tags.extend(section.tags.iter().cloned());

        for item in &section.items {
            let content = format!("[{}]\n{}", section.section, item);
            let embedding = db.embed_text(&content)?.unwrap_or_else(|| vec![0.0; 384]);
            let mut node: MemoryNode =
                MemoryNode::new(agent_id, MemoryType::Semantic, content, embedding);
            node.tags = base_tags.clone();
            let id = node.id;
            let mut source = MemorySource::new(id, "fixture_memory_bank");
            source.conversation_id = Some(fixture.id.clone());
            source.turn_id = Some(slug(&section.section));
            source.actor_id = Some("benchmark".to_string());
            source.payload_json = json!({
                "section": &section.section,
                "item": item,
                "fixture": &fixture.id,
            })
            .to_string();
            db.store_with_source(node, source)?;
            stored_ids.push(id.to_string());
        }
    }

    Ok(stored_ids)
}

fn process_turn(
    db: &MenteDb,
    delta_tracker: &mut DeltaTracker,
    question: &str,
    assistant_response: Option<String>,
    turn_id: u64,
    project_context: &str,
) -> MenteResult<ProcessTurnResult> {
    let input = ProcessTurnInput {
        user_message: question.to_string(),
        assistant_response,
        turn_id,
        project_context: Some(project_context.to_string()),
        agent_id: None,
    };
    db.process_turn(&input, delta_tracker)
}

fn search_context(db: &MenteDb, query: &str, limit: usize) -> MenteResult<ProcessTurnResult> {
    let embedding = db.embed_text(query)?.unwrap_or_else(|| vec![0.0; 384]);
    let now = now_us();
    let ids = db.recall_hybrid_at(&embedding, Some(query), limit, now, None, None)?;
    let mut context = Vec::new();
    for (id, score) in ids {
        if let Ok(memory) = db.get_memory(id) {
            context.push(ScoredMemory { memory, score });
        }
    }

    Ok(ProcessTurnResult {
        context,
        stored_ids: Vec::new(),
        episodic_id: None,
        pain_warnings: Vec::new(),
        cache_hit: false,
        inference_actions: 0,
        detected_actions: Vec::new(),
        proactive_recalls: Vec::new(),
        correction_id: None,
        sentiment: 0.0,
        phantom_count: 0,
        contradiction_count: 0,
        predicted_topics: Vec::new(),
        facts_extracted: 0,
        edges_created: 0,
        enrichment_pending: false,
        delta_added: Vec::new(),
        delta_removed: Vec::new(),
    })
}

fn context_text(context: &[ScoredMemory]) -> String {
    let mut lines = Vec::new();
    for (idx, item) in context.iter().enumerate() {
        lines.push(format!(
            "{}. [score={:.3}] {}",
            idx + 1,
            item.score,
            item.memory.content.replace('\n', " ")
        ));
    }
    lines.join("\n")
}

fn synthetic_answer(turn_id: &str) -> String {
    format!(
        "Diagnostic placeholder answer for {turn_id}. This run is measuring MenteDB retrieval and write behavior, not LLM answer quality."
    )
}

fn compute_coverage(text: &str, terms: &[String]) -> Coverage {
    let haystack = normalize(text);
    let missing: Vec<String> = terms
        .iter()
        .filter(|term| !haystack.contains(&normalize(term)))
        .cloned()
        .collect();
    Coverage {
        total: terms.len(),
        matched: terms.len() - missing.len(),
        missing,
    }
}

fn normalize(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn slug(value: &str) -> String {
    let mut output = String::new();
    let mut previous_dash = false;
    for ch in value.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            output.push(ch);
            previous_dash = false;
        } else if !previous_dash {
            output.push('-');
            previous_dash = true;
        }
    }
    output.trim_matches('-').to_string()
}

fn snippet(value: &str, max_chars: usize) -> String {
    value.replace('\n', " ").chars().take(max_chars).collect()
}

fn print_turn_summary(record: &TurnRecord) {
    let status = if record.context_pass { "PASS" } else { "FAIL" };
    println!(
        "{:02}. {}: {} ({}/{} context terms)",
        record.turn_index, record.turn_id, status, record.context_matched, record.context_total
    );
    if !record.missing_context_terms.is_empty() {
        println!("    missing: {}", record.missing_context_terms.join(", "));
    }
    for snippet in record.context_snippets.iter().take(2) {
        println!("    context: {snippet}");
    }
}

fn write_json_line<T: Serialize>(
    writer: &mut BufWriter<File>,
    value: &T,
) -> Result<(), Box<dyn std::error::Error>> {
    serde_json::to_writer(&mut *writer, value)?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn default_output_path(fixture_id: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    PathBuf::from(DEFAULT_RESULTS_DIR).join(format!("{fixture_id}-rust-{timestamp}.jsonl"))
}

fn count_memory_items(fixture: &Fixture) -> usize {
    fixture
        .memory_bank
        .iter()
        .map(|section| section.items.len())
        .sum()
}

fn format_turn_mode(mode: TurnMode) -> &'static str {
    match mode {
        TurnMode::MimicFlutter => "mimic-flutter",
        TurnMode::SingleProcess => "single-process",
    }
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}
