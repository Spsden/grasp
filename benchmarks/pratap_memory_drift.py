"""Pratap Synapse memory drift benchmark.

This runner loads an editable JSON fixture, seeds a memory bank, and then
executes a chained set of questions through MenteDB. By default it mimics the
Flutter demo turn flow: call process_turn before the model to retrieve context,
then call process_turn again after the model answer to commit the completed
turn.

The first diagnostic layer is retrieval quality. If the required facts are not
present in the returned context, the LLM cannot reliably answer with memory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "benchmarks" / "fixtures" / "pratap_synapse_memory.json"
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
DEFAULT_DB_DIR = REPO_ROOT / "benchmarks" / "test-db"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


def load_env_file(path: Path = DEFAULT_ENV_FILE) -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Coverage:
    total: int
    matched: int
    missing: list[str]

    @property
    def passed(self) -> bool:
        return not self.missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Pratap/Synapse memory drift benchmark."
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Path to the JSON fixture.",
    )
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=None,
        help="MenteDB data directory. Defaults to benchmarks/test-db.",
    )
    parser.add_argument(
        "--keep-db",
        action="store_true",
        help="Kept for compatibility. The default benchmark DB is already persistent.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the fixture without importing or running MenteDB.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL output path. Defaults to benchmarks/results/...",
    )
    parser.add_argument(
        "--turn-mode",
        choices=("mimic-flutter", "single-process"),
        default="mimic-flutter",
        help=(
            "mimic-flutter calls process_turn before and after the answer. "
            "single-process uses search_text for pre-answer context and calls "
            "process_turn only once after the answer."
        ),
    )
    parser.add_argument(
        "--embedding-provider",
        default=os.environ.get("MENTEDB_EMBEDDING_PROVIDER"),
        help="Optional MenteDB embedding provider, for example openai.",
    )
    parser.add_argument(
        "--embedding-api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the embedding provider API key.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("MENTEDB_EMBEDDING_MODEL"),
        help="Optional embedding model name.",
    )
    parser.add_argument(
        "--llm-endpoint",
        default=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL"),
        help=(
            "Optional OpenAI-compatible base URL or /chat/completions URL. "
            "If omitted, the benchmark runs retrieval-only diagnostics."
        ),
    )
    parser.add_argument(
        "--llm-api-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the chat API key.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        help="OpenAI-compatible chat model name.",
    )
    parser.add_argument(
        "--compare-without-memory",
        action="store_true",
        help="Also call the LLM without memory context for each turn.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=900,
        help="Maximum tokens for each optional LLM answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for optional LLM calls.",
    )
    return parser.parse_args()


def load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)
    required = {"id", "project_context", "memory_bank", "turns"}
    missing = sorted(required - fixture.keys())
    if missing:
        raise ValueError(f"fixture missing required keys: {', '.join(missing)}")
    return fixture


def normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower()).strip()


def slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "section"


def compute_coverage(text: str, terms: list[str]) -> Coverage:
    haystack = normalize(text)
    missing = [term for term in terms if normalize(term) not in haystack]
    return Coverage(total=len(terms), matched=len(terms) - len(missing), missing=missing)


def import_mentedb() -> Any:
    try:
        from mentedb import MenteDB

        return MenteDB
    except ImportError as exc:
        raise SystemExit(
            "The mentedb Python package is not installed.\n"
            "Build it from this repo first:\n"
            "  cd sdks/python && maturin develop && cd ../..\n"
            f"Import error: {exc}"
        ) from exc


def open_db(args: argparse.Namespace, db_dir: Path) -> Any:
    MenteDB = import_mentedb()
    embedding_key = os.environ.get(args.embedding_api_key_env)
    return MenteDB(
        str(db_dir),
        embedding_provider=args.embedding_provider,
        embedding_api_key=embedding_key,
        embedding_model=args.embedding_model,
    )


def native_db(db: Any) -> Any:
    return getattr(db, "_db", db)


def seed_memory_bank(db: Any, fixture: dict[str, Any]) -> list[str]:
    stored_ids: list[str] = []
    fixture_tag = f"fixture:{fixture['id']}"
    for section in fixture["memory_bank"]:
        section_name = section["section"]
        section_tag = f"section:{slug(section_name)}"
        base_tags = [fixture_tag, section_tag, *section.get("tags", [])]
        for item in section["items"]:
            content = f"[{section_name}]\n{item}"
            stored_id = db.store(content, memory_type="semantic", tags=base_tags)
            stored_ids.append(str(stored_id))
    return stored_ids


def process_turn(
    db: Any,
    question: str,
    assistant_response: str | None,
    turn_id: int,
    project_context: str,
) -> dict[str, Any]:
    result = native_db(db).process_turn(
        question,
        assistant_response,
        turn_id,
        project_context,
        None,
    )
    return dict(result)


def search_context(db: Any, query: str, limit: int) -> dict[str, Any]:
    context: list[dict[str, Any]] = []
    for result in db.search_text(query, k=limit):
        memory_id = getattr(result, "id", None) or result.get("id")
        score = getattr(result, "score", None)
        if score is None and isinstance(result, dict):
            score = result.get("score")
        memory = db.get_memory(str(memory_id))
        if isinstance(memory, dict):
            content = str(memory.get("content", ""))
        else:
            content = str(getattr(memory, "content", ""))
        context.append({"id": str(memory_id), "content": content, "score": score})
    return {"context": context, "stored_ids": [], "cache_hit": False}


def context_text(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for index, item in enumerate(result.get("context") or [], start=1):
        content = str(item.get("content", "")).replace("\n", " ")
        score = item.get("score")
        if score is None:
            lines.append(f"{index}. {content}")
        else:
            lines.append(f"{index}. [score={float(score):.3f}] {content}")
    return "\n".join(lines)


def synthetic_answer(turn: dict[str, Any]) -> str:
    return (
        f"Diagnostic placeholder answer for {turn['id']}. "
        "This run is measuring MenteDB retrieval and write behavior, not LLM answer quality."
    )


def resolve_chat_completions_url(endpoint: str) -> str:
    value = endpoint.rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return f"{value}/chat/completions"
    return f"{value}/v1/chat/completions"


def chat_completion(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str,
    memory_context: str | None,
    question: str,
    max_tokens: int,
    temperature: float,
) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if memory_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Relevant MenteDB memories follow. Use them when relevant. "
                    "If the memories conflict with the current user message, prefer the current message.\n\n"
                    f"{memory_context}"
                ),
            }
        )
    messages.append({"role": "user", "content": question})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        resolve_chat_completions_url(endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://mentedb.local/pratap-memory-drift",
            "X-Title": "MenteDB Pratap Memory Drift Benchmark",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {body}") from exc
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"LLM response had no choices: {data}")
    message = choices[0].get("message") or {}
    return str(message.get("content") or "")


def default_output_path(fixture: dict[str, Any]) -> Path:
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_RESULTS_DIR / f"{fixture['id']}-{timestamp}.jsonl"


def count_memory_items(fixture: dict[str, Any]) -> int:
    return sum(len(section.get("items", [])) for section in fixture["memory_bank"])


def dry_run(fixture: dict[str, Any]) -> int:
    print(f"Fixture: {fixture['id']}")
    print(f"Project context: {fixture['project_context']}")
    print(f"Memory entries: {count_memory_items(fixture)}")
    print(f"Turns: {len(fixture['turns'])}")
    for index, turn in enumerate(fixture["turns"], start=1):
        terms = ", ".join(turn.get("required_context_terms", []))
        print(f"{index:02d}. {turn['id']} -> required context: {terms}")
    return 0


def print_turn_summary(record: dict[str, Any]) -> None:
    status = "PASS" if record["context_pass"] else "FAIL"
    print(
        f"{record['turn_index']:02d}. {record['turn_id']}: {status} "
        f"({record['context_matched']}/{record['context_total']} context terms)"
    )
    if record["missing_context_terms"]:
        print(f"    missing: {', '.join(record['missing_context_terms'])}")
    snippets = record.get("context_snippets") or []
    for snippet in snippets[:2]:
        print(f"    context: {snippet}")


def run(args: argparse.Namespace) -> int:
    fixture = load_fixture(args.fixture)
    if args.dry_run:
        return dry_run(fixture)

    db_dir = args.db_dir or DEFAULT_DB_DIR
    output = args.output or default_output_path(fixture)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = fixture.get("settings", {})
    retrieve_limit = int(settings.get("retrieve_limit", 8))
    project_context = str(fixture["project_context"])
    llm_api_key = os.environ.get(args.llm_api_key_env)
    use_llm = bool(args.llm_endpoint)
    if use_llm and not llm_api_key:
        raise SystemExit(
            f"--llm-endpoint was provided but {args.llm_api_key_env} is not set."
        )

    db = open_db(args, db_dir)
    records: list[dict[str, Any]] = []
    try:
        seed_ids = seed_memory_bank(db, fixture)
        print(f"Seeded {len(seed_ids)} memory bank entries into {db_dir}")
        print(f"Turn mode: {args.turn_mode}")
        print(f"Output: {output}")

        with output.open("w", encoding="utf-8") as handle:
            header = {
                "kind": "run_start",
                "fixture_id": fixture["id"],
                "db_dir": str(db_dir),
                "turn_mode": args.turn_mode,
                "seeded_memory_ids": seed_ids,
                "llm_enabled": use_llm,
                "compare_without_memory": bool(args.compare_without_memory),
            }
            handle.write(json.dumps(header, sort_keys=True) + "\n")

            for turn_index, turn in enumerate(fixture["turns"], start=1):
                question = str(turn["question"])
                if args.turn_mode == "mimic-flutter":
                    pre_result = process_turn(
                        db,
                        question,
                        None,
                        turn_index,
                        project_context,
                    )
                else:
                    pre_result = search_context(db, question, retrieve_limit)

                retrieved_context = context_text(pre_result)
                context_terms = [str(term) for term in turn.get("required_context_terms", [])]
                context_coverage = compute_coverage(retrieved_context, context_terms)

                if use_llm:
                    with_memory_answer = chat_completion(
                        endpoint=str(args.llm_endpoint),
                        api_key=str(llm_api_key),
                        model=str(args.model),
                        system_prompt="You are a concise senior engineering assistant.",
                        memory_context=retrieved_context,
                        question=question,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    without_memory_answer = None
                    if args.compare_without_memory:
                        without_memory_answer = chat_completion(
                            endpoint=str(args.llm_endpoint),
                            api_key=str(llm_api_key),
                            model=str(args.model),
                            system_prompt="You are a concise senior engineering assistant.",
                            memory_context=None,
                            question=question,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                        )
                else:
                    with_memory_answer = synthetic_answer(turn)
                    without_memory_answer = None

                post_result = process_turn(
                    db,
                    question,
                    with_memory_answer,
                    turn_index,
                    project_context,
                )

                answer_terms = [str(term) for term in turn.get("expected_answer_terms", [])]
                answer_coverage = (
                    compute_coverage(with_memory_answer, answer_terms) if use_llm else None
                )
                snippets = [
                    str(item.get("content", "")).replace("\n", " ")[:220]
                    for item in pre_result.get("context", [])[:3]
                ]
                record = {
                    "kind": "turn",
                    "turn_index": turn_index,
                    "turn_id": turn["id"],
                    "phase": turn.get("phase"),
                    "title": turn.get("title"),
                    "question": question,
                    "context_count": len(pre_result.get("context") or []),
                    "context_total": context_coverage.total,
                    "context_matched": context_coverage.matched,
                    "context_pass": context_coverage.passed,
                    "missing_context_terms": context_coverage.missing,
                    "context_snippets": snippets,
                    "pre_stored_ids": [str(value) for value in pre_result.get("stored_ids", [])],
                    "post_stored_ids": [str(value) for value in post_result.get("stored_ids", [])],
                    "post_episodic_id": post_result.get("episodic_id"),
                    "post_facts_extracted": post_result.get("facts_extracted"),
                    "post_edges_created": post_result.get("edges_created"),
                    "post_enrichment_pending": post_result.get("enrichment_pending"),
                    "with_memory_answer": with_memory_answer if use_llm else None,
                    "without_memory_answer": without_memory_answer,
                    "answer_total": answer_coverage.total if answer_coverage else None,
                    "answer_matched": answer_coverage.matched if answer_coverage else None,
                    "answer_pass": answer_coverage.passed if answer_coverage else None,
                    "missing_answer_terms": answer_coverage.missing if answer_coverage else None,
                }
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                print_turn_summary(record)

            passed = sum(1 for record in records if record["context_pass"])
            summary = {
                "kind": "summary",
                "fixture_id": fixture["id"],
                "turns": len(records),
                "context_passed": passed,
                "context_failed": len(records) - passed,
                "context_pass_rate": passed / len(records) if records else 0.0,
            }
            handle.write(json.dumps(summary, sort_keys=True) + "\n")

        print("")
        print(
            f"Context pass rate: {passed}/{len(records)} "
            f"({(passed / len(records) * 100) if records else 0:.1f}%)"
        )
        print(f"DB kept at: {db_dir}")
        return 0 if passed == len(records) else 1
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


def main() -> None:
    load_env_file()
    raise SystemExit(run(parse_args()))


if __name__ == "__main__":
    main()
