"""Inspect memories written by the Pratap memory drift benchmark.

Examples:

  python benchmarks/db_reader.py --id <memory-id>
  python benchmarks/db_reader.py --jsonl benchmarks/results/latest.jsonl
  python benchmarks/db_reader.py --jsonl benchmarks/results/latest.jsonl --turn turn_08_identity_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_DIR = REPO_ROOT / "benchmarks" / "test-db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read formatted memories from a MenteDB DB.")
    parser.add_argument(
        "--db-dir",
        type=Path,
        default=DEFAULT_DB_DIR,
        help="MenteDB database directory. Defaults to benchmarks/test-db.",
    )
    parser.add_argument(
        "--id",
        action="append",
        default=[],
        help="Memory ID to read. Can be passed more than once.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=None,
        help="Text file containing one memory ID per line.",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Benchmark JSONL. Reads db_dir from the file and prints written IDs grouped by turn.",
    )
    parser.add_argument(
        "--turn",
        default=None,
        help="Optional turn_id filter when using --jsonl.",
    )
    parser.add_argument(
        "--include-seeded",
        action="store_true",
        help="When using --jsonl, also print seeded memory bank entries.",
    )
    parser.add_argument(
        "--content-width",
        type=int,
        default=120,
        help="Wrap content at this width.",
    )
    return parser.parse_args()


def import_mentedb() -> Any:
    try:
        from mentedb import MenteDB

        return MenteDB
    except ImportError as exc:
        raise SystemExit(
            "The mentedb Python package is not installed in this environment. "
            "Activate .venv and install/build the SDK first."
        ) from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
    return rows


def read_ids_file(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if value and not value.startswith("#"):
                ids.append(value)
    return ids


def unwrap_memory(memory: Any) -> dict[str, Any]:
    if isinstance(memory, dict):
        return memory
    fields = [
        "id",
        "content",
        "memory_type",
        "tags",
        "created_at",
        "accessed_at",
        "access_count",
        "salience",
        "confidence",
        "attributes",
    ]
    return {field: getattr(memory, field) for field in fields if hasattr(memory, field)}


def wrap_text(text: str, width: int, indent: str = "  ") -> str:
    words = text.replace("\n", " ").split()
    if not words:
        return ""
    lines: list[str] = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width and current.strip():
            lines.append(current.rstrip())
            current = indent + word + " "
        else:
            current += word + " "
    lines.append(current.rstrip())
    return "\n".join(lines)


def print_memory(db: Any, memory_id: str, *, label: str | None, width: int) -> None:
    print("")
    print("=" * 96)
    if label:
        print(label)
    print(f"id: {memory_id}")
    try:
        raw = db.get_memory(memory_id)
    except Exception as exc:
        print(f"status: NOT FOUND ({exc})")
        return

    memory = unwrap_memory(raw)
    print(f"type: {memory.get('memory_type', 'unknown')}")
    print(f"salience: {memory.get('salience', 'unknown')}")
    print(f"confidence: {memory.get('confidence', 'unknown')}")
    tags = memory.get("tags") or []
    print(f"tags: {', '.join(tags) if tags else '-'}")
    if memory.get("created_at") is not None:
        print(f"created_at: {memory.get('created_at')}")
    attributes = memory.get("attributes") or {}
    if attributes:
        print(f"attributes: {json.dumps(attributes, sort_keys=True)}")
    print("content:")
    print(wrap_text(str(memory.get("content", "")), width))


def ids_from_jsonl(rows: list[dict[str, Any]], *, include_seeded: bool, turn: str | None) -> tuple[Path | None, list[tuple[str, str]]]:
    db_dir: Path | None = None
    entries: list[tuple[str, str]] = []
    for row in rows:
        kind = row.get("kind")
        if kind == "run_start":
            value = row.get("db_dir")
            if value:
                db_dir = Path(str(value))
            if include_seeded:
                for index, memory_id in enumerate(row.get("seeded_memory_ids") or [], start=1):
                    entries.append((f"seeded #{index}", str(memory_id)))
            continue

        if kind != "turn":
            continue
        turn_id = str(row.get("turn_id"))
        if turn is not None and turn_id != turn:
            continue
        for index, memory_id in enumerate(row.get("pre_stored_ids") or [], start=1):
            entries.append((f"{turn_id} pre #{index}", str(memory_id)))
        for index, memory_id in enumerate(row.get("post_stored_ids") or [], start=1):
            entries.append((f"{turn_id} post #{index}", str(memory_id)))
    return db_dir, entries


def main() -> None:
    args = parse_args()
    db_dir = args.db_dir
    entries: list[tuple[str, str]] = []

    if args.jsonl:
        rows = load_jsonl(args.jsonl)
        jsonl_db_dir, jsonl_entries = ids_from_jsonl(
            rows,
            include_seeded=args.include_seeded,
            turn=args.turn,
        )
        if jsonl_db_dir is not None:
            db_dir = jsonl_db_dir
        entries.extend(jsonl_entries)

    for memory_id in args.id:
        entries.append(("manual", memory_id))

    if args.ids_file:
        for memory_id in read_ids_file(args.ids_file):
            entries.append((f"ids-file:{args.ids_file.name}", memory_id))

    if not entries:
        raise SystemExit("No memory IDs provided. Use --id, --ids-file, or --jsonl.")

    MenteDB = import_mentedb()
    print(f"Opening DB: {db_dir}")
    db = MenteDB(str(db_dir))
    try:
        for label, memory_id in entries:
            print_memory(db, memory_id, label=label, width=args.content_width)
    finally:
        db.close()


if __name__ == "__main__":
    main()
