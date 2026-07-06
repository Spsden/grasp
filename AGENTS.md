# MenteDB Development Instructions

## Project overview

Grasp (formerly MenteDB) is a SQLite backed memory layer for AI agents. The durable store is SQLite with sqlite-vec vector search, FTS5 keyword search, and inspectable SQL rows for graph edges, provenance, entities, and retrieval traces. On top of that it provides a custom query language (MQL), context assembly with U curve attention layout, and 7 unique cognitive features (stream cognition, write time inference, trajectory tracking, phantom memories, interference detection, pain signals, speculative pre assembly).

The Rust crate and facade names are still `mentedb*` / `MenteDb` until the rename is completed.

## Workspace structure

```
crates/
  mentedb-core/       Core types, config, error, MVCC, multi agent
  mentedb-sqlite/     SQLite durable store: sqlite-vec, FTS5, graph rows, provenance, retrieval traces
  mentedb-graph/      In-memory graph algorithms: traversal, belief propagation, contradiction
  mentedb-query/      MQL lexer, parser, planner (cross-language query protocol)
  mentedb-context/    U curve attention layout, delta tracker, serializers
  mentedb-cognitive/  Stream cognition, write inference, trajectory, phantoms, interference, pain, speculative
  mentedb-consolidation/ Decay, archival, extraction, compression, GDPR forget
  mentedb-embedding/  Provider trait, hash/HTTP providers, LRU cache
  mentedb-extraction/ LLM extraction pipeline (entities, claims, relationships)
  mentedb-server/     Axum REST API, JWT auth, rate limiting, WebSocket, gRPC
  mentedb/            Unified facade (MenteDb struct)
sdks/
  python/             PyO3 bindings + pure Python client
  typescript/         napi-rs bindings + TypeScript client
  python/integrations/langchain/  LangChain memory, retriever, chat history
  python/integrations/crewai/     CrewAI memory and tool adapter
```

The SDKs are excluded from the Cargo workspace and build independently.

## Build, test, and lint

Always run these before committing:

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

The server binary is `mentedb-server` and runs on axum with JWT auth, rate limiting, and WebSocket support.

## Key types

- `MemoryNode`: The fundamental storage unit (id, content, memory_type, embedding, metadata, timestamps)
- `MemoryEdge`: Typed relationship between memories (caused, contradicts, relates_to, obsoletes, etc.)
- `MenteDb`: The unified facade that coordinates all subsystems
- `MenteConfig`: Top level config with sub configs for every subsystem
- `MenteError` / `MenteResult<T>`: Error handling throughout

## Architecture notes

- Storage is SQLite (WAL mode) as the durable source of truth; sqlite-vec `vec0` powers vector search and FTS5 powers keyword search, both rebuildable projections over canonical memory rows
- Vector index is created lazily via `ensure_vector_index(dim)` and backfilled from the `memories` table, so changing embedding dimensions never loses data
- Graph uses an in-memory CSR/CSC mirror (GraphManager) hydrated from the `edges` table on open and kept write-through on `relate`
- Hybrid recall blends vec0 KNN, FTS5/BM25, tags, temporal, and salience via reciprocal rank fusion, with weights in `RetrievalConfig`
- MQL is a custom query language parsed by hand written recursive descent parser
- Context assembly uses U curve attention layout (primacy/recency bias) with delta aware serving
- 7 cognitive features are unique to Grasp and do not exist in any other memory engine

## Conventions

- Rust edition 2024, Apache 2.0 license
- All heuristics and thresholds must be in Config structs, never hardcoded magic numbers
- Use `MenteResult<T>` for error handling, no `unwrap()` in library code
- Commit messages: conventional style (feat:, fix:, chore:), single line, no emojis
- No emojis or em/en dashes in prose anywhere (docs, README, comments). Use commas instead.
- Trait based design: `EmbeddingProvider`, storage backends, etc.

## Key files

- `crates/mentedb/src/lib.rs`: Unified `MenteDb` facade (open, store, recall, relate, forget, close)
- `crates/mentedb-sqlite/src/lib.rs`: SQLite `Backend`, vec0/FTS5 projections, hybrid recall, retrieval traces
- `docs/SQLITE_MEMORY_SCHEMA.md`: Canonical table schema and projection rationale
- `crates/mentedb-core/src/config.rs`: All configuration (MenteConfig and sub configs)
- `crates/mentedb-core/src/error.rs`: MenteError enum
- `crates/mentedb-cognitive/src/`: 7 cognitive feature modules
- `crates/mentedb-server/src/main.rs`: Axum server entry point
- `.github/workflows/ci.yml`: CI pipeline (check, test, fmt, clippy)
- `.github/workflows/release.yml`: Release pipeline (crates.io, PyPI, npm)

## Do not

- Introduce a second database engine beyond SQLite (SQLite is the durable source of truth; sqlite-vec and FTS5 are rebuildable projections, not separate stores)
- Remove cognitive features or weaken their configurability
- Hardcode language specific patterns, use config structs
- Modify SDK Cargo.toml files to join the root workspace
