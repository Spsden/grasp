# SQLite Memory Schema

This document describes the long-term storage direction for MenteDB in the
`feat/sqlite-vec-storage` branch. SQLite is the durable source of truth.
`sqlite-vec` and FTS5 are rebuildable projections over canonical memory rows.

The product goal is not just persistence. Synapse needs a memory layer that is
scalable enough for local-first agent memory, easy to inspect during testing,
and traceable when recall returns the wrong thing or misses something important.

## Design Goals

- Canonical rows stay simple, durable, and queryable with normal SQL.
- Derived search structures are rebuildable from canonical data.
- Every logical write is atomic and has an audit record.
- Retrieval can be traced without changing recall behavior.
- Ranking parameters are explicit config, not buried constants.
- Graph relationships are stored as rows so they can be inspected and migrated.
- The schema includes provenance and entity resolution, with room for feedback
  and future reranking experiments.

## Implemented Tables

### `schema_meta`

Small key-value table for schema and projection metadata.

```sql
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Current keys:

- `schema_version`, the logical schema version.
- `embedding_dim`, the active sqlite-vec embedding dimension.

### `memories`

Canonical memory table. This is the source of truth for content, metadata,
temporal validity, salience, confidence, and the original embedding bytes.

```sql
CREATE TABLE memories (
    id           TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    space_id     TEXT NOT NULL,
    memory_type  TEXT NOT NULL,
    content      TEXT NOT NULL,
    embedding    BLOB,
    created_at   INTEGER NOT NULL,
    accessed_at  INTEGER NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    salience     REAL    NOT NULL DEFAULT 1.0,
    confidence   REAL    NOT NULL DEFAULT 1.0,
    attributes   TEXT    NOT NULL DEFAULT '{}',
    valid_from   INTEGER,
    valid_until  INTEGER
);
```

Indexes:

```sql
CREATE INDEX idx_mem_agent    ON memories(agent_id);
CREATE INDEX idx_mem_space    ON memories(space_id);
CREATE INDEX idx_mem_type     ON memories(memory_type);
CREATE INDEX idx_mem_salience ON memories(salience DESC);
CREATE INDEX idx_mem_created  ON memories(created_at DESC);
```

Notes:

- `embedding` stores the original vector exactly as supplied.
- `memory_vec` stores the normalized projection used by vector search.
- `valid_from` and `valid_until` model factual validity. A superseded memory can
  remain historically queryable while being filtered from current recall.
- `attributes` is JSON for product metadata, extraction metadata, source labels,
  and experimental flags.

### `memory_tags`

Deterministic tag membership table. Tags are separate rows so filters are
indexable and inspectable.

```sql
CREATE TABLE memory_tags (
    memory_id TEXT NOT NULL,
    tag       TEXT NOT NULL,
    PRIMARY KEY (memory_id, tag)
);

CREATE INDEX idx_tag ON memory_tags(tag);
```

### `memory_vec`

sqlite-vec virtual table. This table is a projection, not the source of truth.

```sql
CREATE VIRTUAL TABLE memory_vec USING vec0(
    embedding float[N]
);
```

Operational rules:

- `N` is stored in `schema_meta.embedding_dim`.
- The rowid matches the SQLite rowid in `memories`.
- Rebuild is safe because the original embedding lives in `memories.embedding`.
- If the embedder dimension changes, the projection is dropped, recreated, and
  backfilled in one transaction.

### `memories_fts`

FTS5 full-text mirror over `memories.content`.

```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid'
);
```

Triggers keep this table synchronized on insert, update, and delete.

### `edges`

Durable knowledge graph edge table. This replaces the custom CSR/CSC storage
path as the durable graph source.

```sql
CREATE TABLE edges (
    source      TEXT NOT NULL,
    target      TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL,
    created_at  INTEGER NOT NULL,
    valid_from  INTEGER,
    valid_until INTEGER,
    label       TEXT
);

CREATE INDEX idx_edge_source ON edges(source, edge_type);
CREATE INDEX idx_edge_target ON edges(target, edge_type);
```

The facade still mirrors edges into `GraphManager` for hot cognitive algorithms.
SQLite remains the durable truth.

### `memory_operations`

Append-only write audit table. This is for debugging and reproducibility.

```sql
CREATE TABLE memory_operations (
    operation_id   TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    memory_id      TEXT,
    source         TEXT,
    target         TEXT,
    payload_json   TEXT NOT NULL,
    created_at     INTEGER NOT NULL
);

CREATE INDEX idx_memory_operations_created
    ON memory_operations(created_at DESC);
CREATE INDEX idx_memory_operations_memory
    ON memory_operations(memory_id, created_at DESC);
CREATE INDEX idx_memory_operations_edge
    ON memory_operations(source, target, created_at DESC);
```

Current operation types:

- `memory_upsert`
- `memory_delete`
- `edge_insert`
- `edge_delete_for_memory`
- `vector_index_rebuild`
- `memory_source_upsert`
- `conversation_upsert`
- `conversation_event_upsert`
- `extraction_run_upsert`
- `entity_upsert`
- `entity_alias_upsert`
- `memory_entity_link_upsert`
- `claim_upsert`
- `claim_entity_link_upsert`
- `claim_evidence_upsert`
- `entity_relationship_upsert`
- `relationship_evidence_upsert`

### `memory_sources`

Track where a memory came from.

```sql
CREATE TABLE memory_sources (
    source_id       TEXT PRIMARY KEY,
    memory_id       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    conversation_id TEXT,
    turn_id         TEXT,
    actor_id        TEXT,
    observed_at     INTEGER,
    extractor       TEXT,
    extractor_hash  TEXT,
    prompt_hash     TEXT,
    payload_json    TEXT NOT NULL DEFAULT '{}',
    created_at      INTEGER NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX idx_memory_sources_memory
    ON memory_sources(memory_id, created_at DESC);
CREATE INDEX idx_memory_sources_type
    ON memory_sources(source_type, created_at DESC);
CREATE INDEX idx_memory_sources_turn
    ON memory_sources(conversation_id, turn_id);
```

Why it helps:

- Debug whether a bad memory came from raw user text, an inferred fact, import,
  sync, or consolidation.
- Compare extraction models and prompt versions.
- Replay extraction from the same source later.

### `conversations`

Durable container for a chat thread, meeting, agent session, or imported
timeline.

```sql
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,
    title           TEXT,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE INDEX idx_conversations_updated
    ON conversations(updated_at DESC);
```

### `conversation_events`

Appendable observed events inside a conversation. A message, tool call,
participant join, participant leave, file observation, or system event can all
be represented here.

```sql
CREATE TABLE conversation_events (
    event_id        TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    turn_id         TEXT,
    event_type      TEXT NOT NULL,
    actor_id        TEXT,
    content         TEXT,
    payload_json    TEXT NOT NULL DEFAULT '{}',
    observed_at     INTEGER NOT NULL,
    created_at      INTEGER NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
        ON DELETE CASCADE
);
```

Why event rows help:

- A turn is no longer guessed from text. It is an observed event boundary.
- Real conversations can include joins, leaves, tool calls, and non-message
  context.
- Memory extraction can be replayed from an exact timeline.

### `extraction_runs`

Versioned extraction attempts over a source memory or conversation.

```sql
CREATE TABLE extraction_runs (
    run_id            TEXT PRIMARY KEY,
    source_memory_id  TEXT,
    conversation_id   TEXT,
    extractor         TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    model             TEXT,
    prompt_hash       TEXT,
    config_hash       TEXT,
    status            TEXT NOT NULL,
    error             TEXT,
    output_json       TEXT NOT NULL DEFAULT '{}',
    started_at        INTEGER NOT NULL,
    completed_at      INTEGER
);
```

Extraction runs make LLM-derived memory auditable. A derived claim should be
traceable to the extractor name, extractor version, model, prompt hash, config
hash, raw output, and source evidence.

### `entities`

Canonical entities for people, projects, apps, files, places, and concepts.

```sql
CREATE TABLE entities (
    entity_id       TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    canonical       TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    confidence      REAL NOT NULL DEFAULT 1.0,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL
);

CREATE INDEX idx_entities_type_canonical
    ON entities(entity_type, canonical);
CREATE INDEX idx_entities_canonical
    ON entities(canonical);
```

### `entity_aliases`

Aliases and resolved names for each entity.

```sql
CREATE TABLE entity_aliases (
    entity_id  TEXT NOT NULL,
    alias      TEXT NOT NULL,
    source     TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(entity_id, alias),
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX idx_entity_aliases_alias
    ON entity_aliases(alias);
```

### `memory_entities`

Many-to-many links between memories and entities.

```sql
CREATE TABLE memory_entities (
    memory_id  TEXT NOT NULL,
    entity_id  TEXT NOT NULL,
    role       TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence   TEXT,
    PRIMARY KEY(memory_id, entity_id, role),
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX idx_memory_entities_entity
    ON memory_entities(entity_id);
CREATE INDEX idx_memory_entities_memory
    ON memory_entities(memory_id);
```

Why the entity tables help:

- Make "the same person/project/app" debuggable instead of hiding it inside
  embeddings.
- Allow deterministic filters before vector search.
- Give the graph better nodes than only memory-to-memory edges.

### `claims`

Atomic derived facts or preferences. Claims are not raw truth. They are
evidence-backed interpretations that can be corrected, superseded, or rebuilt.

```sql
CREATE TABLE claims (
    claim_id          TEXT PRIMARY KEY,
    claim_text        TEXT NOT NULL,
    claim_type        TEXT NOT NULL,
    subject_entity_id TEXT,
    predicate         TEXT,
    object_entity_id  TEXT,
    confidence        REAL NOT NULL DEFAULT 1.0,
    status            TEXT NOT NULL DEFAULT 'active',
    valid_from        INTEGER,
    valid_until       INTEGER,
    attributes_json   TEXT NOT NULL DEFAULT '{}',
    source_run_id     TEXT,
    created_at        INTEGER NOT NULL,
    updated_at        INTEGER NOT NULL
);
```

### `claim_entities`

Links claim rows to participating entities.

```sql
CREATE TABLE claim_entities (
    claim_id   TEXT NOT NULL,
    entity_id  TEXT NOT NULL,
    role       TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY(claim_id, entity_id, role)
);
```

### `claim_evidence`

Evidence spans for claims.

```sql
CREATE TABLE claim_evidence (
    evidence_id   TEXT PRIMARY KEY,
    claim_id      TEXT NOT NULL,
    memory_id     TEXT NOT NULL,
    source_id     TEXT,
    evidence_text TEXT,
    span_start    INTEGER,
    span_end      INTEGER,
    confidence    REAL NOT NULL DEFAULT 1.0,
    created_at    INTEGER NOT NULL
);
```

### `entity_relationships`

Typed entity-to-entity relationships derived from evidence.

```sql
CREATE TABLE entity_relationships (
    relationship_id  TEXT PRIMARY KEY,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL,
    relation_type    TEXT NOT NULL,
    confidence       REAL NOT NULL DEFAULT 1.0,
    status           TEXT NOT NULL DEFAULT 'active',
    valid_from       INTEGER,
    valid_until      INTEGER,
    attributes_json  TEXT NOT NULL DEFAULT '{}',
    source_run_id    TEXT,
    created_at       INTEGER NOT NULL,
    updated_at       INTEGER NOT NULL
);
```

### `relationship_evidence`

Evidence spans for entity relationships.

```sql
CREATE TABLE relationship_evidence (
    evidence_id     TEXT PRIMARY KEY,
    relationship_id TEXT NOT NULL,
    memory_id       TEXT NOT NULL,
    source_id       TEXT,
    evidence_text   TEXT,
    span_start      INTEGER,
    span_end        INTEGER,
    confidence      REAL NOT NULL DEFAULT 1.0,
    created_at      INTEGER NOT NULL
);
```

Why claims and relationships help:

- They separate "what was said" from "what Grasp inferred".
- Corrections can update derived claims without deleting raw episodes.
- Debugging can point to exact evidence spans, extractor versions, and source
  memories.
- Future rerankers can search facts and relationships before falling back to
  raw text.

### `retrieval_traces`

Optional retrieval trace header table. Tracing is disabled by default and can be
enabled during Synapse testing.

```sql
CREATE TABLE retrieval_traces (
    trace_id            TEXT PRIMARY KEY,
    query_text          TEXT,
    query_embedding_dim INTEGER NOT NULL,
    k                   INTEGER NOT NULL,
    fetch_k             INTEGER NOT NULL,
    tags_json           TEXT NOT NULL,
    tags_or             INTEGER NOT NULL,
    time_start          INTEGER,
    time_end            INTEGER,
    candidate_count     INTEGER NOT NULL,
    result_count        INTEGER NOT NULL,
    config_json         TEXT NOT NULL,
    created_at          INTEGER NOT NULL
);

CREATE INDEX idx_retrieval_traces_created
    ON retrieval_traces(created_at DESC);
```

`config_json` captures the retrieval parameters used for that exact run:

- `fetch_multiplier`
- `rrf_k`
- `rrf_weight`
- `salience_weight`
- `recency_weight`
- `recency_prior`
- `multi_query_rrf_k`
- `trace_retention_limit`

### `retrieval_trace_hits`

Final ranked hit details for a retrieval trace.

```sql
CREATE TABLE retrieval_trace_hits (
    trace_id    TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    memory_id   TEXT NOT NULL,
    score       REAL NOT NULL,
    vector_rank INTEGER,
    bm25_rank   INTEGER,
    salience    REAL,
    PRIMARY KEY(trace_id, rank),
    FOREIGN KEY(trace_id) REFERENCES retrieval_traces(trace_id) ON DELETE CASCADE
);

CREATE INDEX idx_retrieval_trace_hits_memory
    ON retrieval_trace_hits(memory_id);
```

This table answers the practical question, "Why did this result appear here?"

## Runtime Flow

### Store

1. Validate embedding dimension.
2. Upsert the canonical row in `memories`.
3. Refresh normalized vector row in `memory_vec`.
4. Refresh deterministic tag rows in `memory_tags`.
5. Let FTS triggers refresh `memories_fts`.
6. Append `memory_operations` row.
7. If the caller provides a `MemorySource`, insert or update `memory_sources`.
8. Commit as one SQLite transaction.

### Relate

1. Insert a durable row in `edges`.
2. Append `memory_operations` row.
3. Commit.
4. Mirror the edge into the in-memory `GraphManager`.

### Recall

1. Build the tag and time candidate set.
2. Fetch vector candidates from `memory_vec`, or use filtered cosine scan when
   tag/time filters are restrictive.
3. Fetch BM25 candidates from `memories_fts` when query text is provided.
4. Fuse candidates with Reciprocal Rank Fusion.
5. Apply salience and recency weights from `RetrievalConfig`.
6. Filter invalidated or superseded memories in the facade.
7. If tracing is enabled, persist a trace header and final ranked hits.

### Forget

1. Delete vector projection row.
2. Delete tag rows.
3. Delete graph edges touching the memory.
4. Delete the canonical memory row.
5. Append `memory_operations` row.
6. Commit.
7. Remove the node from the in-memory graph mirror.

## Debug Queries

Latest retrieval traces:

```sql
SELECT trace_id, query_text, k, fetch_k, candidate_count, result_count,
       config_json, created_at
FROM retrieval_traces
ORDER BY created_at DESC
LIMIT 20;
```

Why a trace ranked its final results:

```sql
SELECT h.rank, h.memory_id, h.score, h.vector_rank, h.bm25_rank, h.salience,
       m.content, m.memory_type, m.confidence, m.valid_from, m.valid_until
FROM retrieval_trace_hits h
JOIN memories m ON m.id = h.memory_id
WHERE h.trace_id = ?
ORDER BY h.rank;
```

What happened to one memory:

```sql
SELECT operation_type, payload_json, created_at
FROM memory_operations
WHERE memory_id = ?
   OR source = ?
   OR target = ?
ORDER BY created_at DESC;
```

Where one memory came from:

```sql
SELECT source_type, conversation_id, turn_id, actor_id, observed_at,
       extractor, extractor_hash, prompt_hash, payload_json, created_at
FROM memory_sources
WHERE memory_id = ?
ORDER BY created_at DESC;
```

Which entities a memory mentions:

```sql
SELECT e.entity_type, e.canonical, me.role, me.confidence, me.evidence
FROM memory_entities me
JOIN entities e ON e.entity_id = me.entity_id
WHERE me.memory_id = ?
ORDER BY me.confidence DESC, e.canonical ASC;
```

Resolve an entity alias:

```sql
SELECT e.entity_id, e.entity_type, e.canonical, a.alias, a.confidence
FROM entity_aliases a
JOIN entities e ON e.entity_id = a.entity_id
WHERE a.alias = ?
ORDER BY a.confidence DESC, e.updated_at DESC;
```

Graph neighborhood:

```sql
SELECT source, edge_type, target, weight, valid_from, valid_until, label
FROM edges
WHERE source = ? OR target = ?
ORDER BY created_at DESC;
```

Tag distribution:

```sql
SELECT tag, COUNT(*) AS n
FROM memory_tags
GROUP BY tag
ORDER BY n DESC, tag ASC;
```

Memories that have embeddings but no current vector projection:

```sql
SELECT m.id, m.content
FROM memories m
LEFT JOIN memory_vec v ON v.rowid = m.rowid
WHERE m.embedding IS NOT NULL
  AND v.rowid IS NULL;
```

Current invalidated memories:

```sql
SELECT id, content, valid_from, valid_until
FROM memories
WHERE valid_until IS NOT NULL
ORDER BY valid_until DESC;
```

## Recommended Next Schema

These tables can still wait while Synapse is in build mode.

### `memory_versions`

Track logical revisions of a memory.

```sql
CREATE TABLE memory_versions (
    version_id   TEXT PRIMARY KEY,
    memory_id    TEXT NOT NULL,
    version      INTEGER NOT NULL,
    content      TEXT NOT NULL,
    attributes   TEXT NOT NULL,
    changed_by   TEXT,
    reason       TEXT,
    created_at   INTEGER NOT NULL
);
```

Why it helps:

- Inspect how a memory changed over time.
- Support user-visible undo and correction flows.
- Keep canonical `memories` small while preserving history.

### `retrieval_experiments`

Record offline and online retrieval experiments.

```sql
CREATE TABLE retrieval_experiments (
    experiment_id TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    notes         TEXT,
    created_at    INTEGER NOT NULL
);
```

### `memory_feedback`

Capture user or evaluator feedback on recall quality.

```sql
CREATE TABLE memory_feedback (
    feedback_id TEXT PRIMARY KEY,
    trace_id    TEXT,
    memory_id   TEXT,
    rating      INTEGER,
    label       TEXT,
    comment     TEXT,
    created_at  INTEGER NOT NULL
);
```

Why it helps:

- Turn manual debugging into training data.
- Compare reranking ideas against real failures.
- Identify memories that are over-recalled, stale, or missing.

## Scaling Notes

SQLite is a good fit for Synapse's local-first memory layer because it is
durable, inspectable, embeddable, and easy to ship. The practical scaling model
should be:

- One SQLite database per user profile or memory space for local isolation.
- WAL mode enabled for concurrent readers and one serialized writer.
- Store raw embeddings once in `memories`, keep vector and FTS projections
  rebuildable.
- Use SQL filters before vector search whenever tags, time, or entities are
  selective.
- Keep tracing off in normal operation, enable it during testing or sample it
  for production diagnostics.
- Periodically run integrity checks that verify `memories`, `memory_vec`,
  `memories_fts`, `memory_tags`, and `edges` are consistent.

If Synapse later needs server-scale shared memory, this schema can still work as
the local cache and debugging substrate while a remote service owns sync,
authorization, compaction, and multi-writer conflict resolution.

## Product Recommendations

1. Add a Synapse memory inspector that reads these tables directly:
   recent operations, current memory row, sources, entities, tags, edges,
   retrieval traces, and trace hits.
2. Add golden recall fixtures for testing. Each fixture should include raw
   conversation input, expected stored memories, expected entities, and expected
   recall answers.
3. Treat retrieval config as experiment state. Store any non-default config used
   during an eval run so results are reproducible.
4. Start writing `memory_sources` from `process_turn`, imports, enrichment, and
   plugin actions before making extraction more aggressive.
5. Start writing `entities`, `entity_aliases`, and `memory_entities` from the
   deterministic extraction or action-schema layer before complex graph work.
6. Keep all ranking and admission thresholds in config structs. Avoid hidden
   constants in search, consolidation, entity merge, and forgetting paths.
