# MenteDB Cognitive Features — How They Actually Work

A code-level walkthrough of `crates/mentedb-cognitive` and an honest verdict on
each feature: **simple heuristic**, **hybrid**, or **genuinely good**.

> Scope: everything under `crates/mentedb-cognitive/src/` (`lib.rs`, `llm.rs`,
> `write_inference.rs`, `entity.rs`, `trajectory.rs`, `phantom.rs`, `pain.rs`,
> `interference.rs`, `speculative.rs`, `stream.rs`) — ~4,600 lines.

---

## The one design idea that ties it all together

Every cognitive feature uses a **three-tier cost model**:

| Tier | Cost | When used |
|------|------|-----------|
| 1. Cache / learned state | Free | Repeat of something seen before |
| 2. Rule / heuristic | Free (CPU only) | Cheap disambiguation |
| 3. LLM judge | $ + latency | Only for genuinely novel/ambiguous cases |

The consistent trick is: **pay the LLM once, remember the answer, never pay
again for the same thing.** Alias tables, topic caches, and transition maps are
persisted to disk (atomic temp-file + rename) so the amortization survives
restarts. This is what makes the "write-time intelligence" affordable rather
than a token firehose.

The actual *intelligence* is **LLM-as-judge** (offloaded to your configured
GPT/Claude/etc.) layered on top of **classical heuristics** (cosine similarity,
keyword overlap, threshold rules). There are no trained neural classifiers
inside this crate. That's a pragmatic, defensible choice — the quality ceiling
is the quality of your underlying model.

---

## Cross-cutting: the LLM judge layer (`llm.rs`, 968 LOC)

`CognitiveLlmService<J: LlmJudge>` is the brain everything else calls. It wraps
any chat-completion provider and exposes **7 typed judgment methods**, each with
its own system prompt:

- `judge_invalidation` — does new memory make old one false? → `Keep | Invalidate | Update`
- `detect_contradiction` — do two memories conflict? → `Compatible | Contradicts | Supersedes(winner)`
- `resolve_entities` — group references that mean the same thing → merge groups
- `consolidate` — what to do with a cluster of similar memories? → `KeepAll | Merge | Deduplicate`
- `canonicalize_topic` — map a raw message to a 1–3 word topic label
- `generate_community_summary` — summarize a cluster of related entities
- `generate_user_profile` — distill accumulated facts into a <200-word profile

**Verdict: genuinely good prompt engineering.** The prompts (llm.rs:171–273) are
well-structured: JSON-only output, few-shot examples, explicit rules, and a real
distinction between *logical contradiction* ("cannot both be true") and *temporal
supersession* ("newer replaces older") — a nuance most memory tools miss. The
JSON parser (llm.rs:489) robustly handles markdown fences, surrounding prose,
and nested braces inside strings. Typed `#[serde(tag = "verdict")]` enums mean
the rest of the engine matches on exhaustive verdicts, not stringly-typed soup.

---

## Feature-by-feature

### 1. Write-time inference — `write_inference.rs` (544 LOC)
**Hybrid — the flagship feature.**

Runs on every write. Two paths:

- **`infer_on_write`** (no LLM): pure cosine-similarity thresholds.
  - `> 0.95` & same agent & different content → flag contradiction
  - `> 0.85` & newer timestamp → mark old obsolete + set `valid_until`
  - `0.60–0.85` → create `Related` graph edge
  - `Correction` memory type → supersede most-similar + decay its confidence

  This is a **simple heuristic** — fast, free, but blind to semantics (it can't
  tell that "Alice works at Acme" is invalidated by "Alice joined Google" because
  those embeddings aren't >0.95 similar).

- **`infer_on_write_with_llm`**: the real version. **Only calls the LLM for pairs
  with similarity > 0.5** (explicit token-cost control — write_inference.rs:224),
  then asks the judge `invalidate → update → keep`, falling through to
  `contradiction` only on `keep` + `sim > 0.7`. This *can* catch the
  Acme→Google case that the heuristic misses.

It emits a typed `InferredAction` stream (`FlagContradiction`, `MarkObsolete`,
`InvalidateMemory`, `UpdateContent`, `CreateEdge`, `UpdateConfidence`,
`PropagateBeliefChange`) that the caller applies transactionally.

**Verdict: genuinely good** in the LLM path; the heuristic path is a sensible
free fallback.

### 2. Entity resolution — `entity.rs` (598 LOC)
**Hybrid — well-engineered.**

Three tiers:
1. **Cache** — `aliases: HashMap<alias, canonical>`. Instant.
2. **Rule-based word-subset** — clever: splits on whitespace/hyphens/underscores
   and checks set-subset. So `"alice"` matches `"alice smith"` but `"java"`
   correctly does **not** match `"javascript"` (entity.rs:207). Returns at
   confidence 0.7.
3. **LLM** — only for the unresolved remainder; results cached as a merge group.

Also maintains a **negative cache** (`"Python"` ≠ `"python snake"`) so the LLM
isn't asked the same disambiguation twice. Persists across sessions.

**Verdict: genuinely good.** The word-subset rule alone solves most real cases
for free, and the LLM is reserved for the long tail.

### 3. Trajectory tracking — `trajectory.rs` (912 LOC)
**Hybrid.**

Models the conversation as a sequence of `TrajectoryNode`s (topic + decision
state: `Investigating → NarrowedTo → Decided → Interrupted → Completed`) and
learns a **first-order Markov chain** of topic transitions
(`HashMap<from, HashMap<to, count>>`).

- `predict_from(topic, n)` — top-n likely next topics by frequency.
- `reinforce(from, to)` — +2 bonus when a prediction hits the speculative cache.
- `decay(from, to)` — saturating-subtract so stale transitions fade.
- Topic canonicalization: cache → LLM (with the canonicalize-topic judge) →
  normalize fallback, and the learned label is cached so the LLM is called once
  per topic ever.

**Verdict: simple-but-appropriate ML.** First-order Markov on bag-of-topics
won't win any benchmarks, but it's the right level of model for the data volume
an agent produces, it's fully transparent, and it improves with use.

### 4. Speculative context cache — `speculative.rs` (420 LOC)
**Heuristic cache.**

Given predicted upcoming topics (from the trajectory tracker), pre-assembles
context windows so the next query can be a cache hit. Match scoring:
cosine on topic embeddings when available, **Jaccard keyword overlap** as
fallback. LRU eviction, stale-sweep by age, atomic on-disk persistence that
keeps only entries with `hit_count ≥ min_hits`.

**Verdict: simple heuristic, done cleanly.** The cache itself is excellent
engineering; its value is entirely a function of how good the trajectory
predictions feeding it are. No magic here.

### 5. Phantom memory detection — `phantom.rs` (578 LOC)
**Hybrid, leaning heuristic.**

Detects references to entities the agent has no stored knowledge about.
Two strategies:
- **Registry-based (precise):** you register entities you care about; any
  referenced-but-unregistered entity is flagged `Medium`/`High`.
- **Heuristic (fallback):** capitalized words, quoted terms, and
  "technical-looking" tokens (`-`/`.`/`_`/ALL_CAPS) flagged `Low`.

`detect_gaps_explicit` is the clean path (caller passes exactly which entities
were mentioned); the heuristic path can be disabled.

**Verdict: sensible hybrid.** The registry path is a solid, low-false-positive
design; the heuristic path is crude but optional.

### 6. Pain signals — `pain.rs` (162 LOC)
**Simple heuristic.**

Records negative experiences (failed actions, corrections) with an intensity
and a **trigger keyword list**. On context assembly, matches context keywords
against triggers (substring, lowercase) and ranks by `intensity × relevance`.
Intensity decays exponentially: `I(t) = I₀ · e^(−rate·Δt)`.

**Verdict: simple heuristic.** Keyword-substring matching is brittle (no
semantics), but the exponential-decay signal is a real, if basic,
signal-processing touch. Lightweight and cheap.

### 7. Interference detection — `interference.rs` (167 LOC)
**Simple heuristic.**

For each pair of memories in a context window, if cosine > 0.8 and content
differs → it's an "interference pair" (LLM-confusable). Generates a
`"Do not confuse A and B"` disambiguation note, then **greedily reorders the
context** so interfering pairs are never adjacent (interference.rs:65).

**Verdict: simple heuristic, with a nice context-engineering twist.** Keeping
confusable memories physically apart in the prompt window is a genuinely useful
trick. Cost is **O(n²)** per context — fine for small windows, a concern at
scale.

### 8. Stream monitoring — `stream.rs` (205 LOC)
**Simple heuristic — the weakest feature.**

Watches the LLM's output token stream via a ring buffer. For each stored fact,
computes keyword overlap against the generated text; if overlap > 0.5, not
identical, **and the text contains a negation keyword** (`"not "`, `"actually "`,
`"instead "`, `"don't "`, …) → flags a `Contradiction` alert; > 0.8 with no
negation → `Reinforcement`.

**Verdict: simple heuristic, brittle.** Negation-by-keyword is exactly the kind
of rule that produces false positives ("I could *not* be happier") and false
negatives ("switched away from"). It's a reasonable *streaming* guard (can't run
an LLM judge per token), but don't trust it as ground truth.

---

## Summary table

| Feature | LOC | Classification | LLM used? | Verdict |
|---|---|---|---|---|
| LLM judge layer (`llm.rs`) | 968 | LLM-backed | yes (7 prompts) | **Genuinely good** |
| Write inference (`write_inference.rs`) | 544 | Hybrid | optional | **Genuinely good** (LLM path) |
| Entity resolution (`entity.rs`) | 598 | Hybrid | optional | **Genuinely good** |
| Trajectory (`trajectory.rs`) | 912 | Hybrid (Markov) | optional | Good for the data volume |
| Speculative cache (`speculative.rs`) | 420 | Heuristic cache | no | Clean, value depends on predictions |
| Phantom gaps (`phantom.rs`) | 578 | Hybrid | no | Sensible (registry path is solid) |
| Pain signals (`pain.rs`) | 162 | Heuristic | no | Simple, the decay is nice |
| Interference (`interference.rs`) | 167 | Heuristic | no | Simple, O(n²), nice reordering trick |
| Stream monitor (`stream.rs`) | 205 | Heuristic | no | **Weakest** — brittle negation keywords |

## So: heuristics or actually good?

**Both, deliberately layered.** The genuinely-good parts are the **LLM judge
prompts and the write-time/entity-resolution logic that uses them** — those are
real intelligence (bounded only by your model) solving problems mem0 doesn't
attempt (temporal invalidation, logical-vs-temporal contradiction, alias
learning). The heuristic parts (interference, pain, phantom, speculative,
stream) are **classical, cheap, transparent rules** that run for free on every
token/turn and cover the cases where an LLM call isn't worth it.

### Strengths
- **Cost discipline.** LLM only fires for sim > 0.5 / novel aliases / unseen
  topics, and every answer is cached to disk. Write-time intelligence without a
  write-time fortune.
- **Graceful degradation.** Every LLM feature falls back to a heuristic on error
  or misconfig — the engine never hard-fails because the judge is down.
- **Typed verdicts.** `InferredAction` / `InvalidationVerdict` / etc. are
  exhaustive enums the storage layer pattern-matches on. No stringly-typed glue.
- **Honest separation** of "logical contradiction" vs "temporal supersession" —
  a distinction most memory tools blur.

### Weaknesses / honest caveats
- **Stream monitoring is brittle** (keyword negation). Treat alerts as hints,
  not truth.
- **Interference is O(n²)** per context assembly — won't scale to huge windows.
- **Pain/phantom use keyword/substring matching**, not semantics — expect
  recall/precision gaps on paraphrases.
- **No learned reranker or trained contradiction classifier** — the heuristics
  are hand-tuned thresholds, not fitted models.
- **Quality ceiling = your LLM.** The judge is only as smart as the model behind
  it; with a weak/local model, the "genuinely good" parts degrade toward the
  heuristic baseline.

### Bottom line
This is **not a toy or a coat of marketing paint over cosine similarity.** The
LLM-judge + write-inference + entity-resolution core is a thoughtfully-designed,
production-minded cognitive layer. The surrounding heuristics are honest,
cheap building blocks. Compared to a plain vector store (or mem0's additive-only
model), the write-time invalidation/contradiction handling is a real, working
advance — exactly the thing that keeps a long-lived agent's context window from
rotting.
