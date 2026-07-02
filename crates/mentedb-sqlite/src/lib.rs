//! SQLite + sqlite-vec storage backend for MenteDB.
//!
//! This crate replaces the bespoke page/WAL/HNSW/CSR storage engine with a
//! single SQLite database file:
//!
//!   * `memories`    — one row per `MemoryNode`, with ordinary B-tree indexes
//!                     on the columns used for filtering (`agent_id`, `space_id`,
//!                     `memory_type`, `salience`, `created_at`).
//!   * `memory_tags` — a junction table replacing the roaring-bitmap tag index
//!                     (`WHERE tag IN (...)`).
//!   * `memory_vec`  — a `sqlite-vec` `vec0` virtual table for brute-force
//!                     exact KNN over embeddings. It is a *derived* index: the
//!                     source-of-truth bytes live in `memories.embedding`, and
//!                     the vec0 table is keyed by the implicit `memories.rowid`
//!                     so it can always be rebuilt from the source rows.
//!   * `schema_meta` — key/value row holding the schema version and the
//!                     embedding dimension the vec0 table was created with.
//!
//! All access goes through a single `parking_lot::Mutex<Connection>`. SQLite's
//! own WAL handles crash recovery; the custom WAL/page-manager/buffer-pool are
//! gone. This is the substrate the facade will eventually talk to instead of
//! `StorageEngine` + `IndexManager` + `GraphManager`.

use std::path::Path;

use mentedb_core::memory::MemoryType;
use mentedb_core::types::{MemoryId, SpaceId, AgentId};
use mentedb_core::{MemoryNode, MenteError};
use parking_lot::Mutex;
use rusqlite::{params, Connection, OptionalExtension};

/// Map any error into `MenteError::Storage` with a human-readable message.
/// We cannot `impl From<rusqlite::Error> for MenteError` here (both are foreign
/// types, blocked by the orphan rule), so callers use `.map_err(store_err)?`.
fn store_err<E: std::fmt::Display>(e: E) -> MenteError {
    MenteError::Storage(e.to_string())
}

/// Register the `vec0` virtual table module as a SQLite auto-extension.
///
/// `sqlite-vec` 0.1.x ships only the raw C init symbol (`sqlite3_vec_init`);
/// there is no `load()` helper. We wire it through SQLite's
/// `sqlite3_auto_extension`, which is process-global and applies to every
/// connection opened *after* this call. Guarded by `Once` so repeated opens are
/// no-ops. Must run before any `Connection::open*` in this backend.
fn ensure_vec0_registered() {
    use rusqlite::ffi::sqlite3_auto_extension;
    use std::sync::Once;
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is the documented entry point of the
        // sqlite-vec extension and has the `sqlite3_loadext_entry` signature
        // SQLite expects. The transmute mirrors the crate's own test.
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

/// Serialize an `f32` embedding into the little-endian byte blob `sqlite-vec`
/// expects for `vec0` inserts and MATCH queries.
fn embedding_to_blob(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice::<f32, u8>(v).to_vec()
}

/// Inverse of [`embedding_to_blob`].
fn blob_to_embedding(b: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice::<u8, f32>(b).to_vec()
}

/// Return a unit-length copy of `v` (L2-normalized). We store the normalized
/// form in the `vec0` table so that L2 distance there is monotonically
/// equivalent to cosine distance, and we can recover cosine similarity from the
/// raw L2 distance without depending on the `distance_metric=` vec0 option.
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return v.to_vec();
    }
    v.iter().map(|x| x / norm).collect()
}

/// Convert L2 squared-distance on unit vectors into cosine similarity in
/// `[0, 1]`. Derivation: for unit vectors `a`, `b`, `|a-b|^2 = 2 - 2·(a·b)`,
/// and `a·b` is cosine similarity, so `sim = 1 - dist^2 / 2`. Clamped because
/// floating point can drift slightly outside the theoretical range.
fn l2_distance_to_similarity(l2_distance: f64) -> f32 {
    let sim = 1.0 - (l2_distance * l2_distance) / 2.0;
    sim.clamp(0.0, 1.0) as f32
}

// --- enum <-> string helpers (SQLite stores these as TEXT) -----------------

fn memory_type_str(t: MemoryType) -> &'static str {
    match t {
        MemoryType::Episodic => "Episodic",
        MemoryType::Semantic => "Semantic",
        MemoryType::Procedural => "Procedural",
        MemoryType::AntiPattern => "AntiPattern",
        MemoryType::Reasoning => "Reasoning",
        MemoryType::Correction => "Correction",
    }
}

fn parse_memory_type(s: &str) -> MemoryType {
    match s {
        "Semantic" => MemoryType::Semantic,
        "Procedural" => MemoryType::Procedural,
        "AntiPattern" => MemoryType::AntiPattern,
        "Reasoning" => MemoryType::Reasoning,
        "Correction" => MemoryType::Correction,
        _ => MemoryType::Episodic,
    }
}

/// Parse a stored TEXT id back into its newtype. The id newtypes wrap `Uuid`
/// and implement `FromStr`; on a malformed value we surface a `Storage` error.
fn parse_id<I: std::str::FromStr>(s: &str) -> Result<I, MenteError>
where
    I::Err: std::fmt::Display,
{
    s.parse::<I>().map_err(store_err)
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// SQLite-backed implementation of MenteDB's storage + vector index.
///
/// Holds a single connection guarded by a mutex. WAL mode is enabled on file
/// opens so readers don't block the (single) writer. The connection keeps the
/// `sqlite-vec` extension loaded for the lifetime of the backend.
pub struct Backend {
    conn: Mutex<Connection>,
    /// The dimension the `memory_vec` vec0 table was created with. Embeddings
    /// of a different length are rejected by [`Backend::store_memory`].
    embedding_dim: usize,
}

impl Backend {
    /// Open (or create) a file-backed database at `path`.
    pub fn open(path: &Path, embedding_dim: usize) -> Result<Self, MenteError> {
        // Ensure the parent directory exists so a fresh path works.
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(store_err)?;
            }
        }
        ensure_vec0_registered();
        let mut conn = Connection::open(path).map_err(store_err)?;
        // WAL: concurrent readers don't block the writer, and crash recovery is
        // handled by SQLite instead of the old custom WAL.
        conn.pragma_update(None, "journal_mode", "WAL").map_err(store_err)?;
        Self::init(&mut conn, embedding_dim)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedding_dim,
        })
    }

    /// Open an ephemeral in-memory database (used by tests and quick spikes).
    pub fn open_in_memory(embedding_dim: usize) -> Result<Self, MenteError> {
        ensure_vec0_registered();
        let mut conn = Connection::open_in_memory().map_err(store_err)?;
        Self::init(&mut conn, embedding_dim)?;
        Ok(Self {
            conn: Mutex::new(conn),
            embedding_dim,
        })
    }

    /// Create all tables/indexes and load the sqlite-vec extension. Idempotent.
    fn init(conn: &mut Connection, embedding_dim: usize) -> Result<(), MenteError> {
        // The vec0 module was registered process-globally by
        // `ensure_vec0_registered()` before this connection was opened, so the
        // `memory_vec` virtual table below will resolve automatically.
        let tx = conn.transaction().map_err(store_err)?;

        tx.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS schema_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
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
            CREATE INDEX IF NOT EXISTS idx_mem_agent     ON memories(agent_id);
            CREATE INDEX IF NOT EXISTS idx_mem_space     ON memories(space_id);
            CREATE INDEX IF NOT EXISTS idx_mem_type      ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_mem_salience  ON memories(salience DESC);
            CREATE INDEX IF NOT EXISTS idx_mem_created   ON memories(created_at DESC);

            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT NOT NULL,
                tag       TEXT NOT NULL,
                PRIMARY KEY (memory_id, tag)
            );
            CREATE INDEX IF NOT EXISTS idx_tag ON memory_tags(tag);
            "#,
        )
        .map_err(store_err)?;

        // The vec0 dimension is fixed at table-creation time. We record the dim
        // in schema_meta so a later open with a different embedder can detect
        // the mismatch and rebuild this table (handled in a follow-up commit).
        tx.execute(
            &format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(embedding float[{embedding_dim}])"
            ),
            [],
        )
        .map_err(store_err)?;

        tx.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('embedding_dim', ?1)",
            params![embedding_dim.to_string()],
        )
        .map_err(store_err)?;

        tx.commit().map_err(store_err)?;
        Ok(())
    }

    /// The embedding dimension the vector index was created with.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Persist (or update) a memory and refresh its vector + tag rows.
    ///
    /// The original embedding is stored verbatim in `memories.embedding`
    /// (returned unchanged by [`Self::get_memory`]); a normalized copy goes
    /// into the `vec0` table so KNN behaves as cosine similarity.
    pub fn store_memory(&self, node: &MemoryNode) -> Result<(), MenteError> {
        if !node.embedding.is_empty() && node.embedding.len() != self.embedding_dim {
            return Err(MenteError::EmbeddingDimensionMismatch {
                got: node.embedding.len(),
                expected: self.embedding_dim,
            });
        }

        let conn = self.conn.lock();
        let attrs_json = serde_json::to_string(&node.attributes).map_err(store_err)?;
        let emb_blob = if node.embedding.is_empty() {
            None
        } else {
            Some(embedding_to_blob(&node.embedding))
        };

        let id_str = node.id.to_string();

        // Upsert the memory row. `ON CONFLICT(id) DO UPDATE` preserves the
        // implicit rowid, which keeps the vec0 key stable across re-stores.
        conn.execute(
            r#"
            INSERT INTO memories (
                id, agent_id, space_id, memory_type, content, embedding,
                created_at, accessed_at, access_count, salience, confidence,
                attributes, valid_from, valid_until
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14
            )
            ON CONFLICT(id) DO UPDATE SET
                agent_id     = excluded.agent_id,
                space_id     = excluded.space_id,
                memory_type  = excluded.memory_type,
                content      = excluded.content,
                embedding    = excluded.embedding,
                accessed_at  = excluded.accessed_at,
                access_count = excluded.access_count,
                salience     = excluded.salience,
                confidence   = excluded.confidence,
                attributes   = excluded.attributes,
                valid_from   = excluded.valid_from,
                valid_until  = excluded.valid_until
            "#,
            params![
                id_str,
                node.agent_id.to_string(),
                node.space_id.to_string(),
                memory_type_str(node.memory_type),
                node.content,
                emb_blob.as_deref(),
                node.created_at as i64,
                node.accessed_at as i64,
                node.access_count as i64,
                node.salience as f64,
                node.confidence as f64,
                attrs_json,
                node.valid_from.map(|t| t as i64),
                node.valid_until.map(|t| t as i64),
            ],
        )
        .map_err(store_err)?;

        // Refresh the derived vector row. vec0 `REPLACE` semantics are version
        // dependent, so delete-then-insert is the portable choice.
        let rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![id_str],
                |r| r.get(0),
            )
            .map_err(store_err)?;
        let _ = conn.execute(
            "DELETE FROM memory_vec WHERE rowid = ?1",
            params![rowid],
        );
        if !node.embedding.is_empty() {
            // Store a normalized copy so L2 KNN approximates cosine ranking.
            let norm = normalize(&node.embedding);
            let norm_blob = embedding_to_blob(&norm);
            let _ = &emb_blob; // original bytes already stored in memories.embedding
            conn.execute(
                "INSERT INTO memory_vec (rowid, embedding) VALUES (?1, ?2)",
                params![rowid, norm_blob.as_slice()],
            )
            .map_err(store_err)?;
        }

        // Refresh tags (delete + reinsert is simplest and correct).
        conn.execute(
            "DELETE FROM memory_tags WHERE memory_id = ?1",
            params![id_str],
        )
        .map_err(store_err)?;
        for tag in &node.tags {
            conn.execute(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?1, ?2)",
                params![id_str, tag],
            )
            .map_err(store_err)?;
        }

        Ok(())
    }

    /// Load a memory by id, including its tags. Returns `None` if absent.
    pub fn get_memory(&self, id: MemoryId) -> Result<Option<MemoryNode>, MenteError> {
        let conn = self.conn.lock();
        let id_str = id.to_string();
        let node: Option<MemoryNode> = conn
            .query_row(
                r#"
                SELECT id, agent_id, space_id, memory_type, content, embedding,
                       created_at, accessed_at, access_count, salience, confidence,
                       attributes, valid_from, valid_until
                FROM memories WHERE id = ?1
                "#,
                params![id_str],
                |row| {
                    let emb_blob: Option<Vec<u8>> = row.get(5)?;
                    Ok(MemoryNode {
                        id: parse_id(&row.get::<_, String>(0)?).unwrap_or_else(|_| MemoryId::nil()),
                        agent_id: parse_id(&row.get::<_, String>(1)?).unwrap_or_else(|_| AgentId::nil()),
                        space_id: parse_id(&row.get::<_, String>(2)?).unwrap_or_else(|_| SpaceId::nil()),
                        memory_type: parse_memory_type(&row.get::<_, String>(3)?),
                        content: row.get(4)?,
                        embedding: emb_blob.as_deref().map(blob_to_embedding).unwrap_or_default(),
                        created_at: row.get::<_, i64>(6)? as u64,
                        accessed_at: row.get::<_, i64>(7)? as u64,
                        access_count: row.get::<_, i64>(8)? as u32,
                        salience: row.get::<_, f64>(9)? as f32,
                        confidence: row.get::<_, f64>(10)? as f32,
                        attributes: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                        valid_from: row.get::<_, Option<i64>>(12)?.map(|t| t as u64),
                        valid_until: row.get::<_, Option<i64>>(13)?.map(|t| t as u64),
                        tags: Vec::new(),
                    })
                },
            )
            .optional()
            .map_err(store_err)?;

        let mut node = match node {
            Some(n) => n,
            None => return Ok(None),
        };

        // Hydrate tags.
        let mut stmt = conn
            .prepare("SELECT tag FROM memory_tags WHERE memory_id = ?1 ORDER BY tag")
            .map_err(store_err)?;
        let tags: Vec<String> = stmt
            .query_map(params![id_str], |r| r.get::<_, String>(0))
            .map_err(store_err)?
            .filter_map(Result::ok)
            .collect();
        node.tags = tags;
        Ok(Some(node))
    }

    /// Delete a memory, its vector row, and its tags. Returns `true` if a row
    /// was removed.
    pub fn delete_memory(&self, id: MemoryId) -> Result<bool, MenteError> {
        let conn = self.conn.lock();
        let id_str = id.to_string();
        let rowid: Option<i64> = conn
            .query_row(
                "SELECT rowid FROM memories WHERE id = ?1",
                params![id_str],
                |r| r.get(0),
            )
            .optional()
            .map_err(store_err)?;
        let Some(rowid) = rowid else {
            return Ok(false);
        };
        conn.execute("DELETE FROM memories WHERE id = ?1", params![id_str])
            .map_err(store_err)?;
        let _ = conn.execute("DELETE FROM memory_vec WHERE rowid = ?1", params![rowid]);
        conn.execute("DELETE FROM memory_tags WHERE memory_id = ?1", params![id_str])
            .map_err(store_err)?;
        Ok(true)
    }

    /// Total number of stored memories.
    pub fn count(&self) -> Result<usize, MenteError> {
        let conn = self.conn.lock();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .map_err(store_err)?;
        Ok(n as usize)
    }

    /// Brute-force exact KNN over the embedding index.
    ///
    /// Returns `(MemoryId, cosine_similarity)` pairs, most similar first. The
    /// query vector is normalized internally; `similarity` is in `[0, 1]`.
    pub fn knn(&self, query: &[f32], k: usize) -> Result<Vec<(MemoryId, f32)>, MenteError> {
        if k == 0 || query.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self.conn.lock();
        let norm = normalize(query);
        let blob = embedding_to_blob(&norm);

        // Two steps so the join can't perturb the KNN plan:
        //   1. rowid + distance from vec0
        //   2. map rowid -> memory id
        let mut stmt = conn
            .prepare(
                "SELECT rowid, distance FROM memory_vec
                 WHERE embedding MATCH ?1
                 ORDER BY distance
                 LIMIT ?2",
            )
            .map_err(store_err)?;
        let hits: Vec<(i64, f64)> = stmt
            .query_map(params![blob.as_slice(), k as i64], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?))
            })
            .map_err(store_err)?
            .filter_map(Result::ok)
            .collect();

        let mut out = Vec::with_capacity(hits.len());
        for (rowid, dist) in hits {
            let id_str: Option<String> = conn
                .query_row(
                    "SELECT id FROM memories WHERE rowid = ?1",
                    params![rowid],
                    |r| r.get::<_, String>(0),
                )
                .optional()
                .map_err(store_err)?;
            if let Some(id_str) = id_str {
                if let Ok(id) = parse_id(&id_str) {
                    out.push((id, l2_distance_to_similarity(dist)));
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mentedb_core::types::{AgentId, MemoryId};

    fn make_node(id: MemoryId, content: &str, embedding: Vec<f32>) -> MemoryNode {
        let mut n = MemoryNode::new(AgentId::new(), MemoryType::Semantic, content.to_string(), embedding);
        n.id = id;
        n.tags = vec!["pref".to_string(), "ui".to_string()];
        n.salience = 0.8;
        n
    }

    #[test]
    fn store_and_get_roundtrips() {
        let db = Backend::open_in_memory(3).unwrap();
        let id = MemoryId::new();
        let node = make_node(id, "User prefers dark mode", vec![0.1, 0.2, 0.3]);
        db.store_memory(&node).unwrap();

        let loaded = db.get_memory(id).unwrap().expect("memory should exist");
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.content, "User prefers dark mode");
        assert_eq!(loaded.embedding, vec![0.1, 0.2, 0.3]); // original preserved
        assert_eq!(loaded.memory_type, MemoryType::Semantic);
        assert_eq!(loaded.salience, 0.8);
        assert_eq!(loaded.tags, vec!["pref".to_string(), "ui".to_string()]);
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn store_upserts_on_same_id() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        db.store_memory(&make_node(id, "first", vec![1.0, 0.0])).unwrap();
        db.store_memory(&make_node(id, "second", vec![0.0, 1.0])).unwrap();
        assert_eq!(db.count().unwrap(), 1);
        let loaded = db.get_memory(id).unwrap().unwrap();
        assert_eq!(loaded.content, "second");
        assert_eq!(loaded.embedding, vec![0.0, 1.0]);
    }

    #[test]
    fn delete_removes_memory_and_vector() {
        let db = Backend::open_in_memory(2).unwrap();
        let id = MemoryId::new();
        db.store_memory(&make_node(id, "gone", vec![1.0, 0.0])).unwrap();
        assert!(db.delete_memory(id).unwrap());
        assert!(db.get_memory(id).unwrap().is_none());
        assert!(db.knn(&[1.0, 0.0], 5).unwrap().is_empty());
    }

    #[test]
    fn knn_returns_nearest_first() {
        let db = Backend::open_in_memory(2).unwrap();
        let a = MemoryId::new();
        let b = MemoryId::new();
        let c = MemoryId::new();
        // Three points on the unit circle.
        db.store_memory(&make_node(a, "east",  vec![1.0, 0.0])).unwrap();
        db.store_memory(&make_node(b, "north", vec![0.0, 1.0])).unwrap();
        db.store_memory(&make_node(c, "sw",    vec![-0.7, -0.7])).unwrap();

        // Query near "east": a must rank first, c last.
        let hits = db.knn(&[0.9, 0.1], 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].0, a, "east should be nearest");
        assert!(hits[0].1 > 0.95, "east similarity should be ~1, got {}", hits[0].1);
        assert_eq!(hits[2].0, c, "sw should be farthest");
        assert!(hits[2].1 < hits[0].1);
    }

    #[test]
    fn rejects_mismatched_embedding_dimension() {
        let db = Backend::open_in_memory(3).unwrap();
        let id = MemoryId::new();
        let res = db.store_memory(&make_node(id, "bad", vec![0.1, 0.2]));
        assert!(matches!(
            res,
            Err(MenteError::EmbeddingDimensionMismatch { got: 2, expected: 3 })
        ));
    }

    #[test]
    fn open_file_backed_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mem.sqlite");
        let id = MemoryId::new();
        {
            let db = Backend::open(&path, 2).unwrap();
            db.store_memory(&make_node(id, "persisted", vec![1.0, 0.0])).unwrap();
        }
        let db = Backend::open(&path, 2).unwrap();
        let loaded = db.get_memory(id).unwrap().expect("should survive reopen");
        assert_eq!(loaded.content, "persisted");
    }
}
