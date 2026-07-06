//! Compatibility storage facade backed by SQLite.
//!
//! Older callers may still depend on the `mentedb-storage` crate name. The
//! maintained storage implementation now lives in `mentedb-sqlite`, so this
//! crate is intentionally a thin wrapper over that backend instead of a custom
//! page manager, buffer pool, and WAL.

pub mod engine;

pub use engine::{StorageEngine, StorageId};
