//! Durable find journal — port of `src/journal/` (`FindJournal`, `FallbackSink`) and the
//! `IFindJournal` contract.
//!
//! Journal-first invariant: [`Journal::append`] returns only after the find is crash-safe on
//! disk. The database is opened with `journal_mode=WAL` and `synchronous=FULL`, so the
//! COMMIT that ends `append` has fsync'd the WAL before control comes back. A WAL checkpoint
//! is *not* part of that guarantee.
//!
//! Concurrency: [`FindJournal`] takes an internal mutex on every call, so one instance is
//! shared by the mining and submission threads. Use exactly one instance per process with a
//! stable database path, and never place the database on NFS/shared storage.
//!
//! Time: every timestamp is a caller-provided ISO-8601 UTC string. The journal never reads a
//! clock, which keeps all of its paths deterministic under test.

mod error;
mod fallback;
mod journal;

pub use error::{JournalError, Result};
pub use fallback::{FallbackSink, ImportStats};
pub use journal::{Counts, FindJournal, Journal, RecoveryStats, MAX_HASH_TO_VERIFY_LENGTH};
