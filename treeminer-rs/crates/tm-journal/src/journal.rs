//! SQLite implementation of the durable find journal. Port of `src/journal/FindJournal.cpp`.

use std::path::Path;
use std::sync::Mutex;
use std::time::Duration;

use rusqlite::{Connection, OpenFlags, OptionalExtension, Row, TransactionBehavior};
use tm_core::{Classification, FindKind, FindRecord, FindStatus, FoundPayload};

use crate::error::{JournalError, Result};

/// Server-side limit on `hash_to_verify`; oversize payloads are rejected outright, so they
/// are journaled as `PermanentlyInvalid` rather than attempted.
pub const MAX_HASH_TO_VERIFY_LENGTH: usize = 150;

const SUPPORTED_SCHEMA_VERSION: i64 = 1;
const BUSY_TIMEOUT: Duration = Duration::from_millis(5000);

const RECORD_COLUMNS: &str = "id, key, hash_to_verify, account, kind, m, worker, attempts, \
     hashes_per_second, found_at, status, status_reason, attempt_count, next_attempt_at, \
     last_attempt_at, last_http_status, last_response, confirmed_at, xuni_windows_tried";

/// Startup recovery counts, logged at boot.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct RecoveryStats {
    pub pending: usize,
    pub accepted_unconfirmed: usize,
    pub parked_difficulty: usize,
    pub parked_xuni: usize,
    pub quarantined: usize,
    pub acked: usize,
    pub dead: usize,
    pub invalid: usize,
}

/// Counters for the stats endpoint and the terminal status line. `parked` aggregates both
/// parked states; `queued_*` is the per-kind count of work the submitter still owes the
/// server (Pending plus AcceptedUnconfirmed).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Counts {
    pub pending: usize,
    pub parked: usize,
    pub parked_difficulty: usize,
    pub parked_xuni: usize,
    pub quarantined: usize,
    pub acked_total: usize,
    pub dead_total: usize,
    pub accepted_unconfirmed: usize,
    pub permanently_invalid: usize,
    pub queued_xen11: usize,
    pub queued_xuni: usize,
}

/// The journal contract the submitter and dashboard code against. Port of `IFindJournal`.
///
/// There is no persisted claim/lease: the submitter leases work by fetching it and holding
/// [`FindStatus::Submitting`] in memory for the duration of one attempt. That state is
/// in-process only and is rejected by [`Journal::record_attempt`], so a crash mid-attempt
/// leaves the row exactly as it was — Pending and due — instead of stranded in a lease.
pub trait Journal {
    /// Durable on return (one fsync'd transaction). A duplicate `key` — the server's dedupe
    /// key — returns the existing row's id without overwriting the original capture.
    fn append(&self, payload: &FoundPayload) -> Result<i64>;

    /// Oldest-first Pending rows whose `next_attempt_at` is NULL or `<= now_utc`.
    fn fetch_eligible(&self, now_utc: &str, limit: usize) -> Result<Vec<FindRecord>>;

    /// As [`Journal::fetch_eligible`], restricted to one kind so neither kind can be starved
    /// out of a mixed LIMIT slice by the other.
    fn fetch_eligible_of_kind(
        &self,
        kind: FindKind,
        now_utc: &str,
        limit: usize,
    ) -> Result<Vec<FindRecord>>;

    /// Oldest-first AcceptedUnconfirmed rows that are due, so `/get_block` confirmation can
    /// be re-driven after a transient lookup failure.
    fn fetch_awaiting_confirmation(&self, now_utc: &str, limit: usize) -> Result<Vec<FindRecord>>;

    fn get_by_id(&self, id: i64) -> Result<Option<FindRecord>>;

    /// Persists one attempt outcome: status, reason, attempt bookkeeping, backoff time,
    /// HTTP status/response, and — for `Acked` — the first confirmation time.
    fn record_attempt(
        &self,
        id: i64,
        classification: &Classification,
        http_status: Option<i32>,
        response_body: &str,
        next_attempt_at: Option<&str>,
        now_utc: &str,
    ) -> Result<()>;

    /// ParkedDifficulty -> Pending for rows with `m >= current_difficulty` (the server check
    /// is strictly `submitted_m < difficulty`, so the boundary un-parks). Returns the count.
    fn unpark_for_difficulty(&self, current_difficulty: u32) -> Result<usize>;

    /// ParkedXuniWindow rows with budget left go to Pending with the window counter
    /// incremented; the rest go to Dead. Returns the number un-parked.
    fn unpark_xuni_for_window(&self, max_windows: i32) -> Result<usize>;

    /// Per-status counts at boot. Also resets any persisted `Submitting` leftover to
    /// Pending; persisted backoffs are deliberately kept, since a restart must not reset
    /// backoff.
    fn recover_on_startup(&self) -> Result<RecoveryStats>;

    fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> Result<()>;
    fn last_known_difficulty(&self) -> Result<Option<u32>>;
    fn counts(&self) -> Result<Counts>;
}

pub struct FindJournal {
    conn: Mutex<Connection>,
}

impl FindJournal {
    /// Opens (creating if absent) the database, applies the durability pragmas and creates
    /// or validates schema version 1.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let path = db_path.as_ref();
        let conn = Connection::open_with_flags(
            path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_CREATE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| JournalError::sqlite(format!("open '{}'", path.display()), e))?;

        apply_pragmas(&conn)?;
        ensure_schema(&conn)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// The mutex is only poisoned if a thread panicked mid-call; the connection itself is
    /// still usable and losing the journal is worse than any inconsistency a panic left, so
    /// the guard is taken regardless.
    fn lock(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn fetch_by_status(
        conn: &Connection,
        status: &'static str,
        kind: Option<&'static str>,
        now_utc: &str,
        limit: usize,
    ) -> Result<Vec<FindRecord>> {
        // Both interpolated fragments are compile-time literals, never caller input; keeping
        // them out of the bound parameters lets the (status, next_attempt_at, kind, id)
        // index drive the scan.
        let kind_clause = match kind {
            Some(k) => format!(" AND kind = '{k}'"),
            None => String::new(),
        };
        let sql = format!(
            "SELECT {RECORD_COLUMNS} FROM finds WHERE status = '{status}'{kind_clause} \
             AND (next_attempt_at IS NULL OR next_attempt_at <= ?1) ORDER BY id ASC LIMIT ?2;"
        );

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| JournalError::sqlite("prepare fetch", e))?;
        let mut rows = stmt
            .query(rusqlite::params![now_utc, limit as i64])
            .map_err(|e| JournalError::sqlite("query fetch", e))?;

        let mut records = Vec::new();
        while let Some(row) = rows
            .next()
            .map_err(|e| JournalError::sqlite("step fetch", e))?
        {
            records.push(row_to_record(row)?);
        }
        Ok(records)
    }
}

impl Drop for FindJournal {
    fn drop(&mut self) {
        // Orderly-shutdown checkpoint; durability never depends on it, so failures are
        // ignored (the WAL replays on the next open).
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.pragma_update(None, "wal_checkpoint", "PASSIVE");
        }
    }
}

fn apply_pragmas(conn: &Connection) -> Result<()> {
    conn.busy_timeout(BUSY_TIMEOUT)
        .map_err(|e| JournalError::sqlite("busy_timeout", e))?;

    // The pragma reports the mode actually in effect; anything but WAL voids the durability
    // model (e.g. an unsupported filesystem), so a fallback is fatal rather than silent.
    let mode: String = conn
        .query_row("PRAGMA journal_mode=WAL;", [], |row| row.get(0))
        .map_err(|e| JournalError::sqlite("PRAGMA journal_mode", e))?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(JournalError::Schema(
            "could not enable WAL journal mode at this path".into(),
        ));
    }

    conn.pragma_update(None, "synchronous", "FULL")
        .map_err(|e| JournalError::sqlite("PRAGMA synchronous", e))?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .map_err(|e| JournalError::sqlite("PRAGMA foreign_keys", e))?;
    Ok(())
}

fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "BEGIN IMMEDIATE;
         CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
         CREATE TABLE IF NOT EXISTS finds (
           id                 INTEGER PRIMARY KEY,
           key                TEXT NOT NULL UNIQUE,
           hash_to_verify     TEXT NOT NULL,
           account            TEXT NOT NULL,
           kind               TEXT NOT NULL CHECK(kind IN ('XEN11','XUNI')),
           m                  INTEGER NOT NULL,
           worker             TEXT,
           attempts           INTEGER,
           hashes_per_second  REAL,
           found_at           TEXT NOT NULL,
           status             TEXT NOT NULL,
           status_reason      TEXT,
           attempt_count      INTEGER NOT NULL DEFAULT 0,
           next_attempt_at    TEXT,
           last_attempt_at    TEXT,
           last_http_status   INTEGER,
           last_response      TEXT,
           confirmed_at       TEXT,
           xuni_windows_tried INTEGER NOT NULL DEFAULT 0
         );
         CREATE INDEX IF NOT EXISTS idx_finds_ready
           ON finds(status, next_attempt_at, kind, id);
         CREATE TABLE IF NOT EXISTS difficulty_seen (at TEXT, value INTEGER);
         COMMIT;",
    )
    .map_err(|e| JournalError::sqlite("create schema", e))?;

    let version: Option<i64> = conn
        .query_row("SELECT MAX(version) FROM schema_version;", [], |row| {
            row.get(0)
        })
        .map_err(|e| JournalError::sqlite("read schema_version", e))?;

    match version {
        None => {
            conn.execute(
                "INSERT INTO schema_version(version) VALUES (?1);",
                [SUPPORTED_SCHEMA_VERSION],
            )
            .map_err(|e| JournalError::sqlite("write schema_version", e))?;
        }
        Some(v) if v != SUPPORTED_SCHEMA_VERSION => {
            return Err(JournalError::Schema(format!(
                "unsupported schema_version {v} (supported: {SUPPORTED_SCHEMA_VERSION})"
            )));
        }
        Some(_) => {}
    }
    Ok(())
}

fn status_from_str(text: &str) -> Result<FindStatus> {
    FindStatus::parse(text)
        .ok_or_else(|| JournalError::Schema(format!("unknown status string in database: '{text}'")))
}

fn kind_from_str(text: &str) -> Result<FindKind> {
    FindKind::parse(text)
        .ok_or_else(|| JournalError::Schema(format!("unknown kind string in database: '{text}'")))
}

fn row_to_record(row: &Row<'_>) -> Result<FindRecord> {
    let get = |idx: usize| -> Result<String> {
        let value: Option<String> = row
            .get(idx)
            .map_err(|e| JournalError::sqlite("read column", e))?;
        Ok(value.unwrap_or_default())
    };
    let get_opt = |idx: usize| -> Result<Option<String>> {
        row.get(idx)
            .map_err(|e| JournalError::sqlite("read column", e))
    };
    let get_i64 = |idx: usize| -> Result<i64> {
        let value: Option<i64> = row
            .get(idx)
            .map_err(|e| JournalError::sqlite("read column", e))?;
        Ok(value.unwrap_or_default())
    };

    let payload = FoundPayload {
        key: get(1)?,
        hash_to_verify: get(2)?,
        account: get(3)?,
        kind: kind_from_str(&get(4)?)?,
        memory_cost: get_i64(5)? as u32,
        worker: get(6)?,
        attempts: get_i64(7)? as u64,
        hashes_per_second: row
            .get::<_, Option<f64>>(8)
            .map_err(|e| JournalError::sqlite("read column", e))?
            .unwrap_or_default(),
        found_at_utc: get(9)?,
    };

    Ok(FindRecord {
        id: get_i64(0)?,
        payload,
        status: status_from_str(&get(10)?)?,
        status_reason: get(11)?,
        attempt_count: get_i64(12)? as i32,
        next_attempt_at: get_opt(13)?,
        last_attempt_at: get_opt(14)?,
        last_http_status: row
            .get::<_, Option<i64>>(15)
            .map_err(|e| JournalError::sqlite("read column", e))?
            .map(|v| v as i32),
        last_response: get(16)?,
        confirmed_at: get_opt(17)?,
        xuni_windows_tried: get_i64(18)? as i32,
    })
}

/// Empty strings are stored as NULL for the nullable text columns, matching the C++ binder.
fn text_or_null(value: &str) -> Option<&str> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

impl Journal for FindJournal {
    fn append(&self, payload: &FoundPayload) -> Result<i64> {
        // Never throw away a find: payloads the server is known to reject are journaled
        // anyway, straight into PermanentlyInvalid with the reason recorded.
        let (status, reason) = if payload.hash_to_verify.len() > MAX_HASH_TO_VERIFY_LENGTH {
            (
                FindStatus::PermanentlyInvalid,
                format!(
                    "hash_to_verify length {} exceeds server limit of {}",
                    payload.hash_to_verify.len(),
                    MAX_HASH_TO_VERIFY_LENGTH
                ),
            )
        } else if payload.account.is_empty() {
            (
                FindStatus::PermanentlyInvalid,
                "account is empty".to_string(),
            )
        } else {
            (FindStatus::Pending, String::new())
        };

        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| JournalError::sqlite("BEGIN IMMEDIATE", e))?;

        let changed = tx
            .execute(
                "INSERT INTO finds (key, hash_to_verify, account, kind, m, worker, attempts,
                                    hashes_per_second, found_at, status, status_reason)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
                 ON CONFLICT(key) DO NOTHING;",
                rusqlite::params![
                    payload.key,
                    payload.hash_to_verify,
                    payload.account,
                    payload.kind.as_str(),
                    payload.memory_cost as i64,
                    text_or_null(&payload.worker),
                    payload.attempts as i64,
                    payload.hashes_per_second,
                    payload.found_at_utc,
                    status.as_str(),
                    text_or_null(&reason),
                ],
            )
            .map_err(|e| JournalError::sqlite("insert find", e))?;

        let id = if changed == 0 {
            // Duplicate local capture: idempotently return the existing row's id.
            tx.query_row(
                "SELECT id FROM finds WHERE key = ?1;",
                [&payload.key],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| JournalError::sqlite("select existing find", e))?
        } else {
            tx.last_insert_rowid()
        };

        tx.commit()
            .map_err(|e| JournalError::sqlite("COMMIT append", e))?;
        // Durable on return: synchronous=FULL fsyncs the WAL as part of the COMMIT above.
        Ok(id)
    }

    fn fetch_eligible(&self, now_utc: &str, limit: usize) -> Result<Vec<FindRecord>> {
        let conn = self.lock();
        Self::fetch_by_status(&conn, "Pending", None, now_utc, limit)
    }

    fn fetch_eligible_of_kind(
        &self,
        kind: FindKind,
        now_utc: &str,
        limit: usize,
    ) -> Result<Vec<FindRecord>> {
        let conn = self.lock();
        Self::fetch_by_status(&conn, "Pending", Some(kind.as_str()), now_utc, limit)
    }

    fn fetch_awaiting_confirmation(&self, now_utc: &str, limit: usize) -> Result<Vec<FindRecord>> {
        let conn = self.lock();
        Self::fetch_by_status(&conn, "AcceptedUnconfirmed", None, now_utc, limit)
    }

    fn get_by_id(&self, id: i64) -> Result<Option<FindRecord>> {
        let conn = self.lock();
        let sql = format!("SELECT {RECORD_COLUMNS} FROM finds WHERE id = ?1;");
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| JournalError::sqlite("prepare getById", e))?;
        let mut rows = stmt
            .query([id])
            .map_err(|e| JournalError::sqlite("query getById", e))?;
        match rows
            .next()
            .map_err(|e| JournalError::sqlite("step getById", e))?
        {
            Some(row) => Ok(Some(row_to_record(row)?)),
            None => Ok(None),
        }
    }

    fn record_attempt(
        &self,
        id: i64,
        classification: &Classification,
        http_status: Option<i32>,
        response_body: &str,
        next_attempt_at: Option<&str>,
        now_utc: &str,
    ) -> Result<()> {
        if classification.next_status == FindStatus::Submitting {
            return Err(JournalError::Contract(
                "record_attempt with status Submitting — that state is in-process only and \
                 must never be persisted"
                    .into(),
            ));
        }

        let conn = self.lock();
        let changed = conn
            .execute(
                "UPDATE finds SET
                   status = ?1,
                   status_reason = ?2,
                   attempt_count = attempt_count + 1,
                   last_attempt_at = ?3,
                   last_http_status = ?4,
                   last_response = ?5,
                   next_attempt_at = ?6,
                   confirmed_at = CASE WHEN ?1 = 'Acked'
                                       THEN COALESCE(confirmed_at, ?3) ELSE confirmed_at END
                 WHERE id = ?7;",
                rusqlite::params![
                    classification.next_status.as_str(),
                    text_or_null(&classification.reason),
                    now_utc,
                    http_status,
                    response_body,
                    next_attempt_at,
                    id,
                ],
            )
            .map_err(|e| JournalError::sqlite("record attempt", e))?;

        if changed == 0 {
            return Err(JournalError::Contract(format!(
                "record_attempt for unknown find id {id}"
            )));
        }
        Ok(())
    }

    fn unpark_for_difficulty(&self, current_difficulty: u32) -> Result<usize> {
        let conn = self.lock();
        // The server rejects strictly `m < difficulty`, so the boundary m == difficulty
        // un-parks. Clearing next_attempt_at makes the row due immediately.
        let changed = conn
            .execute(
                "UPDATE finds SET status = 'Pending', next_attempt_at = NULL
                 WHERE status = 'ParkedDifficulty' AND m >= ?1;",
                [current_difficulty as i64],
            )
            .map_err(|e| JournalError::sqlite("unpark for difficulty", e))?;
        Ok(changed)
    }

    fn unpark_xuni_for_window(&self, max_windows: i32) -> Result<usize> {
        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| JournalError::sqlite("BEGIN IMMEDIATE", e))?;

        tx.execute(
            "UPDATE finds SET status = 'Dead',
               status_reason = 'xuni window budget exhausted',
               next_attempt_at = NULL
             WHERE status = 'ParkedXuniWindow' AND xuni_windows_tried >= ?1;",
            [max_windows as i64],
        )
        .map_err(|e| JournalError::sqlite("exhaust xuni budget", e))?;

        // Whatever is still parked has budget left, by construction of the statement above.
        let unparked = tx
            .execute(
                "UPDATE finds SET status = 'Pending',
                   xuni_windows_tried = xuni_windows_tried + 1,
                   next_attempt_at = NULL
                 WHERE status = 'ParkedXuniWindow';",
                [],
            )
            .map_err(|e| JournalError::sqlite("unpark xuni", e))?;

        tx.commit()
            .map_err(|e| JournalError::sqlite("COMMIT unpark xuni", e))?;
        Ok(unparked)
    }

    fn recover_on_startup(&self) -> Result<RecoveryStats> {
        let mut conn = self.lock();
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| JournalError::sqlite("BEGIN IMMEDIATE", e))?;

        // 'Submitting' is never persisted, so this should always update zero rows; it exists
        // so a bug elsewhere can never strand a find. Persisted next_attempt_at is kept:
        // restarts must not reset backoff.
        tx.execute(
            "UPDATE finds SET status = 'Pending' WHERE status = 'Submitting';",
            [],
        )
        .map_err(|e| JournalError::sqlite("recover Submitting leftovers", e))?;

        let mut stats = RecoveryStats::default();
        {
            let mut stmt = tx
                .prepare("SELECT status, COUNT(*) FROM finds GROUP BY status;")
                .map_err(|e| JournalError::sqlite("prepare recovery counts", e))?;
            let mut rows = stmt
                .query([])
                .map_err(|e| JournalError::sqlite("query recovery counts", e))?;
            while let Some(row) = rows
                .next()
                .map_err(|e| JournalError::sqlite("step recovery counts", e))?
            {
                let status: String = row
                    .get(0)
                    .map_err(|e| JournalError::sqlite("read status", e))?;
                let count: i64 = row
                    .get(1)
                    .map_err(|e| JournalError::sqlite("read count", e))?;
                let count = count as usize;
                match status_from_str(&status)? {
                    FindStatus::Pending => stats.pending += count,
                    FindStatus::AcceptedUnconfirmed => stats.accepted_unconfirmed += count,
                    FindStatus::ParkedDifficulty => stats.parked_difficulty += count,
                    FindStatus::ParkedXuniWindow => stats.parked_xuni += count,
                    FindStatus::Quarantined => stats.quarantined += count,
                    FindStatus::Acked => stats.acked += count,
                    FindStatus::Dead => stats.dead += count,
                    FindStatus::PermanentlyInvalid => stats.invalid += count,
                    FindStatus::Submitting => {} // unreachable: reset above
                }
            }
        }

        tx.commit()
            .map_err(|e| JournalError::sqlite("COMMIT recovery", e))?;
        Ok(stats)
    }

    fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> Result<()> {
        let conn = self.lock();
        conn.execute(
            "INSERT INTO difficulty_seen (at, value) VALUES (?1, ?2);",
            rusqlite::params![at_utc, difficulty as i64],
        )
        .map_err(|e| JournalError::sqlite("record difficulty", e))?;
        Ok(())
    }

    fn last_known_difficulty(&self) -> Result<Option<u32>> {
        let conn = self.lock();
        // Most recently recorded observation (insertion order), independent of timestamp
        // formatting — the journal never interprets caller-provided time strings.
        let value: Option<Option<i64>> = conn
            .query_row(
                "SELECT value FROM difficulty_seen ORDER BY rowid DESC LIMIT 1;",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| JournalError::sqlite("last known difficulty", e))?;
        Ok(value.flatten().map(|v| v as u32))
    }

    fn counts(&self) -> Result<Counts> {
        let conn = self.lock();
        let mut stmt = conn
            .prepare("SELECT status, kind, COUNT(*) FROM finds GROUP BY status, kind;")
            .map_err(|e| JournalError::sqlite("prepare counts", e))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| JournalError::sqlite("query counts", e))?;

        let mut result = Counts::default();
        while let Some(row) = rows
            .next()
            .map_err(|e| JournalError::sqlite("step counts", e))?
        {
            let status = status_from_str(
                &row.get::<_, String>(0)
                    .map_err(|e| JournalError::sqlite("read status", e))?,
            )?;
            let kind = kind_from_str(
                &row.get::<_, String>(1)
                    .map_err(|e| JournalError::sqlite("read kind", e))?,
            )?;
            let count =
                row.get::<_, i64>(2)
                    .map_err(|e| JournalError::sqlite("read count", e))? as usize;

            match status {
                FindStatus::Pending => result.pending += count,
                FindStatus::ParkedDifficulty => {
                    result.parked += count;
                    result.parked_difficulty += count;
                }
                FindStatus::ParkedXuniWindow => {
                    result.parked += count;
                    result.parked_xuni += count;
                }
                FindStatus::Quarantined => result.quarantined += count,
                FindStatus::Acked => result.acked_total += count,
                FindStatus::Dead => result.dead_total += count,
                FindStatus::AcceptedUnconfirmed => result.accepted_unconfirmed += count,
                FindStatus::PermanentlyInvalid => result.permanently_invalid += count,
                FindStatus::Submitting => {} // never persisted
            }

            if matches!(
                status,
                FindStatus::Pending | FindStatus::AcceptedUnconfirmed
            ) {
                match kind {
                    FindKind::Xen11 => result.queued_xen11 += count,
                    FindKind::Xuni => result.queued_xuni += count,
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests;
