//! Last-resort append-only capture for finds the SQLite journal refused. Port of
//! `src/journal/FallbackSink.cpp`.
//!
//! # Why this exists
//!
//! [`crate::FindJournal::append`] returns an error on any SQLite failure. Without this sink
//! the miner's submit path would log that error and *drop* the find — the exact failure
//! class this project was built to eliminate, since the Argon2 key that produced a find is
//! not reproducible on demand.
//!
//! # What it covers, and what it does not
//!
//! The sink writes to a file on the same disk as the database, so it does not survive
//! total-disk failure. It is a second *mechanism*, not a second copy: it covers the disjoint
//! set of failures where SQLite specifically fails while a dumb `O_APPEND` write still
//! succeeds — `SQLITE_BUSY` past the 5 s timeout, a stale POSIX lock from a crashed process,
//! WAL corruption, `SQLITE_FULL` from a condition a few hundred bytes still slip under, or
//! an fsync failing on the WAL. It does not cover a dead disk, a read-only filesystem, or an
//! unwritable directory; the answer to those is off-box replication.
//!
//! # Format
//!
//! JSONL: one find, one self-contained JSON object, one line, fsync'd before `append`
//! returns. Self-contained-per-line is deliberate — a torn tail from a crash mid-write costs
//! exactly one record, and [`FallbackSink::import_into`] steps over the damage instead of
//! aborting recovery.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use tm_core::{FindKind, FoundPayload};

use crate::journal::Journal;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ImportStats {
    /// Records handed to `journal.append()` without error.
    pub imported: usize,
    /// Lines that could not be parsed; skipped, never fatal.
    pub malformed: usize,
    /// False only when the sink file does not exist.
    pub file_present: bool,
}

pub struct FallbackSink {
    path: PathBuf,
}

impl FallbackSink {
    /// Records the path only: no create, no open, no stat. This is constructed on the
    /// miner's startup path, where a filesystem error must not abort boot, and it must be
    /// safe to construct while the disk is the broken thing. The file appears on first
    /// [`FallbackSink::append`].
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one find as a single JSONL line, fsync'd to stable storage before returning.
    /// Returns false on any failure — the caller is already in a failure handler with
    /// nowhere to escalate except a log line, and a false return means the find is genuinely
    /// unrecoverable.
    pub fn append(&self, payload: &FoundPayload) -> bool {
        let line = serialize(payload);

        // Try O_EXCL first so we learn whether we created the file, which decides whether
        // the parent directory also needs an fsync. The plain append-open then covers both
        // "it already existed" and any other error.
        let (file, created) = match open_new(&self.path) {
            Ok(file) => (file, true),
            Err(_) => match open_append(&self.path) {
                Ok(file) => (file, false),
                Err(_) => return false,
            },
        };

        let mut file = file;
        if file.write_all(line.as_bytes()).is_err() {
            return false;
        }
        if file.sync_all().is_err() {
            return false;
        }
        drop(file);

        // fsync on the file alone does not make a newly created file's directory entry
        // durable — after a power loss the data blocks can exist with no name pointing at
        // them. Failure here is non-fatal: the record itself is already fsync'd, and a
        // nameless-but-committed file beats a dropped find.
        if created {
            sync_parent_directory(&self.path);
        }
        true
    }

    /// Startup drain: reads the sink line by line and appends every parseable record to
    /// `journal`.
    ///
    /// Semantics, all chosen so recovery degrades gracefully rather than all-or-nothing:
    ///
    /// * Missing file — the overwhelmingly common case — returns `{0, 0, false}` with no
    ///   error and no side effects.
    /// * An unparseable line increments `malformed` and recovery continues. One corrupt line
    ///   must never strand the good records behind it.
    /// * If `journal.append()` fails, importing stops at that record and the stats so far
    ///   are returned with `file_present = true`. The file is left as-is so the next boot
    ///   retries the whole thing; records already imported dedupe on their key.
    /// * Only after a clean pass is the file renamed to `<path>.imported`: the evidence
    ///   survives and the next boot starts from an empty sink. A failed rename is not an
    ///   error — the records are durable and a replay would dedupe.
    pub fn import_into(journal: &dyn Journal, path: impl AsRef<Path>) -> ImportStats {
        let path = path.as_ref();
        let mut stats = ImportStats::default();

        if !path.exists() {
            return stats; // the normal case: SQLite never failed
        }

        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(_) => {
                // Present but unreadable: report it so the caller can surface "sink pending"
                // rather than claiming a clean boot, and do not rename, so a later boot
                // retries.
                stats.file_present = true;
                return stats;
            }
        };
        stats.file_present = true;

        let mut clean_pass = true;
        for raw_line in bytes.split(|b| *b == b'\n') {
            let line = trim_line(raw_line);
            // Blank lines are not damage — a truncated final write plus a later append
            // leaves them behind. Ignore them silently.
            if line.is_empty() {
                continue;
            }

            let payload = match std::str::from_utf8(line).ok().and_then(parse_payload) {
                Some(payload) => payload,
                None => {
                    stats.malformed += 1;
                    continue;
                }
            };

            if journal.append(&payload).is_err() {
                // The journal is still broken. Stop, leave the file untouched, and let the
                // next boot retry the whole file — everything already imported in this pass
                // dedupes on its unique key, so replay has no side effects.
                clean_pass = false;
                break;
            }
            stats.imported += 1;
        }

        if !clean_pass {
            return stats;
        }

        let archived = archive_path(path);
        let _ = fs::remove_file(&archived);
        let _ = fs::rename(path, &archived);
        stats
    }
}

fn archive_path(path: &Path) -> PathBuf {
    let mut archived = path.as_os_str().to_os_string();
    archived.push(".imported");
    PathBuf::from(archived)
}

/// Strips a trailing CR (a sink written or hand-edited on Windows carries one) and reports
/// whitespace-only lines as empty so the caller skips them.
fn trim_line(raw: &[u8]) -> &[u8] {
    let mut end = raw.len();
    while end > 0 && matches!(raw[end - 1], b'\r' | b'\n' | b' ' | b'\t') {
        end -= 1;
    }
    let mut start = 0;
    while start < end && matches!(raw[start], b' ' | b'\t') {
        start += 1;
    }
    &raw[start..end]
}

#[cfg(unix)]
fn open_new(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    // 0600: the `key` field IS the mining secret for that find. Nothing but this process
    // (and root) has any business reading the sink.
    OpenOptions::new()
        .append(true)
        .create_new(true)
        .mode(0o600)
        .open(path)
}

#[cfg(not(unix))]
fn open_new(path: &Path) -> std::io::Result<File> {
    OpenOptions::new().append(true).create_new(true).open(path)
}

fn open_append(path: &Path) -> std::io::Result<File> {
    // O_APPEND makes the seek-and-write atomic against other appenders, so records from
    // concurrent callers cannot interleave.
    OpenOptions::new().append(true).open(path)
}

#[cfg(unix)]
fn sync_parent_directory(file_path: &Path) {
    let parent = match file_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    if let Ok(dir) = File::open(parent) {
        let _ = dir.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_parent_directory(_file_path: &Path) {}

// -----------------------------------------------------------------------------------------
// Minimal JSON serialization / parsing.
//
// Scope is exactly one flat object of string and number fields — the shape of FoundPayload.
// The serializer is hand-rolled so the bytes match the C++ sink exactly (byte-verbatim
// non-ASCII, C0 controls as \u00XX), which keeps the two implementations' sink files
// mutually importable.
// -----------------------------------------------------------------------------------------

/// Escapes `"`, `\` and every C0 control character. Bytes >= 0x80 are emitted verbatim and
/// never `\u`-escaped: the sink's job is byte-exact round-tripping of whatever the miner
/// captured.
fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for c in value.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn push_field(out: &mut String, key: &str, value: &str, first: bool) {
    if !first {
        out.push(',');
    }
    push_json_string(out, key);
    out.push(':');
    push_json_string(out, value);
}

fn push_raw_field(out: &mut String, key: &str, literal: &str, first: bool) {
    if !first {
        out.push(',');
    }
    push_json_string(out, key);
    out.push(':');
    out.push_str(literal);
}

fn serialize(payload: &FoundPayload) -> String {
    let mut out = String::with_capacity(512);
    out.push('{');
    push_field(&mut out, "key", &payload.key, true);
    push_field(&mut out, "hash_to_verify", &payload.hash_to_verify, false);
    push_field(&mut out, "account", &payload.account, false);
    push_field(&mut out, "kind", payload.kind.as_str(), false);
    push_raw_field(
        &mut out,
        "memory_cost",
        &payload.memory_cost.to_string(),
        false,
    );
    push_field(&mut out, "worker", &payload.worker, false);
    push_raw_field(&mut out, "attempts", &payload.attempts.to_string(), false);

    // `{:?}` on f64 is the shortest representation that round-trips the exact bits, which is
    // what a file whose purpose is evidence needs. NaN/inf have no JSON spelling, so they
    // become 0 rather than an unparseable line.
    let rate = if payload.hashes_per_second.is_finite() {
        format!("{:?}", payload.hashes_per_second)
    } else {
        "0".to_string()
    };
    push_raw_field(&mut out, "hashes_per_second", &rate, false);

    push_field(&mut out, "found_at_utc", &payload.found_at_utc, false);
    out.push('}');
    out.push('\n');
    out
}

/// Parses one sink line. Every field must be present: a record missing one cannot be
/// faithfully reconstructed, and substituting defaults would corrupt the audit trail that is
/// this file's whole reason to exist. Nested objects and arrays are rejected outright —
/// nothing in `FoundPayload` is structured, so encountering one means the line is not ours.
fn parse_payload(line: &str) -> Option<FoundPayload> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let object = value.as_object()?;
    if object.values().any(|v| v.is_object() || v.is_array()) {
        return None;
    }

    // Numbers are accepted in either spelling: the C++ parser keeps bare tokens as raw text
    // and converts them itself, so a hand-written sink using quoted numbers still imports.
    let text = |key: &str| -> Option<String> {
        match object.get(key)? {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Number(n) => Some(n.to_string()),
            _ => None,
        }
    };

    let memory_cost: u32 = text("memory_cost")?.parse().ok()?;
    let attempts: u64 = text("attempts")?.parse().ok()?;
    let hashes_per_second: f64 = text("hashes_per_second")?.parse().ok()?;

    Some(FoundPayload {
        key: text("key")?,
        hash_to_verify: text("hash_to_verify")?,
        account: text("account")?,
        kind: FindKind::parse(&text("kind")?)?,
        memory_cost,
        worker: text("worker")?,
        attempts,
        hashes_per_second,
        found_at_utc: text("found_at_utc")?,
    })
}

#[cfg(test)]
mod tests;
