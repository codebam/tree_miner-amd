//! Port of `tests/unit/journal/FallbackSinkTests.cpp`.
//!
//! The sink's whole contract is "the find reaches the journal intact", so the tests assert
//! on records hydrated back out of the journal rather than on the sink file's bytes.

use std::collections::HashMap;
use std::path::PathBuf;

use tempfile::TempDir;
use tm_core::{FindKind, FindRecord, FindStatus, FoundPayload};

use super::*;
use crate::journal::FindJournal;

const NOW: &str = "2026-08-09T12:34:56Z";

struct Fixture {
    _dir: TempDir,
    sink_path: PathBuf,
    db_path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let sink_path = dir.path().join("finds.jsonl");
        let db_path = dir.path().join("finds.db");
        Self {
            _dir: dir,
            sink_path,
            db_path,
        }
    }

    fn journal(&self) -> FindJournal {
        FindJournal::open(&self.db_path).expect("open journal")
    }

    fn sink(&self) -> FallbackSink {
        FallbackSink::new(&self.sink_path)
    }

    fn sink_exists(&self) -> bool {
        self.sink_path.exists()
    }

    fn archive_exists(&self) -> bool {
        archive_path(&self.sink_path).exists()
    }

    fn read_lines(&self) -> Vec<String> {
        match fs::read_to_string(&self.sink_path) {
            Ok(text) => text.lines().map(str::to_string).collect(),
            Err(_) => Vec::new(),
        }
    }

    fn write_lines(&self, lines: &[&str]) {
        let mut text = String::new();
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
        fs::write(&self.sink_path, text).expect("write sink");
    }
}

fn payload_of(key_suffix: &str, kind: FindKind, memory_cost: u32) -> FoundPayload {
    FoundPayload {
        key: format!("aabbccddeeff00112233445566778899aabbccddeeff001122334455667788{key_suffix}"),
        hash_to_verify: format!(
            "$argon2id$v=19$m={memory_cost},t=1,p=1$c29tZXNhbHRzb21lc2FsdDE5$\
             TFVYRU4xMWFiY2RlZmdoaWprbG1ub3BxcnN0dXZ3eHl6QUJDREVGRw"
        ),
        account: "0x1234567890abcdef1234567890abcdef12345678".to_string(),
        kind,
        memory_cost,
        worker: "rig0-gpu0".to_string(),
        attempts: 123_456,
        hashes_per_second: 1500.5,
        found_at_utc: "2026-08-09T12:00:00Z".to_string(),
    }
}

/// Every journaled Pending record, keyed by payload key.
fn journaled_by_key(journal: &FindJournal) -> HashMap<String, FindRecord> {
    journal
        .fetch_eligible(NOW, 1000)
        .expect("fetch eligible")
        .into_iter()
        .map(|record| (record.payload.key.clone(), record))
        .collect()
}

/// The core path: SQLite was broken, three finds fell into the sink, the next boot drains
/// them back with nothing lost or altered.
#[test]
fn append_and_import_round_trip() {
    let fixture = Fixture::new();

    let a = payload_of("01", FindKind::Xen11, 1727);
    let mut b = payload_of("02", FindKind::Xuni, 1900);
    b.worker = "rig1-gpu3".to_string();
    b.attempts = 987_654_321;
    b.hashes_per_second = 0.125; // exactly representable; the encoding must not perturb it
    b.found_at_utc = "2026-08-09T12:05:01Z".to_string();
    let mut c = payload_of("03", FindKind::Xen11, 2048);
    c.hashes_per_second = 1_234.567_89; // NOT exactly representable: the real round-trip test

    let sink = fixture.sink();
    assert!(sink.append(&a));
    assert!(sink.append(&b));
    assert!(sink.append(&c));
    assert_eq!(fixture.read_lines().len(), 3);

    let journal = fixture.journal();
    let stats = FallbackSink::import_into(&journal, &fixture.sink_path);
    assert_eq!(stats.imported, 3);
    assert_eq!(stats.malformed, 0);
    assert!(stats.file_present);

    let by_key = journaled_by_key(&journal);
    assert_eq!(by_key.len(), 3);
    for expected in [&a, &b, &c] {
        let record = by_key.get(&expected.key).expect("record imported");
        assert_eq!(record.status, FindStatus::Pending);
        assert_eq!(&record.payload, expected);
    }

    // Clean pass: the sink is archived, not deleted (evidence), and the next boot starts
    // from an empty sink.
    assert!(!fixture.sink_exists());
    assert!(fixture.archive_exists());
}

/// Re-import must be free: the sink may hold a record SQLite actually did commit (append
/// failed after the INSERT), or one a previous boot imported before dying ahead of the
/// rename. Both collapse onto the journal's unique key.
#[test]
fn import_is_idempotent_by_key() {
    let fixture = Fixture::new();
    let payload = payload_of("07", FindKind::Xuni, 1800);

    let sink = fixture.sink();
    assert!(sink.append(&payload));
    assert!(sink.append(&payload)); // same key twice in the sink

    let journal = fixture.journal();
    let direct_id = journal.append(&payload).expect("direct append"); // and once directly
    assert!(direct_id > 0);

    let stats = FallbackSink::import_into(&journal, &fixture.sink_path);
    assert_eq!(stats.imported, 2); // both lines were handed over...
    assert_eq!(stats.malformed, 0);

    let by_key = journaled_by_key(&journal);
    assert_eq!(by_key.len(), 1); // ...and both deduped onto the original row
    let record = &by_key[&payload.key];
    assert_eq!(record.id, direct_id);
    assert_eq!(record.payload, payload);
}

/// One damaged line must never strand the good records behind it — that would recreate the
/// exact drop this component exists to prevent.
#[test]
fn malformed_lines_never_block_recovery() {
    let fixture = Fixture::new();
    let first = payload_of("11", FindKind::Xen11, 1727);
    let second = payload_of("12", FindKind::Xuni, 1900);

    // Generate the good lines through the real serializer so the test cannot drift from the
    // sink's actual format, then hand-assemble a file with damage between them.
    let good: Vec<String> = [&first, &second]
        .iter()
        .map(|p| serialize(p).trim_end().to_string())
        .collect();
    let truncated = &good[0][..good[0].len() / 2]; // cut mid-object, likely mid-string

    fixture.write_lines(&[&good[0], "not json", truncated, &good[1]]);

    let journal = fixture.journal();
    let stats = FallbackSink::import_into(&journal, &fixture.sink_path);
    assert_eq!(stats.imported, 2);
    assert_eq!(stats.malformed, 2);
    assert!(stats.file_present);

    let by_key = journaled_by_key(&journal);
    assert_eq!(by_key.len(), 2);
    assert_eq!(by_key[&first.key].payload, first);
    assert_eq!(by_key[&second.key].payload, second);

    // Malformed lines are skipped, not fatal: the pass still counts as clean and archives.
    assert!(!fixture.sink_exists());
    assert!(fixture.archive_exists());
}

/// The overwhelmingly common boot: SQLite never failed, so no sink was ever created. Import
/// must be a silent no-op that neither errors nor creates the file.
#[test]
fn missing_file_is_clean_noop() {
    let fixture = Fixture::new();
    assert!(!fixture.sink_exists());

    let journal = fixture.journal();
    let stats = FallbackSink::import_into(&journal, &fixture.sink_path);
    assert_eq!(stats.imported, 0);
    assert_eq!(stats.malformed, 0);
    assert!(!stats.file_present);

    assert!(!fixture.sink_exists());
    assert!(!fixture.archive_exists());
    assert_eq!(journal.counts().unwrap().pending, 0);
}

/// JSONL is line-oriented and holds fields the miner does not sanitize. Quotes, backslashes,
/// embedded newlines and non-ASCII must survive escape/unescape exactly, or a "recovered"
/// find would be submitted with a corrupted payload.
#[test]
fn append_survives_weird_content() {
    let fixture = Fixture::new();

    let mut payload = payload_of("21", FindKind::Xuni, 4096);
    // hash_to_verify stays under the journal's 150-char server limit so the record lands
    // Pending rather than PermanentlyInvalid; the nastiness goes in the free-text fields.
    payload.hash_to_verify =
        "$argon2id$v=19$m=4096,t=1,p=1$c2FsdA$\"quoted\\backslash\"".to_string();
    payload.worker = "rig\"0\\gpu\nline2\ttab\u{1}ctrl-é-日本".to_string();
    payload.found_at_utc = "2026-08-09T12:00:00Z\r\nspliced".to_string();
    payload.attempts = u64::MAX; // no silent narrowing
    payload.hashes_per_second = 0.1 + 0.2; // 0.30000000000000004; shortest-round-trip or bust

    let sink = fixture.sink();
    assert!(sink.append(&payload));

    // Embedded newlines must have been escaped, not written raw: one find, one line.
    assert_eq!(fixture.read_lines().len(), 1);

    let journal = fixture.journal();
    let stats = FallbackSink::import_into(&journal, &fixture.sink_path);
    assert_eq!(stats.imported, 1);
    assert_eq!(stats.malformed, 0);

    let by_key = journaled_by_key(&journal);
    assert_eq!(by_key.len(), 1);
    assert_eq!(by_key[&payload.key].payload, payload);
}

/// A record the journal rejects stops the pass: the file is kept intact so the next boot
/// retries the whole thing, and the stats still report what was already drained.
#[test]
fn journal_failure_keeps_the_sink_for_the_next_boot() {
    let fixture = Fixture::new();
    let first = payload_of("31", FindKind::Xen11, 1727);
    let second = payload_of("32", FindKind::Xen11, 1727);

    let sink = fixture.sink();
    assert!(sink.append(&first));
    assert!(sink.append(&second));

    /// Journal that accepts the first record and then behaves like a broken SQLite.
    struct FailingJournal {
        inner: FindJournal,
        seen: std::cell::Cell<usize>,
    }

    impl Journal for FailingJournal {
        fn append(&self, payload: &FoundPayload) -> crate::Result<i64> {
            self.seen.set(self.seen.get() + 1);
            if self.seen.get() > 1 {
                return Err(crate::JournalError::Schema("disk on fire".into()));
            }
            self.inner.append(payload)
        }
        fn fetch_eligible(&self, now: &str, limit: usize) -> crate::Result<Vec<FindRecord>> {
            self.inner.fetch_eligible(now, limit)
        }
        fn fetch_eligible_of_kind(
            &self,
            kind: FindKind,
            now: &str,
            limit: usize,
        ) -> crate::Result<Vec<FindRecord>> {
            self.inner.fetch_eligible_of_kind(kind, now, limit)
        }
        fn fetch_awaiting_confirmation(
            &self,
            now: &str,
            limit: usize,
        ) -> crate::Result<Vec<FindRecord>> {
            self.inner.fetch_awaiting_confirmation(now, limit)
        }
        fn get_by_id(&self, id: i64) -> crate::Result<Option<FindRecord>> {
            self.inner.get_by_id(id)
        }
        fn record_attempt(
            &self,
            id: i64,
            classification: &tm_core::Classification,
            http_status: Option<i32>,
            response_body: &str,
            next_attempt_at: Option<&str>,
            now_utc: &str,
        ) -> crate::Result<()> {
            self.inner.record_attempt(
                id,
                classification,
                http_status,
                response_body,
                next_attempt_at,
                now_utc,
            )
        }
        fn unpark_for_difficulty(&self, difficulty: u32) -> crate::Result<usize> {
            self.inner.unpark_for_difficulty(difficulty)
        }
        fn unpark_xuni_for_window(&self, max_windows: i32) -> crate::Result<usize> {
            self.inner.unpark_xuni_for_window(max_windows)
        }
        fn recover_on_startup(&self) -> crate::Result<crate::RecoveryStats> {
            self.inner.recover_on_startup()
        }
        fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> crate::Result<()> {
            self.inner.record_difficulty(difficulty, at_utc)
        }
        fn last_known_difficulty(&self) -> crate::Result<Option<u32>> {
            self.inner.last_known_difficulty()
        }
        fn counts(&self) -> crate::Result<crate::Counts> {
            self.inner.counts()
        }
    }

    let journal = FailingJournal {
        inner: fixture.journal(),
        seen: std::cell::Cell::new(0),
    };
    let stats = FallbackSink::import_into(&journal, &fixture.sink_path);
    assert_eq!(stats.imported, 1);
    assert!(stats.file_present);

    // Nothing archived, nothing lost: the next boot re-reads both lines and dedupes the one
    // that made it.
    assert!(fixture.sink_exists());
    assert!(!fixture.archive_exists());
    assert_eq!(fixture.read_lines().len(), 2);

    let stats = FallbackSink::import_into(&journal.inner, &fixture.sink_path);
    assert_eq!(stats.imported, 2);
    assert_eq!(journaled_by_key(&journal.inner).len(), 2);
    assert!(fixture.archive_exists());
}

/// Blank lines are what a truncated final write plus a later append leaves behind; they are
/// not damage and must not be counted as malformed.
#[test]
fn blank_lines_are_not_damage() {
    let fixture = Fixture::new();
    let find = payload_of("41", FindKind::Xen11, 1727);
    let line = serialize(&find);

    fixture.write_lines(&["", "   ", line.trim_end(), "\t"]);

    let journal = fixture.journal();
    let stats = FallbackSink::import_into(&journal, &fixture.sink_path);
    assert_eq!(stats.imported, 1);
    assert_eq!(stats.malformed, 0);
}

/// The sink holds the mining secret for each find, so it is created 0600.
#[cfg(unix)]
#[test]
fn sink_file_is_owner_only() {
    use std::os::unix::fs::PermissionsExt;

    let fixture = Fixture::new();
    assert!(fixture
        .sink()
        .append(&payload_of("51", FindKind::Xen11, 1727)));

    let mode = fs::metadata(&fixture.sink_path)
        .expect("sink metadata")
        .permissions()
        .mode();
    assert_eq!(mode & 0o777, 0o600);
}
