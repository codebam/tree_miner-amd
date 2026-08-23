//! Port of `tests/unit/journal/FindJournalTests.cpp`.
//!
//! Behaviour is verified through the public API, and additionally through raw SQLite reads
//! of the database file wherever the contract is about persisted bytes (status strings must
//! match the `FindStatus` enumerator names exactly, because they are the on-disk values).

use rusqlite::{Connection, OpenFlags};
use tempfile::TempDir;
use tm_core::{Classification, FindKind, FindStatus, FoundPayload};

use super::*;

const NOW: &str = "2026-08-09T12:34:56Z";

struct TempDb {
    _dir: TempDir,
    path: std::path::PathBuf,
}

impl TempDb {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("finds.db");
        Self { _dir: dir, path }
    }

    fn open(&self) -> FindJournal {
        FindJournal::open(&self.path).expect("open journal")
    }

    /// Raw connection, independent of `FindJournal`, for verifying persisted bytes and for
    /// injecting states the API refuses to write.
    fn raw(&self) -> Connection {
        Connection::open_with_flags(&self.path, OpenFlags::SQLITE_OPEN_READ_WRITE)
            .expect("raw open")
    }
}

fn raw_text(conn: &Connection, sql: &str) -> Option<String> {
    conn.query_row(sql, [], |row| row.get::<_, Option<String>>(0))
        .expect("raw scalar text")
}

fn raw_int(conn: &Connection, sql: &str) -> i64 {
    conn.query_row(sql, [], |row| row.get::<_, i64>(0))
        .expect("raw scalar int")
}

fn payload(key_suffix: &str) -> FoundPayload {
    payload_of(key_suffix, FindKind::Xen11, 1727)
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

fn classify(status: FindStatus, reason: &str) -> Classification {
    Classification {
        next_status: status,
        server_difficulty_hint: None,
        needs_lookup_confirmation: false,
        reason: reason.to_string(),
    }
}

#[test]
fn append_durable_and_idempotent() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    let mut find = payload("01");
    let id = journal.append(&find).expect("append");
    assert!(id > 0);

    // Duplicate key: same id back, still exactly one row, original capture untouched.
    find.worker = "different-worker".to_string();
    assert_eq!(journal.append(&find).expect("re-append"), id);

    let raw = tmp.raw();
    assert_eq!(raw_int(&raw, "SELECT COUNT(*) FROM finds;"), 1);
    assert_eq!(
        raw_text(&raw, "SELECT status FROM finds WHERE id=1;").unwrap(),
        "Pending"
    );
    assert_eq!(
        raw_text(&raw, "SELECT key FROM finds WHERE id=1;").unwrap(),
        find.key
    );
    assert_eq!(
        raw_text(&raw, "SELECT worker FROM finds WHERE id=1;").unwrap(),
        "rig0-gpu0"
    );
    assert_eq!(
        raw_text(&raw, "SELECT kind FROM finds WHERE id=1;").unwrap(),
        "XEN11"
    );
    // The durability plumbing really is in effect.
    assert_eq!(raw_text(&raw, "PRAGMA journal_mode;").unwrap(), "wal");
    assert_eq!(raw_int(&raw, "SELECT version FROM schema_version;"), 1);
}

#[test]
fn invalid_payload_stored() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    // Oversize hash_to_verify (server limit 150): stored, but PermanentlyInvalid.
    let mut oversize = payload("02");
    oversize.hash_to_verify = "x".repeat(151);
    let id_oversize = journal.append(&oversize).expect("append oversize");
    assert!(id_oversize > 0);

    // Empty account: same handling.
    let mut no_account = payload("03");
    no_account.account = String::new();
    let id_no_account = journal.append(&no_account).expect("append no account");
    assert!(id_no_account > 0);
    assert_ne!(id_no_account, id_oversize);

    let raw = tmp.raw();
    assert_eq!(raw_int(&raw, "SELECT COUNT(*) FROM finds;"), 2);
    assert_eq!(
        raw_int(
            &raw,
            "SELECT COUNT(*) FROM finds WHERE status='PermanentlyInvalid' \
             AND status_reason IS NOT NULL;"
        ),
        2
    );
    // Data preserved verbatim (never throw away a find).
    assert_eq!(
        raw_int(
            &raw,
            &format!("SELECT LENGTH(hash_to_verify) FROM finds WHERE id={id_oversize};")
        ) as usize,
        oversize.hash_to_verify.len()
    );

    // Invalid rows are never eligible for submission.
    assert!(journal.fetch_eligible(NOW, 10).expect("fetch").is_empty());
}

#[test]
fn fetch_eligible_backoff_and_ordering() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    let id1 = journal.append(&payload("11")).unwrap();
    let id2 = journal.append(&payload("12")).unwrap();
    let id3 = journal.append(&payload("13")).unwrap();

    // id2 stays Pending but with a future backoff time.
    journal
        .record_attempt(
            id2,
            &classify(FindStatus::Pending, "transient 503"),
            Some(503),
            "server sick",
            Some("2026-08-09T13:00:00Z"),
            NOW,
        )
        .expect("record attempt");

    // Before the backoff expires: id2 excluded, order ascending by id.
    let eligible = journal.fetch_eligible("2026-08-09T12:59:59Z", 10).unwrap();
    assert_eq!(eligible.len(), 2);
    assert_eq!(eligible[0].id, id1);
    assert_eq!(eligible[1].id, id3);

    // next_attempt_at <= now is inclusive.
    let eligible = journal.fetch_eligible("2026-08-09T13:00:00Z", 10).unwrap();
    assert_eq!(
        eligible.iter().map(|r| r.id).collect::<Vec<_>>(),
        vec![id1, id2, id3]
    );

    // LIMIT respected, oldest first.
    let eligible = journal.fetch_eligible("2026-08-09T13:00:00Z", 1).unwrap();
    assert_eq!(eligible.len(), 1);
    assert_eq!(eligible[0].id, id1);
}

#[test]
fn record_attempt_round_trip() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    let find = payload("21");
    let id = journal.append(&find).unwrap();

    // Failed attempt: stays Pending with backoff; every field round-trips.
    journal
        .record_attempt(
            id,
            &classify(FindStatus::Pending, "connect timeout"),
            None,
            "",
            Some("2026-08-09T12:40:00Z"),
            NOW,
        )
        .unwrap();

    let eligible = journal.fetch_eligible("2026-08-09T12:40:00Z", 10).unwrap();
    assert_eq!(eligible.len(), 1);
    let record = &eligible[0];
    assert_eq!(record.id, id);
    assert_eq!(record.status, FindStatus::Pending);
    assert_eq!(record.status_reason, "connect timeout");
    assert_eq!(record.attempt_count, 1);
    assert_eq!(
        record.next_attempt_at.as_deref(),
        Some("2026-08-09T12:40:00Z")
    );
    assert_eq!(record.last_attempt_at.as_deref(), Some(NOW));
    assert!(record.last_http_status.is_none());
    assert!(record.confirmed_at.is_none());
    // Immutable payload intact.
    assert_eq!(record.payload, find);

    // Ack: terminal, confirmed_at set to the ack time.
    let ack_time = "2026-08-09T12:45:00Z";
    journal
        .record_attempt(
            id,
            &classify(FindStatus::Acked, "confirmed via get_block"),
            Some(200),
            "OK",
            None,
            ack_time,
        )
        .unwrap();

    let raw = tmp.raw();
    assert_eq!(
        raw_text(&raw, "SELECT status FROM finds WHERE id=1;").unwrap(),
        "Acked"
    );
    assert_eq!(
        raw_text(&raw, "SELECT confirmed_at FROM finds WHERE id=1;").unwrap(),
        ack_time
    );
    assert_eq!(
        raw_int(&raw, "SELECT attempt_count FROM finds WHERE id=1;"),
        2
    );
    assert_eq!(
        raw_int(&raw, "SELECT last_http_status FROM finds WHERE id=1;"),
        200
    );
    assert_eq!(
        raw_text(&raw, "SELECT last_response FROM finds WHERE id=1;").unwrap(),
        "OK"
    );
    assert!(raw_text(&raw, "SELECT next_attempt_at FROM finds WHERE id=1;").is_none());

    // A repeated Ack must not move the original confirmation time.
    journal
        .record_attempt(
            id,
            &classify(FindStatus::Acked, "re-ack"),
            Some(200),
            "OK",
            None,
            "2026-08-09T23:00:00Z",
        )
        .unwrap();
    assert_eq!(
        raw_text(&raw, "SELECT confirmed_at FROM finds WHERE id=1;").unwrap(),
        ack_time
    );

    // Contract guards.
    let submitting = journal.record_attempt(
        id,
        &classify(FindStatus::Submitting, ""),
        None,
        "",
        None,
        NOW,
    );
    assert!(matches!(submitting, Err(JournalError::Contract(_))));

    let unknown = journal.record_attempt(
        9999,
        &classify(FindStatus::Pending, ""),
        None,
        "",
        None,
        NOW,
    );
    assert!(matches!(unknown, Err(JournalError::Contract(_))));
}

#[test]
fn unpark_for_difficulty_boundary() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    let id_high = journal
        .append(&payload_of("31", FindKind::Xen11, 1500))
        .unwrap();
    let id_low = journal
        .append(&payload_of("32", FindKind::Xen11, 1400))
        .unwrap();
    for id in [id_high, id_low] {
        journal
            .record_attempt(
                id,
                &classify(FindStatus::ParkedDifficulty, "difficulty"),
                Some(401),
                "difficulty too low",
                Some("2026-08-09T13:00:00Z"),
                NOW,
            )
            .unwrap();
    }

    // Boundary: m == current difficulty unparks (the server rejects strictly m < current).
    assert_eq!(journal.unpark_for_difficulty(1500).unwrap(), 1);

    let raw = tmp.raw();
    assert_eq!(
        raw_text(
            &raw,
            &format!("SELECT status FROM finds WHERE id={id_high};")
        )
        .unwrap(),
        "Pending"
    );
    assert_eq!(
        raw_text(
            &raw,
            &format!("SELECT status FROM finds WHERE id={id_low};")
        )
        .unwrap(),
        "ParkedDifficulty"
    );

    // Backoff cleared: eligible immediately, before the old next_attempt_at.
    let eligible = journal.fetch_eligible(NOW, 10).unwrap();
    assert_eq!(eligible.len(), 1);
    assert_eq!(eligible[0].id, id_high);
    assert!(eligible[0].next_attempt_at.is_none());

    // A later call at a lower difficulty releases the rest; nothing double-unparks.
    assert_eq!(journal.unpark_for_difficulty(1400).unwrap(), 1);
    assert_eq!(journal.unpark_for_difficulty(1400).unwrap(), 0);
}

#[test]
fn unpark_xuni_budget_exhaustion() {
    let tmp = TempDb::new();
    let journal = tmp.open();
    let raw = tmp.raw();

    let id = journal
        .append(&payload_of("41", FindKind::Xuni, 1727))
        .unwrap();
    let park = || {
        journal
            .record_attempt(
                id,
                &classify(FindStatus::ParkedXuniWindow, "missed window"),
                Some(401),
                "XUNI Submitted outside of proper time frame.",
                None,
                NOW,
            )
            .unwrap();
    };

    for window in 1..=3 {
        park();
        assert_eq!(journal.unpark_xuni_for_window(3).unwrap(), 1);
        assert_eq!(
            raw_int(&raw, "SELECT xuni_windows_tried FROM finds WHERE id=1;"),
            window
        );
        assert_eq!(
            raw_text(&raw, "SELECT status FROM finds WHERE id=1;").unwrap(),
            "Pending"
        );
    }

    // Budget of 3 exhausted: the 4th window kills it.
    park();
    assert_eq!(journal.unpark_xuni_for_window(3).unwrap(), 0);
    assert_eq!(
        raw_text(&raw, "SELECT status FROM finds WHERE id=1;").unwrap(),
        "Dead"
    );
    assert_eq!(
        raw_text(&raw, "SELECT status_reason FROM finds WHERE id=1;").unwrap(),
        "xuni window budget exhausted"
    );
    assert!(journal.fetch_eligible(NOW, 10).unwrap().is_empty());
}

#[test]
fn recover_on_startup_counts() {
    let tmp = TempDb::new();
    {
        let journal = tmp.open();
        // Two Pending.
        journal.append(&payload("51")).unwrap();
        journal
            .append(&payload_of("52", FindKind::Xuni, 1727))
            .unwrap();
        // One of each non-pending state, driven through the public API.
        let put = |suffix: &str, status: FindStatus| {
            let id = journal.append(&payload(suffix)).unwrap();
            journal
                .record_attempt(id, &classify(status, "test"), None, "", None, NOW)
                .unwrap();
        };
        put("53", FindStatus::AcceptedUnconfirmed);
        put("54", FindStatus::Acked);
        put("55", FindStatus::ParkedDifficulty);
        put("56", FindStatus::ParkedXuniWindow);
        put("57", FindStatus::Quarantined);
        put("58", FindStatus::Dead);
        let mut invalid = payload("59");
        invalid.account = String::new();
        journal.append(&invalid).unwrap(); // PermanentlyInvalid
    } // close so we can tamper below

    {
        // 'Submitting' can only exist through a bug; inject it raw to prove recovery repairs
        // it. Also stamp a stale backoff that recovery must NOT clear.
        let raw = tmp.raw();
        raw.execute_batch(
            "UPDATE finds SET status='Submitting', \
             next_attempt_at='2026-08-09T11:00:00Z' WHERE id=1;",
        )
        .unwrap();
    }

    let journal = tmp.open();
    let stats = journal.recover_on_startup().unwrap();
    assert_eq!(stats.pending, 2); // includes the repaired Submitting row
    assert_eq!(stats.accepted_unconfirmed, 1);
    assert_eq!(stats.parked_difficulty, 1);
    assert_eq!(stats.parked_xuni, 1);
    assert_eq!(stats.quarantined, 1);
    assert_eq!(stats.acked, 1);
    assert_eq!(stats.dead, 1);
    assert_eq!(stats.invalid, 1);

    let raw = tmp.raw();
    assert_eq!(
        raw_int(
            &raw,
            "SELECT COUNT(*) FROM finds WHERE status='Submitting';"
        ),
        0
    );
    // Recovery keeps persisted backoff: restarts must not reset backoff.
    assert_eq!(
        raw_text(&raw, "SELECT next_attempt_at FROM finds WHERE id=1;").unwrap(),
        "2026-08-09T11:00:00Z"
    );

    // counts() mapping: parked aggregates both parked states, strict per-state elsewhere.
    let counts = journal.counts().unwrap();
    assert_eq!(counts.pending, 2);
    assert_eq!(counts.parked, 2);
    assert_eq!(counts.parked_difficulty, 1);
    assert_eq!(counts.parked_xuni, 1);
    assert_eq!(counts.quarantined, 1);
    assert_eq!(counts.acked_total, 1);
    assert_eq!(counts.dead_total, 1);
    assert_eq!(counts.accepted_unconfirmed, 1);
    assert_eq!(counts.permanently_invalid, 1);
    assert_eq!(counts.queued_xen11, 2); // one Pending + one AcceptedUnconfirmed
    assert_eq!(counts.queued_xuni, 1);
}

#[test]
fn fetch_awaiting_confirmation() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    let id_pending = journal.append(&payload("71")).unwrap();
    let id_no_backoff = journal.append(&payload("72")).unwrap();
    let id_past_backoff = journal.append(&payload("73")).unwrap();
    let id_future_backoff = journal.append(&payload("74")).unwrap();

    // Three AcceptedUnconfirmed rows: NULL, past and future next_attempt_at.
    for (id, next) in [
        (id_no_backoff, None),
        (id_past_backoff, Some("2026-08-09T12:00:00Z")),
        (id_future_backoff, Some("2026-08-09T13:00:00Z")),
    ] {
        journal
            .record_attempt(
                id,
                &classify(FindStatus::AcceptedUnconfirmed, "lookup down"),
                Some(200),
                "OK",
                next,
                NOW,
            )
            .unwrap();
    }

    // NULL and past backoffs are fetched oldest-first; the future one is not; Pending rows
    // never leak into the confirmation queue.
    let awaiting = journal.fetch_awaiting_confirmation(NOW, 10).unwrap();
    assert_eq!(awaiting.len(), 2);
    assert_eq!(awaiting[0].id, id_no_backoff);
    assert_eq!(awaiting[1].id, id_past_backoff);
    assert_eq!(awaiting[0].status, FindStatus::AcceptedUnconfirmed);
    assert_eq!(awaiting[0].status_reason, "lookup down");
    assert_eq!(awaiting[0].attempt_count, 1);
    assert_eq!(awaiting[0].last_http_status, Some(200));
    assert_eq!(awaiting[0].payload.key, payload("72").key); // full hydration

    // Backoff boundary is inclusive, and LIMIT applies.
    let awaiting = journal
        .fetch_awaiting_confirmation("2026-08-09T13:00:00Z", 10)
        .unwrap();
    assert_eq!(awaiting.len(), 3);
    assert_eq!(awaiting[2].id, id_future_backoff);
    let awaiting = journal
        .fetch_awaiting_confirmation("2026-08-09T13:00:00Z", 1)
        .unwrap();
    assert_eq!(awaiting.len(), 1);
    assert_eq!(awaiting[0].id, id_no_backoff);

    // Conversely fetch_eligible sees only the Pending row.
    let eligible = journal.fetch_eligible(NOW, 10).unwrap();
    assert_eq!(eligible.len(), 1);
    assert_eq!(eligible[0].id, id_pending);
}

#[test]
fn get_by_id_hit_and_miss() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    let find = payload_of("81", FindKind::Xuni, 1600);
    let id = journal.append(&find).unwrap();
    let ack_time = "2026-08-09T12:45:00Z";
    journal
        .record_attempt(
            id,
            &classify(FindStatus::Acked, "confirmed"),
            Some(200),
            "OK",
            None,
            ack_time,
        )
        .unwrap();

    // Hit: full hydration, including a terminal state fetch_eligible would never expose.
    let found = journal.get_by_id(id).unwrap().expect("record present");
    assert_eq!(found.id, id);
    assert_eq!(found.status, FindStatus::Acked);
    assert_eq!(found.status_reason, "confirmed");
    assert_eq!(found.attempt_count, 1);
    assert_eq!(found.confirmed_at.as_deref(), Some(ack_time));
    assert_eq!(found.last_attempt_at.as_deref(), Some(ack_time));
    assert_eq!(found.last_response, "OK");
    assert_eq!(found.payload, find);

    // Miss: None, not an error.
    assert!(journal.get_by_id(9999).unwrap().is_none());
}

#[test]
fn difficulty_seen_round_trip() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    assert!(journal.last_known_difficulty().unwrap().is_none());

    journal
        .record_difficulty(1727, "2026-08-09T12:00:00Z")
        .unwrap();
    journal
        .record_difficulty(2727, "2026-08-09T12:05:00Z")
        .unwrap();
    journal
        .record_difficulty(1587, "2026-08-09T12:10:00Z")
        .unwrap();

    assert_eq!(journal.last_known_difficulty().unwrap(), Some(1587));

    let raw = tmp.raw();
    assert_eq!(raw_int(&raw, "SELECT COUNT(*) FROM difficulty_seen;"), 3);
    assert_eq!(
        raw_text(
            &raw,
            "SELECT at FROM difficulty_seen ORDER BY rowid LIMIT 1;"
        )
        .unwrap(),
        "2026-08-09T12:00:00Z"
    );
    assert_eq!(
        raw_int(
            &raw,
            "SELECT value FROM difficulty_seen ORDER BY rowid LIMIT 1;"
        ),
        1727
    );
}

#[test]
fn fetch_eligible_of_kind() {
    let tmp = TempDb::new();
    let journal = tmp.open();

    // Three XUNI first — the head-of-line shape that used to starve XEN11 out of a mixed
    // LIMIT slice.
    journal
        .append(&payload_of("71", FindKind::Xuni, 1727))
        .unwrap();
    journal
        .append(&payload_of("72", FindKind::Xuni, 1727))
        .unwrap();
    journal
        .append(&payload_of("73", FindKind::Xuni, 1727))
        .unwrap();
    let xen_id = journal
        .append(&payload_of("74", FindKind::Xen11, 1727))
        .unwrap();
    journal
        .append(&payload_of("75", FindKind::Xuni, 1727))
        .unwrap();

    // A LIMIT smaller than the XUNI backlog still reaches the XEN11: the limit applies after
    // the kind filter. This is the guarantee fetch_eligible cannot give.
    let xen = journal
        .fetch_eligible_of_kind(FindKind::Xen11, NOW, 2)
        .unwrap();
    assert_eq!(xen.len(), 1);
    assert_eq!(xen[0].id, xen_id);
    assert_eq!(xen[0].payload.kind, FindKind::Xen11);

    // Kind slices honour oldest-first ordering and the LIMIT within the kind.
    let xuni = journal
        .fetch_eligible_of_kind(FindKind::Xuni, NOW, 3)
        .unwrap();
    assert_eq!(xuni.len(), 3);
    assert!(xuni[0].id < xuni[1].id && xuni[1].id < xuni[2].id);

    // Backoff eligibility applies identically.
    journal
        .record_attempt(
            xuni[0].id,
            &classify(FindStatus::Pending, "retry later"),
            Some(503),
            "unavailable",
            Some("2999-01-01T00:00:00Z"),
            NOW,
        )
        .unwrap();
    let after = journal
        .fetch_eligible_of_kind(FindKind::Xuni, NOW, 10)
        .unwrap();
    assert_eq!(after.len(), 3); // 4 XUNI total, one backed off
    assert!(after.iter().all(|r| r.id != xuni[0].id));

    // A kind with no eligible rows returns empty, never rows of the other kind.
    journal
        .record_attempt(
            xen_id,
            &classify(FindStatus::Acked, "confirmed"),
            Some(200),
            "OK",
            None,
            NOW,
        )
        .unwrap();
    assert!(journal
        .fetch_eligible_of_kind(FindKind::Xen11, NOW, 10)
        .unwrap()
        .is_empty());
}

#[test]
fn reopen_persistence() {
    let tmp = TempDb::new();
    let keep = payload("61");
    let kept_id;

    {
        let journal = tmp.open();
        kept_id = journal.append(&keep).unwrap();
        let acked_id = journal.append(&payload("62")).unwrap();
        journal
            .record_attempt(
                acked_id,
                &classify(FindStatus::Acked, "confirmed"),
                Some(200),
                "OK",
                None,
                NOW,
            )
            .unwrap();
        journal
            .record_difficulty(1727, "2026-08-09T12:00:00Z")
            .unwrap();
    } // dropped: connection closed

    let reopened = tmp.open();
    let eligible = reopened.fetch_eligible(NOW, 10).unwrap();
    assert_eq!(eligible.len(), 1);
    assert_eq!(eligible[0].id, kept_id);
    assert_eq!(eligible[0].payload, keep); // full payload intact across close/reopen

    let counts = reopened.counts().unwrap();
    assert_eq!(counts.pending, 1);
    assert_eq!(counts.acked_total, 1);
    assert_eq!(counts.queued_xen11, 1);
    assert_eq!(counts.queued_xuni, 0);

    assert_eq!(reopened.last_known_difficulty().unwrap(), Some(1727));

    // Appending the same key after reopen is still idempotent.
    assert_eq!(reopened.append(&keep).unwrap(), kept_id);
}

/// The journal-first invariant is exactly this: a find that `append` returned for is on disk
/// even if the process never closes the database. `mem::forget` leaks the connection without
/// running SQLite's close path, so the reopen below sees only what the fsync'd WAL holds.
#[test]
fn crash_without_clean_shutdown_keeps_appended_finds() {
    let tmp = TempDb::new();
    let find = payload("91");
    let id = {
        let journal = tmp.open();
        let id = journal.append(&find).unwrap();
        journal
            .record_attempt(
                id,
                &classify(FindStatus::Pending, "connect refused"),
                None,
                "",
                Some("2026-08-09T12:40:00Z"),
                NOW,
            )
            .unwrap();
        std::mem::forget(journal); // no close, no checkpoint: as if the process died here
        id
    };

    let reopened = tmp.open();
    let record = reopened.get_by_id(id).unwrap().expect("survived the crash");
    assert_eq!(record.payload, find);
    assert_eq!(record.status, FindStatus::Pending);
    assert_eq!(record.attempt_count, 1);
    // The backoff survives too — a crash must not silently re-arm a find.
    assert_eq!(
        record.next_attempt_at.as_deref(),
        Some("2026-08-09T12:40:00Z")
    );
}

#[test]
fn status_transitions_persist_across_reopen() {
    let tmp = TempDb::new();
    let transitions = [
        ("a1", FindStatus::AcceptedUnconfirmed),
        ("a2", FindStatus::Acked),
        ("a3", FindStatus::ParkedDifficulty),
        ("a4", FindStatus::ParkedXuniWindow),
        ("a5", FindStatus::Quarantined),
        ("a6", FindStatus::Dead),
    ];
    let mut ids = Vec::new();

    {
        let journal = tmp.open();
        for (suffix, status) in transitions {
            let id = journal.append(&payload(suffix)).unwrap();
            journal
                .record_attempt(
                    id,
                    &classify(status, "reason for the transition"),
                    Some(401),
                    "body",
                    None,
                    NOW,
                )
                .unwrap();
            ids.push((id, status));
        }
    }

    let reopened = tmp.open();
    for (id, status) in ids {
        let record = reopened.get_by_id(id).unwrap().expect("record present");
        assert_eq!(record.status, status);
        assert_eq!(record.status_reason, "reason for the transition");
        assert_eq!(record.attempt_count, 1);
        assert_eq!(record.last_http_status, Some(401));
        assert_eq!(record.last_response, "body");
        assert_eq!(record.last_attempt_at.as_deref(), Some(NOW));
    }
}

/// `Submitting` is the submitter's in-process lease and must never reach the disk: a crash
/// mid-attempt has to leave the row Pending and due, not stranded in a lease nobody holds.
#[test]
fn submitting_is_never_persisted() {
    let tmp = TempDb::new();
    {
        let journal = tmp.open();
        let id = journal.append(&payload("c1")).unwrap();
        // Every route that writes a status refuses it.
        assert!(journal
            .record_attempt(
                id,
                &classify(FindStatus::Submitting, "leased"),
                None,
                "",
                None,
                NOW
            )
            .is_err());
        // ...and the failed call did not partially apply.
        assert_eq!(
            journal.get_by_id(id).unwrap().unwrap().status,
            FindStatus::Pending
        );
    }

    let raw = tmp.raw();
    assert_eq!(
        raw_int(
            &raw,
            "SELECT COUNT(*) FROM finds WHERE status='Submitting';"
        ),
        0
    );
    // Nothing anywhere in the file, under any status column, spells the in-process state.
    let reopened = tmp.open();
    assert!(reopened
        .fetch_eligible(NOW, 100)
        .unwrap()
        .iter()
        .all(|r| r.status != FindStatus::Submitting));
}

#[test]
fn unsupported_schema_version_is_rejected() {
    let tmp = TempDb::new();
    drop(tmp.open());
    {
        let raw = tmp.raw();
        raw.execute_batch("UPDATE schema_version SET version = 99;")
            .unwrap();
    }
    match FindJournal::open(&tmp.path) {
        Err(JournalError::Schema(_)) => {}
        _ => panic!("must refuse a future schema"),
    }
}
