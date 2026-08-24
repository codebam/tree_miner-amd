//! The server-clock offset the submitter learns has to reach the mining loop.
//!
//! The XUNI window is the server's, checked against the server's clock (`gpage.py:36-40`).
//! `clock::now_server_ms` is what both the mining loop's gate and the submitter's window
//! check read — but the offset it applies is only ever *published* by the difficulty
//! poller's observer, so this asserts that hand-off rather than assuming it.
//!
//! The transport is a fake that answers in-process. Nothing here may touch xenblocks.io.

use std::sync::Arc;

use parking_lot::Mutex;
use tm_core::{FindKind, FoundPayload};
use tm_journal::{FindJournal, Journal};
use tm_submit::{StepResult, SubmissionManager, Transport, TransportResult};
use treeminer::JournalBridge;

/// Answers `/verify` with a `Date` header, which is where the offset comes from.
#[derive(Default)]
struct DatedServer {
    date: Mutex<Option<String>>,
}

impl Transport for DatedServer {
    fn submit(&self, _payload: &FoundPayload) -> TransportResult {
        let mut result = TransportResult::ok(
            200,
            r#"{"message":"Hash verified successfully and block saved."}"#,
        );
        result.date_header = self.date.lock().clone();
        result
    }

    fn confirm(&self, _key: &str) -> TransportResult {
        TransportResult::ok(404, r#"{"error":"not found"}"#)
    }

    fn difficulty(&self) -> TransportResult {
        TransportResult::ok(200, r#"{"difficulty":"1000"}"#)
    }
}

fn payload() -> FoundPayload {
    FoundPayload {
        key: "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f".to_owned(),
        hash_to_verify: "$argon2id$v=19$m=1000,t=1,p=1$ZTRiYg$abcXEN11def".to_owned(),
        account: "0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc".to_owned(),
        kind: FindKind::Xen11,
        memory_cost: 1000,
        worker: "worker-1".to_owned(),
        attempts: 7,
        hashes_per_second: 100.0,
        found_at_utc: "2026-01-01T00:00:00Z".to_owned(),
    }
}

/// This is the whole point of the wiring: before ec44aa4's follow-up nothing called
/// `set_server_offset_ms`, so the miner and the submitter agreed on plain UTC and both were
/// wrong by the server's real skew.
#[test]
fn the_poller_observer_publishes_the_submitter_s_learned_offset() {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal: Arc<dyn Journal + Send + Sync> =
        Arc::new(FindJournal::open(dir.path().join("journal.db")).expect("journal opens"));
    journal.append(&payload()).expect("append");

    let server = Arc::new(DatedServer::default());
    let manager = SubmissionManager::new(
        Arc::new(JournalBridge::new(Arc::clone(&journal))),
        Arc::clone(&server),
    );

    // Nothing learned yet: the miner must read plain UTC, which is what keeps it exactly in
    // step with the submitter.
    treeminer::clock::set_server_offset_ms(None);
    treeminer::run::observe_difficulty_and_clock(&manager, 1000);
    assert_eq!(treeminer::clock::server_offset_ms(), None);

    // A dated response teaches the manager an offset...
    *server.date.lock() = Some("Thu, 01 Jan 2036 00:00:00 GMT".to_owned());
    assert_eq!(manager.run_once(), StepResult::Submitted);
    let learned = manager
        .server_clock_offset_ms()
        .expect("a dated response sets the offset");
    assert!(learned > 0, "the fixture server's clock is a decade ahead");

    // ...and the next poll observation publishes it to the mining loop.
    treeminer::run::observe_difficulty_and_clock(&manager, 1000);
    assert_eq!(
        treeminer::clock::server_offset_ms(),
        Some(learned),
        "the mining loop's XUNI gate must read the SAME offset the submitter uses"
    );
    let expected = tm_submit::clocktime::now_wall_ms() + learned;
    assert!(
        (treeminer::clock::now_server_ms() - expected).abs() < 5_000,
        "now_server_ms must move with the published offset"
    );

    // And the observation itself still reaches the submitter, which is what un-parks finds.
    assert_eq!(manager.last_observed_difficulty(), Some(1000));

    treeminer::clock::set_server_offset_ms(None);
}
