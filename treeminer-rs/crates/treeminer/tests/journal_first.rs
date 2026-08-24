//! End-to-end wiring of the journal-first pipeline: a find goes into a REAL SQLite journal
//! before any transport call, and the real `SubmissionManager` drains it through the bridge.
//!
//! The transport is a fake that answers in-process. Nothing here may touch xenblocks.io.

use std::sync::Arc;

use parking_lot::Mutex;
use tm_core::{FindKind, FoundPayload};
use tm_journal::{FallbackSink, FindJournal, Journal};
use tm_submit::{StepResult, SubmissionManager, Transport, TransportResult};
use treeminer::{Capture, Find, FindSink, JournalBridge, MiningState};

/// Scripted server. Records every call so "journal before network" can be asserted rather
/// than assumed.
#[derive(Default)]
struct FakeServer {
    calls: Mutex<Vec<String>>,
    verify: Mutex<Option<TransportResult>>,
    stored: Mutex<Vec<FoundPayload>>,
}

impl FakeServer {
    fn set_verify(&self, result: TransportResult) {
        *self.verify.lock() = Some(result);
    }

    fn calls(&self) -> Vec<String> {
        self.calls.lock().clone()
    }
}

impl Transport for FakeServer {
    fn submit(&self, payload: &FoundPayload) -> TransportResult {
        self.calls.lock().push(format!("verify:{}", payload.key));
        let scripted = self.verify.lock().clone();
        match scripted {
            Some(result) => {
                if result.http_status == 200 {
                    self.stored.lock().push(payload.clone());
                }
                result
            }
            None => TransportResult::failed("no scripted response"),
        }
    }

    fn confirm(&self, key: &str) -> TransportResult {
        self.calls.lock().push(format!("get_block:{key}"));
        let stored = self.stored.lock();
        match stored.iter().find(|payload| payload.key == key) {
            Some(payload) => TransportResult::ok(
                200,
                format!(
                    r#"{{"key":"{}","hash_to_verify":"{}","account":"{}"}}"#,
                    payload.key, payload.hash_to_verify, payload.account
                ),
            ),
            None => TransportResult::ok(404, r#"{"error":"not found"}"#),
        }
    }

    fn difficulty(&self) -> TransportResult {
        self.calls.lock().push("difficulty".to_owned());
        TransportResult::ok(200, r#"{"difficulty":"1000"}"#)
    }
}

struct Rig {
    _dir: tempfile::TempDir,
    journal: Arc<dyn Journal + Send + Sync>,
    sink: FindSink,
    server: Arc<FakeServer>,
    manager: Arc<SubmissionManager<Arc<JournalBridge>, Arc<FakeServer>>>,
    state: Arc<MiningState>,
}

fn rig() -> Rig {
    let dir = tempfile::tempdir().expect("tempdir");
    let journal: Arc<dyn Journal + Send + Sync> =
        Arc::new(FindJournal::open(dir.path().join("journal.db")).expect("journal opens"));
    let state = Arc::new(MiningState::for_test(1000));
    let sink = FindSink::new(
        Arc::clone(&journal),
        FallbackSink::new(dir.path().join("fallback.jsonl")),
        Arc::clone(&state),
        "worker-1",
    )
    .with_clock(|| "2026-01-01T00:00:00Z".to_owned())
    // Hand-written digests: the CPU cross-check is unit-tested in `find.rs`.
    .trusting_digests();
    let server = Arc::new(FakeServer::default());
    let manager = Arc::new(SubmissionManager::new(
        Arc::new(JournalBridge::new(Arc::clone(&journal))),
        Arc::clone(&server),
    ));
    Rig {
        _dir: dir,
        journal,
        sink,
        server,
        manager,
        state,
    }
}

fn find(memory_cost: u32, digest: &str) -> Find {
    Find {
        hexsalt: "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc".to_owned(),
        key: "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f".to_owned(),
        digest: digest.to_owned(),
        memory_cost,
        attempts: 7,
        hashes_per_second: 100.0,
        source: "GPU".to_owned(),
    }
}

#[test]
fn a_find_is_durable_before_the_first_network_call() {
    let rig = rig();
    rig.server.set_verify(TransportResult::ok(
        200,
        r#"{"message":"Hash verified successfully and block saved."}"#,
    ));

    let capture = rig.sink.record(&find(1000, "aaaXEN11bbb"));
    assert!(matches!(capture, Capture::Journaled(_)));
    assert!(
        rig.server.calls().is_empty(),
        "capture must not have talked to the server"
    );

    let record = rig.journal.get_by_id(1).expect("read").expect("row exists");
    assert_eq!(record.status, tm_core::FindStatus::Pending);
    assert_eq!(record.payload.memory_cost, 1000);
    assert!(record.payload.hash_to_verify.contains("m=1000,"));
}

#[test]
fn the_drain_submits_then_confirms_a_journaled_find() {
    let rig = rig();
    rig.server.set_verify(TransportResult::ok(
        200,
        r#"{"message":"Hash verified successfully and block saved."}"#,
    ));
    rig.sink.record(&find(1000, "aaaXEN11bbb"));

    // A 200 is never taken at face value: the server answers 200 even when its own insert
    // retries were exhausted, so the ack only lands after the /get_block lookup agrees.
    assert_eq!(rig.manager.run_once(), StepResult::Submitted);
    let record = rig.journal.get_by_id(1).expect("read").expect("row");
    assert_eq!(record.status, tm_core::FindStatus::Acked);

    let calls = rig.server.calls();
    assert_eq!(calls[0], format!("verify:{}", record.payload.key));
    assert_eq!(calls[1], format!("get_block:{}", record.payload.key));
}

#[test]
fn a_difficulty_rejection_parks_the_find_and_a_later_observation_unparks_it() {
    let rig = rig();
    rig.server.set_verify(TransportResult::ok(
        401,
        r#"{"message":"Hash does not contain 'm=1500'."}"#,
    ));
    rig.sink.record(&find(1200, "aaaXEN11bbb"));

    rig.manager.run_once();
    let record = rig.journal.get_by_id(1).expect("read").expect("row");
    assert_eq!(record.status, tm_core::FindStatus::ParkedDifficulty);

    // The poller (or the 401 hint itself) reports a difficulty the find is valid at again.
    let unparked = rig
        .manager
        .observe_difficulty(1200)
        .expect("observation is journaled");
    assert_eq!(unparked, 1, "the boundary case m == difficulty un-parks");
    let record = rig.journal.get_by_id(1).expect("read").expect("row");
    assert_eq!(record.status, tm_core::FindStatus::Pending);
}

#[test]
fn a_difficulty_hint_from_a_401_reaches_the_mining_state() {
    let rig = rig();
    let state = Arc::clone(&rig.state);
    rig.manager
        .set_difficulty_hint_callback(Arc::new(move |difficulty| {
            state.set_difficulty(difficulty);
        }));
    rig.server.set_verify(TransportResult::ok(
        401,
        r#"{"message":"Hash does not contain 'm=1500'."}"#,
    ));
    rig.sink.record(&find(1200, "aaaXEN11bbb"));

    rig.manager.run_once();
    assert_eq!(
        rig.state.difficulty(),
        1500,
        "the server's own number must not wait for a poll interval"
    );
}

#[test]
fn a_dead_network_leaves_the_find_pending_for_the_next_attempt() {
    let rig = rig();
    rig.server
        .set_verify(TransportResult::failed("connection refused"));
    rig.sink.record(&find(1000, "aaaXEN11bbb"));

    rig.manager.run_once();
    let record = rig.journal.get_by_id(1).expect("read").expect("row");
    assert_eq!(
        record.status,
        tm_core::FindStatus::Pending,
        "an outage must never consume a find"
    );
    assert!(record.next_attempt_at.is_some(), "backoff is persisted");
}

#[test]
fn a_xuni_find_survives_a_restart_of_the_process() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("journal.db");
    let state = Arc::new(MiningState::for_test(1000));

    {
        let journal: Arc<dyn Journal + Send + Sync> =
            Arc::new(FindJournal::open(&path).expect("journal opens"));
        let sink = FindSink::new(
            journal,
            FallbackSink::new(dir.path().join("fallback.jsonl")),
            Arc::clone(&state),
            "worker-1",
        )
        .with_clock(|| "2026-01-01T00:00:00Z".to_owned())
        .trusting_digests();
        sink.record(&find(1000, "aaaXUNI7bbb"));
    }

    let reopened = FindJournal::open(&path).expect("journal reopens");
    let recovered = reopened.recover_on_startup().expect("recovery");
    assert_eq!(recovered.pending, 1);
    let record = reopened.get_by_id(1).expect("read").expect("row");
    assert_eq!(record.payload.kind, FindKind::Xuni);
    assert_eq!(record.payload.memory_cost, 1000);
}
