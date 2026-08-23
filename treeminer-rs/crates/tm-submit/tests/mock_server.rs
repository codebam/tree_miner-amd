//! End-to-end drain against a real HTTP server, with fault injection. Rust equivalent of
//! `tests/mockserver/mock_server.py`: same endpoints, same response strings, same status
//! codes as the reference `gpage.py`, plus the fault modes the chaos tests need.
//!
//! Covered here (each is a behaviour the journal exists for):
//!   * a server that answers 200 but does not store the block (the lying-200),
//!   * a 401 difficulty rejection, including the `m={N}` hint and the later un-park,
//!   * a hard outage (nothing listening) followed by recovery,
//!   * a duplicate submission of a block the server already holds.

mod common;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use common::{payload, Clocks, FakeJournal};
use tm_core::{FindKind, FindStatus};
use tm_submit::breaker::BreakerState;
use tm_submit::http::HttpTransport;
use tm_submit::manager::{Config, StepResult, SubmissionManager};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Normal,
    /// `/verify` returns 200 success WITHOUT storing — `/get_block` will 404.
    InsertFail,
    /// `/verify` rejects everything as below the current difficulty.
    DifficultyTooLow,
}

struct MockState {
    mode: Mutex<Mode>,
    difficulty: AtomicU32,
    /// key -> hash_to_verify
    blocks: Mutex<HashMap<String, String>>,
    verify_requests: AtomicU32,
}

impl MockState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            mode: Mutex::new(Mode::Normal),
            difficulty: AtomicU32::new(100_000),
            blocks: Mutex::new(HashMap::new()),
            verify_requests: AtomicU32::new(0),
        })
    }
    fn set_mode(&self, mode: Mode) {
        *self.mode.lock().expect("lock") = mode;
    }
    fn mode(&self) -> Mode {
        *self.mode.lock().expect("lock")
    }
}

async fn verify(State(s): State<Arc<MockState>>, Json(body): Json<Value>) -> impl IntoResponse {
    s.verify_requests.fetch_add(1, Ordering::SeqCst);
    let key = body
        .get("key")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let hash = body
        .get("hash_to_verify")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if key.is_empty() || hash.is_empty() {
        // gpage.py:398
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Missing hash_to_verify, key, or account"})),
        );
    }
    if s.mode() == Mode::DifficultyTooLow {
        // gpage.py:416 — N is the server's CURRENT difficulty.
        let n = s.difficulty.load(Ordering::SeqCst);
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"message": format!(
                "Hash does not contain 'm={n}'. Your memory_cost setting in your miner will be autoadjusted."
            )})),
        );
    }
    if s.blocks.lock().expect("lock").contains_key(&key) {
        // gpage.py:510 — UNIQUE-key duplicate.
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"message": "Block already exists, continue"})),
        );
    }
    if s.mode() != Mode::InsertFail {
        s.blocks.lock().expect("lock").insert(key, hash);
    }
    // gpage.py:515 — returned in insert-fail mode too, with nothing stored: the lying-200.
    (
        StatusCode::OK,
        Json(json!({"message": "Hash verified successfully and block saved."})),
    )
}

async fn get_block(
    State(s): State<Arc<MockState>>,
    Query(q): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    let Some(key) = q.get("key") else {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Invalid key format"})),
        );
    };
    match s.blocks.lock().expect("lock").get(key) {
        // gpage.py:331-364 — a 200 body IS the stored row.
        Some(hash) => (
            StatusCode::OK,
            Json(json!({
                "block_id": 7,
                "hash_to_verify": hash,
                "key": key,
                "account": "0x1111111111111111111111111111111111111111",
                "created_at": "2026-01-01 00:00:00",
            })),
        ),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "Data not found for provided key"})),
        ),
    }
}

async fn difficulty(State(s): State<Arc<MockState>>) -> impl IntoResponse {
    // gpage.py:109-117 — note the JSON *string*.
    Json(json!({"difficulty": s.difficulty.load(Ordering::SeqCst).to_string()}))
}

/// Grab a port the OS just handed out, then release it. The window between release and the
/// server's own bind is the same one every "pick a free port" helper lives with.
fn free_addr() -> SocketAddr {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    listener.local_addr().expect("addr")
}

fn serve(addr: SocketAddr, state: Arc<MockState>) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async move {
            let app = Router::new()
                .route("/verify", post(verify))
                .route("/get_block", get(get_block))
                .route("/difficulty", get(difficulty))
                .with_state(state);
            let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
            axum::serve(listener, app).await.expect("serve");
        });
    })
}

struct Harness {
    state: Arc<MockState>,
    addr: SocketAddr,
    journal: Arc<FakeJournal>,
    clocks: Clocks,
    manager: SubmissionManager<Arc<FakeJournal>, HttpTransport>,
}

impl Harness {
    /// `listening == false` starts with nothing bound: a hard outage.
    fn new(listening: bool) -> Self {
        let state = MockState::new();
        let addr = free_addr();
        if listening {
            serve(addr, Arc::clone(&state));
        }
        let journal = Arc::new(FakeJournal::new());
        let clocks = Clocks::default();
        clocks.set_wall(1_767_227_400_000); // 00:30 — the XUNI window is closed
        let transport =
            HttpTransport::new(&format!("http://{addr}"), "w1", 5000, 2000).expect("client");
        let manager = SubmissionManager::with_config(
            Arc::clone(&journal),
            transport,
            Config::default(),
            Some(clocks.mono_clock()),
            Some(clocks.wall_clock()),
            None,
        );
        let h = Self {
            state,
            addr,
            journal,
            clocks,
            manager,
        };
        if listening {
            h.await_ready();
        }
        h
    }

    fn start_server(&self) {
        serve(self.addr, Arc::clone(&self.state));
        self.await_ready();
    }

    fn await_ready(&self) {
        let url = format!("http://{}/difficulty", self.addr);
        for _ in 0..200 {
            if reqwest::blocking::get(&url).is_ok() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(25));
        }
        panic!("mock server never became ready");
    }
}

#[test]
fn lying_200_is_caught_by_the_confirmation_lookup_and_resubmitted() {
    let h = Harness::new(true);
    h.state.set_mode(Mode::InsertFail);
    let id = h.journal.append(payload(
        "1111111111111111111111111111111111111111111111111111111111111111",
        FindKind::Xen11,
        100_000,
    ));

    // The server answers 200 and stores nothing; /get_block 404s, so the record must come
    // back to Pending rather than being counted as a payout.
    assert_eq!(h.manager.run_once(), StepResult::Submitted);
    assert_eq!(h.journal.record(id).status, FindStatus::Pending);
    assert_eq!(h.manager.metrics().lying_200_detected, 1);
    assert!(h.journal.record(id).next_attempt_at.is_some());

    // With the fault cleared the retry lands and confirms.
    h.state.set_mode(Mode::Normal);
    h.clocks.advance(10_000);
    assert_eq!(h.manager.run_once(), StepResult::Submitted);
    assert_eq!(h.journal.record(id).status, FindStatus::Acked);
    assert_eq!(h.manager.metrics().acked, 1);
    assert_eq!(h.manager.metrics().reconciled_via_get_block, 1);
}

#[test]
fn difficulty_401_parks_the_find_and_a_later_drop_re_pends_it() {
    let h = Harness::new(true);
    h.state.set_mode(Mode::DifficultyTooLow);
    h.state.difficulty.store(104_000, Ordering::SeqCst);
    let id = h.journal.append(payload(
        "2222222222222222222222222222222222222222222222222222222222222222",
        FindKind::Xen11,
        100_000,
    ));

    assert_eq!(h.manager.run_once(), StepResult::Submitted);
    assert_eq!(h.journal.record(id).status, FindStatus::ParkedDifficulty);
    // The hint rides in on the rejection body: no waiting for the difficulty poller.
    assert_eq!(h.manager.last_observed_difficulty(), Some(104_000));
    assert_eq!(h.manager.metrics().parked_difficulty, 1);

    // A parked find is not a lost find: when the floor falls back to its m it re-pends and
    // the next drain step submits it for real.
    h.state.set_mode(Mode::Normal);
    h.state.difficulty.store(100_000, Ordering::SeqCst);
    h.manager.observe_difficulty(100_000).expect("journal ok");
    assert_eq!(h.journal.record(id).status, FindStatus::Pending);
    h.clocks.advance(10_000);
    assert_eq!(h.manager.run_once(), StepResult::Submitted);
    assert_eq!(h.journal.record(id).status, FindStatus::Acked);
}

#[test]
fn hard_outage_then_recovery_drains_the_backlog() {
    let h = Harness::new(false); // nothing is listening: connections are refused
    let id = h.journal.append(payload(
        "3333333333333333333333333333333333333333333333333333333333333333",
        FindKind::Xen11,
        100_000,
    ));

    for i in 0..3 {
        if i > 0 {
            h.clocks.advance(60_000); // clear the per-record backoff and the pacing gate
        }
        assert_eq!(h.manager.run_once(), StepResult::Submitted);
    }
    assert_eq!(h.manager.breaker_state(), BreakerState::Open);
    assert_eq!(h.manager.metrics().transport_failures, 3);
    // The find is untouched and still durable — nothing was dropped during the outage.
    assert_eq!(h.journal.record(id).status, FindStatus::Pending);

    // Open: no /verify traffic at all until the probe falls due.
    assert_eq!(h.manager.run_once(), StepResult::Idle);

    h.start_server();
    h.clocks.advance(6000);
    assert_eq!(h.manager.run_once(), StepResult::Probed);
    assert_eq!(h.manager.breaker_state(), BreakerState::HalfOpen);
    assert_eq!(h.manager.last_observed_difficulty(), Some(100_000));

    // Half-open admits exactly one real submission; success closes the breaker and the
    // recovery drain restarts at 1/s so a just-restored server is not stampeded.
    h.clocks.advance(60_000);
    assert_eq!(h.manager.run_once(), StepResult::Submitted);
    assert_eq!(h.journal.record(id).status, FindStatus::Acked);
    assert_eq!(h.manager.breaker_state(), BreakerState::Closed);
    assert_eq!(h.manager.drain_rate_per_second(), 1.0);
}

#[test]
fn duplicate_submission_confirms_instead_of_being_lost() {
    let h = Harness::new(true);
    let key = "4444444444444444444444444444444444444444444444444444444444444444";
    let p = payload(key, FindKind::Xen11, 100_000);
    // The server already holds this block: a previous attempt landed before we crashed.
    h.state
        .blocks
        .lock()
        .expect("lock")
        .insert(key.to_string(), p.hash_to_verify.clone());
    let id = h.journal.append(p);

    // 400 "Block already exists" is an ack in disguise — confirmed through /get_block.
    assert_eq!(h.manager.run_once(), StepResult::Submitted);
    assert_eq!(h.journal.record(id).status, FindStatus::Acked);
    assert_eq!(h.journal.record(id).last_http_status, Some(400));
    assert_eq!(h.manager.metrics().acked, 1);
    assert_eq!(h.state.verify_requests.load(Ordering::SeqCst), 1);
}
