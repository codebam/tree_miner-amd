//! Port of `tests/unit/submit/test_submission_manager.cpp` — driven synchronously through
//! `run_once()` with a scripted transport, the in-memory journal fake, and fully controlled
//! clocks (no sleeping, no I/O; the one threaded test waits on a channel the fatal callback
//! sends to).

mod common;

use std::sync::{Arc, Mutex};

use common::{block_row, down, ok, payload, Clocks, FakeJournal, FakeTransport, DUP_400, OK_200};
use tm_core::{Classification, FindKind, FindRecord, FindStatus};
use tm_submit::breaker::BreakerState;
use tm_submit::clocktime::{iso_utc, parse_http_date_ms, xuni_window_at};
use tm_submit::drain::DifficultyTrend;
use tm_submit::journal::JournalAccess;
use tm_submit::manager::{Config, ConfirmBodyCheck, StepResult, SubmissionManager};

type Manager = SubmissionManager<Arc<FakeJournal>, Arc<FakeTransport>>;

/// 2026-01-01T00:30:00Z — the XUNI window is CLOSED (XEN11-only scenarios).
const WALL_CLOSED_WINDOW: i64 = 1_767_227_400_000;
/// 2026-01-01T00:55:00Z — the window opens.
const WALL_WINDOW_OPENS: i64 = 1_767_228_900_000;

struct Fixture {
    journal: Arc<FakeJournal>,
    transport: Arc<FakeTransport>,
    clocks: Clocks,
}

impl Fixture {
    fn new() -> Self {
        Self {
            journal: Arc::new(FakeJournal::new()),
            transport: Arc::new(FakeTransport::new()),
            clocks: Clocks::default(),
        }
    }

    fn with_wall(epoch_ms: i64) -> Self {
        let f = Self::new();
        f.clocks.set_wall(epoch_ms);
        f
    }

    fn manager(&self) -> Manager {
        self.manager_with(Config::default())
    }

    fn manager_with(&self, cfg: Config) -> Manager {
        SubmissionManager::with_config(
            Arc::clone(&self.journal),
            Arc::clone(&self.transport),
            cfg,
            Some(self.clocks.mono_clock()),
            Some(self.clocks.wall_clock()),
            None,
        )
    }
}

fn collect_outcomes(m: &Manager) -> Arc<Mutex<Vec<FindStatus>>> {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&seen);
    m.set_outcome_callback(Arc::new(move |_r: &FindRecord, c: &Classification, _h| {
        sink.lock().expect("lock").push(c.next_status);
    }));
    seen
}

// --- pure time helpers ---

#[test]
fn iso_utc_formats_epoch_ms() {
    assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
    assert_eq!(iso_utc(784_111_777_000), "1994-11-06T08:49:37Z");
    assert_eq!(iso_utc(1_767_225_600_000), "2026-01-01T00:00:00Z");
}

#[test]
fn parse_http_date_ms_parses_imf_fixdate() {
    assert_eq!(
        parse_http_date_ms("Sun, 06 Nov 1994 08:49:37 GMT"),
        Some(784_111_777_000)
    );
    assert!(parse_http_date_ms("not a date").is_none());
    assert!(parse_http_date_ms("Sun, 06 Nov 1994 08:49:37 PST").is_none());
}

#[test]
fn xuni_window_at_models_the_55_to_05_server_window() {
    // minute 56 -> open, closes at :05 past the next hour.
    let w = xuni_window_at(56 * 60_000);
    assert!(w.open);
    assert_eq!(w.ms_until_close, 4 * 60_000 + 5 * 60_000);
    // minute 3 -> open, closes at :05.
    let w = xuni_window_at(3 * 60_000);
    assert!(w.open);
    assert_eq!(w.ms_until_close, 2 * 60_000);
    // minute 30 -> closed.
    assert!(!xuni_window_at(30 * 60_000).open);
    // boundary: exactly :55 open, exactly :05 closed.
    assert!(xuni_window_at(55 * 60_000).open);
    assert!(!xuni_window_at(5 * 60_000).open);
}

// --- happy path: 200 + /get_block confirmation -> Acked ---

#[test]
fn success_200_with_confirmed_lookup_becomes_acked() {
    let f = Fixture::new();
    let m = f.manager();
    let outcomes = collect_outcomes(&m);
    let p = payload("aa11", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&p)));

    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
    assert!(f.journal.record(id).confirmed_at.is_some());
    assert_eq!(f.transport.confirmed_keys(), vec!["aa11".to_string()]);
    assert_eq!(m.metrics().acked, 1);
    assert_eq!(m.metrics().reconciled_via_get_block, 1);
    assert_eq!(*outcomes.lock().expect("lock"), vec![FindStatus::Acked]);
}

// --- the lying-200: confirm 404 -> back to Pending ---

#[test]
fn success_200_with_absent_lookup_is_resubmitted() {
    let f = Fixture::new();
    let m = f.manager();
    let p = payload("aa22", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.transport.push_submit(ok(200, OK_200));
    f.transport
        .push_confirm(ok(404, r#"{"error": "Data not found for provided key"}"#));

    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.journal.record(id).status, FindStatus::Pending);
    assert!(f.journal.record(id).next_attempt_at.is_some());
    assert_eq!(m.metrics().lying_200_detected, 1);

    // Second pass (after the backoff) resubmits and this time it sticks.
    f.clocks.advance(10_000);
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&p)));
    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
    assert_eq!(m.metrics().submitted, 2);
    assert_eq!(m.metrics().resubmitted, 1);
}

#[test]
fn duplicate_response_confirms_via_lookup_to_acked() {
    let f = Fixture::new();
    let m = f.manager();
    let p = payload("aa33", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.transport.push_submit(ok(400, DUP_400));
    f.transport.push_confirm(ok(200, &block_row(&p)));

    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
}

#[test]
fn unavailable_lookup_leaves_accepted_unconfirmed() {
    let f = Fixture::new();
    let m = f.manager();
    let id = f.journal.append(payload("aa44", FindKind::Xen11, 100_000));
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(down());

    assert_eq!(m.run_once(), StepResult::Submitted);
    let r = f.journal.record(id);
    assert_eq!(r.status, FindStatus::AcceptedUnconfirmed);
    assert_eq!(m.metrics().accepted_unconfirmed, 1);
    assert!(r.status_reason.contains("unavailable"));
    // A backed-off next_attempt_at is persisted so the confirmation retry re-drives this
    // record later (and does not hot-loop it right now).
    assert_eq!(
        r.next_attempt_at.as_deref(),
        Some(iso_utc(f.clocks.wall_now() + 2000).as_str())
    );
    assert_eq!(f.transport.confirmed_keys().len(), 1);
}

#[test]
fn difficulty_401_parks_and_propagates_the_hint() {
    let f = Fixture::new();
    let m = f.manager();
    let hints = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&hints);
    m.set_difficulty_hint_callback(Arc::new(move |d| sink.lock().expect("lock").push(d)));
    let outcomes = collect_outcomes(&m);
    let id = f.journal.append(payload("aa55", FindKind::Xen11, 100_000));
    f.transport.push_submit(ok(
        401,
        r#"{"message": "Hash does not contain 'm=104000'. Your memory_cost setting in your miner will be autoadjusted."}"#,
    ));

    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.journal.record(id).status, FindStatus::ParkedDifficulty);
    assert_eq!(*hints.lock().expect("lock"), vec![104_000]);
    assert_eq!(m.last_observed_difficulty(), Some(104_000));
    assert_eq!(
        f.journal.difficulty_log().last().map(|(d, _)| *d),
        Some(104_000)
    );
    assert_eq!(
        *outcomes.lock().expect("lock"),
        vec![FindStatus::ParkedDifficulty]
    );
}

#[test]
fn rate_limit_429_retry_after_is_honored_in_next_attempt_at() {
    let f = Fixture::new();
    let m = f.manager();
    let id = f.journal.append(payload("aa66", FindKind::Xen11, 100_000));
    let mut r = ok(429, r#"{"message": "slow down"}"#);
    r.retry_after = Some("30".to_string());
    f.transport.push_submit(r);

    assert_eq!(m.run_once(), StepResult::Submitted);
    let rec = f.journal.record(id);
    assert_eq!(rec.status, FindStatus::Pending);
    assert_eq!(
        rec.next_attempt_at.as_deref(),
        Some(iso_utc(f.clocks.wall_now() + 30_000).as_str())
    );
}

#[test]
fn date_headers_feed_the_server_clock_offset() {
    let f = Fixture::new();
    let m = f.manager();
    assert!(m.server_clock_offset_ms().is_none());
    let p = payload("aa77", FindKind::Xen11, 100_000);
    f.journal.append(p.clone());
    let mut r = ok(200, OK_200);
    // The server is 90 s ahead of our wall clock.
    r.date_header = Some("Thu, 01 Jan 2026 00:01:30 GMT".to_string());
    f.transport.push_submit(r);
    f.transport.push_confirm(ok(200, &block_row(&p)));

    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(m.server_clock_offset_ms(), Some(90_000));
}

// --- outage: breaker opens, /difficulty probes, half-open drains, recovery at 1/s ---

#[test]
fn outage_opens_the_breaker_and_recovery_closes_it_through_a_real_submission() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let states = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&states);
    m.set_network_state_callback(Arc::new(move |s| sink.lock().expect("lock").push(s)));
    let p = payload("bb11", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());

    for i in 0..3 {
        if i > 0 {
            f.clocks.advance(60_000); // clear the per-record backoff and the pacing gate
        }
        f.transport.push_submit(down());
        assert_eq!(m.run_once(), StepResult::Submitted);
    }
    assert_eq!(m.breaker_state(), BreakerState::Open);
    assert_eq!(
        states.lock().expect("lock").last().copied(),
        Some(BreakerState::Open)
    );
    assert_eq!(m.metrics().transport_failures, 3);

    // Open: no /verify traffic; the probe is not due yet right after opening.
    assert_eq!(m.run_once(), StepResult::Idle);
    assert_eq!(f.transport.submitted_keys().len(), 3);

    // A failed probe escalates; a successful probe arms HalfOpen.
    f.clocks.advance(6000);
    f.transport.push_difficulty(down());
    assert_eq!(m.run_once(), StepResult::Probed);
    assert_eq!(m.breaker_state(), BreakerState::Open);
    f.clocks.advance(11_000);
    f.transport
        .push_difficulty(ok(200, r#"{"difficulty": "100000"}"#));
    assert_eq!(m.run_once(), StepResult::Probed);
    assert_eq!(m.breaker_state(), BreakerState::HalfOpen);
    assert_eq!(
        states.lock().expect("lock").last().copied(),
        Some(BreakerState::HalfOpen)
    );
    assert_eq!(m.last_observed_difficulty(), Some(100_000));

    // HalfOpen: one real queued submission; success closes and the drain restarts at 1/s.
    f.clocks.advance(60_000);
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&p)));
    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(m.breaker_state(), BreakerState::Closed);
    assert_eq!(
        states.lock().expect("lock").last().copied(),
        Some(BreakerState::Closed)
    );
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
    assert_eq!(m.drain_rate_per_second(), 1.0);
}

#[test]
fn adaptive_pacing_gates_back_to_back_submissions() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p1 = payload("cc11", FindKind::Xen11, 100_000);
    let p2 = payload("cc22", FindKind::Xen11, 100_000);
    f.journal.append(p1.clone());
    f.journal.append(p2.clone());
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&p1)));

    assert_eq!(m.run_once(), StepResult::Submitted);
    // Immediately after: the pacing gate holds (the rate is finite).
    assert_eq!(m.run_once(), StepResult::Idle);
    f.clocks.advance(1000);
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&p2)));
    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.transport.submitted_keys().len(), 2);
}

#[test]
fn window_opening_unparks_xuni_within_budget() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    assert_eq!(m.run_once(), StepResult::Idle); // closed window: no unpark
    assert_eq!(f.journal.unpark_xuni_calls(), 0);
    f.clocks.set_wall(WALL_WINDOW_OPENS);
    assert_eq!(m.run_once(), StepResult::Idle);
    assert_eq!(f.journal.unpark_xuni_calls(), 1);
    assert_eq!(m.run_once(), StepResult::Idle); // still open: no re-trigger
    assert_eq!(f.journal.unpark_xuni_calls(), 1);
}

#[test]
fn closed_window_xuni_backlog_does_not_hide_a_later_xen11() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager_with(Config {
        fetch_limit: 2,
        ..Config::default()
    });
    f.journal.append(payload("xuni-1", FindKind::Xuni, 100_000));
    f.journal.append(payload("xuni-2", FindKind::Xuni, 100_000));
    f.journal.append(payload("xuni-3", FindKind::Xuni, 100_000));
    let xen = payload("xen-1", FindKind::Xen11, 100_000);
    let xen_id = f.journal.append(xen.clone());
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&xen)));

    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.transport.submitted_keys().first().map(String::as_str), Some("xen-1"));
    assert_eq!(f.journal.record(xen_id).status, FindStatus::Acked);
}

#[test]
fn observed_difficulty_decrease_unparks_records() {
    let f = Fixture::new();
    let m = f.manager();
    // The FIRST observation un-parks even though there is no trend yet: a process that just
    // restarted has no last difficulty, and records parked before the restart may already be
    // valid again at the current floor.
    m.observe_difficulty(104_000).expect("journal ok");
    assert_eq!(m.difficulty_trend(), DifficultyTrend::Unknown);
    assert_eq!(f.journal.unpark_difficulty_calls(), vec![104_000]);

    // A rise cannot re-qualify anything, so it must not touch the journal.
    m.observe_difficulty(106_000).expect("journal ok");
    assert_eq!(m.difficulty_trend(), DifficultyTrend::Rising);
    assert_eq!(f.journal.unpark_difficulty_calls().len(), 1);

    // A fall re-qualifies every parked record with m >= the new floor.
    m.observe_difficulty(100_000).expect("journal ok");
    assert_eq!(m.difficulty_trend(), DifficultyTrend::Falling);
    assert_eq!(f.journal.unpark_difficulty_calls(), vec![104_000, 100_000]);
}

// --- confirmation-retry drain ---

#[test]
fn confirmation_retry_succeeds_to_acked() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p = payload("dd11", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.journal.set_status(id, FindStatus::AcceptedUnconfirmed); // earlier 200, lookup failed
    f.transport.push_confirm(ok(200, &block_row(&p)));

    assert_eq!(m.run_once(), StepResult::ConfirmRetried); // no Pending work: retry only
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
    assert!(f.journal.record(id).confirmed_at.is_some());
    assert!(f.transport.submitted_keys().is_empty()); // never re-POSTs an unconfirmed record
    assert_eq!(m.metrics().confirmation_retries, 1);
    assert_eq!(m.metrics().reconciled_via_get_block, 1);
}

#[test]
fn confirmation_retry_finds_404_and_re_pends() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p = payload("dd22", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.journal.set_status(id, FindStatus::AcceptedUnconfirmed);
    f.transport
        .push_confirm(ok(404, r#"{"error": "Data not found for provided key"}"#));

    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    assert_eq!(f.journal.record(id).status, FindStatus::Pending);
    assert!(f.journal.record(id).next_attempt_at.is_some());
    assert_eq!(m.metrics().lying_200_detected, 1);

    // After the backoff it re-enters the normal submission drain.
    f.clocks.advance(10_000);
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&p)));
    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
}

#[test]
fn confirmation_retry_transport_down_stays_unconfirmed_with_future_backoff() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p = payload("dd33", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.journal.set_status(id, FindStatus::AcceptedUnconfirmed);
    f.transport.push_confirm(down());

    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    assert_eq!(f.journal.record(id).status, FindStatus::AcceptedUnconfirmed);
    assert_eq!(
        f.journal.record(id).next_attempt_at.as_deref(),
        Some(iso_utc(f.clocks.wall_now() + 2000).as_str())
    );
    assert_eq!(f.transport.confirmed_keys().len(), 1);
    // The backoff holds: an immediate second step issues no further lookups.
    assert_eq!(m.run_once(), StepResult::Idle);
    assert_eq!(f.transport.confirmed_keys().len(), 1);
    // Past the backoff the retry lands and the record confirms.
    f.clocks.advance(3000);
    f.transport.push_confirm(ok(200, &block_row(&p)));
    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
}

#[test]
fn breaker_open_suppresses_confirmation_retries() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p1 = payload("ee11", FindKind::Xen11, 100_000);
    f.journal.append(p1.clone());
    for i in 0..3 {
        if i > 0 {
            f.clocks.advance(60_000);
        }
        f.transport.push_submit(down());
        assert_eq!(m.run_once(), StepResult::Submitted);
    }
    assert_eq!(m.breaker_state(), BreakerState::Open);

    // An unconfirmed record eligible right now...
    let p2 = payload("ee22", FindKind::Xen11, 100_000);
    let u = f.journal.append(p2.clone());
    f.journal.set_status(u, FindStatus::AcceptedUnconfirmed);
    // ...is NOT probed while the breaker is Open (the probe is not due yet either).
    assert_eq!(m.run_once(), StepResult::Idle);
    assert!(f.transport.confirmed_keys().is_empty());

    // Once the breaker recovers, the confirmation retry drains again.
    f.clocks.advance(6000);
    f.transport
        .push_difficulty(ok(200, r#"{"difficulty": "100000"}"#));
    assert_eq!(m.run_once(), StepResult::Probed); // HalfOpen now
    f.clocks.advance(60_000);
    f.transport.push_submit(ok(200, OK_200)); // half-open probe: the Pending record
    f.transport.push_confirm(ok(200, &block_row(&p1)));
    f.transport.push_confirm(ok(200, &block_row(&p2))); // then the retry for ee22
    assert_eq!(m.run_once(), StepResult::Submitted);
    assert_eq!(f.journal.record(u).status, FindStatus::Acked);
}

// --- head-of-line blocking: neither kind may starve the other out of the fetch slice ---

#[test]
fn xen11_drains_past_a_deep_closed_window_xuni_backlog() {
    // Regression: a single mixed oldest-first LIMIT slice full of closed-window XUNI left
    // the XEN11 undelivered until the next window (up to ~50 minutes) against a healthy
    // server.
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    // 20 XUNI (> the fetch limit of 16) journaled BEFORE the one XEN11: worst-case ordering.
    for i in 0..20 {
        f.journal.append(payload(&format!("xu{i:02}"), FindKind::Xuni, 100_000));
    }
    let xen = payload("xen1", FindKind::Xen11, 100_000);
    f.journal.append(xen.clone());
    f.transport.push_submit(ok(200, r#"{"message": "Block added"}"#));
    f.transport.push_confirm(ok(200, &block_row(&xen)));

    assert_eq!(m.run_once(), StepResult::Submitted); // was Idle before the fix
    assert_eq!(f.transport.submitted_keys(), vec!["xen1".to_string()]);
    // The closed-window XUNI stayed Pending — parked in place, not starved and not dead.
    assert_eq!(f.journal.counts().expect("counts").pending, 20);
}

#[test]
fn closing_window_xuni_preempts_past_a_deep_xen11_backlog() {
    // The symmetric direction: a XEN11 backlog deeper than the fetch limit used to hide any
    // XUNI from the slice entirely, so the preemption rule had nothing to act on.
    let f = Fixture::with_wall(1_767_225_600_000 + 4 * 60 * 1000); // 00:04, closes at :05
    let m = f.manager();
    for i in 0..20 {
        f.journal.append(payload(&format!("xe{i:02}"), FindKind::Xen11, 100_000));
    }
    let xuni = payload("xuni1", FindKind::Xuni, 100_000);
    f.journal.append(xuni.clone());
    f.transport.push_submit(ok(200, r#"{"message": "Block added"}"#));
    f.transport.push_confirm(ok(200, &block_row(&xuni)));

    assert_eq!(m.run_once(), StepResult::Submitted);
    // With <=60 s to the window close the XUNI goes first even though 20 older XEN11 exist —
    // they remain valid after :05; the XUNI does not.
    assert_eq!(f.transport.submitted_keys(), vec!["xuni1".to_string()]);
}

// =============================================================================
// A confirmation 200 is only an ack when its body proves our row.
// =============================================================================

#[test]
fn confirmation_matches_validates_key_and_hash_byte_for_byte() {
    let mut rec = FindRecord::new(payload("aabb", FindKind::Xen11, 100_000));
    rec.id = 1;
    let check = |body: &str| Manager::confirmation_matches(&rec, body);
    // The real stored row confirms; hash_to_verify may legitimately be absent, and then the
    // key alone decides.
    assert_eq!(check(&block_row(&rec.payload)), ConfirmBodyCheck::Confirmed);
    assert_eq!(check(r#"{"key": "aabb"}"#), ConfirmBodyCheck::Confirmed);
    // Non-JSON, empty, HTML error pages, arrays, and key-less objects are all Malformed: a
    // body that identifies nothing can confirm nothing.
    assert_eq!(check(""), ConfirmBodyCheck::Malformed);
    assert_eq!(check("<html>502</html>"), ConfirmBodyCheck::Malformed);
    assert_eq!(check("OK"), ConfirmBodyCheck::Malformed);
    assert_eq!(check(r#"[{"key": "aabb"}]"#), ConfirmBodyCheck::Malformed);
    assert_eq!(check(r#"{"block_id": 7}"#), ConfirmBodyCheck::Malformed);
    // A different key is some other row. Byte equality: case differences are mismatches too.
    assert_eq!(check(r#"{"key": "ffff"}"#), ConfirmBodyCheck::KeyMismatch);
    assert_eq!(check(r#"{"key": "AABB"}"#), ConfirmBodyCheck::KeyMismatch);
    // Our key with a different stored hash is the serious case.
    assert_eq!(
        check(r#"{"key": "aabb", "hash_to_verify": "$argon2id$other"}"#),
        ConfirmBodyCheck::HashMismatch
    );
}

#[test]
fn initial_confirm_200_with_wrong_key_stays_unconfirmed_then_recovers() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p = payload("ff11", FindKind::Xen11, 100_000);
    let other = payload("attacker", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&other))); // 200, but not OUR row

    assert_eq!(m.run_once(), StepResult::Submitted);
    // Not Acked (nothing proven), not Pending (the server may hold the row): unconfirmed
    // with the normal per-record backoff in the future.
    let rec = f.journal.record(id);
    assert_eq!(rec.status, FindStatus::AcceptedUnconfirmed);
    assert_eq!(
        rec.next_attempt_at.as_deref(),
        Some(iso_utc(f.clocks.wall_now() + 2000).as_str())
    );
    assert_eq!(m.metrics().acked, 0);
    assert_eq!(m.metrics().reconciled_via_get_block, 0);
    assert_eq!(m.metrics().confirm_body_rejected, 1);
    assert!(rec.status_reason.contains("different key"));

    // The retry path re-asks after the backoff; a genuine row then acks it.
    f.clocks.advance(3000);
    f.transport.push_confirm(ok(200, &block_row(&p)));
    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
}

#[test]
fn initial_confirm_200_with_garbage_body_stays_unconfirmed() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let id = f.journal.append(payload("ff22", FindKind::Xen11, 100_000));
    f.transport.push_submit(ok(200, OK_200));
    f.transport
        .push_confirm(ok(200, "<html><body>captive portal</body></html>"));

    assert_eq!(m.run_once(), StepResult::Submitted);
    let rec = f.journal.record(id);
    assert_eq!(rec.status, FindStatus::AcceptedUnconfirmed);
    assert!(rec.next_attempt_at.is_some());
    assert_eq!(m.metrics().acked, 0);
    assert_eq!(m.metrics().confirm_body_rejected, 1);
}

#[test]
fn initial_confirm_200_missing_the_key_field_stays_unconfirmed() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let id = f.journal.append(payload("ff33", FindKind::Xen11, 100_000));
    f.transport.push_submit(ok(200, OK_200));
    f.transport
        .push_confirm(ok(200, r#"{"block_id": 7, "account": "0x1111"}"#));

    assert_eq!(m.run_once(), StepResult::Submitted);
    let rec = f.journal.record(id);
    assert_eq!(rec.status, FindStatus::AcceptedUnconfirmed);
    assert!(rec.next_attempt_at.is_some());
    assert_eq!(m.metrics().confirm_body_rejected, 1);
}

#[test]
fn initial_confirm_matching_key_but_mismatched_hash_is_never_acked() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p = payload("ff44", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    let mut impostor = p.clone(); // same key, different stored hash: NOT our find
    impostor.hash_to_verify = "$argon2id$v=19$m=100000,t=1,p=1$saltsalt$SOMEONEELSE".to_string();
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&impostor)));

    assert_eq!(m.run_once(), StepResult::Submitted);
    let rec = f.journal.record(id);
    assert_eq!(rec.status, FindStatus::AcceptedUnconfirmed);
    assert!(rec.next_attempt_at.is_some());
    assert_eq!(m.metrics().acked, 0);
    assert_eq!(m.metrics().confirm_body_rejected, 1);
    assert!(rec.status_reason.contains("DIFFERENT hash_to_verify"));
}

#[test]
fn confirm_retry_200_with_wrong_key_stays_unconfirmed_with_backoff() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p = payload("gg11", FindKind::Xen11, 100_000);
    let other = payload("attacker", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.journal.set_status(id, FindStatus::AcceptedUnconfirmed);
    f.transport.push_confirm(ok(200, &block_row(&other)));

    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    let rec = f.journal.record(id);
    assert_eq!(rec.status, FindStatus::AcceptedUnconfirmed);
    assert_eq!(
        rec.next_attempt_at.as_deref(),
        Some(iso_utc(f.clocks.wall_now() + 2000).as_str())
    );
    assert_eq!(m.metrics().acked, 0);
    assert_eq!(m.metrics().confirm_body_rejected, 1);

    // Past the backoff, a genuine row still acks it.
    f.clocks.advance(3000);
    f.transport.push_confirm(ok(200, &block_row(&p)));
    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    assert_eq!(f.journal.record(id).status, FindStatus::Acked);
}

#[test]
fn confirm_retry_garbage_and_hash_mismatch_200s_stay_unconfirmed() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    let m = f.manager();
    let p = payload("gg22", FindKind::Xen11, 100_000);
    let id = f.journal.append(p.clone());
    f.journal.set_status(id, FindStatus::AcceptedUnconfirmed);
    f.transport.push_confirm(ok(200, "not json at all"));

    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    assert_eq!(f.journal.record(id).status, FindStatus::AcceptedUnconfirmed);
    assert!(f.journal.record(id).next_attempt_at.is_some());

    // And the serious flavor on the retry path too: same key, different hash.
    let mut impostor = p.clone();
    impostor.hash_to_verify = "$argon2id$v=19$m=100000,t=1,p=1$saltsalt$SOMEONEELSE".to_string();
    f.clocks.advance(3000);
    f.transport.push_confirm(ok(200, &block_row(&impostor)));
    assert_eq!(m.run_once(), StepResult::ConfirmRetried);
    assert_eq!(f.journal.record(id).status, FindStatus::AcceptedUnconfirmed);
    assert_eq!(m.metrics().acked, 0);
    assert_eq!(m.metrics().confirm_body_rejected, 2);
}

// =============================================================================
// A journal failure inside the drain step must never escape or spin.
// =============================================================================

#[test]
fn journal_failure_is_contained_fatal_fires_once_and_the_loop_goes_inert() {
    let f = Fixture::with_wall(WALL_CLOSED_WINDOW);
    f.journal.append(payload("hh11", FindKind::Xen11, 100_000));
    *f.journal.fail_record_attempt.lock().expect("lock") = true;
    let m = f.manager();
    let fatals = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&fatals);
    m.set_fatal_callback(Arc::new(move |what: &str| {
        sink.lock().expect("lock").push(what.to_string())
    }));
    let p = payload("hh11", FindKind::Xen11, 100_000);
    f.transport.push_submit(ok(200, OK_200));
    f.transport.push_confirm(ok(200, &block_row(&p)));

    assert_eq!(m.run_once(), StepResult::Idle); // record_attempt failed inside
    assert_eq!(f.journal.throw_count(), 1);
    assert_eq!(fatals.lock().expect("lock").len(), 1);
    assert!(fatals.lock().expect("lock")[0].contains("disk I/O error"));
    assert_eq!(m.metrics().thread_loop_exceptions, 1);

    // First failure wins: the manager is inert now — no further journal/transport work, no
    // second callback.
    f.clocks.advance(60_000);
    assert_eq!(m.run_once(), StepResult::Idle);
    assert_eq!(f.journal.throw_count(), 1);
    assert_eq!(fatals.lock().expect("lock").len(), 1);
    // stop() is safe even though start() was never called.
    m.stop();
}

#[test]
fn thread_loop_survives_a_journal_failure_and_stop_joins_cleanly() {
    // The one threaded test: the real drain loop hits the failing journal on its first step,
    // halts itself, and stays joinable. The fatal callback sends on a channel, so the wait is
    // event-driven — no sleeps, no polling.
    let journal = Arc::new(FakeJournal::throwing());
    let transport = Arc::new(FakeTransport::new());
    let p = payload("hh22", FindKind::Xen11, 100_000);
    journal.append(p.clone());
    transport.push_submit(ok(200, OK_200));
    transport.push_confirm(ok(200, &block_row(&p)));

    let m = Arc::new(SubmissionManager::new(
        Arc::clone(&journal),
        Arc::clone(&transport),
    ));
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    let tx = Mutex::new(tx);
    m.set_fatal_callback(Arc::new(move |what: &str| {
        let _ = tx.lock().expect("lock").send(what.to_string());
    }));
    m.start();
    let fatal = rx
        .recv_timeout(std::time::Duration::from_secs(10))
        .expect("fatal callback fired");
    assert!(fatal.contains("disk I/O error"));
    // The loop halted itself; stop() must still join the finished-but-joinable thread.
    m.stop();
    assert_eq!(m.metrics().thread_loop_exceptions, 1);
    assert_eq!(journal.throw_count(), 1);
}
