//! Port of `tests/unit/submit/test_circuit_breaker.cpp` — deterministic via the injectable
//! monotonic clock and a zero-jitter source.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use tm_submit::breaker::{BreakerConfig, BreakerState, CircuitBreaker, Jitter};

struct Fixture {
    now: Arc<AtomicI64>,
}

impl Fixture {
    fn new(start: i64) -> Self {
        Self {
            now: Arc::new(AtomicI64::new(start)),
        }
    }
    fn set(&self, v: i64) {
        self.now.store(v, Ordering::SeqCst);
    }
    fn get(&self) -> i64 {
        self.now.load(Ordering::SeqCst)
    }
    fn breaker(&self, cfg: BreakerConfig, jitter: Option<Jitter>) -> CircuitBreaker {
        let now = Arc::clone(&self.now);
        CircuitBreaker::new(cfg, Arc::new(move || now.load(Ordering::SeqCst)), jitter)
    }
    fn default_breaker(&self) -> CircuitBreaker {
        self.breaker(BreakerConfig::default(), None)
    }
}

fn trip(b: &mut CircuitBreaker) {
    for _ in 0..3 {
        b.on_verify_transport_failure();
    }
}

#[test]
fn opens_after_three_consecutive_transport_failures() {
    let f = Fixture::new(0);
    let mut b = f.default_breaker();
    assert_eq!(b.state(), BreakerState::Closed);
    assert!(b.try_admit());
    b.on_verify_transport_failure();
    assert_eq!(b.state(), BreakerState::Closed);
    b.on_verify_transport_failure();
    assert_eq!(b.state(), BreakerState::Closed);
    b.on_verify_transport_failure();
    assert_eq!(b.state(), BreakerState::Open);
    assert!(!b.try_admit());
}

#[test]
fn success_or_conclusive_response_resets_the_failure_streak() {
    let f = Fixture::new(0);
    let mut b = f.default_breaker();
    b.on_verify_transport_failure();
    b.on_verify_transport_failure();
    b.on_verify_success(); // streak reset
    b.on_verify_transport_failure();
    b.on_verify_transport_failure();
    assert_eq!(b.state(), BreakerState::Closed);
    b.on_verify_inconclusive(); // a 401/4xx classification also resets
    b.on_verify_transport_failure();
    b.on_verify_transport_failure();
    assert_eq!(b.state(), BreakerState::Closed);
    b.on_verify_transport_failure();
    assert_eq!(b.state(), BreakerState::Open);
}

#[test]
fn probe_schedule_doubles_from_5s_and_caps_at_60s() {
    let f = Fixture::new(1000);
    let mut b = f.default_breaker();
    trip(&mut b);
    assert_eq!(b.state(), BreakerState::Open);
    assert_eq!(b.next_probe_at_ms(), 1000 + 5000);
    assert!(!b.probe_due());
    f.set(1000 + 5000);
    assert!(b.probe_due());
    for expected in [10_000, 20_000, 40_000, 60_000, 60_000] {
        b.on_probe_failure();
        assert_eq!(b.next_probe_at_ms(), f.get() + expected);
        f.set(b.next_probe_at_ms());
    }
}

#[test]
fn jitter_adds_up_to_jitter_fraction_of_the_interval() {
    let f = Fixture::new(0);
    let cfg = BreakerConfig {
        jitter_fraction: 0.2,
        ..BreakerConfig::default()
    };
    let mut b = f.breaker(cfg, Some(Arc::new(|| 0.5))); // mid-range jitter
    trip(&mut b);
    // 5000 + 0.5 * 0.2 * 5000 = 5500
    assert_eq!(b.next_probe_at_ms(), 5500);
}

#[test]
fn xuni_pressure_caps_probes_at_5s_and_pulls_in_far_probes() {
    let f = Fixture::new(0);
    let mut b = f.default_breaker();
    trip(&mut b);
    f.set(b.next_probe_at_ms());
    b.on_probe_failure(); // 10s
    f.set(b.next_probe_at_ms());
    b.on_probe_failure(); // 20s
    assert_eq!(b.next_probe_at_ms(), f.get() + 20_000);
    b.set_xuni_pressure(true); // an eligible XUNI appeared mid-outage
    assert!(b.next_probe_at_ms() <= f.get() + 5000);
    // Subsequent failed probes stay capped at 5s while pressure holds.
    f.set(b.next_probe_at_ms());
    b.on_probe_failure();
    assert!(b.next_probe_at_ms() <= f.get() + 5000);
    b.set_xuni_pressure(false);
    f.set(b.next_probe_at_ms());
    b.on_probe_failure();
    assert!(b.next_probe_at_ms() > f.get() + 5000); // back to the escalated interval
}

#[test]
fn half_open_admits_one_submission() {
    let f = Fixture::new(0);
    let mut b = f.default_breaker();
    trip(&mut b);
    f.set(b.next_probe_at_ms());
    assert!(b.probe_due());
    b.on_probe_success();
    assert_eq!(b.state(), BreakerState::HalfOpen);
    assert!(b.try_admit());
    assert!(!b.try_admit()); // only one until the outcome is reported
}

#[test]
fn half_open_closes_only_on_verification_success() {
    let f = Fixture::new(0);
    let mut b = f.default_breaker();
    trip(&mut b);
    f.set(b.next_probe_at_ms());
    b.on_probe_success();
    assert!(b.try_admit());
    b.on_verify_inconclusive(); // e.g. a 401 difficulty park: transport fine, NOT a close
    assert_eq!(b.state(), BreakerState::HalfOpen);
    assert!(b.try_admit()); // slot released for the next drain probe
    b.on_verify_success(); // 200 or conclusive duplicate
    assert_eq!(b.state(), BreakerState::Closed);
    assert!(b.try_admit());
}

#[test]
fn half_open_transport_failure_reopens_with_escalated_interval() {
    let f = Fixture::new(0);
    let mut b = f.default_breaker();
    trip(&mut b);
    f.set(b.next_probe_at_ms()); // 5000
    b.on_probe_success();
    assert!(b.try_admit());
    b.on_verify_transport_failure(); // the half-open probe failed
    assert_eq!(b.state(), BreakerState::Open);
    assert!(!b.try_admit());
    assert_eq!(b.next_probe_at_ms(), f.get() + 10_000); // escalated beyond the 5s base
}

#[test]
fn custom_threshold_is_honored() {
    let f = Fixture::new(0);
    let mut b = f.breaker(
        BreakerConfig {
            failure_threshold: 1,
            ..BreakerConfig::default()
        },
        None,
    );
    b.on_verify_transport_failure();
    assert_eq!(b.state(), BreakerState::Open);
    assert_eq!(b.consecutive_failures(), 1);
}
