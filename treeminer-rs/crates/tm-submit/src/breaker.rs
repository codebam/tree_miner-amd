//! Circuit breaker protecting the `/verify` path during server outages. Port of
//! `src/submit/CircuitBreaker.{h,cpp}`.
//!
//! States:
//!   Closed    normal operation; opens after `failure_threshold` CONSECUTIVE transport/5xx
//!             failures on the `/verify` path.
//!   Open      no `/verify` attempts. The owner probes `GET /difficulty` when `probe_due()`:
//!             first probe after 5 s, then x2 + jitter per failed probe, capped at 60 s
//!             (capped at ~5 s while an eligible XUNI exists, so an outage backoff cannot
//!             consume the remainder of a submission window).
//!   HalfOpen  entered on a successful `/difficulty` probe. Admits exactly ONE real queued
//!             submission. Closes ONLY on a verification-path success or a conclusive
//!             duplicate; a transport/5xx failure reopens with an escalated probe interval;
//!             a conclusive non-success (401/4xx) keeps it HalfOpen and releases the
//!             admission slot — the transport is healthy, but only real acceptance proves
//!             the `/verify` path.
//!
//! `/difficulty` health is deliberately separate from `/verify` health: a good `/difficulty`
//! response proves connectivity, not that `/verify` and its database work.

/// Monotonic milliseconds.
pub type Clock = std::sync::Arc<dyn Fn() -> i64 + Send + Sync>;
/// Uniform `[0, 1)`.
pub type Jitter = std::sync::Arc<dyn Fn() -> f64 + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Copy)]
pub struct BreakerConfig {
    /// Consecutive transport/5xx failures needed to open.
    pub failure_threshold: i32,
    /// First probe delay after opening.
    pub probe_base_ms: i64,
    /// Normal probe-interval ceiling.
    pub probe_cap_ms: i64,
    /// Ceiling while an eligible XUNI exists.
    pub probe_cap_xuni_ms: i64,
    /// Adds up to this fraction of the interval.
    pub jitter_fraction: f64,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            probe_base_ms: 5000,
            probe_cap_ms: 60000,
            probe_cap_xuni_ms: 5000,
            jitter_fraction: 0.2,
        }
    }
}

pub struct CircuitBreaker {
    cfg: BreakerConfig,
    clock: Clock,
    jitter: Jitter,
    state: BreakerState,
    consecutive_failures: i32,
    probe_interval_ms: i64,
    next_probe_at_ms: i64,
    /// HalfOpen single-admission latch.
    admission_available: bool,
    xuni_pressure: bool,
}

impl CircuitBreaker {
    /// `jitter` may be `None`: a deterministic zero-jitter source is used (tests) unless a
    /// real one is supplied (production passes a seeded uniform generator).
    pub fn new(cfg: BreakerConfig, clock: Clock, jitter: Option<Jitter>) -> Self {
        Self {
            cfg,
            clock,
            jitter: jitter.unwrap_or_else(|| std::sync::Arc::new(|| 0.0)),
            state: BreakerState::Closed,
            consecutive_failures: 0,
            probe_interval_ms: 0,
            next_probe_at_ms: 0,
            admission_available: false,
            xuni_pressure: false,
        }
    }

    pub fn state(&self) -> BreakerState {
        self.state
    }

    pub fn consecutive_failures(&self) -> i32 {
        self.consecutive_failures
    }

    pub fn next_probe_at_ms(&self) -> i64 {
        self.next_probe_at_ms
    }

    fn active_cap_ms(&self) -> i64 {
        if self.xuni_pressure {
            self.cfg.probe_cap_ms.min(self.cfg.probe_cap_xuni_ms)
        } else {
            self.cfg.probe_cap_ms
        }
    }

    fn schedule_probe(&mut self) {
        let cap = self.active_cap_ms();
        let base = self.probe_interval_ms.min(cap);
        let mut delay = base + ((self.jitter)() * self.cfg.jitter_fraction * base as f64) as i64;
        if self.xuni_pressure {
            delay = delay.min(cap); // hard cap while a XUNI window is at stake
        }
        self.next_probe_at_ms = (self.clock)() + delay;
    }

    fn open(&mut self) {
        self.state = BreakerState::Open;
        self.admission_available = false;
        self.schedule_probe();
    }

    /// True only in `Open`, once the probe time has arrived.
    pub fn probe_due(&self) -> bool {
        self.state == BreakerState::Open && (self.clock)() >= self.next_probe_at_ms
    }

    /// Open -> HalfOpen (one admission granted).
    pub fn on_probe_success(&mut self) {
        if self.state != BreakerState::Open {
            return;
        }
        self.state = BreakerState::HalfOpen;
        self.admission_available = true;
    }

    /// Stay Open; escalate the probe interval.
    pub fn on_probe_failure(&mut self) {
        if self.state != BreakerState::Open {
            return;
        }
        self.probe_interval_ms = (self.probe_interval_ms * 2).min(self.cfg.probe_cap_ms);
        self.schedule_probe();
    }

    /// Closed: always true. HalfOpen: true exactly once until the outcome is reported.
    /// Open: always false.
    pub fn try_admit(&mut self) -> bool {
        match self.state {
            BreakerState::Closed => true,
            BreakerState::HalfOpen => {
                if self.admission_available {
                    self.admission_available = false;
                    true
                } else {
                    false
                }
            }
            BreakerState::Open => false,
        }
    }

    /// Verification success or conclusive duplicate -> Closed.
    pub fn on_verify_success(&mut self) {
        self.state = BreakerState::Closed;
        self.consecutive_failures = 0;
        self.probe_interval_ms = self.cfg.probe_base_ms;
        self.admission_available = false;
    }

    /// Transport/5xx: counts toward opening; HalfOpen -> Open.
    pub fn on_verify_transport_failure(&mut self) {
        match self.state {
            BreakerState::HalfOpen => {
                // The half-open probe submission failed: reopen with an escalated interval.
                self.probe_interval_ms = (self.probe_interval_ms.max(self.cfg.probe_base_ms) * 2)
                    .min(self.cfg.probe_cap_ms);
                self.open();
            }
            BreakerState::Closed => {
                self.consecutive_failures += 1;
                if self.consecutive_failures >= self.cfg.failure_threshold {
                    self.probe_interval_ms = self.cfg.probe_base_ms;
                    self.open();
                }
            }
            // Open: nothing to do — no /verify traffic should be flowing.
            BreakerState::Open => {}
        }
    }

    /// A parsed, conclusive server response (difficulty park, quarantine, ...) proves the
    /// transport and the `/verify` handler are alive — but only real acceptance closes the
    /// breaker from HalfOpen.
    pub fn on_verify_inconclusive(&mut self) {
        self.consecutive_failures = 0;
        if self.state == BreakerState::HalfOpen {
            self.admission_available = true; // keep draining, one admission at a time
        }
    }

    /// While an eligible XUNI exists the probe ceiling drops to `probe_cap_xuni_ms`; an
    /// already scheduled far-future probe is pulled in when the flag turns on.
    pub fn set_xuni_pressure(&mut self, eligible_xuni_exists: bool) {
        let was = self.xuni_pressure;
        self.xuni_pressure = eligible_xuni_exists;
        if !was && self.xuni_pressure && self.state == BreakerState::Open {
            let latest = (self.clock)() + self.active_cap_ms();
            self.next_probe_at_ms = self.next_probe_at_ms.min(latest);
        }
    }
}
