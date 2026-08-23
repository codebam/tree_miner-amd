//! Time, injected.
//!
//! The C++ reads `system_clock` for envelope/timestamp work and `steady_clock` for lease
//! timing. Both are behind this trait so that envelope expiry, lease expiry and reconnect
//! backoff are unit-testable without sleeping — the property the C++ envelope tests bought
//! by taking `now_epoch_s` as a parameter, extended to the whole crate.

use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub trait Clock: Send + Sync {
    /// Wall-clock seconds since the Unix epoch. Used for message timestamps and for the
    /// envelope's issued-at / expires-at window, which are absolute times chosen by a
    /// remote signer, so they can only be compared against a wall clock.
    fn now_epoch_s(&self) -> i64;

    /// A monotonic instant. Lease expiry uses this so that an operator correcting the
    /// system clock cannot end (or extend) a paid lease.
    fn monotonic(&self) -> Duration;
}

/// The real clock.
#[derive(Debug, Default)]
pub struct SystemClock {
    start: Option<std::time::Instant>,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            start: Some(std::time::Instant::now()),
        }
    }
}

impl Clock for SystemClock {
    fn now_epoch_s(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            // A clock before 1970 is nonsense but must not panic in a network path.
            .unwrap_or(0)
    }

    fn monotonic(&self) -> Duration {
        match self.start {
            Some(start) => start.elapsed(),
            None => Duration::ZERO,
        }
    }
}

/// A clock the test drives by hand. Lives outside `#[cfg(test)]` so integration tests in
/// `tests/` can use it too.
#[derive(Debug, Default)]
pub struct TestClock {
    epoch_s: AtomicI64,
    monotonic_ms: AtomicI64,
}

impl TestClock {
    pub fn new(epoch_s: i64) -> Self {
        Self {
            epoch_s: AtomicI64::new(epoch_s),
            monotonic_ms: AtomicI64::new(0),
        }
    }

    /// Move both clocks forward together, the way real time moves.
    pub fn advance(&self, by: Duration) {
        self.epoch_s.fetch_add(by.as_secs() as i64, Ordering::SeqCst);
        self.monotonic_ms
            .fetch_add(by.as_millis() as i64, Ordering::SeqCst);
    }

    /// Move only the wall clock, leaving the monotonic clock alone — an operator running
    /// `date -s`, or NTP stepping the clock.
    pub fn set_epoch(&self, epoch_s: i64) {
        self.epoch_s.store(epoch_s, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn now_epoch_s(&self) -> i64 {
        self.epoch_s.load(Ordering::SeqCst)
    }

    fn monotonic(&self) -> Duration {
        Duration::from_millis(self.monotonic_ms.load(Ordering::SeqCst).max(0) as u64)
    }
}
