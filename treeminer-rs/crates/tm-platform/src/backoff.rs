//! Reconnect backoff. Port of `MqttClient`'s `INITIAL_RECONNECT_DELAY_MS` /
//! `MAX_RECONNECT_DELAY_MS`, which Paho applied internally and rumqttc leaves to us.
//!
//! Exponential with a ceiling, reset on a successful connection. WHY a ceiling and not
//! unbounded growth: a rig that loses the broker for an hour must still come back within
//! `max` of the broker returning, or the fleet stays dark long after the outage ends.

use std::time::Duration;

pub const INITIAL_RECONNECT_DELAY_MS: u64 = 1_000;
pub const MAX_RECONNECT_DELAY_MS: u64 = 30_000;

#[derive(Debug, Clone)]
pub struct ReconnectBackoff {
    initial: Duration,
    max: Duration,
    current: Duration,
}

impl Default for ReconnectBackoff {
    fn default() -> Self {
        Self::new(
            Duration::from_millis(INITIAL_RECONNECT_DELAY_MS),
            Duration::from_millis(MAX_RECONNECT_DELAY_MS),
        )
    }
}

impl ReconnectBackoff {
    pub fn new(initial: Duration, max: Duration) -> Self {
        let initial = initial.max(Duration::from_millis(1));
        Self {
            initial,
            max: max.max(initial),
            current: initial,
        }
    }

    /// The delay to wait before the next attempt, doubling each call up to the ceiling.
    pub fn next_delay(&mut self) -> Duration {
        let delay = self.current;
        self.current = (self.current * 2).min(self.max);
        delay
    }

    /// Called once a connection succeeds, so the next outage starts fast again.
    pub fn reset(&mut self) {
        self.current = self.initial;
    }

    pub fn peek(&self) -> Duration {
        self.current
    }
}
