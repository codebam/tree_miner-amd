//! Ordering and pacing policy for draining the journal backlog. Port of
//! `src/submit/DrainScheduler.{h,cpp}`. No I/O, no clock, no threads.
//!
//! Ordering (`select_next`, a pure function of its inputs):
//!   1. XUNI preempts XEN11 when the (server-clock estimated) XUNI window is open and near
//!      its end — a missed window costs the find, XEN11 can always wait.
//!   2. Otherwise oldest eligible XEN11 first; when the difficulty trend is Rising,
//!      ascending-m first (lowest margin drains before the floor rises past it), oldest as
//!      the tie-break.
//!   3. Remaining XUNI (window open, not near the end) after the XEN11 backlog. XUNI with a
//!      closed window are never selected — submitting them only burns a guaranteed 401.
//!
//! Pacing: start at 1/s when the breaker closes after an outage, double per healthy
//! round-trip, halve on 5xx/429; the configured `max_rate_per_s` is the ceiling. Never
//! stampede a recovering server.

use tm_core::{FindKind, FindRecord};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DifficultyTrend {
    Unknown,
    Flat,
    Rising,
    Falling,
}

/// Server-clock-adjusted view of the XUNI :55-:05 window (computed by the caller from the
/// tracked HTTP `Date` offset).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct XuniWindowState {
    pub open: bool,
    /// Meaningful only when `open`.
    pub ms_until_close: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct DrainConfig {
    /// Rate right after the breaker closes.
    pub start_rate_per_s: f64,
    /// The `drain_rate` config: the ceiling.
    pub max_rate_per_s: f64,
    /// Floor under repeated 5xx/429.
    pub min_rate_per_s: f64,
    /// "Near window end" threshold.
    pub xuni_preempt_window_ms: i64,
}

impl Default for DrainConfig {
    fn default() -> Self {
        Self {
            start_rate_per_s: 1.0,
            max_rate_per_s: 4.0,
            min_rate_per_s: 0.25,
            xuni_preempt_window_ms: 120_000,
        }
    }
}

pub struct DrainScheduler {
    cfg: DrainConfig,
    rate_per_s: f64,
}

impl Default for DrainScheduler {
    fn default() -> Self {
        Self::new(DrainConfig::default())
    }
}

impl DrainScheduler {
    pub fn new(cfg: DrainConfig) -> Self {
        let rate = if cfg.start_rate_per_s <= 0.0 {
            1.0
        } else {
            cfg.start_rate_per_s
        };
        Self {
            cfg,
            rate_per_s: rate,
        }
    }

    /// Pick the next record to submit from the journal's oldest-first eligible slice.
    /// `None` when nothing should be submitted now (empty, or only closed-window XUNI).
    pub fn select_next<'a>(
        &self,
        eligible: &'a [FindRecord],
        trend: DifficultyTrend,
        window: XuniWindowState,
    ) -> Option<&'a FindRecord> {
        let mut oldest_xuni: Option<&FindRecord> = None; // journal order is oldest-first
        let mut best_xen11: Option<&FindRecord> = None;

        for r in eligible {
            if r.payload.kind == FindKind::Xuni {
                if window.open && oldest_xuni.is_none() {
                    oldest_xuni = Some(r);
                }
                continue;
            }
            match best_xen11 {
                None => best_xen11 = Some(r),
                Some(best)
                    if trend == DifficultyTrend::Rising
                        && r.payload.memory_cost < best.payload.memory_cost =>
                {
                    // Rising difficulty: lowest-margin finds drain first, before the floor
                    // climbs past their baked-in m. Oldest wins ties because the journal
                    // slice is already oldest-first.
                    best_xen11 = Some(r);
                }
                _ => {}
            }
        }

        // XUNI preemption: the window is open and closing soon — XEN11 can wait, XUNI cannot.
        if let Some(x) = oldest_xuni {
            if window.ms_until_close <= self.cfg.xuni_preempt_window_ms {
                return Some(x);
            }
        }
        best_xen11.or(oldest_xuni)
    }

    /// Reset to `start_rate_per_s`.
    pub fn on_breaker_close(&mut self) {
        self.rate_per_s = self.cfg.start_rate_per_s;
    }

    /// x2, capped at `max_rate_per_s`.
    pub fn on_healthy_round_trip(&mut self) {
        self.rate_per_s = (self.rate_per_s * 2.0).min(self.cfg.max_rate_per_s);
    }

    /// 5xx/429: /2, floored at `min_rate_per_s`.
    pub fn on_throttle(&mut self) {
        self.rate_per_s = (self.rate_per_s / 2.0).max(self.cfg.min_rate_per_s);
    }

    pub fn rate_per_second(&self) -> f64 {
        self.rate_per_s
    }

    pub fn submit_interval_ms(&self) -> i64 {
        (1000.0 / self.rate_per_s) as i64
    }
}
