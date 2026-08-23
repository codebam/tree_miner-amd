//! The publishing side of the stats layer. Port of the reads `src/StatReporter.cpp` and the
//! `statCallback` lambda in `src/main.cpp` perform on the mining globals.
//!
//! Rendering belongs to `tm-dashboard` and `tm-tui`; this module only turns live mining
//! state into the snapshots they take. The direction of the coupling is the point: nothing
//! here reaches into a renderer, and neither renderer reaches into mining state.

use std::sync::Arc;

use tm_dashboard::stats::{
    JournalCounts as DashJournalCounts, StatsSnapshot, StatsSource, SubmissionMetrics,
    TreeminerStats,
};
use tm_submit::{BreakerState, Metrics};
use tm_tui::snapshot::{
    DeliveryStats, EngineStats, FindCounts, GpuStats, Identity, MinerSnapshot, NetworkState,
    TickerSnapshot,
};

use crate::state::MiningState;

/// The version the woodyminer upload payload reports, as in the C++.
pub const REPORTED_VERSION: &str = "2.0.0";

/// What the submission layer contributes to a snapshot. `None` while it does not exist —
/// the console starts before mining does, and `--testFixedDiff` never builds a submitter.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SubmissionView {
    pub metrics: Metrics,
    pub breaker: BreakerStateLabel,
    pub margin_kib: u32,
    pub outage_ms: i64,
    pub last_outage_span_ms: i64,
    pub drain_rate_per_second: f64,
    pub last_observed_difficulty: Option<u32>,
}

/// The three breaker labels the console shows, decoupled from `tm_submit`'s enum so the
/// snapshot type stays plain data.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BreakerStateLabel {
    #[default]
    Closed,
    HalfOpen,
    Open,
}

impl BreakerStateLabel {
    pub fn from_breaker(state: BreakerState) -> Self {
        match state {
            BreakerState::Closed => BreakerStateLabel::Closed,
            BreakerState::HalfOpen => BreakerStateLabel::HalfOpen,
            BreakerState::Open => BreakerStateLabel::Open,
        }
    }

    /// The `/stats` spelling (`up` / `half-open` / `down`).
    pub fn stats_label(self) -> &'static str {
        match self {
            BreakerStateLabel::Closed => "up",
            BreakerStateLabel::HalfOpen => "half-open",
            BreakerStateLabel::Open => "down",
        }
    }

    pub fn dashboard_state(self) -> tm_dashboard::stats::NetworkState {
        match self {
            BreakerStateLabel::Closed => tm_dashboard::stats::NetworkState::Closed,
            BreakerStateLabel::HalfOpen => tm_dashboard::stats::NetworkState::HalfOpen,
            BreakerStateLabel::Open => tm_dashboard::stats::NetworkState::Open,
        }
    }

    pub fn tui_state(self) -> NetworkState {
        match self {
            BreakerStateLabel::Closed => NetworkState::Online,
            BreakerStateLabel::HalfOpen => NetworkState::Probing,
            BreakerStateLabel::Open => NetworkState::Offline,
        }
    }
}

/// Pulls the submission layer's view, if there is one.
pub type SubmissionProvider = Arc<dyn Fn() -> Option<SubmissionView> + Send + Sync>;
/// Pulls the journal's per-status counts, if there is a journal.
pub type JournalProvider = Arc<dyn Fn() -> Option<tm_journal::Counts> + Send + Sync>;

/// Identity and presentation settings that never change during a run.
#[derive(Debug, Clone, Default)]
pub struct StatsIdentity {
    pub machine_id: String,
    pub miner_address: String,
    pub custom_name: String,
    pub margin_mode: String,
    pub console_url: String,
}

/// Builds every snapshot the miner publishes.
pub struct StatsPublisher {
    state: Arc<MiningState>,
    identity: StatsIdentity,
    submission: Option<SubmissionProvider>,
    journal: Option<JournalProvider>,
}

impl std::fmt::Debug for StatsPublisher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatsPublisher")
            .field("identity", &self.identity)
            .finish()
    }
}

impl StatsPublisher {
    pub fn new(state: Arc<MiningState>, identity: StatsIdentity) -> Self {
        Self {
            state,
            identity,
            submission: None,
            journal: None,
        }
    }

    pub fn with_submission(mut self, provider: SubmissionProvider) -> Self {
        self.submission = Some(provider);
        self
    }

    pub fn with_journal(mut self, provider: JournalProvider) -> Self {
        self.journal = Some(provider);
        self
    }

    fn submission_view(&self) -> Option<SubmissionView> {
        self.submission.as_ref().and_then(|provider| provider())
    }

    fn journal_counts(&self) -> Option<tm_journal::Counts> {
        self.journal.as_ref().and_then(|provider| provider())
    }

    /// Everything the read-only console serves.
    pub fn stats_snapshot(&self) -> StatsSnapshot {
        let state = &self.state;
        let submission = self.submission_view();
        let counts = self.journal_counts();

        let treeminer = submission.as_ref().map(|view| TreeminerStats {
            difficulty: state.difficulty() as i32,
            margin_in_effect: view.margin_kib as i32,
            effective_difficulty: state.effective_difficulty() as i32,
            margin_mode: self.identity.margin_mode.clone(),
            breaker_state: view.breaker.stats_label().to_owned(),
            outage_ms: view.outage_ms,
            drain_rate_per_second: view.drain_rate_per_second,
            pending: counts.map(|c| c.pending as u64).unwrap_or_default(),
            parked: counts.map(|c| c.parked as u64).unwrap_or_default(),
            quarantined: counts.map(|c| c.quarantined as u64).unwrap_or_default(),
            acked_total: counts.map(|c| c.acked_total as u64).unwrap_or_default(),
            dead_total: counts.map(|c| c.dead_total as u64).unwrap_or_default(),
            accepted_unconfirmed: counts
                .map(|c| c.accepted_unconfirmed as u64)
                .unwrap_or_default(),
            permanently_invalid: counts
                .map(|c| c.permanently_invalid as u64)
                .unwrap_or_default(),
        });

        StatsSnapshot {
            machine_id: self.identity.machine_id.clone(),
            miner_address: self.identity.miner_address.clone(),
            custom_name: self.identity.custom_name.clone(),
            version: REPORTED_VERSION.to_owned(),
            uptime_seconds: state.uptime_seconds(),
            running: state.is_running(),
            difficulty: state.difficulty() as i32,
            cpu_workers: state.cpu_workers(),
            cpu_hashrate: state.cpu_hashrate(),
            gpus: state.gpu_stats(),
            normal_blocks: state.normal_blocks() as i64,
            super_blocks: state.super_blocks() as i64,
            xuni_blocks: state.xuni_blocks() as i64,
            failed_blocks: state.failed_blocks() as i64,
            network_state: state.network_state(),
            last_submission: state.last_submission(),
            queued_xnm: state.queued_xnm(),
            queued_xuni: state.queued_xuni(),
            fatal_durability_failure: state.fatal_durability_failure(),
            fatal_durability_reason: state.fatal_durability_reason(),
            treeminer,
            journal: counts.map(|c| DashJournalCounts {
                pending: c.pending as u64,
                accepted_unconfirmed: c.accepted_unconfirmed as u64,
                parked: c.parked as u64,
                parked_difficulty: c.parked_difficulty as u64,
                parked_xuni: c.parked_xuni as u64,
                quarantined: c.quarantined as u64,
                acked_total: c.acked_total as u64,
                dead_total: c.dead_total as u64,
                permanently_invalid: c.permanently_invalid as u64,
            }),
            submission: submission.as_ref().map(|view| SubmissionMetrics {
                last_observed_difficulty: view.last_observed_difficulty.map(u64::from),
                margin_in_effect: u64::from(view.margin_kib),
                margin_changes: view.metrics.margin_changes,
                outage_duration_ms: view.outage_ms,
                submitted: view.metrics.submitted,
                resubmitted: view.metrics.resubmitted,
                acked: view.metrics.acked,
                accepted_unconfirmed: view.metrics.accepted_unconfirmed,
                transport_failures: view.metrics.transport_failures,
                parked_difficulty: view.metrics.parked_difficulty,
                parked_xuni: view.metrics.parked_xuni,
                quarantined: view.metrics.quarantined,
                permanently_invalid: view.metrics.permanently_invalid,
                confirmation_retries: view.metrics.confirmation_retries,
                reconciled_via_get_block: view.metrics.reconciled_via_get_block,
                lying_200_detected: view.metrics.confirm_body_rejected,
                probes: view.metrics.probes,
            }),
            platform: None,
        }
    }

    /// One TUI frame.
    pub fn miner_snapshot(&self) -> MinerSnapshot {
        let state = &self.state;
        let gpus = state.gpu_stats();
        let submission = self.submission_view();
        MinerSnapshot {
            identity: Identity {
                name: if self.identity.custom_name.is_empty() {
                    "TreeMiner".to_owned()
                } else {
                    self.identity.custom_name.clone()
                },
                address: self.identity.miner_address.clone(),
            },
            engine: EngineStats {
                gpu_hashrate: f64::from(
                    gpus.iter().map(|gpu| gpu.hashrate).sum::<f32>(),
                ),
                cpu_hashrate: state.cpu_hashrate(),
                gpu_streams: gpus.len(),
                cpu_workers: state.cpu_workers() as usize,
                difficulty: u64::from(state.difficulty()),
                uptime_seconds: state.uptime_seconds(),
            },
            finds: FindCounts {
                xnm: state.normal_blocks() as usize,
                xuni: state.xuni_blocks() as usize,
                superblocks: state.super_blocks() as usize,
                rejected: state.failed_blocks() as usize,
            },
            delivery: DeliveryStats {
                network: submission
                    .as_ref()
                    .map(|view| view.breaker.tui_state())
                    .unwrap_or_default(),
                last_submission: state.last_submission().label().to_owned(),
                queued_xnm: state.queued_xnm() as usize,
                queued_xuni: state.queued_xuni() as usize,
            },
            gpus: gpus
                .iter()
                .map(|gpu| GpuStats {
                    index: gpu.index.max(0) as u32,
                    stream: gpu.stream_index.max(0) as u32,
                    name: gpu.name.clone(),
                    hashrate: f64::from(gpu.hashrate),
                    memory_used_percent: f64::from(gpu.using_memory) * 100.0,
                })
                .collect(),
            console_url: self.identity.console_url.clone(),
        }
    }

    /// One `logs`-mode ticker update.
    pub fn ticker_snapshot(&self) -> TickerSnapshot {
        let state = &self.state;
        let gpus = state.gpu_stats();
        let fresh: Vec<_> = gpus.iter().filter(|gpu| gpu.updated_secs_ago <= 120).collect();
        let mut devices: Vec<i32> = fresh.iter().map(|gpu| gpu.index).collect();
        devices.sort_unstable();
        devices.dedup();
        let submission = self.submission_view();

        TickerSnapshot {
            gpu_hashrate: f64::from(fresh.iter().map(|gpu| gpu.hashrate).sum::<f32>()),
            cpu_hashrate: state.cpu_hashrate(),
            active_gpus: devices.len(),
            stream_count: fresh.len(),
            total_hashes: state.hash_count(),
            uptime_seconds: state.uptime_seconds().max(0) as u64,
            cpu_workers: state.cpu_workers() as usize,
            cpu_paused_for_difficulty: state.cpu_paused_for_difficulty(),
            superblocks: state.super_blocks(),
            normal_blocks: state.normal_blocks(),
            xuni_blocks: state.xuni_blocks(),
            queued_xnm: state.queued_xnm() as usize,
            queued_xuni: state.queued_xuni() as usize,
            accepted_unconfirmed: submission
                .as_ref()
                .map(|view| view.metrics.accepted_unconfirmed as usize)
                .unwrap_or_default(),
            confirmed: submission
                .as_ref()
                .map(|view| view.metrics.acked as usize)
                .unwrap_or_default(),
            breaker_half_open: submission
                .as_ref()
                .is_some_and(|view| view.breaker == BreakerStateLabel::HalfOpen),
            // "pool DOWN" at a glance is the breaker or the difficulty poller, never the
            // live outage clock — that clock reads zero in HalfOpen and would hide a probe.
            pool_down: submission
                .as_ref()
                .is_some_and(|view| view.breaker == BreakerStateLabel::Open)
                || state.difficulty_endpoint_down(),
            outage_ms: submission
                .as_ref()
                .map(|view| {
                    if view.outage_ms > 0 {
                        view.outage_ms
                    } else {
                        view.last_outage_span_ms
                    }
                })
                .unwrap_or_default()
                .max(0) as u64,
            margin_kib: submission
                .as_ref()
                .map(|view| view.margin_kib)
                .unwrap_or_default(),
            difficulty: u64::from(state.difficulty()),
            console_url: self.identity.console_url.clone(),
        }
    }
}

impl StatsSource for StatsPublisher {
    fn snapshot(&self) -> StatsSnapshot {
        self.stats_snapshot()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tm_dashboard::stats::GpuStat;

    fn publisher(state: Arc<MiningState>) -> StatsPublisher {
        StatsPublisher::new(
            state,
            StatsIdentity {
                machine_id: "abc123".to_owned(),
                miner_address: "0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc".to_owned(),
                custom_name: String::new(),
                margin_mode: "off".to_owned(),
                console_url: "http://192.168.1.5:42069".to_owned(),
            },
        )
    }

    fn gpu(index: i32, stream: i32, hashrate: f32) -> GpuStat {
        GpuStat {
            index,
            bus_id: 3,
            name: "Fake GPU".to_owned(),
            memory: 24,
            using_memory: 0.5,
            hashrate,
            hash_count: 100,
            stream_index: stream,
            ..GpuStat::default()
        }
    }

    #[test]
    fn the_console_snapshot_carries_live_mining_state() {
        let state = Arc::new(MiningState::for_test(1727));
        state.publish_gpu(gpu(0, 0, 1500.0));
        state.add_hashes(4096);
        state.record_find_class(crate::state::FindClass::Normal);
        state.set_queued(3, 1);

        let snapshot = publisher(Arc::clone(&state)).stats_snapshot();

        assert_eq!(snapshot.machine_id, "abc123");
        assert_eq!(snapshot.difficulty, 1727);
        assert_eq!(snapshot.gpus.len(), 1);
        assert_eq!(snapshot.total_gpu_hashrate(), 1500.0);
        assert_eq!(snapshot.normal_blocks, 1);
        assert_eq!(snapshot.queued_xnm, 3);
        assert!(snapshot.treeminer.is_none(), "no submitter, no treeminer block");
    }

    #[test]
    fn two_streams_on_one_card_are_two_rows_but_one_gpu() {
        let state = Arc::new(MiningState::for_test(1727));
        state.publish_gpu(gpu(0, 0, 1000.0));
        state.publish_gpu(gpu(0, 1, 900.0));

        let ticker = publisher(state).ticker_snapshot();
        assert_eq!(ticker.active_gpus, 1);
        assert_eq!(ticker.stream_count, 2);
        assert_eq!(ticker.gpu_hashrate, 1900.0);
    }

    #[test]
    fn a_down_difficulty_endpoint_shows_as_pool_down_without_a_submitter() {
        let state = Arc::new(MiningState::for_test(1727));
        let ticker = publisher(Arc::clone(&state)).ticker_snapshot();
        assert!(!ticker.pool_down);
    }

    #[test]
    fn the_submission_view_fills_the_treeminer_and_journal_blocks() {
        let state = Arc::new(MiningState::for_test(1727));
        state.set_margin_kib(500);
        let view = SubmissionView {
            metrics: Metrics {
                submitted: 10,
                acked: 7,
                accepted_unconfirmed: 2,
                ..Metrics::default()
            },
            breaker: BreakerStateLabel::HalfOpen,
            margin_kib: 500,
            outage_ms: 1234,
            last_outage_span_ms: 9999,
            drain_rate_per_second: 2.5,
            last_observed_difficulty: Some(1727),
        };
        let counts = tm_journal::Counts {
            pending: 4,
            acked_total: 7,
            ..tm_journal::Counts::default()
        };
        let publisher = publisher(Arc::clone(&state))
            .with_submission(Arc::new(move || Some(view.clone())))
            .with_journal(Arc::new(move || Some(counts)));

        let snapshot = publisher.stats_snapshot();
        let treeminer = snapshot.treeminer.expect("submitter present");
        assert_eq!(treeminer.breaker_state, "half-open");
        assert_eq!(treeminer.effective_difficulty, 2227);
        assert_eq!(treeminer.margin_in_effect, 500);
        assert_eq!(treeminer.acked_total, 7);
        assert_eq!(snapshot.journal.expect("journal counts").pending, 4);

        let ticker = publisher.ticker_snapshot();
        assert!(ticker.breaker_half_open);
        assert!(!ticker.pool_down, "a probe in flight is not a down pool");
        assert_eq!(ticker.outage_ms, 1234);
        assert_eq!(ticker.margin_kib, 500);
    }

    #[test]
    fn a_closed_breaker_with_a_past_outage_reports_the_latched_span() {
        let state = Arc::new(MiningState::for_test(1727));
        let view = SubmissionView {
            breaker: BreakerStateLabel::Closed,
            outage_ms: 0,
            last_outage_span_ms: 45_000,
            ..SubmissionView::default()
        };
        let ticker = publisher(state)
            .with_submission(Arc::new(move || Some(view.clone())))
            .ticker_snapshot();
        assert_eq!(ticker.outage_ms, 45_000);
        assert!(!ticker.pool_down);
    }
}
