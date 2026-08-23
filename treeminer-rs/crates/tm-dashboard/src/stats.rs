//! The data the console renders, and the contract the mining side implements.
//!
//! Port of the stats-carrying parts of `src/MiningCommon.h` (`gpuInfo`, `TreeminerStats`,
//! the queue/network globals) plus the reads `StatReporter.cpp` performs on them.
//!
//! The C++ reporter reaches straight into process globals under `globalGpuInfosMutex`.
//! The mining loop is a different crate here, so the coupling is inverted: the owner of
//! those numbers builds a [`StatsSnapshot`] and hands it over through [`StatsSource`].
//! That keeps the dashboard free of mining state, and gives tests a fake source.

use std::sync::Arc;

/// Per-GPU line, mirroring the C++ `gpuInfo` struct field for field.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuStat {
    pub index: i32,
    pub bus_id: i32,
    pub name: String,
    /// Total device memory, GiB (C++ `memory`).
    pub memory: i32,
    /// Fraction in `0.0..=1.0`; the JSON reports it as a percentage, as the C++ does.
    pub using_memory: f32,
    pub temperature: i32,
    pub hashrate: f32,
    /// Free-form power string carried by the mining loop (C++ `gpuInfo::power`).
    pub power: String,
    pub hash_count: u64,
    pub stream_index: i32,
    /// Vendor management-library reading (NVML / ROCm SMI). `None` when no telemetry
    /// library is present, which is the case the sentinel below exists for.
    pub telemetry: Option<GpuTelemetry>,
    /// Age of this entry. The C++ stores a `steady_clock` stamp beside every `gpuInfo`
    /// and drops entries older than two minutes from `/api/rig`; this carries the same
    /// information without exposing a clock to the dashboard.
    pub updated_secs_ago: u64,
}

/// Power/utilisation from the vendor management library. Both halves are independently
/// optional, exactly as `gputelemetry::DeviceTelemetry` models them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuTelemetry {
    pub power_milliwatts: Option<u32>,
    pub utilization_percent: Option<u32>,
}

/// Value reported for power when no telemetry library answered. The C++ writes
/// `static_cast<unsigned int>(-1)`; the page must keep seeing that exact number.
pub const POWER_UNAVAILABLE: u32 = u32::MAX;

impl GpuStat {
    /// Milliwatts, or [`POWER_UNAVAILABLE`] — the C++ sentinel.
    pub fn power_milliwatts_or_sentinel(&self) -> u32 {
        self.telemetry
            .and_then(|t| t.power_milliwatts)
            .unwrap_or(POWER_UNAVAILABLE)
    }

    /// Utilisation percent. The C++ reads `utilizationPercent` without checking
    /// `hasUtilization`, so an unavailable reading surfaces as 0, not a sentinel.
    pub fn utilization_percent(&self) -> u32 {
        self.telemetry
            .and_then(|t| t.utilization_percent)
            .unwrap_or(0)
    }

    fn power_contribution(&self) -> u32 {
        self.telemetry
            .and_then(|t| t.power_milliwatts)
            .unwrap_or(0)
    }
}

/// C++ `treeminer::CircuitBreaker::State`, only the labels the console shows.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum NetworkState {
    #[default]
    Closed,
    HalfOpen,
    Open,
}

impl NetworkState {
    /// `networkStateLabel` in `MiningCommon.cpp`.
    pub fn label(self) -> &'static str {
        match self {
            NetworkState::Open => "offline",
            NetworkState::HalfOpen => "probing",
            NetworkState::Closed => "online",
        }
    }
}

/// C++ `LastSubmissionState`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum LastSubmissionState {
    #[default]
    None,
    Accepted,
    Unconfirmed,
    Retry,
    Parked,
    Failed,
}

impl LastSubmissionState {
    /// `submissionStateLabel` in `MiningCommon.cpp`.
    pub fn label(self) -> &'static str {
        match self {
            LastSubmissionState::Accepted => "accepted",
            LastSubmissionState::Unconfirmed => "confirming",
            LastSubmissionState::Retry => "retrying",
            LastSubmissionState::Parked => "held",
            LastSubmissionState::Failed => "rejected",
            LastSubmissionState::None => "none",
        }
    }
}

/// Port of C++ `TreeminerStats`: the submission layer's view, absent until the journal
/// and submitter exist (the console starts before mining does).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TreeminerStats {
    pub difficulty: i32,
    pub margin_in_effect: i32,
    pub effective_difficulty: i32,
    pub margin_mode: String,
    pub breaker_state: String,
    pub outage_ms: i64,
    pub drain_rate_per_second: f64,
    pub pending: u64,
    pub parked: u64,
    pub quarantined: u64,
    pub acked_total: u64,
    pub dead_total: u64,
    pub accepted_unconfirmed: u64,
    pub permanently_invalid: u64,
}

/// Port of `IFindJournal::counts()`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JournalCounts {
    pub pending: u64,
    pub accepted_unconfirmed: u64,
    pub parked: u64,
    pub parked_difficulty: u64,
    pub parked_xuni: u64,
    pub quarantined: u64,
    pub acked_total: u64,
    pub dead_total: u64,
    pub permanently_invalid: u64,
}

/// Port of `SubmissionManager::metrics()` plus the two scalars `/stats` reads beside it
/// (`lastObservedDifficulty`, `marginInEffect`, `outageDurationMs`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SubmissionMetrics {
    pub last_observed_difficulty: Option<u64>,
    pub margin_in_effect: u64,
    pub margin_changes: u64,
    pub outage_duration_ms: i64,
    pub submitted: u64,
    pub resubmitted: u64,
    pub acked: u64,
    pub accepted_unconfirmed: u64,
    pub transport_failures: u64,
    pub parked_difficulty: u64,
    pub parked_xuni: u64,
    pub quarantined: u64,
    pub permanently_invalid: u64,
    pub confirmation_retries: u64,
    pub reconciled_via_get_block: u64,
    pub lying_200_detected: u64,
    pub probes: u64,
}

impl SubmissionMetrics {
    /// The C++ `failed` sum on `/stats`.
    pub fn failed_attempts(&self) -> u64 {
        self.transport_failures
            + self.parked_difficulty
            + self.parked_xuni
            + self.quarantined
            + self.permanently_invalid
    }

    pub fn failure_rate_pct(&self) -> f64 {
        if self.submitted == 0 {
            0.0
        } else {
            100.0 * self.failed_attempts() as f64 / self.submitted as f64
        }
    }
}

/// Hashpower-marketplace lease, as `/platform/status` reports it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlatformLease {
    pub lease_id: String,
    pub consumer_id: String,
    pub consumer_address: String,
    pub blocks_found: i64,
    pub remaining_sec: i64,
}

/// Port of the `/platform/status` payload. `None` on the snapshot means the platform
/// manager does not exist, which the route renders as the C++ "disabled" shape.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlatformStatus {
    pub platform_mode: bool,
    /// "platform" or "self".
    pub mining_mode: String,
    pub platform_state: String,
    pub running: bool,
    pub lease: Option<PlatformLease>,
}

/// Everything every route serves, captured at one instant.
///
/// One snapshot per response is deliberate: the C++ takes a single identity snapshot and
/// one locked copy of the GPU map per response so a concurrent identity update cannot
/// tear the address mid-serialisation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StatsSnapshot {
    pub machine_id: String,
    /// The public reward address. The only identity field any route may expose.
    pub miner_address: String,
    /// Empty means the console shows "TreeMiner".
    pub custom_name: String,
    /// Reported by the woodyminer upload payload; "2.0.0" in the C++.
    pub version: String,
    pub uptime_seconds: i64,
    pub running: bool,
    pub difficulty: i32,
    pub cpu_workers: u64,
    pub cpu_hashrate: f64,
    pub gpus: Vec<GpuStat>,
    pub normal_blocks: i64,
    pub super_blocks: i64,
    pub xuni_blocks: i64,
    pub failed_blocks: i64,
    pub network_state: NetworkState,
    pub last_submission: LastSubmissionState,
    pub queued_xnm: u64,
    pub queued_xuni: u64,
    pub fatal_durability_failure: bool,
    /// Only serialised when the flag above is set, as in the C++.
    pub fatal_durability_reason: String,
    pub treeminer: Option<TreeminerStats>,
    pub journal: Option<JournalCounts>,
    pub submission: Option<SubmissionMetrics>,
    pub platform: Option<PlatformStatus>,
}

impl StatsSnapshot {
    pub fn total_gpu_hashrate(&self) -> f32 {
        self.gpus.iter().map(|g| g.hashrate).sum()
    }

    pub fn total_hash_count(&self) -> u64 {
        self.gpus.iter().map(|g| g.hash_count).sum()
    }

    /// Sum of the GPUs that actually reported power; unavailable readings contribute 0.
    pub fn total_power_milliwatts(&self) -> u32 {
        self.gpus
            .iter()
            .fold(0u32, |acc, g| acc.saturating_add(g.power_contribution()))
    }

    pub fn accepted_blocks(&self) -> i64 {
        self.normal_blocks + self.super_blocks
    }

    /// `/api/rig` drops GPU entries that have not refreshed in two minutes.
    pub fn fresh_gpus(&self) -> impl Iterator<Item = &GpuStat> {
        self.gpus.iter().filter(|g| g.updated_secs_ago <= 120)
    }
}

/// What the mining side implements so the console can read it.
///
/// Implementations must be cheap and must never block the mining hot path: the C++
/// copies the GPU map under its mutex and releases it before formatting, and the same
/// rule applies here.
pub trait StatsSource: Send + Sync + 'static {
    fn snapshot(&self) -> StatsSnapshot;
}

impl<F> StatsSource for F
where
    F: Fn() -> StatsSnapshot + Send + Sync + 'static,
{
    fn snapshot(&self) -> StatsSnapshot {
        self()
    }
}

impl<T: StatsSource + ?Sized> StatsSource for Arc<T> {
    fn snapshot(&self) -> StatsSnapshot {
        (**self).snapshot()
    }
}

/// A snapshot cell the mining side can publish into and the console reads from — the
/// direct replacement for the C++ `globalGpuInfos` + mutex pairing.
#[derive(Debug, Default)]
pub struct SharedStats {
    inner: parking_lot::RwLock<StatsSnapshot>,
}

impl SharedStats {
    pub fn new(initial: StatsSnapshot) -> Self {
        Self {
            inner: parking_lot::RwLock::new(initial),
        }
    }

    pub fn publish(&self, snapshot: StatsSnapshot) {
        *self.inner.write() = snapshot;
    }

    /// Mutate in place; the write lock is held only for the closure.
    pub fn update(&self, f: impl FnOnce(&mut StatsSnapshot)) {
        f(&mut self.inner.write());
    }
}

impl StatsSource for SharedStats {
    fn snapshot(&self) -> StatsSnapshot {
        self.inner.read().clone()
    }
}
