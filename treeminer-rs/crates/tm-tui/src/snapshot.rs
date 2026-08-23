//! What the UI renders from.
//!
//! The live mining state lives in other crates (`tm-gpu`, `tm-journal`, `tm-submit`), so
//! this crate deliberately owns no state at all: the integrator supplies a plain snapshot,
//! captured at whatever instant it likes, and the UI renders that. It is the Rust stand-in
//! for the JSON blob `getMinerDashboardData()` handed to `TerminalUi::render()`, minus the
//! serialisation round trip.

/// Which way network delivery is currently going.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NetworkState {
    #[default]
    Online,
    /// Circuit breaker half-open: one probe in flight.
    Probing,
    Offline,
}

impl NetworkState {
    pub fn as_str(self) -> &'static str {
        match self {
            NetworkState::Online => "online",
            NetworkState::Probing => "probing",
            NetworkState::Offline => "offline",
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Identity {
    pub name: String,
    pub address: String,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct EngineStats {
    /// Hashes per second, GPU and CPU separately; the header shows their sum.
    pub gpu_hashrate: f64,
    pub cpu_hashrate: f64,
    pub gpu_streams: usize,
    pub cpu_workers: usize,
    pub difficulty: u64,
    pub uptime_seconds: i64,
}

impl EngineStats {
    pub fn total_hashrate(&self) -> f64 {
        self.gpu_hashrate + self.cpu_hashrate
    }

    /// Difficulty-weighted throughput, the figure that is comparable across `m` changes.
    pub fn work_rate_m_units(&self) -> f64 {
        self.total_hashrate() * self.difficulty as f64 / 1_000_000.0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FindCounts {
    pub xnm: usize,
    pub xuni: usize,
    pub superblocks: usize,
    pub rejected: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct DeliveryStats {
    pub network: NetworkState,
    /// Pre-formatted, because "never" and a relative age are both valid here.
    pub last_submission: String,
    pub queued_xnm: usize,
    pub queued_xuni: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuStats {
    pub index: u32,
    /// Zero-based; displayed as `S{stream + 1}`.
    pub stream: u32,
    pub name: String,
    pub hashrate: f64,
    pub memory_used_percent: f64,
}

/// One complete frame's worth of miner state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MinerSnapshot {
    pub identity: Identity,
    pub engine: EngineStats,
    pub finds: FindCounts,
    pub delivery: DeliveryStats,
    pub gpus: Vec<GpuStats>,
    /// Dashboard URL shown in the footer; the caller resolves the real NIC address.
    pub console_url: String,
}

/// State for the one-line ticker in `logs` mode. Separate from [`MinerSnapshot`] because
/// the ticker is driven by the mining loop's own counters at a different cadence.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TickerSnapshot {
    pub gpu_hashrate: f64,
    pub cpu_hashrate: f64,
    pub active_gpus: usize,
    /// Shown only when it exceeds `active_gpus`, i.e. multiple streams per device.
    pub stream_count: usize,
    pub total_hashes: u64,
    pub uptime_seconds: u64,
    pub cpu_workers: usize,
    /// CPU workers idle because the difficulty exceeds their ceiling.
    pub cpu_paused_for_difficulty: bool,
    pub superblocks: u64,
    pub normal_blocks: u64,
    pub xuni_blocks: u64,
    pub queued_xnm: usize,
    pub queued_xuni: usize,
    pub accepted_unconfirmed: usize,
    pub confirmed: usize,
    pub breaker_half_open: bool,
    pub pool_down: bool,
    pub outage_ms: u64,
    /// Difficulty headroom added by the submitter; when set the field reads `m N (+K)`.
    pub margin_kib: u32,
    pub difficulty: u64,
    pub console_url: String,
}
