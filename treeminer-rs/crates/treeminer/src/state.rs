//! Live mining state: the counters, gauges and per-device verdicts every other part of the
//! running miner reads. Port of `src/MiningCommon.{h,cpp}`.
//!
//! The C++ kept all of this in process globals guarded by a handful of ad-hoc mutexes. Here
//! it is one shared object passed by `Arc`, which is what makes the mining loop, the CPU
//! sidecar and the stats publisher testable without a process-wide fixture. The field
//! meanings are unchanged so the two implementations stay comparable.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};
use tm_dashboard::stats::{GpuStat, GpuTelemetry, LastSubmissionState, NetworkState};
use tm_tui::Shutdown;

use crate::difficulty::DifficultyShared;

/// Whether a device derives Argon2's first blocks on the GPU when no self-test verdict was
/// recorded for it. The HIP backend is default-off, exactly as `hashapi::kGpuFirstBlocksEnabled`
/// is for ROCm: the device-side Blake2b prehash is the path that once produced invalid
/// digests, so an unprobed device is never trusted with it.
pub const DEFAULT_GPU_FIRST_BLOCKS: bool = false;

/// The whole live state of a running miner.
#[derive(Debug)]
pub struct MiningState {
    difficulty: Arc<DifficultyShared>,
    margin_kib: AtomicU32,

    normal_blocks: AtomicU64,
    super_blocks: AtomicU64,
    xuni_blocks: AtomicU64,
    failed_blocks: AtomicU64,
    hash_count: AtomicU64,

    queued_xnm: AtomicU64,
    queued_xuni: AtomicU64,
    last_submission: Mutex<LastSubmissionState>,
    network_state: Mutex<NetworkState>,

    cpu_workers: AtomicU64,
    /// f64 bits; the stats threads read it far more often than the workers write it.
    cpu_hashrate_bits: AtomicU64,
    cpu_paused_for_difficulty: AtomicBool,

    gpu_first_blocks: RwLock<BTreeMap<i32, bool>>,
    /// Keyed `device * 16 + stream`, as the C++ `globalGpuInfos` map is.
    gpus: Mutex<BTreeMap<i32, (GpuStat, Instant)>>,
    /// Keyed by device index. Written by the one thread that owns the ROCm SMI session,
    /// merged into every published row — the library is not documented as thread safe, so
    /// it is never touched from a mining thread.
    telemetry: Mutex<BTreeMap<i32, GpuTelemetry>>,

    fatal_durability: AtomicBool,
    fatal_reason: Mutex<String>,

    shutdown: Arc<Shutdown>,
    started_at: Instant,
}

impl MiningState {
    pub fn new(difficulty: Arc<DifficultyShared>, shutdown: Arc<Shutdown>) -> Self {
        Self {
            difficulty,
            margin_kib: AtomicU32::new(0),
            normal_blocks: AtomicU64::new(0),
            super_blocks: AtomicU64::new(0),
            xuni_blocks: AtomicU64::new(0),
            failed_blocks: AtomicU64::new(0),
            hash_count: AtomicU64::new(0),
            queued_xnm: AtomicU64::new(0),
            queued_xuni: AtomicU64::new(0),
            last_submission: Mutex::new(LastSubmissionState::None),
            network_state: Mutex::new(NetworkState::Closed),
            cpu_workers: AtomicU64::new(0),
            cpu_hashrate_bits: AtomicU64::new(0),
            cpu_paused_for_difficulty: AtomicBool::new(false),
            gpu_first_blocks: RwLock::new(BTreeMap::new()),
            gpus: Mutex::new(BTreeMap::new()),
            telemetry: Mutex::new(BTreeMap::new()),
            fatal_durability: AtomicBool::new(false),
            fatal_reason: Mutex::new(String::new()),
            shutdown,
            started_at: Instant::now(),
        }
    }

    /// A state with its own difficulty cell — the shape the tests want.
    pub fn for_test(difficulty: u32) -> Self {
        Self::new(
            Arc::new(DifficultyShared::new(difficulty)),
            Arc::new(Shutdown::new()),
        )
    }

    pub fn shutdown(&self) -> &Arc<Shutdown> {
        &self.shutdown
    }

    pub fn is_running(&self) -> bool {
        self.shutdown.is_running()
    }

    pub fn started_at(&self) -> Instant {
        self.started_at
    }

    pub fn uptime_seconds(&self) -> i64 {
        self.started_at.elapsed().as_secs() as i64
    }

    pub fn difficulty_shared(&self) -> &Arc<DifficultyShared> {
        &self.difficulty
    }

    /// Last observed network difficulty, without headroom.
    pub fn difficulty(&self) -> u32 {
        self.difficulty.difficulty()
    }

    pub fn set_difficulty(&self, value: u32) {
        self.difficulty.set_difficulty(value);
    }

    pub fn difficulty_endpoint_down(&self) -> bool {
        self.difficulty.endpoint_down()
    }

    pub fn margin_kib(&self) -> u32 {
        self.margin_kib.load(Ordering::Acquire)
    }

    /// Returns the previous value, so the caller can log only real changes.
    pub fn set_margin_kib(&self, value: u32) -> u32 {
        self.margin_kib.swap(value, Ordering::AcqRel)
    }

    /// The memory cost new batches must actually mine at. Batch sizing, the kernel request
    /// and the `m=` baked into the PHC all read this one value; if they disagreed the miner
    /// would advertise a cost it did not pay. Port of `effectiveMiningDifficulty()`.
    pub fn effective_difficulty(&self) -> u32 {
        effective_difficulty(self.difficulty(), self.margin_kib())
    }

    // --- find counters ---

    pub fn record_find_class(&self, class: FindClass) {
        match class {
            FindClass::Superblock => &self.super_blocks,
            FindClass::Normal => &self.normal_blocks,
            FindClass::Xuni => &self.xuni_blocks,
        }
        .fetch_add(1, Ordering::Relaxed);
    }

    pub fn normal_blocks(&self) -> u64 {
        self.normal_blocks.load(Ordering::Relaxed)
    }

    pub fn super_blocks(&self) -> u64 {
        self.super_blocks.load(Ordering::Relaxed)
    }

    pub fn xuni_blocks(&self) -> u64 {
        self.xuni_blocks.load(Ordering::Relaxed)
    }

    pub fn failed_blocks(&self) -> u64 {
        self.failed_blocks.load(Ordering::Relaxed)
    }

    pub fn add_hashes(&self, count: u64) {
        self.hash_count.fetch_add(count, Ordering::Relaxed);
    }

    pub fn hash_count(&self) -> u64 {
        self.hash_count.load(Ordering::Relaxed)
    }

    // --- delivery gauges ---

    pub fn set_queued(&self, xnm: u64, xuni: u64) {
        self.queued_xnm.store(xnm, Ordering::Release);
        self.queued_xuni.store(xuni, Ordering::Release);
    }

    pub fn queued_xnm(&self) -> u64 {
        self.queued_xnm.load(Ordering::Acquire)
    }

    pub fn queued_xuni(&self) -> u64 {
        self.queued_xuni.load(Ordering::Acquire)
    }

    pub fn set_last_submission(&self, state: LastSubmissionState) {
        *self.last_submission.lock() = state;
    }

    pub fn last_submission(&self) -> LastSubmissionState {
        *self.last_submission.lock()
    }

    pub fn set_network_state(&self, state: NetworkState) {
        *self.network_state.lock() = state;
    }

    pub fn network_state(&self) -> NetworkState {
        *self.network_state.lock()
    }

    // --- CPU sidecar gauges ---

    pub fn set_cpu_stats(&self, workers: usize, hashrate: f64, paused: bool) {
        self.cpu_workers.store(workers as u64, Ordering::Release);
        self.cpu_hashrate_bits
            .store(hashrate.to_bits(), Ordering::Release);
        self.cpu_paused_for_difficulty
            .store(paused, Ordering::Release);
    }

    pub fn cpu_workers(&self) -> u64 {
        self.cpu_workers.load(Ordering::Acquire)
    }

    pub fn cpu_hashrate(&self) -> f64 {
        f64::from_bits(self.cpu_hashrate_bits.load(Ordering::Acquire))
    }

    pub fn cpu_paused_for_difficulty(&self) -> bool {
        self.cpu_paused_for_difficulty.load(Ordering::Acquire)
    }

    // --- per-device GPU first-blocks verdicts ---

    /// Record the startup self-test's verdict for one device.
    pub fn set_gpu_first_blocks_verified(&self, device_index: i32, verified: bool) {
        self.gpu_first_blocks.write().insert(device_index, verified);
    }

    /// The verdict, or [`DEFAULT_GPU_FIRST_BLOCKS`] for a device that was never probed.
    pub fn gpu_first_blocks_verified(&self, device_index: i32) -> bool {
        self.gpu_first_blocks
            .read()
            .get(&device_index)
            .copied()
            .unwrap_or(DEFAULT_GPU_FIRST_BLOCKS)
    }

    // --- per-device stats ---

    /// Publish one stream's line. The key matches the C++ `index * 16 + streamIndex`, so a
    /// device with two streams keeps two rows rather than overwriting itself.
    pub fn publish_gpu(&self, stat: GpuStat) {
        let key = stat.index * 16 + stat.stream_index;
        self.gpus.lock().insert(key, (stat, Instant::now()));
    }

    /// Publish one device's power/utilisation reading.
    pub fn publish_telemetry(&self, device_index: i32, telemetry: GpuTelemetry) {
        self.telemetry.lock().insert(device_index, telemetry);
    }

    /// Every published line with its age and the latest telemetry for its device, oldest
    /// key first.
    pub fn gpu_stats(&self) -> Vec<GpuStat> {
        let now = Instant::now();
        let telemetry = self.telemetry.lock().clone();
        self.gpus
            .lock()
            .values()
            .map(|(stat, at)| {
                let mut stat = stat.clone();
                stat.updated_secs_ago = now.saturating_duration_since(*at).as_secs();
                stat.telemetry = telemetry.get(&stat.index).copied();
                stat
            })
            .collect()
    }

    // --- fatal durability ---

    /// A find could be persisted by neither the journal nor the fallback sink. From that
    /// moment every future find would be destroyed on arrival, so mining stops and the
    /// process exits nonzero for a supervisor to restart against a recovered disk.
    ///
    /// First declaration wins: a later double-failure on the same broken disk is an echo of
    /// the first, and would overwrite the diagnosis that matters.
    pub fn declare_fatal_durability_failure(&self, reason: &str) {
        {
            let mut held = self.fatal_reason.lock();
            if held.is_empty() {
                *held = if reason.is_empty() {
                    "unspecified durability failure".to_owned()
                } else {
                    reason.to_owned()
                };
            }
        }
        // The reason is published before the flag, so a reader that sees the flag is
        // guaranteed a complete reason behind it.
        self.fatal_durability.store(true, Ordering::Release);
        self.shutdown.request_stop();
    }

    pub fn fatal_durability_failure(&self) -> bool {
        self.fatal_durability.load(Ordering::Acquire)
    }

    pub fn fatal_durability_reason(&self) -> String {
        self.fatal_reason.lock().clone()
    }
}

/// Difficulty plus headroom, with the C++ overflow guard: a sum that cannot be represented
/// would mean mining at a memory cost the server rejects outright, which is worse than
/// ignoring the margin.
pub fn effective_difficulty(difficulty: u32, margin_kib: u32) -> u32 {
    if margin_kib == 0 {
        return difficulty;
    }
    match difficulty.checked_add(margin_kib) {
        Some(sum) if sum <= i32::MAX as u32 => sum,
        _ => difficulty,
    }
}

/// How a find is counted on the status line. `superblock` is the C++ rule: 50 or more
/// capitals in the digest.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FindClass {
    Normal,
    Superblock,
    Xuni,
}

impl FindClass {
    pub fn as_str(self) -> &'static str {
        match self {
            FindClass::Normal => "normal",
            FindClass::Superblock => "superblock",
            FindClass::Xuni => "xuni",
        }
    }
}

/// Classify a find from its bare base64 digest.
pub fn classify_find(digest: &str) -> FindClass {
    if digest.contains("XEN11") {
        if tm_core::is_superblock_hash(digest) {
            FindClass::Superblock
        } else {
            FindClass::Normal
        }
    } else {
        FindClass::Xuni
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margin_is_added_to_the_mined_memory_cost() {
        assert_eq!(effective_difficulty(1000, 0), 1000);
        assert_eq!(effective_difficulty(1000, 250), 1250);
    }

    #[test]
    fn an_overflowing_margin_is_ignored_rather_than_wrapped() {
        assert_eq!(effective_difficulty(u32::MAX - 1, 5), u32::MAX - 1);
        assert_eq!(effective_difficulty(i32::MAX as u32, 1), i32::MAX as u32);
    }

    #[test]
    fn unprobed_devices_keep_first_blocks_on_the_cpu() {
        let state = MiningState::for_test(1000);
        assert_eq!(state.gpu_first_blocks_verified(0), DEFAULT_GPU_FIRST_BLOCKS);
        state.set_gpu_first_blocks_verified(0, true);
        assert!(state.gpu_first_blocks_verified(0));
        assert_eq!(state.gpu_first_blocks_verified(1), DEFAULT_GPU_FIRST_BLOCKS);
    }

    #[test]
    fn the_first_durability_failure_is_the_one_reported() {
        let state = MiningState::for_test(1000);
        state.declare_fatal_durability_failure("disk full");
        state.declare_fatal_durability_failure("later echo");
        assert!(state.fatal_durability_failure());
        assert_eq!(state.fatal_durability_reason(), "disk full");
        assert!(!state.is_running());
    }
}
