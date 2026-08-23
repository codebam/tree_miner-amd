//! CPU sidecar mining. Port of `src/CpuMiningWorker.{h,cpp}`.
//!
//! CPU hashing only pays near the difficulty floor: `m` IS the Argon2 memory cost, so above
//! the ceiling a CPU burns power for a negligible share of the network. Workers idle on a
//! short poll above `--cpuMaxDifficulty` and resume by themselves when it falls, rather than
//! being torn down and rebuilt.
//!
//! Every find goes through the same [`FindSink`](crate::find::FindSink) the GPU loop uses,
//! so the journal-first guarantee is identical on both paths.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tm_argon2::{HashRequest, MIN_ARGON2_CPU_DIFFICULTY};

use crate::find::{Find, FindSink};
use crate::mineunit::{select_work, IdentitySource, MiningIdentity};
use crate::state::MiningState;

/// The C++ `kCpuMiningBatchSize`.
pub const DEFAULT_CPU_BATCH_SIZE: usize = 64;

/// How long an idle (difficulty-gated) worker sleeps between checks. Short enough that a
/// stop request or a difficulty drop is noticed promptly, long enough to cost nothing.
const IDLE_POLL: Duration = Duration::from_millis(250);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuWorkerConfig {
    pub worker_count: usize,
    pub batch_size: usize,
    /// `--cpuMaxDifficulty`; 0 disables the gate.
    pub max_difficulty: u32,
}

impl Default for CpuWorkerConfig {
    fn default() -> Self {
        Self {
            worker_count: 0,
            batch_size: DEFAULT_CPU_BATCH_SIZE,
            max_difficulty: 0,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct CpuStats {
    pub attempts: u64,
    pub matches: u64,
    pub hashrate: f64,
    pub active_workers: usize,
    pub difficulty: u32,
    pub paused_for_difficulty: bool,
    pub max_difficulty: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CpuWorkerError {
    #[error("CPU worker count must be greater than zero")]
    NoWorkers,
    #[error("CPU batch size must be between 1 and {}", tm_argon2::MAX_CPU_BATCH_SIZE)]
    BatchSize,
}

struct Shared {
    config: CpuWorkerConfig,
    state: Arc<MiningState>,
    sink: Arc<FindSink>,
    identity: MiningIdentity,
    /// Live override of `identity`, as on the GPU path: a platform lease must redirect the
    /// CPU sidecar too, or the rig would keep paying part of its output to its owner while
    /// a consumer is being billed for the whole machine.
    identity_source: Option<IdentitySource>,
    /// The devfee rotation slot, shared by every worker so the fee share is per-process.
    work_sequence: AtomicU64,
    /// Injected so the window boundary is testable.
    allow_xuni: Arc<dyn Fn() -> bool + Send + Sync>,
    stop: AtomicBool,
    attempts: AtomicU64,
    matches: AtomicU64,
    active: AtomicUsize,
    difficulty: AtomicU64,
    paused: AtomicBool,
    hashrates: Mutex<Vec<f64>>,
    last_error: Mutex<Option<String>>,
}

/// A group of independent CPU hashing threads.
pub struct CpuMiningWorker {
    shared: Arc<Shared>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

impl std::fmt::Debug for CpuMiningWorker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuMiningWorker")
            .field("config", &self.shared.config)
            .finish()
    }
}

impl CpuMiningWorker {
    pub fn new(
        config: CpuWorkerConfig,
        state: Arc<MiningState>,
        sink: Arc<FindSink>,
        identity: MiningIdentity,
    ) -> Result<Self, CpuWorkerError> {
        if config.worker_count == 0 {
            return Err(CpuWorkerError::NoWorkers);
        }
        if config.batch_size == 0 || config.batch_size > tm_argon2::MAX_CPU_BATCH_SIZE {
            return Err(CpuWorkerError::BatchSize);
        }
        Ok(Self {
            shared: Arc::new(Shared {
                config,
                state,
                sink,
                identity,
                identity_source: None,
                work_sequence: AtomicU64::new(0),
                allow_xuni: Arc::new(crate::mineunit::xuni_window_open_now),
                stop: AtomicBool::new(false),
                attempts: AtomicU64::new(0),
                matches: AtomicU64::new(0),
                active: AtomicUsize::new(0),
                difficulty: AtomicU64::new(0),
                paused: AtomicBool::new(false),
                hashrates: Mutex::new(vec![0.0; config.worker_count]),
                last_error: Mutex::new(None),
            }),
            threads: Mutex::new(Vec::new()),
        })
    }

    /// Install a live identity source. Platform mode calls this before `start`.
    pub fn set_identity_source(&mut self, source: IdentitySource) {
        if let Some(shared) = Arc::get_mut(&mut self.shared) {
            shared.identity_source = Some(source);
        }
    }

    /// Replace the XUNI-window predicate. Tests use it; production keeps the default.
    pub fn set_xuni_window(&mut self, predicate: Arc<dyn Fn() -> bool + Send + Sync>) {
        if let Some(shared) = Arc::get_mut(&mut self.shared) {
            shared.allow_xuni = predicate;
        }
    }

    pub fn start(&self) {
        let mut threads = self.threads.lock();
        if !threads.is_empty() {
            return;
        }
        for index in 0..self.shared.config.worker_count {
            let shared = Arc::clone(&self.shared);
            if let Ok(handle) = std::thread::Builder::new()
                .name(format!("treeminer-cpu-{index}"))
                .spawn(move || run_worker(&shared, index))
            {
                threads.push(handle);
            }
        }
    }

    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
    }

    pub fn join(&self) {
        let handles: Vec<_> = self.threads.lock().drain(..).collect();
        for handle in handles {
            let _ = handle.join();
        }
    }

    pub fn stats(&self) -> CpuStats {
        let shared = &self.shared;
        CpuStats {
            attempts: shared.attempts.load(Ordering::Relaxed),
            matches: shared.matches.load(Ordering::Relaxed),
            hashrate: shared.hashrates.lock().iter().sum(),
            active_workers: shared.active.load(Ordering::Acquire),
            difficulty: shared.difficulty.load(Ordering::Relaxed) as u32,
            paused_for_difficulty: shared.paused.load(Ordering::Acquire),
            max_difficulty: shared.config.max_difficulty,
            last_error: shared.last_error.lock().clone(),
        }
    }

    /// Publish the current stats where the console and dashboard read them.
    pub fn publish(&self) {
        let stats = self.stats();
        self.shared.state.set_cpu_stats(
            stats.active_workers,
            stats.hashrate,
            stats.paused_for_difficulty,
        );
    }
}

impl Drop for CpuMiningWorker {
    fn drop(&mut self) {
        self.stop();
        self.join();
    }
}

fn run_worker(shared: &Arc<Shared>, index: usize) {
    shared.active.fetch_add(1, Ordering::AcqRel);
    let mut attempts_since_match: u64 = 0;

    while !shared.stop.load(Ordering::Acquire) && shared.state.is_running() {
        // The C++ CPU path mines at the bare network difficulty, not difficulty+margin:
        // headroom is a GPU-scale trade and would price the CPU out of the floor entirely.
        let difficulty = shared.state.difficulty();
        shared
            .difficulty
            .store(u64::from(difficulty), Ordering::Relaxed);

        if difficulty < MIN_ARGON2_CPU_DIFFICULTY {
            record_error(shared, "CPU mining difficulty must be at least 8");
            break;
        }

        if shared.config.max_difficulty > 0 && difficulty > shared.config.max_difficulty {
            shared.paused.store(true, Ordering::Release);
            // Report zero rather than a stale figure: the status line must read truthfully.
            set_hashrate(shared, index, 0.0);
            std::thread::sleep(IDLE_POLL);
            continue;
        }
        shared.paused.store(false, Ordering::Release);

        // A platform `pause` idles the sidecar the same way the difficulty gate does: the
        // thread stays alive so `resume` costs nothing.
        if shared.state.is_mining_paused() {
            set_hashrate(shared, index, 0.0);
            std::thread::sleep(IDLE_POLL);
            continue;
        }

        let identity = match &shared.identity_source {
            Some(source) => source(),
            None => shared.identity.clone(),
        };
        let slot = (shared.work_sequence.fetch_add(1, Ordering::Relaxed) % 1000) as i32;
        let work = select_work(&identity, slot);
        let allow_xuni = (shared.allow_xuni)();

        let request = HashRequest {
            backend: "cpu".to_owned(),
            salt_hex: work.salt_hex.clone(),
            key_prefix: work.key_prefix.clone(),
            target_pattern: identity.block_pattern().to_owned(),
            difficulty,
            batch_size: shared.config.batch_size,
            device_id: index as i32,
            allow_xuni,
            ..HashRequest::default()
        };

        let result = tm_argon2::run_batch(&request);
        if !result.ok {
            record_error(shared, &format!("CPU hash batch failed: {}", result.error));
            break;
        }

        shared
            .attempts
            .fetch_add(result.attempts as u64, Ordering::Relaxed);
        set_hashrate(shared, index, result.hashrate);

        let mut next_attempt_index = 0usize;
        for hit in &result.matches {
            if hit.attempt_index >= next_attempt_index {
                attempts_since_match += (hit.attempt_index - next_attempt_index + 1) as u64;
                next_attempt_index = hit.attempt_index + 1;
            }
            // The CPU backend reports PHC strings; the pattern must be re-checked against
            // the digest alone, or a `m=` or salt that happens to contain the pattern would
            // be submitted as a find.
            let Some(digest) = tm_core::phc_digest(&hit.hash) else {
                record_error(shared, "CPU backend returned an invalid PHC string");
                continue;
            };
            if !digest_matches(&request.target_pattern, allow_xuni, &hit.matched_pattern, digest) {
                continue;
            }
            let hashrate = shared.hashrates.lock().iter().sum();
            shared.sink.record(&Find {
                hexsalt: work.salt_hex.clone(),
                key: hit.key.clone(),
                digest: digest.to_owned(),
                memory_cost: difficulty,
                attempts: attempts_since_match,
                hashes_per_second: hashrate,
                source: "CPU".to_owned(),
            });
            shared.matches.fetch_add(1, Ordering::Relaxed);
            attempts_since_match = 0;
        }
        if result.attempts >= next_attempt_index {
            attempts_since_match += (result.attempts - next_attempt_index) as u64;
        }
    }

    set_hashrate(shared, index, 0.0);
    shared.active.fetch_sub(1, Ordering::AcqRel);
}

/// Port of `digestMatches`: a reported match is only real if the DIGEST carries the pattern.
pub fn digest_matches(
    target_pattern: &str,
    allow_xuni: bool,
    matched_pattern: &str,
    digest: &str,
) -> bool {
    if matched_pattern == target_pattern {
        return digest.contains(target_pattern);
    }
    if matched_pattern == "XUNI" {
        return allow_xuni && tm_core::has_xuni_match(digest);
    }
    false
}

fn set_hashrate(shared: &Arc<Shared>, index: usize, value: f64) {
    let mut rates = shared.hashrates.lock();
    if let Some(slot) = rates.get_mut(index) {
        *slot = value;
    }
}

/// The first error is the diagnosis; later ones are echoes of it. Recording one also stops
/// the worker group, as the C++ `recordError` does.
fn record_error(shared: &Arc<Shared>, message: &str) {
    let mut held = shared.last_error.lock();
    if held.is_none() {
        *held = Some(message.to_owned());
    }
    drop(held);
    shared.stop.store(true, Ordering::Release);
}

/// Wall time the worker group has been up; used only for the average-rate figure.
pub fn elapsed_seconds(since: Instant) -> f64 {
    since.elapsed().as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::RecordingJournal;

    fn sink(state: &Arc<MiningState>, dir: &std::path::Path) -> Arc<FindSink> {
        Arc::new(
            FindSink::new(
                Arc::new(RecordingJournal::default()),
                tm_journal::FallbackSink::new(dir.join("fallback.jsonl")),
                Arc::clone(state),
                "worker-1",
            )
            .with_clock(|| "2026-01-01T00:00:00Z".to_owned()),
        )
    }

    #[test]
    fn a_pattern_in_the_phc_envelope_is_not_a_find() {
        // The PHC prefix and the salt are not part of the digest; only the digest counts.
        assert!(!digest_matches("XEN11", true, "XEN11", "no-pattern-here"));
        assert!(digest_matches("XEN11", true, "XEN11", "aaaXEN11bbb"));
    }

    #[test]
    fn a_xuni_match_is_rejected_when_the_window_is_closed() {
        assert!(digest_matches("XEN11", true, "XUNI", "aaaXUNI7bbb"));
        assert!(!digest_matches("XEN11", false, "XUNI", "aaaXUNI7bbb"));
    }

    #[test]
    fn zero_workers_is_a_configuration_error_not_a_silent_no_op() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(100));
        let sink = sink(&state, dir.path());
        let error = CpuMiningWorker::new(
            CpuWorkerConfig::default(),
            state,
            sink,
            MiningIdentity::new("0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc"),
        )
        .expect_err("must reject");
        assert_eq!(error, CpuWorkerError::NoWorkers);
    }

    #[test]
    fn workers_idle_above_the_difficulty_ceiling_and_report_zero() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(50_000));
        let sink = sink(&state, dir.path());
        let worker = CpuMiningWorker::new(
            CpuWorkerConfig {
                worker_count: 1,
                batch_size: 1,
                max_difficulty: 100,
            },
            Arc::clone(&state),
            sink,
            MiningIdentity::new("0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc"),
        )
        .expect("worker");

        worker.start();
        // Long enough for the worker to reach the gate at least once.
        std::thread::sleep(Duration::from_millis(120));
        let stats = worker.stats();
        worker.stop();
        worker.join();

        assert!(stats.paused_for_difficulty, "must idle above the ceiling");
        assert_eq!(stats.hashrate, 0.0);
        assert_eq!(stats.attempts, 0, "no hashing happens above the ceiling");
    }

    #[test]
    fn workers_hash_and_account_their_attempts_below_the_ceiling() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(8));
        let sink = sink(&state, dir.path());
        let worker = CpuMiningWorker::new(
            CpuWorkerConfig {
                worker_count: 1,
                batch_size: 2,
                max_difficulty: 100,
            },
            Arc::clone(&state),
            sink,
            MiningIdentity::new("0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc"),
        )
        .expect("worker");

        worker.start();
        std::thread::sleep(Duration::from_millis(150));
        worker.stop();
        worker.join();

        let stats = worker.stats();
        assert!(stats.attempts > 0, "the CPU path must actually hash");
        assert!(stats.last_error.is_none(), "{:?}", stats.last_error);
    }
}
