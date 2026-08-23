//! The mining loop. Port of `src/MineUnit.{h,cpp}` and `runMiningOnDevice` in
//! `src/main.cpp`.
//!
//! A `MineUnit` is bound to ONE memory cost for its whole life. That is the structural
//! reason the stale-difficulty bug cannot come back: the unit sizes its batch, builds its
//! kernel request and stamps its finds from the same `difficulty` field, and when the
//! network difficulty or the submitter's margin moves the unit ends and a new one is built
//! at the new cost. Nothing downstream ever re-derives the memory cost of a find.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tm_argon2::RandomHexKeyGenerator;
use tm_dashboard::stats::GpuStat;
use tm_gpu::BatchRequest;
use tm_tui::{Console, Level};

use crate::backend::{DeviceFacts, MiningBackend};
use crate::find::{Find, FindSink};
use crate::state::MiningState;

/// Key prefixes that redirect a batch to a fee address. Port of the `MiningCommon.h`
/// constants.
pub const DEVFEE_PREFIX: &str = "FFFFFFFF";
pub const ECODEVFEE_PREFIX: &str = "EEEEEEEE";

/// The C++ devfee cycle length: one batch in every thousand can be a fee batch.
const DEVFEE_CYCLE: i32 = 1000;

/// Length of an Argon2 password in hex characters.
const KEY_LENGTH: usize = 64;

/// Who a batch mines for. Port of `MiningIdentityConfig` plus the devfee globals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MiningIdentity {
    /// `0x`-prefixed reward address.
    pub user_address: String,
    pub devfee_address: String,
    /// Empty disables the ecosystem split.
    pub eco_devfee_address: String,
    pub devfee_permillage: i32,
    /// Remote-controlled key prefix; when set it suppresses the fee rotation entirely, as
    /// in the C++.
    pub self_mining_prefix: String,
    /// `--testBlockPattern`; `None` means `XEN11`.
    pub test_block_pattern: Option<String>,
}

impl MiningIdentity {
    pub fn new(user_address: impl Into<String>) -> Self {
        Self {
            user_address: user_address.into(),
            devfee_address: String::new(),
            eco_devfee_address: String::new(),
            devfee_permillage: 0,
            self_mining_prefix: String::new(),
            test_block_pattern: None,
        }
    }

    pub fn block_pattern(&self) -> &str {
        self.test_block_pattern.as_deref().unwrap_or("XEN11")
    }

    /// The address hex a batch mines against, without `0x`.
    fn user_salt(&self) -> &str {
        self.user_address.strip_prefix("0x").unwrap_or(&self.user_address)
    }
}

/// Salt and key prefix for one batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Work {
    pub salt_hex: String,
    pub key_prefix: String,
}

/// Which address batch number `batch_index` (0..1000) mines for.
///
/// The fee batches are the last `permillage` of every thousand, and the last half of those
/// go to the ecosystem address when one is configured — the exact C++ arithmetic, kept
/// because it decides where real value lands.
pub fn select_work(identity: &MiningIdentity, batch_index: i32) -> Work {
    let user_salt = identity.user_salt().to_owned();
    if !identity.self_mining_prefix.is_empty() {
        return Work {
            salt_hex: user_salt,
            key_prefix: identity.self_mining_prefix.clone(),
        };
    }
    if DEVFEE_CYCLE - batch_index > identity.devfee_permillage {
        return Work {
            salt_hex: user_salt,
            key_prefix: String::new(),
        };
    }
    let eco = DEVFEE_CYCLE - batch_index <= identity.devfee_permillage / 2
        && !identity.eco_devfee_address.is_empty();
    let (address, prefix) = if eco {
        (&identity.eco_devfee_address, ECODEVFEE_PREFIX)
    } else {
        (&identity.devfee_address, DEVFEE_PREFIX)
    };
    Work {
        salt_hex: address.strip_prefix("0x").unwrap_or(address).to_owned(),
        key_prefix: format!("{prefix}{user_salt}"),
    }
}

/// True when the local clock is inside the XUNI window (`:55`–`:05`).
pub fn xuni_window_open_now() -> bool {
    tm_core::is_within_xuni_window_at(u32::from(crate::clock::now_local().minute()))
}

/// Everything a mining thread needs besides its device.
pub struct MineDeps {
    pub state: Arc<MiningState>,
    pub sink: Arc<FindSink>,
    pub identity: MiningIdentity,
    /// `--batchSize`; 0 means "use all free VRAM".
    pub max_batch_size: usize,
    pub streams_per_device: usize,
    /// Injected so the window boundary can be exercised without waiting for :55.
    pub xuni_window_open: Arc<dyn Fn() -> bool + Send + Sync>,
}

impl MineDeps {
    pub fn new(state: Arc<MiningState>, sink: Arc<FindSink>, identity: MiningIdentity) -> Self {
        Self {
            state,
            sink,
            identity,
            max_batch_size: 0,
            streams_per_device: 1,
            xuni_window_open: Arc::new(xuni_window_open_now),
        }
    }
}

/// Why a unit stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoopExit {
    /// `running` cleared: shut down.
    Stopped,
    /// Difficulty or margin moved; rebuild at the new memory cost immediately.
    DifficultyChanged,
    /// Something transient (a failed allocation, a failed batch). The caller backs off
    /// before retrying — retrying instantly is what used to spin a core and flood the log.
    Recoverable(String),
}

/// One device, one memory cost.
pub struct MineUnit<'a> {
    backend: &'a mut dyn MiningBackend,
    deps: &'a MineDeps,
    facts: DeviceFacts,
    difficulty: u32,
    stream_index: i32,
    batch_size: usize,
    used_memory_bytes: u64,
    hash_total: u64,
    /// Attempts since the last submitted match, carried across batches exactly as the C++
    /// does so a find reports how much work preceded it.
    attempts: u64,
    hashrate: f64,
    started_at: Instant,
}

impl<'a> MineUnit<'a> {
    pub fn new(
        backend: &'a mut dyn MiningBackend,
        deps: &'a MineDeps,
        difficulty: u32,
        stream_index: i32,
    ) -> Self {
        let facts = backend.device();
        Self {
            backend,
            deps,
            facts,
            difficulty,
            stream_index,
            batch_size: 1,
            used_memory_bytes: 0,
            hash_total: 0,
            attempts: 0,
            hashrate: 0.0,
            started_at: Instant::now(),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Hash at this unit's memory cost until it stops being the right one.
    pub fn run(&mut self) -> LoopExit {
        // A prior difficulty's pool can hold nearly all of the VRAM. Release it BEFORE
        // measuring, or a difficulty increase sizes its batch from the scraps left over.
        self.backend.release_buffers();
        let decision = match self.backend.plan_batch_size(
            self.difficulty,
            self.deps.max_batch_size,
            self.deps.streams_per_device,
        ) {
            Ok(decision) => decision,
            Err(error) => return LoopExit::Recoverable(error),
        };
        if decision.selected_batch_size == 0 {
            tm_tui::log_to_console("GPU memory allocation unavailable; retrying with backoff");
            return LoopExit::Recoverable("GPU memory allocation unavailable".to_owned());
        }
        self.batch_size = decision.selected_batch_size;
        self.used_memory_bytes = self.batch_size as u64 * u64::from(self.difficulty) * 1024;
        self.started_at = Instant::now();

        let mut batch_index: i32 = 0;
        while self.deps.state.is_running() {
            if self.deps.state.effective_difficulty() != self.difficulty {
                return LoopExit::DifficultyChanged;
            }

            let work = select_work(&self.deps.identity, batch_index);
            let keys = generate_keys(&work.key_prefix, self.batch_size);
            let pattern = self.deps.identity.block_pattern().to_owned();

            let mut request = BatchRequest::new(&keys, &work.salt_hex, self.difficulty);
            request.target_pattern = &pattern;
            // Decided once, at batch start. A XUNI found as the window closes mid-batch is
            // still captured here; the submission layer parks it (ParkedXuniWindow) rather
            // than this loop dropping a find the miner already paid for.
            request.allow_xuni = (self.deps.xuni_window_open)();
            // GPU first blocks: whatever the startup self-test proved about THIS device.
            // Never a build-time guess — that path once produced invalid digests.
            request.gpu_first_blocks = self
                .deps
                .state
                .gpu_first_blocks_verified(self.facts.index);

            let outcome = match self.backend.run_batch(&request) {
                Ok(outcome) => outcome,
                Err(error) => {
                    tm_tui::log_to_console(&format!("Hash batch failed: {error}"));
                    return LoopExit::Recoverable(error);
                }
            };

            self.submit_matches(&work.salt_hex, &outcome);
            self.publish_stats();

            batch_index += 1;
            if batch_index >= DEVFEE_CYCLE {
                batch_index = 0;
            }
        }
        LoopExit::Stopped
    }

    /// Hand every match to the journal-first sink, stamped with THIS unit's memory cost.
    fn submit_matches(&mut self, salt_hex: &str, outcome: &tm_gpu::BatchOutcome) {
        let mut next_attempt_index = 0usize;
        for hit in &outcome.matches {
            if hit.attempt_index >= next_attempt_index {
                self.attempts += (hit.attempt_index - next_attempt_index + 1) as u64;
                next_attempt_index = hit.attempt_index + 1;
            }
            self.deps.sink.record(&Find {
                hexsalt: salt_hex.to_owned(),
                key: hit.key.clone(),
                digest: hit.hash.clone(),
                memory_cost: self.difficulty,
                attempts: self.attempts,
                hashes_per_second: self.hashrate,
                source: "GPU".to_owned(),
            });
            self.attempts = 0;
        }
        if outcome.attempts >= next_attempt_index {
            self.attempts += (outcome.attempts - next_attempt_index) as u64;
        }
    }

    /// Port of `MineUnit::stat()`: cumulative hashrate for this unit, published as one row
    /// per device/stream.
    fn publish_stats(&mut self) {
        self.hash_total += self.batch_size as u64;
        self.deps.state.add_hashes(self.batch_size as u64);

        let elapsed_ms = self.started_at.elapsed().as_millis().max(1) as f64;
        self.hashrate = self.hash_total as f64 / elapsed_ms * 1000.0;

        let total = self.facts.total_memory_bytes.max(1) as f32;
        self.deps.state.publish_gpu(GpuStat {
            index: self.facts.index,
            bus_id: self.facts.bus_id,
            name: self.facts.name.clone(),
            memory: (self.facts.total_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)).round()
                as i32,
            using_memory: self.used_memory_bytes as f32 / total,
            temperature: 0,
            hashrate: self.hashrate as f32,
            power: String::new(),
            hash_count: self.hash_total,
            stream_index: self.stream_index,
            telemetry: None,
            updated_secs_ago: 0,
        });
    }
}

/// Keep one device mining across difficulty changes and transient failures. Port of
/// `runMiningOnDevice`.
///
/// `backoff` is how long a recoverable exit waits before rebuilding. Upstream retried
/// instantly, which spun a core and flooded the log during an allocation failure.
pub fn run_mining_on_device(
    backend: &mut dyn MiningBackend,
    deps: &MineDeps,
    stream_index: i32,
    backoff: Duration,
) {
    while deps.state.is_running() {
        let difficulty = deps.state.effective_difficulty();
        let exit = MineUnit::new(backend, deps, difficulty, stream_index).run();
        match exit {
            LoopExit::Stopped | LoopExit::DifficultyChanged => {}
            LoopExit::Recoverable(reason) => {
                Console::global().event(
                    Level::Warn,
                    "MINING",
                    &format!(
                        "device #{} paused {}s after a recoverable failure | {reason}",
                        backend.device().index,
                        backoff.as_secs_f64()
                    ),
                );
                sleep_interruptible(deps, backoff);
            }
        }
    }
}

/// Wait, but never longer than the shutdown flag allows.
fn sleep_interruptible(deps: &MineDeps, total: Duration) {
    let deadline = Instant::now() + total;
    while deps.state.is_running() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50).min(total));
    }
}

/// One batch's worth of Argon2 passwords.
fn generate_keys(prefix: &str, count: usize) -> Vec<String> {
    let mut generator = RandomHexKeyGenerator::new(prefix, KEY_LENGTH);
    (0..count).map(|_| generator.next_random_key()).collect()
}

/// The default recoverable-exit backoff, as in the C++.
pub const DEFAULT_BACKOFF: Duration = Duration::from_secs(5);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{BackendEvent, FakeBackend, RecordingJournal};
    use tm_gpu::GpuMatch;

    fn identity() -> MiningIdentity {
        MiningIdentity::new("0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc")
    }

    fn deps(state: Arc<MiningState>, journal: Arc<RecordingJournal>, dir: &std::path::Path) -> MineDeps {
        let sink = crate::find::FindSink::new(
            journal,
            tm_journal::FallbackSink::new(dir.join("fallback.jsonl")),
            Arc::clone(&state),
            "worker-1",
        )
        .with_clock(|| "2026-01-01T00:00:00Z".to_owned());
        MineDeps::new(state, Arc::new(sink), identity())
    }

    fn a_match(pattern: &str, digest: &str) -> GpuMatch {
        GpuMatch {
            key: "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f".to_owned(),
            hash: digest.to_owned(),
            matched_pattern: pattern.to_owned(),
            attempt_index: 3,
            is_superblock: false,
        }
    }

    #[test]
    fn fee_batches_are_the_tail_of_every_thousand() {
        let mut identity = identity();
        identity.devfee_address = "0x24691E54aFafe2416a8252097C9Ca67557271475".to_owned();
        identity.devfee_permillage = 10;

        let own = select_work(&identity, 0);
        assert_eq!(own.salt_hex, "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc");
        assert!(own.key_prefix.is_empty());

        let fee = select_work(&identity, 995);
        assert_eq!(fee.salt_hex, "24691E54aFafe2416a8252097C9Ca67557271475");
        assert!(fee.key_prefix.starts_with(DEVFEE_PREFIX));
    }

    #[test]
    fn the_ecosystem_address_takes_the_second_half_of_the_fee_batches() {
        let mut identity = identity();
        identity.devfee_address = "0x24691E54aFafe2416a8252097C9Ca67557271475".to_owned();
        identity.eco_devfee_address = "0x1111111111111111111111111111111111111111".to_owned();
        identity.devfee_permillage = 10;

        assert!(select_work(&identity, 991).key_prefix.starts_with(DEVFEE_PREFIX));
        assert!(select_work(&identity, 996)
            .key_prefix
            .starts_with(ECODEVFEE_PREFIX));
    }

    #[test]
    fn a_remote_prefix_suppresses_the_fee_rotation() {
        let mut identity = identity();
        identity.devfee_address = "0x24691E54aFafe2416a8252097C9Ca67557271475".to_owned();
        identity.devfee_permillage = 500;
        identity.self_mining_prefix = "ABCD".to_owned();

        let work = select_work(&identity, 999);
        assert_eq!(work.key_prefix, "ABCD");
        assert_eq!(work.salt_hex, "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc");
    }

    #[test]
    fn one_match_produces_exactly_one_journaled_payload_at_the_batch_s_own_cost() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(1000));
        let journal = Arc::new(RecordingJournal::default());
        let deps = deps(Arc::clone(&state), Arc::clone(&journal), dir.path());

        let mut backend = FakeBackend::new(8);
        backend.push_batch(vec![a_match("XEN11", "aaaXEN11bbb")]);
        // The network difficulty moves the instant the batch comes back, ending the unit —
        // this is the upstream race that used to rewrite (or drop) the find.
        let moved = Arc::clone(&state);
        backend.on_batch_complete(move || moved.set_difficulty(2000));

        let exit = MineUnit::new(&mut backend, &deps, 1000, 0).run();

        assert_eq!(exit, LoopExit::DifficultyChanged);
        let appended = journal.appended();
        assert_eq!(appended.len(), 1);
        assert_eq!(appended[0].memory_cost, 1000);
        assert!(appended[0].hash_to_verify.contains("m=1000,"));
    }

    #[test]
    fn a_xuni_found_as_the_window_closes_is_journaled_not_dropped() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(1000));
        let journal = Arc::new(RecordingJournal::default());
        let mut deps = deps(Arc::clone(&state), Arc::clone(&journal), dir.path());

        // Open when the batch is built, closed by the time the match comes back.
        let open = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let flag = Arc::clone(&open);
        deps.xuni_window_open =
            Arc::new(move || flag.load(std::sync::atomic::Ordering::SeqCst));

        let mut backend = FakeBackend::new(8);
        backend.push_batch(vec![a_match("XUNI", "aaaXUNI7bbb")]);
        let flag = Arc::clone(&open);
        let stop = Arc::clone(&state);
        backend.on_batch_complete(move || {
            flag.store(false, std::sync::atomic::Ordering::SeqCst);
            stop.shutdown().request_stop();
        });

        MineUnit::new(&mut backend, &deps, 1000, 0).run();

        let appended = journal.appended();
        assert_eq!(appended.len(), 1, "the closing window must not eat the find");
        assert_eq!(appended[0].kind, tm_core::FindKind::Xuni);
    }

    #[test]
    fn the_pool_is_released_before_every_batch_size_decision() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(1000));
        let journal = Arc::new(RecordingJournal::default());
        let deps = deps(Arc::clone(&state), journal, dir.path());

        let mut backend = FakeBackend::new(8);
        backend.push_batch(Vec::new());
        let moved = Arc::clone(&state);
        backend.on_batch_complete(move || moved.set_difficulty(2000));
        MineUnit::new(&mut backend, &deps, 1000, 0).run();

        // Rebuild at the new cost, as runMiningOnDevice would.
        backend.push_batch(Vec::new());
        let stop = Arc::clone(&state);
        backend.on_batch_complete(move || stop.shutdown().request_stop());
        MineUnit::new(&mut backend, &deps, 2000, 0).run();

        let plans: Vec<_> = backend
            .events()
            .into_iter()
            .filter(|event| {
                matches!(event, BackendEvent::Release | BackendEvent::Plan { .. })
            })
            .collect();
        assert_eq!(
            plans,
            vec![
                BackendEvent::Release,
                BackendEvent::Plan { difficulty: 1000 },
                BackendEvent::Release,
                BackendEvent::Plan { difficulty: 2000 },
            ],
            "free VRAM must be measured with the previous pool already gone"
        );
    }

    #[test]
    fn a_zero_batch_size_backs_off_instead_of_spinning() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(1000));
        let journal = Arc::new(RecordingJournal::default());
        let deps = deps(Arc::clone(&state), journal, dir.path());

        let mut backend = FakeBackend::new(0);
        let exit = MineUnit::new(&mut backend, &deps, 1000, 0).run();

        assert!(matches!(exit, LoopExit::Recoverable(_)));
        assert!(!backend
            .events()
            .iter()
            .any(|event| matches!(event, BackendEvent::Run { .. })));
    }

    #[test]
    fn a_failed_batch_ends_the_unit_recoverably() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(1000));
        let journal = Arc::new(RecordingJournal::default());
        let deps = deps(Arc::clone(&state), journal, dir.path());

        let mut backend = FakeBackend::new(8);
        backend.push_failure("out of memory");
        let exit = MineUnit::new(&mut backend, &deps, 1000, 0).run();

        assert_eq!(exit, LoopExit::Recoverable("out of memory".to_owned()));
    }

    #[test]
    fn the_devices_own_first_blocks_verdict_reaches_the_kernel_request() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(1000));
        state.set_gpu_first_blocks_verified(0, true);
        let journal = Arc::new(RecordingJournal::default());
        let deps = deps(Arc::clone(&state), journal, dir.path());

        let mut backend = FakeBackend::new(8);
        backend.push_batch(Vec::new());
        let stop = Arc::clone(&state);
        backend.on_batch_complete(move || stop.shutdown().request_stop());
        MineUnit::new(&mut backend, &deps, 1000, 0).run();

        let ran = backend
            .events()
            .into_iter()
            .find_map(|event| match event {
                BackendEvent::Run {
                    gpu_first_blocks, ..
                } => Some(gpu_first_blocks),
                _ => None,
            })
            .expect("a batch ran");
        assert!(ran);
    }

    #[test]
    fn a_recoverable_exit_stops_retrying_once_shutdown_is_requested() {
        let dir = tempfile::tempdir().expect("tempdir");
        let state = Arc::new(MiningState::for_test(1000));
        let journal = Arc::new(RecordingJournal::default());
        let deps = deps(Arc::clone(&state), journal, dir.path());

        // Never any capacity: every unit exits recoverably.
        let mut backend = FakeBackend::new(0);
        let stop = Arc::clone(&state);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(30));
            stop.shutdown().request_stop();
        });
        run_mining_on_device(&mut backend, &deps, 0, Duration::from_millis(10));
        assert!(!state.is_running());
    }
}
