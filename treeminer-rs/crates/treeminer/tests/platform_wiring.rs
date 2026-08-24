//! Platform mode, wired into the miner.
//!
//! Every test here drives the real [`PlatformRuntime`] over the crate's `Transport` trait,
//! so a signed command travels the same path it would from a broker — enqueue, envelope
//! check, handler, coordinator — and the assertion is on what the MINER does about it, not
//! on the platform state machine (that is `tm-platform`'s own test suite).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::{json, Value};
use tm_core::batch::BatchSizeDecision;
use tm_gpu::{BatchOutcome, BatchRequest, GpuMatch};
use tm_platform::clock::{Clock, TestClock};
use tm_platform::envelope::sign_command;
use tm_platform::manager::PlatformConfig;
use tm_platform::secret::Secret;
use tm_platform::transport::{Transport, TransportError};
use tm_platform::PlatformState;

use treeminer::backend::{DeviceFacts, MiningBackend};
use treeminer::find::FindSink;
use treeminer::mineunit::{run_mining_on_device, select_work, MineDeps, MineUnit, MiningIdentity};
use treeminer::platform::{PlatformOptions, PlatformRuntime};
use treeminer::state::MiningState;
use treeminer::stats::{StatsIdentity, StatsPublisher};
use treeminer::testsupport::RecordingJournal;

const WORKER: &str = "rig-01";
const SECRET: &str = "correct horse battery staple";
const NOW: i64 = 1_700_000_000;
const SELF_ADDRESS: &str = "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359";
const CONSUMER_ADDRESS: &str = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";
const PREFIX: &str = "a1b2c3d4e5f6a7b8";
const DEVFEE_ADDRESS: &str = "0x24691E54aFafe2416a8252097C9Ca67557271475";

/// A broker that goes nowhere.
#[derive(Debug, Default)]
struct FakeTransport {
    published: Mutex<Vec<(String, String)>>,
}

impl FakeTransport {
    fn published_on(&self, suffix: &str) -> Vec<Value> {
        self.published
            .lock()
            .iter()
            .filter(|(topic, _)| topic.ends_with(&format!("/{suffix}")))
            .filter_map(|(_, body)| serde_json::from_str(body).ok())
            .collect()
    }
}

impl Transport for FakeTransport {
    fn publish(&self, topic: &str, payload: &str) -> Result<(), TransportError> {
        self.published
            .lock()
            .push((topic.to_owned(), payload.to_owned()));
        Ok(())
    }
    fn subscribe(&self, _topic: &str) -> Result<(), TransportError> {
        Ok(())
    }
    fn is_connected(&self) -> bool {
        true
    }
}

/// A device that never runs out of work, so a test can watch the loop keep going — or stop.
struct EndlessBackend {
    facts: DeviceFacts,
    batches: Arc<AtomicU64>,
    /// The salt of the most recent batch: what the rig is actually mining for.
    last_salt: Arc<Mutex<String>>,
    matches: Arc<Mutex<Vec<GpuMatch>>>,
}

impl EndlessBackend {
    fn new() -> Self {
        Self {
            facts: DeviceFacts {
                index: 0,
                name: "Fake GPU".to_owned(),
                bus_id: 3,
                total_memory_bytes: 24 * 1024 * 1024 * 1024,
            },
            batches: Arc::new(AtomicU64::new(0)),
            last_salt: Arc::new(Mutex::new(String::new())),
            matches: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl MiningBackend for EndlessBackend {
    fn device(&self) -> DeviceFacts {
        self.facts.clone()
    }
    fn release_buffers(&mut self) {}
    fn plan_batch_size(&mut self, _d: u32, _m: usize, _s: usize) -> Result<BatchSizeDecision, String> {
        Ok(BatchSizeDecision {
            memory_limited_batch_size: 1,
            tuned_batch_size: 1,
            selected_batch_size: 1,
            explicit_limit_applied: false,
            tuned_default_applied: false,
        })
    }
    fn run_batch(&mut self, request: &BatchRequest<'_>) -> Result<BatchOutcome, String> {
        *self.last_salt.lock() = request.salt_hex.to_owned();
        self.batches.fetch_add(1, Ordering::Release);
        // A tiny sleep keeps a paused-vs-running comparison from being decided by how fast
        // this machine spins rather than by the flag.
        std::thread::sleep(Duration::from_millis(2));
        Ok(BatchOutcome {
            attempts: request.passwords.len(),
            gpu_first_blocks: request.gpu_first_blocks,
            matches: std::mem::take(&mut *self.matches.lock()),
            ..BatchOutcome::default()
        })
    }
}

fn base_identity() -> MiningIdentity {
    MiningIdentity {
        user_address: SELF_ADDRESS.to_owned(),
        devfee_address: DEVFEE_ADDRESS.to_owned(),
        eco_devfee_address: String::new(),
        devfee_permillage: 10,
        self_mining_prefix: String::new(),
        test_block_pattern: None,
    }
}

struct Harness {
    runtime: Arc<PlatformRuntime<Arc<FakeTransport>>>,
    transport: Arc<FakeTransport>,
    clock: Arc<TestClock>,
    state: Arc<MiningState>,
    nonce: AtomicU64,
}

impl Harness {
    /// A runtime with a secret, announced but with no threads running: commands are
    /// dispatched synchronously.
    fn new() -> Self {
        let transport = Arc::new(FakeTransport::default());
        let clock = Arc::new(TestClock::new(NOW));
        let mut config = PlatformConfig::new(WORKER, SELF_ADDRESS);
        config.command_secret = Some(Secret::new(SECRET));
        let runtime = PlatformRuntime::new(
            config,
            Arc::clone(&transport),
            base_identity(),
            8,
            clock.clone(),
        );
        runtime.manager().announce();
        Self {
            runtime,
            transport,
            clock,
            state: Arc::new(MiningState::for_test(8)),
            nonce: AtomicU64::new(0),
        }
    }

    fn sign(&self, msg: &Value) -> Value {
        let n = self.nonce.fetch_add(1, Ordering::Relaxed) + 1;
        let now = self.clock.now_epoch_s();
        sign_command(
            msg,
            SECRET,
            WORKER,
            &format!("cmd-{n}"),
            &format!("{n:032x}"),
            now,
            now + 60,
        )
    }

    /// Deliver a signed command through the intake path a broker would use.
    fn deliver(&self, suffix: &str, msg: &Value) {
        let signed = self.sign(msg);
        self.runtime.manager().enqueue_command(
            &format!("xenminer/{WORKER}/{suffix}"),
            signed.to_string().as_bytes(),
        );
        self.runtime.manager().dispatch_pending();
    }

    /// Deliver without waiting for a synchronous dispatch — for a manager whose own thread
    /// is draining the queue.
    fn post(&self, suffix: &str, msg: &Value) {
        let signed = self.sign(msg);
        self.runtime.manager().enqueue_command(
            &format!("xenminer/{WORKER}/{suffix}"),
            signed.to_string().as_bytes(),
        );
    }

    fn assign(&self, lease_id: &str) {
        self.deliver(
            "task",
            &json!({
                "command": "assign_task",
                "lease_id": lease_id,
                "consumer_id": "consumer-3",
                "consumer_address": CONSUMER_ADDRESS,
                "prefix": PREFIX,
                "duration_sec": 3600,
            }),
        );
    }
}

/// Poll until `predicate` holds, or fail. Never sleeps longer than it must; the deadline is
/// generous because the whole suite runs in parallel on a machine that may also be mining.
fn wait_for(what: &str, predicate: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if predicate() {
            return;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    panic!("timed out waiting for {what}");
}

fn sink(state: &Arc<MiningState>, dir: &std::path::Path) -> (Arc<FindSink>, Arc<RecordingJournal>) {
    sink_with(state, dir, None)
}

/// The find sink, optionally reporting to a platform runtime — the wiring `run` installs.
fn sink_with(
    state: &Arc<MiningState>,
    dir: &std::path::Path,
    observer: Option<treeminer::find::FindObserver>,
) -> (Arc<FindSink>, Arc<RecordingJournal>) {
    let journal = Arc::new(RecordingJournal::default());
    let mut sink = FindSink::new(
        Arc::clone(&journal) as Arc<dyn tm_journal::Journal + Send + Sync>,
        tm_journal::FallbackSink::new(dir.join("fallback.jsonl")),
        Arc::clone(state),
        WORKER,
    )
    .with_clock(|| "2026-01-01T00:00:00Z".to_owned())
    .trusting_digests();
    if let Some(observer) = observer {
        sink = sink.with_observer(observer);
    }
    (Arc::new(sink), journal)
}

// --- platform mode off ---

/// The flag gates EVERYTHING. With it clear, the credential lookup that would otherwise be
/// a hard startup failure never happens — which is the observable proof that nothing else
/// did either.
#[test]
fn platform_mode_off_reads_no_credentials() {
    let state = Arc::new(MiningState::for_test(8));
    let result = treeminer::platform::start_if_enabled(
        &PlatformOptions {
            enabled: false,
            broker_uri: "tcp://broker.invalid:1883",
            worker_id: WORKER,
            eth_address: SELF_ADDRESS,
        },
        &[],
        &BTreeMap::new(),
        &base_identity(),
        &state,
    );
    assert!(
        matches!(result, Ok(None)),
        "platform mode off must build nothing: {result:?}"
    );
    assert!(!state.is_mining_paused());
}

/// Without a runtime the mining loop reads the resolved identity, the dashboard reports the
/// C++ "disabled" shape, and nothing about a batch changes.
#[test]
fn without_platform_mode_nothing_changes() {
    let state = Arc::new(MiningState::for_test(8));
    let dir = tempfile::tempdir().expect("tempdir");
    let (sink, _journal) = sink(&state, dir.path());

    let deps = MineDeps::new(Arc::clone(&state), sink, base_identity());
    assert!(deps.identity_source.is_none());
    assert_eq!(deps.current_identity(), base_identity());
    // Batch 0 is an ordinary batch and batch 995 is a devfee batch: the fee rotation is
    // untouched when nothing is leasing the rig.
    assert_eq!(
        select_work(&deps.current_identity(), 0).salt_hex,
        SELF_ADDRESS.trim_start_matches("0x")
    );
    assert_eq!(
        select_work(&deps.current_identity(), 995).salt_hex,
        DEVFEE_ADDRESS.trim_start_matches("0x")
    );

    let publisher = StatsPublisher::new(Arc::clone(&state), StatsIdentity::default());
    let snapshot = publisher.stats_snapshot();
    assert!(snapshot.platform.is_none());
    let payload = tm_dashboard::platform_payload(&snapshot);
    assert_eq!(payload["platform_mode"], false);
    assert_eq!(payload["platform_state"], "disabled");
    assert_eq!(payload["running"], false);
    assert!(payload.get("lease_id").is_none());
}

// --- assign_task redirects the mining address ---

/// The whole point of a lease: the salt every batch hashes against becomes the consumer's
/// address, the platform prefix is applied, and the devfee rotation is suppressed because
/// the consumer is paying for the whole machine.
#[test]
fn a_signed_assign_task_redirects_what_the_miner_mines_to() {
    let harness = Harness::new();
    let state = Arc::clone(&harness.state);
    // Started, because a find is only reported to a platform that is actually running —
    // the same guard the C++ applies before `onBlockFound`.
    harness.runtime.start(&state);
    let dir = tempfile::tempdir().expect("tempdir");
    let (sink, journal) = sink_with(
        &state,
        dir.path(),
        Some(harness.runtime.find_observer()),
    );

    let deps = MineDeps::new(Arc::clone(&state), sink, base_identity())
        .with_identity_source(harness.runtime.identity_source());
    assert_eq!(deps.current_identity().user_address, SELF_ADDRESS);

    harness.assign("lease-7");
    wait_for("the lease to start", || {
        harness.runtime.manager().state() == PlatformState::Mining
    });

    let leased = deps.current_identity();
    assert_eq!(leased.user_address, CONSUMER_ADDRESS);
    assert_eq!(leased.self_mining_prefix, PREFIX);
    assert_eq!(leased.devfee_permillage, 0, "a leased rig pays no devfee");
    // Every batch of the thousand, including the ones that would have been fee batches.
    for batch in [0, 500, 995, 999] {
        let work = select_work(&leased, batch);
        assert_eq!(
            work.salt_hex,
            CONSUMER_ADDRESS.trim_start_matches("0x"),
            "batch {batch} did not mine for the consumer"
        );
        assert_eq!(work.key_prefix, PREFIX);
    }

    // ...and a find made under the lease is journaled against the consumer's account.
    let mut backend = EndlessBackend::new();
    *backend.matches.lock() = vec![GpuMatch {
        key: "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f".to_owned(),
        hash: "XEN11abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"
            .to_owned(),
        matched_pattern: "XEN11".to_owned(),
        attempt_index: 0,
        is_superblock: false,
    }];
    let salt = Arc::clone(&backend.last_salt);
    let stop = Arc::clone(state.shutdown());
    let batches = Arc::clone(&backend.batches);
    std::thread::spawn(move || {
        wait_for("one batch", || batches.load(Ordering::Acquire) > 0);
        stop.request_stop();
    });
    MineUnit::new(&mut backend, &deps, 8, 0).run();

    assert_eq!(*salt.lock(), CONSUMER_ADDRESS.trim_start_matches("0x"));
    let appended = journal.appended();
    assert_eq!(appended.len(), 1, "the find was not journaled");
    assert_eq!(appended[0].account, CONSUMER_ADDRESS);

    // The lease shows up in the operator console, with the key set the C++ served.
    let publisher = StatsPublisher::new(Arc::clone(&state), StatsIdentity::default())
        .with_platform(harness.runtime.status_provider());
    let payload = tm_dashboard::platform_payload(&publisher.stats_snapshot());
    assert_eq!(payload["platform_mode"], true);
    assert_eq!(payload["mining_mode"], "platform");
    assert_eq!(payload["platform_state"], "MINING");
    assert_eq!(payload["lease_id"], "lease-7");
    assert_eq!(payload["consumer_id"], "consumer-3");
    assert_eq!(payload["consumer_address"], CONSUMER_ADDRESS);
    assert_eq!(payload["blocks_found"], 1, "the find was not attributed to the lease");
    assert_eq!(payload["remaining_sec"], 3600);
    // The find reached the broker too, against this lease.
    let blocks = harness.transport.published_on("block");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["lease_id"], "lease-7");
    assert_eq!(blocks[0]["account"], CONSUMER_ADDRESS);

    harness.runtime.stop();
}

/// An UNSIGNED assign_task must change nothing — the security decision this wiring rests
/// on, asserted here on the miner side rather than the manager's.
#[test]
fn an_unsigned_assign_task_does_not_redirect_anything() {
    let harness = Harness::new();
    let deps = MineDeps::new(
        Arc::clone(&harness.state),
        sink(&harness.state, tempfile::tempdir().expect("tempdir").path()).0,
        base_identity(),
    )
    .with_identity_source(harness.runtime.identity_source());

    harness.runtime.manager().enqueue_command(
        &format!("xenminer/{WORKER}/task"),
        json!({
            "command": "assign_task",
            "lease_id": "lease-7",
            "consumer_id": "consumer-3",
            "consumer_address": CONSUMER_ADDRESS,
            "prefix": PREFIX,
            "duration_sec": 3600,
        })
        .to_string()
        .as_bytes(),
    );
    harness.runtime.manager().dispatch_pending();

    assert_eq!(deps.current_identity(), base_identity());
    assert_eq!(harness.runtime.manager().state(), PlatformState::Available);
    assert!(harness.runtime.status().lease.is_none());
}

// --- pause / resume ---

/// `pause` idles the hashing loop without ending the run; `resume` starts it again on the
/// same thread and the same device context.
#[test]
fn pause_stops_the_mining_loop_and_resume_restarts_it() {
    let harness = Harness::new();
    let state = Arc::clone(&harness.state);
    harness.runtime.start(&state);
    let dir = tempfile::tempdir().expect("tempdir");
    let (sink, _journal) = sink(&state, dir.path());
    let deps = Arc::new(
        MineDeps::new(Arc::clone(&state), sink, base_identity())
            .with_identity_source(harness.runtime.identity_source()),
    );

    let mut backend = EndlessBackend::new();
    let batches = Arc::clone(&backend.batches);
    let mining = std::thread::spawn({
        let deps = Arc::clone(&deps);
        move || run_mining_on_device(&mut backend, &deps, 0, Duration::from_millis(10))
    });

    wait_for("mining to start", || batches.load(Ordering::Acquire) > 0);

    harness.post("control", &json!({ "action": "pause" }));
    wait_for("the pause to take effect", || state.is_mining_paused());

    // Let any batch already in flight finish, then prove the loop is genuinely idle.
    std::thread::sleep(Duration::from_millis(150));
    let frozen = batches.load(Ordering::Acquire);
    std::thread::sleep(Duration::from_millis(250));
    assert_eq!(
        batches.load(Ordering::Acquire),
        frozen,
        "the mining loop kept hashing while paused"
    );
    assert!(!mining.is_finished(), "pause killed the mining thread");

    harness.post("control", &json!({ "action": "resume" }));
    wait_for("the resume to take effect", || !state.is_mining_paused());
    wait_for("mining to restart", || {
        batches.load(Ordering::Acquire) > frozen
    });

    state.shutdown().request_stop();
    mining.join().expect("mining thread");
    harness.runtime.stop();
}

// --- remote shutdown ---

/// A signed `shutdown` must take the SIGINT path: flip the one shutdown flag and let the
/// main thread tear everything down in order. Anything abrupt loses whatever the mining
/// loop was holding.
#[test]
fn a_signed_shutdown_takes_the_same_path_as_a_signal() {
    let harness = Harness::new();
    let state = Arc::clone(&harness.state);
    harness.runtime.start(&state);
    let dir = tempfile::tempdir().expect("tempdir");
    let (sink, _journal) = sink(&state, dir.path());
    let deps = Arc::new(MineDeps::new(Arc::clone(&state), sink, base_identity()));

    let mut backend = EndlessBackend::new();
    let batches = Arc::clone(&backend.batches);
    let mining = std::thread::spawn({
        let deps = Arc::clone(&deps);
        move || run_mining_on_device(&mut backend, &deps, 0, Duration::from_millis(10))
    });
    wait_for("mining to start", || batches.load(Ordering::Acquire) > 0);
    assert!(state.is_running());

    harness.post("control", &json!({ "action": "shutdown" }));

    // The same flag a signal flips, so the same orderly teardown follows.
    wait_for("the shutdown flag", || !state.is_running());
    mining.join().expect("the mining thread exited on its own");
    harness.runtime.stop();
    assert!(!harness.runtime.manager().is_running());
    assert!(harness.runtime.manager().shutdown_requested());
    // The graceful goodbye, not a dropped socket.
    let offline = harness.transport.published_on("status");
    assert!(
        offline.iter().any(|m| m["status"] == "offline"),
        "no offline notice was published: {offline:?}"
    );
}

/// An UNSIGNED shutdown is refused, and the miner keeps running. Same fixture, opposite
/// outcome, so the test above cannot be passing for the wrong reason.
#[test]
fn an_unsigned_shutdown_does_not_stop_the_miner() {
    let harness = Harness::new();
    let state = Arc::clone(&harness.state);
    harness.runtime.start(&state);

    harness.runtime.manager().enqueue_command(
        &format!("xenminer/{WORKER}/control"),
        json!({ "action": "shutdown" }).to_string().as_bytes(),
    );
    std::thread::sleep(Duration::from_millis(300));

    assert!(state.is_running());
    assert!(!harness.runtime.manager().shutdown_requested());

    state.shutdown().request_stop();
    harness.runtime.stop();
}

// --- credentials ---

/// A missing credential is a startup failure that names the variable. It must not degrade
/// into an unauthenticated connection.
#[test]
fn missing_credentials_fail_at_startup_naming_the_variable() {
    static ENV_LOCK: Mutex<()> = Mutex::new(());
    let _guard = ENV_LOCK.lock();
    for var in [
        "TREEMINER_PLATFORM_COMMAND_SECRET",
        "TREEMINER_MQTT_USERNAME",
        "TREEMINER_MQTT_PASSWORD",
        "TREEMINER_MQTT_ANONYMOUS",
    ] {
        std::env::remove_var(var);
    }
    let state = Arc::new(MiningState::for_test(8));
    let options = PlatformOptions {
        enabled: true,
        broker_uri: "tcp://broker.invalid:1883",
        worker_id: WORKER,
        eth_address: SELF_ADDRESS,
    };

    let error = treeminer::platform::start_if_enabled(
        &options,
        &[],
        &BTreeMap::new(),
        &base_identity(),
        &state,
    )
    .expect_err("platform mode started without a command secret");
    let message = error.to_string();
    assert!(
        message.contains("TREEMINER_PLATFORM_COMMAND_SECRET"),
        "the error must name the variable: {message}"
    );
    assert!(message.contains("--platform-mode"), "{message}");

    // With the secret set, the broker credentials are the next thing named — never
    // silently skipped.
    std::env::set_var("TREEMINER_PLATFORM_COMMAND_SECRET", SECRET);
    let error = treeminer::platform::start_if_enabled(
        &options,
        &[],
        &BTreeMap::new(),
        &base_identity(),
        &state,
    )
    .expect_err("platform mode connected anonymously by accident");
    let message = error.to_string();
    assert!(message.contains("TREEMINER_MQTT_ANONYMOUS"), "{message}");
    assert!(!message.contains(SECRET), "the error leaked the secret: {message}");
    std::env::remove_var("TREEMINER_PLATFORM_COMMAND_SECRET");
}

/// The pause flag is a miner-wide gate, not a platform detail: nothing may leave it set on
/// the way out, or a join would hang on a politely idling thread.
#[test]
fn a_paused_miner_still_shuts_down() {
    let state = Arc::new(MiningState::for_test(8));
    state.set_mining_paused(true);
    let dir = tempfile::tempdir().expect("tempdir");
    let (sink, _journal) = sink(&state, dir.path());
    let deps = Arc::new(MineDeps::new(Arc::clone(&state), sink, base_identity()));

    let mut backend = EndlessBackend::new();
    let batches = Arc::clone(&backend.batches);
    let mining = std::thread::spawn({
        let deps = Arc::clone(&deps);
        move || run_mining_on_device(&mut backend, &deps, 0, Duration::from_millis(10))
    });
    std::thread::sleep(Duration::from_millis(150));
    assert_eq!(batches.load(Ordering::Acquire), 0, "a paused loop hashed");

    state.shutdown().request_stop();
    let deadline = Instant::now() + Duration::from_secs(30);
    while !mining.is_finished() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(mining.is_finished(), "a paused mining thread did not stop");
    mining.join().expect("mining thread");
}

/// A `set_config` difficulty change reaches the mining loop, as it does in the C++ (which
/// writes `globalDifficulty` directly).
#[test]
fn a_signed_set_config_moves_the_miners_difficulty_and_address() {
    let harness = Harness::new();
    let state = Arc::clone(&harness.state);
    harness.runtime.start(&state);

    harness.post(
        "control",
        &json!({
            "action": "set_config",
            "config": { "difficulty": 4096, "address": CONSUMER_ADDRESS },
        }),
    );
    wait_for("the difficulty to reach the mining state", || {
        state.difficulty() == 4096
    });
    assert_eq!(harness.runtime.current_identity().user_address, CONSUMER_ADDRESS);
    // Still self-mining: `set_config` changes the operator's own identity, it does not
    // start a lease.
    assert_eq!(harness.runtime.status().mining_mode, "self");

    state.shutdown().request_stop();
    harness.runtime.stop();
}
