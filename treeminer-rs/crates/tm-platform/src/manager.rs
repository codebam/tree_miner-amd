//! Registration, heartbeat, command dispatch and status. Port of
//! `PlatformManager.{h,cpp}`.
//!
//! # Command intake
//!
//! The broker's pump thread only ever *enqueues*; a single worker thread drains the
//! bounded queue in FIFO arrival order. This is the shape the C++ arrived at after the
//! hardening pass: the original detached-thread-per-message dispatch allowed unbounded
//! thread creation under flooding, use-after-free after destruction, and command
//! reordering.
//!
//! [`PlatformManager::dispatch_pending`] drains the same queue through the same code
//! synchronously, which is how the tests drive every handler without a broker or a thread.

use crate::clock::Clock;
use crate::coordinator::{MiningContext, MiningCoordinator, MiningMode};
use crate::envelope::{self, NonceCache, VerifyStatus};
use crate::lease::LeaseManager;
use crate::proto::{
    self, AssignTask, Command, ControlAction, GpuInfo, RegisterAck, Release, SetConfig,
    PLATFORM_PREFIX_LENGTH,
};
use crate::reporter::{WorkerReporter, WorkerStats};
use crate::secret::Secret;
use crate::transport::{build_topic, Transport};
use parking_lot::{Condvar, Mutex};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

pub const HEARTBEAT_INTERVAL_SEC: u64 = 30;
pub const WATCHDOG_INTERVAL_SEC: u64 = 5;

/// Legitimate platform traffic is a handful of commands per lease lifecycle; 256 gives
/// ample headroom while keeping worst-case memory (256 * 64 KiB) at 16 MiB. Overflow drops
/// the NEWEST message so a flood cannot displace commands already accepted.
pub const COMMAND_QUEUE_CAPACITY: usize = 256;
/// 4096 nonces at <= 15 min lifetime comfortably covers any legitimate signed command
/// rate; only authentically signed traffic can occupy a slot.
pub const NONCE_CACHE_CAPACITY: usize = 4096;

/// Bounds on `assign_task` fields, checked before any state changes.
const MAX_LEASE_ID_LEN: usize = 64;
const MIN_LEASE_DURATION_SEC: i64 = 60;
const MAX_LEASE_DURATION_SEC: i64 = 7 * 24 * 3600;
/// Argon2 memory cost in KiB. The upper bound is far above any plausible network value but
/// stops a hostile `set_config` from OOMing every GPU in the rig.
const MIN_DIFFICULTY: i64 = 1;
const MAX_DIFFICULTY: i64 = 10_000_000;
/// More prefix than half the 64-character key guts the search entropy.
const MAX_SELF_PREFIX_LEN: usize = 32;
const MAX_BLOCK_PATTERN_LEN: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlatformState {
    /// Not connected to the platform.
    #[default]
    Idle,
    /// Registered and waiting for a lease assignment.
    Available,
    /// Lease assigned, preparing to mine.
    Leased,
    /// Actively mining for a consumer.
    Mining,
    /// Lease completed, transitioning back.
    Completed,
    /// Error state; the watchdog will attempt recovery.
    Error,
}

impl PlatformState {
    /// The spelling that goes on the wire and into `/platform/status`.
    pub fn as_str(self) -> &'static str {
        match self {
            PlatformState::Idle => "IDLE",
            PlatformState::Available => "AVAILABLE",
            PlatformState::Leased => "LEASED",
            PlatformState::Mining => "MINING",
            PlatformState::Completed => "COMPLETED",
            PlatformState::Error => "ERROR",
        }
    }
}

/// Why a command was not acted on. Every variant is a log line; none of them carries
/// attacker-controlled text, so none can forge a log entry.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RejectReason {
    #[error("payload rejected: {0}")]
    Parse(#[from] proto::ParseError),
    #[error("envelope rejected: {0}")]
    Unauthenticated(VerifyStatus),
    #[error("unsigned mutating command refused — set {} to enable signed control", crate::secret::ENV_COMMAND_SECRET)]
    UnsignedMutating,
    #[error("assign_task rejected: invalid lease_id/consumer_id")]
    BadLeaseIdentifiers,
    #[error("assign_task rejected: invalid consumer_address")]
    BadConsumerAddress,
    #[error("rejected: invalid prefix")]
    BadPrefix,
    #[error("assign_task rejected: duration_sec out of range")]
    BadDuration,
    #[error("release rejected: invalid lease_id")]
    BadReleaseLeaseId,
    #[error("set_config rejected: difficulty out of range")]
    BadDifficulty,
    #[error("set_config rejected: invalid payout address")]
    BadPayoutAddress,
    #[error("set_config rejected: invalid block_pattern")]
    BadBlockPattern,
    /// Not an error: a well-formed command that does not apply in the current state.
    #[error("ignored in state {0}")]
    WrongState(&'static str),
}

/// A command as it sits in the queue.
#[derive(Debug, Clone, PartialEq, Eq)]
struct QueuedCommand {
    topic: String,
    payload: Vec<u8>,
}

#[derive(Debug, Default)]
struct CommandQueue {
    items: VecDeque<QueuedCommand>,
    dropped: u64,
}

/// Static worker facts and policy knobs.
#[derive(Debug, Clone)]
pub struct PlatformConfig {
    /// The worker id every envelope must be addressed to. Snapshot taken once, so the
    /// dispatch path never reads a mutable global.
    pub worker_id: String,
    /// The operator's own payout address, used for registration and for self-mining.
    pub eth_address: String,
    pub gpus: Vec<GpuInfo>,
    /// Shared secret for the HMAC envelope, from the environment (see [`crate::secret`]).
    ///
    /// `None` is the legacy deployment: only the commands that cannot move money keep
    /// working (`register_ack`, `release`, pause/resume); every mutating command —
    /// `assign_task` included — is refused. See [`envelope::is_mutating_command`].
    pub command_secret: Option<Secret>,
    pub queue_capacity: usize,
    pub nonce_cache_capacity: usize,
    pub heartbeat_interval: Duration,
    pub watchdog_interval: Duration,
}

impl PlatformConfig {
    pub fn new(worker_id: impl Into<String>, eth_address: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            eth_address: eth_address.into(),
            gpus: Vec::new(),
            command_secret: None,
            queue_capacity: COMMAND_QUEUE_CAPACITY,
            nonce_cache_capacity: NONCE_CACHE_CAPACITY,
            heartbeat_interval: Duration::from_secs(HEARTBEAT_INTERVAL_SEC),
            watchdog_interval: Duration::from_secs(WATCHDOG_INTERVAL_SEC),
        }
    }
}

/// Live statistics for the heartbeat, supplied by the binary so this crate touches no
/// global mining state.
pub type StatsFn = Arc<dyn Fn() -> WorkerStats + Send + Sync>;
/// Notified on every state transition (old, new).
pub type StateChangeFn = Arc<dyn Fn(PlatformState, PlatformState) + Send + Sync>;

pub struct PlatformManager<T: Transport> {
    config: PlatformConfig,
    reporter: WorkerReporter<T>,
    leases: LeaseManager,
    coordinator: Arc<MiningCoordinator>,
    clock: Arc<dyn Clock>,

    state: Mutex<PlatformState>,
    running: AtomicBool,

    queue: Mutex<CommandQueue>,
    queue_cv: Condvar,
    /// Guarded rather than thread-confined as in the C++: `dispatch_pending` is callable
    /// from any thread, and a mutex costs nothing at platform command rates.
    nonces: Mutex<NonceCache>,

    stats_fn: Mutex<Option<StatsFn>>,
    state_change_fn: Mutex<Option<StateChangeFn>>,
    /// Set when a signed `shutdown` arrives, so the caller can exit the process.
    shutdown_requested: AtomicBool,
    /// Latched by `stop`. Distinct from `running` because a manager that has not started
    /// yet must still accept commands into its queue — that is how a message arriving
    /// during startup is not lost — whereas one that is tearing down must not.
    stopping: AtomicBool,
    threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    dropped_total: AtomicU64,
}

impl<T: Transport + 'static> PlatformManager<T> {
    pub fn new(
        config: PlatformConfig,
        transport: T,
        coordinator: Arc<MiningCoordinator>,
        clock: Arc<dyn Clock>,
    ) -> Self {
        let nonces = NonceCache::new(config.nonce_cache_capacity);
        let reporter = WorkerReporter::new(transport, config.worker_id.clone(), Arc::clone(&clock));
        Self {
            leases: LeaseManager::new(Arc::clone(&clock)),
            reporter,
            coordinator,
            clock,
            state: Mutex::new(PlatformState::Idle),
            running: AtomicBool::new(false),
            queue: Mutex::new(CommandQueue::default()),
            queue_cv: Condvar::new(),
            nonces: Mutex::new(nonces),
            stats_fn: Mutex::new(None),
            state_change_fn: Mutex::new(None),
            shutdown_requested: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            threads: Mutex::new(Vec::new()),
            dropped_total: AtomicU64::new(0),
            config,
        }
    }

    pub fn set_stats_source(&self, f: StatsFn) {
        *self.stats_fn.lock() = Some(f);
    }

    pub fn set_state_change_callback(&self, f: StateChangeFn) {
        *self.state_change_fn.lock() = Some(f);
    }

    pub fn state(&self) -> PlatformState {
        *self.state.lock()
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// True once a signed remote `shutdown` has been obeyed.
    pub fn shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::SeqCst)
    }

    pub fn leases(&self) -> &LeaseManager {
        &self.leases
    }

    pub fn coordinator(&self) -> &Arc<MiningCoordinator> {
        &self.coordinator
    }

    pub fn worker_id(&self) -> &str {
        &self.config.worker_id
    }

    /// Total commands dropped by the intake gate (oversized or queue full).
    pub fn dropped_commands(&self) -> u64 {
        self.dropped_total.load(Ordering::Relaxed)
    }

    // --- Lifecycle ---

    /// Subscribe to the command topics, announce this worker, and go AVAILABLE.
    ///
    /// Separated from [`PlatformManager::start`] so that the state machine can be exercised
    /// without spawning threads — every handler behaves identically either way.
    pub fn announce(&self) {
        let transport = self.reporter.transport();
        let _ = transport.subscribe(&build_topic(&self.config.worker_id, proto::topic::TASK));
        let _ = transport.subscribe(&build_topic(&self.config.worker_id, proto::topic::CONTROL));

        let _ = self
            .reporter
            .send_registration(&self.config.eth_address, &self.config.gpus);

        // Optimistically AVAILABLE; `register_ack` confirms or rejects it, as in the C++.
        self.transition_to(PlatformState::Available);
    }

    /// Subscribe, register, and start the heartbeat / watchdog / dispatch threads.
    pub fn start(self: &Arc<Self>) -> bool {
        if self.running.swap(true, Ordering::SeqCst) {
            return true;
        }
        self.stopping.store(false, Ordering::SeqCst);

        if self.config.command_secret.is_none() {
            tracing::warn!(
                "SECURITY: {} is not set — MQTT commands are UNAUTHENTICATED. Lease \
                 release and pause/resume remain enabled; lease ASSIGNMENT, \
                 payout-address/difficulty/prefix/pattern changes and remote shutdown are \
                 DISABLED. This rig cannot take platform work until the secret is set.",
                crate::secret::ENV_COMMAND_SECRET
            );
        }

        self.announce();

        let mut threads = self.threads.lock();
        for (name, body) in [
            ("tm-plat-dispatch", DispatchKind::Dispatch),
            ("tm-plat-heartbeat", DispatchKind::Heartbeat),
            ("tm-plat-watchdog", DispatchKind::Watchdog),
        ] {
            let me = Arc::clone(self);
            if let Ok(handle) = std::thread::Builder::new()
                .name(name.into())
                .spawn(move || me.run_loop(body))
            {
                threads.push(handle);
            }
        }
        true
    }

    /// Idempotent, and safe to call from a dispatch thread (a signed remote `shutdown`
    /// does exactly that): a thread never tries to join itself.
    pub fn stop(&self) {
        self.stopping.store(true, Ordering::SeqCst);
        if !self.running.swap(false, Ordering::SeqCst) {
            return;
        }
        // Wake every loop so they observe `running == false`. Commands still queued are
        // deliberately dropped: obeying remote commands during teardown races it.
        self.queue_cv.notify_all();

        if self.leases.has_active_lease() {
            self.leases.end_lease();
            self.switch_to_self_mining();
        }

        let handles: Vec<_> = self.threads.lock().drain(..).collect();
        let me = std::thread::current().id();
        for handle in handles {
            if handle.thread().id() != me {
                let _ = handle.join();
            }
        }

        let _ = self.reporter.send_offline();
        self.transition_to(PlatformState::Idle);
    }

    // --- Command intake ---

    /// Enqueue a raw payload from the broker. Never blocks, never parses.
    ///
    /// The size gate is applied BEFORE the queue: an attacker with broker access must not
    /// be able to park megabytes per slot or feed the JSON parser unbounded input.
    pub fn enqueue_command(&self, topic: &str, payload: &[u8]) {
        let oversized = payload.len() > envelope::MAX_PAYLOAD_BYTES;

        let mut queue = self.queue.lock();
        if self.stopping.load(Ordering::SeqCst) {
            // Tearing down: nothing may touch the queue any more.
            return;
        }
        if oversized || queue.items.len() >= self.config.queue_capacity {
            queue.dropped += 1;
            let total = self.dropped_total.fetch_add(1, Ordering::Relaxed) + 1;
            tracing::warn!(
                total_dropped = total,
                reason = if oversized { "oversized" } else { "queue full" },
                "platform command dropped"
            );
            // Drop the NEWEST: already-accepted commands keep their FIFO order, so a flood
            // cannot displace a command the platform already handed us.
            return;
        }
        queue.items.push_back(QueuedCommand {
            topic: topic.to_string(),
            payload: payload.to_vec(),
        });
        self.queue_cv.notify_one();
    }

    /// Drain and handle everything queued right now. Returns how many were handled.
    ///
    /// The production dispatch thread and the tests both go through here, so there is one
    /// code path from bytes to behaviour.
    pub fn dispatch_pending(&self) -> usize {
        let mut handled = 0;
        loop {
            let Some(cmd) = self.queue.lock().items.pop_front() else {
                return handled;
            };
            self.handle_payload(&cmd.topic, &cmd.payload);
            handled += 1;
        }
    }

    /// Authenticate, parse and act on one payload. Every failure path logs and returns;
    /// nothing here can panic on hostile input, and nothing is applied partially.
    pub fn handle_payload(&self, topic: &str, payload: &[u8]) {
        if let Err(reason) = self.try_handle_payload(payload) {
            match reason {
                // A well-formed command that does not apply is normal traffic, not an
                // attack; keep it out of the warning stream.
                RejectReason::WrongState(_) => {
                    tracing::debug!(topic = %topic, %reason, "platform command ignored")
                }
                RejectReason::UnsignedMutating => {
                    tracing::error!(topic = %topic, %reason, "platform command REFUSED")
                }
                _ => tracing::warn!(topic = %topic, %reason, "platform command rejected"),
            }
        }
    }

    fn try_handle_payload(&self, payload: &[u8]) -> Result<(), RejectReason> {
        let msg = proto::value_from_payload(payload)?;
        self.authorize_command(&msg)?;
        match proto::command_from_value(&msg)? {
            Command::RegisterAck(ack) => self.handle_register_ack(&ack),
            Command::AssignTask(task) => self.handle_assign_task(&task),
            Command::Release(release) => self.handle_release(&release),
            Command::Control(action) => self.handle_control(&action),
        }
    }

    /// Envelope verification, plus the no-secret legacy policy.
    ///
    /// The topic is deliberately not consulted: authorisation is envelope-based, and the
    /// envelope's `worker_id` is what stops a command addressed to another rig from being
    /// obeyed here. A broker that mis-routes, or an attacker that publishes to our topic,
    /// gains nothing.
    fn authorize_command(&self, msg: &Value) -> Result<(), RejectReason> {
        match &self.config.command_secret {
            Some(secret) => {
                let status = envelope::verify_envelope(
                    msg,
                    secret.expose(),
                    &self.config.worker_id,
                    self.clock.now_epoch_s(),
                    &mut self.nonces.lock(),
                );
                if status != VerifyStatus::Ok {
                    return Err(RejectReason::Unauthenticated(status));
                }
                Ok(())
            }
            // Legacy deployment: keep working the commands that cannot move money, so live
            // operators are not broken, but refuse every command that could redirect
            // payouts, change mining parameters, or kill the miner — `assign_task` among
            // them, because its `consumer_address` is exactly such a redirection.
            None if envelope::is_mutating_command(msg) => Err(RejectReason::UnsignedMutating),
            None => Ok(()),
        }
    }

    // --- Handlers ---

    fn handle_register_ack(&self, ack: &RegisterAck) -> Result<(), RejectReason> {
        if ack.accepted {
            tracing::info!("registration accepted by platform");
            if self.state() != PlatformState::Available {
                self.transition_to(PlatformState::Available);
            }
            return Ok(());
        }
        // The reason string is attacker-influenced text headed for a log line, so it is
        // charset- and length-bounded before it gets there.
        let reason = if envelope::is_printable_ascii(&ack.reason, 1, 128) {
            ack.reason.as_str()
        } else if ack.reason.is_empty() {
            "unknown"
        } else {
            "<unparseable>"
        };
        tracing::error!(reason = %reason, "registration rejected by platform");
        self.transition_to(PlatformState::Error);
        Ok(())
    }

    fn handle_assign_task(&self, task: &AssignTask) -> Result<(), RejectReason> {
        if self.state() != PlatformState::Available {
            return Err(RejectReason::WrongState(self.state().as_str()));
        }

        // Strict field validation BEFORE any state change. Rejecting while staying
        // AVAILABLE (rather than the pre-hardening jump to ERROR) means a malformed
        // message cannot knock the rig out of service.
        if !envelope::is_safe_identifier(&task.lease_id, 1, MAX_LEASE_ID_LEN)
            || !envelope::is_safe_identifier(&task.consumer_id, 1, MAX_LEASE_ID_LEN)
        {
            return Err(RejectReason::BadLeaseIdentifiers);
        }
        // The consumer address becomes the Argon2 salt — the payout identity for the whole
        // lease — so it gets full EIP-55 validation, not a length check.
        if !tm_core::is_valid_ethereum_address(&task.consumer_address) {
            return Err(RejectReason::BadConsumerAddress);
        }
        // The prefix feeds hex key generation directly: exact platform length, hex only.
        if !task.prefix.is_empty()
            && !envelope::is_hex_string(&task.prefix, PLATFORM_PREFIX_LENGTH, PLATFORM_PREFIX_LENGTH)
        {
            return Err(RejectReason::BadPrefix);
        }
        // A minute to a week: outside that is either a typo or an attempt to pin the rig
        // to one consumer indefinitely.
        if !(MIN_LEASE_DURATION_SEC..=MAX_LEASE_DURATION_SEC).contains(&task.duration_sec) {
            return Err(RejectReason::BadDuration);
        }

        self.transition_to(PlatformState::Leased);

        if self
            .leases
            .start_lease(
                &task.lease_id,
                &task.consumer_id,
                &task.consumer_address,
                &task.prefix,
                task.duration_sec,
            )
            .is_err()
        {
            tracing::error!("failed to start lease: one is already active");
            self.transition_to(PlatformState::Error);
            return Ok(());
        }

        let ctx = self.leases.to_mining_context(&self.config.eth_address);
        self.switch_to_platform_mining(ctx);
        self.transition_to(PlatformState::Mining);
        Ok(())
    }

    fn handle_release(&self, release: &Release) -> Result<(), RejectReason> {
        if !release.lease_id.is_empty()
            && !envelope::is_safe_identifier(&release.lease_id, 1, MAX_LEASE_ID_LEN)
        {
            return Err(RejectReason::BadReleaseLeaseId);
        }

        let Some(current) = self.leases.lease() else {
            return Err(RejectReason::WrongState("no active lease"));
        };
        // An empty lease_id releases whatever is active; a mismatched one is for another
        // lease and must not end this one.
        if !release.lease_id.is_empty() && current.lease_id != release.lease_id {
            return Err(RejectReason::WrongState("lease_id mismatch"));
        }

        self.transition_to(PlatformState::Completed);
        self.leases.end_lease();
        self.switch_to_self_mining();
        self.transition_to(PlatformState::Available);
        Ok(())
    }

    fn handle_control(&self, action: &ControlAction) -> Result<(), RejectReason> {
        match action {
            ControlAction::Pause => {
                if self.leases.has_active_lease() {
                    self.leases.end_lease();
                    self.switch_to_self_mining();
                }
                self.transition_to(PlatformState::Idle);
                Ok(())
            }
            ControlAction::Resume => {
                if self.state() == PlatformState::Idle {
                    let _ = self
                        .reporter
                        .send_registration(&self.config.eth_address, &self.config.gpus);
                    self.transition_to(PlatformState::Available);
                }
                Ok(())
            }
            ControlAction::Shutdown => {
                // Only reachable with a valid signature: `is_mutating_command` classifies
                // shutdown as mutating, so the unsigned legacy path refuses it.
                tracing::warn!("signed remote shutdown accepted — stopping platform manager");
                self.shutdown_requested.store(true, Ordering::SeqCst);
                self.stop();
                Ok(())
            }
            ControlAction::SetConfig { config } => self.handle_set_config(config),
        }
    }

    /// Apply a remote configuration change.
    ///
    /// Only reachable through a valid HMAC envelope. Every present field is validated
    /// FIRST and the command is applied only if all of them pass — unlike the C++, which
    /// applies the good fields and ignores the bad ones. A half-applied `set_config` is a
    /// state nobody asked for, and the operator reading the rejection has no way to know
    /// which half landed.
    fn handle_set_config(&self, config: &SetConfig) -> Result<(), RejectReason> {
        if let Some(difficulty) = config.difficulty {
            if !(MIN_DIFFICULTY..=MAX_DIFFICULTY).contains(&difficulty) {
                return Err(RejectReason::BadDifficulty);
            }
        }
        if let Some(address) = &config.address {
            if !tm_core::is_valid_ethereum_address(address) {
                return Err(RejectReason::BadPayoutAddress);
            }
        }
        if let Some(prefix) = &config.prefix {
            // Empty clears it.
            if !prefix.is_empty() && !envelope::is_hex_string(prefix, 1, MAX_SELF_PREFIX_LEN) {
                return Err(RejectReason::BadPrefix);
            }
        }
        if let Some(pattern) = &config.block_pattern {
            // Empty resets to the default.
            if !pattern.is_empty()
                && !envelope::is_safe_identifier(pattern, 1, MAX_BLOCK_PATTERN_LEN)
            {
                return Err(RejectReason::BadBlockPattern);
            }
        }

        if let Some(difficulty) = config.difficulty {
            self.coordinator.set_difficulty(difficulty);
            tracing::info!(difficulty, "difficulty set by signed platform command");
        }
        if let Some(address) = &config.address {
            // The highest-risk command in the protocol: it redirects every future block
            // reward. Logged at error level so an operator scanning the console cannot
            // miss it.
            self.coordinator.set_user_address(address);
            tracing::error!(
                address = %address,
                "REMOTE PAYOUT ADDRESS CHANGE via signed platform command"
            );
        }
        if let Some(prefix) = &config.prefix {
            self.coordinator.set_self_mining_prefix(prefix);
            tracing::info!(prefix = %prefix, "self-mining prefix set by signed platform command");
        }
        if let Some(pattern) = &config.block_pattern {
            self.coordinator.set_test_block_pattern(pattern);
            tracing::info!(pattern = %pattern, "block pattern set by signed platform command");
        }

        // Immediate heartbeat so the server and dashboard see the change without waiting
        // out the interval.
        self.publish_heartbeat();
        Ok(())
    }

    // --- Reporting ---

    /// Called by the submitter when a find lands. Reports it against the active lease when
    /// mining for a consumer, and bare when self-mining.
    pub fn on_block_found(
        &self,
        hash: &str,
        key: &str,
        account: &str,
        attempts: u64,
        hashrate: f32,
    ) {
        match self.state() {
            PlatformState::Mining => {
                let Some(lease) = self.leases.lease() else {
                    return;
                };
                self.leases.record_block();
                let _ = self.reporter.send_block_found(
                    &lease.lease_id,
                    hash,
                    key,
                    account,
                    attempts,
                    hashrate,
                );
            }
            PlatformState::Available => {
                let _ = self
                    .reporter
                    .send_block_found("", hash, key, account, attempts, hashrate);
            }
            _ => {}
        }
    }

    pub fn publish_heartbeat(&self) {
        let stats = self.stats_fn.lock().clone();
        let mut stats = match stats {
            Some(f) => f(),
            None => WorkerStats::default(),
        };
        // The coordinator is the authority on these, whatever the caller filled in.
        let identity = self.coordinator.identity();
        stats.difficulty = self.coordinator.difficulty();
        stats.address = identity.user_address;
        stats.prefix = identity.self_mining_prefix;
        stats.block_pattern = identity.test_block_pattern;
        let _ = self.reporter.send_heartbeat(&stats);
    }

    fn transition_to(&self, new_state: PlatformState) {
        let old_state = {
            let mut state = self.state.lock();
            let old = *state;
            if old == new_state {
                return;
            }
            *state = new_state;
            old
        };
        tracing::info!(from = old_state.as_str(), to = new_state.as_str(), "platform state");

        if self.reporter.transport().is_connected() {
            let lease_id = self.leases.lease().map(|l| l.lease_id).unwrap_or_default();
            let _ = self
                .reporter
                .send_status_update(new_state.as_str(), &lease_id, "");
        }

        let cb = self.state_change_fn.lock().clone();
        if let Some(cb) = cb {
            cb(old_state, new_state);
        }
    }

    fn switch_to_self_mining(&self) {
        self.coordinator.update_context(MiningContext {
            mode: MiningMode::SelfMining,
            address: self.coordinator.identity().user_address,
            ..MiningContext::default()
        });
    }

    fn switch_to_platform_mining(&self, ctx: MiningContext) {
        tracing::info!(address = %ctx.address, "switched to platform mining");
        self.coordinator.update_context(ctx);
    }

    /// One watchdog tick: expire a finished lease, and try to climb out of ERROR.
    /// Public so a test can step it without waiting for the interval.
    pub fn watchdog_tick(&self) {
        if self.state() == PlatformState::Mining && self.leases.is_expired() {
            tracing::info!("lease expired");
            self.transition_to(PlatformState::Completed);
            self.leases.end_lease();
            self.switch_to_self_mining();
            self.transition_to(PlatformState::Available);
        }

        if self.state() == PlatformState::Error {
            self.transition_to(PlatformState::Idle);
            if self.reporter.transport().is_connected() {
                let _ = self
                    .reporter
                    .send_registration(&self.config.eth_address, &self.config.gpus);
                self.transition_to(PlatformState::Available);
            }
        }
    }

    // --- Threads ---

    fn run_loop(&self, kind: DispatchKind) {
        match kind {
            DispatchKind::Dispatch => self.dispatch_loop(),
            DispatchKind::Heartbeat => self.interval_loop(self.config.heartbeat_interval, || {
                self.publish_heartbeat()
            }),
            DispatchKind::Watchdog => {
                self.interval_loop(self.config.watchdog_interval, || self.watchdog_tick())
            }
        }
    }

    fn dispatch_loop(&self) {
        // Single consumer; commands run strictly in FIFO arrival order.
        loop {
            let cmd = {
                let mut queue = self.queue.lock();
                while queue.items.is_empty() {
                    if !self.running.load(Ordering::SeqCst) {
                        return;
                    }
                    self.queue_cv
                        .wait_for(&mut queue, Duration::from_millis(250));
                }
                if !self.running.load(Ordering::SeqCst) {
                    return;
                }
                queue.items.pop_front()
            };
            if let Some(cmd) = cmd {
                self.handle_payload(&cmd.topic, &cmd.payload);
            }
        }
    }

    /// Sleep in slices so shutdown is prompt, as the C++ loops do.
    fn interval_loop(&self, interval: Duration, mut body: impl FnMut()) {
        let slice = Duration::from_millis(200).min(interval);
        let mut waited = Duration::ZERO;
        while self.running.load(Ordering::SeqCst) {
            std::thread::sleep(slice);
            waited += slice;
            if waited < interval {
                continue;
            }
            waited = Duration::ZERO;
            if !self.running.load(Ordering::SeqCst) {
                return;
            }
            body();
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DispatchKind {
    Dispatch,
    Heartbeat,
    Watchdog,
}
