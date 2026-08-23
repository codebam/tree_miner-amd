//! Platform mode: everything that connects `tm-platform` to a running miner.
//!
//! Port of the `globalPlatformMode` block in `src/main.cpp` plus the reads of
//! `MiningCoordinator::getInstance()` scattered through `MineUnit.cpp`, `main.cpp` and
//! `LocalServer.cpp`. The crate already owns the protocol, the envelope and the state
//! machine; this module owns only the seams into the miner:
//!
//! - **identity** — [`PlatformRuntime::identity_source`] is read at every batch boundary by
//!   both the GPU loop and the CPU sidecar, so a lease redirects the salt without a
//!   restart, and a signed `set_config` moves the payout address the same way.
//! - **finds** — [`PlatformRuntime::find_observer`] reports a block to the broker, strictly
//!   after it is durably captured.
//! - **stats** — the heartbeat reads the live [`MiningState`].
//! - **pause / resume** — mapped onto [`MiningState::set_mining_paused`].
//! - **shutdown** — a signed remote `shutdown` flips the same [`Shutdown`] flag SIGINT
//!   does, so teardown takes the one orderly path in `run`.
//!
//! Without `--platform-mode` none of this exists: no runtime is built, no environment
//! variable is read, no socket is opened, and every seam above falls back to the value the
//! configuration resolved at startup.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tm_dashboard::stats::{PlatformLease, PlatformStatus};
use tm_platform::clock::{Clock, SystemClock};
use tm_platform::coordinator::{MiningCoordinator, MiningIdentity as RemoteIdentity, MiningMode};
use tm_platform::manager::{PlatformConfig, PlatformManager, StatsFn};
use tm_platform::proto::GpuInfo;
use tm_platform::reporter::WorkerStats;
use tm_platform::secret::{CredentialError, PlatformCredentials};
use tm_platform::transport::{BrokerConfig, ConnectError, MqttTransport, Transport};
use tm_tui::{Console, Level};

use crate::find::{Find, FindObserver};
use crate::mineunit::{IdentitySource, MiningIdentity};
use crate::state::MiningState;
use crate::stats::PlatformProvider;

/// How often the supervisor thread samples the manager. Short enough that a remote
/// `shutdown` is not visibly slower than a Ctrl-C, cheap enough to ignore.
const SUPERVISOR_TICK: Duration = Duration::from_millis(200);

/// Why platform mode could not start. Every variant is printed verbatim to the operator,
/// and every one of them names what to fix.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("--platform-mode: {0}")]
    Credentials(#[from] CredentialError),
    #[error("--platform-mode: cannot open the broker connection to {uri} | {source}")]
    Broker {
        /// Already redacted: a broker URI may carry `user:pass@`.
        uri: String,
        source: ConnectError,
    },
}

/// The MQTT-backed runtime the binary builds. Named so `run` has one concrete type.
pub type MqttRuntime = PlatformRuntime<Arc<MqttTransport>>;

/// Platform mode, attached to a running miner.
///
/// Generic over [`Transport`] so the whole wiring — identity redirection, pause, remote
/// shutdown, the dashboard payload — is testable without a broker.
pub struct PlatformRuntime<T: Transport + 'static> {
    manager: Arc<PlatformManager<T>>,
    coordinator: Arc<MiningCoordinator>,
    /// The identity the configuration resolved: devfee addresses, permillage, and the
    /// operator's own address. Remote commands never change these fields, only the ones
    /// [`RemoteIdentity`] carries.
    base: MiningIdentity,
    supervisor: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl<T: Transport + 'static> std::fmt::Debug for PlatformRuntime<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlatformRuntime")
            .field("worker_id", &self.manager.worker_id())
            .field("state", &self.manager.state())
            .finish_non_exhaustive()
    }
}

impl<T: Transport + 'static> PlatformRuntime<T> {
    /// Build a runtime around an already-open transport.
    pub fn new(
        config: PlatformConfig,
        transport: T,
        base: MiningIdentity,
        difficulty: u32,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        let coordinator = Arc::new(MiningCoordinator::new(
            RemoteIdentity {
                user_address: base.user_address.clone(),
                self_mining_prefix: base.self_mining_prefix.clone(),
                test_block_pattern: base.test_block_pattern.clone().unwrap_or_default(),
            },
            i64::from(difficulty),
        ));
        let manager = Arc::new(PlatformManager::new(
            config,
            transport,
            Arc::clone(&coordinator),
            clock,
        ));
        Arc::new(Self {
            manager,
            coordinator,
            base,
            supervisor: Mutex::new(None),
        })
    }

    pub fn manager(&self) -> &Arc<PlatformManager<T>> {
        &self.manager
    }

    pub fn coordinator(&self) -> &Arc<MiningCoordinator> {
        &self.coordinator
    }

    /// The identity the next batch must mine for.
    ///
    /// Under a lease that is the consumer's address and the platform key prefix, with the
    /// devfee rotation suppressed — the consumer is paying for the whole machine, and the
    /// C++ takes no fee from leased work either. Otherwise it is the resolved identity with
    /// whatever a signed `set_config` has changed.
    pub fn current_identity(&self) -> MiningIdentity {
        let context = self.coordinator.context();
        let remote = self.coordinator.identity();
        let pattern = if remote.test_block_pattern.is_empty() {
            self.base.test_block_pattern.clone()
        } else {
            Some(remote.test_block_pattern.clone())
        };
        match context.mode {
            MiningMode::PlatformMining => MiningIdentity {
                user_address: context.address,
                devfee_address: String::new(),
                eco_devfee_address: String::new(),
                devfee_permillage: 0,
                self_mining_prefix: context.prefix,
                test_block_pattern: pattern,
            },
            MiningMode::SelfMining => MiningIdentity {
                user_address: remote.user_address,
                self_mining_prefix: remote.self_mining_prefix,
                test_block_pattern: pattern,
                ..self.base.clone()
            },
        }
    }

    /// The live identity the mining loops read. Installed on `MineDeps` and the CPU worker.
    pub fn identity_source(self: &Arc<Self>) -> IdentitySource {
        let me = Arc::clone(self);
        Arc::new(move || me.current_identity())
    }

    /// Report durable finds to the platform. Installed on the find sink.
    pub fn find_observer(self: &Arc<Self>) -> FindObserver {
        let manager = Arc::clone(&self.manager);
        Arc::new(move |find: &Find, payload: &tm_core::FoundPayload| {
            if !manager.is_running() {
                return;
            }
            manager.on_block_found(
                &payload.hash_to_verify,
                &find.key,
                &payload.account,
                find.attempts,
                find.hashes_per_second as f32,
            );
        })
    }

    /// What `/platform/status` serves while platform mode is on.
    pub fn status(&self) -> PlatformStatus {
        let leases = self.manager.leases();
        PlatformStatus {
            platform_mode: true,
            mining_mode: self.coordinator.mode().as_str().to_owned(),
            platform_state: self.manager.state().as_str().to_owned(),
            running: self.manager.is_running(),
            lease: leases.lease().map(|lease| PlatformLease {
                lease_id: lease.lease_id,
                consumer_id: lease.consumer_id,
                consumer_address: lease.consumer_address,
                blocks_found: lease.blocks_found,
                remaining_sec: leases.remaining_seconds(),
            }),
        }
    }

    /// The dashboard provider. Installed on the stats publisher.
    pub fn status_provider(self: &Arc<Self>) -> PlatformProvider {
        let me = Arc::clone(self);
        Arc::new(move || Some(me.status()))
    }

    /// The heartbeat's view of the miner. The manager overwrites the identity fields from
    /// the coordinator, so only the measurements are filled in here.
    fn stats_source(state: &Arc<MiningState>) -> StatsFn {
        let state = Arc::clone(state);
        Arc::new(move || {
            let gpus = state.gpu_stats();
            WorkerStats {
                total_hashrate: gpus.iter().map(|gpu| gpu.hashrate).sum::<f32>()
                    + state.cpu_hashrate() as f32,
                active_gpus: gpus.len() as i64,
                accepted_blocks: (state.normal_blocks() + state.super_blocks() + state.xuni_blocks())
                    as i64,
                uptime_sec: state.uptime_seconds(),
                ..WorkerStats::default()
            }
        })
    }

    /// Connect the runtime to the miner's live state and start its threads.
    ///
    /// After this returns, a signed command changes what the rig mines; before it, the
    /// runtime is inert.
    pub fn start(self: &Arc<Self>, state: &Arc<MiningState>) {
        self.manager.set_stats_source(Self::stats_source(state));

        // `pause` is the only way to reach IDLE from a serving state while the manager is
        // running: the watchdog's own ERROR -> IDLE recovery is exempted below, and a
        // broker outage does not move the state machine at all. Deriving the pause flag
        // from the transition rather than from the state means a dead broker leaves the rig
        // self-mining instead of silently idling it.
        let paused_state = Arc::clone(state);
        // Weak, because the manager owns this callback: an Arc here would be a cycle, and
        // the manager would outlive the process it is supposed to shut down.
        let manager = Arc::downgrade(&self.manager);
        self.manager
            .set_state_change_callback(Arc::new(move |old, new| {
                use tm_platform::PlatformState::*;
                // `stop` transitions to IDLE on its way out, after clearing `running`.
                // That is teardown, not a pause, and the distinction matters: `finish`
                // joins the producer threads, and a producer that thinks it is paused is
                // a producer that never reaches its exit check.
                if !manager.upgrade().is_some_and(|m| m.is_running()) {
                    return;
                }
                let paused = match (old, new) {
                    (Error, Idle) => return,
                    (_, Idle) => true,
                    (_, Available) => false,
                    _ => return,
                };
                if paused_state.set_mining_paused(paused) != paused {
                    Console::global().event(
                        Level::Warn,
                        "PLATFORM",
                        if paused {
                            "mining PAUSED by platform command"
                        } else {
                            "mining resumed by platform command"
                        },
                    );
                }
            }));

        self.manager.start();

        let me = Arc::clone(self);
        let state = Arc::clone(state);
        let handle = std::thread::Builder::new()
            .name("treeminer-platform".into())
            .spawn(move || me.supervise(&state))
            .ok();
        *self.supervisor.lock() = handle;
    }

    /// The one thread this module owns. It carries a remote `shutdown` into the miner's own
    /// shutdown path, and keeps the coordinator's difficulty and the miner's in step.
    fn supervise(&self, state: &Arc<MiningState>) {
        let mut last_remote_difficulty = self.coordinator.difficulty();
        while state.is_running() {
            if self.manager.shutdown_requested() {
                Console::global().event(
                    Level::Warn,
                    "PLATFORM",
                    "signed remote shutdown — stopping the miner",
                );
                // Exactly what the signal handler does. Everything else is torn down on the
                // main thread afterwards, in order.
                state.shutdown().request_stop();
                return;
            }

            // A signed `set_config` wins for one tick and then the network poller takes
            // over again, which is the C++ behaviour (it writes `globalDifficulty` and lets
            // the poller correct it).
            let remote = self.coordinator.difficulty();
            if remote != last_remote_difficulty {
                if let Ok(difficulty) = u32::try_from(remote) {
                    state.set_difficulty(difficulty);
                }
                last_remote_difficulty = remote;
            } else {
                let live = i64::from(state.difficulty());
                if live != remote {
                    self.coordinator.set_difficulty(live);
                    last_remote_difficulty = live;
                }
            }

            std::thread::sleep(SUPERVISOR_TICK);
        }
    }

    /// Stop the manager, join every thread it owns, and release any lease.
    ///
    /// Idempotent. Called from the miner's teardown, after the mining threads have joined,
    /// so a find in flight is still reported.
    pub fn stop(&self) {
        self.manager.stop();
        let supervisor = self.supervisor.lock().take();
        if let Some(handle) = supervisor {
            // The supervisor only exits when the shutdown flag is set; `run` sets it before
            // reaching here, and a remote shutdown set it itself.
            let _ = handle.join();
        }
    }
}

/// What platform mode needs from the resolved configuration.
#[derive(Debug, Clone, Copy)]
pub struct PlatformOptions<'a> {
    /// `--platform-mode`. When false NOTHING below happens.
    pub enabled: bool,
    /// `--mqtt-broker`; `resolve` has already refused an empty one with the flag set.
    pub broker_uri: &'a str,
    /// The machine id, or `--worker-id` if the operator overrode it.
    pub worker_id: &'a str,
    /// The operator's own payout address, for registration and for self-mining.
    pub eth_address: &'a str,
}

/// Build and start platform mode, or nothing at all.
///
/// `Ok(None)` is the ordinary case, and it is reached before anything is read or opened:
/// without `--platform-mode` there is no environment lookup, no device query, and no
/// socket. That ordering is the test `platform_mode_off_reads_no_credentials` pins.
pub fn start_if_enabled(
    options: &PlatformOptions<'_>,
    devices: &[i32],
    device_names: &std::collections::BTreeMap<i32, String>,
    base: &MiningIdentity,
    state: &Arc<MiningState>,
) -> Result<Option<Arc<MqttRuntime>>, PlatformError> {
    if !options.enabled {
        return Ok(None);
    }
    let runtime = start_mqtt(
        options.broker_uri,
        options.worker_id,
        options.eth_address,
        gpu_infos(devices, device_names),
        base.clone(),
        state.difficulty(),
    )?;
    runtime.start(state);
    Ok(Some(runtime))
}

/// Build the MQTT runtime for `--platform-mode`, or explain why it cannot exist.
///
/// Credentials come from the environment and nowhere else (see [`tm_platform::secret`]); a
/// missing one is a startup failure naming the variable, because starting platform mode
/// without a command secret means the rig can be reached by anyone who can publish to the
/// broker and can take no work at all.
pub fn start_mqtt(
    broker_uri: &str,
    worker_id: &str,
    eth_address: &str,
    gpus: Vec<GpuInfo>,
    base: MiningIdentity,
    difficulty: u32,
) -> Result<Arc<MqttRuntime>, PlatformError> {
    let credentials = PlatformCredentials::from_env()?;

    let mut broker = BrokerConfig::new(broker_uri, worker_id);
    broker.credentials = credentials.broker_auth.clone();
    let safe_uri = broker.safe_uri();

    // The transport hands inbound payloads to the manager's bounded queue, and the manager
    // does not exist yet — hence the slot. It is filled before `start`, and an early message
    // that finds it empty is simply dropped, exactly as one arriving before we subscribed
    // would be.
    let slot: Arc<std::sync::OnceLock<Arc<PlatformManager<Arc<MqttTransport>>>>> =
        Arc::new(std::sync::OnceLock::new());
    let sink = Arc::clone(&slot);
    let transport = MqttTransport::start(
        broker,
        Arc::new(move |topic: &str, payload: &[u8]| {
            if let Some(manager) = sink.get() {
                manager.enqueue_command(topic, payload);
            }
        }),
    )
    .map_err(|source| PlatformError::Broker {
        uri: safe_uri.clone(),
        source,
    })?;

    let mut config = PlatformConfig::new(worker_id, eth_address);
    config.gpus = gpus;
    config.command_secret = Some(credentials.command_secret);

    let runtime = PlatformRuntime::new(
        config,
        transport,
        base,
        difficulty,
        Arc::new(SystemClock::new()),
    );
    let _unused = slot.set(Arc::clone(runtime.manager()));
    Console::global().event(
        Level::Info,
        "PLATFORM",
        &format!("platform mode enabled | broker={safe_uri} | worker={worker_id}"),
    );
    Ok(runtime)
}

/// The GPU descriptors sent with the registration.
pub fn gpu_infos(devices: &[i32], names: &std::collections::BTreeMap<i32, String>) -> Vec<GpuInfo> {
    devices
        .iter()
        .map(|&index| {
            let (name, memory_gb, bus_id) = match tm_gpu::Device::open(index) {
                Ok(device) => (
                    device.name().to_owned(),
                    (device.total_memory_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)).round()
                        as i64,
                    device.bus_id(),
                ),
                // Registration is telemetry: a device that will not open is reported by
                // name rather than dropped from the fleet view.
                Err(_) => (
                    names.get(&index).cloned().unwrap_or_else(|| "unknown".into()),
                    0,
                    -1,
                ),
            };
            GpuInfo {
                index,
                name,
                memory_gb,
                bus_id,
            }
        })
        .collect()
}
