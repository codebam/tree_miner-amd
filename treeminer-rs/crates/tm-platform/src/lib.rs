//! Platform mode: the hashpower marketplace / fleet telemetry side of TreeMiner.
//!
//! A worker in platform mode connects to an MQTT broker, registers its GPUs, heartbeats
//! its statistics, and accepts leases that redirect its mining at a consumer's address for
//! a bounded time. Port of the C++ `PlatformManager`, `MqttClient`, `LeaseManager`,
//! `WorkerReporter`, `MiningCoordinator` and `platform/CommandEnvelope`.
//!
//! # The threat model, in one paragraph
//!
//! The broker is a shared rendezvous point that nobody in this system fully controls.
//! Anyone able to publish to `xenminer/{worker_id}/task` or `.../control` can otherwise
//! redirect the miner's payout address, change its Argon2 difficulty, or shut it down. So:
//! every command carries an HMAC-SHA256 envelope ([`envelope`]) bound to this worker's id,
//! with a bounded lifetime and a single-use nonce; the signature is checked before the
//! command is even interpreted; and when no secret is configured, the miner still refuses
//! every command that could move money — including `assign_task`, whose `consumer_address`
//! redirects every block found for the length of the lease. Nothing parsed from the network
//! can panic, and nothing is applied partially.
//!
//! # Secrets
//!
//! Broker credentials and the command secret come from **environment variables only** —
//! never a CLI flag (`/proc/<pid>/cmdline` is world-readable) and never the config file.
//! See [`secret`] for the variable names. Values live in [`secret::Secret`], whose `Debug`
//! prints a redaction, so a stray `{:?}` cannot leak one.
//!
//! # Wiring it up
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use tm_platform::*;
//! let creds = secret::PlatformCredentials::from_env()?;                 // fails loudly if unset
//! let mut broker = transport::BrokerConfig::new("ssl://broker:8883", "rig-01");
//! broker.credentials = creds.broker_auth.clone();
//!
//! let coordinator = Arc::new(coordinator::MiningCoordinator::new(
//!     coordinator::MiningIdentity { user_address: "0x…".into(), ..Default::default() },
//!     8,
//! ));
//! let clock: Arc<dyn clock::Clock> = Arc::new(clock::SystemClock::new());
//!
//! // The transport hands raw payloads straight to the manager's bounded queue.
//! let slot: Arc<std::sync::OnceLock<Arc<manager::PlatformManager<Arc<transport::MqttTransport>>>>> =
//!     Arc::new(std::sync::OnceLock::new());
//! let sink = Arc::clone(&slot);
//! let transport = transport::MqttTransport::start(
//!     broker,
//!     Arc::new(move |topic: &str, payload: &[u8]| {
//!         if let Some(m) = sink.get() { m.enqueue_command(topic, payload) }
//!     }),
//! )?;
//!
//! let mut config = manager::PlatformConfig::new("rig-01", "0x…");
//! config.command_secret = Some(creds.command_secret);
//! let manager = Arc::new(manager::PlatformManager::new(
//!     config, transport, coordinator, clock,
//! ));
//! let _ = slot.set(Arc::clone(&manager));
//! manager.start();
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod backoff;
pub mod clock;
pub mod coordinator;
pub mod envelope;
pub mod lease;
pub mod manager;
pub mod proto;
pub mod reporter;
pub mod secret;
pub mod transport;

pub use clock::{Clock, SystemClock, TestClock};
pub use coordinator::{MiningContext, MiningCoordinator, MiningIdentity, MiningMode};
pub use envelope::{verify_envelope, NonceCache, VerifyStatus};
pub use lease::{LeaseInfo, LeaseManager};
pub use manager::{PlatformConfig, PlatformManager, PlatformState, RejectReason};
pub use proto::{Command, ControlAction, GpuInfo};
pub use reporter::{WorkerReporter, WorkerStats};
pub use secret::{PlatformCredentials, Secret};
pub use transport::{BrokerConfig, MqttTransport, Transport, TransportError};
