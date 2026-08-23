//! TreeMiner command-line front end: argument parsing, configuration resolution, machine
//! identity, the startup clock, and the network-difficulty poller.
//!
//! Rust port of `src/main.cpp` (option handling), `src/ConfigManager.*`, `src/AppConfig.*`,
//! `src/DifficultyManager.*` and `src/MachineIDGetter.*`.
//!
//! The mining loop (`mineunit`), the GPU self-test (`selftest`), the journal-first find
//! capture (`find`), the CPU sidecar (`cpuworker`), the stats publishers (`stats`) and the
//! startup/shutdown wiring (`run`) sit on top of that; `state` is the live mining state they
//! all share.

pub mod appconfig;
pub mod backend;
pub mod bridge;
pub mod cli;
pub mod clock;
pub mod config;
pub mod cpuworker;
pub mod difficulty;
pub mod find;
pub mod hashcli;
pub mod machineid;
pub mod mineunit;
pub mod platform;
pub mod prompt;
pub mod resolve;
pub mod run;
pub mod selftest;
pub mod state;
pub mod stats;
pub mod testsupport;

pub use appconfig::{AppConfig, AppConfigError};
pub use cli::Cli;
pub use clock::{clock_hms, init_local_offset, local_offset, log_timestamp, now_local};
pub use config::{ConfigManager, CONFIG_FILENAME};
pub use hashcli::{is_hash_api_command, run as run_hash_cli};
pub use difficulty::{
    load_cached_difficulty, persist_difficulty, seed_initial_difficulty, DifficultyHandle,
    DifficultyPoller, DifficultyShared, PollOutcome, DIFFICULTY_CACHE_FILE, FALLBACK_DIFFICULTY,
    POLL_INTERVAL, POOL_DOWN_FAILURE_THRESHOLD,
};
pub use machineid::{
    derive_machine_id, device_info_text, machine_id_for_devices, machine_identity,
    parse_device_list, MachineFacts,
};
pub use prompt::{NoPrompter, Prompter, ScriptedPrompter, StdioPrompter};
pub use resolve::{resolve, ResolveError, ResolveOptions, ResolvedConfig};
pub use backend::{DeviceFacts, GpuMiningBackend, GpuSelfTestProbe, MiningBackend};
pub use bridge::JournalBridge;
pub use cpuworker::{CpuMiningWorker, CpuStats, CpuWorkerConfig, CpuWorkerError};
pub use find::{Capture, Find, FindObserver, FindSink};
pub use mineunit::{
    run_mining_on_device, select_work, xuni_window_open_now, IdentitySource, LoopExit, MineDeps,
    MineUnit, MiningIdentity, Work,
};
pub use platform::{
    gpu_infos, start_if_enabled, start_mqtt, MqttRuntime, PlatformError, PlatformOptions,
    PlatformRuntime,
};
pub use run::run;
pub use selftest::{run_self_test, DeviceDecision, SelfTestProbe, SelfTestReport};
pub use state::{classify_find, effective_difficulty, FindClass, MiningState};
pub use stats::{BreakerStateLabel, StatsIdentity, StatsPublisher, SubmissionView};
