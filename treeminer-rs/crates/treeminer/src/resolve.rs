//! Turning the command line plus `config.txt` into the one settled configuration the rest
//! of the miner reads. Port of the option-handling half of `src/main.cpp` (everything
//! before device enumeration).
//!
//! PRECEDENCE, EXACTLY AS THE C++ HAD IT
//! explicit CLI flag > `config.txt` value > compiled-in default. Only a handful of keys are
//! readable from the file at all (the margin trio, the journal path, the dashboard bind and
//! port, plus the identity keys `AppConfig` owns); everything else is CLI-or-default.
//!
//! A bad value is fatal here rather than ignored. Silently mining at the wrong memory cost,
//! or journalling to the wrong file, shows up hours later as unexplained 401s or as finds
//! stranded in an orphaned database — both far more expensive than refusing to start.

use std::path::{Path, PathBuf};

use tm_dashboard::url::is_valid_dashboard_bind;
use tm_submit::{MarginConfig, MarginMode};
use tm_tui::{resolve_display_mode, resolve_prompt_selection, DisplayEnv, DisplayMode, PROMPT_TEXT};

use crate::appconfig::{AppConfig, AppConfigError};
use crate::cli::Cli;
use crate::config::{stoi, ConfigManager, CONFIG_FILENAME};
use crate::difficulty::{seed_initial_difficulty, DIFFICULTY_CACHE_FILE};
use crate::prompt::Prompter;

/// `MiningCommon.cpp` defaults.
pub const DEFAULT_RPC_LINK: &str = "http://xenblocks.io";
pub const DEFAULT_DASHBOARD_BIND: &str = "0.0.0.0";
pub const DEFAULT_DASHBOARD_PORT: u16 = 42069;
/// CWD-relative for drop-in compatibility with existing deployments; the resolved ABSOLUTE
/// path is what gets logged at startup, because a miner launched from an unexpected working
/// directory would otherwise open a fresh empty journal and strand every queued find.
pub const DEFAULT_JOURNAL_PATH: &str = "treeminer-journal.db";
/// CPU hashing only pays near the difficulty floor.
pub const DEFAULT_CPU_MAX_DIFFICULTY: u32 = 100;
const MAX_CONFIGURED_KIB: i64 = 100_000_000;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ResolveError {
    #[error("Invalid dashboard port '{0}'.")]
    DashboardPort(i32),
    #[error("The display mode must be logs, terminal, or prompt.")]
    DisplayMode,
    #[error("Invalid difficulty margin mode '{0}' (expected: off | fixed | auto).")]
    MarginMode(String),
    #[error("Value for {key} ({text}) is out of range (0-100000000).")]
    ValueOutOfRange { key: String, text: String },
    #[error("Value for {key} ({text}) is not a number.")]
    ValueNotANumber { key: String, text: String },
    #[error("Invalid dashboard bind address '{0}': expected an IPv4 or IPv6 address.")]
    DashboardBind(String),
    #[error("The argument ({0}) for GPU streams must be 1 or 2.")]
    GpuStreams(i32),
    #[error("The argument ({requested}) for CPU workers must be between 0 and {ceiling}.")]
    CpuWorkers { requested: i32, ceiling: u32 },
    #[error("The argument ({0}) for the CPU difficulty ceiling must be 0-100000000.")]
    CpuMaxDifficulty(i32),
    #[error("The argument ({0}) for total developer fee (0-1000) is invalid.")]
    TotalDevFee(i32),
    #[error("The argument ({0}) for ecosystem developer fee address (EIP55) is invalid.")]
    EcoDevAddr(String),
    #[error("The argument ({0}) for miner address (EIP55) is invalid.")]
    MinerAddr(String),
    #[error("Platform mode requires --mqtt-broker to be set.")]
    PlatformModeWithoutBroker,
    #[error("{0}")]
    Config(String),
}

impl From<AppConfigError> for ResolveError {
    fn from(error: AppConfigError) -> Self {
        ResolveError::Config(error.to_string())
    }
}

/// Ambient facts the resolution depends on. Injected so the whole thing is testable without
/// a tty, a `/proc`, or the real working directory.
#[derive(Debug, Clone)]
pub struct ResolveOptions {
    pub config_path: PathBuf,
    pub difficulty_cache_path: PathBuf,
    pub display_env: DisplayEnv,
    /// `std::thread::hardware_concurrency()`; 0 means "unknown", which disables the upper
    /// bound on `--cpuWorkers` exactly as the C++ did.
    pub logical_threads: u32,
}

impl Default for ResolveOptions {
    fn default() -> Self {
        Self {
            config_path: PathBuf::from(CONFIG_FILENAME),
            difficulty_cache_path: PathBuf::from(DIFFICULTY_CACHE_FILE),
            display_env: DisplayEnv::default(),
            logical_threads: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(0),
        }
    }
}

impl ResolveOptions {
    /// Real process environment and real working directory.
    pub fn from_process() -> Self {
        Self { display_env: DisplayEnv::from_process(), ..Self::default() }
    }
}

/// The settled configuration. This is the type the integrator threads through the mining
/// loop; nothing downstream should re-read `config.txt` or the command line.
#[derive(Debug, Clone)]
pub struct ResolvedConfig {
    pub execute: bool,
    pub donotupload: bool,
    pub save_config: bool,

    pub miner_address: String,
    pub eco_devfee_address: String,
    pub devfee_permillage: i32,

    pub device_list: String,
    /// Independent device work queues per device (`--gpuStreams`). Keeps the C++ field
    /// name because the mining loop reads it under that name.
    pub gpu_streams_per_device: usize,
    pub cpu_worker_count: usize,
    pub cpu_max_difficulty: u32,
    /// 0 = auto (use all free GPU memory).
    pub max_batch_size: usize,

    pub display_mode: DisplayMode,
    /// Written to stderr by the binary when the requested mode was downgraded.
    pub display_warning: Option<String>,

    pub dashboard_bind: String,
    pub dashboard_port: u16,

    pub rpc_link: String,
    pub custom_name: String,
    pub platform_mode: bool,
    pub mqtt_broker: String,
    pub worker_id: String,

    pub journal_path: PathBuf,
    pub difficulty_cache_path: PathBuf,

    pub margin: MarginConfig,
    /// Initial `globalDifficultyMargin`: non-zero only in fixed mode, which pays for its
    /// headroom on every single hash forever.
    pub difficulty_margin: u32,

    /// `--testFixedDiff`: mine at a constant difficulty with a null identity, no polling.
    pub test_fixed_diff: Option<u32>,
    pub test_block_pattern: Option<String>,
    /// Difficulty to start at: the fixed test value, the cached value, or the 42069 fallback.
    pub initial_difficulty: u32,
    /// `DIFFICULTY seeded from cache | current=N`, when the cache supplied the start value.
    pub difficulty_seed_note: Option<String>,

    /// Startup banner lines, in the order the C++ printed them to stdout.
    pub startup_messages: Vec<String>,
}

impl ResolvedConfig {
    pub fn is_test_fixed_diff(&self) -> bool {
        self.test_fixed_diff.is_some()
    }
}

/// Resolve the command line against `config.txt`. `prompter` is only consulted for the
/// paths that were interactive in the C++ too: first-run identity setup, repairing an
/// invalid stored value, and `--display prompt`.
pub fn resolve(
    cli: &Cli,
    options: &ResolveOptions,
    prompter: &mut dyn Prompter,
) -> Result<ResolvedConfig, ResolveError> {
    let mut startup_messages = Vec::new();

    // --- dashboard (command line half) ---
    let mut dashboard_bind = cli
        .dashboard_bind
        .clone()
        .unwrap_or_else(|| DEFAULT_DASHBOARD_BIND.to_string());
    let mut dashboard_port = DEFAULT_DASHBOARD_PORT;
    if let Some(port) = cli.dashboard_port {
        if !(1..=65535).contains(&port) {
            return Err(ResolveError::DashboardPort(port));
        }
        dashboard_port = port as u16;
    }

    // --- display ---
    let requested = match cli.display.as_deref() {
        None => DisplayMode::Logs,
        Some(text) => DisplayMode::parse(text).ok_or(ResolveError::DisplayMode)?,
    };
    let decision = resolve_display_mode(requested, &options.display_env);
    let display_warning = decision.warning.clone();
    let display_mode = if decision.prompt_required {
        let selection = prompter
            .prompt_line(PROMPT_TEXT)
            .map_err(|e| ResolveError::Config(e.to_string()))?
            .unwrap_or_default();
        resolve_prompt_selection(&selection, &options.display_env)
    } else {
        decision.mode
    };

    // --- test mode ---
    let test_fixed_diff = cli.test_fixed_diff.map(|value| value.max(0) as u32);

    if let Some(pattern) = &cli.test_block_pattern {
        startup_messages.push(format!("Test block pattern override: {pattern}"));
    }

    let max_batch_size = match cli.batch_size {
        Some(size) => {
            let size = size.max(0) as usize;
            startup_messages.push(format!("Max batch size override: {size}"));
            size
        }
        None => 0,
    };

    // --- config-file backed keys ---
    let mut config = ConfigManager::new(&options.config_path);
    config.load();

    let mut margin = MarginConfig::default();
    let mode_text = match &cli.difficulty_margin_mode {
        Some(text) => text.clone(),
        None => config.get("difficulty_margin_mode").to_string(),
    };
    if !mode_text.is_empty() {
        margin.mode =
            MarginMode::parse(&mode_text).ok_or(ResolveError::MarginMode(mode_text.clone()))?;
    }
    margin.margin_kib = read_positive_int(
        &config,
        "difficulty_margin",
        cli.difficulty_margin,
        margin.margin_kib,
    )?;
    margin.max_kib = read_positive_int(
        &config,
        "difficulty_margin_max",
        cli.difficulty_margin_max,
        margin.max_kib,
    )?;

    let mut difficulty_margin = 0;
    if margin.mode != MarginMode::Off {
        let mut line = format!(
            "Difficulty margin: mode={} step={} KiB",
            margin.mode.as_str(),
            margin.margin_kib
        );
        if margin.mode == MarginMode::Auto {
            line.push_str(&format!(" max={} KiB", margin.max_kib));
        } else {
            difficulty_margin = margin.margin_kib;
        }
        startup_messages.push(line);
    }

    let journal_text = match &cli.journal_path {
        Some(text) => text.clone(),
        None => config.get("journal_path").to_string(),
    };
    let journal_path = if journal_text.is_empty() {
        PathBuf::from(DEFAULT_JOURNAL_PATH)
    } else {
        PathBuf::from(journal_text)
    };

    if cli.dashboard_bind.is_none() {
        let configured = config.get("dashboard_bind");
        if !configured.is_empty() {
            dashboard_bind = configured.to_string();
        }
    }
    if cli.dashboard_port.is_none() {
        // A bad port in the file is ignored rather than fatal, matching the C++ `catch(...)`:
        // an unreachable dashboard must never stop a rig from mining.
        if let Some(port) = stoi(config.get("dashboard_port")) {
            if (1..=65535).contains(&port) {
                dashboard_port = port as u16;
            }
        }
    }

    if !is_valid_dashboard_bind(&dashboard_bind) {
        return Err(ResolveError::DashboardBind(dashboard_bind));
    }

    // --- device parallelism ---
    let mut gpu_streams_per_device = 1usize;
    if let Some(requested) = cli.gpu_streams {
        if !(1..=2).contains(&requested) {
            return Err(ResolveError::GpuStreams(requested));
        }
        gpu_streams_per_device = requested as usize;
        startup_messages.push(format!("GPU streams per device: {gpu_streams_per_device}"));
    }

    let mut cpu_worker_count = 0usize;
    if let Some(requested) = cli.cpu_workers {
        let threads = options.logical_threads;
        if requested < 0 || (threads > 0 && requested as u32 > threads) {
            return Err(ResolveError::CpuWorkers {
                requested,
                ceiling: if threads > 0 { threads } else { 256 },
            });
        }
        cpu_worker_count = requested as usize;
        startup_messages.push(format!("CPU sidecar workers: {cpu_worker_count}"));
    }

    let mut cpu_max_difficulty = DEFAULT_CPU_MAX_DIFFICULTY;
    if let Some(requested) = cli.cpu_max_difficulty {
        if requested < 0 || i64::from(requested) > MAX_CONFIGURED_KIB {
            return Err(ResolveError::CpuMaxDifficulty(requested));
        }
        cpu_max_difficulty = requested as u32;
    }

    // --- identity ---
    let mut app_config = AppConfig::new(&options.config_path);
    let (mut miner_address, mut eco_devfee_address, mut devfee_permillage);
    if test_fixed_diff.is_none() {
        if cli.miner_addr.is_none() || cli.total_dev_fee.is_none() {
            app_config.load(prompter)?;
        } else {
            app_config.try_load();
        }
        miner_address = app_config.account_address().to_string();
        eco_devfee_address = app_config.eco_dev_addr().to_string();
        devfee_permillage = app_config.devfee_permillage();
    } else {
        miner_address = "0x0000000000000000000000000000000000000000".to_string();
        eco_devfee_address = "0x0000000000000000000000000000000000000000".to_string();
        devfee_permillage = 0;
    }

    if let Some(total_dev_fee) = cli.total_dev_fee {
        if !(0..=1000).contains(&total_dev_fee) {
            return Err(ResolveError::TotalDevFee(total_dev_fee));
        }
        devfee_permillage = total_dev_fee;
        startup_messages.push(format!("Total developer fee set to: {total_dev_fee}"));
    }
    if let Some(address) = &cli.eco_dev_addr {
        if !tm_core::is_valid_ethereum_address(address) {
            return Err(ResolveError::EcoDevAddr(address.clone()));
        }
        eco_devfee_address = address.clone();
        startup_messages.push(format!("Ecosystem developer fee address: {address}"));
    }
    if let Some(address) = &cli.miner_addr {
        if !tm_core::is_valid_ethereum_address(address) {
            return Err(ResolveError::MinerAddr(address.clone()));
        }
        miner_address = address.clone();
        startup_messages.push(format!("Miner address: {address}"));
    }

    if !eco_devfee_address.is_empty() && !tm_core::is_valid_ethereum_address(&eco_devfee_address) {
        return Err(ResolveError::EcoDevAddr(eco_devfee_address));
    }
    if !tm_core::is_valid_ethereum_address(&miner_address) {
        return Err(ResolveError::MinerAddr(miner_address));
    }
    if !(0..=1000).contains(&devfee_permillage) {
        return Err(ResolveError::TotalDevFee(devfee_permillage));
    }

    if cli.save_config {
        app_config.set_account_address(miner_address.clone());
        if !eco_devfee_address.is_empty() {
            app_config.set_eco_dev_addr(eco_devfee_address.clone());
        }
        app_config.set_devfee_permillage(devfee_permillage);
        app_config.save()?;
        startup_messages.push("Configuration file updated with console inputs.".to_string());
    }

    let rpc_link = cli.rpc_link.clone().unwrap_or_else(|| DEFAULT_RPC_LINK.to_string());
    let custom_name = cli.custom_name.clone().unwrap_or_default();
    let mqtt_broker = cli.mqtt_broker.clone().unwrap_or_default();
    let worker_id = cli.worker_id.clone().unwrap_or_default();

    if cli.platform_mode && mqtt_broker.is_empty() {
        return Err(ResolveError::PlatformModeWithoutBroker);
    }

    startup_messages.push(format!("RPC Link: {rpc_link}"));
    let eco_note = if devfee_permillage != 0 && !eco_devfee_address.is_empty() {
        format!(" Ecosystem devfee address: {eco_devfee_address}")
    } else {
        String::new()
    };
    startup_messages.push(format!(
        "Logged in as {miner_address}. Devfee set at {devfee_permillage}/1000.{eco_note}"
    ));

    // Boot difficulty: the fixed test value, or the cache, or the 42069 fallback.
    let (initial_difficulty, difficulty_seed_note) = match test_fixed_diff {
        Some(fixed) => (fixed, None),
        None => seed_initial_difficulty(&options.difficulty_cache_path),
    };

    Ok(ResolvedConfig {
        execute: cli.execute,
        donotupload: cli.donotupload,
        save_config: cli.save_config,
        miner_address,
        eco_devfee_address,
        devfee_permillage,
        device_list: cli.device.clone().unwrap_or_default(),
        gpu_streams_per_device,
        cpu_worker_count,
        cpu_max_difficulty,
        max_batch_size,
        display_mode,
        display_warning,
        dashboard_bind,
        dashboard_port,
        rpc_link,
        custom_name,
        platform_mode: cli.platform_mode,
        mqtt_broker,
        worker_id,
        journal_path,
        difficulty_cache_path: options.difficulty_cache_path.clone(),
        margin,
        difficulty_margin,
        test_fixed_diff,
        test_block_pattern: cli.test_block_pattern.clone(),
        initial_difficulty,
        difficulty_seed_note,
        startup_messages,
    })
}

/// The C++ `readPositiveInt` lambda: CLI wins over the file, an empty value keeps the
/// current default, and anything unparseable or out of range is fatal.
fn read_positive_int(
    config: &ConfigManager,
    key: &str,
    flag: Option<i32>,
    default: u32,
) -> Result<u32, ResolveError> {
    let text = match flag {
        Some(value) => value.to_string(),
        None => config.get(key).to_string(),
    };
    if text.is_empty() {
        return Ok(default);
    }
    let Some(value) = stoi(&text) else {
        return Err(ResolveError::ValueNotANumber { key: key.to_string(), text });
    };
    if !(0..=MAX_CONFIGURED_KIB).contains(&value) {
        return Err(ResolveError::ValueOutOfRange { key: key.to_string(), text });
    }
    Ok(value as u32)
}

/// Absolute form of the journal path, for the startup log. A relative path resolved against
/// an unexpected working directory is the whole reason this is logged.
pub fn absolute_path(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| match std::env::current_dir() {
        Ok(cwd) if path.is_relative() => cwd.join(path),
        _ => path.to_path_buf(),
    })
}
