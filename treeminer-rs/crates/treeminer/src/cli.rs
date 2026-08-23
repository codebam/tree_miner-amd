//! Command line surface. Port of the `add_options()` block in `src/main.cpp`.
//!
//! Every flag keeps its exact C++ spelling, including the camelCase ones and the three
//! kebab-case outliers (`--platform-mode`, `--mqtt-broker`, `--worker-id`,
//! `--dashboard-bind`, `--dashboard-port`), because deployment scripts, systemd units and
//! HiveOS wrappers in the field pass them verbatim. clap accepts both `--flag value` and
//! `--flag=value` for every one of them, which is a superset of what boost accepted.
//!
//! Nothing here validates or applies a value: an `Option` that is `Some` means "the
//! operator typed it", which is precisely the input the precedence rules in [`crate::resolve`]
//! need.

use clap::Parser;

#[derive(Debug, Clone, Default, Parser)]
#[command(
    name = "treeminer",
    about = "XenblocksMiner options",
    disable_version_flag = true,
    // boost accepted `--cudaStreams -1`; without this clap reads `-1` as an unknown short
    // flag and reports a parse error instead of the C++'s bounds message.
    allow_negative_numbers = true
)]
pub struct Cli {
    /// set total developer fee
    #[arg(long = "totalDevFee", value_name = "int")]
    pub total_dev_fee: Option<i32>,

    /// set ecosystem developer address (will receive half of the total dev fee)
    #[arg(long = "ecoDevAddr", value_name = "string")]
    pub eco_dev_addr: Option<String>,

    /// set miner address
    #[arg(long = "minerAddr", value_name = "string")]
    pub miner_addr: Option<String>,

    /// execute the miner otherwise it will run as a mointor server
    #[arg(long = "execute")]
    pub execute: bool,

    /// do not upload the data to the server
    #[arg(long = "donotupload")]
    pub donotupload: bool,

    /// device index list[--device=1,2,7] to run the miner on
    #[arg(long = "device", value_name = "string")]
    pub device: Option<String>,

    /// update configuration file with console inputs
    #[arg(long = "saveConfig")]
    pub save_config: bool,

    /// run in test mode with a fixed difficulty
    #[arg(long = "testFixedDiff", value_name = "int")]
    pub test_fixed_diff: Option<i32>,

    /// set rpc link
    #[arg(long = "rpcLink", value_name = "string")]
    pub rpc_link: Option<String>,

    /// set custom name
    #[arg(long = "customName", value_name = "string")]
    pub custom_name: Option<String>,

    /// enable hashpower marketplace platform mode
    #[arg(long = "platform-mode")]
    pub platform_mode: bool,

    /// MQTT broker URI for platform mode (e.g. tcp://broker:1883)
    #[arg(long = "mqtt-broker", value_name = "string")]
    pub mqtt_broker: Option<String>,

    /// override worker ID for platform registration
    #[arg(long = "worker-id", value_name = "string")]
    pub worker_id: Option<String>,

    /// override block detection pattern for testing (default: XEN11)
    #[arg(long = "testBlockPattern", value_name = "string")]
    pub test_block_pattern: Option<String>,

    /// limit GPU batch size (reduces VRAM usage)
    #[arg(long = "batchSize", value_name = "int")]
    pub batch_size: Option<i32>,

    /// difficulty headroom policy: off (default) | fixed | auto
    #[arg(long = "difficultyMarginMode", value_name = "string")]
    pub difficulty_margin_mode: Option<String>,

    /// headroom in KiB; fixed mode: the constant, auto mode: one ramp step (default 1000)
    #[arg(long = "difficultyMargin", value_name = "int")]
    pub difficulty_margin: Option<i32>,

    /// auto mode only: ceiling on the headroom ramp in KiB (default 5000)
    #[arg(long = "difficultyMarginMax", value_name = "int")]
    pub difficulty_margin_max: Option<i32>,

    /// find journal database file (default: treeminer-journal.db in the working directory)
    #[arg(long = "journalPath", value_name = "string")]
    pub journal_path: Option<String>,

    /// independent CUDA work streams per device (1-2)
    #[arg(long = "cudaStreams", value_name = "int")]
    pub cuda_streams: Option<i32>,

    /// independent CPU sidecar mining workers (0 disables)
    #[arg(long = "cpuWorkers", value_name = "int")]
    pub cpu_workers: Option<i32>,

    /// CPU workers hash only while difficulty <= this ceiling; they idle above it and
    /// resume when it falls (default 100; 0 = no ceiling)
    #[arg(long = "cpuMaxDifficulty", value_name = "int")]
    pub cpu_max_difficulty: Option<i32>,

    /// dashboard listen IP (default: 0.0.0.0 for Vast.ai/Docker/LAN; 127.0.0.1 for this
    /// machine only)
    #[arg(long = "dashboard-bind", value_name = "string")]
    pub dashboard_bind: Option<String>,

    /// dashboard listen port (default 42069; use 8080 if the host firewall already allows
    /// the old xen.pub API port)
    #[arg(long = "dashboard-port", value_name = "int")]
    pub dashboard_port: Option<i32>,

    /// terminal display: logs, terminal, or prompt
    #[arg(long = "display", value_name = "string")]
    pub display: Option<String>,
}

impl Cli {
    /// Parse the real process arguments.
    pub fn from_env() -> Self {
        Self::parse()
    }
}
