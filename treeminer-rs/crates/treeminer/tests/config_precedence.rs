//! Config resolution: CLI > config.txt > default, the bounded options, `--saveConfig`, and
//! the validation messages the C++ printed verbatim.

use std::fs;
use std::path::PathBuf;

use clap::Parser;
use tm_submit::MarginMode;
use tm_tui::{DisplayEnv, DisplayMode};
use treeminer::{
    resolve, Cli, ConfigManager, NoPrompter, ResolveError, ResolveOptions, ResolvedConfig,
    ScriptedPrompter,
};

/// A real EIP-55 address, checksummed here rather than pasted so the test cannot rot.
fn address(seed: u8) -> String {
    let body: String = (0..40).map(|i| char::from(b'a' + ((seed + i as u8) % 6))).collect();
    tm_core::to_checksum_address(&format!("0x{body}")).expect("checksum")
}

struct Fixture {
    _dir: tempfile::TempDir,
    config_path: PathBuf,
    cache_path: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let config_path = dir.path().join("config.txt");
        let cache_path = dir.path().join("difficulty.cache");
        Self { _dir: dir, config_path, cache_path }
    }

    fn write_config(&self, pairs: &[(&str, &str)]) {
        let mut config = ConfigManager::new(&self.config_path);
        for (key, value) in pairs {
            config.set(key, *value);
        }
        config.save().expect("save config");
    }

    fn options(&self) -> ResolveOptions {
        ResolveOptions {
            config_path: self.config_path.clone(),
            difficulty_cache_path: self.cache_path.clone(),
            display_env: DisplayEnv {
                stdout_is_tty: false,
                no_tui_env: false,
                invocation_id_env: false,
            },
            logical_threads: 8,
        }
    }

    /// Resolve with an identity supplied on the command line, so nothing blocks on stdin.
    fn resolve(&self, args: &[&str]) -> Result<ResolvedConfig, ResolveError> {
        let miner = address(0);
        let mut full = vec!["treeminer", "--minerAddr", &miner];
        // The fixture supplies an identity so nothing blocks on stdin, but a test that sets
        // its own fee must not end up passing the flag twice.
        if !args.contains(&"--totalDevFee") {
            full.extend_from_slice(&["--totalDevFee", "0"]);
        }
        full.extend_from_slice(args);
        let cli = Cli::try_parse_from(full).expect("parse");
        resolve(&cli, &self.options(), &mut NoPrompter)
    }
}

// ---------------------------------------------------------------- precedence

#[test]
fn margin_precedence_cli_beats_config_beats_default() {
    let fixture = Fixture::new();

    // default
    let resolved = fixture.resolve(&[]).expect("resolve");
    assert_eq!(resolved.margin.mode, MarginMode::Off);
    assert_eq!(resolved.margin.margin_kib, 1000);
    assert_eq!(resolved.margin.max_kib, 5000);

    // config file
    fixture.write_config(&[
        ("difficulty_margin_mode", "auto"),
        ("difficulty_margin", "1500"),
        ("difficulty_margin_max", "9000"),
    ]);
    let resolved = fixture.resolve(&[]).expect("resolve");
    assert_eq!(resolved.margin.mode, MarginMode::Auto);
    assert_eq!(resolved.margin.margin_kib, 1500);
    assert_eq!(resolved.margin.max_kib, 9000);

    // command line wins over the file
    let resolved = fixture
        .resolve(&[
            "--difficultyMarginMode",
            "fixed",
            "--difficultyMargin",
            "2222",
            "--difficultyMarginMax",
            "3333",
        ])
        .expect("resolve");
    assert_eq!(resolved.margin.mode, MarginMode::Fixed);
    assert_eq!(resolved.margin.margin_kib, 2222);
    assert_eq!(resolved.margin.max_kib, 3333);
    // Fixed headroom is applied immediately; auto ramps at runtime instead.
    assert_eq!(resolved.difficulty_margin, 2222);
}

#[test]
fn auto_mode_does_not_prepay_headroom() {
    let fixture = Fixture::new();
    let resolved = fixture
        .resolve(&["--difficultyMarginMode", "auto", "--difficultyMargin", "1234"])
        .expect("resolve");
    assert_eq!(resolved.difficulty_margin, 0);
    assert!(resolved
        .startup_messages
        .iter()
        .any(|m| m == "Difficulty margin: mode=auto step=1234 KiB max=5000 KiB"));
}

#[test]
fn journal_path_precedence() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture.resolve(&[]).expect("resolve").journal_path,
        PathBuf::from("treeminer-journal.db")
    );

    fixture.write_config(&[("journal_path", "/var/lib/from-config.db")]);
    assert_eq!(
        fixture.resolve(&[]).expect("resolve").journal_path,
        PathBuf::from("/var/lib/from-config.db")
    );

    assert_eq!(
        fixture
            .resolve(&["--journalPath", "/srv/from-cli.db"])
            .expect("resolve")
            .journal_path,
        PathBuf::from("/srv/from-cli.db")
    );
}

#[test]
fn dashboard_precedence() {
    let fixture = Fixture::new();
    let resolved = fixture.resolve(&[]).expect("resolve");
    assert_eq!(resolved.dashboard_bind, "0.0.0.0");
    assert_eq!(resolved.dashboard_port, 42069);

    fixture.write_config(&[("dashboard_bind", "127.0.0.1"), ("dashboard_port", "8080")]);
    let resolved = fixture.resolve(&[]).expect("resolve");
    assert_eq!(resolved.dashboard_bind, "127.0.0.1");
    assert_eq!(resolved.dashboard_port, 8080);

    let resolved = fixture
        .resolve(&["--dashboard-bind", "::1", "--dashboard-port", "9000"])
        .expect("resolve");
    assert_eq!(resolved.dashboard_bind, "::1");
    assert_eq!(resolved.dashboard_port, 9000);
}

/// An unreachable dashboard must never stop a rig mining, so a bad port in the file is
/// ignored — unlike the same value on the command line, which is fatal.
#[test]
fn a_bad_dashboard_port_in_the_file_falls_back_to_the_default() {
    let fixture = Fixture::new();
    fixture.write_config(&[("dashboard_port", "99999")]);
    assert_eq!(fixture.resolve(&[]).expect("resolve").dashboard_port, 42069);
    assert_eq!(
        fixture.resolve(&["--dashboard-port", "99999"]).unwrap_err(),
        ResolveError::DashboardPort(99999)
    );
}

#[test]
fn identity_precedence_cli_beats_config() {
    let fixture = Fixture::new();
    let stored = address(1);
    fixture.write_config(&[
        ("account_address", stored.as_str()),
        ("devfee_permillage", "25"),
    ]);

    // No CLI identity: the file wins over the (zero) default.
    let cli = Cli::try_parse_from(["treeminer"]).expect("parse");
    let resolved =
        resolve(&cli, &fixture.options(), &mut ScriptedPrompter::new(Vec::<String>::new()))
            .expect("resolve");
    assert_eq!(resolved.miner_address, stored);
    assert_eq!(resolved.devfee_permillage, 25);

    // CLI identity wins.
    let from_cli = address(2);
    let cli = Cli::try_parse_from([
        "treeminer",
        "--minerAddr",
        &from_cli,
        "--totalDevFee",
        "77",
    ])
    .expect("parse");
    let resolved = resolve(&cli, &fixture.options(), &mut NoPrompter).expect("resolve");
    assert_eq!(resolved.miner_address, from_cli);
    assert_eq!(resolved.devfee_permillage, 77);
}

// ---------------------------------------------------------------- bounded options

#[test]
fn gpu_streams_is_bounded_to_one_or_two() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture.resolve(&["--gpuStreams", "1"]).expect("resolve").gpu_streams_per_device,
        1
    );
    assert_eq!(
        fixture.resolve(&["--gpuStreams", "2"]).expect("resolve").gpu_streams_per_device,
        2
    );
    for bad in [0, 3, -1] {
        let error = fixture.resolve(&["--gpuStreams", &bad.to_string()]).unwrap_err();
        assert_eq!(error, ResolveError::GpuStreams(bad));
        assert_eq!(
            error.to_string(),
            format!("The argument ({bad}) for GPU streams must be 1 or 2.")
        );
    }
    // Absent means the C++ default of one stream, with no banner line.
    let resolved = fixture.resolve(&[]).expect("resolve");
    assert_eq!(resolved.gpu_streams_per_device, 1);
    assert!(!resolved.startup_messages.iter().any(|m| m.starts_with("GPU streams")));
}

/// The deployed spelling must resolve identically, including the banner line, so an
/// existing unit file that passes `--cudaStreams` keeps its second queue.
#[test]
fn the_cuda_streams_alias_resolves_like_the_primary_spelling() {
    let fixture = Fixture::new();
    let aliased = fixture.resolve(&["--cudaStreams", "2"]).expect("resolve");
    let primary = fixture.resolve(&["--gpuStreams", "2"]).expect("resolve");
    assert_eq!(aliased.gpu_streams_per_device, 2);
    assert_eq!(aliased.gpu_streams_per_device, primary.gpu_streams_per_device);
    assert!(aliased
        .startup_messages
        .iter()
        .any(|m| m == "GPU streams per device: 2"));
    assert_eq!(
        fixture.resolve(&["--cudaStreams", "0"]).unwrap_err(),
        ResolveError::GpuStreams(0)
    );
}

#[test]
fn cpu_workers_is_bounded_by_the_logical_thread_count() {
    let fixture = Fixture::new();
    assert_eq!(fixture.resolve(&["--cpuWorkers", "8"]).expect("resolve").cpu_worker_count, 8);
    assert_eq!(
        fixture.resolve(&["--cpuWorkers", "9"]).unwrap_err().to_string(),
        "The argument (9) for CPU workers must be between 0 and 8."
    );
    assert_eq!(
        fixture.resolve(&["--cpuWorkers", "-1"]).unwrap_err().to_string(),
        "The argument (-1) for CPU workers must be between 0 and 8."
    );
}

/// `hardware_concurrency() == 0` (unknown) removed the upper bound but still said 256.
#[test]
fn cpu_workers_with_unknown_thread_count_only_rejects_negatives() {
    let fixture = Fixture::new();
    let mut options = fixture.options();
    options.logical_threads = 0;
    let miner = address(0);

    let cli = Cli::try_parse_from([
        "treeminer", "--minerAddr", &miner, "--totalDevFee", "0", "--cpuWorkers", "999",
    ])
    .expect("parse");
    assert_eq!(
        resolve(&cli, &options, &mut NoPrompter).expect("resolve").cpu_worker_count,
        999
    );

    let cli = Cli::try_parse_from([
        "treeminer", "--minerAddr", &miner, "--totalDevFee", "0", "--cpuWorkers", "-1",
    ])
    .expect("parse");
    assert_eq!(
        resolve(&cli, &options, &mut NoPrompter).unwrap_err().to_string(),
        "The argument (-1) for CPU workers must be between 0 and 256."
    );
}

#[test]
fn cpu_max_difficulty_is_bounded() {
    let fixture = Fixture::new();
    assert_eq!(fixture.resolve(&[]).expect("resolve").cpu_max_difficulty, 100);
    assert_eq!(
        fixture.resolve(&["--cpuMaxDifficulty", "0"]).expect("resolve").cpu_max_difficulty,
        0
    );
    assert_eq!(
        fixture.resolve(&["--cpuMaxDifficulty", "-5"]).unwrap_err().to_string(),
        "The argument (-5) for the CPU difficulty ceiling must be 0-100000000."
    );
}

#[test]
fn margin_mode_rejects_a_typo_with_the_cpp_message() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture.resolve(&["--difficultyMarginMode", "atuo"]).unwrap_err().to_string(),
        "Invalid difficulty margin mode 'atuo' (expected: off | fixed | auto)."
    );
    // The same typo in the file is equally fatal — it would otherwise mine at the wrong
    // memory cost and show up only as unexplained 401s.
    fixture.write_config(&[("difficulty_margin_mode", "atuo")]);
    assert_eq!(
        fixture.resolve(&[]).unwrap_err().to_string(),
        "Invalid difficulty margin mode 'atuo' (expected: off | fixed | auto)."
    );
}

#[test]
fn margin_values_report_range_and_type_errors_per_key() {
    let fixture = Fixture::new();
    fixture.write_config(&[("difficulty_margin", "nine")]);
    assert_eq!(
        fixture.resolve(&[]).unwrap_err().to_string(),
        "Value for difficulty_margin (nine) is not a number."
    );

    fixture.write_config(&[("difficulty_margin_max", "100000001")]);
    assert_eq!(
        fixture.resolve(&[]).unwrap_err().to_string(),
        "Value for difficulty_margin_max (100000001) is out of range (0-100000000)."
    );
}

#[test]
fn display_modes_are_the_three_documented_ones() {
    let fixture = Fixture::new();
    assert_eq!(fixture.resolve(&[]).expect("resolve").display_mode, DisplayMode::Logs);
    assert_eq!(
        fixture.resolve(&["--display", "logs"]).expect("resolve").display_mode,
        DisplayMode::Logs
    );
    assert_eq!(
        fixture.resolve(&["--display", "bogus"]).unwrap_err().to_string(),
        "The display mode must be logs, terminal, or prompt."
    );
}

/// A non-interactive run (the fixture's env) must downgrade and say so, never enter the
/// alternate screen.
#[test]
fn terminal_and_prompt_downgrade_to_logs_when_the_tui_is_forbidden() {
    let fixture = Fixture::new();
    for requested in ["terminal", "prompt"] {
        let resolved = fixture.resolve(&["--display", requested]).expect("resolve");
        assert_eq!(resolved.display_mode, DisplayMode::Logs);
        assert_eq!(
            resolved.display_warning.as_deref(),
            Some(
                format!(
                    "Display '{requested}' is disabled for service/non-interactive runs; using logs."
                )
                .as_str()
            )
        );
    }
}

#[test]
fn prompt_on_a_real_tty_asks_once_and_honours_the_answer() {
    let fixture = Fixture::new();
    let mut options = fixture.options();
    options.display_env.stdout_is_tty = true;
    let miner = address(0);
    let cli = Cli::try_parse_from([
        "treeminer", "--minerAddr", &miner, "--totalDevFee", "0", "--display", "prompt",
    ])
    .expect("parse");

    let mut prompter = ScriptedPrompter::new(["2"]);
    let resolved = resolve(&cli, &options, &mut prompter).expect("resolve");
    assert_eq!(resolved.display_mode, DisplayMode::Logs);
    assert_eq!(prompter.prompts.len(), 1);

    let mut prompter = ScriptedPrompter::new([""]);
    let resolved = resolve(&cli, &options, &mut prompter).expect("resolve");
    assert_eq!(resolved.display_mode, DisplayMode::Terminal);
}

// ---------------------------------------------------------------- validation

#[test]
fn a_non_eip55_miner_address_is_rejected_with_the_cpp_message() {
    let fixture = Fixture::new();
    // All-lowercase: a valid hex address, but not the EIP-55 checksummed form.
    let lowercase = address(0).to_lowercase();
    let cli = Cli::try_parse_from([
        "treeminer", "--minerAddr", &lowercase, "--totalDevFee", "0",
    ])
    .expect("parse");
    let error = resolve(&cli, &fixture.options(), &mut NoPrompter).unwrap_err();
    assert_eq!(
        error.to_string(),
        format!("The argument ({lowercase}) for miner address (EIP55) is invalid.")
    );
}

#[test]
fn a_non_eip55_ecosystem_address_is_rejected_with_the_cpp_message() {
    let fixture = Fixture::new();
    let bad = "0xnothexatall";
    let error = fixture.resolve(&["--ecoDevAddr", bad]).unwrap_err();
    assert_eq!(
        error.to_string(),
        format!("The argument ({bad}) for ecosystem developer fee address (EIP55) is invalid.")
    );
}

#[test]
fn total_dev_fee_is_bounded_to_permillage() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture.resolve(&["--totalDevFee", "1001"]).unwrap_err().to_string(),
        "The argument (1001) for total developer fee (0-1000) is invalid."
    );
    assert_eq!(
        fixture.resolve(&["--totalDevFee", "1000"]).expect("resolve").devfee_permillage,
        1000
    );
}

#[test]
fn platform_mode_requires_a_broker() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture.resolve(&["--platform-mode"]).unwrap_err().to_string(),
        "Platform mode requires --mqtt-broker to be set."
    );
    assert!(fixture
        .resolve(&["--platform-mode", "--mqtt-broker", "tcp://b:1883"])
        .is_ok());
}

#[test]
fn a_bad_dashboard_bind_is_rejected() {
    let fixture = Fixture::new();
    assert_eq!(
        fixture.resolve(&["--dashboard-bind", "not-an-ip"]).unwrap_err().to_string(),
        "Invalid dashboard bind address 'not-an-ip': expected an IPv4 or IPv6 address."
    );
}

// ---------------------------------------------------------------- --saveConfig

#[test]
fn save_config_writes_the_console_inputs_back_to_the_file() {
    let fixture = Fixture::new();
    fixture.write_config(&[("account_address", address(1).as_str()), ("devfee_permillage", "5")]);

    let miner = address(3);
    let eco = address(4);
    let cli = Cli::try_parse_from([
        "treeminer",
        "--minerAddr",
        &miner,
        "--totalDevFee",
        "42",
        "--ecoDevAddr",
        &eco,
        "--saveConfig",
    ])
    .expect("parse");
    let resolved = resolve(&cli, &fixture.options(), &mut NoPrompter).expect("resolve");
    assert!(resolved
        .startup_messages
        .iter()
        .any(|m| m == "Configuration file updated with console inputs."));

    let mut written = ConfigManager::new(&fixture.config_path);
    written.load();
    assert_eq!(written.get("account_address"), miner);
    assert_eq!(written.get("ecodev_address"), eco);
    assert_eq!(written.get("devfee_permillage"), "42");
}

#[test]
fn without_save_config_the_file_keeps_its_previous_identity() {
    let fixture = Fixture::new();
    let stored = address(1);
    fixture.write_config(&[("account_address", stored.as_str()), ("devfee_permillage", "5")]);

    let miner = address(3);
    let cli =
        Cli::try_parse_from(["treeminer", "--minerAddr", &miner, "--totalDevFee", "42"])
            .expect("parse");
    resolve(&cli, &fixture.options(), &mut NoPrompter).expect("resolve");

    let mut written = ConfigManager::new(&fixture.config_path);
    written.load();
    assert_eq!(written.get("account_address"), stored);
    assert_eq!(written.get("devfee_permillage"), "5");
}

// ---------------------------------------------------------------- test mode + banner

#[test]
fn test_fixed_diff_uses_the_null_identity_and_skips_the_cache() {
    let fixture = Fixture::new();
    fs::write(&fixture.cache_path, "5000\n").expect("cache");
    let cli = Cli::try_parse_from(["treeminer", "--testFixedDiff", "8"]).expect("parse");
    let resolved = resolve(&cli, &fixture.options(), &mut NoPrompter).expect("resolve");

    assert!(resolved.is_test_fixed_diff());
    assert_eq!(resolved.initial_difficulty, 8);
    assert!(resolved.difficulty_seed_note.is_none());
    assert_eq!(resolved.miner_address, "0x0000000000000000000000000000000000000000");
    assert_eq!(resolved.devfee_permillage, 0);
}

#[test]
fn the_banner_reports_the_identity_and_rpc_link_last() {
    let fixture = Fixture::new();
    let miner = address(0);
    let eco = address(4);
    let resolved = fixture
        .resolve(&["--rpcLink", "http://example.test", "--ecoDevAddr", &eco, "--totalDevFee", "10"])
        .expect("resolve");

    let messages = &resolved.startup_messages;
    assert_eq!(messages[messages.len() - 2], "RPC Link: http://example.test");
    assert_eq!(
        messages[messages.len() - 1],
        format!("Logged in as {miner}. Devfee set at 10/1000. Ecosystem devfee address: {eco}")
    );
}

/// A zero fee means nothing is ever sent to the ecosystem address, so naming it would be a
/// lie; the C++ suppressed it and so does this.
#[test]
fn a_zero_devfee_suppresses_the_ecosystem_address_in_the_banner() {
    let fixture = Fixture::new();
    let eco = address(4);
    let resolved = fixture.resolve(&["--ecoDevAddr", &eco]).expect("resolve");
    let last = resolved.startup_messages.last().expect("banner");
    assert!(last.ends_with("Devfee set at 0/1000."), "{last}");
}

// ------------------------------------------------- config-file keys are frozen vocabulary

/// The CLI vocabulary moved off vendor names; the config-file keys did not, and must not.
/// This is a `config.txt` exactly as an older build wrote it, byte for byte.
#[test]
fn a_config_file_from_the_old_build_still_loads() {
    let fixture = Fixture::new();
    let stored = address(3);
    fs::write(
        &fixture.config_path,
        format!(
            "account_address={stored}\n\
             dashboard_bind=127.0.0.1\n\
             dashboard_port=8080\n\
             devfee_permillage=25\n\
             difficulty_margin=1500\n\
             difficulty_margin_mode=fixed\n\
             journal_path=/var/lib/treeminer/journal.db\n",
        ),
    )
    .expect("write legacy config");

    let cli = Cli::try_parse_from(["treeminer"]).expect("parse");
    let resolved = resolve(&cli, &fixture.options(), &mut NoPrompter).expect("resolve");
    assert_eq!(resolved.miner_address, stored);
    assert_eq!(resolved.devfee_permillage, 25);
    assert_eq!(resolved.dashboard_bind, "127.0.0.1");
    assert_eq!(resolved.dashboard_port, 8080);
    assert_eq!(resolved.margin.mode, MarginMode::Fixed);
    assert_eq!(resolved.difficulty_margin, 1500);
    assert_eq!(
        resolved.journal_path,
        PathBuf::from("/var/lib/treeminer/journal.db")
    );
}
