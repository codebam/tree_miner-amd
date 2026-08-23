//! Every flag in the C++ `add_options()` block, parsed the way deployment scripts write it.

use clap::Parser;
use treeminer::Cli;

fn parse(args: &[&str]) -> Cli {
    let mut full = vec!["treeminer"];
    full.extend_from_slice(args);
    Cli::try_parse_from(full).expect("parse")
}

fn fails(args: &[&str]) -> String {
    let mut full = vec!["treeminer"];
    full.extend_from_slice(args);
    Cli::try_parse_from(full).expect_err("must fail").to_string()
}

#[test]
fn every_option_from_the_cpp_block_is_present() {
    let cli = parse(&[
        "--totalDevFee",
        "10",
        "--ecoDevAddr",
        "0xAbC",
        "--minerAddr",
        "0xdEf",
        "--execute",
        "--donotupload",
        "--device",
        "1,2,7",
        "--saveConfig",
        "--testFixedDiff",
        "8",
        "--rpcLink",
        "http://example.test",
        "--customName",
        "rig-01",
        "--platform-mode",
        "--mqtt-broker",
        "tcp://broker:1883",
        "--worker-id",
        "w-9",
        "--testBlockPattern",
        "XUNI",
        "--batchSize",
        "512",
        "--difficultyMarginMode",
        "auto",
        "--difficultyMargin",
        "1500",
        "--difficultyMarginMax",
        "7000",
        "--journalPath",
        "/var/lib/treeminer.db",
        "--gpuStreams",
        "2",
        "--cpuWorkers",
        "4",
        "--cpuMaxDifficulty",
        "250",
        "--dashboard-bind",
        "127.0.0.1",
        "--dashboard-port",
        "8080",
        "--display",
        "logs",
    ]);

    assert_eq!(cli.total_dev_fee, Some(10));
    assert_eq!(cli.eco_dev_addr.as_deref(), Some("0xAbC"));
    assert_eq!(cli.miner_addr.as_deref(), Some("0xdEf"));
    assert!(cli.execute);
    assert!(cli.donotupload);
    assert_eq!(cli.device.as_deref(), Some("1,2,7"));
    assert!(cli.save_config);
    assert_eq!(cli.test_fixed_diff, Some(8));
    assert_eq!(cli.rpc_link.as_deref(), Some("http://example.test"));
    assert_eq!(cli.custom_name.as_deref(), Some("rig-01"));
    assert!(cli.platform_mode);
    assert_eq!(cli.mqtt_broker.as_deref(), Some("tcp://broker:1883"));
    assert_eq!(cli.worker_id.as_deref(), Some("w-9"));
    assert_eq!(cli.test_block_pattern.as_deref(), Some("XUNI"));
    assert_eq!(cli.batch_size, Some(512));
    assert_eq!(cli.difficulty_margin_mode.as_deref(), Some("auto"));
    assert_eq!(cli.difficulty_margin, Some(1500));
    assert_eq!(cli.difficulty_margin_max, Some(7000));
    assert_eq!(cli.journal_path.as_deref(), Some("/var/lib/treeminer.db"));
    assert_eq!(cli.gpu_streams, Some(2));
    assert_eq!(cli.cpu_workers, Some(4));
    assert_eq!(cli.cpu_max_difficulty, Some(250));
    assert_eq!(cli.dashboard_bind.as_deref(), Some("127.0.0.1"));
    assert_eq!(cli.dashboard_port, Some(8080));
    assert_eq!(cli.display.as_deref(), Some("logs"));
}

#[test]
fn nothing_is_set_when_nothing_is_passed() {
    let cli = parse(&[]);
    assert_eq!(cli.total_dev_fee, None);
    assert_eq!(cli.miner_addr, None);
    assert!(!cli.execute);
    assert!(!cli.save_config);
    assert!(!cli.platform_mode);
    assert_eq!(cli.display, None);
}

/// The C++ help text documented `--device=1,2,7`; existing HiveOS wrappers use that form.
#[test]
fn equals_and_space_forms_are_both_accepted() {
    assert_eq!(parse(&["--device=1,2,7"]).device.as_deref(), Some("1,2,7"));
    assert_eq!(parse(&["--device", "1,2,7"]).device.as_deref(), Some("1,2,7"));
    assert_eq!(parse(&["--dashboard-port=8080"]).dashboard_port, Some(8080));
    assert_eq!(parse(&["--gpuStreams=2"]).gpu_streams, Some(2));
}

/// A comma list must stay one value; splitting it would silently drop devices.
#[test]
fn device_list_is_not_split_on_commas() {
    assert_eq!(parse(&["--device", "0,1"]).device.as_deref(), Some("0,1"));
}

#[test]
fn integer_options_reject_non_numbers() {
    assert!(fails(&["--gpuStreams", "two"]).contains("two"));
    assert!(fails(&["--dashboard-port", "abc"]).contains("abc"));
    assert!(fails(&["--totalDevFee", "x"]).contains("x"));
}

#[test]
fn negative_integers_parse_so_resolution_can_report_the_cpp_message() {
    assert_eq!(parse(&["--cpuWorkers=-1"]).cpu_workers, Some(-1));
    assert_eq!(parse(&["--gpuStreams=-3"]).gpu_streams, Some(-3));
}

#[test]
fn value_options_require_a_value() {
    assert!(!fails(&["--minerAddr"]).is_empty());
    assert!(!fails(&["--display"]).is_empty());
}

#[test]
fn unknown_flags_are_rejected() {
    assert!(fails(&["--nope"]).contains("nope"));
}

#[test]
fn help_lists_every_flag() {
    let mut buffer = Vec::new();
    <Cli as clap::CommandFactory>::command()
        .write_help(&mut buffer)
        .expect("help");
    let help = String::from_utf8(buffer).expect("utf8");
    for flag in [
        "--totalDevFee",
        "--ecoDevAddr",
        "--minerAddr",
        "--execute",
        "--donotupload",
        "--device",
        "--saveConfig",
        "--testFixedDiff",
        "--rpcLink",
        "--customName",
        "--platform-mode",
        "--mqtt-broker",
        "--worker-id",
        "--testBlockPattern",
        "--batchSize",
        "--difficultyMarginMode",
        "--difficultyMargin",
        "--difficultyMarginMax",
        "--journalPath",
        "--gpuStreams",
        "--cpuWorkers",
        "--cpuMaxDifficulty",
        "--dashboard-bind",
        "--dashboard-port",
        "--display",
    ] {
        assert!(help.contains(flag), "missing {flag} in --help");
    }
}

// ------------------------------------------------------- vendor-neutral stream spelling

/// `--gpuStreams` is the spelling operators should see; `--cudaStreams` is what the C++
/// miner called it and what every deployed unit file passes today.
#[test]
fn both_stream_spellings_parse_to_the_same_field() {
    assert_eq!(parse(&["--gpuStreams", "2"]).gpu_streams, Some(2));
    assert_eq!(parse(&["--cudaStreams", "2"]).gpu_streams, Some(2));
    assert_eq!(parse(&["--cudaStreams=1"]).gpu_streams, Some(1));
}

/// The alias must not be advertised: the kernels are HIP, and help text that names CUDA
/// tells the operator they are on hardware they are not on.
#[test]
fn the_cuda_stream_alias_is_hidden_from_help() {
    let mut buffer = Vec::new();
    <Cli as clap::CommandFactory>::command()
        .write_help(&mut buffer)
        .expect("help");
    let help = String::from_utf8(buffer).expect("utf8");
    assert!(help.contains("--gpuStreams"));
    assert!(!help.contains("cudaStreams"), "the alias leaked into --help");
    assert!(!help.contains("CUDA"), "help text still names CUDA");
}
