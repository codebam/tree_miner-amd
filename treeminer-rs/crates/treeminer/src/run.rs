//! Startup, wiring and shutdown: everything `src/main.cpp` does from device enumeration to
//! the final `return`.
//!
//! ORDER IS THE CONTRACT HERE.
//! Devices are enumerated and self-tested BEFORE a journal, a socket or a thread exists, so
//! a miner that would produce invalid digests never reaches the network. The journal opens
//! before the submitter, the submitter before the difficulty poller (it wants the poller's
//! observations), and the dashboard last, because it must advertise the port it actually
//! got. On the way down the order reverses: HTTP stops before the journal is torn down, the
//! terminal is handed back before the process exits, and a durability failure becomes a
//! nonzero exit so a supervisor restarts the miner against a recovered disk.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use tm_dashboard::{DashboardConfig, DashboardServer};
use tm_journal::{FallbackSink, FindJournal, Journal};
use tm_submit::{BreakerState, HttpTransport, SubmissionManager};
use tm_tui::{Console, DisplayMode, FileLogger, Level, Shutdown, TerminalUi};

use crate::backend::{GpuMiningBackend, GpuSelfTestProbe};
use crate::bridge::JournalBridge;
use crate::cpuworker::{CpuMiningWorker, CpuWorkerConfig, DEFAULT_CPU_BATCH_SIZE};
use crate::find::FindSink;
use crate::mineunit::{run_mining_on_device, MineDeps, MiningIdentity, DEFAULT_BACKOFF};
use crate::resolve::{absolute_path, ResolvedConfig};
use crate::selftest::{run_self_test, BACKEND_NAME};
use crate::state::{MiningState, DEFAULT_GPU_FIRST_BLOCKS};
use crate::stats::{BreakerStateLabel, StatsIdentity, StatsPublisher, SubmissionView};

/// The devfee address the C++ ships with.
const DEVFEE_ADDRESS: &str = "0x24691E54aFafe2416a8252097C9Ca67557271475";
/// Per-request HTTP budgets for `/verify` and `/get_block`.
const SUBMIT_TIMEOUT_MS: u64 = 10_000;
const GET_TIMEOUT_MS: u64 = 10_000;
/// How often the ticker/telemetry pass runs. The C++ status line redraws per batch; a fixed
/// cadence keeps it readable when batches are sub-second.
const REPORT_INTERVAL: Duration = Duration::from_secs(1);
const TELEMETRY_INTERVAL: Duration = Duration::from_secs(5);

/// Process exit codes, as the C++ used them.
pub const EXIT_OK: u8 = 0;
pub const EXIT_FAILURE: u8 = 1;
/// A find could be persisted nowhere; the supervisor should restart us.
pub const EXIT_DURABILITY: u8 = 2;

/// Run the miner. Returns the process exit code.
pub fn run(config: &ResolvedConfig) -> u8 {
    // Every timestamp this process prints is rendered in the offset captured in `main`,
    // while it was still legal to look one up.
    tm_tui::set_time_offset(crate::clock::local_offset());

    if !config.execute {
        // The C++ re-executes itself as a watchdog here. Process supervision belongs to the
        // service manager, not to the miner, so this build declines rather than forking.
        println!(
            "Monitor mode: no mining was started. Pass --execute to mine (and let systemd, \
             Docker or your rig manager supervise the process)."
        );
        return EXIT_OK;
    }

    let (devices, device_names) = match enumerate_devices(&config.device_list) {
        Ok(devices) => devices,
        Err(message) => {
            eprintln!("{message}");
            return EXIT_FAILURE;
        }
    };

    let machine_id = if config.worker_id.is_empty() {
        crate::machineid::machine_id_for_devices(&devices)
    } else {
        config.worker_id.clone()
    };
    println!("Machine ID: {machine_id}");

    // Fail closed before journals, network threads, dashboards or mining exist.
    let host = Arc::new(tm_argon2::CpuArgon2Host::new());
    let selected: Vec<i32> = devices.iter().copied().collect();
    let report = run_self_test(
        &selected,
        DEFAULT_GPU_FIRST_BLOCKS,
        &mut GpuSelfTestProbe::new(host.clone()),
    );
    report.emit();
    if report.is_fatal() {
        eprintln!("{}", report.fatal_message());
        return EXIT_FAILURE;
    }
    let mining_devices = report.mining_devices();

    let shutdown = Arc::new(Shutdown::new());
    let difficulty = Arc::new(crate::difficulty::DifficultyShared::new(
        config.initial_difficulty,
    ));
    let state = Arc::new(MiningState::new(
        Arc::clone(&difficulty),
        Arc::clone(&shutdown),
    ));
    report.apply(&state);
    state.set_margin_kib(config.difficulty_margin);

    let logger = FileLogger::new("log", tm_tui::DEFAULT_MAX_FILE_SIZE)
        .ok()
        .map(Arc::new);

    // --- journal ---
    let journal_path = config.journal_path.clone();
    let journal = match FindJournal::open(&journal_path) {
        Ok(journal) => Arc::new(journal),
        Err(error) => {
            Console::global().event(
                Level::Error,
                "JOURNAL",
                &format!("cannot open {} | {error}", journal_path.display()),
            );
            return EXIT_FAILURE;
        }
    };
    let journal: Arc<dyn Journal + Send + Sync> = journal;
    let fallback_path = fallback_path(&journal_path);
    open_journal_banner(journal.as_ref(), &journal_path, &fallback_path, &state);

    // --- submission ---
    let submission = if config.test_fixed_diff.is_some() {
        None
    } else {
        match build_submitter(config, &machine_id, Arc::clone(&journal), &state, logger.clone()) {
            Ok(manager) => Some(manager),
            Err(error) => {
                Console::global().event(
                    Level::Error,
                    "NETWORK",
                    &format!("cannot build the HTTP transport | {error}"),
                );
                return EXIT_FAILURE;
            }
        }
    };

    // --- difficulty ---
    let poller_thread = if config.test_fixed_diff.is_some() {
        println!(
            "Running in TEST MODE with fixed difficulty {}",
            config.initial_difficulty
        );
        None
    } else {
        if let Some(note) = &config.difficulty_seed_note {
            Console::global().event(Level::Info, "DIFFICULTY", note);
        }
        spawn_difficulty_poller(config, &state, submission.as_ref())
    };

    // --- find capture ---
    let mut sink = FindSink::new(
        Arc::clone(&journal),
        FallbackSink::new(&fallback_path),
        Arc::clone(&state),
        &machine_id,
    );
    if let Some(logger) = &logger {
        sink = sink.with_logger(Arc::clone(logger));
    }
    if let Some(manager) = &submission {
        let manager = Arc::clone(manager);
        sink = sink.with_notifier(Arc::new(move || manager.notify_find_appended()));
    }
    let sink = Arc::new(sink);

    // --- stats ---
    let console_url = tm_dashboard::console_url(
        &config.dashboard_bind,
        config.dashboard_port,
        &tm_dashboard::SystemInterfaces,
    );
    let publisher = Arc::new(
        stats_publisher(config, &machine_id, &console_url, &state, &journal, submission.as_ref()),
    );

    // --- display ---
    let terminal_ui = start_display(config, &publisher);

    // --- CPU sidecar ---
    let identity = mining_identity(config);
    let cpu = start_cpu_workers(config, &state, &sink, &identity);

    for index in &mining_devices {
        let name = device_names
            .get(index)
            .map(String::as_str)
            .unwrap_or("unknown device");
        tm_tui::log_to_console(&format!("Device #{index}: {name}"));
    }

    // --- GPU mining threads ---
    let deps = Arc::new(MineDeps {
        state: Arc::clone(&state),
        sink: Arc::clone(&sink),
        identity,
        max_batch_size: config.max_batch_size,
        streams_per_device: config.gpu_streams_per_device,
        xuni_window_open: Arc::new(crate::mineunit::xuni_window_open_now),
    });
    let mining_threads = spawn_mining_threads(&mining_devices, config, &deps, &host);
    if mining_threads.is_empty() {
        Console::global().event(
            Level::Error,
            "MINING",
            "no device could be opened for mining after the self-test passed",
        );
        shutdown.request_stop();
    }

    // --- dashboard + signals ---
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build();
    let runtime = match runtime {
        Ok(runtime) => runtime,
        Err(error) => {
            Console::global().event(
                Level::Error,
                "CONSOLE",
                &format!("cannot start the async runtime | {error}"),
            );
            shutdown.request_stop();
            return finish(
                &state,
                mining_threads,
                cpu,
                terminal_ui,
                poller_thread,
                submission,
            );
        }
    };
    install_signal_handlers(&runtime, Arc::clone(&shutdown));
    let dashboard_stop = start_dashboard(&runtime, config, Arc::clone(&publisher), &logger);

    // --- run ---
    report_loop(&state, &publisher, &cpu, &mining_devices, terminal_ui.is_some());

    // --- shutdown ---
    // HTTP stops before the journal is torn down: a request being served while the journal
    // closes was a live source of use-after-free in the C++ (the server used to be detached
    // and stopped from the signal handler).
    if let Some(stop) = dashboard_stop {
        let _ = stop.send(());
    }
    runtime.shutdown_timeout(Duration::from_secs(2));

    finish(
        &state,
        mining_threads,
        cpu,
        terminal_ui,
        poller_thread,
        submission,
    )
}

/// Enumerate and select devices, or explain why mining cannot start. Returns the selected
/// indices and every device's display name.
type DeviceNames = std::collections::BTreeMap<i32, String>;

fn enumerate_devices(device_list: &str) -> Result<(BTreeSet<i32>, DeviceNames), String> {
    let devices = tm_gpu::Device::enumerate().map_err(|error| {
        format!(
            "GPU device enumeration failed ({error})! Do you have a {BACKEND_NAME}-capable \
             GPU and driver installed?"
        )
    })?;
    if devices.is_empty() {
        return Err(format!(
            "No {BACKEND_NAME}-capable GPU was found. Mining was not started."
        ));
    }
    let names: DeviceNames = devices
        .iter()
        .map(|device| (device.index(), device.full_name()))
        .collect();
    let selected = crate::machineid::parse_device_list(device_list, devices.len() as i32);
    Ok((selected, names))
}

fn fallback_path(journal_path: &std::path::Path) -> std::path::PathBuf {
    let mut path = journal_path.as_os_str().to_os_string();
    path.push(".fallback.jsonl");
    std::path::PathBuf::from(path)
}

/// Log the resolved path, drain the fallback sink, then recover. The sink is drained BEFORE
/// recovery so a find that fell into it while SQLite was broken is counted and re-drained
/// like every other journaled find this boot.
fn open_journal_banner(
    journal: &(dyn Journal + Send + Sync),
    journal_path: &std::path::Path,
    fallback_path: &std::path::Path,
    state: &MiningState,
) {
    Console::global().event(
        Level::Info,
        "JOURNAL",
        &format!("path={}", absolute_path(journal_path).display()),
    );

    let sink_stats = FallbackSink::import_into(journal, fallback_path);
    if sink_stats.file_present {
        Console::global().event(
            if sink_stats.malformed > 0 {
                Level::Error
            } else {
                Level::Warn
            },
            "JOURNAL",
            &format!(
                "fallback sink drained | imported={} | malformed={} — a previous run could \
                 not write the journal; investigate why",
                sink_stats.imported, sink_stats.malformed
            ),
        );
    }

    match journal.recover_on_startup() {
        Ok(recovered) => {
            Console::global().event(
                Level::Info,
                "JOURNAL",
                &format!(
                    "recovered | pending={} | unconfirmed={} | parked={} | acked={} | \
                     quarantined={}",
                    recovered.pending,
                    recovered.accepted_unconfirmed,
                    recovered.parked_difficulty + recovered.parked_xuni,
                    recovered.acked,
                    recovered.quarantined
                ),
            );
        }
        Err(error) => Console::global().event(
            Level::Error,
            "JOURNAL",
            &format!("startup recovery failed | {error}"),
        ),
    }
    if let Ok(counts) = journal.counts() {
        state.set_queued(counts.queued_xen11 as u64, counts.queued_xuni as u64);
    }
}

type Manager = SubmissionManager<Arc<JournalBridge>, HttpTransport>;

/// The submitter and every callback it reports through.
fn build_submitter(
    config: &ResolvedConfig,
    machine_id: &str,
    journal: Arc<dyn Journal + Send + Sync>,
    state: &Arc<MiningState>,
    logger: Option<Arc<FileLogger>>,
) -> Result<Arc<Manager>, reqwest::Error> {
    let transport = HttpTransport::new(
        &config.rpc_link,
        machine_id,
        SUBMIT_TIMEOUT_MS,
        GET_TIMEOUT_MS,
    )?;
    let bridge = Arc::new(JournalBridge::new(Arc::clone(&journal)));
    let submit_config = tm_submit::Config {
        margin: config.margin,
        ..tm_submit::Config::default()
    };

    let manager = Arc::new(SubmissionManager::with_config(
        bridge,
        transport,
        submit_config,
        None,
        None,
        None,
    ));

    // The drain thread's own unrecoverable-journal detection converges on the same fatal
    // state as the find sink's double-failure path. This only sets flags — it never joins,
    // which the fatal-callback contract forbids.
    let fatal_state = Arc::clone(state);
    manager.set_fatal_callback(Arc::new(move |reason: &str| {
        fatal_state.declare_fatal_durability_failure(&format!(
            "submission drain thread halted: {reason}"
        ));
    }));

    // The margin ramp publishes here; the mining loop picks it up at its next batch
    // boundary, because a margin change ends the current unit.
    let margin_state = Arc::clone(state);
    manager.set_margin_callback(Arc::new(move |kib: u32| {
        let previous = margin_state.set_margin_kib(kib);
        if previous != kib {
            Console::global().event(
                Level::Info,
                "MARGIN",
                &format!(
                    "{previous} -> {kib} | effective_m={} | headroom costs proportional \
                     hashrate",
                    margin_state.effective_difficulty()
                ),
            );
        }
    }));

    let outcome_state = Arc::clone(state);
    let outcome_journal = Arc::clone(&journal);
    manager.set_outcome_callback(Arc::new(
        move |record: &tm_core::FindRecord,
              classification: &tm_core::Classification,
              http_status: Option<i32>| {
            let (label, detail) = outcome_labels(classification);
            outcome_state.set_last_submission(last_submission_for(classification.next_status));
            if let Ok(counts) = outcome_journal.counts() {
                outcome_state.set_queued(counts.queued_xen11 as u64, counts.queued_xuni as u64);
            }
            let mut detail = detail;
            const MAX_DETAIL: usize = 64;
            if detail.len() > MAX_DETAIL {
                detail.truncate(MAX_DETAIL - 3);
                detail.push_str("...");
            }
            let http = http_status
                .map(|status| format!(" HTTP={status}"))
                .unwrap_or_default();
            let message = format!(
                "{label} [{}] id={}{http} - {detail}",
                record.payload.kind.as_str(),
                record.id
            );
            if let Some(logger) = &logger {
                let _ = logger.log(&message);
            }
            let (level, console_message) =
                outcome_console(classification, record.payload.kind, record.id, http_status, &detail);
            Console::global().event(level, "UPLINK", &console_message);
        },
    ));

    let network_state = Arc::clone(state);
    manager.set_network_state_callback(Arc::new(move |breaker: BreakerState| {
        network_state.set_network_state(BreakerStateLabel::from_breaker(breaker).dashboard_state());
    }));

    // A 401's embedded `m={N}` updates the difficulty immediately rather than waiting out a
    // poll interval, so the next batch is already mining at the cost the server wants.
    let hint_state = Arc::clone(state);
    manager.set_difficulty_hint_callback(Arc::new(move |difficulty: u32| {
        if hint_state.difficulty() != difficulty {
            hint_state.set_difficulty(difficulty);
            Console::global().event(
                Level::Info,
                "DIFFICULTY",
                &format!("updated from server hint | current={difficulty}"),
            );
        }
    }));

    manager.start();
    Ok(manager)
}

fn outcome_labels(classification: &tm_core::Classification) -> (&'static str, String) {
    use tm_core::FindStatus::*;
    match classification.next_status {
        Acked => ("UPLINK CONFIRMED", "server record verified".to_owned()),
        AcceptedUnconfirmed => ("UPLINK ACCEPTED", "confirmation pending".to_owned()),
        Pending => (
            "UPLINK RETRY",
            if classification.reason.starts_with("transport failure") {
                "network unavailable; retry scheduled".to_owned()
            } else {
                classification.reason.clone()
            },
        ),
        ParkedDifficulty | ParkedXuniWindow => ("UPLINK PARKED", classification.reason.clone()),
        Quarantined => ("UPLINK QUARANTINED", classification.reason.clone()),
        Dead | PermanentlyInvalid => ("UPLINK REJECTED", classification.reason.clone()),
        _ => ("UPLINK UPDATED", classification.reason.clone()),
    }
}

/// Console rendering for a submission outcome.
///
/// The C++ wrote uplink outcomes to the log file only, so an operator watching the console
/// saw every find appear and none of them get delivered — the one thing this miner exists to
/// guarantee. The file line is kept verbatim for log parsers; this adds the on-screen half.
fn outcome_console(
    classification: &tm_core::Classification,
    kind: tm_core::FindKind,
    id: i64,
    http_status: Option<i32>,
    detail: &str,
) -> (Level, String) {
    use tm_core::FindStatus::*;
    let (level, verb) = match classification.next_status {
        Acked => (Level::Ok, "confirmed"),
        AcceptedUnconfirmed => (Level::Info, "accepted"),
        Pending => (Level::Retry, "retrying"),
        ParkedDifficulty | ParkedXuniWindow => (Level::Park, "parked"),
        Quarantined => (Level::Error, "quarantined"),
        Dead | PermanentlyInvalid => (Level::Error, "rejected"),
        _ => (Level::Info, "updated"),
    };
    let http = http_status
        .map(|status| format!("  \u{2022}  HTTP={status}"))
        .unwrap_or_default();
    (
        level,
        format!(
            "#{id}  \u{2022}  {}  \u{2022}  {verb}{http}  \u{2022}  {detail}",
            kind.as_str()
        ),
    )
}

fn last_submission_for(status: tm_core::FindStatus) -> tm_dashboard::stats::LastSubmissionState {
    use tm_core::FindStatus::*;
    use tm_dashboard::stats::LastSubmissionState as L;
    match status {
        Acked => L::Accepted,
        AcceptedUnconfirmed => L::Unconfirmed,
        Pending => L::Retry,
        ParkedDifficulty | ParkedXuniWindow => L::Parked,
        Quarantined | Dead | PermanentlyInvalid => L::Failed,
        _ => L::None,
    }
}

/// Poll `/difficulty` forever, feeding both the mining loop and the submitter's unparking.
fn spawn_difficulty_poller(
    config: &ResolvedConfig,
    state: &Arc<MiningState>,
    submission: Option<&Arc<Manager>>,
) -> Option<std::thread::JoinHandle<()>> {
    let transport = HttpTransport::new(
        &config.rpc_link,
        "",
        crate::difficulty::DIFFICULTY_TIMEOUT_MS,
        crate::difficulty::DIFFICULTY_TIMEOUT_MS,
    )
    .ok()?;
    let mut poller = crate::difficulty::DifficultyPoller::new(
        transport,
        config.difficulty_cache_path.clone(),
        Arc::clone(state.difficulty_shared()),
    );
    if let Some(manager) = submission {
        // Every observation, changed or not: it is what un-parks finds whose `m` the
        // network has since caught up with, and what feeds the trend the margin ramp uses.
        let manager = Arc::clone(manager);
        poller = poller.with_observer(move |difficulty| {
            let _ = manager.observe_difficulty(difficulty);
        });
    }
    // The first poll happens on this thread, as in the C++, so a reachable server has
    // corrected the difficulty before the first batch is sized.
    poller.poll_once();

    let state = Arc::clone(state);
    std::thread::Builder::new()
        .name("treeminer-difficulty".into())
        .spawn(move || {
            while state.is_running() {
                sleep_while(&state, crate::difficulty::POLL_INTERVAL);
                if !state.is_running() {
                    break;
                }
                poller.poll_once();
            }
        })
        .ok()
}

fn mining_identity(config: &ResolvedConfig) -> MiningIdentity {
    MiningIdentity {
        user_address: config.miner_address.clone(),
        devfee_address: DEVFEE_ADDRESS.to_owned(),
        eco_devfee_address: config.eco_devfee_address.clone(),
        devfee_permillage: config.devfee_permillage,
        self_mining_prefix: String::new(),
        test_block_pattern: config.test_block_pattern.clone(),
    }
}

fn stats_publisher(
    config: &ResolvedConfig,
    machine_id: &str,
    console_url: &str,
    state: &Arc<MiningState>,
    journal: &Arc<dyn Journal + Send + Sync>,
    submission: Option<&Arc<Manager>>,
) -> StatsPublisher {
    let mut publisher = StatsPublisher::new(
        Arc::clone(state),
        StatsIdentity {
            machine_id: machine_id.to_owned(),
            miner_address: config.miner_address.clone(),
            custom_name: config.custom_name.clone(),
            margin_mode: config.margin.mode.as_str().to_owned(),
            console_url: console_url.to_owned(),
        },
    );
    let journal = Arc::clone(journal);
    publisher = publisher.with_journal(Arc::new(move || journal.counts().ok()));
    if let Some(manager) = submission {
        let manager = Arc::clone(manager);
        publisher = publisher.with_submission(Arc::new(move || {
            Some(SubmissionView {
                metrics: manager.metrics(),
                breaker: BreakerStateLabel::from_breaker(manager.breaker_state()),
                margin_kib: manager.margin_in_effect(),
                outage_ms: manager.outage_duration_ms(),
                last_outage_span_ms: manager.last_outage_span_ms(),
                drain_rate_per_second: manager.drain_rate_per_second(),
                last_observed_difficulty: manager.last_observed_difficulty(),
            })
        }));
    }
    publisher
}

/// Start the alternate-screen UI, if that is the resolved mode, and route log output into
/// its event pane so nothing writes to the tty behind its back.
fn start_display(
    config: &ResolvedConfig,
    publisher: &Arc<StatsPublisher>,
) -> Option<Arc<TerminalUi>> {
    if config.display_mode != DisplayMode::Terminal {
        return None;
    }
    let provider = Arc::clone(publisher);
    let ui = TerminalUi::new(Arc::new(move || provider.miner_snapshot()));
    if ui.start().is_err() {
        return None;
    }
    let ui = Arc::new(ui);
    let forward = Arc::clone(&ui);
    Console::global().set_event_forwarder(Arc::new(move |_level, component, message| {
        forward.post_event(&format!("{component}  {message}"));
    }));
    let sink = Arc::clone(&ui);
    Console::global().set_line_sink(Arc::new(move |message| sink.post_event(message)));
    Some(ui)
}

fn start_cpu_workers(
    config: &ResolvedConfig,
    state: &Arc<MiningState>,
    sink: &Arc<FindSink>,
    identity: &MiningIdentity,
) -> Option<CpuMiningWorker> {
    if config.cpu_worker_count == 0 {
        return None;
    }
    if config.cpu_max_difficulty > 0 {
        println!(
            "CPU workers hash only at difficulty <= {} (idle above; auto-resume)",
            config.cpu_max_difficulty
        );
    }
    let worker = CpuMiningWorker::new(
        CpuWorkerConfig {
            worker_count: config.cpu_worker_count,
            batch_size: DEFAULT_CPU_BATCH_SIZE,
            max_difficulty: config.cpu_max_difficulty,
        },
        Arc::clone(state),
        Arc::clone(sink),
        identity.clone(),
    );
    match worker {
        Ok(worker) => {
            worker.start();
            Some(worker)
        }
        Err(error) => {
            Console::global().event(Level::Error, "CPU", &error.to_string());
            None
        }
    }
}

/// One thread per device per stream, exactly as the C++ builds them.
fn spawn_mining_threads(
    devices: &[i32],
    config: &ResolvedConfig,
    deps: &Arc<MineDeps>,
    host: &Arc<tm_argon2::CpuArgon2Host>,
) -> Vec<std::thread::JoinHandle<()>> {
    let mut threads = Vec::new();
    for &index in devices {
        for stream in 0..config.gpu_streams_per_device {
            let backend = match GpuMiningBackend::open(index, host.clone()) {
                Ok(backend) => backend,
                Err(error) => {
                    Console::global().event(
                        Level::Error,
                        "MINING",
                        &format!("device #{index} could not be opened | {error}"),
                    );
                    continue;
                }
            };
            let deps = Arc::clone(deps);
            let stream = stream as i32;
            if let Ok(handle) = std::thread::Builder::new()
                .name(format!("treeminer-gpu-{index}-{stream}"))
                .spawn(move || {
                    let mut backend = backend;
                    run_mining_on_device(&mut backend, &deps, stream, DEFAULT_BACKOFF);
                })
            {
                threads.push(handle);
            }
        }
    }
    threads
}

/// SIGINT and SIGTERM only flip the shutdown flag. Everything that has to be torn down is
/// torn down on the main thread afterwards — doing it from a handler is what corrupted the
/// heap in the C++.
fn install_signal_handlers(runtime: &tokio::runtime::Runtime, shutdown: Arc<Shutdown>) {
    runtime.spawn(async move {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{signal, SignalKind};
            let (Ok(mut interrupt), Ok(mut terminate)) = (
                signal(SignalKind::interrupt()),
                signal(SignalKind::terminate()),
            ) else {
                return;
            };
            tokio::select! {
                _unused = interrupt.recv() => {}
                _unused = terminate.recv() => {}
            }
        }
        #[cfg(not(unix))]
        {
            let _unused = tokio::signal::ctrl_c().await;
        }
        shutdown.request_stop();
    });
}

/// Bind the console, print the banner it can actually be reached on, and serve until the
/// returned sender fires.
fn start_dashboard(
    runtime: &tokio::runtime::Runtime,
    config: &ResolvedConfig,
    publisher: Arc<StatsPublisher>,
    logger: &Option<Arc<FileLogger>>,
) -> Option<tokio::sync::oneshot::Sender<()>> {
    let dashboard_config =
        DashboardConfig::new(config.dashboard_bind.clone(), config.dashboard_port);
    let server = runtime.block_on(DashboardServer::bind(dashboard_config, publisher));
    let server = match server {
        Ok(server) => server,
        Err(error) => {
            Console::global().event(
                Level::Error,
                "CONSOLE",
                &format!("dashboard unavailable | {error} — mining continues"),
            );
            return None;
        }
    };

    // Through the console, not `print!`: when the TUI owns the screen this has to land in
    // its event pane instead of being spliced into a frame.
    let ready = server.ready_message();
    if let Some(logger) = logger {
        let _ = logger.log(&ready);
    }
    tm_tui::log_to_console(&ready);
    // Written so a rig manager can read back the URL without scraping the log.
    let _ = std::fs::write("dashboard.url", &ready);

    let (tx, rx) = tokio::sync::oneshot::channel();
    runtime.spawn(async move {
        let _unused = server
            .serve_with_shutdown(async move {
                let _unused = rx.await;
            })
            .await;
    });
    Some(tx)
}

/// The main thread's loop: publish stats until something asks us to stop.
fn report_loop(
    state: &Arc<MiningState>,
    publisher: &Arc<StatsPublisher>,
    cpu: &Option<CpuMiningWorker>,
    devices: &[i32],
    tui_active: bool,
) {
    let telemetry_devices: Vec<i32> = devices.to_vec();
    let telemetry_state = Arc::clone(state);
    // ROCm SMI is not documented as thread safe, so exactly one thread ever holds a session.
    let telemetry = std::thread::Builder::new()
        .name("treeminer-telemetry".into())
        .spawn(move || {
            let session = tm_gpu::TelemetrySession::new();
            while telemetry_state.is_running() {
                for &index in &telemetry_devices {
                    let reading = session.query(index, -1);
                    telemetry_state.publish_telemetry(
                        index,
                        tm_dashboard::stats::GpuTelemetry {
                            power_milliwatts: reading.power_milliwatts,
                            utilization_percent: reading.utilization_percent,
                        },
                    );
                }
                sleep_while(&telemetry_state, TELEMETRY_INTERVAL);
            }
        })
        .ok();

    while state.is_running() {
        if let Some(cpu) = cpu {
            cpu.publish();
        }
        if !tui_active {
            Console::global().progress(&tm_tui::format_ticker(
                &publisher.ticker_snapshot(),
                Console::global().color_enabled(),
            ));
        }
        sleep_while(state, REPORT_INTERVAL);
    }

    if let Some(handle) = telemetry {
        let _ = handle.join();
    }
}

/// Sleep in slices so shutdown is never delayed by a full interval.
fn sleep_while(state: &Arc<MiningState>, total: Duration) {
    let deadline = std::time::Instant::now() + total;
    while state.is_running() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100).min(total));
    }
}

/// Join everything, hand back the terminal, and translate a durability failure into an exit
/// code a supervisor can act on.
fn finish(
    state: &Arc<MiningState>,
    mining_threads: Vec<std::thread::JoinHandle<()>>,
    cpu: Option<CpuMiningWorker>,
    terminal_ui: Option<Arc<TerminalUi>>,
    poller: Option<std::thread::JoinHandle<()>>,
    submission: Option<Arc<Manager>>,
) -> u8 {
    state.shutdown().request_stop();

    if let Some(cpu) = &cpu {
        cpu.stop();
        cpu.join();
    }
    for thread in mining_threads {
        let _ = thread.join();
    }
    if let Some(poller) = poller {
        let _ = poller.join();
    }

    // The terminal comes back before anything else prints, or the epilogue lands inside the
    // alternate screen and is lost with it.
    if let Some(ui) = terminal_ui {
        Console::global().clear_event_forwarder();
        Console::global().clear_line_sink();
        ui.stop();
    }

    if let Some(manager) = submission {
        manager.stop();
    }

    if state.fatal_durability_failure() {
        eprintln!(
            "FATAL: durability failure — {} — exiting nonzero for supervisor restart",
            state.fatal_durability_reason()
        );
        return EXIT_DURABILITY;
    }
    println!();
    EXIT_OK
}

#[cfg(test)]
mod tests {
    use super::*;
    use tm_core::{Classification, FindKind, FindStatus};

    fn classification(next_status: FindStatus, reason: &str) -> Classification {
        Classification {
            next_status,
            server_difficulty_hint: None,
            needs_lookup_confirmation: false,
            reason: reason.to_owned(),
        }
    }

    #[test]
    fn a_confirmed_upload_reaches_the_console_as_an_ok_event() {
        let (level, message) = outcome_console(
            &classification(FindStatus::Acked, "server record verified"),
            FindKind::Xen11,
            376,
            Some(200),
            "server record verified",
        );
        assert_eq!(level, Level::Ok);
        assert!(message.contains("#376"), "{message}");
        assert!(message.contains("XEN11"), "{message}");
        assert!(message.contains("confirmed"), "{message}");
        assert!(message.contains("HTTP=200"), "{message}");
    }

    #[test]
    fn every_outcome_has_a_console_level_matching_its_severity() {
        let cases = [
            (FindStatus::Acked, Level::Ok, "confirmed"),
            (FindStatus::AcceptedUnconfirmed, Level::Info, "accepted"),
            (FindStatus::Pending, Level::Retry, "retrying"),
            (FindStatus::ParkedDifficulty, Level::Park, "parked"),
            (FindStatus::ParkedXuniWindow, Level::Park, "parked"),
            (FindStatus::Quarantined, Level::Error, "quarantined"),
            (FindStatus::Dead, Level::Error, "rejected"),
            (FindStatus::PermanentlyInvalid, Level::Error, "rejected"),
        ];
        for (status, expected_level, expected_verb) in cases {
            let (level, message) = outcome_console(
                &classification(status, "reason"),
                FindKind::Xuni,
                7,
                None,
                "reason",
            );
            assert_eq!(level, expected_level, "{status:?}");
            assert!(message.contains(expected_verb), "{status:?}: {message}");
            assert!(!message.contains("HTTP="), "no status means no HTTP segment");
        }
    }

    /// The log file is what operators grep after the fact; its wording must not drift.
    #[test]
    fn the_log_line_labels_stay_as_the_cpp_wrote_them() {
        let (label, detail) = outcome_labels(&classification(FindStatus::Acked, "ignored"));
        assert_eq!(label, "UPLINK CONFIRMED");
        assert_eq!(detail, "server record verified");
        let (label, _) = outcome_labels(&classification(FindStatus::ParkedXuniWindow, "window"));
        assert_eq!(label, "UPLINK PARKED");
    }
}
