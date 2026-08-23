//! Route behaviour against a live server on an ephemeral port, driven by a fake stats
//! source. Field names are asserted literally: they are the contract third-party fleet
//! dashboards and the embedded page depend on.

use std::net::IpAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde_json::Value;
use tm_dashboard::{
    DashboardConfig, DashboardServer, GpuStat, GpuTelemetry, JournalCounts, LastSubmissionState,
    NetworkState, PlatformLease, PlatformStatus, StaticInterfaces, StatsSnapshot, StatsSource,
    SubmissionMetrics, TreeminerStats, POWER_UNAVAILABLE,
};

/// Counts reads so a test can prove no route mutates anything the source owns.
struct FakeStats {
    snapshot: StatsSnapshot,
    reads: AtomicUsize,
}

impl FakeStats {
    fn new(snapshot: StatsSnapshot) -> Self {
        Self {
            snapshot,
            reads: AtomicUsize::new(0),
        }
    }
}

impl StatsSource for FakeStats {
    fn snapshot(&self) -> StatsSnapshot {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.snapshot.clone()
    }
}

fn gpu(index: i32) -> GpuStat {
    GpuStat {
        index,
        bus_id: 10 + index,
        name: format!("Radeon RX 7900 XTX #{index}"),
        memory: 24,
        using_memory: 0.4231,
        temperature: 61,
        hashrate: 1234.5,
        power: String::new(),
        hash_count: 9_000 + index as u64,
        stream_index: 0,
        telemetry: Some(GpuTelemetry {
            power_milliwatts: Some(310_000),
            utilization_percent: Some(97),
        }),
        updated_secs_ago: 3,
    }
}

fn full_snapshot() -> StatsSnapshot {
    StatsSnapshot {
        machine_id: "machine-1".into(),
        miner_address: "0x1234567890abcdef1234567890abcdef12345678".into(),
        custom_name: "rig-a".into(),
        version: String::new(),
        uptime_seconds: 3600,
        running: true,
        difficulty: 60_000,
        cpu_workers: 4,
        cpu_hashrate: 12.5,
        gpus: vec![gpu(0), gpu(1)],
        normal_blocks: 7,
        super_blocks: 2,
        xuni_blocks: 5,
        failed_blocks: 1,
        network_state: NetworkState::Open,
        last_submission: LastSubmissionState::Retry,
        queued_xnm: 3,
        queued_xuni: 2,
        fatal_durability_failure: false,
        fatal_durability_reason: String::new(),
        treeminer: Some(TreeminerStats {
            difficulty: 60_000,
            margin_in_effect: 512,
            effective_difficulty: 60_512,
            margin_mode: "adaptive".into(),
            breaker_state: "open".into(),
            outage_ms: 45_000,
            drain_rate_per_second: 1.5,
            pending: 4,
            parked: 1,
            quarantined: 0,
            acked_total: 20,
            dead_total: 1,
            accepted_unconfirmed: 2,
            permanently_invalid: 1,
        }),
        journal: Some(JournalCounts {
            pending: 4,
            accepted_unconfirmed: 2,
            parked: 1,
            parked_difficulty: 1,
            parked_xuni: 0,
            quarantined: 0,
            acked_total: 20,
            dead_total: 1,
            permanently_invalid: 1,
        }),
        submission: Some(SubmissionMetrics {
            last_observed_difficulty: Some(60_000),
            margin_in_effect: 512,
            margin_changes: 3,
            outage_duration_ms: 45_000,
            submitted: 25,
            resubmitted: 4,
            acked: 20,
            accepted_unconfirmed: 2,
            transport_failures: 2,
            parked_difficulty: 1,
            parked_xuni: 0,
            quarantined: 0,
            permanently_invalid: 1,
            confirmation_retries: 6,
            reconciled_via_get_block: 3,
            lying_200_detected: 1,
            probes: 9,
        }),
        platform: None,
    }
}

struct Harness {
    base: String,
    client: reqwest::Client,
    stats: Arc<FakeStats>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
    joined: Option<tokio::task::JoinHandle<()>>,
}

impl Harness {
    async fn start(snapshot: StatsSnapshot) -> Self {
        Self::start_with(snapshot, DashboardConfig::default()).await
    }

    async fn start_with(snapshot: StatsSnapshot, mut config: DashboardConfig) -> Self {
        // Ephemeral port on loopback: tests must not open a LAN port.
        config.bind = "127.0.0.1".into();
        config.port = 0;
        config.interfaces = Arc::new(StaticInterfaces(vec![
            "192.168.1.5".parse::<IpAddr>().expect("test address")
        ]));

        let stats = Arc::new(FakeStats::new(snapshot));
        let server = DashboardServer::bind(config, stats.clone())
            .await
            .expect("bind ephemeral port");
        let base = format!("http://{}", server.local_addr());
        let (tx, rx) = tokio::sync::oneshot::channel();
        let joined = tokio::spawn(async move {
            let _ = server
                .serve_with_shutdown(async {
                    let _ = rx.await;
                })
                .await;
        });
        Self {
            base,
            client: reqwest::Client::new(),
            stats,
            shutdown: Some(tx),
            joined: Some(joined),
        }
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        self.client
            .get(format!("{}{path}", self.base))
            .send()
            .await
            .expect("request")
    }

    async fn json(&self, path: &str) -> Value {
        let response = self.get(path).await;
        assert_eq!(response.status(), 200, "GET {path}");
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok()),
            Some("application/json"),
            "GET {path} content type"
        );
        assert_eq!(
            response
                .headers()
                .get("cache-control")
                .and_then(|v| v.to_str().ok()),
            Some("no-store"),
            "GET {path} must not be cached by a browser"
        );
        response.json().await.expect("json body")
    }

    async fn stop(mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        if let Some(handle) = self.joined.take() {
            let _ = handle.await;
        }
    }
}

#[tokio::test]
async fn healthz_reports_ok() {
    let harness = Harness::start(full_snapshot()).await;
    let body = harness.json("/healthz").await;
    assert_eq!(body, serde_json::json!({"ok": true}));
    harness.stop().await;
}

#[tokio::test]
async fn stats_keeps_the_cpp_field_names() {
    let harness = Harness::start(full_snapshot()).await;
    let body = harness.json("/stats").await;

    assert_eq!(body["totalHashrate"].as_f64().expect("f64").round(), 2469.0);
    assert_eq!(body["uptime"], 3600);
    assert_eq!(body["acceptedBlocks"], 9);
    assert_eq!(body["rejectedBlocks"], 1);
    assert_eq!(body["fatalDurabilityFailure"], false);
    assert!(body.get("fatalDurabilityReason").is_none());
    assert_eq!(body["stats_cache_seconds"], 2);

    let gpus = body["gpus"].as_array().expect("gpus array");
    assert_eq!(gpus.len(), 2);
    assert_eq!(gpus[0]["index"], 0);
    assert_eq!(gpus[0]["busId"], 10);
    assert!(gpus[0]["hashrate"].is_number());

    assert_eq!(body["difficulty"], 60_000);
    assert_eq!(body["marginInEffect"], 512);
    assert_eq!(body["effectiveDifficulty"], 60_512);
    assert_eq!(body["marginMode"], "adaptive");
    assert_eq!(body["serverState"], "open");
    assert_eq!(body["outageMs"], 45_000);
    assert_eq!(body["drainRatePerSecond"], 1.5);
    assert!(body["rawHashrate"].is_number());
    // 20 acked of 22 resolved (acked + dead + permanently invalid).
    assert!((body["acceptedYieldRatio"].as_f64().expect("ratio") - 20.0 / 22.0).abs() < 1e-9);
    assert!(body["acceptedYieldHashrate"].is_number());

    assert_eq!(body["difficultyStats"]["last_observed"], 60_000);
    assert_eq!(body["difficultyStats"]["margin_in_effect"], 512);
    assert_eq!(body["difficultyStats"]["effective_mining_difficulty"], 60_512);
    assert_eq!(body["difficultyStats"]["margin_changes_total"], 3);
    assert_eq!(body["pool"]["outage_duration_ms"], 45_000);

    let submissions = &body["submissions"];
    assert_eq!(submissions["attempts_total"], 25);
    assert_eq!(submissions["resubmissions_total"], 4);
    assert_eq!(submissions["acked_total"], 20);
    assert_eq!(submissions["accepted_unconfirmed_total"], 2);
    assert_eq!(submissions["transport_failures_total"], 2);
    assert_eq!(submissions["difficulty_rejections_total"], 1);
    assert_eq!(submissions["xuni_window_rejections_total"], 0);
    assert_eq!(submissions["quarantined_total"], 0);
    assert_eq!(submissions["permanently_invalid_total"], 1);
    assert_eq!(submissions["confirmation_retries_total"], 6);
    assert_eq!(submissions["confirmed_via_lookup_total"], 3);
    assert_eq!(submissions["lying_200_total"], 1);
    assert_eq!(submissions["difficulty_probes_total"], 9);
    assert_eq!(submissions["failed_attempts_total"], 4);
    assert_eq!(submissions["failure_rate_pct"], 16.0);

    // The durable journal block wins over the in-memory one, with snake_case names.
    let journal = &body["journal"];
    assert_eq!(journal["pending"], 4);
    assert_eq!(journal["accepted_unconfirmed"], 2);
    assert_eq!(journal["parked_total"], 1);
    assert_eq!(journal["parked_difficulty"], 1);
    assert_eq!(journal["parked_xuni"], 0);
    assert_eq!(journal["quarantined"], 0);
    assert_eq!(journal["acked_total"], 20);
    assert_eq!(journal["dead_total"], 1);
    assert_eq!(journal["permanently_invalid"], 1);
    assert!(journal.get("acceptedUnconfirmed").is_none());

    assert_eq!(body["console"]["bind"], "127.0.0.1");
    // The advertised URL names the port actually bound, not the requested one.
    assert_eq!(body["console"]["open"], harness.base);
    harness.stop().await;
}

#[tokio::test]
async fn api_v1_status_is_the_same_body_as_stats() {
    let harness = Harness::start(full_snapshot()).await;
    let stats = harness.json("/stats").await;
    let status = harness.json("/api/v1/status").await;
    assert_eq!(stats, status);
    harness.stop().await;
}

#[tokio::test]
async fn stats_without_a_submission_layer_still_answers() {
    let mut snapshot = full_snapshot();
    snapshot.treeminer = None;
    snapshot.journal = None;
    snapshot.submission = None;
    snapshot.fatal_durability_failure = true;
    snapshot.fatal_durability_reason = "journal write failed: disk full".into();

    let harness = Harness::start(snapshot).await;
    let body = harness.json("/stats").await;
    assert_eq!(body["fatalDurabilityFailure"], true);
    assert_eq!(
        body["fatalDurabilityReason"],
        "journal write failed: disk full"
    );
    assert!(body.get("submissions").is_none());
    assert!(body.get("journal").is_none());
    assert!(body["totalHashrate"].is_number());
    harness.stop().await;
}

#[tokio::test]
async fn accepted_yield_is_null_until_a_find_resolves() {
    let mut snapshot = full_snapshot();
    if let Some(tm) = snapshot.treeminer.as_mut() {
        tm.acked_total = 0;
        tm.dead_total = 0;
        tm.permanently_invalid = 0;
    }
    let harness = Harness::start(snapshot).await;
    let body = harness.json("/stats").await;
    assert!(body["acceptedYieldRatio"].is_null());
    assert!(body["acceptedYieldHashrate"].is_null());
    harness.stop().await;
}

#[tokio::test]
async fn rig_serves_what_the_embedded_page_reads() {
    let harness = Harness::start(full_snapshot()).await;
    let body = harness.json("/api/rig").await;

    assert_eq!(body["identity"]["name"], "rig-a");
    assert_eq!(body["identity"]["machine_id"], "machine-1");
    assert_eq!(
        body["identity"]["address"],
        "0x1234567890abcdef1234567890abcdef12345678"
    );

    let engine = &body["engine"];
    assert_eq!(engine["running"], true);
    assert_eq!(engine["fatal_durability_failure"], false);
    assert_eq!(engine["uptime_seconds"], 3600);
    assert_eq!(engine["difficulty"], 60_000);
    assert_eq!(engine["gpu_devices"], 2);
    assert_eq!(engine["cuda_streams"], 2);
    assert_eq!(engine["cpu_workers"], 4);
    assert_eq!(engine["cpu_hashrate"], 12.5);
    assert!(engine["gpu_hashrate"].is_number());
    assert!(engine["total_hashrate"].is_number());

    assert_eq!(body["finds"]["xnm"], 9);
    assert_eq!(body["finds"]["xuni"], 5);
    assert_eq!(body["finds"]["super"], 2);
    assert_eq!(body["finds"]["rejected"], 1);

    assert_eq!(body["delivery"]["network"], "offline");
    assert_eq!(body["delivery"]["last_submission"], "retrying");
    assert_eq!(body["delivery"]["queued_xnm"], 3);
    assert_eq!(body["delivery"]["queued_xuni"], 2);
    assert_eq!(body["delivery"]["queued_total"], 5);

    assert_eq!(body["console"]["bind"], "127.0.0.1");
    assert_eq!(body["console"]["urls"][0], "127.0.0.1");

    let gpus = body["gpus"].as_array().expect("gpus array");
    assert_eq!(gpus.len(), 2);
    assert_eq!(gpus[0]["index"], 0);
    assert_eq!(gpus[0]["stream"], 0);
    assert_eq!(gpus[0]["name"], "Radeon RX 7900 XTX #0");
    assert_eq!(gpus[0]["memory_gib"], 24);
    assert!((gpus[0]["memory_used_percent"].as_f64().expect("percent") - 42.31).abs() < 0.01);
    assert_eq!(gpus[0]["hash_count"], 9000);
    harness.stop().await;
}

#[tokio::test]
async fn rig_drops_gpu_entries_that_stopped_reporting() {
    let mut snapshot = full_snapshot();
    snapshot.gpus[1].updated_secs_ago = 121;
    let harness = Harness::start(snapshot).await;
    let body = harness.json("/api/rig").await;
    let gpus = body["gpus"].as_array().expect("gpus array");
    assert_eq!(gpus.len(), 1);
    assert_eq!(body["engine"]["gpu_devices"], 1);
    // `/stats` is unfiltered, as in the C++.
    let stats = harness.json("/stats").await;
    assert_eq!(stats["gpus"].as_array().expect("gpus").len(), 2);
    harness.stop().await;
}

#[tokio::test]
async fn a_rig_with_no_gpus_still_renders_valid_json() {
    let snapshot = StatsSnapshot {
        machine_id: "bare".into(),
        ..StatsSnapshot::default()
    };
    let harness = Harness::start(snapshot).await;

    let rig = harness.json("/api/rig").await;
    assert_eq!(rig["gpus"], serde_json::json!([]));
    assert_eq!(rig["engine"]["gpu_devices"], 0);
    assert_eq!(rig["engine"]["cuda_streams"], 0);
    assert_eq!(rig["engine"]["gpu_hashrate"], 0.0);
    assert_eq!(rig["engine"]["total_hashrate"], 0.0);
    assert_eq!(rig["identity"]["name"], "TreeMiner");
    assert_eq!(rig["delivery"]["network"], "online");
    assert_eq!(rig["delivery"]["last_submission"], "none");

    let stats = harness.json("/stats").await;
    assert_eq!(stats["gpus"], serde_json::json!([]));
    assert_eq!(stats["totalHashrate"], 0.0);
    assert_eq!(stats["acceptedBlocks"], 0);
    harness.stop().await;
}

#[tokio::test]
async fn platform_status_reports_disabled_without_a_platform_manager() {
    let harness = Harness::start(full_snapshot()).await;
    let body = harness.json("/platform/status").await;
    assert_eq!(body["platform_mode"], false);
    assert_eq!(body["mining_mode"], "self");
    assert_eq!(body["platform_state"], "disabled");
    assert_eq!(body["running"], false);
    assert!(body.get("lease_id").is_none());
    harness.stop().await;
}

#[tokio::test]
async fn platform_status_reports_the_lease_when_leased() {
    let mut snapshot = full_snapshot();
    snapshot.platform = Some(PlatformStatus {
        platform_mode: true,
        mining_mode: "platform".into(),
        platform_state: "leased".into(),
        running: true,
        lease: Some(PlatformLease {
            lease_id: "lease-7".into(),
            consumer_id: "consumer-3".into(),
            consumer_address: "0xabc".into(),
            blocks_found: 2,
            remaining_sec: 900,
        }),
    });
    let harness = Harness::start(snapshot).await;
    let body = harness.json("/platform/status").await;
    assert_eq!(body["platform_mode"], true);
    assert_eq!(body["mining_mode"], "platform");
    assert_eq!(body["platform_state"], "leased");
    assert_eq!(body["lease_id"], "lease-7");
    assert_eq!(body["consumer_id"], "consumer-3");
    assert_eq!(body["consumer_address"], "0xabc");
    assert_eq!(body["blocks_found"], 2);
    assert_eq!(body["remaining_sec"], 900);
    harness.stop().await;
}

#[tokio::test]
async fn index_serves_the_embedded_page() {
    let harness = Harness::start(full_snapshot()).await;
    let response = harness.get("/").await;
    assert_eq!(response.status(), 200);
    assert_eq!(
        response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/html; charset=utf-8")
    );
    let body = response.text().await.expect("body");
    assert!(body.starts_with("<!doctype html>"));
    assert!(body.contains("fetch('/api/rig'"));
    harness.stop().await;
}

#[tokio::test]
async fn hashfield_asset_is_served_when_present_and_404s_when_not() {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = DashboardConfig {
        asset_root: dir.path().to_path_buf(),
        ..DashboardConfig::default()
    };

    let harness = Harness::start_with(full_snapshot(), config.clone()).await;
    let missing = harness.get("/assets/hashfield.webp").await;
    assert_eq!(missing.status(), 404);
    assert_eq!(missing.text().await.expect("body"), "asset unavailable");
    harness.stop().await;

    std::fs::create_dir_all(dir.path().join("res/dashboard")).expect("asset dir");
    std::fs::write(dir.path().join(tm_dashboard::HASHFIELD_ASSET), b"RIFFwebp")
        .expect("write asset");
    let harness = Harness::start_with(full_snapshot(), config).await;
    let present = harness.get("/assets/hashfield.webp").await;
    assert_eq!(present.status(), 200);
    assert_eq!(
        present
            .headers()
            .get("cache-control")
            .and_then(|v| v.to_str().ok()),
        Some("public, max-age=86400")
    );
    assert_eq!(present.bytes().await.expect("bytes").as_ref(), b"RIFFwebp");
    harness.stop().await;
}

#[tokio::test]
async fn every_route_is_read_only() {
    let harness = Harness::start(full_snapshot()).await;
    let before = harness.stats.snapshot.clone();

    for path in [
        "/",
        "/healthz",
        "/stats",
        "/api/v1/status",
        "/api/rig",
        "/platform/status",
        "/assets/hashfield.webp",
    ] {
        for method in [
            reqwest::Method::POST,
            reqwest::Method::PUT,
            reqwest::Method::DELETE,
            reqwest::Method::PATCH,
        ] {
            let response = harness
                .client
                .request(method.clone(), format!("{}{path}", harness.base))
                .send()
                .await
                .expect("request");
            assert_eq!(
                response.status(),
                405,
                "{method} {path} must not be routed"
            );
        }
    }

    // Unknown paths are not silently handled either.
    assert_eq!(harness.get("/config").await.status(), 404);
    assert_eq!(harness.get("/api/v1/submit").await.status(), 404);

    assert_eq!(harness.stats.snapshot, before, "a GET mutated the source");
    harness.stop().await;
}

#[tokio::test]
async fn no_route_leaks_more_than_the_public_address() {
    let mut snapshot = full_snapshot();
    snapshot.miner_address = "0x1234567890abcdef1234567890abcdef12345678".into();
    let harness = Harness::start(snapshot).await;

    // JSON routes only: the static page's own copy says "no secrets included", which is
    // a claim about the payloads below rather than a leak.
    for path in ["/healthz", "/stats", "/api/v1/status", "/api/rig", "/platform/status"] {
        let body = harness.get(path).await.text().await.expect("body");
        let lowered = body.to_lowercase();
        for secret in ["private", "secret", "seed", "token", "password", "mnemonic"] {
            assert!(!lowered.contains(secret), "{path} mentions {secret}");
        }
    }
    harness.stop().await;
}

#[tokio::test]
async fn stats_is_cached_while_rig_reads_live() {
    let harness = Harness::start(full_snapshot()).await;
    let _ = harness.json("/stats").await;
    let after_first = harness.stats.reads.load(Ordering::Relaxed);
    let _ = harness.json("/stats").await;
    let _ = harness.json("/api/v1/status").await;
    assert_eq!(
        harness.stats.reads.load(Ordering::Relaxed),
        after_first,
        "/stats must serve the 2s cache"
    );

    let _ = harness.json("/api/rig").await;
    assert_eq!(harness.stats.reads.load(Ordering::Relaxed), after_first + 1);
    harness.stop().await;
}

#[tokio::test]
async fn ready_message_reports_the_port_actually_bound() {
    let config = DashboardConfig {
        interfaces: Arc::new(StaticInterfaces(vec![])),
        ..DashboardConfig::new("127.0.0.1", 0)
    };
    let server = DashboardServer::bind(config, Arc::new(FakeStats::new(full_snapshot())))
        .await
        .expect("bind");
    let port = server.local_addr().port();
    assert_ne!(port, 0);
    assert_eq!(
        server.ready_message(),
        format!("Dashboard ready — open http://127.0.0.1:{port} (this machine only)\n")
    );
}

#[tokio::test]
async fn a_hostname_bind_is_rejected_rather_than_guessed() {
    let result = DashboardServer::bind(
        DashboardConfig::new("localhost", 0),
        Arc::new(FakeStats::new(full_snapshot())),
    )
    .await;
    match result {
        Err(error) => assert!(error.to_string().contains("localhost")),
        Ok(_) => panic!("a hostname bind must be rejected, not resolved"),
    }
}

#[test]
fn upload_payload_reports_the_sentinel_when_telemetry_is_unavailable() {
    let mut snapshot = full_snapshot();
    snapshot.gpus[1].telemetry = None;
    let body = tm_dashboard::stat_upload_payload(&snapshot);

    assert_eq!(body["gpus"][0]["power"], 310_000);
    assert_eq!(body["gpus"][0]["utiliz"], 97);
    assert_eq!(body["gpus"][1]["power"], POWER_UNAVAILABLE);
    // Utilisation has no sentinel in the C++: an unavailable reading is reported as 0.
    assert_eq!(body["gpus"][1]["utiliz"], 0);
    // Only GPUs that answered contribute to the total.
    assert_eq!(body["totalPower"], 310_000);
    // Formatted-string fields stay strings.
    assert_eq!(body["gpus"][0]["hashrate"], "1234.50");
    assert_eq!(body["gpus"][0]["usingMemory"], "42.3");
    assert_eq!(body["totalHashrate"], "2469.00");
    assert_eq!(body["version"], "2.0.0");
    assert_eq!(body["customName"], "rig-a");
    assert_eq!(body["minerAddr"], snapshot.miner_address);
}
