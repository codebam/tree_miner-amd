//! JSON payloads. Port of `src/StatReporter.cpp` plus `buildStatsSnapshot` from
//! `src/LocalServer.cpp`.
//!
//! Field names are the compatibility surface: the embedded page and third-party fleet
//! dashboards read them, so they are spelled out literally here rather than derived from
//! Rust identifiers. Values keep the C++ types too — several are formatted strings, not
//! numbers, and changing that would break consumers just as badly as a rename.

use serde_json::{json, Map, Value};

use crate::stats::StatsSnapshot;
use crate::url::{advertised_addresses, console_url, InterfaceSource};

/// Seconds `/stats` and `/api/v1/status` may serve a cached body for.
pub const STATS_CACHE_SECONDS: u64 = 2;

/// Version string the woodyminer upload payload reports when the snapshot leaves it unset.
pub const REPORTED_VERSION: &str = "2.0.0";

fn console_block(bind: &str, port: u16, interfaces: &dyn InterfaceSource) -> Value {
    json!({
        "bind": bind,
        "open": console_url(bind, port, interfaces),
        "urls": advertised_addresses(bind, interfaces),
    })
}

/// `getGpuStatsJson()` — the base of the `/stats` body.
fn gpu_stats_json(snapshot: &StatsSnapshot) -> Value {
    let total_hashrate = snapshot.total_gpu_hashrate();
    let gpus: Vec<Value> = snapshot
        .gpus
        .iter()
        .map(|gpu| {
            json!({
                "index": gpu.index,
                "hashrate": gpu.hashrate,
                "busId": gpu.bus_id,
            })
        })
        .collect();

    let mut result = Map::new();
    result.insert("totalHashrate".into(), json!(total_hashrate));
    result.insert("gpus".into(), Value::Array(gpus));
    result.insert("uptime".into(), json!(snapshot.uptime_seconds));
    result.insert("acceptedBlocks".into(), json!(snapshot.accepted_blocks()));
    result.insert("rejectedBlocks".into(), json!(snapshot.failed_blocks));

    if let Some(tm) = &snapshot.treeminer {
        result.insert("difficulty".into(), json!(tm.difficulty));
        result.insert("marginInEffect".into(), json!(tm.margin_in_effect));
        result.insert("effectiveDifficulty".into(), json!(tm.effective_difficulty));
        result.insert("marginMode".into(), json!(tm.margin_mode));
        result.insert("serverState".into(), json!(tm.breaker_state));
        result.insert("outageMs".into(), json!(tm.outage_ms));
        result.insert(
            "drainRatePerSecond".into(),
            json!(tm.drain_rate_per_second),
        );
        result.insert(
            "journal".into(),
            json!({
                "pending": tm.pending,
                "parked": tm.parked,
                "quarantined": tm.quarantined,
                "acked": tm.acked_total,
                "dead": tm.dead_total,
                "acceptedUnconfirmed": tm.accepted_unconfirmed,
                "permanentlyInvalid": tm.permanently_invalid,
            }),
        );
        result.insert("rawHashrate".into(), json!(total_hashrate));

        // Resolved = reached a terminal state. In-flight finds are excluded rather than
        // counted as losses, so the ratio does not dip merely because a drain is running.
        let resolved = tm.acked_total + tm.dead_total + tm.permanently_invalid;
        if resolved > 0 {
            let ratio = tm.acked_total as f64 / resolved as f64;
            result.insert("acceptedYieldRatio".into(), json!(ratio));
            result.insert(
                "acceptedYieldHashrate".into(),
                json!(total_hashrate as f64 * ratio),
            );
        } else {
            // No find has resolved yet: unknown, not an implied 0% or 100%.
            result.insert("acceptedYieldRatio".into(), Value::Null);
            result.insert("acceptedYieldHashrate".into(), Value::Null);
        }
    }

    // Outside the block above on purpose: a broken disk must be visible even before the
    // submission layer exists.
    result.insert(
        "fatalDurabilityFailure".into(),
        json!(snapshot.fatal_durability_failure),
    );
    if snapshot.fatal_durability_failure {
        result.insert(
            "fatalDurabilityReason".into(),
            json!(snapshot.fatal_durability_reason),
        );
    }

    Value::Object(result)
}

/// `buildStatsSnapshot()` — the body served by `/stats` and `/api/v1/status`.
pub fn stats_payload(snapshot: &StatsSnapshot, bind: &str, port: u16, ifaces: &dyn InterfaceSource) -> Value {
    let mut result = gpu_stats_json(snapshot);
    let object = result
        .as_object_mut()
        .expect("gpu_stats_json always builds an object");

    if let Some(sub) = &snapshot.submission {
        let last_observed = sub
            .last_observed_difficulty
            .map(|d| json!(d))
            .unwrap_or(Value::Null);
        let effective = sub
            .last_observed_difficulty
            .map(|d| json!(d + sub.margin_in_effect))
            .unwrap_or(Value::Null);
        object.insert(
            "difficultyStats".into(),
            json!({
                "last_observed": last_observed,
                "margin_in_effect": sub.margin_in_effect,
                "effective_mining_difficulty": effective,
                "margin_changes_total": sub.margin_changes,
            }),
        );
        object.insert(
            "pool".into(),
            json!({ "outage_duration_ms": sub.outage_duration_ms }),
        );
        object.insert(
            "submissions".into(),
            json!({
                "attempts_total": sub.submitted,
                "resubmissions_total": sub.resubmitted,
                "acked_total": sub.acked,
                "accepted_unconfirmed_total": sub.accepted_unconfirmed,
                "transport_failures_total": sub.transport_failures,
                "difficulty_rejections_total": sub.parked_difficulty,
                "xuni_window_rejections_total": sub.parked_xuni,
                "quarantined_total": sub.quarantined,
                "permanently_invalid_total": sub.permanently_invalid,
                "confirmation_retries_total": sub.confirmation_retries,
                "confirmed_via_lookup_total": sub.reconciled_via_get_block,
                "lying_200_total": sub.lying_200_detected,
                "difficulty_probes_total": sub.probes,
                "failed_attempts_total": sub.failed_attempts(),
                "failure_rate_pct": sub.failure_rate_pct(),
            }),
        );
    }

    // Replaces the camelCase journal block written above when the durable journal is
    // present — the same overwrite the C++ performs, with the same snake_case names.
    if let Some(journal) = &snapshot.journal {
        object.insert(
            "journal".into(),
            json!({
                "pending": journal.pending,
                "accepted_unconfirmed": journal.accepted_unconfirmed,
                "parked_total": journal.parked,
                "parked_difficulty": journal.parked_difficulty,
                "parked_xuni": journal.parked_xuni,
                "quarantined": journal.quarantined,
                "acked_total": journal.acked_total,
                "dead_total": journal.dead_total,
                "permanently_invalid": journal.permanently_invalid,
            }),
        );
    }

    object.insert("stats_cache_seconds".into(), json!(STATS_CACHE_SECONDS));
    object.insert("console".into(), console_block(bind, port, ifaces));
    result
}

/// `getMinerDashboardData()` — the body the embedded page polls from `/api/rig`.
pub fn rig_payload(snapshot: &StatsSnapshot, bind: &str, port: u16, ifaces: &dyn InterfaceSource) -> Value {
    let mut gpus = Vec::new();
    let mut gpu_hashrate = 0.0f64;
    let mut devices: Vec<i32> = Vec::new();
    for gpu in snapshot.fresh_gpus() {
        gpus.push(json!({
            "index": gpu.index,
            "stream": gpu.stream_index,
            "name": gpu.name,
            "memory_gib": gpu.memory,
            "memory_used_percent": gpu.using_memory as f64 * 100.0,
            "hashrate": gpu.hashrate,
            "hash_count": gpu.hash_count,
        }));
        if !devices.contains(&gpu.index) {
            devices.push(gpu.index);
        }
        gpu_hashrate += gpu.hashrate as f64;
    }

    let name = if snapshot.custom_name.is_empty() {
        "TreeMiner"
    } else {
        snapshot.custom_name.as_str()
    };

    json!({
        "identity": {
            "name": name,
            "machine_id": snapshot.machine_id,
            "address": snapshot.miner_address,
        },
        "engine": {
            // `running: false` with `fatal_durability_failure: true` means the miner is
            // shutting down because it cannot persist finds — not an operator stop.
            "running": snapshot.running,
            "fatal_durability_failure": snapshot.fatal_durability_failure,
            "uptime_seconds": snapshot.uptime_seconds,
            "difficulty": snapshot.difficulty,
            "gpu_devices": devices.len(),
            "cuda_streams": gpus.len(),
            "cpu_workers": snapshot.cpu_workers,
            "gpu_hashrate": gpu_hashrate,
            "cpu_hashrate": snapshot.cpu_hashrate,
            "total_hashrate": gpu_hashrate + snapshot.cpu_hashrate,
        },
        "finds": {
            "xnm": snapshot.accepted_blocks(),
            "xuni": snapshot.xuni_blocks,
            "super": snapshot.super_blocks,
            "rejected": snapshot.failed_blocks,
        },
        "delivery": {
            "network": snapshot.network_state.label(),
            "last_submission": snapshot.last_submission.label(),
            "last_submission_age_seconds": snapshot.last_submission_age_seconds,
            "queued_xnm": snapshot.queued_xnm,
            "queued_xuni": snapshot.queued_xuni,
            "queued_total": snapshot.queued_xnm + snapshot.queued_xuni,
        },
        "console": console_block(bind, port, ifaces),
        "gpus": gpus,
    })
}

/// `/platform/status`. Without a platform manager the C++ still answers, reporting the
/// disabled shape, so a fleet poller never sees a 404 for a supported route.
pub fn platform_payload(snapshot: &StatsSnapshot) -> Value {
    let Some(platform) = &snapshot.platform else {
        return json!({
            "platform_mode": false,
            "mining_mode": "self",
            "platform_state": "disabled",
            "running": false,
        });
    };

    let mut result = Map::new();
    result.insert("platform_mode".into(), json!(platform.platform_mode));
    result.insert("mining_mode".into(), json!(platform.mining_mode));
    result.insert("platform_state".into(), json!(platform.platform_state));
    result.insert("running".into(), json!(platform.running));
    if let Some(lease) = &platform.lease {
        result.insert("lease_id".into(), json!(lease.lease_id));
        result.insert("consumer_id".into(), json!(lease.consumer_id));
        result.insert("consumer_address".into(), json!(lease.consumer_address));
        result.insert("blocks_found".into(), json!(lease.blocks_found));
        result.insert("remaining_sec".into(), json!(lease.remaining_sec));
    }
    Value::Object(result)
}

/// `getStatData()` — the periodic upload payload. Not served by any route; exported
/// because the reporting thread lives in another crate and must send the same shape.
pub fn stat_upload_payload(snapshot: &StatsSnapshot) -> Value {
    let gpus: Vec<Value> = snapshot
        .gpus
        .iter()
        .map(|gpu| {
            json!({
                "index": gpu.index,
                "name": gpu.name,
                "hashrate": format!("{:.2}", gpu.hashrate),
                "memory": gpu.memory,
                "power": gpu.power_milliwatts_or_sentinel(),
                "utiliz": gpu.utilization_percent(),
                "usingMemory": format!("{:.1}", gpu.using_memory * 100.0),
                "hashCount": gpu.hash_count,
            })
        })
        .collect();

    let mut result = Map::new();
    result.insert("machineId".into(), json!(snapshot.machine_id));
    result.insert("minerAddr".into(), json!(snapshot.miner_address));
    result.insert(
        "totalHashrate".into(),
        json!(format!("{:.2}", snapshot.total_gpu_hashrate())),
    );
    result.insert("totalHashCount".into(), json!(snapshot.total_hash_count()));
    result.insert(
        "totalPower".into(),
        json!(snapshot.total_power_milliwatts()),
    );
    result.insert("difficulty".into(), json!(snapshot.difficulty));
    result.insert("gpus".into(), Value::Array(gpus));
    result.insert("uptime".into(), json!(snapshot.uptime_seconds));
    result.insert("acceptedBlocks".into(), json!(snapshot.accepted_blocks()));
    result.insert("normalBlocks".into(), json!(snapshot.normal_blocks));
    result.insert("superBlocks".into(), json!(snapshot.super_blocks));
    result.insert("rejectedBlocks".into(), json!(snapshot.failed_blocks));
    // The C++ hardcodes the reporting version; an unset field keeps that value.
    let version = if snapshot.version.is_empty() {
        REPORTED_VERSION
    } else {
        snapshot.version.as_str()
    };
    result.insert("version".into(), json!(version));
    if !snapshot.custom_name.is_empty() {
        result.insert("customName".into(), json!(snapshot.custom_name));
    }
    Value::Object(result)
}
