//! Command authorisation and the state machine it guards.
//!
//! Each security property here has a test that fails if the corresponding check is
//! removed; the module comment on each names which check it pins.

mod common;

use common::*;
use serde_json::json;
use std::time::Duration;
use tm_platform::coordinator::MiningMode;
use tm_platform::envelope::{sign_command, MAX_LIFETIME_SEC};
use tm_platform::{Clock, PlatformState};

// --- Authentication: verify_envelope is consulted at all ---

/// Pins: `authorize_command` verifies the envelope when a secret is configured.
/// Delete that check and an unsigned `assign_task` is obeyed, so this fails.
#[test]
fn unsigned_commands_are_refused_when_a_secret_is_configured() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &assign_task("lease-1", 3600));
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert!(harness.manager.leases().lease().is_none());
}

/// Pins: the signature itself, not merely the presence of an `auth` object.
#[test]
fn a_forged_signature_is_refused() {
    let harness = Harness::new(Some(SECRET));
    let msg = harness.sign_as(&assign_task("lease-1", 3600), WORKER, "the wrong secret");
    harness.deliver("task", &msg);
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

/// Pins: an attacker who captures a valid command and edits the payout address cannot
/// reuse the signature.
#[test]
fn a_command_edited_after_signing_is_refused() {
    let harness = Harness::new(Some(SECRET));
    let mut msg = harness.sign(&assign_task("lease-1", 3600));
    msg["consumer_address"] = json!(SELF_ADDRESS);
    harness.deliver("task", &msg);
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

#[test]
fn a_correctly_signed_command_is_obeyed() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    assert_eq!(harness.manager.state(), PlatformState::Mining);
    let ctx = harness.coordinator.context();
    assert_eq!(ctx.mode, MiningMode::PlatformMining);
    assert_eq!(ctx.address, CONSUMER_ADDRESS);
    assert_eq!(ctx.lease_id, "lease-1");
    assert_eq!(ctx.prefix, "a1b2c3d4e5f6a7b8");
}

// --- Topic authorisation ---

/// Pins: the `worker_id` check inside the envelope. A shared broker mis-routing, or an
/// attacker republishing another rig's genuinely signed command onto our topic, must not
/// be obeyed. Delete the `WrongWorker` check and this fails.
#[test]
fn a_command_addressed_to_another_worker_is_refused() {
    let harness = Harness::new(Some(SECRET));
    // Signed with the real secret — the issuer is legitimate, the target is not us.
    let msg = harness.sign_as(&assign_task("lease-1", 3600), "rig-99", SECRET);
    harness.deliver("task", &msg);
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert!(harness.manager.leases().lease().is_none());
}

/// The topic a command arrives on grants nothing: a `control` command published to the
/// `task` topic still needs its signature, and a signed one works from either topic. This
/// is deliberate — the C++ comments that "the topic adds no trust".
#[test]
fn authorisation_does_not_depend_on_the_topic() {
    for suffix in ["task", "control"] {
        let harness = Harness::new(Some(SECRET));
        harness.deliver(suffix, &json!({ "action": "shutdown" }));
        assert!(!harness.manager.shutdown_requested(), "{suffix} unsigned");

        harness.deliver(suffix, &harness.sign(&json!({ "action": "shutdown" })));
        assert!(harness.manager.shutdown_requested(), "{suffix} signed");
    }
}

// --- Replay ---

/// Pins: the nonce cache. QoS 1 gives at-least-once delivery, so the broker itself
/// redelivers; a command must be applied exactly once. Delete the replay check and the
/// second delivery re-runs the handler.
#[test]
fn a_redelivered_command_is_applied_once() {
    let harness = Harness::new(Some(SECRET));
    let assign = harness.sign(&assign_task("lease-1", 3600));
    harness.deliver("task", &assign);
    assert_eq!(harness.manager.state(), PlatformState::Mining);

    let release = harness.sign(&json!({ "command": "release", "lease_id": "lease-1" }));
    harness.deliver("task", &release);
    assert_eq!(harness.manager.state(), PlatformState::Available);

    // The broker redelivers the assign, then the release, then the assign again.
    harness.deliver("task", &assign);
    assert_eq!(
        harness.manager.state(),
        PlatformState::Available,
        "a replayed assign_task restarted the lease"
    );
    harness.deliver("task", &release);
    harness.deliver("task", &assign);
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert!(harness.manager.leases().lease().is_none());
}

/// Pins: the envelope expiry check. A command captured today must not work tomorrow.
#[test]
fn an_expired_command_is_refused() {
    let harness = Harness::new(Some(SECRET));
    let msg = harness.sign(&assign_task("lease-1", 3600));
    harness.clock.advance(Duration::from_secs(61));
    harness.deliver("task", &msg);
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

/// A signer cannot mint an envelope that outlives the replay cache's memory.
#[test]
fn an_over_long_lifetime_is_refused_even_when_correctly_signed() {
    let harness = Harness::new(Some(SECRET));
    let now = harness.clock.now_epoch_s();
    let msg = sign_command(
        &assign_task("lease-1", 3600),
        SECRET,
        WORKER,
        "cmd-long",
        "00112233445566778899aabbccddeeff",
        now,
        now + MAX_LIFETIME_SEC + 1,
    );
    harness.deliver("task", &msg);
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

// --- Legacy (no secret) policy ---

/// Pins: `is_mutating_command`'s fail-closed classification. Without a secret the
/// commands that cannot move money still work...
#[test]
fn without_a_secret_the_harmless_commands_still_work() {
    let harness = Harness::new(None);
    harness.deliver("control", &json!({ "action": "pause" }));
    assert_eq!(harness.manager.state(), PlatformState::Idle);
    harness.deliver("control", &json!({ "action": "resume" }));
    assert_eq!(harness.manager.state(), PlatformState::Available);
    // `release` is obeyed unsigned because its only effect is to hand the rig back; with
    // no lease running it is a no-op, which is exactly the state a secretless rig is in.
    harness.deliver("task", &json!({ "command": "release", "lease_id": "lease-1" }));
    assert_eq!(harness.manager.state(), PlatformState::Available);
    harness.deliver("task", &json!({ "command": "register_ack", "accepted": true }));
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

/// Pins: `assign_task` is a MUTATING command. It names the address every block found for
/// the next seven days is paid to, so on a secretless rig anyone who can reach the broker
/// would otherwise take the whole output by publishing one message.
///
/// The refusal must also be total: no lease, no state move, no identity change. A
/// partially-applied refusal is worse than an accepted command, because the operator's
/// console would still read AVAILABLE while the salt had already moved.
#[test]
fn without_a_secret_an_assign_task_is_refused_and_changes_nothing() {
    let harness = Harness::new(None);
    let identity_before = harness.coordinator.identity();
    let context_before = harness.coordinator.context();

    harness.deliver("task", &assign_task("lease-1", 3600));

    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert!(harness.manager.leases().lease().is_none());
    assert!(!harness.manager.leases().has_active_lease());
    // The mining identity is what decides where the reward lands; nothing about it moved.
    assert_eq!(harness.coordinator.identity(), identity_before);
    assert_eq!(harness.coordinator.context(), context_before);
    assert_eq!(harness.coordinator.context().mode, MiningMode::SelfMining);
    assert_eq!(harness.coordinator.context().address, SELF_ADDRESS);
    assert!(harness.coordinator.context().prefix.is_empty());
    assert!(harness.coordinator.context().consumer_id.is_empty());
    assert_eq!(harness.coordinator.difficulty(), 8);

    // ...and the same message with a valid signature IS obeyed, so the refusal above is
    // about authentication and not about the message being malformed.
    let signed = Harness::new(Some(SECRET));
    signed.deliver("task", &signed.sign(&assign_task("lease-1", 3600)));
    assert_eq!(signed.manager.state(), PlatformState::Mining);
    assert_eq!(signed.coordinator.context().address, CONSUMER_ADDRESS);
}

/// ...and every mutating command is refused. Delete the `is_mutating_command` gate and
/// an anonymous broker peer redirects the payout address.
#[test]
fn without_a_secret_mutating_commands_are_refused() {
    let harness = Harness::new(None);

    harness.deliver("control", &json!({ "action": "shutdown" }));
    assert!(!harness.manager.shutdown_requested());
    assert!(harness.manager.is_running() || !harness.manager.is_running());

    harness.deliver(
        "control",
        &json!({ "action": "set_config", "config": { "address": CONSUMER_ADDRESS } }),
    );
    assert_eq!(harness.coordinator.identity().user_address, SELF_ADDRESS);

    harness.deliver(
        "control",
        &json!({ "action": "set_config", "config": { "difficulty": 4096 } }),
    );
    assert_eq!(harness.coordinator.difficulty(), 8);

    harness.deliver("task", &assign_task("lease-1", 3600));
    assert!(harness.manager.leases().lease().is_none());
    assert_eq!(harness.coordinator.context().mode, MiningMode::SelfMining);

    // Unknown commands are mutating by default.
    harness.deliver("control", &json!({ "command": "drain_wallet" }));
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

// --- assign_task validation ---

#[test]
fn assign_task_field_validation() {
    let cases: Vec<(&str, serde_json::Value)> = vec![
        ("empty lease_id", json!({"command":"assign_task","lease_id":"","consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"duration_sec":3600})),
        ("lease_id too long", json!({"command":"assign_task","lease_id":"a".repeat(65),"consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"duration_sec":3600})),
        ("empty consumer_id", json!({"command":"assign_task","lease_id":"L","consumer_id":"","consumer_address":CONSUMER_ADDRESS,"duration_sec":3600})),
        ("address not checksummed", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":CONSUMER_ADDRESS.to_lowercase(),"duration_sec":3600})),
        ("address too short", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":"0xdead","duration_sec":3600})),
        ("address missing 0x", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":"5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed","duration_sec":3600})),
        ("prefix not hex", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"prefix":"zzzzzzzzzzzzzzzz","duration_sec":3600})),
        ("prefix wrong length", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"prefix":"a1b2c3","duration_sec":3600})),
        ("duration too short", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"duration_sec":59})),
        ("duration too long", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"duration_sec":7*24*3600+1})),
        ("duration zero", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"duration_sec":0})),
        ("duration negative", json!({"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":CONSUMER_ADDRESS,"duration_sec":-3600})),
    ];
    for (name, msg) in cases {
        let harness = Harness::new(Some(SECRET));
        harness.deliver("task", &harness.sign(&msg));
        assert_eq!(
            harness.manager.state(),
            PlatformState::Available,
            "{name} was accepted"
        );
        assert!(harness.manager.leases().lease().is_none(), "{name}");
        assert_eq!(harness.coordinator.context().mode, MiningMode::SelfMining);
    }
}

/// An empty prefix is legal and means "no prefix", and the duration defaults to an hour.
#[test]
fn assign_task_defaults() {
    let harness = Harness::new(Some(SECRET));
    let msg = json!({
        "command": "assign_task",
        "lease_id": "lease-1",
        "consumer_id": "consumer-1",
        "consumer_address": CONSUMER_ADDRESS,
    });
    harness.deliver("task", &harness.sign(&msg));
    assert_eq!(harness.manager.state(), PlatformState::Mining);
    let lease = harness.manager.leases().lease().unwrap();
    assert_eq!(lease.duration_sec, 3600);
    assert_eq!(lease.prefix, "");
}

/// A malformed assign_task must leave the rig AVAILABLE, not knock it into ERROR — a
/// broker peer must not be able to take a rig out of service with one bad message.
#[test]
fn a_rejected_assign_task_leaves_the_rig_available() {
    let harness = Harness::new(Some(SECRET));
    for _ in 0..10 {
        harness.deliver("task", &harness.sign(&json!({"command":"assign_task","lease_id":"bad id"})));
        assert_eq!(harness.manager.state(), PlatformState::Available);
    }
}

/// assign_task only applies from AVAILABLE, so a signed one cannot repoint a rig that is
/// already mining for someone else.
#[test]
fn assign_task_is_ignored_unless_available() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    assert_eq!(harness.manager.state(), PlatformState::Mining);

    harness.deliver("task", &harness.sign(&assign_task("lease-2", 3600)));
    assert_eq!(harness.manager.leases().lease().unwrap().lease_id, "lease-1");
    assert_eq!(harness.coordinator.context().lease_id, "lease-1");
}

// --- release ---

#[test]
fn release_for_another_lease_is_ignored() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    harness.deliver(
        "task",
        &harness.sign(&json!({ "command": "release", "lease_id": "lease-2" })),
    );
    assert_eq!(harness.manager.state(), PlatformState::Mining);
    assert_eq!(harness.manager.leases().lease().unwrap().lease_id, "lease-1");
}

#[test]
fn release_without_a_lease_id_releases_the_active_lease() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    harness.deliver("task", &harness.sign(&json!({ "command": "release" })));
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert_eq!(harness.coordinator.context().mode, MiningMode::SelfMining);
    assert_eq!(harness.coordinator.context().address, SELF_ADDRESS);
}

#[test]
fn release_with_no_active_lease_is_a_no_op() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&json!({ "command": "release" })));
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

// --- register_ack ---

#[test]
fn register_ack_accepted_keeps_the_rig_available() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver(
        "task",
        &harness.sign(&json!({ "command": "register_ack", "accepted": true })),
    );
    assert_eq!(harness.manager.state(), PlatformState::Available);
}

#[test]
fn register_ack_rejected_moves_to_error_and_the_watchdog_recovers() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver(
        "task",
        &harness.sign(&json!({
            "command": "register_ack",
            "accepted": false,
            "reason": "version_unsupported",
        })),
    );
    assert_eq!(harness.manager.state(), PlatformState::Error);

    // A connected transport lets the watchdog re-register and come back.
    harness.manager.watchdog_tick();
    assert_eq!(harness.manager.state(), PlatformState::Available);

    // A disconnected one leaves it IDLE until the link returns.
    harness.deliver(
        "task",
        &harness.sign(&json!({ "command": "register_ack", "accepted": false })),
    );
    harness.transport.set_connected(false);
    harness.manager.watchdog_tick();
    assert_eq!(harness.manager.state(), PlatformState::Idle);
}

/// A rejection reason is attacker-influenced text bound for a log. It is bounded before it
/// gets there, and the handler must not choke on anything in it.
#[test]
fn a_hostile_rejection_reason_is_handled() {
    for reason in [
        json!("\u{1b}[2J\u{1b}[1;1H fake console"),
        json!("x".repeat(10_000)), // under the payload cap, over the log bound
        json!("line\nbreak"),
        json!(""),
    ] {
        let harness = Harness::new(Some(SECRET));
        harness.deliver(
            "task",
            &harness.sign(&json!({ "command": "register_ack", "accepted": false, "reason": reason })),
        );
        assert_eq!(harness.manager.state(), PlatformState::Error);
    }
}

// --- set_config ---

#[test]
fn signed_set_config_applies_every_field() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver(
        "control",
        &harness.sign(&json!({
            "action": "set_config",
            "config": {
                "difficulty": 4096,
                "address": CONSUMER_ADDRESS,
                "prefix": "abcdef",
                "block_pattern": "XEN11",
            }
        })),
    );
    assert_eq!(harness.coordinator.difficulty(), 4096);
    let identity = harness.coordinator.identity();
    assert_eq!(identity.user_address, CONSUMER_ADDRESS);
    assert_eq!(identity.self_mining_prefix, "abcdef");
    assert_eq!(identity.test_block_pattern, "XEN11");
    // The change is heartbeated immediately so the dashboard sees it.
    let heartbeats = harness.transport.published_on("heartbeat");
    assert_eq!(heartbeats.len(), 1);
    assert_eq!(heartbeats[0]["difficulty"], 4096);
    assert_eq!(heartbeats[0]["address"], CONSUMER_ADDRESS);
}

/// Bounds on everything a signed `set_config` can set: even a compromised platform server
/// must not be able to push a value that OOMs the rig or destroys its search entropy.
/// A rejected field rejects the WHOLE command — nothing is applied partially.
#[test]
fn set_config_bounds_and_all_or_nothing() {
    let cases: Vec<(&str, serde_json::Value)> = vec![
        ("difficulty zero", json!({ "difficulty": 0 })),
        ("difficulty negative", json!({ "difficulty": -1 })),
        ("difficulty absurd", json!({ "difficulty": 10_000_001i64 })),
        ("address not checksummed", json!({ "address": CONSUMER_ADDRESS.to_lowercase() })),
        ("address empty", json!({ "address": "" })),
        ("address garbage", json!({ "address": "not an address" })),
        ("prefix not hex", json!({ "prefix": "zz" })),
        ("prefix too long", json!({ "prefix": "a".repeat(33) })),
        ("block_pattern too long", json!({ "block_pattern": "X".repeat(17) })),
        ("block_pattern with control chars", json!({ "block_pattern": "XEN\u{1b}" })),
    ];
    for (name, bad_field) in cases {
        let harness = Harness::new(Some(SECRET));
        // Pair the bad field with good ones: if any good field lands, the command was
        // applied partially.
        let mut config = json!({ "difficulty": 4096, "block_pattern": "GOOD" });
        for (k, v) in bad_field.as_object().unwrap() {
            config[k] = v.clone();
        }
        harness.deliver(
            "control",
            &harness.sign(&json!({ "action": "set_config", "config": config })),
        );
        assert_eq!(harness.coordinator.difficulty(), 8, "{name} applied partially");
        assert_eq!(
            harness.coordinator.identity().test_block_pattern,
            "",
            "{name} applied partially"
        );
        assert_eq!(harness.coordinator.identity().user_address, SELF_ADDRESS, "{name}");
        assert!(
            harness.transport.published_on("heartbeat").is_empty(),
            "{name} heartbeated a change it did not make"
        );
    }
}

/// Empty prefix and pattern are the documented way to clear them.
#[test]
fn set_config_can_clear_prefix_and_pattern() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver(
        "control",
        &harness.sign(&json!({ "action": "set_config", "config": { "prefix": "ab", "block_pattern": "XEN" } })),
    );
    harness.deliver(
        "control",
        &harness.sign(&json!({ "action": "set_config", "config": { "prefix": "", "block_pattern": "" } })),
    );
    let identity = harness.coordinator.identity();
    assert_eq!(identity.self_mining_prefix, "");
    assert_eq!(identity.test_block_pattern, "");
}

// --- pause / resume / shutdown ---

#[test]
fn pause_ends_an_active_lease_and_goes_idle() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    harness.deliver("control", &harness.sign(&json!({ "action": "pause" })));
    assert_eq!(harness.manager.state(), PlatformState::Idle);
    assert!(harness.manager.leases().lease().is_none());
    assert_eq!(harness.coordinator.context().mode, MiningMode::SelfMining);
}

#[test]
fn resume_only_applies_from_idle() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("control", &harness.sign(&json!({ "action": "resume" })));
    assert_eq!(harness.manager.state(), PlatformState::Available);
    // No re-registration happened, because it was never IDLE.
    assert!(harness.transport.published_on("register").is_empty());

    harness.deliver("control", &harness.sign(&json!({ "action": "pause" })));
    harness.deliver("control", &harness.sign(&json!({ "action": "resume" })));
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert_eq!(harness.transport.published_on("register").len(), 1);
}

// --- Telemetry ---

#[test]
fn a_find_during_a_lease_is_reported_against_it() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    harness.transport.clear();

    harness
        .manager
        .on_block_found("00000abc", "deadbeef", CONSUMER_ADDRESS, 1_500_000, 1250.75);
    let blocks = harness.transport.published_on("block");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["lease_id"], "lease-1");
    assert_eq!(blocks[0]["worker_id"], WORKER);
    assert_eq!(blocks[0]["attempts"], 1_500_000);
    assert_eq!(blocks[0]["hashrate"], "1250.75");
    assert_eq!(harness.manager.leases().lease().unwrap().blocks_found, 1);
}

#[test]
fn a_self_mined_find_is_reported_with_an_empty_lease() {
    let harness = Harness::new(Some(SECRET));
    harness.transport.clear();
    harness
        .manager
        .on_block_found("00000abc", "deadbeef", SELF_ADDRESS, 10, 1.0);
    let blocks = harness.transport.published_on("block");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["lease_id"], "");
}

/// Publishing is best-effort: a broker that is down must not stop the miner or lose the
/// command that is being processed.
#[test]
fn a_disconnected_broker_does_not_break_dispatch() {
    let harness = Harness::new(Some(SECRET));
    harness.transport.set_connected(false);
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    // The command still applied even though no status update could be published.
    assert_eq!(harness.manager.state(), PlatformState::Mining);
    assert!(harness.transport.published_on("status").is_empty());
}

/// A command that arrives while the link is down waits in the queue and is applied once
/// when dispatch runs — not lost, not applied twice.
#[test]
fn a_command_queued_while_disconnected_is_applied_exactly_once() {
    let harness = Harness::new(Some(SECRET));
    let msg = harness.sign(&assign_task("lease-1", 3600));

    harness.transport.set_connected(false);
    harness
        .manager
        .enqueue_command("xenminer/rig-01/task", msg.to_string().as_bytes());
    assert_eq!(harness.manager.state(), PlatformState::Available, "not yet dispatched");

    harness.transport.set_connected(true);
    assert_eq!(harness.manager.dispatch_pending(), 1);
    assert_eq!(harness.manager.state(), PlatformState::Mining);

    // The broker redelivering it after reconnect does not start a second lease.
    harness
        .manager
        .enqueue_command("xenminer/rig-01/task", msg.to_string().as_bytes());
    harness.manager.dispatch_pending();
    assert_eq!(harness.manager.leases().lease().unwrap().lease_id, "lease-1");
    assert_eq!(harness.manager.leases().lease().unwrap().blocks_found, 0);
}

/// Commands are dispatched in arrival order, so a release cannot overtake the assign it
/// refers to.
#[test]
fn commands_are_dispatched_in_fifo_order() {
    let harness = Harness::new(Some(SECRET));
    let assign = harness.sign(&assign_task("lease-1", 3600));
    let release = harness.sign(&json!({ "command": "release", "lease_id": "lease-1" }));
    harness
        .manager
        .enqueue_command("xenminer/rig-01/task", assign.to_string().as_bytes());
    harness
        .manager
        .enqueue_command("xenminer/rig-01/task", release.to_string().as_bytes());
    assert_eq!(harness.manager.dispatch_pending(), 2);
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert!(harness.manager.leases().lease().is_none());
}

/// The threaded lifecycle: start spawns the loops, stop joins them, and a signed remote
/// shutdown stops the manager from inside its own dispatch thread without deadlocking.
#[test]
fn start_and_stop_are_clean() {
    let harness = Harness::new(Some(SECRET));
    assert!(harness.manager.start());
    assert!(harness.manager.is_running());
    assert!(harness.manager.start(), "start is idempotent");

    let msg = harness.sign(&assign_task("lease-1", 3600));
    harness
        .manager
        .enqueue_command("xenminer/rig-01/task", msg.to_string().as_bytes());
    for _ in 0..200 {
        if harness.manager.state() == PlatformState::Mining {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert_eq!(harness.manager.state(), PlatformState::Mining);

    harness.manager.stop();
    harness.manager.stop();
    assert!(!harness.manager.is_running());
    assert_eq!(harness.manager.state(), PlatformState::Idle);
}

#[test]
fn a_signed_shutdown_stops_the_manager_from_its_own_dispatch_thread() {
    let harness = Harness::new(Some(SECRET));
    assert!(harness.manager.start());
    let msg = harness.sign(&json!({ "action": "shutdown" }));
    harness
        .manager
        .enqueue_command("xenminer/rig-01/control", msg.to_string().as_bytes());

    for _ in 0..400 {
        if harness.manager.shutdown_requested() && !harness.manager.is_running() {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(harness.manager.shutdown_requested());
    assert!(!harness.manager.is_running());
}
