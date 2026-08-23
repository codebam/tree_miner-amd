//! Everything a party with broker access can publish, and none of it may panic.
//!
//! A panic on the dispatch path is a remote DoS: the pump thread dies, the miner stops
//! taking leases, and the operator sees nothing until the fleet goes quiet. So the bar
//! here is not "rejected correctly" but "rejected without unwinding, and without changing
//! any state".

mod common;

use common::*;
use serde_json::json;
use tm_platform::envelope::MAX_PAYLOAD_BYTES;
use tm_platform::proto::{command_from_value, value_from_payload, ParseError};
use tm_platform::PlatformState;

/// Raw byte payloads that a broker peer can send. None may panic; none may be accepted.
fn hostile_payloads() -> Vec<(&'static str, Vec<u8>)> {
    let mut cases: Vec<(&'static str, Vec<u8>)> = vec![
        ("empty", b"".to_vec()),
        ("whitespace", b"   ".to_vec()),
        ("nul bytes", vec![0, 0, 0, 0]),
        ("invalid utf-8", vec![0xff, 0xfe, 0xfd]),
        ("truncated object", br#"{"command":"assign_task""#.to_vec()),
        ("truncated string", br#"{"command":"assig"#.to_vec()),
        ("truncated array", b"[1,2,".to_vec()),
        ("bare scalar", b"42".to_vec()),
        ("bare string", br#""assign_task""#.to_vec()),
        ("bare null", b"null".to_vec()),
        ("bare true", b"true".to_vec()),
        ("top-level array", br#"[{"command":"assign_task"}]"#.to_vec()),
        ("duplicate keys", br#"{"command":"release","command":"assign_task"}"#.to_vec()),
        ("command is a number", br#"{"command":42}"#.to_vec()),
        ("command is an object", br#"{"command":{"$ne":null}}"#.to_vec()),
        ("command is an array", br#"{"command":["assign_task"]}"#.to_vec()),
        ("action is a number", br#"{"action":7}"#.to_vec()),
        ("no discriminator at all", br#"{"lease_id":"L-1"}"#.to_vec()),
        (
            "oversized command field",
            format!(r#"{{"command":"{}"}}"#, "a".repeat(4096)).into_bytes(),
        ),
        (
            "oversized action field",
            format!(r#"{{"action":"{}"}}"#, "a".repeat(4096)).into_bytes(),
        ),
        (
            "assign_task with wrong field types",
            br#"{"command":"assign_task","lease_id":[],"consumer_id":1,"consumer_address":null}"#
                .to_vec(),
        ),
        (
            "assign_task duration as a string",
            br#"{"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":"0x0","duration_sec":"3600"}"#.to_vec(),
        ),
        (
            "assign_task duration overflowing i64",
            br#"{"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":"0x0","duration_sec":99999999999999999999}"#.to_vec(),
        ),
        (
            "assign_task duration negative",
            br#"{"command":"assign_task","lease_id":"L","consumer_id":"C","consumer_address":"0x0","duration_sec":-1}"#.to_vec(),
        ),
        (
            "register_ack accepted as a string",
            br#"{"command":"register_ack","accepted":"true"}"#.to_vec(),
        ),
        (
            "release lease_id as an object",
            br#"{"command":"release","lease_id":{}}"#.to_vec(),
        ),
        (
            "set_config config not an object",
            br#"{"action":"set_config","config":"all of it"}"#.to_vec(),
        ),
        (
            "set_config difficulty as a float",
            br#"{"action":"set_config","config":{"difficulty":1e309}}"#.to_vec(),
        ),
        (
            "auth is a string",
            br#"{"command":"release","auth":"trust me"}"#.to_vec(),
        ),
        (
            "auth is an array",
            br#"{"command":"release","auth":[]}"#.to_vec(),
        ),
    ];

    // Deep nesting: serde_json enforces a recursion limit, so this must come back as a
    // parse error rather than a stack overflow.
    cases.push(("512-deep nesting", nested(512)));
    cases.push(("4096-deep nesting", nested(4096)));
    cases.push(("100k-deep nesting", nested(100_000)));

    // Size: one byte over the cap, and far over it.
    cases.push((
        "one byte over the payload cap",
        oversized(MAX_PAYLOAD_BYTES + 1),
    ));
    cases.push(("8 MiB payload", oversized(8 * 1024 * 1024)));

    // A huge but structurally valid command, under the cap.
    cases.push((
        "64 KiB of lease_id",
        format!(
            r#"{{"command":"assign_task","lease_id":"{}"}}"#,
            "a".repeat(60_000)
        )
        .into_bytes(),
    ));
    cases
}

fn nested(depth: usize) -> Vec<u8> {
    let mut s = String::with_capacity(depth * 2 + 32);
    s.push_str(r#"{"command":"release","x":"#);
    s.push_str(&"[".repeat(depth));
    s.push_str(&"]".repeat(depth));
    s.push('}');
    s.into_bytes()
}

fn oversized(bytes: usize) -> Vec<u8> {
    let filler = "a".repeat(bytes);
    format!(r#"{{"command":"release","pad":"{filler}"}}"#).into_bytes()
}

#[test]
fn hostile_payloads_never_panic_and_never_apply() {
    // With a secret configured, none of these carries a valid envelope, so not one of them
    // may reach a handler at all — the state machine must be exactly where it started.
    let harness = Harness::new(Some(SECRET));
    harness.assign("lease-1", 3600);
    assert_eq!(harness.manager.state(), PlatformState::Mining);
    let before = harness.coordinator.context();
    let lease_before = harness.manager.leases().lease();

    for (name, payload) in hostile_payloads() {
        for suffix in ["task", "control"] {
            harness.deliver_raw(suffix, &payload);
            assert_eq!(
                harness.manager.state(),
                PlatformState::Mining,
                "{name} on {suffix} changed the platform state"
            );
        }
    }
    assert_eq!(harness.coordinator.context(), before, "mining context moved");
    assert_eq!(harness.manager.leases().lease(), lease_before, "lease moved");
    assert_eq!(harness.coordinator.difficulty(), 8);
    assert_eq!(harness.coordinator.identity().user_address, SELF_ADDRESS);
}

/// The same corpus against a secretless (legacy) manager. Here a well-formed *unsigned*
/// `release` or `pause` is legitimately obeyed, so the invariant is narrower: nothing may
/// panic, and nothing that moves money or kills the process may be applied — which now
/// includes taking a lease at all, since `assign_task` names the address it pays.
#[test]
fn hostile_payloads_against_a_secretless_manager_change_nothing_that_matters() {
    let harness = Harness::new(None);
    harness.assign("lease-1", 3600);
    assert_eq!(
        harness.manager.state(),
        PlatformState::Available,
        "an unsigned assign_task leased the rig"
    );

    for (_, payload) in hostile_payloads() {
        for suffix in ["task", "control"] {
            harness.deliver_raw(suffix, &payload);
        }
    }
    assert_eq!(harness.coordinator.difficulty(), 8, "difficulty changed unsigned");
    assert_eq!(
        harness.coordinator.identity().user_address,
        SELF_ADDRESS,
        "payout address changed unsigned"
    );
    assert!(!harness.manager.shutdown_requested(), "shut down unsigned");
    assert!(harness.manager.leases().lease().is_none(), "leased unsigned");
}

/// The same corpus straight at the codec, so a future caller that skips the manager still
/// cannot get a panic out of it.
#[test]
fn codec_rejects_the_same_corpus() {
    for (name, payload) in hostile_payloads() {
        match value_from_payload(&payload) {
            Err(_) => {}
            Ok(value) => {
                // Anything that parses as JSON must still fail to become a command, or be
                // a command whose fields the handler will reject.
                let _ = command_from_value(&value);
                // Reaching here without panicking is the assertion.
                let _ = name;
            }
        }
    }
}

#[test]
fn payload_cap_is_enforced_on_bytes_not_on_parsed_json() {
    let over = oversized(MAX_PAYLOAD_BYTES + 1);
    assert!(over.len() > MAX_PAYLOAD_BYTES);
    assert!(matches!(
        value_from_payload(&over),
        Err(ParseError::TooLarge(_, _))
    ));

    // Exactly at the cap is allowed: the bound must be inclusive or a legitimate large
    // command becomes unusable at an arbitrary boundary.
    let mut at_cap = br#"{"command":"release","pad":""#.to_vec();
    at_cap.resize(MAX_PAYLOAD_BYTES - 2, b'a');
    at_cap.extend_from_slice(br#""}"#);
    assert_eq!(at_cap.len(), MAX_PAYLOAD_BYTES);
    assert!(value_from_payload(&at_cap).is_ok());
}

/// Oversized payloads are dropped at intake, before the JSON parser, and counted.
#[test]
fn oversized_payloads_are_dropped_at_intake() {
    let harness = Harness::new(Some(SECRET));
    let before = harness.manager.dropped_commands();
    harness
        .manager
        .enqueue_command("xenminer/rig-01/task", &oversized(MAX_PAYLOAD_BYTES + 1));
    assert_eq!(harness.manager.dropped_commands(), before + 1);
    assert_eq!(harness.manager.dispatch_pending(), 0, "nothing was queued");
}

/// A flood must not grow the queue without bound, and must not displace commands the
/// platform already handed us.
#[test]
fn command_queue_is_bounded_and_drops_the_newest() {
    let harness = Harness::new(Some(SECRET));
    // The first command in is a genuine, signed assign_task.
    let genuine = harness.sign(&assign_task("lease-keeper", 3600));
    harness
        .manager
        .enqueue_command("xenminer/rig-01/task", genuine.to_string().as_bytes());

    // Then 10_000 junk messages arrive.
    for _ in 0..10_000 {
        harness
            .manager
            .enqueue_command("xenminer/rig-01/task", br#"{"command":"release"}"#);
    }
    assert!(harness.manager.dropped_commands() > 9_000);

    let handled = harness.manager.dispatch_pending();
    assert!(handled <= 256, "queue grew past its capacity: {handled}");
    // The genuine command was at the head and survived the flood.
    assert_eq!(harness.manager.state(), PlatformState::Mining);
    assert_eq!(
        harness.manager.leases().lease().unwrap().lease_id,
        "lease-keeper"
    );
}

/// Log-forging: identifiers that carry control characters must never reach a handler.
#[test]
fn control_characters_in_identifiers_are_refused() {
    let harness = Harness::new(Some(SECRET));
    for evil in [
        "lease\n2026-01-01 INFO forged log line",
        "lease\u{1b}[2J",
        "lease\r\n",
        "lease id",
        "../../etc/passwd",
        "lease/#",
        "lease+",
    ] {
        let msg = harness.sign(&json!({
            "command": "assign_task",
            "lease_id": evil,
            "consumer_id": "consumer-1",
            "consumer_address": CONSUMER_ADDRESS,
            "duration_sec": 3600,
        }));
        harness.deliver("task", &msg);
        assert_eq!(
            harness.manager.state(),
            PlatformState::Available,
            "accepted lease_id {evil:?}"
        );
    }
}
