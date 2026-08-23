//! Every example payload in `../treeminer/proto/examples/` round-trips through the codec.
//!
//! The examples are the protocol's own statement of what goes on the wire, so a change to
//! a field name or type in this crate has to break here.

use serde_json::Value;
use std::path::PathBuf;
use tm_platform::proto::*;

fn examples_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../treeminer/proto/examples")
}

/// The examples carry `_description` / `_topic` / `_source` documentation keys that the
/// protocol README explicitly excludes from the wire format.
fn load(name: &str) -> Value {
    let path = examples_dir().join(name);
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut value: Value = serde_json::from_str(&raw).expect("example is valid JSON");
    let obj = value.as_object_mut().expect("example is an object");
    obj.retain(|k, _| !k.starts_with('_'));
    value
}

/// Decode into `T`, re-encode, and require the JSON to be identical to what we were given.
/// Anything the codec silently drops or renames shows up here.
fn round_trip<T>(name: &str)
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let original = load(name);
    let typed: T = serde_json::from_value(original.clone())
        .unwrap_or_else(|e| panic!("{name} failed to decode: {e}"));
    let reencoded = serde_json::to_value(&typed).expect("re-encode");
    assert_eq!(reencoded, original, "{name} did not round-trip");
}

#[test]
fn every_example_file_is_covered() {
    // A new example must not be able to appear without a test noticing.
    let mut found: Vec<String> = std::fs::read_dir(examples_dir())
        .expect("examples directory")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".json"))
        .collect();
    found.sort();
    let expected = vec![
        "assign_task.json",
        "block_found.json",
        "control_pause.json",
        "control_resume.json",
        "control_shutdown.json",
        "heartbeat.json",
        "register.json",
        "register_ack_accepted.json",
        "register_ack_rejected.json",
        "release.json",
        "status.json",
        "status_offline.json",
    ];
    assert_eq!(found, expected);
}

// --- Worker -> platform: built here, so a full round-trip is meaningful ---

#[test]
fn register_round_trips() {
    round_trip::<Register>("register.json");
    let register: Register = serde_json::from_value(load("register.json")).unwrap();
    assert_eq!(register.gpus.len(), register.gpu_count as usize);
    assert_eq!(
        register.total_memory_gb,
        register.gpus.iter().map(|g| g.memory_gb).sum::<i64>()
    );
    assert_eq!(register.version, WORKER_VERSION);
}

#[test]
fn heartbeat_round_trips() {
    round_trip::<Heartbeat>("heartbeat.json");
}

#[test]
fn status_round_trips() {
    round_trip::<StatusUpdate>("status.json");
    round_trip::<StatusUpdate>("status_offline.json");
    // The optional fields are omitted, not sent as empty strings: `status_offline.json`
    // has no `lease_id` key at all and must not gain one.
    let offline: StatusUpdate = serde_json::from_value(load("status_offline.json")).unwrap();
    let json = serde_json::to_value(&offline).unwrap();
    assert!(json.get("lease_id").is_none());
    assert!(json.get("detail").is_none());
}

#[test]
fn block_found_round_trips() {
    round_trip::<BlockFound>("block_found.json");
    let block: BlockFound = serde_json::from_value(load("block_found.json")).unwrap();
    // hashrate is a *string* on the wire, formatted to two decimals.
    assert_eq!(block.hashrate, "1250.75");
}

// --- Platform -> worker: parsed here ---

#[test]
fn assign_task_example_parses() {
    let Command::AssignTask(task) = command_from_value(&load("assign_task.json")).unwrap() else {
        panic!("dispatched to the wrong handler");
    };
    assert_eq!(task.lease_id, "lease-550e8400-e29b-41d4-a716-446655440000");
    assert_eq!(task.consumer_id, "consumer-7c9e6679-7425-40de-944b-e07fc1f90ae7");
    assert_eq!(task.prefix, "a1b2c3d4e5f6a7b8");
    assert_eq!(task.prefix.len(), PLATFORM_PREFIX_LENGTH);
    assert_eq!(task.duration_sec, 3600);
}

#[test]
fn release_example_parses() {
    let Command::Release(release) = command_from_value(&load("release.json")).unwrap() else {
        panic!("dispatched to the wrong handler");
    };
    assert_eq!(release.lease_id, "lease-550e8400-e29b-41d4-a716-446655440000");
}

#[test]
fn register_ack_examples_parse() {
    let Command::RegisterAck(ok) =
        command_from_value(&load("register_ack_accepted.json")).unwrap()
    else {
        panic!("wrong handler");
    };
    assert!(ok.accepted);

    let Command::RegisterAck(rejected) =
        command_from_value(&load("register_ack_rejected.json")).unwrap()
    else {
        panic!("wrong handler");
    };
    assert!(!rejected.accepted);
    assert_eq!(rejected.reason, "version_unsupported");
}

#[test]
fn control_examples_parse() {
    for (file, expected) in [
        ("control_pause.json", ControlAction::Pause),
        ("control_resume.json", ControlAction::Resume),
        ("control_shutdown.json", ControlAction::Shutdown),
    ] {
        let Command::Control(action) = command_from_value(&load(file)).unwrap() else {
            panic!("{file} dispatched to the wrong handler");
        };
        assert_eq!(action, expected, "{file}");
    }
}

/// The server's `/api/workers/{id}/control` always sends `{"action": ..., "config": {...}}`,
/// including for pause and resume. Those extra keys must not stop the action parsing.
#[test]
fn control_with_the_servers_extra_config_key_parses() {
    let msg = serde_json::json!({ "action": "pause", "config": {} });
    assert_eq!(
        command_from_value(&msg).unwrap(),
        Command::Control(ControlAction::Pause)
    );
    let msg = serde_json::json!({ "action": "set_config", "config": { "difficulty": 12 } });
    let Command::Control(ControlAction::SetConfig { config }) = command_from_value(&msg).unwrap()
    else {
        panic!("wrong handler");
    };
    assert_eq!(config.difficulty, Some(12));
}

/// Dispatch rule from the protocol README: the `task` topic keys on `command`, and anything
/// that is not one of the three task commands falls through to the control handler.
#[test]
fn dispatch_falls_through_to_control() {
    let msg = serde_json::json!({ "action": "shutdown" });
    assert!(matches!(
        command_from_value(&msg),
        Ok(Command::Control(ControlAction::Shutdown))
    ));
    // ...but an unrecognised action is not a command at all.
    let msg = serde_json::json!({ "action": "self_destruct" });
    assert_eq!(command_from_value(&msg), Err(ParseError::UnknownAction));
}

/// The signed form of every platform-to-worker example still parses: the `auth` object is
/// an extra key the command types have to tolerate.
#[test]
fn signed_examples_still_parse() {
    for file in [
        "assign_task.json",
        "release.json",
        "register_ack_accepted.json",
        "control_pause.json",
        "control_shutdown.json",
    ] {
        let signed = tm_platform::envelope::sign_command(
            &load(file),
            "secret",
            "abc123def456",
            "cmd-1",
            "0123456789abcdef",
            1_700_000_000,
            1_700_000_060,
        );
        assert!(command_from_value(&signed).is_ok(), "{file}");
    }
}
