//! The MQTT wire protocol, as specified in `../treeminer/proto/`.
//!
//! Worker -> platform messages ([`Register`], [`Heartbeat`], [`StatusUpdate`],
//! [`BlockFound`]) are built here and published on `xenminer/{worker_id}/{suffix}`.
//! Platform -> worker messages ([`Command`]) arrive on the `task` and `control` suffixes
//! and are parsed here.
//!
//! # Parsing policy
//!
//! Fields are explicitly typed: nothing coerces, so a `"duration_sec": "3600"` is a parse
//! failure and the whole command is dropped, exactly as nlohmann's `value()` type_error
//! aborts the C++ handler. Unknown fields are *tolerated* — the same as the C++, and
//! harmless because the HMAC in [`crate::envelope`] covers the entire canonical body, so
//! an unknown field cannot be smuggled past the signature. Bounds and charsets are
//! enforced at the handler, not here, so a rejection can name the offending field.

use serde::{Deserialize, Serialize};

/// MQTT topic suffixes. Full topic is `xenminer/{worker_id}/{suffix}`.
pub mod topic {
    pub const REGISTER: &str = "register";
    pub const HEARTBEAT: &str = "heartbeat";
    pub const STATUS: &str = "status";
    pub const BLOCK: &str = "block";
    pub const TASK: &str = "task";
    pub const CONTROL: &str = "control";
}

/// Required length of an `assign_task` key prefix, in hex characters.
pub const PLATFORM_PREFIX_LENGTH: usize = 16;
/// Software version reported in [`Register`]; "2.0.0" in the C++.
pub const WORKER_VERSION: &str = "2.0.0";

// --- Worker -> platform ---

/// Individual GPU descriptor, from the C++ `gpuInfo` struct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: i32,
    pub name: String,
    pub memory_gb: i64,
    pub bus_id: i32,
}

/// Worker registration. Published on startup and on resume from `IDLE`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Register {
    pub worker_id: String,
    pub eth_address: String,
    pub gpu_count: i64,
    pub total_memory_gb: i64,
    pub gpus: Vec<GpuInfo>,
    pub version: String,
    pub timestamp: i64,
}

/// Periodic heartbeat, published every `HEARTBEAT_INTERVAL_SEC`.
///
/// `address`, `prefix` and `block_pattern` are absent from `proto/worker_to_platform.json`
/// but are sent by the C++ `WorkerReporter::sendHeartbeat` and *read* by the server
/// (`MatchingEngine.update_heartbeat` stores them as `current_address`/`current_prefix`/
/// `current_block_pattern` for the dashboard). The schema is the stale party, so they are
/// sent; they are optional on parse so an older heartbeat still decodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Heartbeat {
    pub worker_id: String,
    pub hashrate: f64,
    pub active_gpus: i64,
    pub accepted_blocks: i64,
    pub difficulty: i64,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub address: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub prefix: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub block_pattern: String,
    pub uptime_sec: i64,
    pub timestamp: i64,
}

/// State-change notification, published on every transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatusUpdate {
    pub worker_id: String,
    pub state: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub lease_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
    pub timestamp: i64,
}

/// Block-found report. `lease_id` is empty for a self-mined block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockFound {
    pub worker_id: String,
    pub lease_id: String,
    pub hash: String,
    pub key: String,
    pub account: String,
    pub attempts: u64,
    /// A *string* on the wire, formatted to two decimals — the C++ builds it with
    /// `std::setprecision(2)` and the schema types it as a string.
    pub hashrate: String,
    pub timestamp: i64,
}

/// The Last Will and Testament, and the message published just before a clean disconnect.
///
/// Note the field is `status`, not `state`: `MqttClient::connect`/`disconnect` in the C++
/// build this shape, while every other status message uses [`StatusUpdate`]. The server's
/// `update_worker_state` reads `state`, so it records an empty state for this message.
/// Preserved verbatim because changing it is a server-visible protocol change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OfflineNotice {
    pub worker_id: String,
    pub status: String,
    pub timestamp: i64,
}

impl OfflineNotice {
    pub fn new(worker_id: impl Into<String>, timestamp: i64) -> Self {
        Self {
            worker_id: worker_id.into(),
            status: "offline".into(),
            timestamp,
        }
    }
}

// --- Platform -> worker ---

/// The `auth` envelope, as carried on a signed command. Parsed structurally by
/// [`crate::envelope::verify_envelope`] straight off the `serde_json::Value`; this type
/// exists so a command struct can carry the field without losing it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Auth {
    pub worker_id: String,
    pub command_id: String,
    pub issued_at: i64,
    pub expires_at: i64,
    pub nonce: String,
    pub sig: String,
}

/// Registration acknowledgement (`task` topic, `command = "register_ack"`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct RegisterAck {
    /// Missing or absent counts as a rejection — fail closed, as in the C++.
    #[serde(default)]
    pub accepted: bool,
    #[serde(default)]
    pub reason: String,
}

/// Lease assignment (`task` topic, `command = "assign_task"`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct AssignTask {
    #[serde(default)]
    pub lease_id: String,
    #[serde(default)]
    pub consumer_id: String,
    #[serde(default)]
    pub consumer_address: String,
    #[serde(default)]
    pub prefix: String,
    #[serde(default = "default_duration_sec")]
    pub duration_sec: i64,
}

fn default_duration_sec() -> i64 {
    3600
}

/// Lease release (`task` topic, `command = "release"`). An empty `lease_id` releases
/// whatever lease is active.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Release {
    #[serde(default)]
    pub lease_id: String,
}

/// Remote configuration change, the `config` object of a `set_config` control command.
///
/// Every field is optional: only the ones present are applied, and each is validated
/// independently so one bad field does not discard the rest — matching the C++, whose
/// `handleSetConfig` checks `config.contains(...)` per key.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct SetConfig {
    pub difficulty: Option<i64>,
    pub address: Option<String>,
    pub prefix: Option<String>,
    pub block_pattern: Option<String>,
}

/// Control action (`control` topic, dispatched by `action`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action")]
pub enum ControlAction {
    Pause,
    Resume,
    Shutdown,
    SetConfig {
        #[serde(default)]
        config: SetConfig,
    },
}

/// Every platform-to-worker message the worker acts on.
///
/// Dispatch mirrors the C++ and the protocol README: the `task` topic is keyed by
/// `command`, and anything whose `command` is not one of the three task commands falls
/// through to the control handler, which keys on `action`. The topic itself carries no
/// trust — authorisation is envelope-based — so the same parse serves both topics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    RegisterAck(RegisterAck),
    AssignTask(AssignTask),
    Release(Release),
    Control(ControlAction),
}

/// Why a payload could not be turned into a [`Command`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    #[error("payload is {0} bytes, over the {1}-byte cap")]
    TooLarge(usize, usize),
    #[error("payload is not valid JSON")]
    NotJson,
    #[error("payload is not a JSON object")]
    NotObject,
    #[error("oversized command field")]
    OversizedCommand,
    #[error("oversized action field")]
    OversizedAction,
    #[error("unknown control action")]
    UnknownAction,
    #[error("malformed {0} command")]
    Malformed(&'static str),
}

/// Largest `command` string the dispatcher will even look at, as in the C++.
const MAX_COMMAND_FIELD: usize = 64;
/// Largest `action` string the control handler will look at, as in the C++.
const MAX_ACTION_FIELD: usize = 32;

/// Parse an already-size-checked JSON object into a [`Command`].
///
/// Kept separate from byte-level parsing because the dispatcher must verify the envelope
/// on the raw `Value` *before* the message is interpreted: a command that fails
/// authentication is never turned into a typed command at all.
pub fn command_from_value(msg: &serde_json::Value) -> Result<Command, ParseError> {
    let obj = msg.as_object().ok_or(ParseError::NotObject)?;

    let command = match obj.get("command") {
        Some(serde_json::Value::String(s)) => s.as_str(),
        // A non-string `command` is not a task command; the C++'s `value("command", "")`
        // would throw and drop the message, so do the same rather than falling through to
        // the control handler with attacker-chosen typing.
        Some(_) => return Err(ParseError::Malformed("command")),
        None => "",
    };
    if command.len() > MAX_COMMAND_FIELD {
        return Err(ParseError::OversizedCommand);
    }

    match command {
        "register_ack" => serde_json::from_value(msg.clone())
            .map(Command::RegisterAck)
            .map_err(|_| ParseError::Malformed("register_ack")),
        "assign_task" => serde_json::from_value(msg.clone())
            .map(Command::AssignTask)
            .map_err(|_| ParseError::Malformed("assign_task")),
        "release" => serde_json::from_value(msg.clone())
            .map(Command::Release)
            .map_err(|_| ParseError::Malformed("release")),
        _ => {
            let action = match obj.get("action") {
                Some(serde_json::Value::String(s)) => s.as_str(),
                Some(_) => return Err(ParseError::Malformed("action")),
                // No command and no action: nothing to do. The C++ silently ignores it.
                None => return Err(ParseError::UnknownAction),
            };
            if action.len() > MAX_ACTION_FIELD {
                return Err(ParseError::OversizedAction);
            }
            if !matches!(action, "pause" | "resume" | "shutdown" | "set_config") {
                return Err(ParseError::UnknownAction);
            }
            serde_json::from_value(msg.clone())
                .map(Command::Control)
                .map_err(|_| ParseError::Malformed("control"))
        }
    }
}

/// Parse a raw MQTT payload into a `serde_json::Value`, applying the size cap first.
///
/// The cap is checked on the *bytes*, before the JSON parser sees them: a broker-connected
/// attacker must not be able to make the dispatch thread chew megabytes of nested JSON.
pub fn value_from_payload(payload: &[u8]) -> Result<serde_json::Value, ParseError> {
    if payload.len() > crate::envelope::MAX_PAYLOAD_BYTES {
        return Err(ParseError::TooLarge(
            payload.len(),
            crate::envelope::MAX_PAYLOAD_BYTES,
        ));
    }
    let value: serde_json::Value = serde_json::from_slice(payload).map_err(|_| ParseError::NotJson)?;
    if !value.is_object() {
        return Err(ParseError::NotObject);
    }
    Ok(value)
}
