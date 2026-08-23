//! The journal/submitter contract. Port of `src/treeminer/Types.h`; the C++ names and
//! semantics are kept so the two implementations can be diffed against each other.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FindKind {
    Xen11,
    Xuni,
}

impl FindKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FindKind::Xen11 => "XEN11",
            FindKind::Xuni => "XUNI",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "XEN11" => Some(FindKind::Xen11),
            "XUNI" => Some(FindKind::Xuni),
            _ => None,
        }
    }
}

/// Full lifecycle of a find. Terminal states: `Acked`, `Dead`, `PermanentlyInvalid`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FindStatus {
    /// Durable, awaiting (re)submission.
    Pending,
    /// Claimed by the submitter (in-process only; never persisted).
    Submitting,
    /// Got HTTP 200; awaiting `/get_block` confirmation.
    AcceptedUnconfirmed,
    /// Confirmed stored server-side (200 + lookup, or "already exists").
    Acked,
    /// 401 difficulty-too-low; re-pends when the current difficulty allows.
    ParkedDifficulty,
    /// XUNI missed its window; eligible again next :55-:05, bounded by a budget.
    ParkedXuniWindow,
    /// Unknown 4xx / unknown schema; never auto-unparks, operator visible.
    Quarantined,
    /// XUNI window budget exhausted.
    Dead,
    /// Malformed or failed local verification; high-severity log.
    PermanentlyInvalid,
}

impl FindStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            FindStatus::Pending => "Pending",
            FindStatus::Submitting => "Submitting",
            FindStatus::AcceptedUnconfirmed => "AcceptedUnconfirmed",
            FindStatus::Acked => "Acked",
            FindStatus::ParkedDifficulty => "ParkedDifficulty",
            FindStatus::ParkedXuniWindow => "ParkedXuniWindow",
            FindStatus::Quarantined => "Quarantined",
            FindStatus::Dead => "Dead",
            FindStatus::PermanentlyInvalid => "PermanentlyInvalid",
        }
    }

    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "Pending" => FindStatus::Pending,
            "Submitting" => FindStatus::Submitting,
            "AcceptedUnconfirmed" => FindStatus::AcceptedUnconfirmed,
            "Acked" => FindStatus::Acked,
            "ParkedDifficulty" => FindStatus::ParkedDifficulty,
            "ParkedXuniWindow" => FindStatus::ParkedXuniWindow,
            "Quarantined" => FindStatus::Quarantined,
            "Dead" => FindStatus::Dead,
            "PermanentlyInvalid" => FindStatus::PermanentlyInvalid,
            _ => return None,
        })
    }

    /// Terminal states are never re-attempted by the drain loop.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            FindStatus::Acked | FindStatus::Dead | FindStatus::PermanentlyInvalid
        )
    }
}

/// Immutable capture of a find at discovery time, built only from the parameters the GPU
/// batch actually used. `hash_to_verify` is never recomputed after construction — that is
/// what fixes the upstream stale-difficulty silent drop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundPayload {
    /// 64-hex Argon2 password; the server's dedupe key.
    pub key: String,
    /// Complete PHC string including the `m=` of the originating batch.
    pub hash_to_verify: String,
    /// 0x-prefixed reward address (also the salt source).
    pub account: String,
    pub kind: FindKind,
    /// The `m` baked into `hash_to_verify`.
    pub memory_cost: u32,
    pub worker: String,
    pub attempts: u64,
    pub hashes_per_second: f64,
    /// ISO-8601; local bookkeeping only, never sent to the server.
    pub found_at_utc: String,
}

/// A journaled find: payload plus durable lifecycle state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FindRecord {
    pub id: i64,
    pub payload: FoundPayload,
    pub status: FindStatus,
    pub status_reason: String,
    pub attempt_count: i32,
    /// ISO-8601 UTC; persisted so restarts keep their backoff.
    pub next_attempt_at: Option<String>,
    pub last_attempt_at: Option<String>,
    pub last_http_status: Option<i32>,
    pub last_response: String,
    pub confirmed_at: Option<String>,
    pub xuni_windows_tried: i32,
}

impl FindRecord {
    pub fn new(payload: FoundPayload) -> Self {
        Self {
            id: -1,
            payload,
            status: FindStatus::Pending,
            status_reason: String::new(),
            attempt_count: 0,
            next_attempt_at: None,
            last_attempt_at: None,
            last_http_status: None,
            last_response: String::new(),
            confirmed_at: None,
            xuni_windows_tried: 0,
        }
    }
}

/// Outcome of classifying one server response. Pure data, produced by the classifier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    pub next_status: FindStatus,
    /// When a 401 difficulty message embeds `m={N}`, it is surfaced here so the difficulty
    /// cache updates without waiting for the poller.
    pub server_difficulty_hint: Option<u32>,
    /// True for 200 and duplicate responses: the accept must be re-verified via `/get_block`.
    pub needs_lookup_confirmation: bool,
    /// Human-readable, stored in `status_reason`.
    pub reason: String,
}
