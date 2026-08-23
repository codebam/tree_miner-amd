//! HMAC-SHA256 signed-command envelope for platform/MQTT control messages. Port of
//! `src/platform/CommandEnvelope.{h,cpp}`.
//!
//! WHY this exists: the MQTT broker is a shared rendezvous point. Anyone who can publish
//! to `xenminer/{worker_id}/task` or `.../control` could otherwise redirect the miner's
//! payout address, change its difficulty, or shut it down. Every command carries a
//! verifiable envelope so the miner only obeys a party holding the shared secret.
//!
//! Wire format — the signer adds an `auth` object to the command JSON:
//!
//! ```json
//! {
//!   "command": "assign_task", "...": "...",
//!   "auth": {
//!     "worker_id":  "<target worker's machine id>",
//!     "command_id": "<issuer-unique id, 1..128 chars [A-Za-z0-9._-]>",
//!     "issued_at":  1700000000,
//!     "expires_at": 1700000060,
//!     "nonce":      "<random hex, 16..128 chars>",
//!     "sig":        "<lowercase hex HMAC-SHA256, 64 chars>"
//!   }
//! }
//! ```
//!
//! The signature covers a canonical string (see [`signing_string`]):
//!
//! ```text
//! "TMv1\n" + worker_id + "\n" + command_id + "\n" + issued_at + "\n"
//!          + expires_at + "\n" + nonce + "\n" + canonical_body(msg)
//! ```
//!
//! `canonical_body` is the message with `auth` removed, serialised compactly with sorted
//! keys. The C++ relies on `nlohmann::json::dump()` for exactly that property;
//! `serde_json::Map` is a `BTreeMap` here (the `preserve_order` feature is off across the
//! workspace), so the two produce the same bytes for the same logical message.
//!
//! Replay defence: the nonce of every ACCEPTED command is remembered in a bounded
//! [`NonceCache`]; a repeat within its validity window is rejected. Nonces are recorded
//! only after the signature verifies, so an attacker without the secret cannot poison or
//! fill the cache.

use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

// --- Bounds (shared by signer, verifier, and the transport layer) ---

/// Reject payloads before JSON parsing ever sees them; a broker-connected attacker must
/// not be able to make the dispatch thread chew megabytes of nested JSON.
pub const MAX_PAYLOAD_BYTES: usize = 64 * 1024;

/// Hard cap on envelope lifetime. Short lifetimes bound the replay window the
/// [`NonceCache`] has to remember, which is what lets the cache stay small.
pub const MAX_LIFETIME_SEC: i64 = 15 * 60;

/// Tolerated clock skew between issuer and miner for the issued-at check.
pub const CLOCK_SKEW_SEC: i64 = 30;

/// >= 64 bits of randomness.
pub const MIN_NONCE_HEX_LEN: usize = 16;
pub const MAX_NONCE_HEX_LEN: usize = 128;
/// `worker_id` / `command_id` bound.
pub const MAX_ID_LEN: usize = 128;

// --- Small validation helpers (also used for command payload field checks) ---

/// True iff `s` is entirely `[0-9a-fA-F]` and `min_len <= len <= max_len`.
pub fn is_hex_string(s: &str, min_len: usize, max_len: usize) -> bool {
    (min_len..=max_len).contains(&s.len()) && s.bytes().all(|c| c.is_ascii_hexdigit())
}

/// True iff `s` is entirely `[A-Za-z0-9._-]` and `min_len <= len <= max_len`.
///
/// WHY this charset: identifiers travel into logs and MQTT topics; a conservative set
/// prevents log forging (embedded `\n` / ANSI) and topic injection.
pub fn is_safe_identifier(s: &str, min_len: usize, max_len: usize) -> bool {
    (min_len..=max_len).contains(&s.len())
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'.' || c == b'_' || c == b'-')
}

/// True iff `s` is printable ASCII (0x20..=0x7E) and `min_len <= len <= max_len`.
pub fn is_printable_ascii(s: &str, min_len: usize, max_len: usize) -> bool {
    (min_len..=max_len).contains(&s.len()) && s.bytes().all(|c| (0x20..=0x7E).contains(&c))
}

/// Lowercase-hex HMAC-SHA256.
pub fn hmac_sha256_hex(key: &str, data: &str) -> String {
    hex::encode(hmac_sha256(key, data))
}

fn hmac_sha256(key: &str, data: &str) -> [u8; 32] {
    // `new_from_slice` only errors for key sizes HMAC cannot take, and HMAC takes any
    // length (RFC 2104), so this cannot fail — including for the empty key used in tests.
    let mut mac = HmacSha256::new_from_slice(key.as_bytes()).expect("HMAC accepts any key length");
    mac.update(data.as_bytes());
    mac.finalize().into_bytes().into()
}

/// Constant-time equality for hex digests (case-insensitive; both sides are compared as
/// decoded bytes). WHY: a naive string compare leaks the matching prefix length through
/// timing, which is enough to forge a signature one nibble at a time.
pub fn constant_time_hex_equals(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        // Length is not secret: the digest length is fixed and public.
        return false;
    }
    let (Ok(a), Ok(b)) = (hex::decode(a), hex::decode(b)) else {
        return false;
    };
    a.ct_eq(&b).into()
}

// --- Bounded replay cache ---

/// Remembers accepted nonces until they expire. FIFO-bounded: when full, the oldest entry
/// is evicted. Because envelope lifetime is capped at [`MAX_LIFETIME_SEC`], insertion
/// order approximates expiry order, so eviction rarely discards a still-live nonce unless
/// capacity is exceeded by genuinely signed traffic — which an attacker without the
/// secret cannot cause.
#[derive(Debug)]
pub struct NonceCache {
    capacity: usize,
    by_nonce: std::collections::HashMap<String, i64>,
    insertion_order: std::collections::VecDeque<String>,
}

impl NonceCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            by_nonce: std::collections::HashMap::new(),
            insertion_order: std::collections::VecDeque::new(),
        }
    }

    /// Returns true and records the nonce if it is fresh; false if it was already seen
    /// (replay). Expired entries are purged lazily as a side effect.
    pub fn check_and_insert(&mut self, nonce: &str, expires_at: i64, now_epoch_s: i64) -> bool {
        self.purge_expired(now_epoch_s);

        if self.by_nonce.contains_key(nonce) {
            return false;
        }

        // FIFO eviction keeps memory bounded no matter how much *validly signed* traffic
        // arrives; unsigned floods never reach this point.
        while self.by_nonce.len() >= self.capacity {
            match self.insertion_order.pop_front() {
                Some(oldest) => {
                    self.by_nonce.remove(&oldest);
                }
                None => break,
            }
        }

        self.by_nonce.insert(nonce.to_string(), expires_at);
        self.insertion_order.push_back(nonce.to_string());
        true
    }

    pub fn len(&self) -> usize {
        self.by_nonce.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_nonce.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn purge_expired(&mut self, now_epoch_s: i64) {
        // Insertion order approximates expiry order (lifetime is capped), so sweeping
        // from the front is enough and stays O(expired) amortised.
        while let Some(front) = self.insertion_order.front() {
            match self.by_nonce.get(front) {
                Some(&expires_at) if expires_at > now_epoch_s => break,
                Some(_) => {
                    let front = front.clone();
                    self.by_nonce.remove(&front);
                    self.insertion_order.pop_front();
                }
                None => {
                    self.insertion_order.pop_front();
                }
            }
        }
    }
}

// --- Verification ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyStatus {
    Ok,
    /// No `auth` object at all (unsigned command).
    MissingAuth,
    /// `auth` present but fields missing / wrong type / out of bounds.
    MalformedAuth,
    /// Envelope addressed to a different worker id.
    WrongWorker,
    /// `issued_at` ahead of our clock beyond [`CLOCK_SKEW_SEC`].
    IssuedInFuture,
    /// `expires_at <= issued_at`, or lifetime > [`MAX_LIFETIME_SEC`].
    LifetimeInvalid,
    /// Now past `expires_at`.
    Expired,
    BadSignature,
    ReplayedNonce,
}

impl VerifyStatus {
    pub fn name(self) -> &'static str {
        match self {
            VerifyStatus::Ok => "ok",
            VerifyStatus::MissingAuth => "missing auth envelope",
            VerifyStatus::MalformedAuth => "malformed auth envelope",
            VerifyStatus::WrongWorker => "wrong worker id",
            VerifyStatus::IssuedInFuture => "issued in the future",
            VerifyStatus::LifetimeInvalid => "invalid lifetime",
            VerifyStatus::Expired => "expired",
            VerifyStatus::BadSignature => "bad signature",
            VerifyStatus::ReplayedNonce => "replayed nonce",
        }
    }
}

impl std::fmt::Display for VerifyStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The message with `auth` removed, serialised deterministically (sorted keys, compact).
pub fn canonical_body(msg: &Value) -> String {
    let mut msg = msg.clone();
    if let Some(obj) = msg.as_object_mut() {
        obj.remove("auth");
    }
    msg.to_string()
}

/// The exact byte string the HMAC covers. Exposed so tests and a reference signer share
/// one definition with the verifier.
///
/// `TMv1` domain-separates this MAC from any other use of the same secret and gives a
/// version handle if the format ever changes. Every field is newline-delimited and none
/// of them may contain `\n` (the identifier/hex/number charsets enforce that), so the
/// encoding is unambiguous.
pub fn signing_string(
    worker_id: &str,
    command_id: &str,
    issued_at: i64,
    expires_at: i64,
    nonce: &str,
    body: &str,
) -> String {
    format!("TMv1\n{worker_id}\n{command_id}\n{issued_at}\n{expires_at}\n{nonce}\n{body}")
}

/// Attach a valid `auth` envelope to `msg` (test/reference-signer helper; the miner itself
/// only ever verifies).
pub fn sign_command(
    msg: &Value,
    secret: &str,
    worker_id: &str,
    command_id: &str,
    nonce: &str,
    issued_at: i64,
    expires_at: i64,
) -> Value {
    let body = canonical_body(msg);
    let sig = hmac_sha256_hex(
        secret,
        &signing_string(worker_id, command_id, issued_at, expires_at, nonce, &body),
    );
    let mut out = msg.clone();
    if let Some(obj) = out.as_object_mut() {
        obj.insert(
            "auth".into(),
            json!({
                "worker_id": worker_id,
                "command_id": command_id,
                "issued_at": issued_at,
                "expires_at": expires_at,
                "nonce": nonce,
                "sig": sig,
            }),
        );
    }
    out
}

/// Fetch a signed-integer field, rejecting floats/strings/bools. The envelope wants a
/// hard, typed schema rather than any coercion.
fn int64_field(obj: &Value, key: &str) -> Option<i64> {
    match obj.get(key) {
        Some(Value::Number(n)) => n.as_i64(),
        _ => None,
    }
}

fn string_field<'a>(obj: &'a Value, key: &str) -> Option<&'a str> {
    match obj.get(key) {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Full envelope check in the order schema -> addressing -> time window -> signature ->
/// replay. The nonce is consumed only when everything else passed, so failed attempts
/// cannot poison the cache.
pub fn verify_envelope(
    msg: &Value,
    secret: &str,
    expected_worker_id: &str,
    now_epoch_s: i64,
    nonces: &mut NonceCache,
) -> VerifyStatus {
    if !msg.is_object() {
        return VerifyStatus::MissingAuth;
    }
    let auth = match msg.get("auth") {
        None => return VerifyStatus::MissingAuth,
        Some(v) if !v.is_object() => return VerifyStatus::MalformedAuth,
        Some(v) => v,
    };

    // 1. Schema: every field typed and bounded before anything else looks at it.
    let (Some(worker_id), Some(command_id), Some(nonce), Some(sig)) = (
        string_field(auth, "worker_id"),
        string_field(auth, "command_id"),
        string_field(auth, "nonce"),
        string_field(auth, "sig"),
    ) else {
        return VerifyStatus::MalformedAuth;
    };
    let (Some(issued_at), Some(expires_at)) = (
        int64_field(auth, "issued_at"),
        int64_field(auth, "expires_at"),
    ) else {
        return VerifyStatus::MalformedAuth;
    };
    if !is_safe_identifier(worker_id, 1, MAX_ID_LEN)
        || !is_safe_identifier(command_id, 1, MAX_ID_LEN)
        || !is_hex_string(nonce, MIN_NONCE_HEX_LEN, MAX_NONCE_HEX_LEN)
        // HMAC-SHA256 is exactly 32 bytes / 64 hex characters.
        || !is_hex_string(sig, 64, 64)
    {
        return VerifyStatus::MalformedAuth;
    }

    // 2. Addressing: an envelope signed for a different rig must not work here — this is
    //    what stops cross-worker replay on a shared broker.
    if worker_id != expected_worker_id {
        return VerifyStatus::WrongWorker;
    }

    // 3. Time window. `issued_at` gets skew tolerance; expiry is strict because the signer
    //    controls it and can add margin. Saturating arithmetic: an i64::MAX `issued_at`
    //    must not wrap into a valid window.
    if issued_at > now_epoch_s.saturating_add(CLOCK_SKEW_SEC) {
        return VerifyStatus::IssuedInFuture;
    }
    if expires_at <= issued_at || expires_at.saturating_sub(issued_at) > MAX_LIFETIME_SEC {
        return VerifyStatus::LifetimeInvalid;
    }
    if now_epoch_s > expires_at {
        return VerifyStatus::Expired;
    }

    // 4. Signature — over the canonical body plus every envelope field checked above.
    let expected_sig = hmac_sha256_hex(
        secret,
        &signing_string(
            worker_id,
            command_id,
            issued_at,
            expires_at,
            nonce,
            &canonical_body(msg),
        ),
    );
    if !constant_time_hex_equals(sig, &expected_sig) {
        return VerifyStatus::BadSignature;
    }

    // 5. Replay — last, so only authentically signed commands consume cache slots.
    if !nonces.check_and_insert(nonce, expires_at, now_epoch_s) {
        return VerifyStatus::ReplayedNonce;
    }
    VerifyStatus::Ok
}

// --- Policy classification ---

/// When NO secret is configured (legacy deployments), the miner keeps accepting the
/// non-mutating marketplace flow it always accepted — registration acks, lease
/// assignment/release, pause/resume — but never state-mutating commands (payout address /
/// difficulty / prefix / pattern changes, remote shutdown).
///
/// Fail-closed: unknown commands and unknown control actions count as mutating.
pub fn is_mutating_command(msg: &Value) -> bool {
    if !msg.is_object() {
        return true; // fail closed
    }
    let command = string_field(msg, "command").unwrap_or("");

    // The historical non-mutating marketplace flow: these only steer which lease the rig
    // serves or pause/resume availability — they never change payout identity, difficulty,
    // key prefix, or block pattern, and cannot kill the process.
    if matches!(command, "register_ack" | "assign_task" | "release") {
        return false;
    }

    if command.is_empty() {
        // Control-topic message, keyed by "action".
        let action = string_field(msg, "action").unwrap_or("");
        if matches!(action, "pause" | "resume") {
            return false;
        }
        // "set_config" (payout address / difficulty / prefix / pattern) and "shutdown"
        // (kills mining outright) always require a signature.
        return true;
    }

    true // unknown command: fail closed
}
