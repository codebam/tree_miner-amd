//! Command-envelope tests. Ported case for case from
//! `../treeminer/tests/unit/platform/test_command_envelope.cpp`, plus the cases that
//! Rust's type system does not make redundant.

use serde_json::{json, Value};
use tm_platform::envelope::*;

/// One named way of corrupting a signed message.
type Mutation = (&'static str, Box<dyn Fn(&mut Value)>);

const SECRET: &str = "correct horse battery staple";
const WORKER: &str = "rig-01";
const NONCE: &str = "0123456789abcdef0011223344556677";
const NOW: i64 = 1_700_000_000;

fn base_command() -> Value {
    json!({
        "command": "assign_task",
        "lease_id": "L-1",
        "consumer_id": "C-1",
        "duration_sec": 3600,
    })
}

fn signed(now: i64) -> Value {
    signed_with(now, NONCE, WORKER, SECRET)
}

fn signed_with(now: i64, nonce: &str, worker: &str, secret: &str) -> Value {
    sign_command(&base_command(), secret, worker, "cmd-1", nonce, now, now + 60)
}

#[test]
fn hmac_matches_known_answer_vector() {
    // The same known-answer test the C++ suite uses, so a signature produced by either
    // implementation verifies against the other.
    assert_eq!(
        hmac_sha256_hex("key", "The quick brown fox jumps over the lazy dog"),
        "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
    );
}

#[test]
fn valid_signed_command_verifies() {
    let mut cache = NonceCache::new(16);
    assert_eq!(
        verify_envelope(&signed(NOW), SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::Ok
    );
}

#[test]
fn unsigned_command_is_missing_auth() {
    let mut cache = NonceCache::new(16);
    assert_eq!(
        verify_envelope(&base_command(), SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::MissingAuth
    );
    // A non-object payload has no envelope either.
    assert_eq!(
        verify_envelope(&json!([1, 2, 3]), SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::MissingAuth
    );
}

#[test]
fn malformed_auth_fields_are_rejected() {
    // Every mutation here keeps the envelope structurally present but breaks one field.
    let mutations: Vec<Mutation> = vec![
        ("auth not an object", Box::new(|m: &mut Value| m["auth"] = json!("nope"))),
        ("missing sig", Box::new(|m: &mut Value| { m["auth"].as_object_mut().unwrap().remove("sig"); })),
        ("missing nonce", Box::new(|m: &mut Value| { m["auth"].as_object_mut().unwrap().remove("nonce"); })),
        ("issued_at a string", Box::new(|m: &mut Value| m["auth"]["issued_at"] = json!("1700000000"))),
        ("issued_at a float", Box::new(|m: &mut Value| m["auth"]["issued_at"] = json!(1.7e9))),
        ("expires_at a bool", Box::new(|m: &mut Value| m["auth"]["expires_at"] = json!(true))),
        ("worker_id not a string", Box::new(|m: &mut Value| m["auth"]["worker_id"] = json!(7))),
        ("worker_id empty", Box::new(|m: &mut Value| m["auth"]["worker_id"] = json!(""))),
        ("worker_id with a newline", Box::new(|m: &mut Value| m["auth"]["worker_id"] = json!("rig\n01"))),
        ("command_id over 128 chars", Box::new(|m: &mut Value| m["auth"]["command_id"] = json!("c".repeat(129)))),
        ("nonce too short", Box::new(|m: &mut Value| m["auth"]["nonce"] = json!("abc"))),
        ("nonce too long", Box::new(|m: &mut Value| m["auth"]["nonce"] = json!("a".repeat(129)))),
        ("nonce not hex", Box::new(|m: &mut Value| m["auth"]["nonce"] = json!("zzzzzzzzzzzzzzzz"))),
        ("sig wrong length", Box::new(|m: &mut Value| m["auth"]["sig"] = json!("ab"))),
        ("sig not hex", Box::new(|m: &mut Value| m["auth"]["sig"] = json!("z".repeat(64)))),
    ];
    for (name, mutate) in mutations {
        let mut msg = signed(NOW);
        mutate(&mut msg);
        let mut cache = NonceCache::new(16);
        assert_eq!(
            verify_envelope(&msg, SECRET, WORKER, NOW, &mut cache),
            VerifyStatus::MalformedAuth,
            "{name}"
        );
        assert_eq!(cache.len(), 0, "{name} must not consume a nonce slot");
    }
}

/// The topic-authorisation property: an envelope signed for another rig, by a party that
/// genuinely holds the secret, must not be obeyed here.
#[test]
fn envelope_for_another_worker_is_rejected() {
    let msg = signed_with(NOW, NONCE, "rig-02", SECRET);
    let mut cache = NonceCache::new(16);
    assert_eq!(
        verify_envelope(&msg, SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::WrongWorker
    );
    assert_eq!(cache.len(), 0);
}

#[test]
fn time_window_is_enforced() {
    let mut cache = NonceCache::new(16);

    // Issued beyond the tolerated skew.
    let future = signed(NOW + CLOCK_SKEW_SEC + 1);
    assert_eq!(
        verify_envelope(&future, SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::IssuedInFuture
    );
    // Inside the skew window it is fine, so the check is a window and not a ban on any
    // clock difference at all.
    let barely = signed(NOW + CLOCK_SKEW_SEC);
    assert_eq!(
        verify_envelope(&barely, SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::Ok
    );

    // expires_at <= issued_at.
    let mut inverted = signed(NOW);
    inverted["auth"]["expires_at"] = json!(NOW);
    let sig = hmac_sha256_hex(
        SECRET,
        &signing_string(WORKER, "cmd-1", NOW, NOW, NONCE, &canonical_body(&inverted)),
    );
    inverted["auth"]["sig"] = json!(sig);
    assert_eq!(
        verify_envelope(&inverted, SECRET, WORKER, NOW, &mut NonceCache::new(16)),
        VerifyStatus::LifetimeInvalid
    );

    // Lifetime beyond the cap, correctly signed — the cap is policy, not a signature check.
    let long = sign_command(
        &base_command(),
        SECRET,
        WORKER,
        "cmd-1",
        NONCE,
        NOW,
        NOW + MAX_LIFETIME_SEC + 1,
    );
    assert_eq!(
        verify_envelope(&long, SECRET, WORKER, NOW, &mut NonceCache::new(16)),
        VerifyStatus::LifetimeInvalid
    );

    // Expired.
    assert_eq!(
        verify_envelope(&signed(NOW), SECRET, WORKER, NOW + 61, &mut NonceCache::new(16)),
        VerifyStatus::Expired
    );
}

/// i64 extremes must not wrap a hostile envelope into a valid window.
#[test]
fn extreme_timestamps_do_not_overflow() {
    for (issued, expires) in [
        (i64::MAX, i64::MAX),
        (i64::MIN, i64::MAX),
        (i64::MIN, i64::MIN),
        (0, i64::MAX),
        (i64::MAX, i64::MIN),
    ] {
        let msg = sign_command(&base_command(), SECRET, WORKER, "cmd-1", NONCE, issued, expires);
        let status = verify_envelope(&msg, SECRET, WORKER, NOW, &mut NonceCache::new(16));
        assert_ne!(status, VerifyStatus::Ok, "issued={issued} expires={expires}");
    }
}

#[test]
fn tampering_breaks_the_signature() {
    // Changing any signed field, adding one, or removing one must invalidate the MAC.
    let tamper: Vec<Mutation> = vec![
        ("body field changed", Box::new(|m: &mut Value| m["lease_id"] = json!("L-2"))),
        ("body field added", Box::new(|m: &mut Value| m["consumer_address"] = json!("0xdead"))),
        ("body field removed", Box::new(|m: &mut Value| { m.as_object_mut().unwrap().remove("duration_sec"); })),
        ("command swapped", Box::new(|m: &mut Value| m["command"] = json!("release"))),
        ("command_id changed", Box::new(|m: &mut Value| m["auth"]["command_id"] = json!("cmd-2"))),
        ("issued_at changed", Box::new(|m: &mut Value| m["auth"]["issued_at"] = json!(NOW - 1))),
        ("expires_at changed", Box::new(|m: &mut Value| m["auth"]["expires_at"] = json!(NOW + 61))),
        ("nonce changed", Box::new(|m: &mut Value| m["auth"]["nonce"] = json!("ffffffffffffffff"))),
        ("last sig nibble flipped", Box::new(|m: &mut Value| {
            let mut sig = m["auth"]["sig"].as_str().unwrap().to_string();
            let last = sig.pop().unwrap();
            sig.push(if last == '0' { '1' } else { '0' });
            m["auth"]["sig"] = json!(sig);
        })),
    ];
    for (name, mutate) in tamper {
        let mut msg = signed(NOW);
        mutate(&mut msg);
        assert_eq!(
            verify_envelope(&msg, SECRET, WORKER, NOW, &mut NonceCache::new(16)),
            VerifyStatus::BadSignature,
            "{name}"
        );
    }
}

#[test]
fn a_different_secret_does_not_verify() {
    let msg = signed_with(NOW, NONCE, WORKER, "not the secret");
    assert_eq!(
        verify_envelope(&msg, SECRET, WORKER, NOW, &mut NonceCache::new(16)),
        VerifyStatus::BadSignature
    );
}

#[test]
fn replay_of_an_accepted_command_is_rejected() {
    let msg = signed(NOW);
    let mut cache = NonceCache::new(16);
    assert_eq!(verify_envelope(&msg, SECRET, WORKER, NOW, &mut cache), VerifyStatus::Ok);
    assert_eq!(
        verify_envelope(&msg, SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::ReplayedNonce
    );
    // Still rejected later in the same validity window.
    assert_eq!(
        verify_envelope(&msg, SECRET, WORKER, NOW + 30, &mut cache),
        VerifyStatus::ReplayedNonce
    );
}

/// Failed attempts must not consume cache slots, or an unauthenticated flood could evict
/// the nonces of real commands.
#[test]
fn failed_signature_does_not_consume_the_nonce() {
    let mut cache = NonceCache::new(16);
    let mut bad = signed(NOW);
    bad["lease_id"] = json!("tampered");
    assert_eq!(
        verify_envelope(&bad, SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::BadSignature
    );
    assert_eq!(cache.len(), 0);

    // The genuine command with the same nonce still goes through.
    assert_eq!(
        verify_envelope(&signed(NOW), SECRET, WORKER, NOW, &mut cache),
        VerifyStatus::Ok
    );
}

#[test]
fn nonces_expire_out_of_the_cache() {
    let msg = signed(NOW);
    let mut cache = NonceCache::new(16);
    assert_eq!(verify_envelope(&msg, SECRET, WORKER, NOW, &mut cache), VerifyStatus::Ok);

    // Past the envelope's own expiry the nonce is forgotten, but the replay is then caught
    // by the expiry check instead — the two defences overlap by design.
    assert_eq!(
        verify_envelope(&msg, SECRET, WORKER, NOW + 3600, &mut cache),
        VerifyStatus::Expired
    );
    // Purging is lazy: it happens on the next insertion, not on the expiry check, which
    // is why the two defences have to overlap.
    assert_eq!(cache.len(), 1);
    let fresh = signed_with(NOW + 3600, "ffffffffffffffff", WORKER, SECRET);
    assert_eq!(
        verify_envelope(&fresh, SECRET, WORKER, NOW + 3600, &mut cache),
        VerifyStatus::Ok
    );
    assert_eq!(cache.len(), 1, "the stale entry was purged, not accumulated");
}

#[test]
fn nonce_cache_is_bounded_with_fifo_eviction() {
    let mut cache = NonceCache::new(4);
    for i in 0..64u64 {
        assert!(cache.check_and_insert(&format!("{i:016x}"), NOW + 60, NOW));
    }
    assert_eq!(cache.capacity(), 4);
    assert!(cache.len() <= 4);
    // The most recent are still remembered; the oldest were evicted.
    assert!(!cache.check_and_insert(&format!("{:016x}", 63u64), NOW + 60, NOW));
    assert!(cache.check_and_insert(&format!("{:016x}", 0u64), NOW + 60, NOW));
}

#[test]
fn zero_capacity_cache_still_works() {
    let mut cache = NonceCache::new(0);
    assert_eq!(cache.capacity(), 1);
    assert!(cache.check_and_insert("aaaaaaaaaaaaaaaa", NOW + 60, NOW));
}

#[test]
fn canonical_body_is_stable_under_key_insertion_order() {
    let a: Value = serde_json::from_str(
        r#"{"command":"release","lease_id":"L-1","duration_sec":10}"#,
    )
    .unwrap();
    let b: Value = serde_json::from_str(
        r#"{"duration_sec":10,"lease_id":"L-1","command":"release"}"#,
    )
    .unwrap();
    assert_eq!(canonical_body(&a), canonical_body(&b));
    // And the auth object never appears in what it signs.
    let signed = sign_command(&a, SECRET, WORKER, "cmd-1", NONCE, NOW, NOW + 60);
    assert_eq!(canonical_body(&signed), canonical_body(&a));
    assert!(!canonical_body(&signed).contains("auth"));
}

#[test]
fn signing_string_is_the_documented_shape() {
    assert_eq!(
        signing_string("w", "c", 1, 2, "ff", "{}"),
        "TMv1\nw\nc\n1\n2\nff\n{}"
    );
}

#[test]
fn mutating_command_policy() {
    // The historical marketplace flow is non-mutating.
    for command in ["register_ack", "assign_task", "release"] {
        assert!(!is_mutating_command(&json!({ "command": command })));
    }
    for action in ["pause", "resume"] {
        assert!(!is_mutating_command(&json!({ "action": action })));
    }
    // Anything that moves money or kills the process is mutating, and so is anything
    // unrecognised — fail closed.
    for action in ["shutdown", "set_config", "reboot", ""] {
        assert!(is_mutating_command(&json!({ "action": action })), "{action}");
    }
    assert!(is_mutating_command(&json!({ "command": "unknown_thing" })));
    assert!(is_mutating_command(&json!({})));
    assert!(is_mutating_command(&json!([])));
    assert!(is_mutating_command(&json!("string")));
    // A non-string discriminator must not be coerced into a known command.
    assert!(is_mutating_command(&json!({ "command": 1 })));
    assert!(is_mutating_command(&json!({ "action": ["pause"] })));
}

#[test]
fn field_validators() {
    assert!(is_hex_string("deadBEEF", 1, 16));
    assert!(!is_hex_string("deadbeeg", 1, 16));
    assert!(!is_hex_string("", 1, 16));
    assert!(!is_hex_string("abcdef", 8, 16));

    assert!(is_safe_identifier("lease-1_a.b", 1, 32));
    assert!(!is_safe_identifier("lease 1", 1, 32));
    assert!(!is_safe_identifier("lease\n1", 1, 32));
    assert!(!is_safe_identifier("lease\u{1b}[31m", 1, 32));
    assert!(!is_safe_identifier("", 1, 32));

    assert!(is_printable_ascii("version unsupported!", 1, 64));
    assert!(!is_printable_ascii("bell\u{7}", 1, 64));
    assert!(!is_printable_ascii("emoji \u{1f600}", 1, 64));
}

#[test]
fn constant_time_hex_equals_behaviour() {
    let a = hmac_sha256_hex(SECRET, "x");
    assert!(constant_time_hex_equals(&a, &a));
    assert!(constant_time_hex_equals(&a, &a.to_uppercase()));
    assert!(!constant_time_hex_equals(&a, &hmac_sha256_hex(SECRET, "y")));
    assert!(!constant_time_hex_equals(&a, "short"));
    assert!(!constant_time_hex_equals(&"z".repeat(64), &a));
}
