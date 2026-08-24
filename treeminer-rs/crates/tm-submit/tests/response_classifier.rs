//! Port of `tests/unit/submit/test_response_classifier.cpp` — one test per truth-table row,
//! with the REAL server strings from `repos/xenminer/gpage.py` (line refs in comments).

use tm_core::{FindKind, FindStatus};
use tm_submit::classifier::{
    classify, extract_json_field, is_difficulty_mismatch, is_xuni_window_rejection,
    parse_difficulty_hint, parse_retry_after_seconds, terminal_marker, TRANSPORT_ERROR,
};

fn c(status: i32, body: &str, kind: FindKind) -> tm_core::Classification {
    classify(status, body, kind, None)
}

// --- 200 -> AcceptedUnconfirmed, needs_lookup_confirmation (gpage.py:515, lying-200 risk
// at gpage.py:492-494) ---
#[test]
fn success_200_is_only_accepted_unconfirmed() {
    let r = c(
        200,
        r#"{"message": "Hash verified successfully and block saved."}"#,
        FindKind::Xen11,
    );
    assert_eq!(r.next_status, FindStatus::AcceptedUnconfirmed);
    assert!(r.needs_lookup_confirmation);
    assert!(r.server_difficulty_hint.is_none());

    let r = c(
        200,
        r#"{"message": "Hash verified successfully and block saved."}"#,
        FindKind::Xuni,
    );
    assert_eq!(r.next_status, FindStatus::AcceptedUnconfirmed);
    assert!(r.needs_lookup_confirmation);
}

// --- 400 "already exists" -> AcceptedUnconfirmed duplicate (gpage.py:510) ---
#[test]
fn duplicate_400_acks_via_lookup() {
    let r = c(
        400,
        r#"{"message": "Block already exists, continue"}"#,
        FindKind::Xen11,
    );
    assert_eq!(r.next_status, FindStatus::AcceptedUnconfirmed);
    assert!(r.needs_lookup_confirmation);
    assert!(r.reason.contains("duplicate"));

    // Substring fallback: a non-JSON body carrying the marker still classifies.
    let r = c(400, "Block already exists, continue", FindKind::Xuni);
    assert_eq!(r.next_status, FindStatus::AcceptedUnconfirmed);
    assert!(r.needs_lookup_confirmation);
}

// --- 409 "already exists" is the same duplicate ack. The reference source only emits 400;
// a 2026 third-party client also handles 409, which points at production drift, and the
// confirmation lookup is what actually decides either way. ---
#[test]
fn duplicate_409_acks_via_lookup_too() {
    for kind in [FindKind::Xen11, FindKind::Xuni] {
        let r = c(409, r#"{"message": "Block already exists, continue"}"#, kind);
        assert_eq!(r.next_status, FindStatus::AcceptedUnconfirmed);
        assert!(r.needs_lookup_confirmation);
        assert!(r.reason.contains("duplicate"));
    }
    // A 409 that is NOT a duplicate is still an unknown response, not an ack.
    assert_eq!(
        c(409, r#"{"message": "conflict"}"#, FindKind::Xen11).next_status,
        FindStatus::Quarantined
    );
}

// --- 401 difficulty message -> ParkedDifficulty + m={N} hint (gpage.py:416) ---
#[test]
fn difficulty_401_parks_and_surfaces_the_hint() {
    let r = c(
        401,
        r#"{"message": "Hash does not contain 'm=104000'. Your memory_cost setting in your miner will be autoadjusted."}"#,
        FindKind::Xen11,
    );
    assert_eq!(r.next_status, FindStatus::ParkedDifficulty);
    assert_eq!(r.server_difficulty_hint, Some(104_000));
    assert!(!r.needs_lookup_confirmation);

    // Substring fallback (no JSON wrapper).
    let r = c(
        401,
        "Hash does not contain 'm=99000'. Your memory_cost setting in your miner will be autoadjusted.",
        FindKind::Xen11,
    );
    assert_eq!(r.next_status, FindStatus::ParkedDifficulty);
    assert_eq!(r.server_difficulty_hint, Some(99_000));
}

// --- 401 XUNI window, current (gpage.py:434) and legacy (gpage.py:497) strings ---
#[test]
fn xuni_window_401_parks_xuni() {
    let r = c(
        401,
        r#"{"message": "XUNI Submitted outside of proper time frame."}"#,
        FindKind::Xuni,
    );
    assert_eq!(r.next_status, FindStatus::ParkedXuniWindow);

    let r = c(
        401,
        r#"{"message": "XUNI found outside of time window"}"#,
        FindKind::Xuni,
    );
    assert_eq!(r.next_status, FindStatus::ParkedXuniWindow);
}

// --- a XUNI-window rejection for a XEN11 record is impossible (docs/05 §2) ---
#[test]
fn xuni_window_401_for_xen11_is_quarantined_loudly() {
    for body in [
        r#"{"message": "XUNI Submitted outside of proper time frame."}"#,
        r#"{"message": "XUNI found outside of time window"}"#,
    ] {
        let r = c(401, body, FindKind::Xen11);
        assert_eq!(r.next_status, FindStatus::Quarantined);
        assert!(r.reason.contains("IMPOSSIBLE"));
    }
}

// --- TERMINAL class: responses no retry can ever turn into an accept ---
//
// Table-driven, one row per verbatim server string, with the status code gpage.py actually
// answers with. Each must reach PermanentlyInvalid for BOTH kinds and must keep the body in
// the reason, because that reason is the whole diagnostic record of a written-off find.
#[test]
fn terminal_rejections_are_permanently_invalid_for_both_kinds() {
    let rows: &[(i32, &str, &str)] = &[
        (400, r#"{"error": "Invalid key format"}"#, "Invalid key format"), // gpage.py:391
        (400, r#"{"error": "Invalid salt format"}"#, "Invalid salt format"), // :395
        (
            400,
            r#"{"error": "Missing hash_to_verify, key, or account"}"#, // :399
            "Missing hash_to_verify, key, or account",
        ),
        (
            401,
            r#"{"message": "Hash does not contain any of the valid targets ['XEN11'] in the last 87 characters. Adjust target_substr in your miner."}"#, // :439
            "valid targets",
        ),
        (
            400,
            r#"{"message": "Length of hash_to_verify should not be greater than 150 characters."}"#, // :445
            "150 characters",
        ),
        (
            401,
            r#"{"message": "Hash verification failed."}"#, // :519
            "Hash verification failed",
        ),
    ];
    for (status, body, marker) in rows {
        for kind in [FindKind::Xen11, FindKind::Xuni] {
            let r = c(*status, body, kind);
            assert_eq!(
                r.next_status,
                FindStatus::PermanentlyInvalid,
                "{body} ({kind:?}) should be terminal"
            );
            assert!(!r.needs_lookup_confirmation);
            assert!(r.server_difficulty_hint.is_none());
            assert!(r.reason.contains(marker), "reason lost the body: {}", r.reason);
        }
        assert!(terminal_marker(body).is_some(), "table miss: {body}");
    }

    // Raw-body (non-JSON) forms classify identically — the classifier falls back to the
    // whole body when there is no message/error field.
    assert_eq!(
        c(400, "Invalid key format", FindKind::Xen11).next_status,
        FindStatus::PermanentlyInvalid
    );

    // The length message is matched on its stable tail, so the exact number the server
    // interpolates in front of it does not matter.
    assert_eq!(
        c(
            400,
            "Length of hash_to_verify should not be greater than 150 characters.",
            FindKind::Xuni
        )
        .next_status,
        FindStatus::PermanentlyInvalid
    );
}

// --- the terminal class must be unreachable from "the server/network was unwell" ---
#[test]
fn transient_failures_never_become_terminal_even_carrying_a_terminal_string() {
    // A 5xx or 408/425 body is a statement about the server, not about our payload: a
    // reverse proxy interpolating an upstream error page must not write off a real find.
    for status in [500, 502, 503, 408, 425] {
        assert_eq!(
            c(status, r#"{"message": "Hash verification failed."}"#, FindKind::Xen11).next_status,
            FindStatus::Pending,
            "http {status} must stay transient"
        );
    }
    // Transport failure and blank bodies short-circuit before any taxonomy runs.
    assert_eq!(
        c(TRANSPORT_ERROR, "Invalid key format", FindKind::Xen11).next_status,
        FindStatus::Pending
    );
    assert_eq!(c(400, "   ", FindKind::Xen11).next_status, FindStatus::Pending);
    // 429 is backpressure, never a verdict on the payload.
    assert_eq!(
        c(429, r#"{"message": "Invalid salt format"}"#, FindKind::Xen11).next_status,
        FindStatus::Pending
    );
}

// --- the non-terminal classes stay non-terminal ---
#[test]
fn window_and_difficulty_rejections_are_never_terminal() {
    // XUNI window: re-park, retryable in the next window.
    for body in [
        "XUNI Submitted outside of proper time frame.",
        "XUNI found outside of time window",
        "rejected: bad time frame",
        "rejected: bad time window",
    ] {
        assert!(is_xuni_window_rejection(body), "{body}");
        assert_eq!(
            c(401, body, FindKind::Xuni).next_status,
            FindStatus::ParkedXuniWindow,
            "{body}"
        );
        assert!(terminal_marker(body).is_none(), "{body}");
    }

    // Difficulty mismatch: park until the floor falls.
    for body in [
        "Hash does not contain 'm=104000'. Your memory_cost setting in your miner will be autoadjusted.",
        "your hash does not contain m=104000 anywhere",
    ] {
        assert!(is_difficulty_mismatch(body), "{body}");
        let r = c(401, body, FindKind::Xen11);
        assert_eq!(r.next_status, FindStatus::ParkedDifficulty, "{body}");
        assert_eq!(r.server_difficulty_hint, Some(104_000));
        assert!(terminal_marker(body).is_none(), "{body}");
    }

    // Shared "does not contain" prefix: the valid-targets message is terminal and must win
    // over the loose difficulty test, and the difficulty message must not be swallowed by
    // the terminal table.
    assert!(!is_difficulty_mismatch(
        "Hash does not contain any of the valid targets ['XEN11'] in the last 87 characters."
    ));
}

// --- 429 -> Pending, Retry-After honored via the reason hint ---
#[test]
fn rate_limit_429_stays_pending_and_honors_retry_after() {
    let r = c(429, r#"{"message": "slow down"}"#, FindKind::Xen11);
    assert_eq!(r.next_status, FindStatus::Pending);

    let r = classify(429, r#"{"message": "slow down"}"#, FindKind::Xen11, Some("30"));
    assert_eq!(r.next_status, FindStatus::Pending);
    assert!(r.reason.contains("retry_after_s=30"));

    // HTTP-date Retry-After is unsupported: still Pending, no hint.
    let r = classify(
        429,
        r#"{"message": "slow down"}"#,
        FindKind::Xen11,
        Some("Wed, 21 Oct 2015 07:28:00 GMT"),
    );
    assert_eq!(r.next_status, FindStatus::Pending);
    assert!(!r.reason.contains("retry_after_s="));
}

// --- 408/425/5xx/transport error/empty body -> Pending with backoff ---
#[test]
fn transport_class_failures_stay_pending() {
    assert_eq!(
        c(TRANSPORT_ERROR, "", FindKind::Xen11).next_status,
        FindStatus::Pending
    );
    assert_eq!(
        c(408, "Request Timeout", FindKind::Xen11).next_status,
        FindStatus::Pending
    );
    assert_eq!(
        c(425, "Too Early", FindKind::Xen11).next_status,
        FindStatus::Pending
    );
    assert_eq!(
        c(500, "<html>Internal Server Error</html>", FindKind::Xen11).next_status,
        FindStatus::Pending
    );
    assert_eq!(
        c(502, "Bad Gateway", FindKind::Xuni).next_status,
        FindStatus::Pending
    );
    assert_eq!(
        c(503, r#"{"message": "Service Unavailable"}"#, FindKind::Xen11).next_status,
        FindStatus::Pending
    );
    // Empty body, even on 200: never conclusive (the mock server's "empty-body" fault).
    assert_eq!(c(200, "", FindKind::Xen11).next_status, FindStatus::Pending);
    assert_eq!(
        c(200, "   \n", FindKind::Xen11).next_status,
        FindStatus::Pending
    );
}

// --- any other 4xx / unrecognized body -> Quarantined (never dropped, never terminal) ---
#[test]
fn unknown_responses_quarantine_never_drop() {
    // A 400 the taxonomy has never seen: operator-visible, not written off.
    assert_eq!(
        c(400, r#"{"error": "some new validation rule"}"#, FindKind::Xen11).next_status,
        FindStatus::Quarantined
    );
    // A 401 with no m= digits, no XUNI marker and no terminal marker.
    assert_eq!(
        c(401, r#"{"message": "brand new rejection"}"#, FindKind::Xen11).next_status,
        FindStatus::Quarantined
    );
    assert_eq!(
        c(403, "Forbidden", FindKind::Xen11).next_status,
        FindStatus::Quarantined
    );
    assert_eq!(
        c(404, "Not Found", FindKind::Xen11).next_status,
        FindStatus::Quarantined
    );
    assert_eq!(
        c(301, "Moved Permanently", FindKind::Xen11).next_status,
        FindStatus::Quarantined
    );
}

// --- structured parse before substring fallback ---
#[test]
fn json_message_field_wins_over_incidental_body_text() {
    // The marker appears only in a different JSON field; the "message" field is
    // unrecognized -> Quarantined (no naive whole-body substring hit).
    let r = c(
        401,
        r#"{"debug": "Hash verification failed.", "message": "totally new response"}"#,
        FindKind::Xen11,
    );
    assert_eq!(r.next_status, FindStatus::Quarantined);

    // Escaped quotes inside the message decode correctly.
    let r = c(
        401,
        "{\"message\": \"Hash does not contain \\u0027m=123456\\u0027.\"}",
        FindKind::Xen11,
    );
    assert_eq!(r.next_status, FindStatus::ParkedDifficulty);
    assert_eq!(r.server_difficulty_hint, Some(123_456));
}

#[test]
fn helper_coverage() {
    assert_eq!(
        extract_json_field(r#"{"difficulty": "104000"}"#, "difficulty").as_deref(),
        Some("104000")
    );
    assert_eq!(
        extract_json_field(r#"{"a": 1, "difficulty": 98000}"#, "difficulty").as_deref(),
        Some("98000")
    );
    assert!(extract_json_field("not json", "difficulty").is_none());
    assert!(extract_json_field("{}", "difficulty").is_none());
    // A structured value is not a scalar: it can never be mistaken for a message.
    assert!(extract_json_field(r#"{"message": {"nested": 1}}"#, "message").is_none());

    assert_eq!(parse_difficulty_hint("m=1234 tail"), Some(1234));
    assert!(parse_difficulty_hint("no hint here").is_none());
    assert!(parse_difficulty_hint("m=99999999999999").is_none()); // > u32
    assert_eq!(parse_difficulty_hint("memory m=x then m=42"), Some(42));

    assert!(terminal_marker("Invalid key format").is_some());
    assert!(terminal_marker("perfectly fine").is_none());
    assert!(!is_xuni_window_rejection("Hash verification failed."));
    assert!(!is_difficulty_mismatch("Hash verification failed."));

    assert_eq!(parse_retry_after_seconds("120"), Some(120));
    assert_eq!(parse_retry_after_seconds(" 5 "), Some(5));
    assert!(parse_retry_after_seconds("Wed, 21 Oct 2015 07:28:00 GMT").is_none());
}

#[test]
fn classification_is_deterministic() {
    let body = r#"{"message": "Hash does not contain 'm=104000'."}"#;
    let a = c(401, body, FindKind::Xen11);
    let b = c(401, body, FindKind::Xen11);
    assert_eq!(a, b);
}
