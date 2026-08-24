//! Pure, deterministic classification of XenBlocks `/verify` responses. Port of
//! `src/submit/ResponseClassifier.{h,cpp}`.
//!
//! Server facts this encodes (line refs into `repos/xenminer/gpage.py`):
//!   200  "Hash verified successfully and block saved." (:515) — can LIE: the server
//!        answers 200 even when its insert retries were exhausted (:492-494), so a 200 is
//!        `AcceptedUnconfirmed` + a `/get_block` lookup, never a straight ack.
//!   400  "Block already exists, continue" (:510) — UNIQUE-key duplicate: a prior attempt
//!        landed, so confirm it the same way. Also accepted on 409: the reference source
//!        only ever emits 400, but a 2026 third-party client handles 409 as well, which
//!        suggests production has drifted behind some proxy. Treating both as the duplicate
//!        ack is safe — the confirmation lookup is what actually decides, and a wrong guess
//!        here costs one `/get_block` and re-pends.
//!   401  "Hash does not contain 'm={N}'. ..." (:416) — N is the server's CURRENT
//!        difficulty, and the server check is strictly-`<`, so the find becomes valid
//!        again once difficulty falls back to <= its m.
//!   401  XUNI window rejections (:434 current, :497 legacy) — server-clock gated.
//!   4xx  TERMINAL rejections ([`TERMINAL_MARKERS`]) — malformed payloads and hashes the
//!        server will refuse identically forever: retrying cannot fix them, so they land in
//!        `PermanentlyInvalid` with the body preserved for diagnosis rather than looping.
//!   429 / 408 / 425 / 5xx / transport failure / empty body — Pending with backoff.
//!   anything else — Quarantined, never silently dropped.

use tm_core::{Classification, FindKind, FindStatus};

/// Sentinel `http_status` for transport-level failures (connect error, timeout, DNS).
pub const TRANSPORT_ERROR: i32 = 0;

fn is_blank(s: &str) -> bool {
    s.bytes().all(|c| c.is_ascii_whitespace())
}

fn skip_ws(s: &[u8], i: &mut usize) {
    while *i < s.len() && s[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

/// Parse a JSON string literal at `s[i] == b'"'`, advancing `i` past the closing quote.
/// Non-ASCII `\uXXXX` escapes become `?` — the server never emits them, and widening what
/// an attacker-controlled body may decode to buys nothing.
fn parse_json_string(s: &[u8], i: &mut usize) -> Option<String> {
    if *i >= s.len() || s[*i] != b'"' {
        return None;
    }
    *i += 1;
    let mut out: Vec<u8> = Vec::new();
    while *i < s.len() {
        let c = s[*i];
        if c == b'"' {
            *i += 1;
            return Some(String::from_utf8_lossy(&out).into_owned());
        }
        if c == b'\\' {
            *i += 1;
            if *i >= s.len() {
                return None;
            }
            match s[*i] {
                b'"' => out.push(b'"'),
                b'\\' => out.push(b'\\'),
                b'/' => out.push(b'/'),
                b'b' => out.push(0x08),
                b'f' => out.push(0x0c),
                b'n' => out.push(b'\n'),
                b'r' => out.push(b'\r'),
                b't' => out.push(b'\t'),
                b'u' => {
                    if *i + 4 >= s.len() {
                        return None;
                    }
                    let mut code: u32 = 0;
                    for k in 1..=4 {
                        let h = s[*i + k] as char;
                        let digit = h.to_digit(16)?;
                        code = (code << 4) | digit;
                    }
                    out.push(if code < 0x80 { code as u8 } else { b'?' });
                    *i += 4;
                }
                _ => return None,
            }
            *i += 1;
        } else {
            out.push(c);
            *i += 1;
        }
    }
    None // unterminated
}

/// Skip one JSON value (string, number, literal, object, array) starting at `s[i]`.
fn skip_json_value(s: &[u8], i: &mut usize) -> bool {
    skip_ws(s, i);
    if *i >= s.len() {
        return false;
    }
    let c = s[*i];
    if c == b'"' {
        return parse_json_string(s, i).is_some();
    }
    if c == b'{' || c == b'[' {
        let open = c;
        let close = if c == b'{' { b'}' } else { b']' };
        let mut depth = 0i32;
        while *i < s.len() {
            let d = s[*i];
            if d == b'"' {
                if parse_json_string(s, i).is_none() {
                    return false;
                }
                continue;
            }
            if d == open {
                depth += 1;
            }
            if d == close {
                depth -= 1;
                if depth == 0 {
                    *i += 1;
                    return true;
                }
            }
            *i += 1;
        }
        return false;
    }
    while *i < s.len() && s[*i] != b',' && s[*i] != b'}' && s[*i] != b']' && !s[*i].is_ascii_whitespace() {
        *i += 1;
    }
    true
}

/// Capture a scalar value (string, number, literal) as text; `None` for object/array.
fn capture_scalar(s: &[u8], i: &mut usize) -> Option<String> {
    skip_ws(s, i);
    if *i >= s.len() {
        return None;
    }
    if s[*i] == b'"' {
        return parse_json_string(s, i);
    }
    if s[*i] == b'{' || s[*i] == b'[' {
        skip_json_value(s, i); // structured value: not a scalar
        return None;
    }
    let start = *i;
    while *i < s.len() && s[*i] != b',' && s[*i] != b'}' && s[*i] != b']' && !s[*i].is_ascii_whitespace() {
        *i += 1;
    }
    Some(String::from_utf8_lossy(&s[start..*i]).into_owned())
}

/// Extract a top-level scalar field from a JSON object body. `None` when the body is not a
/// JSON object or lacks the key.
pub fn extract_json_field(body: &str, key: &str) -> Option<String> {
    let s = body.as_bytes();
    let mut i = 0usize;
    skip_ws(s, &mut i);
    if i >= s.len() || s[i] != b'{' {
        return None;
    }
    i += 1;
    skip_ws(s, &mut i);
    if i < s.len() && s[i] == b'}' {
        return None; // empty object
    }
    while i < s.len() {
        skip_ws(s, &mut i);
        let k = parse_json_string(s, &mut i)?;
        skip_ws(s, &mut i);
        if i >= s.len() || s[i] != b':' {
            return None;
        }
        i += 1;
        if k == key {
            return capture_scalar(s, &mut i);
        }
        if !skip_json_value(s, &mut i) {
            return None;
        }
        skip_ws(s, &mut i);
        if i < s.len() && s[i] == b',' {
            i += 1;
            continue;
        }
        break;
    }
    None
}

/// The server wraps human messages as `{"message": ...}` and validation errors as
/// `{"error": ...}`. Structured parse first; the classifier falls back to the raw body.
pub fn extract_json_message(body: &str) -> Option<String> {
    extract_json_field(body, "message").or_else(|| extract_json_field(body, "error"))
}

/// Parse the first `m=<digits>` occurrence (the current-difficulty hint embedded in the
/// 401 difficulty message). `None` when absent or out of `u32` range.
pub fn parse_difficulty_hint(message: &str) -> Option<u32> {
    let s = message.as_bytes();
    let mut pos = 0usize;
    while pos + 1 < s.len() {
        if !(s[pos] == b'm' && s[pos + 1] == b'=') {
            pos += 1;
            continue;
        }
        let mut d = pos + 2;
        if d >= s.len() || !s[d].is_ascii_digit() {
            pos += 1;
            continue;
        }
        let mut value: u64 = 0;
        while d < s.len() && s[d].is_ascii_digit() {
            value = value * 10 + u64::from(s[d] - b'0');
            if value > u64::from(u32::MAX) {
                return None; // absurd; treat as no hint
            }
            d += 1;
        }
        return Some(value as u32);
    }
    None
}

/// Parse a `Retry-After` header in delay-seconds form. HTTP-date form returns `None`.
pub fn parse_retry_after_seconds(header_value: &str) -> Option<i64> {
    let s = header_value.as_bytes();
    let mut i = 0usize;
    skip_ws(s, &mut i);
    if i >= s.len() || !s[i].is_ascii_digit() {
        return None; // HTTP-date form unsupported; caller falls back to backoff
    }
    let mut value: i64 = 0;
    while i < s.len() && s[i].is_ascii_digit() {
        if value > 100_000_000 {
            return None;
        }
        value = value * 10 + i64::from(s[i] - b'0');
        i += 1;
    }
    skip_ws(s, &mut i);
    if i != s.len() {
        return None;
    }
    Some(value)
}

/// Server responses that can NEVER succeed on retry. Each is verbatim from the reference
/// server (line refs into `repos/xenminer/gpage.py`); the stored substring is the stable
/// part of the message, because several of them interpolate server-side values.
///
/// The classifier only consults this table for a conclusive 4xx. A transport failure, a
/// blank body or a 5xx is a statement about the SERVER, never about our payload, and those
/// paths return `Pending` long before we get here — a terminal verdict must never be
/// reachable from "the network was down".
pub const TERMINAL_MARKERS: &[&str] = &[
    "Invalid key format",                          // :391 (400)
    "Invalid salt format",                         // :395 (400)
    "Missing hash_to_verify, key, or account",     // :399 (400)
    "Hash does not contain any of the valid targets", // :439 (401)
    "should not be greater than 150 characters",   // :445 — full text names the length
    "Hash verification failed",                    // :519 (401)
];

/// First [`TERMINAL_MARKERS`] entry the message carries, if any.
pub fn terminal_marker(message: &str) -> Option<&'static str> {
    TERMINAL_MARKERS
        .iter()
        .copied()
        .find(|marker| message.contains(marker))
}

/// XUNI submitted outside the server-clock :55-:05 window (`:434` current wording, `:497`
/// legacy). NOT permanent: the same find is valid in the next window, so it re-parks. The
/// two loose forms catch a reworded message that keeps the same meaning.
pub fn is_xuni_window_rejection(message: &str) -> bool {
    message.contains("outside of proper time frame")
        || message.contains("outside of time window")
        || message.contains("time frame")
        || message.contains("time window")
}

/// "Hash does not contain 'm={N}'" (:416). NOT permanent: the server test is strictly-`<`,
/// so the find becomes valid again the moment difficulty falls back to <= its baked-in m.
/// The loose form requires BOTH halves, so it cannot swallow the "valid targets" rejection
/// (which is terminal and is checked first anyway).
pub fn is_difficulty_mismatch(message: &str) -> bool {
    message.contains("Hash does not contain 'm=")
        || (message.contains("does not contain") && message.contains("m="))
}

fn pending(reason: String) -> Classification {
    Classification {
        next_status: FindStatus::Pending,
        server_difficulty_hint: None,
        needs_lookup_confirmation: false,
        reason,
    }
}

fn quarantined(reason: String) -> Classification {
    Classification {
        next_status: FindStatus::Quarantined,
        server_difficulty_hint: None,
        needs_lookup_confirmation: false,
        reason,
    }
}

/// Classify one `/verify` response. Fully deterministic: same inputs, same answer.
pub fn classify(
    http_status: i32,
    body: &str,
    kind: FindKind,
    retry_after: Option<&str>,
) -> Classification {
    // Transport-level failure (connect error / timeout / DNS): retry forever with backoff.
    if http_status <= 0 {
        return pending("transport failure; will retry with backoff".to_string());
    }

    // An empty body is indistinguishable from a proxy/serving failure — never conclusive.
    if body.is_empty() || is_blank(body) {
        return pending(format!(
            "empty response body (http {http_status}); will retry with backoff"
        ));
    }

    // Structured parse first, raw-body substring fallback second — never both.
    let message = extract_json_message(body).unwrap_or_else(|| body.to_string());

    if http_status == 200 {
        return Classification {
            next_status: FindStatus::AcceptedUnconfirmed,
            server_difficulty_hint: None,
            needs_lookup_confirmation: true,
            reason: "http 200; awaiting /get_block confirmation".to_string(),
        };
    }

    if http_status == 429 {
        let mut c = pending("rate limited (429)".to_string());
        if let Some(secs) = retry_after.and_then(parse_retry_after_seconds) {
            c.reason += &format!("; retry_after_s={secs}");
        }
        return c;
    }

    if http_status == 408 || http_status == 425 || http_status >= 500 {
        return pending(format!(
            "server unhealthy (http {http_status}); will retry with backoff"
        ));
    }

    // Accepted-as-duplicate. 400 is what the reference server emits; 409 is the semantically
    // correct code and is handled because production may already answer it.
    if (http_status == 400 || http_status == 409) && message.contains("already exists") {
        return Classification {
            next_status: FindStatus::AcceptedUnconfirmed,
            server_difficulty_hint: None,
            needs_lookup_confirmation: true,
            reason: "duplicate key (already exists); confirming via /get_block".to_string(),
        };
    }

    // TERMINAL, before any of the retryable taxonomies: these responses describe our
    // payload, not the server's mood, and resubmitting one forever is the failure mode this
    // class exists to prevent. The whole message is preserved so the journal row and the
    // operator log say exactly WHY a find was written off.
    //
    // Checked ahead of the difficulty test on purpose: "Hash does not contain any of the
    // valid targets ..." shares the "does not contain" prefix with the difficulty rejection
    // but is not fixable by waiting.
    if let Some(marker) = terminal_marker(&message) {
        return Classification {
            next_status: FindStatus::PermanentlyInvalid,
            server_difficulty_hint: None,
            needs_lookup_confirmation: false,
            reason: format!(
                "server rejected permanently (http {http_status}, matched \"{marker}\"); \
                 retrying cannot change this answer: {message}"
            ),
        };
    }

    if is_xuni_window_rejection(&message) {
        if kind == FindKind::Xuni {
            return Classification {
                next_status: FindStatus::ParkedXuniWindow,
                server_difficulty_hint: None,
                needs_lookup_confirmation: false,
                reason: "XUNI outside server time window; parked for a later window".to_string(),
            };
        }
        // docs/05 §2: a XEN11 submission can never receive this response. If it does,
        // the server has changed — quarantine and make it loud.
        return quarantined(format!(
            "IMPOSSIBLE: XUNI-window rejection for a XEN11 record — server semantics changed, investigate: {message}"
        ));
    }

    // Difficulty mismatch: park until the floor falls, never a terminal verdict. The hint is
    // the server's CURRENT difficulty, which the manager also feeds to the un-park path.
    if is_difficulty_mismatch(&message) || http_status == 401 {
        if let Some(hint) = parse_difficulty_hint(&message) {
            return Classification {
                next_status: FindStatus::ParkedDifficulty,
                server_difficulty_hint: Some(hint),
                needs_lookup_confirmation: false,
                reason: format!(
                    "difficulty too low (server currently at m={hint}); parked until difficulty falls"
                ),
            };
        }
    }

    quarantined(format!(
        "unrecognized response (http {http_status}): {message}"
    ))
}
