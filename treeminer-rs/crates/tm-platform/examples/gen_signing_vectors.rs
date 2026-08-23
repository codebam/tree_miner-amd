//! Regenerate `treeminer/proto/signing_vectors.json`, the cross-language contract for the
//! HMAC-SHA256 command envelope.
//!
//! The Rust verifier ([`tm_platform::envelope::verify_envelope`]) is the canonical
//! implementation, so the fixture is produced *from* it rather than written by hand; the
//! `signing_vectors` integration test then reads the file back and re-verifies every
//! vector, which is what stops the two drifting apart.
//!
//! ```sh
//! ./rs cargo run -p tm-platform --example gen_signing_vectors
//! ```

use serde_json::{json, Map, Value};
use tm_platform::envelope::{canonical_body, hmac_sha256_hex, sign_command, signing_string};

const WORKER: &str = "rig-01";
const SECRET: &str = "correct horse battery staple";
/// A non-ASCII secret pins that the HMAC key is the UTF-8 bytes of the string.
const UNICODE_SECRET: &str = "s\u{e9}same-\u{5f00}\u{95e8}-\u{1f511}";
const NOW: i64 = 1_700_000_000;

const CONSUMER_ADDRESS: &str = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";
const SELF_ADDRESS: &str = "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359";

/// One vector, rendered exactly as it lands in the JSON file.
struct Vector {
    name: &'static str,
    description: &'static str,
    secret: &'static str,
    worker_id: &'static str,
    command_id: &'static str,
    nonce: &'static str,
    issued_at: i64,
    expires_at: i64,
    body: Value,
}

impl Vector {
    fn positive(&self) -> Value {
        let message = sign_command(
            &self.body,
            self.secret,
            self.worker_id,
            self.command_id,
            self.nonce,
            self.issued_at,
            self.expires_at,
        );
        self.render(message, true, "ok", None)
    }

    /// Render whatever `message` actually is, positive or not: the fields describe the
    /// message as DELIVERED, so a Python verifier can reproduce the same computation and
    /// arrive at the same verdict.
    fn render(
        &self,
        message: Value,
        must_verify: bool,
        expected_status: &str,
        note: Option<&str>,
    ) -> Value {
        let auth = message
            .get("auth")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let signature = auth
            .get("sig")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let canonical = canonical_body(&message);
        let signing = signing_string(
            auth.get("worker_id").and_then(Value::as_str).unwrap_or(""),
            auth.get("command_id").and_then(Value::as_str).unwrap_or(""),
            auth.get("issued_at").and_then(Value::as_i64).unwrap_or(0),
            auth.get("expires_at").and_then(Value::as_i64).unwrap_or(0),
            auth.get("nonce").and_then(Value::as_str).unwrap_or(""),
            &canonical,
        );

        let mut out = Map::new();
        out.insert("name".into(), json!(self.name));
        out.insert("description".into(), json!(self.description));
        out.insert("must_verify".into(), json!(must_verify));
        out.insert("expected_status".into(), json!(expected_status));
        if let Some(note) = note {
            out.insert("note".into(), json!(note));
        }
        out.insert("secret".into(), json!(self.secret));
        // The worker id the VERIFIER is configured with — an envelope addressed elsewhere
        // is refused, so this is not always the envelope's own `worker_id`.
        out.insert("expected_worker_id".into(), json!(WORKER));
        out.insert("verify_at".into(), json!(self.issued_at + 1));
        out.insert("worker_id".into(), auth.get("worker_id").cloned().unwrap_or(Value::Null));
        out.insert("command_id".into(), auth.get("command_id").cloned().unwrap_or(Value::Null));
        out.insert("issued_at".into(), auth.get("issued_at").cloned().unwrap_or(Value::Null));
        out.insert("expires_at".into(), auth.get("expires_at").cloned().unwrap_or(Value::Null));
        out.insert(
            "lifetime_sec".into(),
            json!(auth.get("expires_at").and_then(Value::as_i64).unwrap_or(0)
                - auth.get("issued_at").and_then(Value::as_i64).unwrap_or(0)),
        );
        out.insert("nonce".into(), auth.get("nonce").cloned().unwrap_or(Value::Null));
        out.insert("body".into(), body_without_auth(&message));
        out.insert("canonical_body".into(), json!(canonical));
        out.insert("signing_string".into(), json!(signing));
        out.insert("signature".into(), json!(signature));
        out.insert("message".into(), message);
        Value::Object(out)
    }
}

fn body_without_auth(message: &Value) -> Value {
    let mut body = message.clone();
    if let Some(obj) = body.as_object_mut() {
        obj.remove("auth");
    }
    body
}

fn vectors() -> Vec<Value> {
    let mut out = Vec::new();

    out.push(
        Vector {
            name: "minimal_command",
            description: "The smallest real command: one string field beside the envelope.",
            secret: SECRET,
            worker_id: WORKER,
            command_id: "cmd-0001",
            nonce: "9f86d081884c7d65",
            issued_at: NOW,
            expires_at: NOW + 60,
            body: json!({ "command": "release", "lease_id": "lease-1" }),
        }
        .positive(),
    );

    out.push(
        Vector {
            name: "assign_task",
            description:
                "The command that redirects the payout address. Mutating: refused outright \
                 unless this signature verifies.",
            secret: SECRET,
            worker_id: WORKER,
            command_id: "cmd-0002",
            nonce: "3b1f2c0d4e5a6b7c8d9e0f1a2b3c4d5e",
            issued_at: NOW,
            expires_at: NOW + 300,
            // Deliberately NOT in sorted order here: canonicalisation sorts it.
            body: json!({
                "command": "assign_task",
                "lease_id": "lease-7",
                "duration_sec": 3600,
                "consumer_id": "consumer-3",
                "consumer_address": CONSUMER_ADDRESS,
                "prefix": "a1b2c3d4e5f6a7b8",
            }),
        }
        .positive(),
    );

    out.push(
        Vector {
            name: "nested_config",
            description:
                "A nested object. Every level is key-sorted, not just the top one, and \
                 integers keep their exact spelling.",
            secret: SECRET,
            worker_id: WORKER,
            command_id: "cmd-0003",
            nonce: "aabbccddeeff00112233445566778899",
            issued_at: NOW,
            expires_at: NOW + 120,
            body: json!({
                "action": "set_config",
                "config": {
                    "prefix": "ff00",
                    "difficulty": 60000,
                    "address": SELF_ADDRESS,
                    "block_pattern": "XEN11",
                    "limits": { "min": 0, "max": 10000000, "nested": { "z": 1, "a": -1 } },
                    "flags": [true, false, null, 42],
                },
            }),
        }
        .positive(),
    );

    out.push(
        Vector {
            name: "unicode_string_field",
            description:
                "Non-ASCII text, an emoji outside the BMP, and the characters JSON must \
                 escape. The body is UTF-8 with NO \\uXXXX escaping of non-ASCII; the \
                 secret is non-ASCII too, so the HMAC key is its UTF-8 bytes.",
            secret: UNICODE_SECRET,
            worker_id: WORKER,
            command_id: "cmd-0004",
            nonce: "0f0e0d0c0b0a09080706050403020100",
            issued_at: NOW,
            expires_at: NOW + 60,
            body: json!({
                "command": "register_ack",
                "accepted": false,
                // caf\u{e9} \u{2615}, an em dash, CJK, an astral emoji, then the escapees:
                // quote, backslash, tab, newline, and a bare control character.
                "reason": "caf\u{e9} \u{2615} \u{2014} \u{6771}\u{4eac} \u{1f680} \" \\ \t \n \u{1}",
                "\u{5e73}\u{53f0}": "\u{4e2d}\u{6587}\u{952e}",
            }),
        }
        .positive(),
    );

    out.push(
        Vector {
            name: "empty_body",
            description:
                "Nothing but the envelope. The canonical body is the two characters `{}`.",
            secret: SECRET,
            worker_id: WORKER,
            command_id: "cmd-0005",
            nonce: "00112233445566778899aabbccddeeff",
            issued_at: NOW,
            expires_at: NOW + 60,
            body: json!({}),
        }
        .positive(),
    );

    // Large but legal: the whole published message must stay under MAX_PAYLOAD_BYTES
    // (64 KiB), which is checked on the raw bytes before JSON parsing.
    let filler = "x".repeat(60_000);
    out.push(
        Vector {
            name: "large_body",
            description:
                "A body just under the 65536-byte payload cap, to pin that nothing \
                 truncates or chunks the input to the MAC.",
            secret: SECRET,
            worker_id: WORKER,
            command_id: "cmd-0006",
            nonce: "cafebabedeadbeefcafebabedeadbeef",
            issued_at: NOW,
            expires_at: NOW + 60,
            body: json!({ "command": "register_ack", "accepted": true, "reason": filler }),
        }
        .positive(),
    );

    // --- Negative vectors ---

    let tampered = Vector {
        name: "body_edited_after_signing",
        description:
            "A genuine assign_task whose consumer_address was swapped in flight. The \
             signature is untouched and still well-formed; it must NOT verify.",
        secret: SECRET,
        worker_id: WORKER,
        command_id: "cmd-0007",
        nonce: "1234567890abcdef1234567890abcdef",
        issued_at: NOW,
        expires_at: NOW + 60,
        body: json!({
            "command": "assign_task",
            "lease_id": "lease-9",
            "consumer_id": "consumer-9",
            "consumer_address": SELF_ADDRESS,
            "duration_sec": 3600,
        }),
    };
    let mut edited = sign_command(
        &tampered.body,
        tampered.secret,
        tampered.worker_id,
        tampered.command_id,
        tampered.nonce,
        tampered.issued_at,
        tampered.expires_at,
    );
    edited["consumer_address"] = json!(CONSUMER_ADDRESS);
    out.push(tampered.render(
        edited,
        false,
        "bad signature",
        Some("consumer_address was changed from the self address to the consumer address \
              after signing; the `signature` field is the one produced over the ORIGINAL \
              body, so recomputing over `canonical_body` here yields a different digest"),
    ));

    let other_rig = Vector {
        name: "signed_for_another_worker",
        description:
            "A perfectly valid envelope, signed with the right secret — for a different \
             rig. Replaying it onto our topic must NOT work.",
        secret: SECRET,
        worker_id: "rig-99",
        command_id: "cmd-0008",
        nonce: "fedcba9876543210fedcba9876543210",
        issued_at: NOW,
        expires_at: NOW + 60,
        body: json!({ "action": "shutdown" }),
    };
    let message = sign_command(
        &other_rig.body,
        other_rig.secret,
        other_rig.worker_id,
        other_rig.command_id,
        other_rig.nonce,
        other_rig.issued_at,
        other_rig.expires_at,
    );
    out.push(other_rig.render(
        message,
        false,
        "wrong worker id",
        Some("the signature over this signing string is correct; the envelope is addressed \
              to rig-99 while the verifier is rig-01"),
    ));

    out
}

fn main() {
    let document = json!({
        "_comment":
            "Cross-language test vectors for the TreeMiner platform command envelope \
             (HMAC-SHA256, domain TMv1). GENERATED by \
             `cargo run -p tm-platform --example gen_signing_vectors`; do not edit by hand. \
             The canonicalisation rules are documented in proto/README.md, section \
             'Command Signing'.",
        "version": 1,
        "algorithm": "HMAC-SHA256",
        "domain": "TMv1",
        "signature_encoding": "lowercase hex, 64 characters",
        "signing_string_format":
            "TMv1\\n{worker_id}\\n{command_id}\\n{issued_at}\\n{expires_at}\\n{nonce}\\n{canonical_body}",
        "self_check": {
            "description":
                "HMAC-SHA256 of the literal signing string below, keyed with the literal \
                 secret below. Reproduce this first: it isolates the MAC from the JSON \
                 canonicalisation.",
            "secret": SECRET,
            "signing_string": signing_string(WORKER, "cmd-0000", 1, 2, "ff00ff00ff00ff00", "{}"),
            "signature": hmac_sha256_hex(
                SECRET,
                &signing_string(WORKER, "cmd-0000", 1, 2, "ff00ff00ff00ff00", "{}"),
            ),
        },
        "vectors": vectors(),
    });

    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/../../../treeminer/proto/signing_vectors.json",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let mut rendered = serde_json::to_string_pretty(&document).expect("vectors serialise");
    rendered.push('\n');
    std::fs::write(&path, rendered).expect("write vectors");
    println!("wrote {path}");
}
