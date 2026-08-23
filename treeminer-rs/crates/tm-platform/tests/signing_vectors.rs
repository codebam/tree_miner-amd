//! The cross-language signing contract: `treeminer/proto/signing_vectors.json`.
//!
//! The FastAPI server signs commands in Python; this crate verifies them in Rust. The
//! fixture is the only thing that keeps the two honest, so it is checked here against the
//! live verifier rather than trusted. If this test fails, either the envelope
//! implementation changed (regenerate with
//! `cargo run -p tm-platform --example gen_signing_vectors`, and tell whoever maintains
//! the Python signer, because it is a wire-format change) or the file was edited by hand.

use serde_json::Value;
use tm_platform::envelope::{
    canonical_body, hmac_sha256_hex, signing_string, verify_envelope, NonceCache, VerifyStatus,
    MAX_PAYLOAD_BYTES,
};

const VECTORS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../treeminer/proto/signing_vectors.json"
);

fn load() -> Value {
    let raw = std::fs::read_to_string(VECTORS).unwrap_or_else(|error| {
        panic!(
            "cannot read {VECTORS} ({error}) — regenerate it with \
             `cargo run -p tm-platform --example gen_signing_vectors`"
        )
    });
    serde_json::from_str(&raw).expect("signing_vectors.json is valid JSON")
}

fn vectors(document: &Value) -> &Vec<Value> {
    document["vectors"].as_array().expect("vectors array")
}

fn field<'a>(vector: &'a Value, key: &str) -> &'a str {
    vector[key]
        .as_str()
        .unwrap_or_else(|| panic!("vector {} has no string {key}", vector["name"]))
}

/// The whole point of the file: what the Python signer produces must be what this verifier
/// accepts, and what it must reject must be rejected.
#[test]
fn every_vector_verifies_exactly_as_it_claims() {
    let document = load();
    assert!(!vectors(&document).is_empty(), "no vectors in the file");

    let mut positives = 0;
    let mut negatives = 0;
    for vector in vectors(&document) {
        let name = field(vector, "name");
        // A fresh cache per vector: replay is a separate property, tested elsewhere, and
        // sharing one here would make the file order-dependent.
        let mut nonces = NonceCache::new(64);
        let status = verify_envelope(
            &vector["message"],
            field(vector, "secret"),
            field(vector, "expected_worker_id"),
            vector["verify_at"].as_i64().expect("verify_at"),
            &mut nonces,
        );
        assert_eq!(
            status.name(),
            field(vector, "expected_status"),
            "vector {name} verified as {status}"
        );

        match vector["must_verify"].as_bool().expect("must_verify") {
            true => {
                assert_eq!(status, VerifyStatus::Ok, "positive vector {name} was refused");
                positives += 1;
            }
            false => {
                assert_ne!(status, VerifyStatus::Ok, "negative vector {name} was ACCEPTED");
                negatives += 1;
            }
        }
    }
    assert!(positives >= 5, "expected the documented positive coverage");
    assert!(negatives >= 1, "the file must carry a negative vector");
}

/// The intermediate values are published so a Python implementer can bisect a mismatch;
/// they are only useful if they are the ones this crate actually computes.
#[test]
fn the_published_intermediates_match_the_implementation() {
    let document = load();
    for vector in vectors(&document) {
        let name = field(vector, "name");
        let message = &vector["message"];
        let auth = &message["auth"];

        let canonical = canonical_body(message);
        assert_eq!(canonical, field(vector, "canonical_body"), "{name}: body");
        assert!(
            !canonical.contains("\"auth\""),
            "{name}: the auth object must be removed before signing"
        );

        let signing = signing_string(
            auth["worker_id"].as_str().expect("worker_id"),
            auth["command_id"].as_str().expect("command_id"),
            auth["issued_at"].as_i64().expect("issued_at"),
            auth["expires_at"].as_i64().expect("expires_at"),
            auth["nonce"].as_str().expect("nonce"),
            &canonical,
        );
        assert_eq!(signing, field(vector, "signing_string"), "{name}: signing string");

        // `signature` is what the vector's own envelope carries; for the tampered vector
        // that is deliberately NOT the MAC of the delivered body.
        assert_eq!(
            field(vector, "signature"),
            auth["sig"].as_str().expect("sig"),
            "{name}: signature field disagrees with the envelope"
        );
        let recomputed = hmac_sha256_hex(field(vector, "secret"), &signing);
        let should_match = field(vector, "expected_status") != "bad signature";
        assert_eq!(
            recomputed == field(vector, "signature"),
            should_match,
            "{name}: MAC over the delivered body"
        );
    }
}

/// The self-check exists so a Python implementer can prove their HMAC before touching
/// JSON canonicalisation at all. It has to be right.
#[test]
fn the_self_check_is_reproducible() {
    let document = load();
    let check = &document["self_check"];
    assert_eq!(
        hmac_sha256_hex(
            check["secret"].as_str().expect("secret"),
            check["signing_string"].as_str().expect("signing_string"),
        ),
        check["signature"].as_str().expect("signature")
    );
    assert_eq!(document["domain"], "TMv1");
    assert_eq!(document["algorithm"], "HMAC-SHA256");
}

/// Every vector must be a message the miner would actually accept off the wire: the size
/// gate runs on the raw bytes before anything parses them.
#[test]
fn every_vector_fits_under_the_payload_cap() {
    let document = load();
    for vector in vectors(&document) {
        let encoded = vector["message"].to_string();
        assert!(
            encoded.len() <= MAX_PAYLOAD_BYTES,
            "vector {} is {} bytes, over the {MAX_PAYLOAD_BYTES}-byte cap",
            field(vector, "name"),
            encoded.len()
        );
    }
    // ...and one of them exercises the large end of that range.
    assert!(
        vectors(&document)
            .iter()
            .any(|v| v["message"].to_string().len() > MAX_PAYLOAD_BYTES / 2),
        "no vector covers a large-but-legal body"
    );
}
