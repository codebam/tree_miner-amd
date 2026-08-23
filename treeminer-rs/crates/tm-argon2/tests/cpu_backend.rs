//! CPU backend parity tests. The oracle is `fixtures/argon2_vectors.json`, produced by the
//! C++ `hash-one` command and cross-checked cpu == gpu == gpu+first-blocks, so a failure
//! here means the Rust port diverged from the binary that is actually mining.

use std::collections::HashSet;

use serde::Deserialize;
use tm_argon2::{
    argon2id_phc, run_batch, validate_request, HashBackend, HashRequest, RandomHexKeyGenerator,
    HASH_API_KEY_LENGTH,
};
use tm_core::encoding::phc_digest;

#[derive(Debug, Deserialize)]
struct Vector {
    salt_hex: String,
    key: String,
    difficulty: u32,
    phc: String,
    digest_b64: String,
}

#[derive(Debug, Deserialize)]
struct Vectors {
    vectors: Vec<Vector>,
}

fn load_vectors() -> Vec<Vector> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/argon2_vectors.json");
    let text = std::fs::read_to_string(path).expect("fixture readable");
    serde_json::from_str::<Vectors>(&text)
        .expect("fixture parses")
        .vectors
}

/// The vector named in PORT.md, kept as its own test so the headline reference value is
/// visible in the source and not only in a data file.
const PORT_MD_SALT: &str = "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc";
const PORT_MD_KEY: &str = "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f";
const PORT_MD_DIGEST: &str =
    "2PKfnaEX2s+Yf/Drzi92D8HJ+B6K+FppyT7g5glp2knIMlFGWhnyOb9r1QIPf0GaVUEw8KumqQZ/pK2dkNTDxA";

#[test]
fn port_md_reference_vector_reproduces_byte_for_byte() {
    let phc = argon2id_phc(PORT_MD_SALT, PORT_MD_KEY, 8).expect("hash");
    assert_eq!(
        phc,
        "$argon2id$v=19$m=8,t=1,p=1$5LsYR4G7yccATo2v1Km0nSA7ybw$".to_string() + PORT_MD_DIGEST
    );
    assert_eq!(phc_digest(&phc), Some(PORT_MD_DIGEST));
}

#[test]
fn every_fixture_vector_reproduces_byte_for_byte() {
    let vectors = load_vectors();
    assert!(vectors.len() >= 24, "fixture should carry the full matrix");
    for vector in &vectors {
        let phc = argon2id_phc(&vector.salt_hex, &vector.key, vector.difficulty)
            .unwrap_or_else(|err| panic!("m={} failed: {err}", vector.difficulty));
        assert_eq!(phc, vector.phc, "m={}", vector.difficulty);
        assert_eq!(phc_digest(&phc), Some(vector.digest_b64.as_str()));
    }
}

#[test]
fn run_batch_single_key_matches_the_fixture() {
    for vector in load_vectors() {
        let request = HashRequest {
            salt_hex: vector.salt_hex.clone(),
            key: vector.key.clone(),
            difficulty: vector.difficulty,
            batch_size: 64,
            ..Default::default()
        };
        let result = run_batch(&request);
        assert!(result.ok, "error: {}", result.error);
        assert_eq!(result.hash, vector.phc);
        // A fixed key collapses the batch to a single attempt.
        assert_eq!(result.attempts, 1);
        assert_eq!(result.batch_size, 1);
        assert_eq!(result.batch_size_min, 1);
        assert_eq!(result.batch_size_max, 1);
    }
}

#[test]
fn run_batch_accepts_the_0x_prefixed_and_uppercase_salt_form() {
    let vector = &load_vectors()[0];
    let request = HashRequest {
        salt_hex: format!("0x{}", vector.salt_hex.to_ascii_uppercase()),
        key: vector.key.to_ascii_uppercase(),
        difficulty: vector.difficulty,
        ..Default::default()
    };
    let result = run_batch(&request);
    assert!(result.ok, "error: {}", result.error);
    assert_eq!(result.hash, vector.phc);
}

#[test]
fn backend_trait_dispatches_to_the_same_code_path() {
    let vector = &load_vectors()[0];
    let request = HashRequest {
        salt_hex: vector.salt_hex.clone(),
        key: vector.key.clone(),
        difficulty: vector.difficulty,
        ..Default::default()
    };
    let mut backend = tm_argon2::CpuHashBackend;
    assert_eq!(backend.run_batch(&request).hash, run_batch(&request).hash);
}

#[test]
fn generated_keys_honour_the_prefix_and_fill_out_the_key_length() {
    let request = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        key_prefix: "DEADBEEF".to_string(),
        difficulty: 8,
        batch_size: 8,
        // Match everything so each attempt is reported and its key is observable.
        target_pattern: "$argon2id$".to_string(),
        allow_xuni: false,
        ..Default::default()
    };
    let result = run_batch(&request);
    assert!(result.ok, "error: {}", result.error);
    assert_eq!(result.attempts, 8);
    assert_eq!(result.matches.len(), 8);
    for (index, found) in result.matches.iter().enumerate() {
        assert_eq!(found.attempt_index, index);
        // normalizeHex lowercases the prefix before it reaches the generator.
        assert!(found.key.starts_with("deadbeef"), "key: {}", found.key);
        assert_eq!(found.key.len(), HASH_API_KEY_LENGTH);
        assert!(found.key.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

#[test]
fn batch_reports_no_matches_when_the_target_is_absent() {
    let request = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        key: PORT_MD_KEY.to_string(),
        difficulty: 8,
        allow_xuni: false,
        ..Default::default()
    };
    let result = run_batch(&request);
    assert!(result.ok);
    assert!(result.matches.is_empty());
    assert!(result.hashrate > 0.0);
    assert!(result.timings.total_ms >= result.timings.compute_ms);
}

/// `appendMatches` runs against the assembled PHC string, so a synthetic hash exercises the
/// reporting rules without needing to mine a real XEN11.
fn synthetic_matches(hash: &str, target: &str, allow_xuni: bool) -> Vec<tm_argon2::HashMatch> {
    let request = HashRequest {
        target_pattern: target.to_string(),
        allow_xuni,
        ..Default::default()
    };
    let mut matches = Vec::new();
    tm_argon2::append_matches(&request, &mut matches, "key", hash, 7);
    matches
}

#[test]
fn xen11_match_is_reported_with_its_attempt_index() {
    let matches = synthetic_matches("$argon2id$v=19$m=8,t=1,p=1$salt$aaXEN11bb", "XEN11", false);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].matched_pattern, "XEN11");
    assert_eq!(matches[0].attempt_index, 7);
    assert_eq!(matches[0].key, "key");
    assert!(!matches[0].is_superblock);
}

#[test]
fn xen11_match_carrying_fifty_uppercase_letters_is_a_superblock() {
    let hash = format!("$XEN11${}", "A".repeat(50));
    let matches = synthetic_matches(&hash, "XEN11", false);
    assert_eq!(matches.len(), 1);
    assert!(matches[0].is_superblock);
}

#[test]
fn xuni_match_is_reported_only_when_allowed() {
    let hash = "$argon2id$v=19$m=8,t=1,p=1$salt$aaXUNI7bb";
    assert!(synthetic_matches(hash, "XEN11", false).is_empty());

    let matches = synthetic_matches(hash, "XEN11", true);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].matched_pattern, "XUNI");
    assert_eq!(matches[0].attempt_index, 7);
    // XUNI finds are never superblocks, whatever the digest looks like.
    assert!(!matches[0].is_superblock);
}

#[test]
fn a_hash_satisfying_both_rules_is_reported_twice() {
    let matches = synthetic_matches("$XEN11$XUNI3$", "XEN11", true);
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].matched_pattern, "XEN11");
    assert_eq!(matches[0].attempt_index, 7);
    assert_eq!(matches[1].matched_pattern, "XUNI");
    assert_eq!(matches[1].attempt_index, 7);
}

#[test]
fn validation_rejects_each_malformed_input() {
    let ok = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        difficulty: 8,
        ..Default::default()
    };
    assert!(validate_request(&ok).is_ok());

    let cases: Vec<(HashRequest, &str)> = vec![
        (
            HashRequest {
                algorithm: "scrypt".to_string(),
                ..ok.clone()
            },
            "unsupported algorithm: scrypt",
        ),
        (
            HashRequest {
                backend: "opencl".to_string(),
                ..ok.clone()
            },
            "unsupported backend: opencl",
        ),
        (
            HashRequest {
                salt_hex: String::new(),
                ..ok.clone()
            },
            "salt_hex is required",
        ),
        (
            HashRequest {
                salt_hex: "e4bb184781bbc9c7004e8dafd4a9b49d203bc9b".to_string(),
                ..ok.clone()
            },
            "salt_hex must contain an even number of hex characters",
        ),
        (
            HashRequest {
                salt_hex: "e4bb1847".to_string(),
                ..ok.clone()
            },
            "salt_hex must be at least 16 hex characters",
        ),
        (
            HashRequest {
                salt_hex: "zzzz1847zzzz1847".to_string(),
                ..ok.clone()
            },
            "salt_hex must contain only hex characters",
        ),
        (
            HashRequest {
                key_prefix: "a".repeat(65),
                ..ok.clone()
            },
            "key_prefix cannot exceed 64 hex characters",
        ),
        (
            HashRequest {
                key_prefix: "zz".to_string(),
                ..ok.clone()
            },
            "key_prefix must contain only hex characters",
        ),
        (
            HashRequest {
                key: "abcd".to_string(),
                ..ok.clone()
            },
            "key must contain exactly 64 hex characters",
        ),
        (
            HashRequest {
                key: "z".repeat(64),
                ..ok.clone()
            },
            "key must contain only hex characters",
        ),
        (
            HashRequest {
                key: PORT_MD_KEY.to_string(),
                key_prefix: "ffff".to_string(),
                ..ok.clone()
            },
            "key must start with key_prefix when both are provided",
        ),
        (
            HashRequest {
                target_pattern: String::new(),
                ..ok.clone()
            },
            "target_pattern is required",
        ),
        (
            HashRequest {
                target_pattern: "X".repeat(129),
                ..ok.clone()
            },
            "target_pattern is too long",
        ),
        (
            HashRequest {
                difficulty: 0,
                ..ok.clone()
            },
            "difficulty must be greater than zero",
        ),
        (
            HashRequest {
                batch_size: 0,
                ..ok.clone()
            },
            "batch_size must be greater than zero",
        ),
        (
            HashRequest {
                batch_size: 10_001,
                ..ok.clone()
            },
            "cpu batch_size exceeds safe limit",
        ),
        (
            HashRequest {
                device_id: -1,
                ..ok.clone()
            },
            "device_id must be non-negative",
        ),
        (
            HashRequest {
                gpu_first_blocks: true,
                ..ok.clone()
            },
            "gpu_first_blocks requires backend=cuda",
        ),
    ];

    for (request, expected) in cases {
        let errors = validate_request(&request).expect_err(expected);
        assert!(
            errors.messages().iter().any(|m| m == expected),
            "expected {expected:?}, got {:?}",
            errors.messages()
        );
    }
}

#[test]
fn the_cpu_batch_limit_does_not_apply_to_the_cuda_backend() {
    let request = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        backend: "cuda".to_string(),
        batch_size: 10_001,
        difficulty: 8,
        ..Default::default()
    };
    assert!(validate_request(&request).is_ok());
}

#[test]
fn multiple_validation_failures_are_joined_like_join_errors() {
    let request = HashRequest {
        salt_hex: String::new(),
        difficulty: 0,
        ..Default::default()
    };
    let errors = validate_request(&request).expect_err("two failures");
    assert_eq!(errors.messages().len(), 2);
    assert_eq!(
        errors.to_string(),
        "salt_hex is required; difficulty must be greater than zero"
    );
}

#[test]
fn a_validation_failure_is_reported_in_the_result_not_as_a_panic() {
    let result = run_batch(&HashRequest::default());
    assert!(!result.ok);
    assert_eq!(result.error, "salt_hex is required");
    assert_eq!(result.attempts, 0);
}

#[test]
fn the_cpu_backend_refuses_the_cuda_backend_and_sub_minimum_difficulty() {
    let cuda = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        backend: "cuda".to_string(),
        difficulty: 8,
        ..Default::default()
    };
    assert_eq!(
        run_batch(&cuda).error,
        "cuda backend is not available in CpuHashBackend"
    );

    let too_easy = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        difficulty: 7,
        ..Default::default()
    };
    assert_eq!(
        run_batch(&too_easy).error,
        "cpu/reference difficulty must be at least 8"
    );
}

#[test]
fn the_reference_backend_keeps_its_own_label() {
    let request = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        key: PORT_MD_KEY.to_string(),
        backend: "reference".to_string(),
        difficulty: 8,
        ..Default::default()
    };
    assert_eq!(run_batch(&request).backend, "reference");
}

#[test]
fn two_generated_key_batches_do_not_collide() {
    let batch = |_| {
        let mut generator = RandomHexKeyGenerator::new("", HASH_API_KEY_LENGTH);
        (0..512)
            .map(|_| generator.next_random_key())
            .collect::<Vec<_>>()
    };
    let first: Vec<String> = batch(0);
    let second: Vec<String> = batch(1);

    let unique: HashSet<&String> = first.iter().chain(second.iter()).collect();
    assert_eq!(unique.len(), first.len() + second.len());
    assert!(first.iter().all(|k| k.len() == HASH_API_KEY_LENGTH));
}

#[test]
fn a_prefix_at_or_beyond_the_key_length_truncates_instead_of_generating() {
    let mut generator = RandomHexKeyGenerator::new(&"AB".repeat(40), HASH_API_KEY_LENGTH);
    let key = generator.next_random_key();
    assert_eq!(key, "ab".repeat(32));
    assert_eq!(key.len(), HASH_API_KEY_LENGTH);
}

#[test]
fn a_seeded_generator_is_reproducible_and_a_reseed_changes_the_stream() {
    let seed = [7u8; 32];
    let mut a = RandomHexKeyGenerator::from_seed("", HASH_API_KEY_LENGTH, seed);
    let mut b = RandomHexKeyGenerator::from_seed("", HASH_API_KEY_LENGTH, seed);
    assert_eq!(a.next_random_key(), b.next_random_key());

    let mut c = RandomHexKeyGenerator::from_seed("", HASH_API_KEY_LENGTH, [8u8; 32]);
    assert_ne!(a.next_random_key(), c.next_random_key());

    b.set_prefix("CAFE");
    assert_eq!(b.prefix(), "cafe");
    assert!(b.next_random_key().starts_with("cafe"));
}

#[test]
fn the_result_serializes_with_the_cpp_json_keys() {
    let request = HashRequest {
        salt_hex: PORT_MD_SALT.to_string(),
        key: PORT_MD_KEY.to_string(),
        difficulty: 8,
        ..Default::default()
    };
    let json = serde_json::to_value(run_batch(&request)).expect("serializes");
    let object = json.as_object().expect("object");
    for key in [
        "request_id",
        "ok",
        "error",
        "algorithm",
        "backend",
        "device_id",
        "batch_size",
        "batch_size_min",
        "batch_size_max",
        "attempts",
        "first_block_dynamic_chunk_size",
        "first_block_dynamic_chunk_auto",
        "first_block_worker_count",
        "first_block_chunk_size",
        "first_block_dynamic_chunk_size_min",
        "first_block_dynamic_chunk_size_max",
        "first_block_chunk_size_min",
        "first_block_chunk_size_max",
        "gpu_first_blocks",
        "elapsed_ms",
        "hashrate",
        "timings",
        "hash",
        "matches",
    ] {
        assert!(object.contains_key(key), "missing {key}");
    }
    assert_eq!(object.len(), 24);
    assert_eq!(json["hash"], PORT_MD_DIGEST_PHC);
}

const PORT_MD_DIGEST_PHC: &str = concat!(
    "$argon2id$v=19$m=8,t=1,p=1$5LsYR4G7yccATo2v1Km0nSA7ybw$",
    "2PKfnaEX2s+Yf/Drzi92D8HJ+B6K+FppyT7g5glp2knIMlFGWhnyOb9r1QIPf0GaVUEw8KumqQZ/pK2dkNTDxA"
);
