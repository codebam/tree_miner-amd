//! CPU Argon2id backend. Port of `src/Argon2idHasher.cpp` and
//! `src/hashapi/CpuHashBackend.cpp`.
//!
//! The C++ hasher calls libargon2's `argon2id_hash_encoded`, which formats the PHC string
//! itself. Here the raw digest is taken instead and the PHC string is assembled by
//! `tm_core::encoding::assemble_phc` — the same code path the GPU backend will use, so
//! CPU and GPU finds cannot disagree about their encoding.

use std::time::Instant;

use argon2::{Algorithm, Argon2, Params, Version};
use tm_core::encoding::{assemble_phc, HexError};
use tm_core::matching::{has_xuni_match, is_superblock_hash};

use crate::keygen::RandomHexKeyGenerator;
use crate::types::{
    HashBackend, HashMatch, HashRequest, HashResult, DEFAULT_HASH_LENGTH, HASH_API_KEY_LENGTH,
    MIN_ARGON2_CPU_DIFFICULTY,
};
use crate::validation::{normalize_hex, validate_request};

#[derive(Debug, thiserror::Error)]
pub enum HashError {
    #[error("salt_hex is not valid hex: {0}")]
    Salt(#[from] HexError),
    #[error("argon2 rejected the parameters: {0}")]
    Params(argon2::Error),
    #[error("argon2 failed: {0}")]
    Hash(argon2::Error),
}

/// One Argon2id hash in PHC form: v=19, t=1, p=1, m=`difficulty` KiB, 64-byte digest.
/// `salt_hex` is the address hex (with or without `0x`); `key` is hashed as ASCII, not
/// decoded — the server verifies it as the literal password string.
pub fn argon2id_phc(salt_hex: &str, key: &str, difficulty: u32) -> Result<String, HashError> {
    let salt_hex = normalize_hex(salt_hex);
    let salt_bytes = tm_core::encoding::hex_to_bytes(&salt_hex)?;
    let digest = argon2id_digest(&salt_bytes, key.as_bytes(), difficulty)?;
    let digest_b64 = tm_core::encoding::base64_encode(&digest);
    Ok(assemble_phc(difficulty, &salt_hex, &digest_b64)?)
}

fn argon2id_digest(
    salt: &[u8],
    password: &[u8],
    difficulty: u32,
) -> Result<[u8; DEFAULT_HASH_LENGTH], HashError> {
    let params = Params::new(difficulty, 1, 1, Some(DEFAULT_HASH_LENGTH)).map_err(HashError::Params)?;
    let hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut digest = [0u8; DEFAULT_HASH_LENGTH];
    hasher
        .hash_password_into(password, salt, &mut digest)
        .map_err(HashError::Hash)?;
    Ok(digest)
}

/// Port of `hashapi::appendMatches`. A find can satisfy both rules, and then it is reported
/// twice — once per pattern — because the two are submitted to different endpoints.
pub fn append_matches(
    request: &HashRequest,
    matches: &mut Vec<HashMatch>,
    key: &str,
    hash: &str,
    attempt_index: usize,
) {
    if hash.contains(&request.target_pattern) {
        matches.push(HashMatch {
            key: key.to_string(),
            hash: hash.to_string(),
            matched_pattern: request.target_pattern.clone(),
            attempt_index,
            is_superblock: is_superblock_hash(hash),
        });
    }
    if request.allow_xuni && has_xuni_match(hash) {
        matches.push(HashMatch {
            key: key.to_string(),
            hash: hash.to_string(),
            matched_pattern: "XUNI".to_string(),
            attempt_index,
            is_superblock: false,
        });
    }
}

#[derive(Debug, Default)]
pub struct CpuHashBackend;

impl HashBackend for CpuHashBackend {
    fn run_batch(&mut self, request: &HashRequest) -> HashResult {
        run_batch(request)
    }
}

fn millis_since(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

pub fn run_batch(request: &HashRequest) -> HashResult {
    let total_start = Instant::now();
    let mut result = HashResult {
        request_id: request.request_id.clone(),
        algorithm: request.algorithm.clone(),
        backend: if request.backend == "reference" {
            "reference".to_string()
        } else {
            "cpu".to_string()
        },
        device_id: request.device_id,
        batch_size: request.batch_size,
        ..Default::default()
    };

    let validation_start = Instant::now();
    let validation = validate_request(request);
    result.timings.validation_ms = millis_since(validation_start);

    let fail = |result: &mut HashResult, message: String| {
        result.error = message;
        result.timings.total_ms = millis_since(total_start);
    };

    if let Err(errors) = validation {
        fail(&mut result, errors.to_string());
        return result;
    }
    if request.backend == "cuda" {
        fail(
            &mut result,
            "cuda backend is not available in CpuHashBackend".to_string(),
        );
        return result;
    }
    if request.difficulty < MIN_ARGON2_CPU_DIFFICULTY {
        fail(
            &mut result,
            "cpu/reference difficulty must be at least 8".to_string(),
        );
        return result;
    }

    let start = Instant::now();

    let setup_start = Instant::now();
    let salt_hex = normalize_hex(&request.salt_hex);
    let prefix = normalize_hex(&request.key_prefix);
    let fixed_key = normalize_hex(&request.key);
    let single_key = !fixed_key.is_empty();
    let attempts = if single_key { 1 } else { request.batch_size };
    let salt_bytes = match tm_core::encoding::hex_to_bytes(&salt_hex) {
        Ok(bytes) => bytes,
        Err(err) => {
            fail(&mut result, HashError::Salt(err).to_string());
            return result;
        }
    };
    let mut key_generator = RandomHexKeyGenerator::new(&prefix, HASH_API_KEY_LENGTH);
    result.timings.setup_ms = millis_since(setup_start);

    let compute_start = Instant::now();
    for index in 0..attempts {
        let keygen_start = Instant::now();
        let key = if single_key {
            fixed_key.clone()
        } else {
            key_generator.next_random_key()
        };
        result.timings.keygen_ms += millis_since(keygen_start);

        let digest = match argon2id_digest(&salt_bytes, key.as_bytes(), request.difficulty) {
            Ok(digest) => digest,
            Err(err) => {
                fail(&mut result, err.to_string());
                return result;
            }
        };
        let digest_b64 = tm_core::encoding::base64_encode(&digest);
        let hash = match assemble_phc(request.difficulty, &salt_hex, &digest_b64) {
            Ok(hash) => hash,
            Err(err) => {
                fail(&mut result, HashError::Salt(err).to_string());
                return result;
            }
        };
        if single_key {
            result.hash.clone_from(&hash);
        }
        append_matches(request, &mut result.matches, &key, &hash, index);
    }
    result.timings.compute_ms = millis_since(compute_start);

    result.ok = true;
    result.attempts = attempts;
    result.batch_size = attempts;
    result.batch_size_min = attempts;
    result.batch_size_max = attempts;

    result.elapsed_ms = millis_since(start);
    result.timings.total_ms = millis_since(total_start);
    if result.elapsed_ms > 0.0 && result.attempts > 0 {
        result.hashrate = result.attempts as f64 / (result.elapsed_ms / 1000.0);
    }
    result
}
