//! Request validation. Port of `src/hashapi/HashApiValidation.cpp`.
//!
//! The C++ side collects every problem before returning so an operator sees all of them at
//! once; that is worth keeping, so the error type carries a list and renders it with the
//! same `"; "` join `joinErrors` used.

use crate::types::{
    HashRequest, HASH_API_KEY_LENGTH, MAX_CPU_BATCH_SIZE, MAX_TARGET_PATTERN_LENGTH,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{}", .messages.join("; "))]
pub struct ValidationErrors {
    pub messages: Vec<String>,
}

impl ValidationErrors {
    pub fn messages(&self) -> &[String] {
        &self.messages
    }
}

pub fn is_hex_string(value: &str) -> bool {
    value.chars().all(|c| c.is_ascii_hexdigit())
}

/// Strips a `0x`/`0X` prefix and lowercases. Applied before every other hex check so the
/// operator can paste an address in any of the forms the ecosystem uses.
pub fn normalize_hex(value: &str) -> String {
    let body = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .unwrap_or(value);
    body.to_ascii_lowercase()
}

fn is_supported_algorithm(algorithm: &str) -> bool {
    algorithm == crate::types::DEFAULT_ALGORITHM
}

fn is_supported_backend(backend: &str) -> bool {
    matches!(backend, "cpu" | "reference" | "cuda")
}

pub fn validate_request(request: &HashRequest) -> Result<(), ValidationErrors> {
    let mut messages = Vec::new();

    if !is_supported_algorithm(&request.algorithm) {
        messages.push(format!("unsupported algorithm: {}", request.algorithm));
    }
    if !is_supported_backend(&request.backend) {
        messages.push(format!("unsupported backend: {}", request.backend));
    }

    let salt = normalize_hex(&request.salt_hex);
    if salt.is_empty() {
        messages.push("salt_hex is required".to_string());
    } else {
        if salt.len() % 2 != 0 {
            messages.push("salt_hex must contain an even number of hex characters".to_string());
        }
        if salt.len() < 16 {
            messages.push("salt_hex must be at least 16 hex characters".to_string());
        }
        if !is_hex_string(&salt) {
            messages.push("salt_hex must contain only hex characters".to_string());
        }
    }

    let prefix = normalize_hex(&request.key_prefix);
    if !prefix.is_empty() {
        if prefix.len() > HASH_API_KEY_LENGTH {
            messages.push("key_prefix cannot exceed 64 hex characters".to_string());
        }
        if !is_hex_string(&prefix) {
            messages.push("key_prefix must contain only hex characters".to_string());
        }
    }

    let key = normalize_hex(&request.key);
    if !key.is_empty() {
        if key.len() != HASH_API_KEY_LENGTH {
            messages.push("key must contain exactly 64 hex characters".to_string());
        }
        if !is_hex_string(&key) {
            messages.push("key must contain only hex characters".to_string());
        }
        if !prefix.is_empty() && !key.starts_with(&prefix) {
            messages.push("key must start with key_prefix when both are provided".to_string());
        }
    }

    if request.target_pattern.is_empty() {
        messages.push("target_pattern is required".to_string());
    }
    if request.target_pattern.len() > MAX_TARGET_PATTERN_LENGTH {
        messages.push("target_pattern is too long".to_string());
    }

    if request.difficulty == 0 {
        messages.push("difficulty must be greater than zero".to_string());
    }

    if request.batch_size == 0 {
        messages.push("batch_size must be greater than zero".to_string());
    }
    if (request.backend == "cpu" || request.backend == "reference")
        && request.batch_size > MAX_CPU_BATCH_SIZE
    {
        messages.push("cpu batch_size exceeds safe limit".to_string());
    }

    if request.device_id < 0 {
        messages.push("device_id must be non-negative".to_string());
    }

    if request.gpu_first_blocks && request.backend != "cuda" {
        // Wording matches the C++ miner verbatim (HashApiValidation.cpp:120) because the
        // hash CLI's JSON is diffed against that binary. "cuda" here is the backend's wire
        // name, not a claim about the vendor — the CLI advertises it as `gpu`.
        messages.push("gpu_first_blocks requires backend=cuda".to_string());
    }

    if messages.is_empty() {
        Ok(())
    } else {
        Err(ValidationErrors { messages })
    }
}

pub fn is_valid_request(request: &HashRequest) -> bool {
    validate_request(request).is_ok()
}
