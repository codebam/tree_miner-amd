//! Ethereum address validation and the salt derived from it. Port of
//! `src/EthereumAddressValidator.cpp`: the miner requires a full EIP-55 checksummed
//! address, because the address is compacted into the Argon2 salt and a typo would bind
//! every find to an address nobody controls.

use sha3::{Digest, Keccak256};

pub fn keccak256_hex(input: &str) -> String {
    let mut hasher = Keccak256::new();
    hasher.update(input.as_bytes());
    hex::encode_upper(hasher.finalize())
}

/// EIP-55 checksum form of a `0x`-prefixed address.
pub fn to_checksum_address(address: &str) -> Option<String> {
    let body = address.strip_prefix("0x")?;
    if body.len() != 40 || !body.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let lower = body.to_ascii_lowercase();
    let hash = keccak256_hex(&lower);
    let mut out = String::with_capacity(42);
    out.push_str("0x");
    for (index, c) in lower.chars().enumerate() {
        if hash.as_bytes()[index] >= b'8' {
            out.push(c.to_ascii_uppercase());
        } else {
            out.push(c);
        }
    }
    Some(out)
}

pub fn is_valid_ethereum_address(address: &str) -> bool {
    match to_checksum_address(address) {
        Some(checksummed) => checksummed == address,
        None => false,
    }
}

/// The Argon2 salt for an address: its 40 hex characters, without the `0x` prefix.
pub fn salt_hex_for_address(address: &str) -> Option<&str> {
    let body = address.strip_prefix("0x")?;
    (body.len() == 40 && body.chars().all(|c| c.is_ascii_hexdigit())).then_some(body)
}
