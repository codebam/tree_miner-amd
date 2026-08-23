//! PHC-compatible unpadded base64 and hex helpers. Port of `src/hashapi/HashApiEncoding.cpp`
//! and `src/treeminer/PhcAssembler.h`.

const BASE64_CHARS: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 alphabet with the padding omitted — the form Argon2 PHC strings use, and
/// byte-for-byte what the C++ miner emits.
pub fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(base64_encoded_len(bytes.len()));
    let mut chunks = bytes.chunks_exact(3);
    for chunk in &mut chunks {
        let value = ((chunk[0] as u32) << 16) | ((chunk[1] as u32) << 8) | chunk[2] as u32;
        out.push(BASE64_CHARS[(value >> 18) as usize & 0x3f] as char);
        out.push(BASE64_CHARS[(value >> 12) as usize & 0x3f] as char);
        out.push(BASE64_CHARS[(value >> 6) as usize & 0x3f] as char);
        out.push(BASE64_CHARS[value as usize & 0x3f] as char);
    }
    match chunks.remainder() {
        [a] => {
            let value = (*a as u32) << 16;
            out.push(BASE64_CHARS[(value >> 18) as usize & 0x3f] as char);
            out.push(BASE64_CHARS[(value >> 12) as usize & 0x3f] as char);
        }
        [a, b] => {
            let value = ((*a as u32) << 16) | ((*b as u32) << 8);
            out.push(BASE64_CHARS[(value >> 18) as usize & 0x3f] as char);
            out.push(BASE64_CHARS[(value >> 12) as usize & 0x3f] as char);
            out.push(BASE64_CHARS[(value >> 6) as usize & 0x3f] as char);
        }
        _ => {}
    }
    out
}

pub fn base64_encoded_len(input_len: usize) -> usize {
    let full = input_len / 3;
    let remaining = input_len % 3;
    full * 4 + if remaining == 0 { 0 } else { remaining + 1 }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HexError {
    #[error("odd-length hex string")]
    OddLength,
    #[error("non-hex character")]
    NonHexCharacter,
}

pub fn hex_to_bytes(text: &str) -> Result<Vec<u8>, HexError> {
    if text.len() % 2 != 0 {
        return Err(HexError::OddLength);
    }
    hex::decode(text).map_err(|_| HexError::NonHexCharacter)
}

/// Assembles the complete PHC-encoded Argon2id string for a find from the parameters the
/// batch actually used. `hexsalt` is the 40-hex-char address without the `0x` prefix;
/// `digest_b64` is the unpadded base64 digest as produced on the GPU.
pub fn assemble_phc(memory_cost: u32, hexsalt: &str, digest_b64: &str) -> Result<String, HexError> {
    let salt_bytes = hex_to_bytes(hexsalt)?;
    let salt_b64 = base64_encode(&salt_bytes);
    Ok(format!(
        "$argon2id$v=19$m={memory_cost},t=1,p=1${salt_b64}${digest_b64}"
    ))
}

/// The digest half of a PHC string: everything after the last `$`.
pub fn phc_digest(phc: &str) -> Option<&str> {
    let (_, digest) = phc.rsplit_once('$')?;
    if digest.is_empty() {
        None
    } else {
        Some(digest)
    }
}
