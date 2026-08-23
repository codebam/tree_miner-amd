//! Random Argon2 password generation. Port of `src/RandomHexKeyGenerator.h`.
//!
//! The key is the server's dedupe key, so two miners that ever generate the same key throw
//! away one of the two finds. Upstream seeded `mt19937` from a single 32-bit value, giving
//! only 2^32 distinct key streams — a birthday collision across a fleet, not a theoretical
//! one (see `CHANGES-FROM-UPSTREAM.md`). `StdRng::from_entropy` seeds a 256-bit ChaCha
//! state straight from the OS, which removes the seed space as a factor entirely.

use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::types::HASH_API_KEY_LENGTH;

const HEX_CHARS: &[u8; 16] = b"0123456789abcdef";

pub struct RandomHexKeyGenerator {
    prefix: String,
    total_length: usize,
    rng: StdRng,
}

impl RandomHexKeyGenerator {
    pub fn new(initial_prefix: &str, key_length: usize) -> Self {
        Self {
            prefix: initial_prefix.to_ascii_lowercase(),
            total_length: key_length,
            rng: StdRng::from_entropy(),
        }
    }

    /// Deterministic construction, for tests that need a reproducible key stream.
    pub fn from_seed(initial_prefix: &str, key_length: usize, seed: [u8; 32]) -> Self {
        Self {
            prefix: initial_prefix.to_ascii_lowercase(),
            total_length: key_length,
            rng: StdRng::from_seed(seed),
        }
    }

    pub fn set_prefix(&mut self, new_prefix: &str) {
        self.prefix = new_prefix.to_ascii_lowercase();
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// A prefix at or beyond the full key length leaves no random suffix; the C++ version
    /// warns on stdout and returns the truncated prefix. Truncating silently is the same
    /// behaviour minus the log line, which does not belong in a library.
    pub fn next_random_key(&mut self) -> String {
        if self.prefix.len() >= self.total_length {
            return self.prefix[..self.total_length].to_string();
        }
        let mut key = String::with_capacity(self.total_length);
        key.push_str(&self.prefix);
        while key.len() < self.total_length {
            let mut random_bits: u64 = self.rng.gen();
            for _ in 0..16 {
                if key.len() >= self.total_length {
                    break;
                }
                key.push(HEX_CHARS[(random_bits & 0x0f) as usize] as char);
                random_bits >>= 4;
            }
        }
        key
    }
}

impl Default for RandomHexKeyGenerator {
    fn default() -> Self {
        Self::new("", HASH_API_KEY_LENGTH)
    }
}
