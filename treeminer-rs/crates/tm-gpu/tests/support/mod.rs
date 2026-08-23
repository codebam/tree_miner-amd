//! Test-only Argon2 host helpers and fixture loading.
//!
//! `ReferenceArgon2Host` is a straight transcription of `Argon2Params::initialHash`,
//! `fillFirstBlocks` and `finalize` from the C++ miner. It lives in the tests, not in the
//! crate, because production wires in `tm-argon2`'s implementation of the same trait — but
//! the GPU path cannot be tested at all without *some* CPU side, and an independent
//! transcription is a better oracle than the crate under test.

#![allow(dead_code)]

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

use tm_gpu::{Argon2Host, Argon2Shape, HostError};

const BLAKE2B_OUT_BYTES: usize = 64;
const PREHASH_DIGEST_LENGTH: usize = 64;
const PREHASH_SEED_LENGTH: usize = 72;
const BLOCK_SIZE: usize = 1024;

pub struct ReferenceArgon2Host;

impl Argon2Host for ReferenceArgon2Host {
    fn fill_first_blocks(
        &self,
        out: &mut [u8],
        password: &[u8],
        salt: &[u8],
        shape: &Argon2Shape,
    ) -> Result<(), HostError> {
        if out.len() != 2 * BLOCK_SIZE {
            return Err(format!("first-block slot is {} bytes, want 2048", out.len()).into());
        }
        if shape.lanes != 1 {
            return Err("the reference host only implements lanes=1".into());
        }
        let mut seed = [0u8; PREHASH_SEED_LENGTH];
        initial_hash(&mut seed[..PREHASH_DIGEST_LENGTH], password, salt, shape);

        seed[PREHASH_DIGEST_LENGTH..PREHASH_DIGEST_LENGTH + 4].copy_from_slice(&0u32.to_le_bytes());
        seed[PREHASH_DIGEST_LENGTH + 4..].copy_from_slice(&0u32.to_le_bytes());
        digest_long(&mut out[..BLOCK_SIZE], &seed);

        seed[PREHASH_DIGEST_LENGTH..PREHASH_DIGEST_LENGTH + 4].copy_from_slice(&1u32.to_le_bytes());
        seed[PREHASH_DIGEST_LENGTH + 4..].copy_from_slice(&0u32.to_le_bytes());
        digest_long(&mut out[BLOCK_SIZE..], &seed);
        Ok(())
    }

    fn finalize(&self, last_block: &[u8], out: &mut [u8]) -> Result<(), HostError> {
        if last_block.len() != BLOCK_SIZE {
            return Err(format!("final block is {} bytes, want 1024", last_block.len()).into());
        }
        digest_long(out, last_block);
        Ok(())
    }
}

fn blake2b(out: &mut [u8], parts: &[&[u8]]) {
    let mut hasher = Blake2bVar::new(out.len()).expect("blake2b output length is in range");
    for part in parts {
        hasher.update(part);
    }
    hasher
        .finalize_variable(out)
        .expect("output buffer matches the configured length");
}

/// Argon2's variable-length hash H'.
fn digest_long(out: &mut [u8], input: &[u8]) {
    let out_len = out.len();
    let length_prefix = (out_len as u32).to_le_bytes();
    if out_len <= BLAKE2B_OUT_BYTES {
        blake2b(out, &[&length_prefix, input]);
        return;
    }
    let mut buffer = [0u8; BLAKE2B_OUT_BYTES];
    blake2b(&mut buffer, &[&length_prefix, input]);
    let half = BLAKE2B_OUT_BYTES / 2;
    out[..half].copy_from_slice(&buffer[..half]);

    let mut written = half;
    let mut to_produce = out_len - half;
    while to_produce > BLAKE2B_OUT_BYTES {
        let previous = buffer;
        blake2b(&mut buffer, &[&previous]);
        out[written..written + half].copy_from_slice(&buffer[..half]);
        written += half;
        to_produce -= half;
    }
    let mut tail = vec![0u8; to_produce];
    blake2b(&mut tail, &[&buffer]);
    out[written..].copy_from_slice(&tail);
}

/// Argon2's H0. Note the salt is the raw address bytes, not its hex text.
fn initial_hash(out: &mut [u8], password: &[u8], salt: &[u8], shape: &Argon2Shape) {
    let mut header = Vec::with_capacity(7 * 4);
    for value in [
        shape.lanes,
        shape.output_length,
        shape.memory_cost,
        shape.passes,
        shape.version,
        shape.type_,
        password.len() as u32,
    ] {
        header.extend_from_slice(&value.to_le_bytes());
    }
    let salt_length = (salt.len() as u32).to_le_bytes();
    let empty_lengths = [0u8; 8];
    blake2b(
        out,
        &[&header, password, &salt_length, salt, &empty_lengths],
    );
}

/// One entry of `fixtures/argon2_vectors.json`.
#[derive(Debug, serde::Deserialize)]
pub struct Vector {
    pub salt_hex: String,
    pub key: String,
    pub difficulty: u32,
    pub phc: String,
    pub digest_b64: String,
}

#[derive(Debug, serde::Deserialize)]
struct Fixture {
    vectors: Vec<Vector>,
}

pub fn load_vectors() -> Vec<Vector> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/argon2_vectors.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("reading {}: {error}", path.display()));
    let fixture: Fixture = serde_json::from_str(&text).expect("fixture is valid JSON");
    fixture.vectors
}

/// Returns the first GPU, or `None` after printing why the test is being skipped.
pub fn first_gpu_or_skip(test: &str) -> Option<tm_gpu::Device> {
    match tm_gpu::Device::enumerate() {
        Ok(devices) if !devices.is_empty() => devices.into_iter().next(),
        Ok(_) => {
            eprintln!("skipping {test}: no GPU present");
            None
        }
        Err(error) => {
            eprintln!("skipping {test}: GPU unavailable ({error})");
            None
        }
    }
}

/// Serialises the GPU tests inside one test binary. The card is shared with whatever else
/// is mining on the box, so two tests each grabbing a difficulty-60000 pool would make the
/// free-memory assertions meaningless (and can genuinely run the card out of VRAM).
pub fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}
