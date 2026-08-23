//! The CPU half of Argon2: first blocks in, digest out. Port of
//! `src/argon2params.cpp` (`Argon2Params::initialHash`, `fillFirstBlocks`, `finalize`,
//! `digestLong`) and the `Blake2b` wrapper in `src/blake2b.cpp`.
//!
//! This is the production implementation of [`tm_core::Argon2Host`]: the GPU chains the
//! memory blocks, this crate produces the two blocks it starts from and reduces the block
//! it ends with. It deliberately lives in `tm-argon2` rather than `tm-gpu` so a CPU-only
//! build never links HIP to hash.
//!
//! The batch entry point spreads the work across threads, as `fillPasswordBlocks` in
//! `src/hashapi/CudaHashBackend.cpp` does. That matters on ROCm, where the device-side
//! first-blocks kernel is disabled and every batch pays for these on the CPU.

use std::collections::VecDeque;
use std::sync::Mutex;

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use tm_core::argon2host::{
    Argon2Host, Argon2Shape, HostError, ARGON2_BLOCK_SIZE, INPUT_BLOCKS_PER_JOB,
};

const BLAKE2B_OUT_BYTES: usize = 64;
/// `argon2::ARGON2_PREHASH_DIGEST_LENGTH`.
const PREHASH_DIGEST_LENGTH: usize = 64;
/// `argon2::ARGON2_PREHASH_SEED_LENGTH` — H0 plus the block index and the lane index.
const PREHASH_SEED_LENGTH: usize = 72;

/// Below this many jobs the C++ miner does not bother with threads
/// (`kMinParallelFirstBlockAttempts`).
pub const MIN_PARALLEL_FIRST_BLOCK_ATTEMPTS: usize = 8;

/// Port of `firstBlockWorkerCount`. `worker_cap` of 0 means "no operator cap".
pub fn first_block_worker_count(attempts: usize, worker_cap: usize) -> usize {
    if attempts < MIN_PARALLEL_FIRST_BLOCK_ATTEMPTS {
        return 1;
    }
    let hardware_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    if hardware_threads < 2 {
        return 1;
    }
    let mut worker_count = attempts.min(hardware_threads);
    if worker_cap > 0 {
        worker_count = worker_count.min(worker_cap);
    }
    worker_count.max(1)
}

/// Port of `firstBlockChunkSize`: an even split, rounded up.
pub fn first_block_chunk_size(attempts: usize, worker_count: usize) -> usize {
    if attempts == 0 || worker_count == 0 {
        return 0;
    }
    attempts.div_ceil(worker_count)
}

/// Port of `firstBlockSelectedChunkSize`. A dynamic chunk size only applies once there is
/// more than one worker to steal between.
pub fn first_block_selected_chunk_size(
    attempts: usize,
    worker_count: usize,
    dynamic_chunk_size: usize,
) -> usize {
    if attempts == 0 || worker_count == 0 {
        return 0;
    }
    if dynamic_chunk_size > 0 && worker_count > 1 {
        return attempts.min(dynamic_chunk_size);
    }
    first_block_chunk_size(attempts, worker_count)
}

/// Port of `recommendedFirstBlockDynamicChunkSize`: the measured sweet spots where small
/// stolen chunks beat an even split, for the GPU path only.
pub fn recommended_first_block_dynamic_chunk_size(
    dynamic_chunk_auto: bool,
    backend: &str,
    has_fixed_key: bool,
    difficulty: u32,
    attempts: usize,
    worker_count: usize,
) -> usize {
    if !dynamic_chunk_auto
        || backend != "cuda"
        || has_fixed_key
        || attempts < 1024
        || worker_count <= 1
    {
        return 0;
    }
    match difficulty {
        1 => 16,
        8 => {
            if attempts >= 2048 {
                16
            } else {
                32
            }
        }
        64 if attempts <= 2048 => 16,
        _ => 0,
    }
}

/// The production [`Argon2Host`].
///
/// `workers` caps the thread count (0 = up to `available_parallelism`), `dynamic_chunk_size`
/// switches the batch path from an even split to stolen chunks of that many jobs.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuArgon2Host {
    workers: usize,
    dynamic_chunk_size: usize,
}

impl CpuArgon2Host {
    /// Auto-sized worker pool, evenly split chunks — what the mining loop wants.
    pub const fn new() -> Self {
        Self {
            workers: 0,
            dynamic_chunk_size: 0,
        }
    }

    /// One thread, no chunking. Used by the tests as the oracle the parallel path must
    /// reproduce byte for byte.
    pub const fn single_threaded() -> Self {
        Self {
            workers: 1,
            dynamic_chunk_size: 0,
        }
    }

    pub const fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers;
        self
    }

    pub const fn with_dynamic_chunk_size(mut self, jobs: usize) -> Self {
        self.dynamic_chunk_size = jobs;
        self
    }

    /// The operator-supplied cap, verbatim; `worker_count` applies it.
    pub const fn workers(&self) -> usize {
        self.workers
    }

    pub const fn dynamic_chunk_size(&self) -> usize {
        self.dynamic_chunk_size
    }

    /// How many threads a batch of `attempts` jobs would actually use.
    pub fn worker_count(&self, attempts: usize) -> usize {
        first_block_worker_count(attempts, self.workers)
    }

    /// How many jobs each unit of work covers.
    pub fn chunk_size(&self, attempts: usize) -> usize {
        first_block_selected_chunk_size(
            attempts,
            self.worker_count(attempts),
            self.dynamic_chunk_size,
        )
    }
}

/// Bytes one job's first blocks occupy.
fn slot_bytes(shape: &Argon2Shape) -> usize {
    INPUT_BLOCKS_PER_JOB * ARGON2_BLOCK_SIZE * shape.lanes as usize
}

impl Argon2Host for CpuArgon2Host {
    fn fill_first_blocks(
        &self,
        out: &mut [u8],
        password: &[u8],
        salt: &[u8],
        shape: &Argon2Shape,
    ) -> Result<(), HostError> {
        let want = slot_bytes(shape);
        if out.len() != want {
            return Err(format!("first-block slot is {} bytes, want {want}", out.len()).into());
        }
        if shape.lanes == 0 {
            return Err("shape must have at least one lane".into());
        }

        let mut seed = [0u8; PREHASH_SEED_LENGTH];
        initial_hash(&mut seed[..PREHASH_DIGEST_LENGTH], password, salt, shape);

        let mut written = 0;
        // Block 0 of every lane, then block 1 of every lane — the layout the kernel reads.
        for block_index in 0u32..2 {
            seed[PREHASH_DIGEST_LENGTH..PREHASH_DIGEST_LENGTH + 4]
                .copy_from_slice(&block_index.to_le_bytes());
            for lane in 0..shape.lanes {
                seed[PREHASH_DIGEST_LENGTH + 4..].copy_from_slice(&lane.to_le_bytes());
                digest_long(&mut out[written..written + ARGON2_BLOCK_SIZE], &seed);
                written += ARGON2_BLOCK_SIZE;
            }
        }
        Ok(())
    }

    fn fill_first_blocks_batch(
        &self,
        out: &mut [u8],
        passwords: &[String],
        salt: &[u8],
        shape: &Argon2Shape,
    ) -> Result<(), HostError> {
        let slot = slot_bytes(shape);
        let attempts = passwords.len();
        if out.len() != attempts * slot {
            return Err(format!(
                "first-block buffer is {} bytes, want {}",
                out.len(),
                attempts * slot
            )
            .into());
        }
        if attempts == 0 {
            return Ok(());
        }

        let worker_count = self.worker_count(attempts);
        if worker_count <= 1 {
            for (chunk, password) in out.chunks_mut(slot).zip(passwords) {
                self.fill_first_blocks(chunk, password.as_bytes(), salt, shape)?;
            }
            return Ok(());
        }

        let chunk_jobs = first_block_selected_chunk_size(
            attempts,
            worker_count,
            self.dynamic_chunk_size,
        )
        .max(1);

        // With an even split there is exactly one chunk per worker, so a shared queue is
        // the C++ static path; with a smaller dynamic chunk size it is the C++ work-stealing
        // path. One loop covers both, and neither needs unsafe aliasing of `out`.
        let queue: Mutex<VecDeque<(usize, &mut [u8])>> = Mutex::new(
            out.chunks_mut(chunk_jobs * slot)
                .enumerate()
                .map(|(index, chunk)| (index * chunk_jobs, chunk))
                .collect(),
        );

        let mut failure: Option<String> = None;
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..worker_count)
                .map(|_| {
                    let queue = &queue;
                    scope.spawn(move || -> Result<(), String> {
                        loop {
                            let next = queue
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .pop_front();
                            let Some((first_job, chunk)) = next else {
                                return Ok(());
                            };
                            for (offset, slot_out) in chunk.chunks_mut(slot).enumerate() {
                                let password = &passwords[first_job + offset];
                                self.fill_first_blocks(
                                    slot_out,
                                    password.as_bytes(),
                                    salt,
                                    shape,
                                )
                                .map_err(|error| error.to_string())?;
                            }
                        }
                    })
                })
                .collect();
            for handle in handles {
                match handle.join() {
                    Ok(Err(error)) => failure.get_or_insert(error),
                    // A panicking worker is a bug, not operator input; surface it as an
                    // error rather than resuming the unwind into the caller's thread.
                    Err(_) => failure.get_or_insert_with(|| {
                        "a first-block worker panicked".to_owned()
                    }),
                    Ok(Ok(())) => continue,
                };
            }
        });

        match failure {
            Some(error) => Err(error.into()),
            None => Ok(()),
        }
    }

    fn finalize(&self, last_block: &[u8], out: &mut [u8]) -> Result<(), HostError> {
        if last_block.is_empty() || last_block.len() % ARGON2_BLOCK_SIZE != 0 {
            return Err(format!(
                "final block is {} bytes, want a multiple of 1024",
                last_block.len()
            )
            .into());
        }
        if last_block.len() == ARGON2_BLOCK_SIZE {
            digest_long(out, last_block);
            return Ok(());
        }
        // lanes > 1: XOR the last block of every lane before reducing.
        let mut xored = [0u8; ARGON2_BLOCK_SIZE];
        for block in last_block.chunks(ARGON2_BLOCK_SIZE) {
            for (accumulated, byte) in xored.iter_mut().zip(block) {
                *accumulated ^= byte;
            }
        }
        digest_long(out, &xored);
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

/// Argon2's variable-length hash H'. Port of `Argon2Params::digestLong`.
pub fn digest_long(out: &mut [u8], input: &[u8]) {
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
    let mut tail = [0u8; BLAKE2B_OUT_BYTES];
    blake2b(&mut tail[..to_produce], &[&buffer]);
    out[written..].copy_from_slice(&tail[..to_produce]);
}

/// Argon2's H0. Port of `Argon2Params::initialHash` for the `secretLen == adLen == 0` case
/// the miner always takes. `salt` is the decoded address bytes, not its hex text — the C++
/// side stores the hex and decodes it here.
pub fn initial_hash(out: &mut [u8], password: &[u8], salt: &[u8], shape: &Argon2Shape) {
    let mut header = [0u8; 7 * 4];
    for (slot, value) in header.chunks_mut(4).zip([
        shape.lanes,
        shape.output_length,
        shape.memory_cost,
        shape.passes,
        shape.version,
        shape.type_,
        password.len() as u32,
    ]) {
        slot.copy_from_slice(&value.to_le_bytes());
    }
    let salt_length = (salt.len() as u32).to_le_bytes();
    // The lengths of the (always absent) secret and associated data.
    let empty_lengths = [0u8; 8];
    blake2b(
        out,
        &[&header, password, &salt_length, salt, &empty_lengths],
    );
}
