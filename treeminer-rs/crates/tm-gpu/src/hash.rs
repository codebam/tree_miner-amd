//! The GPU hash backend: first blocks in, digests and matches out. Port of
//! `src/hashapi/CudaHashBackend.cpp`.
//!
//! The Argon2 *host* helpers (deriving a job's two first blocks, and reducing its final
//! block to a 64-byte digest) are injected through [`Argon2Host`] rather than implemented
//! here — they are Blake2b, they are shared with the CPU miner, and tm-gpu must not carry a
//! second copy of them.

use std::time::Instant;

use tm_core::encoding::{base64_encode, hex_to_bytes};
use tm_core::matching::{has_xuni_match, is_superblock_hash};

use crate::backend::GpuBackend;
use crate::error::{GpuError, Result};
use crate::params::{Argon2Shape, ARGON2_BLOCK_SIZE, DEFAULT_HASH_LENGTH};

pub use tm_core::argon2host::{Argon2Host, HostError};

/// One batch of Argon2 work. Key generation belongs to the caller: the mining loop, the
/// self-test and the CLI all want different key sources, and none of them belong on the GPU.
#[derive(Debug, Clone)]
pub struct BatchRequest<'a> {
    /// One password per job; the batch size is `passwords.len()`.
    pub passwords: &'a [String],
    /// The 40 hex characters of the miner address (no `0x`).
    pub salt_hex: &'a str,
    /// Argon2 `m` in KiB.
    pub difficulty: u32,
    /// Substring that marks a payable find, normally `XEN11`.
    pub target_pattern: &'a str,
    pub allow_xuni: bool,
    /// Derive the first blocks on the device instead of on the CPU.
    pub gpu_first_blocks: bool,
    /// Keep every digest, not just the matches. Off in mining; on in tests.
    pub collect_digests: bool,
}

impl<'a> BatchRequest<'a> {
    pub fn new(passwords: &'a [String], salt_hex: &'a str, difficulty: u32) -> Self {
        Self {
            passwords,
            salt_hex,
            difficulty,
            target_pattern: "XEN11",
            allow_xuni: true,
            gpu_first_blocks: false,
            collect_digests: false,
        }
    }
}

/// One hit inside a batch. Mirrors `HashApiMatch`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuMatch {
    pub key: String,
    /// Unpadded base64 digest.
    pub hash: String,
    pub matched_pattern: String,
    pub attempt_index: usize,
    pub is_superblock: bool,
}

/// Wall-clock breakdown of a batch, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BatchTimings {
    pub setup_ms: f64,
    pub first_block_ms: f64,
    pub compute_ms: f64,
    pub kernel_ms: f64,
    pub host_to_device_ms: f64,
    pub gpu_first_block_ms: f64,
    pub device_to_host_ms: f64,
    pub finalize_ms: f64,
    pub total_ms: f64,
}

/// The result of one batch.
#[derive(Debug, Clone, Default)]
pub struct BatchOutcome {
    pub attempts: usize,
    pub gpu_first_blocks: bool,
    pub matches: Vec<GpuMatch>,
    /// Every digest, in job order; empty unless `collect_digests` was set.
    pub digests: Vec<String>,
    /// The single digest, when the batch had exactly one job.
    pub hash: Option<String>,
    pub elapsed_ms: f64,
    pub hashrate: f64,
    pub timings: BatchTimings,
}

/// Runs Argon2 batches on one GPU.
#[derive(Debug)]
pub struct GpuHashBackend {
    backend: GpuBackend,
}

impl GpuHashBackend {
    pub fn new(backend: GpuBackend) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &GpuBackend {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut GpuBackend {
        &mut self.backend
    }

    pub fn into_backend(self) -> GpuBackend {
        self.backend
    }

    /// Hashes one batch and collects the matches.
    pub fn run_batch(
        &mut self,
        request: &BatchRequest<'_>,
        host: &dyn Argon2Host,
    ) -> Result<BatchOutcome> {
        let total_start = Instant::now();
        if request.passwords.is_empty() {
            return Err(GpuError::Invalid("batch has no passwords".to_owned()));
        }
        if request.difficulty == 0 {
            return Err(GpuError::Invalid("difficulty must be non-zero".to_owned()));
        }
        let salt = hex_to_bytes(request.salt_hex)
            .map_err(|error| GpuError::Invalid(format!("salt: {error}")))?;
        if salt.is_empty() {
            return Err(GpuError::Invalid("salt is empty".to_owned()));
        }

        let attempts = request.passwords.len();
        let shape = Argon2Shape::for_difficulty(request.difficulty);

        let setup_start = Instant::now();
        self.backend.activate()?;
        self.backend.init(&shape, attempts)?;
        let setup_ms = elapsed_ms(setup_start);

        let first_block_start = Instant::now();
        let gpu_first_blocks = if request.gpu_first_blocks {
            let runner = self.backend.runner_mut()?;
            if !runner.prepare_input_blocks_on_device(request.passwords, &salt, &shape)? {
                return Err(GpuError::Invalid(
                    "the device first-blocks kernel does not support this batch shape".to_owned(),
                ));
            }
            true
        } else {
            let runner = self.backend.runner_mut()?;
            let slots = runner.input_blocks_batch_mut(attempts)?;
            host.fill_first_blocks_batch(slots, request.passwords, &salt, &shape)
                .map_err(|error| GpuError::Host(error.to_string()))?;
            false
        };
        let first_block_ms = elapsed_ms(first_block_start);

        let compute_start = Instant::now();
        let runner = self.backend.runner_mut()?;
        runner.run()?;
        let kernel_ms = f64::from(runner.finish()?);
        let run_timings = runner.timings();
        let compute_ms = elapsed_ms(compute_start);

        let finalize_start = Instant::now();
        let blocks = self
            .backend
            .runner()
            .ok_or_else(|| GpuError::Invalid("GPU pool is not initialised".to_owned()))?
            .output_blocks(attempts)?;
        let finalized = finalize_batch(request, host, blocks, finalize_worker_count(attempts))?;
        let mut outcome = BatchOutcome {
            attempts,
            gpu_first_blocks,
            matches: finalized.matches,
            digests: finalized.digests,
            hash: finalized.hash,
            ..Default::default()
        };
        let finalize_ms = elapsed_ms(finalize_start);

        let total_ms = elapsed_ms(total_start);
        outcome.elapsed_ms = total_ms;
        outcome.hashrate = if total_ms > 0.0 {
            attempts as f64 / (total_ms / 1000.0)
        } else {
            0.0
        };
        outcome.timings = BatchTimings {
            setup_ms,
            first_block_ms,
            compute_ms,
            kernel_ms,
            host_to_device_ms: f64::from(run_timings.host_to_device_ms),
            gpu_first_block_ms: f64::from(run_timings.gpu_first_block_ms),
            device_to_host_ms: f64::from(run_timings.device_to_host_ms),
            finalize_ms,
            total_ms,
        };
        Ok(outcome)
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Below this many jobs the finalize pass stays on the calling thread: a batch this small
/// finishes in less time than it takes to spawn the workers. Sized like
/// `tm_argon2::MIN_PARALLEL_FIRST_BLOCK_ATTEMPTS`, but higher, because finalizing one job is
/// roughly half the Blake2b work of deriving its first blocks.
pub const MIN_PARALLEL_FINALIZE_ATTEMPTS: usize = 64;

/// How many threads a finalize pass over `attempts` jobs uses. Mirrors
/// `tm_argon2::first_block_worker_count`; there is no operator cap on this one.
fn finalize_worker_count(attempts: usize) -> usize {
    if attempts < MIN_PARALLEL_FINALIZE_ATTEMPTS {
        return 1;
    }
    let hardware_threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    if hardware_threads < 2 {
        return 1;
    }
    attempts.min(hardware_threads).max(1)
}

/// What one finalize pass produces. Kept separate from [`BatchOutcome`] so a worker can
/// build one for its own slice of the batch and the parts can be concatenated in job order.
#[derive(Debug, Default)]
struct Finalized {
    matches: Vec<GpuMatch>,
    digests: Vec<String>,
    /// Set only for a single-job batch, mirroring `BatchOutcome::hash`.
    hash: Option<String>,
}

/// Reduces the final Argon2 blocks of one batch to digests and collects the matches.
///
/// `blocks` is the whole batch's output, `attempts * ARGON2_BLOCK_SIZE` bytes; the job index
/// of a chunk is its offset divided by the block size, which is what keeps the matches in
/// ascending `attempt_index` order however the work is split.
///
/// `worker_count` of 1 runs on the calling thread. Any larger value splits the batch into
/// that many contiguous ranges, one scoped thread each, and concatenates the per-range
/// results in range order — so the output is identical to the serial path, byte for byte
/// and element for element.
fn finalize_batch(
    request: &BatchRequest<'_>,
    host: &dyn Argon2Host,
    blocks: &[u8],
    worker_count: usize,
) -> Result<Finalized> {
    let attempts = request.passwords.len();
    if blocks.len() != attempts * ARGON2_BLOCK_SIZE {
        return Err(GpuError::Invalid(format!(
            "output buffer is {} bytes, want {}",
            blocks.len(),
            attempts * ARGON2_BLOCK_SIZE
        )));
    }
    // A one-job batch reports its digest whether or not the caller asked for all of them.
    let keep_digests = request.collect_digests || attempts == 1;

    if worker_count <= 1 || attempts <= 1 {
        let mut finalized = Finalized {
            digests: if keep_digests {
                Vec::with_capacity(attempts)
            } else {
                Vec::new()
            },
            ..Default::default()
        };
        finalize_range(request, host, blocks, 0, keep_digests, &mut finalized)?;
        return Ok(collect(finalized, request, attempts));
    }

    let chunk_jobs = attempts.div_ceil(worker_count).max(1);
    let mut parts: Vec<Result<Finalized>> = Vec::new();
    std::thread::scope(|scope| {
        let handles: Vec<_> = blocks
            .chunks(chunk_jobs * ARGON2_BLOCK_SIZE)
            .enumerate()
            .map(|(chunk_index, chunk)| {
                let first_job = chunk_index * chunk_jobs;
                scope.spawn(move || {
                    let jobs = chunk.len() / ARGON2_BLOCK_SIZE;
                    let mut part = Finalized {
                        digests: if keep_digests {
                            Vec::with_capacity(jobs)
                        } else {
                            Vec::new()
                        },
                        ..Default::default()
                    };
                    finalize_range(request, host, chunk, first_job, keep_digests, &mut part)?;
                    Ok(part)
                })
            })
            .collect();
        for handle in handles {
            parts.push(handle.join().unwrap_or_else(|_| {
                // A panicking worker is a bug, not operator input; surface it as an error
                // rather than resuming the unwind into the caller's thread.
                Err(GpuError::Host("a finalize worker panicked".to_owned()))
            }));
        }
    });

    let mut finalized = Finalized {
        digests: if keep_digests {
            Vec::with_capacity(attempts)
        } else {
            Vec::new()
        },
        ..Default::default()
    };
    for part in parts {
        let part = part?;
        finalized.matches.extend(part.matches);
        finalized.digests.extend(part.digests);
    }
    Ok(collect(finalized, request, attempts))
}

/// Drops the digests the caller never asked for, and lifts the single-job digest out.
fn collect(mut finalized: Finalized, request: &BatchRequest<'_>, attempts: usize) -> Finalized {
    if attempts == 1 {
        finalized.hash = finalized.digests.first().cloned();
    }
    if !request.collect_digests {
        finalized.digests = Vec::new();
    }
    finalized
}

/// Finalizes one contiguous run of jobs. `blocks` holds the jobs `first_job ..`, and nothing
/// in here depends on any job outside that range — which is what makes the split safe.
fn finalize_range(
    request: &BatchRequest<'_>,
    host: &dyn Argon2Host,
    blocks: &[u8],
    first_job: usize,
    keep_digests: bool,
    out: &mut Finalized,
) -> Result<()> {
    let mut digest = [0u8; DEFAULT_HASH_LENGTH];
    for (offset, block) in blocks.chunks(ARGON2_BLOCK_SIZE).enumerate() {
        debug_assert_eq!(block.len(), ARGON2_BLOCK_SIZE);
        let index = first_job + offset;
        host.finalize(block, &mut digest)
            .map_err(|error| GpuError::Host(error.to_string()))?;
        let hash = base64_encode(&digest);
        append_matches(
            request,
            &mut out.matches,
            &request.passwords[index],
            &hash,
            index,
        );
        if keep_digests {
            out.digests.push(hash);
        }
    }
    Ok(())
}

/// Port of `hashapi::appendMatches` — a digest can match both the target pattern and XUNI,
/// and is then reported twice, once per pattern.
fn append_matches(
    request: &BatchRequest<'_>,
    matches: &mut Vec<GpuMatch>,
    key: &str,
    hash: &str,
    attempt_index: usize,
) {
    if !request.target_pattern.is_empty() && hash.contains(request.target_pattern) {
        matches.push(GpuMatch {
            key: key.to_owned(),
            hash: hash.to_owned(),
            matched_pattern: request.target_pattern.to_owned(),
            attempt_index,
            is_superblock: is_superblock_hash(hash),
        });
    }
    if request.allow_xuni && has_xuni_match(hash) {
        matches.push(GpuMatch {
            key: key.to_owned(),
            hash: hash.to_owned(),
            matched_pattern: "XUNI".to_owned(),
            attempt_index,
            // NOT an oversight, and not symmetric with the arm above: the server never
            // credits a superblock on the XUNI path, however many capitals the digest has.
            //   * `/verify` routes a XUNI[0-9] hash into the `xuni` table and a XEN11 hash
            //     into `blocks` (`gpage.py:468-490`). `make_superblocks.py:33-47` counts
            //     capitals over `blocks` ONLY, so a XUNI row is never seen by it.
            //   * The X.BLK balance credit is `check_and_credit_for_capital_count`, and it
            //     is called from inside the `elif 'XEN' in hash_to_verify` branch of
            //     `utils/gen_balances.py:119-124` — the XUNI branch above it credits
            //     currency 2 (XUNI) and returns without ever reaching the capital count.
            // Flagging it true here would only make the miner claim a reward the network
            // does not pay.
            is_superblock: false,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tm_core::argon2host::Argon2Shape as HostShape;

    /// The six bytes whose base64 is `XEN11AAA`, at a 3-byte boundary.
    const XEN11_BYTES: [u8; 6] = [92, 67, 117, 212, 0, 0];
    /// The six bytes whose base64 is `XUNI0AAA`.
    const XUNI_BYTES: [u8; 6] = [93, 67, 72, 208, 0, 0];

    /// A stand-in for `CpuArgon2Host` that is cheap, deterministic, and — unlike Blake2b —
    /// lets the test decide which jobs produce a payable digest. The job index and the
    /// marker are carried in the block itself, because `finalize` is not told either.
    #[derive(Debug)]
    struct MarkerHost;

    impl Argon2Host for MarkerHost {
        fn fill_first_blocks(
            &self,
            _out: &mut [u8],
            _password: &[u8],
            _salt: &[u8],
            _shape: &HostShape,
        ) -> std::result::Result<(), HostError> {
            Err("the marker host only finalizes".into())
        }

        fn finalize(
            &self,
            last_block: &[u8],
            out: &mut [u8],
        ) -> std::result::Result<(), HostError> {
            if last_block.len() != ARGON2_BLOCK_SIZE {
                return Err("bad block".into());
            }
            let index = u32::from_le_bytes(last_block[1..5].try_into().unwrap());
            // Something digest-shaped that differs per job, so no two hashes collide.
            let mut state = index.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9);
            for byte in out.iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }
            match last_block[0] {
                1 => out[3..9].copy_from_slice(&XEN11_BYTES),
                2 => out[3..9].copy_from_slice(&XUNI_BYTES),
                // Both patterns in one digest: `append_matches` must report it twice.
                3 => {
                    out[3..9].copy_from_slice(&XEN11_BYTES);
                    out[9..15].copy_from_slice(&XUNI_BYTES);
                }
                _ => {}
            }
            Ok(())
        }
    }

    /// One output block per job, carrying its index and a marker that decides whether the
    /// digest will match. The markers are spaced so that hits land at the start, in the
    /// middle and at the end of the batch, and therefore on both sides of any chunk split.
    fn output_blocks(attempts: usize) -> Vec<u8> {
        let mut blocks = vec![0u8; attempts * ARGON2_BLOCK_SIZE];
        for (index, block) in blocks.chunks_mut(ARGON2_BLOCK_SIZE).enumerate() {
            block[0] = match index % 17 {
                0 => 1,
                5 => 2,
                11 => 3,
                _ => 0,
            };
            block[1..5].copy_from_slice(&(index as u32).to_le_bytes());
        }
        blocks
    }

    fn passwords(attempts: usize) -> Vec<String> {
        (0..attempts).map(|index| format!("key-{index}")).collect()
    }

    fn request<'a>(passwords: &'a [String], collect_digests: bool) -> BatchRequest<'a> {
        let mut request = BatchRequest::new(passwords, "00", 100);
        request.collect_digests = collect_digests;
        request
    }

    #[test]
    fn the_marker_host_really_produces_the_patterns() {
        let blocks = output_blocks(12);
        let keys = passwords(12);
        let finalized =
            finalize_batch(&request(&keys, true), &MarkerHost, &blocks, 1).expect("finalize");
        assert!(finalized.digests[0].contains("XEN11"), "{:?}", finalized.digests[0]);
        assert!(has_xuni_match(&finalized.digests[5]), "{:?}", finalized.digests[5]);
        assert!(finalized.digests[11].contains("XEN11"));
        assert!(has_xuni_match(&finalized.digests[11]));
        // Job 11 matches both patterns, so it is reported twice — under the same index.
        let both: Vec<_> = finalized
            .matches
            .iter()
            .filter(|hit| hit.attempt_index == 11)
            .collect();
        assert_eq!(both.len(), 2);
        assert_eq!(both[0].matched_pattern, "XEN11");
        assert_eq!(both[1].matched_pattern, "XUNI");
    }

    #[test]
    fn parallel_finalize_is_identical_to_serial() {
        for attempts in [1usize, 2, 7, 16, 63, 64, 65, 100, 999, 1024, 4097] {
            let keys = passwords(attempts);
            let blocks = output_blocks(attempts);
            for collect_digests in [false, true] {
                let request = request(&keys, collect_digests);
                let serial =
                    finalize_batch(&request, &MarkerHost, &blocks, 1).expect("serial finalize");
                for workers in [2usize, 3, 8, 16, 64] {
                    let parallel = finalize_batch(&request, &MarkerHost, &blocks, workers)
                        .expect("parallel finalize");
                    assert_eq!(
                        parallel.matches, serial.matches,
                        "attempts={attempts} workers={workers}"
                    );
                    assert_eq!(
                        parallel.digests, serial.digests,
                        "attempts={attempts} workers={workers}"
                    );
                    assert_eq!(
                        parallel.hash, serial.hash,
                        "attempts={attempts} workers={workers}"
                    );
                }
                assert_eq!(serial.digests.len(), if collect_digests { attempts } else { 0 });
                assert_eq!(serial.hash.is_some(), attempts == 1);
                assert!(!serial.matches.is_empty());
                assert!(
                    serial
                        .matches
                        .windows(2)
                        .all(|pair| pair[0].attempt_index <= pair[1].attempt_index),
                    "matches are out of attempt order at attempts={attempts}"
                );
            }
        }
    }

    #[test]
    fn a_short_batch_stays_on_the_calling_thread() {
        assert_eq!(finalize_worker_count(0), 1);
        assert_eq!(finalize_worker_count(1), 1);
        assert_eq!(finalize_worker_count(MIN_PARALLEL_FINALIZE_ATTEMPTS - 1), 1);
        let threads = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1);
        if threads > 1 {
            assert!(finalize_worker_count(MIN_PARALLEL_FINALIZE_ATTEMPTS) > 1);
            assert_eq!(finalize_worker_count(1_000_000), threads);
        }
    }

    #[test]
    fn a_mis_sized_output_buffer_is_rejected() {
        let keys = passwords(4);
        let blocks = output_blocks(3);
        let error = finalize_batch(&request(&keys, false), &MarkerHost, &blocks, 4)
            .expect_err("length mismatch must not be silently truncated");
        assert!(error.to_string().contains("output buffer"), "{error}");
    }

    #[test]
    fn a_failing_host_surfaces_from_a_worker() {
        #[derive(Debug)]
        struct AlwaysFails;
        impl Argon2Host for AlwaysFails {
            fn fill_first_blocks(
                &self,
                _out: &mut [u8],
                _password: &[u8],
                _salt: &[u8],
                _shape: &HostShape,
            ) -> std::result::Result<(), HostError> {
                Err("unused".into())
            }
            fn finalize(
                &self,
                _last_block: &[u8],
                _out: &mut [u8],
            ) -> std::result::Result<(), HostError> {
                Err("host exploded".into())
            }
        }
        let keys = passwords(256);
        let blocks = output_blocks(256);
        let error = finalize_batch(&request(&keys, false), &AlwaysFails, &blocks, 8)
            .expect_err("a host failure must not be swallowed");
        assert!(error.to_string().contains("host exploded"), "{error}");
    }
}
