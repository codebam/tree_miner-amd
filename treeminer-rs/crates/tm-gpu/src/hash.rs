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
        let mut outcome = BatchOutcome {
            attempts,
            gpu_first_blocks,
            digests: if request.collect_digests {
                Vec::with_capacity(attempts)
            } else {
                Vec::new()
            },
            ..Default::default()
        };
        let mut digest = [0u8; DEFAULT_HASH_LENGTH];
        for index in 0..attempts {
            let block = self.backend.runner_mut()?.output_block(index)?;
            debug_assert_eq!(block.len(), ARGON2_BLOCK_SIZE);
            host.finalize(block, &mut digest)
                .map_err(|error| GpuError::Host(error.to_string()))?;
            let hash = base64_encode(&digest);
            append_matches(request, &mut outcome, &request.passwords[index], &hash, index);
            if attempts == 1 {
                outcome.hash = Some(hash.clone());
            }
            if request.collect_digests {
                outcome.digests.push(hash);
            }
        }
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

/// Port of `hashapi::appendMatches` — a digest can match both the target pattern and XUNI,
/// and is then reported twice, once per pattern.
fn append_matches(
    request: &BatchRequest<'_>,
    outcome: &mut BatchOutcome,
    key: &str,
    hash: &str,
    attempt_index: usize,
) {
    if !request.target_pattern.is_empty() && hash.contains(request.target_pattern) {
        outcome.matches.push(GpuMatch {
            key: key.to_owned(),
            hash: hash.to_owned(),
            matched_pattern: request.target_pattern.to_owned(),
            attempt_index,
            is_superblock: is_superblock_hash(hash),
        });
    }
    if request.allow_xuni && has_xuni_match(hash) {
        outcome.matches.push(GpuMatch {
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
