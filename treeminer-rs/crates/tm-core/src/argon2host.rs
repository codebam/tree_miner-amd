//! The host (CPU) half of Argon2, as a contract.
//!
//! Argon2 splits cleanly in three: derive a job's two first blocks from the password and
//! salt, chain the memory blocks, reduce the last block to a digest. Only the middle part
//! belongs to the GPU. The two ends are Blake2b and are shared by every backend, so the
//! shape they agree on and the trait that produces them live here rather than in `tm-gpu`
//! — a CPU-only build must be able to name them without linking HIP.
//!
//! `Argon2Shape` is a port of the fields `Argon2Params` carries; the implementation of
//! [`Argon2Host`] is `tm_argon2::CpuArgon2Host`.

/// Argon2 primitive type. The miner only ever uses `Argon2id`.
pub const ARGON2_ID: u32 = 2;
/// Argon2 v1.3 (0x13 == 19), the version XenBlocks pins.
pub const ARGON2_VERSION_13: u32 = 0x13;

pub const ARGON2_BLOCK_SIZE: usize = 1024;
pub const ARGON2_SYNC_POINTS: u32 = 4;
/// XenBlocks digests are 64 bytes.
pub const DEFAULT_HASH_LENGTH: usize = 64;

/// Two Argon2 blocks per job go into the pool, one comes back out.
pub const INPUT_BLOCKS_PER_JOB: usize = 2;

/// Everything both the kernel launch and the CPU first-blocks path need to agree on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Argon2Shape {
    pub type_: u32,
    pub version: u32,
    /// `t_cost`; the kernel is a one-shot single-pass implementation, so this is always 1.
    pub passes: u32,
    pub lanes: u32,
    /// `m_cost` in KiB — the mining difficulty.
    pub memory_cost: u32,
    pub output_length: u32,
}

impl Argon2Shape {
    /// The shape XenBlocks mines with at a given difficulty.
    pub fn for_difficulty(difficulty: u32) -> Self {
        Self {
            type_: ARGON2_ID,
            version: ARGON2_VERSION_13,
            passes: 1,
            lanes: 1,
            memory_cost: difficulty,
            output_length: DEFAULT_HASH_LENGTH as u32,
        }
    }

    /// Blocks per segment, rounded exactly as `Argon2Params`' constructor does — including
    /// the `2 * segments` floor that keeps tiny difficulties legal.
    pub fn segment_blocks(&self) -> u32 {
        let segments = self.lanes * ARGON2_SYNC_POINTS;
        self.memory_cost.max(2 * segments) / segments
    }

    pub fn lane_blocks(&self) -> u32 {
        self.segment_blocks() * ARGON2_SYNC_POINTS
    }

    /// Device bytes one job of this shape occupies.
    pub fn job_bytes(&self) -> usize {
        self.lane_blocks() as usize * self.lanes as usize * ARGON2_BLOCK_SIZE
    }
}

/// Anything an implementer can fail with; boxed so `tm-argon2` keeps its own error type.
pub type HostError = Box<dyn std::error::Error + Send + Sync>;

/// The CPU half of Argon2 that the GPU path needs.
///
/// Implemented by `tm_argon2::CpuArgon2Host`. Neither operation touches the device: the
/// GPU computes the block chain in between.
pub trait Argon2Host: Send + Sync {
    /// Writes the two 1024-byte first blocks for `password` into `out` (exactly
    /// `2 * lanes` blocks), for the given salt and shape. Equivalent to
    /// `Argon2Params::fillFirstBlocks`.
    fn fill_first_blocks(
        &self,
        out: &mut [u8],
        password: &[u8],
        salt: &[u8],
        shape: &Argon2Shape,
    ) -> Result<(), HostError>;

    /// Fills the first blocks of a whole batch: `out` is `passwords.len()` consecutive
    /// per-job slots, in job order.
    ///
    /// The default is the obvious loop. `CpuArgon2Host` overrides it to spread the work
    /// across threads, which is what the C++ miner does — on ROCm the first blocks are on
    /// the hot path of every batch, because the device-side kernel is disabled there.
    fn fill_first_blocks_batch(
        &self,
        out: &mut [u8],
        passwords: &[String],
        salt: &[u8],
        shape: &Argon2Shape,
    ) -> Result<(), HostError> {
        let slot = INPUT_BLOCKS_PER_JOB * ARGON2_BLOCK_SIZE * shape.lanes as usize;
        if out.len() != passwords.len() * slot {
            return Err(format!(
                "first-block buffer is {} bytes, want {}",
                out.len(),
                passwords.len() * slot
            )
            .into());
        }
        for (chunk, password) in out.chunks_mut(slot).zip(passwords) {
            self.fill_first_blocks(chunk, password.as_bytes(), salt, shape)?;
        }
        Ok(())
    }

    /// Reduces the final 1024-byte Argon2 block to a digest of `out.len()` bytes.
    /// Equivalent to `Argon2Params::finalize` for `lanes == 1`.
    fn finalize(&self, last_block: &[u8], out: &mut [u8]) -> Result<(), HostError>;
}

#[cfg(test)]
mod tests {
    use super::Argon2Shape;

    #[test]
    fn segment_blocks_matches_the_cpp_rounding() {
        // m=8, lanes=1 -> segments=4 -> max(8, 8)/4 = 2
        assert_eq!(Argon2Shape::for_difficulty(8).segment_blocks(), 2);
        // The 2*segments floor: m=1 still gets two blocks per segment.
        assert_eq!(Argon2Shape::for_difficulty(1).segment_blocks(), 2);
        assert_eq!(Argon2Shape::for_difficulty(9).segment_blocks(), 2);
        assert_eq!(Argon2Shape::for_difficulty(60000).segment_blocks(), 15000);
    }

    #[test]
    fn job_bytes_tracks_the_rounded_segment_size() {
        assert_eq!(Argon2Shape::for_difficulty(8).job_bytes(), 8 * 1024);
        // Difficulty 9 rounds down to 8 KiB of blocks, as in the C++ miner.
        assert_eq!(Argon2Shape::for_difficulty(9).job_bytes(), 8 * 1024);
    }
}
