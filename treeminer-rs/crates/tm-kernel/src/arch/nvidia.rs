//! NVPTX (`nvptx64-nvidia-cuda`) implementation of [`super`].
//!
//! **UNTESTED.** No NVIDIA GPU has ever executed this code. It compiles, the emitted PTX
//! has been inspected (`shfl.sync.idx.b32` / `shfl.sync.bfly.b32`, member mask `-1`, clamp
//! `31`), and the host-side argument packing is unit-tested — but not one digest has been
//! produced by it. Treat every claim about it as a claim about the *source*, not about a
//! result. The first person with an NVIDIA card must run `tests/parity/run_parity.sh` and
//! the fixture suite before trusting a submission from this path.
//!
//! The semantics being reproduced are HIP/CUDA's `__shfl_sync(0xffffffff, v, src, 32)` and
//! `__shfl_xor_sync(0xffffffff, v, mask, 32)` as used by
//! `../tm-gpu/kernel/argon2_kernel.hip`.

use super::THREADS_PER_LANE;

/// Full warp. Both kernels are launched with a block that is a whole number of warps and
/// every lane reaches every shuffle — the one-shot kernel's block *is* one warp — so no
/// lane can be inactive at a `shfl.sync` and the mask is unconditional.
const FULL_WARP: u32 = 0xffff_ffff;

/// The packed `c` operand of `shfl.sync`: `((32 - width) << 8) | 0x1f`. Width 32, so the
/// segment mask is empty and `maxLane` is 31 — the clamp CUDA's `width = 32` argument
/// means. It shows up in the PTX as the literal `31`.
const CLAMP_WIDTH_32: u32 = ((32 - THREADS_PER_LANE) << 8) | 0x1f;

extern "C" {
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
    fn tid_x() -> u32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
    fn ctaid_x() -> u32;
    #[link_name = "llvm.nvvm.shfl.sync.idx.i32"]
    fn shfl_sync_idx(member_mask: u32, value: u32, src_lane: u32, clamp: u32) -> u32;
    #[link_name = "llvm.nvvm.shfl.sync.bfly.i32"]
    fn shfl_sync_bfly(member_mask: u32, value: u32, lane_mask: u32, clamp: u32) -> u32;
}

/// `threadIdx.x`.
#[inline(always)]
pub fn workitem_id_x() -> u32 {
    // SAFETY: a pure read of the `%tid.x` special register.
    unsafe { tid_x() }
}

/// `blockIdx.x`.
#[inline(always)]
pub fn workgroup_id_x() -> u32 {
    // SAFETY: a pure read of the `%ctaid.x` special register.
    unsafe { ctaid_x() }
}

/// `TM_SHFL` for a 32-bit value.
///
/// `src` is masked to the warp rather than left to the hardware clamp so that an
/// out-of-range index wraps exactly the way the AMD path wraps it. Every caller already
/// passes a value below 32, so this only pins the two vendors together; it does not change
/// any result the kernels actually produce.
#[inline(always)]
pub fn shfl32(value: u32, src: u32) -> u32 {
    // SAFETY: a cross-lane register read with the whole warp in the member mask; the block
    // shape guarantees every lane executes this.
    unsafe {
        shfl_sync_idx(
            FULL_WARP,
            value,
            src & (THREADS_PER_LANE - 1),
            CLAMP_WIDTH_32,
        )
    }
}

/// `TM_SHFL_XOR`. `shfl.sync.bfly` *is* the xor-shuffle, so unlike the AMD path this does
/// not have to compute a lane id first.
#[inline(always)]
pub fn shfl_xor32(value: u32, mask: u32) -> u32 {
    // SAFETY: as for [`shfl32`].
    unsafe { shfl_sync_bfly(FULL_WARP, value, mask, CLAMP_WIDTH_32) }
}
