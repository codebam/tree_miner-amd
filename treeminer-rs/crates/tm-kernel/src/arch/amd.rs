//! AMDGPU (`amdgcn-amd-amdhsa`) implementation of [`super`].
//!
//! This is the *tested* path: gfx1100, all 44 fixture vectors byte-exact, raw-block
//! differential against the hipcc kernels green. Nothing here may change without re-running
//! that gate.

use super::THREADS_PER_LANE;

extern "C" {
    #[link_name = "llvm.amdgcn.workitem.id.x"]
    fn workitem_id_x_raw() -> u32;
    #[link_name = "llvm.amdgcn.workgroup.id.x"]
    fn workgroup_id_x_raw() -> u32;
    #[link_name = "llvm.amdgcn.mbcnt.lo"]
    fn mbcnt_lo(mask: u32, base: u32) -> u32;
    #[link_name = "llvm.amdgcn.mbcnt.hi"]
    fn mbcnt_hi(mask: u32, base: u32) -> u32;
    #[link_name = "llvm.amdgcn.ds.bpermute"]
    fn ds_bpermute(byte_index: u32, src: u32) -> u32;
}

/// `threadIdx.x`.
#[inline(always)]
pub fn workitem_id_x() -> u32 {
    // SAFETY: a pure read of the dispatch-provided work-item id.
    unsafe { workitem_id_x_raw() }
}

/// `blockIdx.x`.
#[inline(always)]
pub fn workgroup_id_x() -> u32 {
    // SAFETY: a pure read of the dispatch-provided workgroup id.
    unsafe { workgroup_id_x_raw() }
}

/// This work-item's index inside its wavefront.
#[inline(always)]
fn lane_id() -> u32 {
    // SAFETY: both intrinsics are pure reads of the lane-mask registers.
    unsafe { mbcnt_hi(!0, mbcnt_lo(!0, 0)) }
}

/// `TM_SHFL` for a 32-bit value: read `value` from lane `src` of this work-item's *group of
/// 32*. `ds_bpermute` addresses the whole wavefront, so the group base is put back — on a
/// wave64 part lanes 32..63 must read from their own half, exactly as HIP's `__shfl(...,
/// THREADS_PER_LANE)` does.
#[inline(always)]
pub fn shfl32(value: u32, src: u32) -> u32 {
    let lane = (lane_id() & !(THREADS_PER_LANE - 1)) | (src & (THREADS_PER_LANE - 1));
    // SAFETY: `ds_bpermute` is a cross-lane register read; the byte index is a lane number
    // scaled by 4, which is the intrinsic's documented encoding.
    unsafe { ds_bpermute(lane << 2, value) }
}

/// `TM_SHFL_XOR`. For masks below 32 — the only ones used — this is the same lane the HIP
/// builtin picks, because the group base is restored by [`shfl32`].
#[inline(always)]
pub fn shfl_xor32(value: u32, mask: u32) -> u32 {
    shfl32(value, lane_id() ^ mask)
}
