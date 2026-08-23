//! The vendor-specific floor the Argon2 kernels stand on.
//!
//! Everything above this module is written once. Only three things actually differ between
//! an AMD and an NVIDIA GPU as far as these kernels are concerned:
//!
//! 1. the work-item / workgroup (thread / block) index intrinsics,
//! 2. the 32-lane shuffles — `ds_bpermute` on AMD, `shfl.sync` on NVIDIA — which must give
//!    the *same* semantics HIP's `__shfl_sync(m, v, src, 32)` and
//!    `__shfl_xor_sync(m, v, mask, 32)` give in `../tm-gpu/kernel/argon2_kernel.hip`,
//! 3. the kernel entry ABI: `extern "gpu-kernel"` for amdgcn, `extern "ptx-kernel"` for
//!    nvptx64. [`gpu_kernel!`] hides that difference.
//!
//! Every function here is `#[inline(always)]`: the one-shot kernel's inner loop runs
//! `4 * m` times per hash and a single scratch spill would gut the hashrate, so nothing is
//! allowed to become a real call or to hold a value across one.
//!
//! **Support status.** The AMD path is the tested one (gfx1100, all 44 fixture vectors,
//! raw-block differential against the hipcc kernels). The NVIDIA path has *never been
//! executed*: this machine has no NVIDIA GPU. It compiles to PTX and the emitted
//! instructions have been read by eye; that is all. See PORT.md.

#[cfg(target_arch = "amdgpu")]
mod amd;
#[cfg(target_arch = "amdgpu")]
pub use amd::{shfl32, shfl_xor32, workgroup_id_x, workitem_id_x};

#[cfg(target_arch = "nvptx64")]
mod nvidia;
#[cfg(target_arch = "nvptx64")]
pub use nvidia::{shfl32, shfl_xor32, workgroup_id_x, workitem_id_x};

/// The width of the shuffle group. Argon2's block is spread over exactly this many lanes,
/// which is a warp on NVIDIA and half a wavefront on a wave64 AMD part — hence the
/// group-base arithmetic in the AMD implementation.
pub const THREADS_PER_LANE: u32 = 32;

/// Declares a kernel entry point with whichever ABI this target spells it.
///
/// `extern "gpu-kernel"` (amdgcn) and `extern "ptx-kernel"` (nvptx64) are different ABI
/// strings, and an ABI string cannot be produced by `cfg_attr`, so the whole item is
/// generated instead. The body is passed through untouched.
#[cfg(target_arch = "amdgpu")]
macro_rules! gpu_kernel {
    ($(#[$attr:meta])* fn $name:ident ($($arg:ident : $ty:ty),* $(,)?) $body:block) => {
        $(#[$attr])*
        #[no_mangle]
        pub unsafe extern "gpu-kernel" fn $name($($arg: $ty),*) $body
    };
}

#[cfg(target_arch = "nvptx64")]
macro_rules! gpu_kernel {
    ($(#[$attr:meta])* fn $name:ident ($($arg:ident : $ty:ty),* $(,)?) $body:block) => {
        $(#[$attr])*
        #[no_mangle]
        pub unsafe extern "ptx-kernel" fn $name($($arg: $ty),*) $body
    };
}

pub(crate) use gpu_kernel;
