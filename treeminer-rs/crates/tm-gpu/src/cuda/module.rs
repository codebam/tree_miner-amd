//! Loads the PTX built from `../tm-kernel` and launches its two kernels — the NVIDIA mirror
//! of [`crate::module`].
//!
//! **UNTESTED.** No NVIDIA GPU has run this. `cuModuleLoadData` has never been called with
//! this PTX, so the JIT has never had an opinion about it. What *is* checked, at test time
//! on any machine: that the PTX contains both kernels, that it is self-contained (no
//! `.extern .func` for the JIT to fail to resolve), and that the host-side parameter list
//! matches the kernel signature the PTX declares, parameter for parameter.
//!
//! Unlike the HIP path, arguments go through `cuLaunchKernel`'s `kernelParams` — an array of
//! pointers, one per parameter. That is the driver API's normal convention and it works for
//! module-loaded functions because PTX carries the parameter list with it, which is exactly
//! what a HIP code object does not. The packed-buffer form (`CU_LAUNCH_PARAM_BUFFER_POINTER`)
//! exists here too, but there is no reason to hand-pack a layout the driver can read.

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::sync::Mutex;

use crate::cuda::{self, CUfunction, CUmodule, Stream};
use crate::error::{GpuError, Result};

/// The PTX, produced by `build.rs`. Embedded rather than loaded from a path so the miner
/// cannot be separated from its kernel by a move or a `cargo clean`.
static KERNEL_PTX: &str = include_str!(env!("TM_PTX_KERNEL"));

/// A loaded kernel pair. The driver hands back opaque handles into its own state, which
/// lives as long as the context; they are only ever passed back, never dereferenced.
#[derive(Clone, Copy)]
struct LoadedKernel {
    first_blocks: CUfunction,
    oneshot: CUfunction,
}

// SAFETY: `CUfunction` is an opaque handle into the driver, usable from any host thread
// whose current context owns the module; nothing here dereferences it.
unsafe impl Send for LoadedKernel {}

/// Modules belong to a context, and there is one primary context per device, so the cache
/// is keyed by device ordinal — as on the HIP side.
static MODULES: Mutex<BTreeMap<i32, LoadedKernel>> = Mutex::new(BTreeMap::new());

fn loaded_kernel() -> Result<LoadedKernel> {
    let device = cuda::current_device()?;
    let mut cache = MODULES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(kernel) = cache.get(&device) {
        return Ok(*kernel);
    }

    // `cuModuleLoadData` reads a NUL-terminated image; `include_str!` gives text without
    // one. The copy is made once per device, behind the cache.
    let image = CString::new(KERNEL_PTX)
        .map_err(|_| GpuError::Invalid("the embedded PTX contains a NUL byte".to_owned()))?;

    let mut module: CUmodule = std::ptr::null_mut();
    // SAFETY: `image` is a NUL-terminated buffer that outlives the call and `module` is a
    // live out-parameter. The driver JITs the PTX during this call.
    cuda::check("cuModuleLoadData", unsafe {
        cuda::cuModuleLoadData(&mut module, image.as_ptr().cast::<c_void>())
    })
    .map_err(|error| {
        GpuError::Invalid(format!(
            "loading the Rust PTX kernel failed ({error}); the PTX targets sm_70, so a \
             pre-Volta card cannot run it. NOTE: this NVIDIA path has never been executed \
             on real hardware — see PORT.md before trusting any result from it"
        ))
    })?;

    let kernel = LoadedKernel {
        first_blocks: function(module, "argon2_first_blocks_kernel")?,
        oneshot: function(module, "argon2_kernel_oneshot")?,
    };
    cache.insert(device, kernel);
    Ok(kernel)
}

/// Looks one kernel up in a loaded module.
fn function(module: CUmodule, name: &str) -> Result<CUfunction> {
    let name = CString::new(name).map_err(|_| GpuError::Invalid("kernel name".to_owned()))?;
    let mut function: CUfunction = std::ptr::null_mut();
    // SAFETY: `module` was just loaded, `name` outlives the call, `function` is a live
    // out-parameter.
    cuda::check("cuModuleGetFunction", unsafe {
        cuda::cuModuleGetFunction(&mut function, module, name.as_ptr())
    })?;
    Ok(function)
}

/// `argon2_first_blocks_kernel`'s parameters, in declaration order.
///
/// The order is the kernel's, not the C++'s: pointers, then the 64-bit count, then the
/// 32-bit scalars. That regrouping was done for the AMDGPU kernarg segment, and CUDA does
/// not care — but the two vendors share one kernel source, so the order is shared too.
#[repr(C)]
struct FirstBlocksArgs {
    memory: *mut c_void,
    keys: *const c_void,
    salt: *const c_void,
    batch_size: u64,
    key_length: u32,
    salt_length: u32,
    output_length: u32,
    memory_cost: u32,
    time_cost: u32,
    version: u32,
    type_: u32,
    lanes: u32,
    segment_blocks: u32,
    threads_per_block: u32,
}

/// The size of each `FirstBlocksArgs` field, in declaration order.
///
/// This is the bridge the tests check in both directions: against the PTX's `.param` list
/// (so the host agrees with the kernel) and against the struct's own field offsets (so the
/// table agrees with the pointer array actually handed to `cuLaunchKernel`). Neither the
/// compiler nor the driver can catch a reordering here, because the two sides are built by
/// separate compilations for separate targets.
// Read only by the tests, but it belongs next to the struct it describes: it is the written
// form of an ABI contract, not a test fixture.
#[allow(dead_code)]
const FIRST_BLOCKS_PARAM_SIZES: [usize; 14] = [8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4];

/// The `kernelParams` array: one pointer per parameter, in declaration order.
fn first_blocks_params(args: &mut FirstBlocksArgs) -> [*mut c_void; 14] {
    [
        std::ptr::addr_of_mut!(args.memory).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.keys).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.salt).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.batch_size).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.key_length).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.salt_length).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.output_length).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.memory_cost).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.time_cost).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.version).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.type_).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.lanes).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.segment_blocks).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.threads_per_block).cast::<c_void>(),
    ]
}

/// `argon2_kernel_oneshot`'s parameters.
#[repr(C)]
struct OneshotArgs {
    memory: *mut c_void,
    segment_blocks: u32,
}

#[allow(dead_code)]
const ONESHOT_PARAM_SIZES: [usize; 2] = [8, 4];

fn oneshot_params(args: &mut OneshotArgs) -> [*mut c_void; 2] {
    [
        std::ptr::addr_of_mut!(args.memory).cast::<c_void>(),
        std::ptr::addr_of_mut!(args.segment_blocks).cast::<c_void>(),
    ]
}

/// Matches `THREADS_PER_BLOCK` in `crate::module`, so both vendors shard the batch the same
/// way and a difference in results can never be a difference in launch geometry.
const THREADS_PER_BLOCK: u32 = 128;

/// One block per job, 32 threads wide: the Argon2 lane *is* 32 threads, and on NVIDIA that
/// is exactly one warp, which is what makes the kernel's `shfl.sync` full-warp masks valid.
const THREADS_PER_LANE: u32 = 32;

/// Hands a `kernelParams` array to `cuLaunchKernel`.
///
/// # Safety
/// `function` must belong to the current context's module, and every pointer in `params`
/// must address a live value of the type the corresponding kernel parameter declares.
unsafe fn launch(
    function: CUfunction,
    grid: u32,
    block: u32,
    stream: &Stream,
    params: &mut [*mut c_void],
) -> Result<()> {
    // SAFETY: the caller guarantees the handle and the parameter pointers; the driver reads
    // both the array and the values it points at before returning, so the borrow is enough.
    cuda::check("cuLaunchKernel", unsafe {
        cuda::cuLaunchKernel(
            function,
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    })
}

/// Launches the first-blocks kernel. Signature and semantics mirror
/// [`crate::hip::launch_first_blocks`] exactly.
///
/// # Safety
/// `memory` must be the batch pool, `keys` must hold `batch_size * key_length` device bytes
/// and `salt` `salt_length`, and all three must stay alive until `stream` is synchronised.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_first_blocks(
    stream: &Stream,
    memory: *mut c_void,
    keys: *const c_void,
    key_length: u32,
    salt: *const c_void,
    salt_length: u32,
    output_length: u32,
    memory_cost: u32,
    time_cost: u32,
    version: u32,
    type_: u32,
    lanes: u32,
    segment_blocks: u32,
    batch_size: usize,
) -> Result<()> {
    let kernel = loaded_kernel()?;

    let mut args = FirstBlocksArgs {
        memory,
        keys,
        salt,
        batch_size: batch_size as u64,
        key_length,
        salt_length,
        output_length,
        memory_cost,
        time_cost,
        version,
        type_,
        lanes,
        segment_blocks,
        threads_per_block: THREADS_PER_BLOCK,
    };
    let mut params = first_blocks_params(&mut args);

    let grid = batch_size.div_ceil(THREADS_PER_BLOCK as usize);
    let grid = u32::try_from(grid)
        .map_err(|_| GpuError::Invalid(format!("batch of {batch_size} exceeds one grid")))?;

    // SAFETY: the handle came from cuModuleGetFunction, and every pointer in `params`
    // addresses a field of `args`, which outlives the call.
    unsafe { launch(kernel.first_blocks, grid, THREADS_PER_BLOCK, stream, &mut params) }
}

/// Launches the one-shot Argon2 kernel. Mirrors [`crate::hip::launch_oneshot`].
///
/// # Safety
/// `memory` must hold `batch_size` jobs of `4 * segment_blocks` 1 KiB blocks with blocks 0
/// and 1 of each already filled, and must stay alive until `stream` is synchronised.
pub unsafe fn launch_oneshot(
    stream: &Stream,
    memory: *mut c_void,
    segment_blocks: u32,
    batch_size: usize,
) -> Result<()> {
    let kernel = loaded_kernel()?;

    let mut args = OneshotArgs {
        memory,
        segment_blocks,
    };
    let mut params = oneshot_params(&mut args);

    let grid = u32::try_from(batch_size)
        .map_err(|_| GpuError::Invalid(format!("batch of {batch_size} exceeds one grid")))?;

    // SAFETY: as above.
    unsafe { launch(kernel.oneshot, grid, THREADS_PER_LANE, stream, &mut params) }
}

/// What can be checked about the NVIDIA path without an NVIDIA GPU: the PTX itself, and the
/// agreement between the kernel signatures it declares and the arguments the host packs.
///
/// None of this proves the kernels compute Argon2 correctly on NVIDIA hardware. That needs
/// `fixtures/argon2_vectors.json` and `tests/parity/run_parity.sh` on a real card, and until
/// someone runs them the NVIDIA path is unproven — see PORT.md.
#[cfg(test)]
mod tests {
    use super::*;

    /// The `.param` sizes of one `.visible .entry` in the embedded PTX, in order.
    fn ptx_entry_params(name: &str) -> Vec<usize> {
        let header = format!(".visible .entry {name}(");
        let start = KERNEL_PTX
            .find(&header)
            .unwrap_or_else(|| panic!("the PTX declares no kernel named {name}"));
        let body = &KERNEL_PTX[start + header.len()..];
        let end = body.find(')').expect("the parameter list is closed");
        body[..end]
            .split(',')
            .map(str::trim)
            .filter(|parameter| !parameter.is_empty())
            // `.param .u64 .ptr .align 1 name` — the width is the token after `.param`.
            .map(|parameter| match parameter.split_whitespace().nth(1) {
                // PTX spells a parameter's width in its type: `.u64`/`.b64` and `.u32`/`.b32`
                // are the only two the kernels use.
                Some(".u64" | ".b64" | ".s64" | ".f64") => 8,
                Some(".u32" | ".b32" | ".s32" | ".f32") => 4,
                other => panic!("unexpected PTX parameter type {other:?} in {parameter:?}"),
            })
            .collect()
    }

    /// The byte offset of each entry of a `kernelParams` array within its argument struct.
    fn offsets(base: *const u8, params: &[*mut c_void]) -> Vec<usize> {
        params
            .iter()
            .map(|pointer| pointer.cast::<u8>() as usize - base as usize)
            .collect()
    }

    /// A size table turned into the offsets those sizes imply if nothing is padded.
    fn prefix_sums(sizes: &[usize]) -> Vec<usize> {
        sizes
            .iter()
            .scan(0usize, |total, size| {
                let offset = *total;
                *total += size;
                Some(offset)
            })
            .collect()
    }

    #[test]
    fn the_ptx_declares_both_kernels() {
        assert!(KERNEL_PTX.contains(".visible .entry argon2_first_blocks_kernel("));
        assert!(KERNEL_PTX.contains(".visible .entry argon2_kernel_oneshot("));
    }

    /// An unresolved `.extern .func` is a module the driver cannot JIT. The kernel crate
    /// avoids it by keeping every helper in-crate and by the `sigma!` / `buf_slot!` masks
    /// that stop `core::panicking::panic_bounds_check` from being referenced; this is the
    /// test that notices when a future edit undoes that.
    #[test]
    fn the_ptx_is_self_contained() {
        let externals: Vec<&str> = KERNEL_PTX
            .lines()
            .filter(|line| line.trim_start().starts_with(".extern "))
            .collect();
        assert!(
            externals.is_empty(),
            "the PTX references symbols nothing defines, so cuModuleLoadData will fail to \
             JIT it: {externals:?}"
        );
    }

    #[test]
    fn the_ptx_targets_a_gpu_the_shuffles_exist_on() {
        assert!(KERNEL_PTX.contains(".target sm_"), "no .target directive");
        // Both shuffle forms the Argon2 permutation needs must actually be there; a silent
        // fallback to a non-`sync` shuffle would be a correctness bug on Volta and later.
        assert!(KERNEL_PTX.contains("shfl.sync.idx.b32"));
        assert!(KERNEL_PTX.contains("shfl.sync.bfly.b32"));
    }

    /// Every `shfl.sync` must name the whole warp (`-1`) and clamp at lane 31 — the PTX
    /// spelling of CUDA's `__shfl_sync(0xffffffff, v, i, 32)`. A narrower member mask or a
    /// smaller width would silently change which lane each Argon2 word comes from.
    #[test]
    fn every_shuffle_is_a_full_warp_width_32_shuffle() {
        let shuffles: Vec<&str> = KERNEL_PTX
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("shfl.sync."))
            .collect();
        assert!(!shuffles.is_empty(), "the PTX contains no shuffles at all");
        for shuffle in shuffles {
            let operands: Vec<&str> = shuffle
                .trim_end_matches(';')
                .split(',')
                .map(str::trim)
                .collect();
            assert_eq!(
                operands.len(),
                5,
                "shfl.sync takes d, a, b, c, membermask: {shuffle:?}"
            );
            assert_eq!(operands[3], "31", "width is not 32 in {shuffle:?}");
            assert_eq!(operands[4], "-1", "member mask is not full in {shuffle:?}");
        }
    }

    #[test]
    fn the_first_blocks_arguments_match_the_kernel_signature() {
        assert_eq!(ptx_entry_params("argon2_first_blocks_kernel"), FIRST_BLOCKS_PARAM_SIZES);
        assert_eq!(
            FIRST_BLOCKS_PARAM_SIZES.iter().sum::<usize>(),
            std::mem::size_of::<FirstBlocksArgs>(),
            "the argument struct has padding the size table does not describe"
        );

        let mut args = FirstBlocksArgs {
            memory: std::ptr::null_mut(),
            keys: std::ptr::null(),
            salt: std::ptr::null(),
            batch_size: 0,
            key_length: 0,
            salt_length: 0,
            output_length: 0,
            memory_cost: 0,
            time_cost: 0,
            version: 0,
            type_: 0,
            lanes: 0,
            segment_blocks: 0,
            threads_per_block: 0,
        };
        let base = std::ptr::addr_of!(args).cast::<u8>();
        let params = first_blocks_params(&mut args);
        assert_eq!(offsets(base, &params), prefix_sums(&FIRST_BLOCKS_PARAM_SIZES));
    }

    #[test]
    fn the_oneshot_arguments_match_the_kernel_signature() {
        assert_eq!(ptx_entry_params("argon2_kernel_oneshot"), ONESHOT_PARAM_SIZES);

        let mut args = OneshotArgs {
            memory: std::ptr::null_mut(),
            segment_blocks: 0,
        };
        let base = std::ptr::addr_of!(args).cast::<u8>();
        let params = oneshot_params(&mut args);
        assert_eq!(offsets(base, &params), prefix_sums(&ONESHOT_PARAM_SIZES));
    }
}
