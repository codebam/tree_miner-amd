//! Loads the Rust-compiled AMDGPU code object (`../tm-kernel`) and launches its kernels.
//!
//! Only compiled with the `rust-kernel` feature. The HIP kernel in `kernel/argon2_kernel.hip`
//! remains the default and the fallback: while the feature is on both the first-blocks and
//! the one-shot launch come from here instead, and `hip::launch_*_hip` still reach the C++
//! kernels directly. See PORT.md, "Rust GPU kernels".

use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::sync::Mutex;

use crate::error::{GpuError, Result};
use crate::hip::{self, hipFunction_t, hipModule_t, Stream};

/// The amdgcn code object, produced by `build.rs`. Embedding it rather than loading it from
/// a path means the miner cannot be separated from its kernel by a move or a `cargo clean`.
static KERNEL_IMAGE: &[u8] = include_bytes!(env!("TM_RUST_KERNEL_ELF"));

/// The architecture the image above was compiled for, for a legible error when the code
/// object does not match the installed card.
const KERNEL_ARCH: &str = env!("TM_RUST_KERNEL_ARCH");

/// A loaded kernel handle. HIP hands back raw pointers into runtime-owned state that lives
/// for the process; they are only ever passed back to HIP, never dereferenced here.
#[derive(Clone, Copy)]
struct LoadedKernel {
    first_blocks: hipFunction_t,
    oneshot: hipFunction_t,
}

// SAFETY: `hipFunction_t` is an opaque handle into the HIP runtime, which is documented as
// usable from any host thread; nothing here dereferences it.
unsafe impl Send for LoadedKernel {}

/// Modules are per-device, so the cache is keyed by device ordinal. Loading is idempotent
/// but not free, and a mining loop launches this kernel every batch.
static MODULES: Mutex<BTreeMap<i32, LoadedKernel>> = Mutex::new(BTreeMap::new());

fn loaded_kernel() -> Result<LoadedKernel> {
    let device = hip::current_device()?;
    let mut cache = MODULES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(kernel) = cache.get(&device) {
        return Ok(*kernel);
    }

    let mut module: hipModule_t = std::ptr::null_mut();
    // SAFETY: `KERNEL_IMAGE` is a static, correctly sized code object and `module` is a
    // live out-parameter. HIP copies what it needs out of the image during this call.
    hip::check("hipModuleLoadData", unsafe {
        hip::hipModuleLoadData(&mut module, KERNEL_IMAGE.as_ptr().cast::<c_void>())
    })
    .map_err(|error| {
        GpuError::Invalid(format!(
            "loading the Rust {KERNEL_ARCH} kernel failed ({error}); rebuild without the \
             `rust-kernel` feature, or set TM_GPU_OFFLOAD_ARCH to this device's architecture"
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
fn function(module: hipModule_t, name: &str) -> Result<hipFunction_t> {
    let name = CString::new(name).map_err(|_| GpuError::Invalid("kernel name".to_owned()))?;
    let mut function: hipFunction_t = std::ptr::null_mut();
    // SAFETY: `module` was just loaded, `name` outlives the call, `function` is a live
    // out-parameter.
    hip::check("hipModuleGetFunction", unsafe {
        hip::hipModuleGetFunction(&mut function, module, name.as_ptr())
    })?;
    Ok(function)
}

/// Hands a kernel argument buffer to `hipModuleLaunchKernel`.
///
/// HIP takes the buffer through this sentinel-terminated list. An array of
/// pointers-to-arguments is the *other* HIP calling convention and faults the device here,
/// because a module-loaded code object has no argument metadata for the runtime to marshal
/// against.
///
/// # Safety
/// `args` must point to `args_size` readable bytes matching the kernel's kernarg segment,
/// and `function` must belong to the current device's module.
unsafe fn launch(
    function: hipFunction_t,
    grid: u32,
    block: u32,
    stream: &Stream,
    args: *mut c_void,
    mut args_size: usize,
) -> Result<()> {
    let mut config: [*mut c_void; 5] = [
        hip::HIP_LAUNCH_PARAM_BUFFER_POINTER,
        args,
        hip::HIP_LAUNCH_PARAM_BUFFER_SIZE,
        std::ptr::addr_of_mut!(args_size).cast::<c_void>(),
        hip::HIP_LAUNCH_PARAM_END,
    ];
    // SAFETY: the caller guarantees the handle and the buffer; `args_size` and `config`
    // outlive the call, and HIP copies the kernarg buffer before returning.
    hip::check("hipModuleLaunchKernel", unsafe {
        hip::hipModuleLaunchKernel(
            function,
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            std::ptr::null_mut(),
            config.as_mut_ptr(),
        )
    })
}

/// The kernel argument buffer, laid out to match `argon2_first_blocks_kernel`'s parameter
/// list in `tm-kernel`.
///
/// The AMDGPU kernarg segment places each argument at its natural alignment, so this struct
/// is a faithful mirror only because the kernel's parameters are ordered pointers, then the
/// 64-bit count, then the 32-bit scalars: every field lands where `#[repr(C)]` puts it and
/// there is no interior padding to disagree about. `debug_assert`s below pin the size.
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

/// Matches `tm_launch_argon2_first_blocks` in the HIP shim, so both paths shard the batch
/// the same way.
const THREADS_PER_BLOCK: u32 = 128;

/// Launches the Rust first-blocks kernel. Signature and semantics mirror
/// [`crate::hip::launch_first_blocks`] exactly, so the two are interchangeable.
///
/// # Safety
/// As for the HIP path: `memory` must be the batch pool, `keys` must hold
/// `batch_size * key_length` device bytes and `salt` `salt_length`, and all three must stay
/// alive until `stream` is synchronised.
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
    debug_assert_eq!(std::mem::size_of::<FirstBlocksArgs>(), 72);

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
    let grid = batch_size.div_ceil(THREADS_PER_BLOCK as usize);
    let grid = u32::try_from(grid)
        .map_err(|_| GpuError::Invalid(format!("batch of {batch_size} exceeds one grid")))?;

    // SAFETY: the handle came from hipModuleGetFunction and the argument buffer matches the
    // kernel's kernarg segment (asserted above, cross-checked by the differential test).
    unsafe {
        launch(
            kernel.first_blocks,
            grid,
            THREADS_PER_BLOCK,
            stream,
            std::ptr::addr_of_mut!(args).cast::<c_void>(),
            std::mem::size_of::<FirstBlocksArgs>(),
        )
    }
}

/// `argon2_kernel_oneshot`'s kernarg segment: a pointer then a 32-bit count, 12 bytes with
/// no trailing padding. `#[repr(C)]` rounds the Rust struct up to 16 for its own alignment,
/// so only the first [`ONESHOT_KERNARG_SIZE`] bytes are handed to HIP.
#[repr(C)]
struct OneshotArgs {
    memory: *mut c_void,
    segment_blocks: u32,
}

/// `.kernarg_segment_size` of `argon2_kernel_oneshot`, from `llvm-readelf --notes`.
const ONESHOT_KERNARG_SIZE: usize = 12;

/// One workgroup per job, `THREADS_PER_LANE` work-items wide — the same shape as
/// `tm_launch_argon2_oneshot` in the HIP shim, because the Argon2 lane *is* 32 threads and
/// the work-item id is the thread index.
const THREADS_PER_LANE: u32 = 32;

/// Launches the Rust one-shot Argon2 kernel. Mirrors [`crate::hip::launch_oneshot`].
///
/// # Safety
/// As for the HIP path: `memory` must hold `batch_size` jobs of `4 * segment_blocks` 1 KiB
/// blocks with blocks 0 and 1 of each already filled, and must stay alive until `stream` is
/// synchronised.
pub unsafe fn launch_oneshot(
    stream: &Stream,
    memory: *mut c_void,
    segment_blocks: u32,
    batch_size: usize,
) -> Result<()> {
    let kernel = loaded_kernel()?;
    debug_assert_eq!(std::mem::offset_of!(OneshotArgs, memory), 0);
    debug_assert_eq!(std::mem::offset_of!(OneshotArgs, segment_blocks), 8);
    debug_assert!(std::mem::size_of::<OneshotArgs>() >= ONESHOT_KERNARG_SIZE);

    let mut args = OneshotArgs {
        memory,
        segment_blocks,
    };

    let grid = u32::try_from(batch_size)
        .map_err(|_| GpuError::Invalid(format!("batch of {batch_size} exceeds one grid")))?;

    // SAFETY: the handle came from hipModuleGetFunction and the first 12 bytes of `args`
    // are exactly the kernel's kernarg segment (asserted above, cross-checked by the
    // differential test).
    unsafe {
        launch(
            kernel.oneshot,
            grid,
            THREADS_PER_LANE,
            stream,
            std::ptr::addr_of_mut!(args).cast::<c_void>(),
            ONESHOT_KERNARG_SIZE,
        )
    }
}
