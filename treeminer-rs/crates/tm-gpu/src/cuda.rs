//! Minimal, hand-declared bindings to the CUDA **driver** API — the NVIDIA mirror of
//! [`crate::hip`], with the same public surface so `runner.rs` and `device.rs` do not know
//! which one they are talking to.
//!
//! # This has never been executed
//!
//! There is no NVIDIA GPU on the machine where this was written, and no CUDA driver: not
//! one digest, not one `cuInit`, has ever come out of this file. It compiles. Every claim
//! below is a claim about the CUDA documentation and about the source, not about a result.
//! Before trusting a submission produced through this path, run `tests/parity/run_parity.sh`
//! and `cargo test -p tm-gpu` on real hardware. See PORT.md's support matrix.
//!
//! # Why the driver API and not the runtime API
//!
//! The kernels ship as PTX and are loaded with `cuModuleLoadData`, which is a driver-API
//! call; mixing in the runtime API would mean linking `libcudart` as well and reasoning
//! about two context stacks. The driver library (`libcuda`) is part of the *driver*, so it
//! is present wherever a GPU is, with no toolkit installed.
//!
//! As in the HIP bindings, `cuDeviceGetAttribute`-style structs are avoided: only opaque
//! handles, integers and pointers cross this boundary, plus `CUDA_MEMCPY2D`, which is
//! versioned (`_v2`) precisely so that its layout is fixed.
//!
//! Every entry point is spelled with its explicit `_v2` suffix where one exists. `cuda.h`
//! applies those suffixes with `#define`s that a hand-written `extern` block does not get,
//! and the unsuffixed symbols are the 32-bit-size 1.0 ABI — binding those by accident would
//! truncate every allocation above 4 GiB.

#![allow(non_camel_case_types)]

use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
use std::sync::{Mutex, Once};

use crate::error::{GpuError, Result};

pub mod module;

pub type CUresult = c_int;
pub type CUdevice = c_int;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

/// `CUmemorytype`. Only the two the miner copies between.
const CU_MEMORYTYPE_HOST: c_uint = 1;
const CU_MEMORYTYPE_DEVICE: c_uint = 2;

/// Direction of a 2D copy. The values match HIP's `hipMemcpyKind` so that `runner.rs` can
/// pass the same constant to either backend.
pub const MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// `CUDA_MEMCPY2D_v2`. Field order and types are from `cuda.h`; the four-byte hole after
/// each `CUmemorytype` is what `#[repr(C)]` inserts anyway, which is the only reason
/// declaring this by hand is defensible.
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaMemcpy2D {
    src_x_in_bytes: usize,
    src_y: usize,
    src_memory_type: c_uint,
    src_host: *const c_void,
    src_device: CUdeviceptr,
    src_array: *mut c_void,
    src_pitch: usize,

    dst_x_in_bytes: usize,
    dst_y: usize,
    dst_memory_type: c_uint,
    dst_host: *mut c_void,
    dst_device: CUdeviceptr,
    dst_array: *mut c_void,
    dst_pitch: usize,

    width_in_bytes: usize,
    height: usize,
}

extern "C" {
    fn cuGetErrorString(error: CUresult, string: *mut *const c_char) -> CUresult;

    fn cuInit(flags: c_uint) -> CUresult;
    fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    fn cuDeviceGetName(name: *mut c_char, len: c_int, device: CUdevice) -> CUresult;
    fn cuDeviceGetPCIBusId(pci_bus_id: *mut c_char, len: c_int, device: CUdevice) -> CUresult;
    fn cuDeviceTotalMem_v2(bytes: *mut usize, device: CUdevice) -> CUresult;
    fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

    fn cuDevicePrimaryCtxRetain(context: *mut CUcontext, device: CUdevice) -> CUresult;
    fn cuCtxSetCurrent(context: CUcontext) -> CUresult;
    fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;

    fn cuMemAlloc_v2(pointer: *mut CUdeviceptr, size: usize) -> CUresult;
    fn cuMemFree_v2(pointer: CUdeviceptr) -> CUresult;
    fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, size: usize) -> CUresult;
    fn cuMemcpy2DAsync_v2(copy: *const CudaMemcpy2D, stream: CUstream) -> CUresult;

    fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    fn cuStreamSynchronize(stream: CUstream) -> CUresult;

    fn cuEventCreate(event: *mut CUevent, flags: c_uint) -> CUresult;
    fn cuEventDestroy_v2(event: CUevent) -> CUresult;
    fn cuEventRecord(event: CUevent, stream: CUstream) -> CUresult;
    fn cuEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;
}

extern "C" {
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleGetFunction(
        function: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    #[allow(clippy::too_many_arguments)]
    pub fn cuLaunchKernel(
        function: CUfunction,
        grid_dim_x: c_uint,
        grid_dim_y: c_uint,
        grid_dim_z: c_uint,
        block_dim_x: c_uint,
        block_dim_y: c_uint,
        block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

/// `cuInit` must precede every other driver call and is documented as idempotent.
///
/// HIP has no equivalent — its runtime initialises itself lazily — so this is the one piece
/// of the CUDA layer with no counterpart in `hip.rs`. Every entry point below goes through
/// it, because any of them can be the first call the miner makes.
fn initialize() -> Result<()> {
    static INIT: Once = Once::new();
    static mut RESULT: CUresult = CUDA_SUCCESS;
    INIT.call_once(|| {
        // SAFETY: `cuInit` takes an integer; the write is inside `call_once`, so it happens
        // exactly once and is ordered before every later read by `Once`'s barrier.
        unsafe { RESULT = cuInit(0) };
    });
    // SAFETY: `call_once` has returned, so the write above happened-before this read and no
    // further write can occur.
    check("cuInit", unsafe { RESULT })
}

/// Turns a `CUresult` into a `Result`, attaching the driver's own message.
pub fn check(call: &'static str, code: CUresult) -> Result<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    Err(GpuError::Cuda {
        call,
        code,
        message: error_string(code),
    })
}

pub fn error_string(code: CUresult) -> String {
    let mut text: *const c_char = std::ptr::null();
    // SAFETY: `text` is a live out-parameter. On success the driver stores a pointer to one
    // of its own static strings; on failure it leaves it null, which is handled below.
    let status = unsafe { cuGetErrorString(code, &mut text) };
    if status != CUDA_SUCCESS || text.is_null() {
        return format!("unknown cuda error {code}");
    }
    // SAFETY: non-null and NUL-terminated as documented; the storage is static, so the
    // borrow cannot dangle before it is copied into a String.
    unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
}

/// The primary context of each device the miner has touched.
///
/// The driver API, unlike HIP, has no implicit per-thread current device: a context has to
/// be made current before any allocation or launch. Retaining the *primary* context is what
/// makes this interoperate with anything else in the process that uses the runtime API,
/// which shares that same context.
static CONTEXTS: Mutex<BTreeMap<i32, usize>> = Mutex::new(BTreeMap::new());

pub fn device_count() -> Result<i32> {
    initialize()?;
    let mut count: c_int = 0;
    // SAFETY: `count` is a live, aligned i32 for the duration of the call.
    check("cuDeviceGetCount", unsafe { cuDeviceGetCount(&mut count) })?;
    Ok(count)
}

/// The `CUdevice` handle for an ordinal. It happens to be an `int` on every platform, but
/// the driver still insists it be produced by `cuDeviceGet` rather than assumed.
fn device_handle(index: i32) -> Result<CUdevice> {
    initialize()?;
    let mut device: CUdevice = 0;
    // SAFETY: `device` is a live out-parameter.
    check("cuDeviceGet", unsafe { cuDeviceGet(&mut device, index) })?;
    Ok(device)
}

/// Binds `index`'s primary context to the calling thread — the CUDA equivalent of
/// `hipSetDevice`.
pub fn set_device(index: i32) -> Result<()> {
    let device = device_handle(index)?;
    let mut cache = CONTEXTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let context = match cache.get(&index) {
        Some(context) => *context as CUcontext,
        None => {
            let mut context: CUcontext = std::ptr::null_mut();
            // SAFETY: `context` is a live out-parameter; the retained context is released
            // only by process exit, which is why the handle can be cached.
            check("cuDevicePrimaryCtxRetain", unsafe {
                cuDevicePrimaryCtxRetain(&mut context, device)
            })?;
            cache.insert(index, context as usize);
            context
        }
    };
    drop(cache);
    // SAFETY: the handle came from cuDevicePrimaryCtxRetain and is still retained.
    check("cuCtxSetCurrent", unsafe { cuCtxSetCurrent(context) })
}

/// The device whose context is current on this thread. Modules are per context, so the
/// module cache needs this as its key.
pub fn current_device() -> Result<i32> {
    initialize()?;
    let mut device: CUdevice = 0;
    // SAFETY: `device` is a live out-parameter.
    check("cuCtxGetDevice", unsafe { cuCtxGetDevice(&mut device) })?;
    Ok(device)
}

/// Reads a NUL-terminated string the driver writes into a caller-provided buffer.
fn read_into_buffer(
    call: &'static str,
    capacity: usize,
    fill: impl FnOnce(*mut c_char, c_int) -> CUresult,
) -> Result<String> {
    let mut buffer = vec![0u8; capacity];
    // SAFETY: the buffer is `capacity` bytes and the driver is handed exactly that length,
    // so it cannot write past the end. `buffer` outlives the call.
    check(call, fill(buffer.as_mut_ptr().cast::<c_char>(), capacity as c_int))?;
    let end = buffer.iter().position(|byte| *byte == 0).unwrap_or(capacity);
    buffer.truncate(end);
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

pub fn device_name(index: i32) -> Result<String> {
    let device = device_handle(index)?;
    read_into_buffer("cuDeviceGetName", 256, |pointer, len| unsafe {
        cuDeviceGetName(pointer, len, device)
    })
}

/// The `0000:03:00.0`-style bus id string, in the same format HIP reports.
pub fn device_pci_bus_id(index: i32) -> Result<String> {
    let device = device_handle(index)?;
    read_into_buffer("cuDeviceGetPCIBusId", 64, |pointer, len| unsafe {
        cuDeviceGetPCIBusId(pointer, len, device)
    })
}

pub fn device_total_memory(index: i32) -> Result<usize> {
    let device = device_handle(index)?;
    let mut bytes = 0usize;
    // SAFETY: `bytes` is a live, aligned usize for the duration of the call.
    check("cuDeviceTotalMem", unsafe {
        cuDeviceTotalMem_v2(&mut bytes, device)
    })?;
    Ok(bytes)
}

/// Free and total bytes on the device whose context is current.
pub fn mem_get_info() -> Result<(usize, usize)> {
    initialize()?;
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: both out-parameters are live, aligned usize values.
    check("cuMemGetInfo", unsafe {
        cuMemGetInfo_v2(&mut free, &mut total)
    })?;
    Ok((free, total))
}

/// An owned device allocation. Mirrors [`crate::hip::DeviceBuffer`], including exposing the
/// address as `*mut c_void`: a `CUdeviceptr` is a 64-bit integer rather than a pointer, but
/// the two are the same value on every 64-bit platform CUDA supports, and keeping the type
/// identical is what lets `runner.rs` be vendor-agnostic.
#[derive(Debug)]
pub struct DeviceBuffer {
    pointer: CUdeviceptr,
    size: usize,
}

// SAFETY: a device pointer is just an address; CUDA allows it to be used from any host
// thread that has the owning context current, and `DeviceBuffer` hands out no interior
// references.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn new(size: usize) -> Result<Self> {
        initialize()?;
        let mut pointer: CUdeviceptr = 0;
        // SAFETY: `pointer` is a live out-parameter; on success the driver owns `size` bytes
        // at the returned address and this struct becomes their sole owner.
        check("cuMemAlloc", unsafe { cuMemAlloc_v2(&mut pointer, size) })?;
        Ok(Self { pointer, size })
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.pointer as *mut c_void
    }

    pub fn len(&self) -> usize {
        self.size
    }

    /// Blocking host-to-device copy of `source` into the start of this buffer.
    pub fn copy_from_host(&mut self, source: &[u8]) -> Result<()> {
        if source.len() > self.size {
            return Err(GpuError::Invalid(format!(
                "host->device copy of {} bytes into a {}-byte buffer",
                source.len(),
                self.size
            )));
        }
        // SAFETY: the length check above guarantees the destination holds `source.len()`
        // bytes, and `source` is a live host slice for the duration of this blocking call.
        check("cuMemcpyHtoD", unsafe {
            cuMemcpyHtoD_v2(
                self.pointer,
                source.as_ptr().cast::<c_void>(),
                source.len(),
            )
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if self.pointer != 0 {
            // SAFETY: the pointer came from cuMemAlloc and is freed exactly once, here.
            unsafe { cuMemFree_v2(self.pointer) };
        }
    }
}

/// A strided host<->device copy, with [`crate::hip::memcpy_2d_async`]'s signature so the
/// runner can call either.
///
/// CUDA has no `kind` argument: the direction is expressed by which memory-type fields of
/// `CUDA_MEMCPY2D` are set. The HIP-shaped `kind` is translated here rather than at the
/// call site so that the runner stays vendor-neutral.
///
/// # Safety
/// The caller guarantees that both pointers address at least `height` rows of `width` bytes
/// at the given pitches, that the host side stays alive and untouched until the stream is
/// synchronised, and that `kind` matches which pointer is which.
#[allow(clippy::too_many_arguments)]
pub unsafe fn memcpy_2d_async(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: c_int,
    stream: CUstream,
) -> Result<()> {
    let mut copy = CudaMemcpy2D {
        src_x_in_bytes: 0,
        src_y: 0,
        src_memory_type: 0,
        src_host: std::ptr::null(),
        src_device: 0,
        src_array: std::ptr::null_mut(),
        src_pitch: spitch,

        dst_x_in_bytes: 0,
        dst_y: 0,
        dst_memory_type: 0,
        dst_host: std::ptr::null_mut(),
        dst_device: 0,
        dst_array: std::ptr::null_mut(),
        dst_pitch: dpitch,

        width_in_bytes: width,
        height,
    };
    match kind {
        MEMCPY_HOST_TO_DEVICE => {
            copy.src_memory_type = CU_MEMORYTYPE_HOST;
            copy.src_host = src;
            copy.dst_memory_type = CU_MEMORYTYPE_DEVICE;
            copy.dst_device = dst as CUdeviceptr;
        }
        MEMCPY_DEVICE_TO_HOST => {
            copy.src_memory_type = CU_MEMORYTYPE_DEVICE;
            copy.src_device = src as CUdeviceptr;
            copy.dst_memory_type = CU_MEMORYTYPE_HOST;
            copy.dst_host = dst;
        }
        other => {
            return Err(GpuError::Invalid(format!(
                "unsupported 2D copy direction {other}"
            )))
        }
    }
    // SAFETY: `copy` is fully initialised and outlives the call — the driver reads the
    // descriptor synchronously even though the transfer itself is queued. The buffers it
    // names are the caller's responsibility, as documented above.
    check("cuMemcpy2DAsync", unsafe {
        cuMemcpy2DAsync_v2(std::ptr::addr_of!(copy), stream)
    })
}

/// A CUDA stream, destroyed on drop.
#[derive(Debug)]
pub struct Stream(CUstream);

// SAFETY: CUDA streams may be used from any host thread with the owning context current;
// the handle is an opaque pointer with no host-side aliasing.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        initialize()?;
        let mut stream: CUstream = std::ptr::null_mut();
        // SAFETY: `stream` is a live out-parameter; flag 0 is CU_STREAM_DEFAULT.
        check("cuStreamCreate", unsafe { cuStreamCreate(&mut stream, 0) })?;
        Ok(Self(stream))
    }

    pub fn raw(&self) -> CUstream {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: the handle is valid until `Drop`, which is the only place it is destroyed.
        check("cuStreamSynchronize", unsafe { cuStreamSynchronize(self.0) })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // Queued work must finish before the stream and the buffers it references go
            // away, exactly as on the HIP side.
            // SAFETY: valid handle, destroyed exactly once.
            unsafe {
                cuStreamSynchronize(self.0);
                cuStreamDestroy_v2(self.0);
            }
        }
    }
}

/// A CUDA event, destroyed on drop.
#[derive(Debug)]
pub struct Event(CUevent);

// SAFETY: as for `Stream` — an opaque handle usable from any host thread.
unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        initialize()?;
        let mut event: CUevent = std::ptr::null_mut();
        // SAFETY: `event` is a live out-parameter; flag 0 is CU_EVENT_DEFAULT, which is the
        // timing-enabled one `cuEventElapsedTime` requires.
        check("cuEventCreate", unsafe { cuEventCreate(&mut event, 0) })?;
        Ok(Self(event))
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        // SAFETY: both handles are owned and alive.
        check("cuEventRecord", unsafe {
            cuEventRecord(self.0, stream.raw())
        })
    }

    /// Milliseconds between two recorded events. Both must have been recorded and the
    /// stream synchronised, or the driver reports `CUDA_ERROR_NOT_READY`.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        let mut ms = 0.0f32;
        // SAFETY: `ms` is a live out-parameter; both handles are owned and alive.
        check("cuEventElapsedTime", unsafe {
            cuEventElapsedTime(&mut ms, start.0, end.0)
        })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: valid handle, destroyed exactly once.
            unsafe { cuEventDestroy_v2(self.0) };
        }
    }
}

/// Launches the one-shot Argon2 kernel. Mirrors [`crate::hip::launch_oneshot`].
///
/// # Safety
/// `memory` must point to at least `batch_size * segment_blocks * 4 * 1024` device bytes
/// whose first two blocks per job are already filled, and must stay alive until `stream` is
/// synchronised.
pub unsafe fn launch_oneshot(
    stream: &Stream,
    memory: *mut c_void,
    segment_blocks: u32,
    batch_size: usize,
) -> Result<()> {
    // SAFETY: forwarded verbatim; the contract above is the module's contract.
    unsafe { module::launch_oneshot(stream, memory, segment_blocks, batch_size) }
}

/// Launches the device-side first-blocks kernel. Mirrors
/// [`crate::hip::launch_first_blocks`].
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
    // SAFETY: forwarded verbatim; the contract above is the module's contract.
    unsafe {
        module::launch_first_blocks(
            stream,
            memory,
            keys,
            key_length,
            salt,
            salt_length,
            output_length,
            memory_cost,
            time_cost,
            version,
            type_,
            lanes,
            segment_blocks,
            batch_size,
        )
    }
}
