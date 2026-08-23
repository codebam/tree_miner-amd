//! Minimal, hand-declared bindings to the HIP runtime.
//!
//! Only the calls the miner actually makes are declared, and all of them are part of the
//! C ABI HIP guarantees: opaque handles, integers and pointers. Deliberately absent is
//! `hipGetDeviceProperties`, whose `hipDeviceProp_t` layout changes between ROCm releases
//! (ROCm 6 even renamed the symbol) — reproducing that struct by hand would be unsound the
//! moment the runtime is upgraded. The three fields the C++ miner read from it (name, PCI
//! bus id, total memory) each have their own stable entry point, used here instead.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void, CStr};

use crate::error::{GpuError, Result};

pub type hipError_t = c_int;
pub type hipStream_t = *mut c_void;
pub type hipEvent_t = *mut c_void;

pub const HIP_SUCCESS: hipError_t = 0;

pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;

extern "C" {
    fn hipGetErrorString(error: hipError_t) -> *const c_char;

    fn hipGetDeviceCount(count: *mut c_int) -> hipError_t;
    fn hipSetDevice(device: c_int) -> hipError_t;
    fn hipDeviceGetName(name: *mut c_char, len: c_int, device: c_int) -> hipError_t;
    fn hipDeviceGetPCIBusId(pci_bus_id: *mut c_char, len: c_int, device: c_int) -> hipError_t;
    fn hipDeviceTotalMem(bytes: *mut usize, device: c_int) -> hipError_t;
    fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> hipError_t;

    fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> hipError_t;
    fn hipFree(ptr: *mut c_void) -> hipError_t;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: c_int) -> hipError_t;
    fn hipMemcpy2DAsync(
        dst: *mut c_void,
        dpitch: usize,
        src: *const c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: c_int,
        stream: hipStream_t,
    ) -> hipError_t;

    fn hipStreamCreate(stream: *mut hipStream_t) -> hipError_t;
    fn hipStreamDestroy(stream: hipStream_t) -> hipError_t;
    fn hipStreamSynchronize(stream: hipStream_t) -> hipError_t;

    fn hipEventCreate(event: *mut hipEvent_t) -> hipError_t;
    fn hipEventDestroy(event: hipEvent_t) -> hipError_t;
    fn hipEventRecord(event: hipEvent_t, stream: hipStream_t) -> hipError_t;
    fn hipEventElapsedTime(ms: *mut f32, start: hipEvent_t, end: hipEvent_t) -> hipError_t;
}

/// Turns a `hipError_t` into a `Result`, attaching the driver's own message.
pub fn check(call: &'static str, code: hipError_t) -> Result<()> {
    if code == HIP_SUCCESS {
        return Ok(());
    }
    Err(GpuError::Hip {
        call,
        code,
        message: error_string(code),
    })
}

pub fn error_string(code: hipError_t) -> String {
    // SAFETY: hipGetErrorString returns a pointer to a static, NUL-terminated string for
    // every input, including codes it does not recognise.
    let text = unsafe { hipGetErrorString(code) };
    if text.is_null() {
        return format!("unknown hip error {code}");
    }
    // SAFETY: non-null and NUL-terminated as documented; the storage is static so the
    // borrow cannot dangle before it is copied into a String.
    unsafe { CStr::from_ptr(text) }.to_string_lossy().into_owned()
}

pub fn device_count() -> Result<i32> {
    let mut count: c_int = 0;
    // SAFETY: `count` is a live, aligned i32 for the duration of the call.
    check("hipGetDeviceCount", unsafe {
        hipGetDeviceCount(&mut count)
    })?;
    Ok(count)
}

pub fn set_device(index: i32) -> Result<()> {
    // SAFETY: takes an integer only.
    check("hipSetDevice", unsafe { hipSetDevice(index) })
}

/// Reads a NUL-terminated string HIP writes into a caller-provided buffer.
fn read_into_buffer(
    call: &'static str,
    capacity: usize,
    fill: impl FnOnce(*mut c_char, c_int) -> hipError_t,
) -> Result<String> {
    let mut buffer = vec![0u8; capacity];
    // SAFETY: the buffer is `capacity` bytes and we hand HIP exactly that length, so it
    // cannot write past the end. `buffer` outlives the call.
    check(call, fill(buffer.as_mut_ptr().cast::<c_char>(), capacity as c_int))?;
    let end = buffer.iter().position(|byte| *byte == 0).unwrap_or(capacity);
    buffer.truncate(end);
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

pub fn device_name(index: i32) -> Result<String> {
    read_into_buffer("hipDeviceGetName", 256, |pointer, len| unsafe {
        hipDeviceGetName(pointer, len, index)
    })
}

/// The `0000:03:00.0`-style bus id string HIP reports for a device.
pub fn device_pci_bus_id(index: i32) -> Result<String> {
    read_into_buffer("hipDeviceGetPCIBusId", 64, |pointer, len| unsafe {
        hipDeviceGetPCIBusId(pointer, len, index)
    })
}

pub fn device_total_memory(index: i32) -> Result<usize> {
    let mut bytes = 0usize;
    // SAFETY: `bytes` is a live, aligned usize for the duration of the call.
    check("hipDeviceTotalMem", unsafe {
        hipDeviceTotalMem(&mut bytes, index)
    })?;
    Ok(bytes)
}

/// Free and total bytes on the *currently active* device.
pub fn mem_get_info() -> Result<(usize, usize)> {
    let mut free = 0usize;
    let mut total = 0usize;
    // SAFETY: both out-parameters are live, aligned usize values.
    check("hipMemGetInfo", unsafe {
        hipMemGetInfo(&mut free, &mut total)
    })?;
    Ok((free, total))
}

/// An owned device allocation. Freeing is the only thing `Drop` does, so a leak on a failed
/// free is impossible and a double free is prevented by ownership.
#[derive(Debug)]
pub struct DeviceBuffer {
    pointer: *mut c_void,
    size: usize,
}

// SAFETY: a device pointer is just an address; HIP allows it to be used from any host
// thread, and `DeviceBuffer` hands out no interior references.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    pub fn new(size: usize) -> Result<Self> {
        let mut pointer: *mut c_void = std::ptr::null_mut();
        // SAFETY: `pointer` is a live out-parameter; on success HIP owns `size` bytes at
        // the returned address and this struct becomes their sole owner.
        check("hipMalloc", unsafe { hipMalloc(&mut pointer, size) })?;
        Ok(Self { pointer, size })
    }

    pub fn as_ptr(&self) -> *mut c_void {
        self.pointer
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
        check("hipMemcpy", unsafe {
            hipMemcpy(
                self.pointer,
                source.as_ptr().cast::<c_void>(),
                source.len(),
                HIP_MEMCPY_HOST_TO_DEVICE,
            )
        })
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.pointer.is_null() {
            // SAFETY: the pointer came from hipMalloc and is freed exactly once, here.
            unsafe { hipFree(self.pointer) };
        }
    }
}

/// A strided host<->device copy. `dst`/`src` must each be valid for `height * pitch` bytes
/// on their respective sides; the callers in `runner.rs` size both from the same batch
/// shape, which is the invariant that makes this sound.
///
/// # Safety
/// The caller guarantees that both pointers address at least `height` rows of `width`
/// bytes at the given pitches, that the host side stays alive and untouched until the
/// stream is synchronised, and that `kind` matches which pointer is which.
// Mirrors hipMemcpy2DAsync one for one; renaming or grouping its arguments would only
// obscure which HIP parameter each one is.
#[allow(clippy::too_many_arguments)]
pub unsafe fn memcpy_2d_async(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: c_int,
    stream: hipStream_t,
) -> Result<()> {
    check("hipMemcpy2DAsync", {
        hipMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
    })
}

/// A HIP stream, destroyed on drop.
#[derive(Debug)]
pub struct Stream(hipStream_t);

// SAFETY: HIP streams may be used from any host thread; the handle is an opaque pointer
// with no host-side aliasing.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new() -> Result<Self> {
        let mut stream: hipStream_t = std::ptr::null_mut();
        // SAFETY: `stream` is a live out-parameter.
        check("hipStreamCreate", unsafe { hipStreamCreate(&mut stream) })?;
        Ok(Self(stream))
    }

    pub fn raw(&self) -> hipStream_t {
        self.0
    }

    pub fn synchronize(&self) -> Result<()> {
        // SAFETY: the handle is valid until `Drop`, which is the only place it is destroyed.
        check("hipStreamSynchronize", unsafe {
            hipStreamSynchronize(self.0)
        })
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // Queued work must finish before the stream and the buffers it references go
            // away; the C++ runner does the same drain in its destructor.
            // SAFETY: valid handle, destroyed exactly once.
            unsafe {
                hipStreamSynchronize(self.0);
                hipStreamDestroy(self.0);
            }
        }
    }
}

/// A HIP event, destroyed on drop.
#[derive(Debug)]
pub struct Event(hipEvent_t);

// SAFETY: as for `Stream` — an opaque handle usable from any host thread.
unsafe impl Send for Event {}

impl Event {
    pub fn new() -> Result<Self> {
        let mut event: hipEvent_t = std::ptr::null_mut();
        // SAFETY: `event` is a live out-parameter.
        check("hipEventCreate", unsafe { hipEventCreate(&mut event) })?;
        Ok(Self(event))
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        // SAFETY: both handles are owned and alive.
        check("hipEventRecord", unsafe {
            hipEventRecord(self.0, stream.raw())
        })
    }

    /// Milliseconds between two recorded events. Both must have been recorded and the
    /// stream synchronised, or HIP reports `hipErrorNotReady`.
    pub fn elapsed_ms(start: &Event, end: &Event) -> Result<f32> {
        let mut ms = 0.0f32;
        // SAFETY: `ms` is a live out-parameter; both handles are owned and alive.
        check("hipEventElapsedTime", unsafe {
            hipEventElapsedTime(&mut ms, start.0, end.0)
        })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: valid handle, destroyed exactly once.
            unsafe { hipEventDestroy(self.0) };
        }
    }
}

/// Bindings to the two kernel entry points in `kernel/argon2_kernel.hip`.
#[cfg(not(tm_gpu_stub))]
mod kernel {
    use super::{hipError_t, hipStream_t};
    use std::ffi::{c_uint, c_void};

    extern "C" {
        pub fn tm_launch_argon2_oneshot(
            stream: hipStream_t,
            memory: *mut c_void,
            segment_blocks: c_uint,
            batch_size: usize,
        ) -> hipError_t;

        #[allow(clippy::too_many_arguments)]
        pub fn tm_launch_argon2_first_blocks(
            stream: hipStream_t,
            memory: *mut c_void,
            keys: *const c_void,
            key_length: c_uint,
            salt: *const c_void,
            salt_length: c_uint,
            output_length: c_uint,
            memory_cost: c_uint,
            time_cost: c_uint,
            version: c_uint,
            type_: c_uint,
            lanes: c_uint,
            segment_blocks: c_uint,
            batch_size: usize,
        ) -> hipError_t;
    }
}

/// Launches the one-shot Argon2 kernel.
///
/// # Safety
/// `memory` must point to at least `batch_size * segment_blocks * 4 * 1024` device bytes
/// whose first two blocks per job are already filled, and must stay alive until `stream`
/// is synchronised.
pub unsafe fn launch_oneshot(
    stream: &Stream,
    memory: *mut c_void,
    segment_blocks: u32,
    batch_size: usize,
) -> Result<()> {
    #[cfg(tm_gpu_stub)]
    {
        let _ = (stream, memory, segment_blocks, batch_size);
        Err(GpuError::NoKernel)
    }
    #[cfg(not(tm_gpu_stub))]
    check(
        "argon2_kernel_oneshot",
        kernel::tm_launch_argon2_oneshot(stream.raw(), memory, segment_blocks, batch_size),
    )
}

/// Launches the device-side first-blocks kernel.
///
/// # Safety
/// `memory` must be the pool described above, `keys` must hold `batch_size * key_length`
/// bytes and `salt` `salt_length` bytes, all on the device and alive until `stream` is
/// synchronised.
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
    #[cfg(tm_gpu_stub)]
    {
        let _ = (
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
        );
        Err(GpuError::NoKernel)
    }
    #[cfg(not(tm_gpu_stub))]
    check(
        "argon2_first_blocks_kernel",
        kernel::tm_launch_argon2_first_blocks(
            stream.raw(),
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
        ),
    )
}
