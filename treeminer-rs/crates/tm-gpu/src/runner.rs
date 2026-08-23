//! The device-side batch pool and its launch sequence. Port of the host half of
//! `src/kernelrunner.cu` (class `KernelRunner`).

use std::ffi::c_void;

use crate::error::{GpuError, Result};
use crate::hip::{
    self, DeviceBuffer, Event, Stream, HIP_MEMCPY_DEVICE_TO_HOST, HIP_MEMCPY_HOST_TO_DEVICE,
};
use crate::params::{Argon2Shape, ARGON2_BLOCK_SIZE, DEFAULT_HASH_LENGTH};

/// Two Argon2 blocks per job go in, one comes out.
const INPUT_BLOCKS_PER_JOB: usize = 2;

/// The parameters a queued device-first-blocks launch was prepared with.
#[derive(Debug, Clone, Copy)]
struct FirstBlockConfig {
    key_length: u32,
    salt_length: u32,
    shape: Argon2Shape,
}

/// The timings the dashboard reports, refreshed by every [`KernelRunner::finish`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RunTimings {
    /// Argon2 kernel only.
    pub kernel_ms: f32,
    /// Staging the first blocks: the host->device copy, or zero on the GPU path.
    pub host_to_device_ms: f32,
    /// The device-side first-blocks kernel; zero when the CPU produced them.
    pub gpu_first_block_ms: f32,
    /// Reading the final block of every job back.
    pub device_to_host_ms: f32,
    /// Everything between the first and last event.
    pub total_ms: f32,
}

/// Owns one batch-sized device pool plus the stream and events that drive it.
///
/// Field order is load-bearing: `stream` is declared first so it is dropped — and therefore
/// synchronised — before the buffers whose memory its queued work still references.
#[derive(Debug)]
pub struct KernelRunner {
    stream: Stream,
    start: Event,
    end: Event,
    copy_start: Event,
    copy_end: Event,
    first_block_start: Event,
    first_block_end: Event,
    kernel_start: Event,
    kernel_end: Event,

    memory: DeviceBuffer,
    device_keys: Option<DeviceBuffer>,
    device_salt: Option<DeviceBuffer>,

    shape: Argon2Shape,
    /// The segment size the pool was sized for; a reconfigure may only shrink below it.
    allocated_segment_blocks: u32,
    batch_size: usize,

    blocks_in: Vec<u8>,
    blocks_out: Vec<u8>,

    /// Set by `prepare_input_blocks_on_device`, consumed by the next `run`.
    pending_first_blocks: Option<FirstBlockConfig>,
    last_used_device_first_blocks: bool,
    timings: RunTimings,
}

impl KernelRunner {
    /// Allocates the pool. The device must already be active on this thread.
    pub fn new(shape: Argon2Shape, batch_size: usize) -> Result<Self> {
        if batch_size == 0 {
            return Err(GpuError::Invalid("batch size must be non-zero".to_owned()));
        }
        let segment_blocks = shape.segment_blocks();
        let memory_size = batch_size
            .checked_mul(shape.job_bytes())
            .ok_or_else(|| GpuError::Invalid("batch size overflows the device pool".to_owned()))?;

        Ok(Self {
            stream: Stream::new()?,
            start: Event::new()?,
            end: Event::new()?,
            copy_start: Event::new()?,
            copy_end: Event::new()?,
            first_block_start: Event::new()?,
            first_block_end: Event::new()?,
            kernel_start: Event::new()?,
            kernel_end: Event::new()?,
            memory: DeviceBuffer::new(memory_size)?,
            device_keys: None,
            device_salt: None,
            shape,
            allocated_segment_blocks: segment_blocks,
            batch_size,
            blocks_in: vec![0u8; batch_size * INPUT_BLOCKS_PER_JOB * ARGON2_BLOCK_SIZE],
            blocks_out: vec![0u8; batch_size * ARGON2_BLOCK_SIZE],
            pending_first_blocks: None,
            last_used_device_first_blocks: false,
            timings: RunTimings::default(),
        })
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn shape(&self) -> Argon2Shape {
        self.shape
    }

    /// True when this pool can serve the requested shape without reallocating: same
    /// algorithm and batch, and a segment size that fits inside what was allocated.
    pub fn can_reuse(&self, shape: &Argon2Shape, batch_size: usize) -> bool {
        self.shape.type_ == shape.type_
            && self.shape.version == shape.version
            && self.shape.passes == shape.passes
            && self.shape.lanes == shape.lanes
            && self.batch_size == batch_size
            && shape.segment_blocks() <= self.allocated_segment_blocks
    }

    /// Retargets an existing pool at a smaller shape. Any prepared first blocks are
    /// discarded: they were staged for the old parameters.
    pub fn reconfigure(&mut self, shape: &Argon2Shape, batch_size: usize) {
        self.shape = *shape;
        self.batch_size = batch_size;
        self.pending_first_blocks = None;
        self.last_used_device_first_blocks = false;
    }

    /// The 2 KiB staging slot for one job's first blocks, for the CPU path to fill.
    pub fn input_blocks_mut(&mut self, job_id: usize) -> Result<&mut [u8]> {
        let size = INPUT_BLOCKS_PER_JOB * ARGON2_BLOCK_SIZE;
        let start = job_id * size;
        self.blocks_in
            .get_mut(start..start + size)
            .ok_or_else(|| GpuError::Invalid(format!("job {job_id} is outside the batch")))
    }

    /// The staging slots of the first `jobs` jobs as one contiguous slice, so a host can
    /// fill the whole batch in parallel instead of one job at a time.
    pub fn input_blocks_batch_mut(&mut self, jobs: usize) -> Result<&mut [u8]> {
        let size = INPUT_BLOCKS_PER_JOB * ARGON2_BLOCK_SIZE;
        self.blocks_in
            .get_mut(..jobs * size)
            .ok_or_else(|| GpuError::Invalid(format!("batch of {jobs} is outside the pool")))
    }

    /// The final Argon2 block of one job, ready to be reduced to a digest.
    pub fn output_block(&self, job_id: usize) -> Result<&[u8]> {
        let start = job_id * ARGON2_BLOCK_SIZE;
        self.blocks_out
            .get(start..start + ARGON2_BLOCK_SIZE)
            .ok_or_else(|| GpuError::Invalid(format!("job {job_id} is outside the batch")))
    }

    /// Stages passwords and salt on the device so the next `run` derives the first blocks
    /// there instead of copying them up.
    ///
    /// Returns `Ok(false)` — never an error — when the request has a shape the kernel does
    /// not implement, so the caller can fall back to the CPU path exactly as the C++ miner
    /// does.
    pub fn prepare_input_blocks_on_device(
        &mut self,
        passwords: &[String],
        salt: &[u8],
        shape: &Argon2Shape,
    ) -> Result<bool> {
        self.pending_first_blocks = None;
        if passwords.is_empty()
            || passwords.len() != self.batch_size
            || shape.output_length as usize != DEFAULT_HASH_LENGTH
            || shape.passes != 1
            || shape.lanes != 1
            || salt.is_empty()
            || u32::try_from(salt.len()).is_err()
        {
            return Ok(false);
        }

        let key_length = passwords[0].len();
        if key_length == 0 || u32::try_from(key_length).is_err() {
            return Ok(false);
        }
        if passwords.iter().any(|password| password.len() != key_length) {
            return Ok(false);
        }

        let mut flat_keys = Vec::with_capacity(key_length * passwords.len());
        for password in passwords {
            flat_keys.extend_from_slice(password.as_bytes());
        }

        // Grow-only, like the C++ runner: a shrinking batch keeps the larger allocation.
        if self
            .device_keys
            .as_ref()
            .is_none_or(|buffer| buffer.len() < flat_keys.len())
        {
            self.device_keys = None;
            self.device_keys = Some(DeviceBuffer::new(flat_keys.len())?);
        }
        if self
            .device_salt
            .as_ref()
            .is_none_or(|buffer| buffer.len() < salt.len())
        {
            self.device_salt = None;
            self.device_salt = Some(DeviceBuffer::new(salt.len())?);
        }

        let keys = self
            .device_keys
            .as_mut()
            .ok_or_else(|| GpuError::Invalid("key buffer missing".to_owned()))?;
        keys.copy_from_host(&flat_keys)?;
        let salt_buffer = self
            .device_salt
            .as_mut()
            .ok_or_else(|| GpuError::Invalid("salt buffer missing".to_owned()))?;
        salt_buffer.copy_from_host(salt)?;

        self.pending_first_blocks = Some(FirstBlockConfig {
            key_length: key_length as u32,
            salt_length: salt.len() as u32,
            shape: *shape,
        });
        Ok(true)
    }

    /// Bytes one job occupies in the pool at the *current* shape. A reconfigure to a
    /// smaller difficulty makes this smaller than the allocated stride, which is exactly
    /// how the reuse path packs jobs.
    fn job_bytes(&self) -> usize {
        self.shape.job_bytes()
    }

    fn copy_input_blocks(&self) -> Result<()> {
        let copy_size = INPUT_BLOCKS_PER_JOB * ARGON2_BLOCK_SIZE;
        // SAFETY: `memory` holds `batch_size * allocated job_bytes` device bytes and
        // `job_bytes()` never exceeds that stride, so `batch_size` rows of `copy_size`
        // bytes fit. `blocks_in` is `batch_size * copy_size` long and is owned by `self`,
        // which cannot be mutated again before `finish` synchronises the stream.
        unsafe {
            hip::memcpy_2d_async(
                self.memory.as_ptr(),
                self.job_bytes(),
                self.blocks_in.as_ptr().cast::<c_void>(),
                copy_size,
                copy_size,
                self.batch_size,
                HIP_MEMCPY_HOST_TO_DEVICE,
                self.stream.raw(),
            )
        }
    }

    fn copy_output_blocks(&mut self) -> Result<()> {
        let job_bytes = self.job_bytes();
        let copy_size = self.shape.lanes as usize * ARGON2_BLOCK_SIZE;
        let batch_size = self.batch_size;
        let stream = self.stream.raw();
        // SAFETY: the source is the last `copy_size` bytes of each job's stride, which lies
        // inside the pool because `job_bytes <= allocated stride`. `blocks_out` is
        // `batch_size * ARGON2_BLOCK_SIZE` long and is not read until `finish` has
        // synchronised the stream.
        unsafe {
            let source = self.memory.as_ptr().cast::<u8>().add(job_bytes - copy_size);
            hip::memcpy_2d_async(
                self.blocks_out.as_mut_ptr().cast::<c_void>(),
                copy_size,
                source.cast::<c_void>(),
                job_bytes,
                copy_size,
                batch_size,
                HIP_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        }
    }

    /// Queues one batch: first blocks (host copy or device kernel), the Argon2 kernel, and
    /// the readback. Nothing has happened yet when this returns — call [`Self::finish`].
    pub fn run(&mut self) -> Result<()> {
        self.start.record(&self.stream)?;
        self.copy_start.record(&self.stream)?;

        let pending = self.pending_first_blocks.take();
        self.last_used_device_first_blocks = pending.is_some();
        match pending {
            Some(config) => {
                self.copy_end.record(&self.stream)?;
                self.first_block_start.record(&self.stream)?;
                self.launch_first_blocks(&config)?;
                self.first_block_end.record(&self.stream)?;
            }
            None => {
                self.copy_input_blocks()?;
                self.copy_end.record(&self.stream)?;
                // The GPU first-block window is empty on this path, but both events still
                // have to be recorded or the elapsed-time query fails.
                self.first_block_start.record(&self.stream)?;
                self.first_block_end.record(&self.stream)?;
            }
        }

        self.kernel_start.record(&self.stream)?;
        // SAFETY: the pool holds `batch_size` jobs of `segment_blocks * 4` blocks each and
        // outlives the stream (field order), and every job's first two blocks were just
        // filled by one of the two branches above.
        unsafe {
            hip::launch_oneshot(
                &self.stream,
                self.memory.as_ptr(),
                self.shape.segment_blocks(),
                self.batch_size,
            )?;
        }
        self.kernel_end.record(&self.stream)?;

        self.copy_output_blocks()?;
        self.end.record(&self.stream)
    }

    fn launch_first_blocks(&self, config: &FirstBlockConfig) -> Result<()> {
        let keys = self
            .device_keys
            .as_ref()
            .ok_or_else(|| GpuError::Invalid("first blocks prepared without keys".to_owned()))?;
        let salt = self
            .device_salt
            .as_ref()
            .ok_or_else(|| GpuError::Invalid("first blocks prepared without a salt".to_owned()))?;
        // SAFETY: `keys` holds batch_size * key_length bytes and `salt` salt_length bytes,
        // both checked when they were staged; all three buffers outlive the stream.
        unsafe {
            hip::launch_first_blocks(
                &self.stream,
                self.memory.as_ptr(),
                keys.as_ptr(),
                config.key_length,
                salt.as_ptr(),
                config.salt_length,
                config.shape.output_length,
                config.shape.memory_cost,
                config.shape.passes,
                config.shape.version,
                config.shape.type_,
                config.shape.lanes,
                self.shape.segment_blocks(),
                self.batch_size,
            )
        }
    }

    /// Waits for the queued batch and returns the Argon2 kernel time in milliseconds.
    /// After this returns, [`Self::output_block`] is safe to read.
    pub fn finish(&mut self) -> Result<f32> {
        self.stream.synchronize()?;
        self.timings = RunTimings {
            kernel_ms: Event::elapsed_ms(&self.kernel_start, &self.kernel_end)?,
            host_to_device_ms: Event::elapsed_ms(&self.copy_start, &self.copy_end)?,
            gpu_first_block_ms: if self.last_used_device_first_blocks {
                Event::elapsed_ms(&self.first_block_start, &self.first_block_end)?
            } else {
                0.0
            },
            device_to_host_ms: Event::elapsed_ms(&self.kernel_end, &self.end)?,
            total_ms: Event::elapsed_ms(&self.start, &self.end)?,
        };
        Ok(self.timings.kernel_ms)
    }

    /// Timings from the most recent [`Self::finish`].
    pub fn timings(&self) -> RunTimings {
        self.timings
    }
}
