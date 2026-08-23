//! One GPU as the mining loop sees it: a device plus the pool currently allocated on it.
//! Port of `src/CudaBackend.cpp` / `src/ComputeBackend.h`.

use tm_core::batch::{select_batch_size, BatchSizeDecision, GpuRuntimeKind};

use crate::device::Device;
use crate::error::{GpuError, Result};
use crate::params::Argon2Shape;
use crate::runner::{KernelRunner, RunTimings};

/// A GPU backend. The pool is created lazily by [`GpuBackend::init`] and reused across
/// batches whenever the shape allows.
#[derive(Debug)]
pub struct GpuBackend {
    device: Device,
    runner: Option<KernelRunner>,
}

impl GpuBackend {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            runner: None,
        }
    }

    /// Every GPU the runtime reports, each wrapped in an idle backend.
    pub fn enumerate() -> Result<Vec<Self>> {
        Ok(Device::enumerate()?.into_iter().map(Self::new).collect())
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn activate(&self) -> Result<()> {
        self.device.activate()
    }

    pub fn free_memory_bytes(&self) -> Result<usize> {
        self.device.free_memory_bytes()
    }

    /// Allocates or reuses the pool for `batch_size` jobs of `shape`.
    pub fn init(&mut self, shape: &Argon2Shape, batch_size: usize) -> Result<()> {
        if let Some(runner) = self.runner.as_mut() {
            if runner.can_reuse(shape, batch_size) {
                runner.reconfigure(shape, batch_size);
                return Ok(());
            }
        }
        // Drop the old pool before allocating the new one, or the two coexist and the
        // allocation fails on a device that was nearly full.
        self.runner = None;
        self.runner = Some(KernelRunner::new(*shape, batch_size)?);
        Ok(())
    }

    /// Frees the pool now.
    ///
    /// Difficulty changes must call this before re-measuring free memory: a retained pool
    /// sized for the previous difficulty is counted as used VRAM and starves the batch
    /// estimate, which is how the C++ miner used to shrink its batch on every difficulty
    /// bump until it was mining a handful of hashes per batch.
    pub fn release_buffers(&mut self) {
        self.runner = None;
    }

    /// Frees the pool, measures free VRAM, and asks `tm_core` for the batch size.
    ///
    /// The release is the point: see [`Self::release_buffers`].
    pub fn plan_batch_size(
        &mut self,
        difficulty: u32,
        explicit_max_batch_size: usize,
    ) -> Result<BatchSizeDecision> {
        self.release_buffers();
        let free = self.free_memory_bytes()?;
        Ok(select_batch_size(
            GpuRuntimeKind::Hip,
            free,
            difficulty,
            explicit_max_batch_size,
        ))
    }

    pub fn runner(&self) -> Option<&KernelRunner> {
        self.runner.as_ref()
    }

    pub fn runner_mut(&mut self) -> Result<&mut KernelRunner> {
        self.runner
            .as_mut()
            .ok_or_else(|| GpuError::Invalid("GPU pool is not initialised".to_owned()))
    }

    /// Timings of the last finished batch; all zero when nothing has run.
    pub fn timings(&self) -> RunTimings {
        self.runner
            .as_ref()
            .map_or_else(RunTimings::default, KernelRunner::timings)
    }
}
