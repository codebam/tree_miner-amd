//! The mining loop's view of one hashing device, and the real GPU behind it.
//!
//! The loop is written against [`MiningBackend`] rather than `tm_gpu::GpuHashBackend`
//! directly, for the same reason the submitter is written against a journal trait: the
//! behaviours that cost money — building the payload from the batch's own parameters,
//! releasing buffers before re-measuring free VRAM, never dropping a XUNI at a window
//! boundary — have to be testable on a machine whose GPU is busy mining.

use std::sync::Arc;

use tm_core::batch::{select_batch_size, BatchSizeDecision, GpuRuntimeKind};
use tm_gpu::{
    Argon2Host, BatchOutcome, BatchRequest, Device, GpuBackend, GpuHashBackend,
};

use crate::selftest::{ProbeOutcome, SelfTestProbe, SELF_TEST_DIFFICULTY, SELF_TEST_KEY, SELF_TEST_PATTERN, SELF_TEST_SALT};

/// Which runtime `tm_core` should size batches for; follows the vendor `tm-gpu` was built
/// with. ROCm needs a much larger VRAM cushion than CUDA — see `tm_core::batch`.
#[cfg(feature = "amd")]
const RUNTIME: GpuRuntimeKind = GpuRuntimeKind::Hip;
#[cfg(feature = "nvidia")]
const RUNTIME: GpuRuntimeKind = GpuRuntimeKind::Cuda;

/// Headroom left on a device whose memory is shared between streams, so a second stream's
/// allocation does not have to come out of the first one's slack. Port of the
/// `kDeviceHeadroom` constant in `MineUnit::runMineLoop`.
const DEVICE_HEADROOM_BYTES: usize = 512 * 1024 * 1024;

/// What the stats line needs to know about a device, captured once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceFacts {
    pub index: i32,
    pub name: String,
    pub bus_id: i32,
    pub total_memory_bytes: usize,
}

/// One device the mining loop can drive.
pub trait MiningBackend: Send {
    fn device(&self) -> DeviceFacts;

    /// Free the batch pool now.
    fn release_buffers(&mut self);

    /// Size the next batch. Implementations MUST release the current pool before measuring
    /// free memory: a pool sized for the previous difficulty otherwise counts as used VRAM
    /// and starves the estimate, which is how the C++ miner used to shrink its batch on
    /// every difficulty bump until it was hashing a handful of jobs per batch.
    fn plan_batch_size(
        &mut self,
        difficulty: u32,
        explicit_max_batch_size: usize,
        streams_per_device: usize,
    ) -> Result<BatchSizeDecision, String>;

    fn run_batch(&mut self, request: &BatchRequest<'_>) -> Result<BatchOutcome, String>;
}

/// The real GPU: a `tm-gpu` backend plus the CPU Argon2 half it needs for first blocks and
/// digest finalisation.
pub struct GpuMiningBackend {
    backend: GpuHashBackend,
    host: Arc<dyn Argon2Host>,
    facts: DeviceFacts,
}

impl std::fmt::Debug for GpuMiningBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuMiningBackend")
            .field("device", &self.facts)
            .finish()
    }
}

impl GpuMiningBackend {
    pub fn new(device: Device, host: Arc<dyn Argon2Host>) -> Self {
        let facts = DeviceFacts {
            index: device.index(),
            name: device.name().to_owned(),
            bus_id: device.bus_id(),
            total_memory_bytes: device.total_memory_bytes(),
        };
        Self {
            backend: GpuHashBackend::new(GpuBackend::new(device)),
            host,
            facts,
        }
    }

    /// Open device `index` and wrap it.
    pub fn open(index: i32, host: Arc<dyn Argon2Host>) -> Result<Self, String> {
        Device::open(index)
            .map(|device| Self::new(device, host))
            .map_err(|error| error.to_string())
    }
}

impl MiningBackend for GpuMiningBackend {
    fn device(&self) -> DeviceFacts {
        self.facts.clone()
    }

    fn release_buffers(&mut self) {
        self.backend.backend_mut().release_buffers();
    }

    fn plan_batch_size(
        &mut self,
        difficulty: u32,
        explicit_max_batch_size: usize,
        streams_per_device: usize,
    ) -> Result<BatchSizeDecision, String> {
        let gpu = self.backend.backend_mut();
        if streams_per_device <= 1 {
            // Releases the pool, re-measures, and asks tm-core — all of it in one place.
            return gpu
                .plan_batch_size(difficulty, explicit_max_batch_size)
                .map_err(|error| error.to_string());
        }

        // Multi-stream: the same release-before-measure rule, but each stream may only
        // count on its share of the device. Free memory alone would let two streams each
        // size a batch for the whole card and then fail to allocate the second one.
        gpu.release_buffers();
        let free = gpu.free_memory_bytes().map_err(|error| error.to_string())?;
        let shareable = self
            .facts
            .total_memory_bytes
            .saturating_sub(DEVICE_HEADROOM_BYTES);
        let share = shareable / streams_per_device;
        Ok(select_batch_size(
            RUNTIME,
            free.min(share),
            difficulty,
            explicit_max_batch_size,
        ))
    }

    fn run_batch(&mut self, request: &BatchRequest<'_>) -> Result<BatchOutcome, String> {
        self.backend
            .run_batch(request, self.host.as_ref())
            .map_err(|error| error.to_string())
    }
}

/// The startup self-test's real probe: the CPU reference from `tm-argon2`, each device
/// digest from a one-job GPU batch.
pub struct GpuSelfTestProbe {
    host: Arc<dyn Argon2Host>,
}

impl GpuSelfTestProbe {
    pub fn new(host: Arc<dyn Argon2Host>) -> Self {
        Self { host }
    }
}

impl SelfTestProbe for GpuSelfTestProbe {
    fn cpu_reference(&mut self) -> ProbeOutcome {
        crate::selftest::cpu_reference_digest()
    }

    fn gpu_digest(&mut self, device_index: i32, gpu_first_blocks: bool) -> ProbeOutcome {
        // A fresh backend per probe: the device is opened, exercised and released, so a
        // device that cannot even be opened fails here rather than during mining.
        let mut backend = GpuMiningBackend::open(device_index, Arc::clone(&self.host))?;
        let passwords = vec![SELF_TEST_KEY.to_owned()];
        let mut request = BatchRequest::new(&passwords, SELF_TEST_SALT, SELF_TEST_DIFFICULTY);
        request.target_pattern = SELF_TEST_PATTERN;
        request.allow_xuni = false;
        request.gpu_first_blocks = gpu_first_blocks;
        let outcome = backend.run_batch(&request)?;
        outcome
            .hash
            .ok_or_else(|| "the batch produced no digest".to_owned())
    }
}
