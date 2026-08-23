//! The GPU layer of TreeMiner: HIP device management, the Argon2 batch pool, and the hash
//! backend the mining loop drives.
//!
//! Port of `src/kernelrunner.cu` (host half), `src/CudaBackend.cpp`, `src/CudaDevice.cpp`,
//! `src/ComputeBackend.h`, `src/gpu/GpuTelemetry.cpp` and `src/hashapi/CudaHashBackend.cpp`.
//! The Argon2 *device* kernels stay in C++ (`kernel/argon2_kernel.hip`, compiled by
//! `build.rs`) and are reached through a two-function `extern "C"` shim — see `PORT.md` for
//! why rewriting them is a separate, later phase.
//!
//! This is the only crate in the workspace allowed to use `unsafe`.
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use tm_gpu::{BatchRequest, Device, GpuBackend, GpuHashBackend};
//! # fn host() -> &'static dyn tm_gpu::Argon2Host { unimplemented!() }
//! let device = Device::enumerate()?.into_iter().next().ok_or("no GPU")?;
//! let mut backend = GpuHashBackend::new(GpuBackend::new(device));
//! let passwords = vec!["52a1...".to_owned()];
//! let request = BatchRequest::new(&passwords, "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc", 8);
//! let outcome = backend.run_batch(&request, host())?;
//! println!("{:?}", outcome.hash);
//! # Ok(())
//! # }
//! ```

pub mod backend;
pub mod device;
pub mod error;
mod hip;
pub mod hash;
#[cfg(feature = "rust-kernel")]
mod module;
pub mod params;
pub mod runner;
pub mod telemetry;

pub use backend::GpuBackend;
pub use device::Device;
pub use error::{GpuError, Result};
pub use hash::{
    Argon2Host, BatchOutcome, BatchRequest, BatchTimings, GpuHashBackend, GpuMatch, HostError,
};
pub use params::{Argon2Shape, ARGON2_BLOCK_SIZE, ARGON2_ID, ARGON2_VERSION_13, DEFAULT_HASH_LENGTH};
pub use runner::{KernelRunner, RunTimings};
pub use telemetry::{DeviceTelemetry, TelemetrySession};

/// True when at least one GPU is present. Used by the tests to skip rather than fail on a
/// machine without a device, and by the miner to decide whether to offer GPU mining.
pub fn gpu_available() -> bool {
    matches!(device::Device::enumerate(), Ok(devices) if !devices.is_empty())
}
