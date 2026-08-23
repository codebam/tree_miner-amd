//! The GPU layer of TreeMiner: device management, the Argon2 batch pool, and the hash
//! backend the mining loop drives.
//!
//! # Vendor support
//!
//! | vendor | feature | status |
//! | --- | --- | --- |
//! | AMD (HIP/ROCm) | `amd`, the default | **tested** on gfx1100: 44/44 fixture vectors byte-exact, raw-block differential against the hipcc kernels green |
//! | NVIDIA (CUDA/PTX) | `nvidia` | **compiles; never executed.** No NVIDIA GPU has ever run it |
//!
//! The two features are mutually exclusive — they link different device runtimes — so the
//! NVIDIA build is `--no-default-features --features nvidia`. Everything above the driver
//! layer is shared: `runner.rs`, `device.rs`, `hash.rs` and `backend.rs` talk to
//! [`driver`], which is [`hip`] or [`cuda`] depending on the feature, and the Argon2
//! kernels themselves are one Rust source (`../tm-kernel`) built for two targets.
//!
//! The NVIDIA path is honest about what it is: it is compile-verified, its PTX has been
//! read, and its argument packing is unit-tested against the kernel signature the PTX
//! declares — but no digest has ever come out of it. The first person with an NVIDIA card
//! should run `tests/parity/run_parity.sh` and the fixture suite before trusting a
//! submission from it. See PORT.md.
//!
//! Port of `src/kernelrunner.cu` (host half), `src/CudaBackend.cpp`, `src/CudaDevice.cpp`,
//! `src/ComputeBackend.h`, `src/gpu/GpuTelemetry.cpp` and `src/hashapi/CudaHashBackend.cpp`.
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

#[cfg(all(feature = "amd", feature = "nvidia"))]
compile_error!(
    "tm-gpu's `amd` and `nvidia` features are mutually exclusive: they link different device \
     runtimes. Use `--no-default-features --features nvidia` for the (untested) NVIDIA path."
);
#[cfg(not(any(feature = "amd", feature = "nvidia")))]
compile_error!(
    "tm-gpu needs a vendor feature: `amd` (the default, tested) or `nvidia` (compile-verified \
     only, never run on hardware)."
);

pub mod backend;
#[cfg(feature = "nvidia")]
mod cuda;
pub mod device;
pub mod error;
#[cfg(feature = "amd")]
mod hip;
pub mod hash;
#[cfg(all(feature = "amd", feature = "rust-kernel"))]
mod module;

/// The device runtime everything above this layer talks to.
///
/// `hip` and `cuda` deliberately expose the same names and signatures — `DeviceBuffer`,
/// `Stream`, `Event`, `memcpy_2d_async`, `launch_oneshot`, `launch_first_blocks` — so this
/// alias is the whole of the vendor abstraction on the host side. A call that exists on only
/// one of them (`launch_oneshot_hip`, the differential oracle) is reached through a cfg at
/// the call site, not smuggled in here.
#[cfg(feature = "amd")]
use hip as driver;
#[cfg(feature = "nvidia")]
use cuda as driver;
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
