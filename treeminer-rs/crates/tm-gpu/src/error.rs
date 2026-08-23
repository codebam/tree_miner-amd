//! Errors from the GPU layer. Every device-runtime call is checked; a failure becomes a
//! [`GpuError::Hip`] or [`GpuError::Cuda`] carrying the driver's own message, which is what
//! the C++ miner's `CudaException` did.
//!
//! Both variants exist in every build even though only one vendor is compiled in: which one
//! the caller sees is a build-time choice, but matching on the enum should not be.

use std::fmt;

pub type Result<T> = std::result::Result<T, GpuError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuError {
    /// A HIP runtime call failed. `message` is `hipGetErrorString`'s text.
    Hip {
        call: &'static str,
        code: i32,
        message: String,
    },
    /// A CUDA driver call failed. `message` is `cuGetErrorString`'s text.
    ///
    /// Only produced by the `nvidia` feature, which has never run on real hardware.
    Cuda {
        call: &'static str,
        code: i32,
        message: String,
    },
    /// The crate was built without a HIP toolchain, so no kernel exists to launch.
    NoKernel,
    /// The request cannot be served as stated (bad batch shape, bad salt, ...).
    Invalid(String),
    /// The device index does not exist.
    NoSuchDevice(i32),
    /// The injected CPU Argon2 helper failed (first blocks or digest finalisation).
    Host(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::Hip {
                call,
                code,
                message,
            } => write!(formatter, "{call} failed: {message} (hipError {code})"),
            GpuError::Cuda {
                call,
                code,
                message,
            } => write!(formatter, "{call} failed: {message} (CUresult {code})"),
            GpuError::NoKernel => formatter.write_str(
                "tm-gpu was built without hipcc; no Argon2 device kernel is available",
            ),
            GpuError::Invalid(reason) => write!(formatter, "invalid GPU request: {reason}"),
            GpuError::NoSuchDevice(index) => write!(formatter, "no GPU with index {index}"),
            GpuError::Host(message) => write!(formatter, "CPU Argon2 helper failed: {message}"),
        }
    }
}

impl std::error::Error for GpuError {}
