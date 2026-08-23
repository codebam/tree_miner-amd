//! The Argon2 shape a batch is run with.
//!
//! These are host-side Argon2 concepts with no device state, so they live in `tm-core`
//! (a CPU-only build needs them without linking HIP) and are re-exported here for the
//! callers that reach for them through the GPU crate.

pub use tm_core::argon2host::{
    Argon2Shape, ARGON2_BLOCK_SIZE, ARGON2_ID, ARGON2_SYNC_POINTS, ARGON2_VERSION_13,
    DEFAULT_HASH_LENGTH, INPUT_BLOCKS_PER_JOB,
};
