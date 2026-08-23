//! Shared TreeMiner contract types and the pure logic every other crate agrees on.
//!
//! This crate is the Rust side of `src/treeminer/Types.h` plus the parts of the C++ miner
//! that are pure functions of their inputs: PHC assembly, find-detection rules, address
//! validation, and GPU batch sizing. No I/O, no GPU, no network — everything here is unit
//! testable and is what the other crates are written against.

pub mod address;
pub mod argon2host;
pub mod batch;
pub mod encoding;
pub mod matching;
pub mod types;

pub use argon2host::{
    Argon2Host, Argon2Shape, HostError, ARGON2_BLOCK_SIZE, ARGON2_ID, ARGON2_SYNC_POINTS,
    ARGON2_VERSION_13, DEFAULT_HASH_LENGTH, INPUT_BLOCKS_PER_JOB,
};
pub use address::{is_valid_ethereum_address, salt_hex_for_address, to_checksum_address};
pub use batch::{select_batch_size, BatchSizeDecision, GpuRuntimeKind};
pub use encoding::{assemble_phc, base64_encode, hex_to_bytes, phc_digest};
pub use matching::{has_xuni_match, is_superblock_hash, is_within_xuni_window_at};
pub use types::{Classification, FindKind, FindRecord, FindStatus, FoundPayload};
