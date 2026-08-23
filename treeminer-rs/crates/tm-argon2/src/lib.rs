//! Argon2id hashing for TreeMiner: the request/result contract, its validation, random key
//! generation, and the CPU backend that the GPU path is checked against.
//!
//! Ports `src/Argon2idHasher.cpp`, `src/hashapi/CpuHashBackend.cpp`,
//! `src/hashapi/HashApiValidation.cpp`, `src/hashapi/HashApiTypes.h` and
//! `src/RandomHexKeyGenerator.h`.

pub mod cpu;
pub mod host;
pub mod keygen;
pub mod types;
pub mod validation;

pub use cpu::{append_matches, argon2id_phc, run_batch, CpuHashBackend, HashError};
pub use host::{
    digest_long, first_block_chunk_size, first_block_selected_chunk_size,
    first_block_worker_count, initial_hash, recommended_first_block_dynamic_chunk_size,
    CpuArgon2Host, MIN_PARALLEL_FIRST_BLOCK_ATTEMPTS,
};
pub use keygen::RandomHexKeyGenerator;
pub use types::{
    HashBackend, HashMatch, HashRequest, HashResult, HashTimings, DEFAULT_ALGORITHM,
    DEFAULT_HASH_LENGTH, DEFAULT_TARGET_PATTERN, HASH_API_KEY_LENGTH, MAX_CPU_BATCH_SIZE,
    MAX_TARGET_PATTERN_LENGTH, MIN_ARGON2_CPU_DIFFICULTY,
};
pub use validation::{is_valid_request, normalize_hex, validate_request, ValidationErrors};
