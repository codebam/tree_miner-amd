//! `CpuArgon2Host`: the threaded batch path against the single-threaded one, and the
//! worker/chunk arithmetic against the C++ constants it was ported from.

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;
use tm_argon2::{
    first_block_chunk_size, first_block_selected_chunk_size, first_block_worker_count,
    recommended_first_block_dynamic_chunk_size, CpuArgon2Host,
};
use tm_core::argon2host::{Argon2Host, Argon2Shape, ARGON2_BLOCK_SIZE};

const SALT: &[u8] = &[
    0xe4, 0xbb, 0x18, 0x47, 0x81, 0xbb, 0xc9, 0xc7, 0x00, 0x4e, 0x8d, 0xaf, 0xd4, 0xa9, 0xb4,
    0x9d, 0x20, 0x3b, 0xc9, 0xbc,
];

fn passwords(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| format!("{index:064x}"))
        .collect()
}

fn fill(host: &CpuArgon2Host, keys: &[String], shape: &Argon2Shape) -> Vec<u8> {
    let mut out = vec![0u8; keys.len() * 2 * ARGON2_BLOCK_SIZE];
    host.fill_first_blocks_batch(&mut out, keys, SALT, shape)
        .expect("batch fills");
    out
}

#[test]
fn the_parallel_path_reproduces_the_single_threaded_one() {
    let shape = Argon2Shape::for_difficulty(8);
    // 7 stays under the parallel threshold, 64 is comfortably over it.
    for count in [1, 7, 8, 9, 33, 64] {
        let keys = passwords(count);
        let expected = fill(&CpuArgon2Host::single_threaded(), &keys, &shape);
        assert_eq!(
            fill(&CpuArgon2Host::new(), &keys, &shape),
            expected,
            "auto worker count, {count} jobs"
        );
        for workers in [2, 3, 8] {
            assert_eq!(
                fill(&CpuArgon2Host::new().with_workers(workers), &keys, &shape),
                expected,
                "{workers} workers, {count} jobs"
            );
        }
    }
}

/// Stolen chunks reorder which thread does which job; the buffer must come out identical.
#[test]
fn dynamic_chunking_reproduces_the_even_split() {
    let shape = Argon2Shape::for_difficulty(8);
    let keys = passwords(100);
    let expected = fill(&CpuArgon2Host::single_threaded(), &keys, &shape);
    for chunk in [1, 3, 16, 512] {
        let host = CpuArgon2Host::new().with_dynamic_chunk_size(chunk);
        assert_eq!(fill(&host, &keys, &shape), expected, "chunk size {chunk}");
    }
}

#[test]
fn the_batch_path_matches_filling_each_job_alone() {
    let shape = Argon2Shape::for_difficulty(64);
    let host = CpuArgon2Host::new();
    let keys = passwords(16);
    let batched = fill(&host, &keys, &shape);
    for (index, key) in keys.iter().enumerate() {
        let mut slot = [0u8; 2 * ARGON2_BLOCK_SIZE];
        host.fill_first_blocks(&mut slot, key.as_bytes(), SALT, &shape)
            .expect("single job fills");
        assert_eq!(
            &batched[index * slot.len()..(index + 1) * slot.len()],
            &slot[..],
            "job {index}"
        );
    }
}

/// The two blocks of a job are H'(H0 || 0 || 0) and H'(H0 || 1 || 0) — different seeds, so
/// a copy-paste slip that wrote the same block twice would show here.
#[test]
fn the_two_first_blocks_differ() {
    let shape = Argon2Shape::for_difficulty(8);
    let mut slot = [0u8; 2 * ARGON2_BLOCK_SIZE];
    CpuArgon2Host::new()
        .fill_first_blocks(&mut slot, b"abc", SALT, &shape)
        .expect("fills");
    assert_ne!(slot[..ARGON2_BLOCK_SIZE], slot[ARGON2_BLOCK_SIZE..]);
    assert!(slot.iter().any(|byte| *byte != 0));
}

/// Difficulty is part of H0, so two shapes cannot produce the same first blocks.
#[test]
fn the_shape_feeds_the_first_blocks() {
    let host = CpuArgon2Host::new();
    let mut low = [0u8; 2 * ARGON2_BLOCK_SIZE];
    let mut high = [0u8; 2 * ARGON2_BLOCK_SIZE];
    host.fill_first_blocks(&mut low, b"abc", SALT, &Argon2Shape::for_difficulty(8))
        .expect("fills");
    host.fill_first_blocks(&mut high, b"abc", SALT, &Argon2Shape::for_difficulty(64))
        .expect("fills");
    assert_ne!(low[..], high[..]);
}

/// H' for a 64-byte output is a single Blake2b over the little-endian length and the input.
#[test]
fn finalize_is_argon2s_h_prime() {
    let block: Vec<u8> = (0..ARGON2_BLOCK_SIZE).map(|i| (i % 251) as u8).collect();
    let mut digest = [0u8; 64];
    CpuArgon2Host::new()
        .finalize(&block, &mut digest)
        .expect("finalizes");

    let mut hasher = Blake2bVar::new(64).expect("64 is a legal blake2b length");
    hasher.update(&64u32.to_le_bytes());
    hasher.update(&block);
    let mut expected = [0u8; 64];
    hasher.finalize_variable(&mut expected).expect("finalizes");
    assert_eq!(digest, expected);
}

#[test]
fn wrong_sized_buffers_are_errors_not_panics() {
    let host = CpuArgon2Host::new();
    let shape = Argon2Shape::for_difficulty(8);
    let mut short = [0u8; 1024];
    assert!(host
        .fill_first_blocks(&mut short, b"abc", SALT, &shape)
        .is_err());

    let keys = passwords(4);
    let mut wrong = vec![0u8; 3 * 2 * ARGON2_BLOCK_SIZE];
    assert!(host
        .fill_first_blocks_batch(&mut wrong, &keys, SALT, &shape)
        .is_err());

    let mut digest = [0u8; 64];
    assert!(host.finalize(&[0u8; 999], &mut digest).is_err());
    assert!(host.finalize(&[], &mut digest).is_err());
}

#[test]
fn an_empty_batch_is_a_no_op() {
    assert!(CpuArgon2Host::new()
        .fill_first_blocks_batch(&mut [], &[], SALT, &Argon2Shape::for_difficulty(8))
        .is_ok());
}

#[test]
fn worker_count_follows_the_cpp_rules() {
    // Below kMinParallelFirstBlockAttempts there is exactly one worker, cap or no cap.
    for attempts in 0..8 {
        assert_eq!(first_block_worker_count(attempts, 0), 1);
        assert_eq!(first_block_worker_count(attempts, 16), 1);
    }
    assert_eq!(first_block_worker_count(64, 3), 3);
    assert_eq!(first_block_worker_count(8, 64), 8, "capped by the job count");
    let uncapped = first_block_worker_count(1024, 0);
    assert!(uncapped >= 1);
    assert_eq!(first_block_worker_count(1024, uncapped + 10), uncapped);
}

#[test]
fn chunk_size_rounds_up_and_honours_dynamic_chunks() {
    assert_eq!(first_block_chunk_size(0, 4), 0);
    assert_eq!(first_block_chunk_size(10, 0), 0);
    assert_eq!(first_block_chunk_size(10, 4), 3);
    assert_eq!(first_block_chunk_size(12, 4), 3);

    // A dynamic chunk size only applies with more than one worker.
    assert_eq!(first_block_selected_chunk_size(100, 1, 16), 100);
    assert_eq!(first_block_selected_chunk_size(100, 4, 16), 16);
    assert_eq!(first_block_selected_chunk_size(100, 4, 0), 25);
    assert_eq!(first_block_selected_chunk_size(8, 4, 16), 8, "clamped to attempts");
}

#[test]
fn the_recommended_dynamic_chunk_table_matches_the_cpp() {
    let recommend = |difficulty, attempts| {
        recommended_first_block_dynamic_chunk_size(true, "cuda", false, difficulty, attempts, 8)
    };
    assert_eq!(recommend(1, 2048), 16);
    assert_eq!(recommend(8, 2048), 16);
    assert_eq!(recommend(8, 1024), 32);
    assert_eq!(recommend(64, 2048), 16);
    assert_eq!(recommend(64, 4096), 0);
    assert_eq!(recommend(42069, 4096), 0);

    // Every gate: opt-out, CPU backend, a fixed key, too few jobs, a single worker.
    assert_eq!(
        recommended_first_block_dynamic_chunk_size(false, "cuda", false, 8, 2048, 8),
        0
    );
    assert_eq!(
        recommended_first_block_dynamic_chunk_size(true, "cpu", false, 8, 2048, 8),
        0
    );
    assert_eq!(
        recommended_first_block_dynamic_chunk_size(true, "cuda", true, 8, 2048, 8),
        0
    );
    assert_eq!(recommend(8, 1023), 0);
    assert_eq!(
        recommended_first_block_dynamic_chunk_size(true, "cuda", false, 8, 2048, 1),
        0
    );
}
