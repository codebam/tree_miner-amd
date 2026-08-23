//! The production host (`tm_argon2::CpuArgon2Host`) against the independent transcription
//! in `support`, and then end to end through the GPU against the C++ fixtures.
//!
//! The two transcriptions were written separately from `src/argon2params.cpp`; a bug in
//! either shows up here as a mismatch rather than as two copies agreeing with each other.

mod support;

use support::ReferenceArgon2Host;
use tm_argon2::CpuArgon2Host;
use tm_gpu::{Argon2Host, Argon2Shape, BatchRequest, GpuBackend, GpuHashBackend};

const BLOCK: usize = 1024;

fn salt_bytes(salt_hex: &str) -> Vec<u8> {
    tm_core::encoding::hex_to_bytes(salt_hex).expect("fixture salt is hex")
}

#[test]
fn first_blocks_match_the_reference_transcription() {
    let vectors = support::load_vectors();
    assert!(!vectors.is_empty());
    for vector in &vectors {
        let shape = Argon2Shape::for_difficulty(vector.difficulty);
        let salt = salt_bytes(&vector.salt_hex);
        let mut mine = [0u8; 2 * BLOCK];
        let mut reference = [0u8; 2 * BLOCK];
        CpuArgon2Host::new()
            .fill_first_blocks(&mut mine, vector.key.as_bytes(), &salt, &shape)
            .expect("cpu host fills");
        ReferenceArgon2Host
            .fill_first_blocks(&mut reference, vector.key.as_bytes(), &salt, &shape)
            .expect("reference host fills");
        assert_eq!(
            mine, reference,
            "m={} key={}",
            vector.difficulty, vector.key
        );
    }
}

#[test]
fn finalize_matches_the_reference_transcription() {
    let block: Vec<u8> = (0..BLOCK).map(|index| (index * 7 % 256) as u8).collect();
    let mut mine = [0u8; 64];
    let mut reference = [0u8; 64];
    CpuArgon2Host::new()
        .finalize(&block, &mut mine)
        .expect("cpu host finalizes");
    ReferenceArgon2Host
        .finalize(&block, &mut reference)
        .expect("reference host finalizes");
    assert_eq!(mine, reference);
}

/// The whole point of the host half: with it, the GPU path reproduces the digests the C++
/// miner produced. Batches stay tiny because the card is shared with a live miner.
#[test]
fn fixture_vectors_reproduce_through_the_cpu_host() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("cpu-host fixture vectors") else {
        return;
    };
    let host = CpuArgon2Host::new();
    let mut backend = GpuHashBackend::new(GpuBackend::new(device));
    let vectors = support::load_vectors();
    for vector in &vectors {
        let passwords = vec![vector.key.clone()];
        let mut request = BatchRequest::new(&passwords, &vector.salt_hex, vector.difficulty);
        request.target_pattern = "";
        request.allow_xuni = false;
        let outcome = backend
            .run_batch(&request, &host)
            .unwrap_or_else(|error| panic!("m={}: {error}", vector.difficulty));
        assert_eq!(
            outcome.hash.as_deref(),
            Some(vector.digest_b64.as_str()),
            "m={} key={}",
            vector.difficulty,
            vector.key
        );
    }
    eprintln!("verified {} vectors through CpuArgon2Host", vectors.len());
}

/// A multi-job batch takes the threaded fill path, which is where a stride or chunking bug
/// would live. Eight jobs is the smallest batch that crosses the parallel threshold, so the
/// fixture group is padded to that with a filler key whose digest is not checked.
#[test]
fn a_threaded_batch_reproduces_its_fixture_digests() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("threaded batch") else {
        return;
    };
    let vectors = support::load_vectors();
    let salt = vectors
        .iter()
        .find(|vector| vector.difficulty == 8)
        .map(|vector| vector.salt_hex.clone())
        .expect("a difficulty-8 fixture");
    let group: Vec<&support::Vector> = vectors
        .iter()
        .filter(|vector| vector.difficulty == 8 && vector.salt_hex == salt)
        .take(8)
        .collect();
    assert!(!group.is_empty(), "no difficulty-8 fixtures");

    let mut passwords: Vec<String> = group.iter().map(|vector| vector.key.clone()).collect();
    while passwords.len() < tm_argon2::MIN_PARALLEL_FIRST_BLOCK_ATTEMPTS {
        passwords.push(format!("{:064x}", passwords.len()));
    }
    let mut request = BatchRequest::new(&passwords, &salt, 8);
    request.target_pattern = "";
    request.allow_xuni = false;
    request.collect_digests = true;

    let host = CpuArgon2Host::new().with_workers(4);
    assert!(
        host.worker_count(passwords.len()) > 1,
        "the batch must be big enough to be filled in parallel"
    );

    let mut backend = GpuHashBackend::new(GpuBackend::new(device));
    let outcome = backend
        .run_batch(&request, &host)
        .expect("threaded batch runs");
    for (index, vector) in group.iter().enumerate() {
        assert_eq!(outcome.digests[index], vector.digest_b64, "job {index}");
    }
}
