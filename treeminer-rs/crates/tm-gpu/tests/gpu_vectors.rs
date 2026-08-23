//! Every fixture vector, produced by the C++ miner, must reproduce on the Rust GPU path —
//! on both first-block paths.

mod support;

use support::ReferenceArgon2Host;
use tm_gpu::{BatchRequest, GpuBackend, GpuHashBackend};

#[test]
fn fixture_vectors_reproduce_on_the_gpu() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("fixture vectors") else {
        return;
    };
    let vectors = support::load_vectors();
    assert!(
        vectors.len() >= 32,
        "expected at least 32 cross-checked vectors, found {}",
        vectors.len()
    );

    let mut backend = GpuHashBackend::new(GpuBackend::new(device));
    for vector in &vectors {
        for gpu_first_blocks in [false, true] {
            let passwords = vec![vector.key.clone()];
            let mut request =
                BatchRequest::new(&passwords, &vector.salt_hex, vector.difficulty);
            request.target_pattern = "";
            request.allow_xuni = false;
            request.gpu_first_blocks = gpu_first_blocks;
            let outcome = backend
                .run_batch(&request, &ReferenceArgon2Host)
                .unwrap_or_else(|error| {
                    panic!(
                        "m={} key={} gpu_first_blocks={gpu_first_blocks}: {error}",
                        vector.difficulty, vector.key
                    )
                });
            assert_eq!(
                outcome.hash.as_deref(),
                Some(vector.digest_b64.as_str()),
                "m={} key={} gpu_first_blocks={gpu_first_blocks}",
                vector.difficulty,
                vector.key
            );
            assert!(vector.phc.ends_with(&vector.digest_b64));
        }
    }
    eprintln!("verified {} vectors on both first-block paths", vectors.len());
}

/// A batch is not just N independent single hashes: the pool is strided, so a wrong stride
/// only shows up when several jobs share one allocation.
#[test]
fn a_vector_still_matches_when_batched_with_others() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("batched vector") else {
        return;
    };
    let vectors = support::load_vectors();
    let mut backend = GpuHashBackend::new(GpuBackend::new(device));

    // Group by (salt, difficulty) so a whole group can go through one batch.
    let mut groups: std::collections::BTreeMap<(String, u32), Vec<&support::Vector>> =
        std::collections::BTreeMap::new();
    for vector in &vectors {
        groups
            .entry((vector.salt_hex.clone(), vector.difficulty))
            .or_default()
            .push(vector);
    }

    let mut batched_groups = 0;
    for ((salt, difficulty), group) in groups {
        if group.len() < 2 {
            continue;
        }
        batched_groups += 1;
        let passwords: Vec<String> = group.iter().map(|vector| vector.key.clone()).collect();
        let mut request = BatchRequest::new(&passwords, &salt, difficulty);
        request.target_pattern = "";
        request.allow_xuni = false;
        request.collect_digests = true;
        let outcome = backend
            .run_batch(&request, &ReferenceArgon2Host)
            .expect("batched vectors run");
        for (index, vector) in group.iter().enumerate() {
            assert_eq!(
                outcome.digests[index], vector.digest_b64,
                "batched m={difficulty} index {index}"
            );
        }
    }
    assert!(batched_groups > 0, "no fixture group had two vectors to batch");
}
