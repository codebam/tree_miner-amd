//! The port of `runCpuCudaSelfTest`: the known-good vector from `PORT.md` must reproduce
//! byte for byte on the GPU, with the first blocks computed on the CPU and on the device.
//!
//! This is the test that stands between the miner and submitting invalid blocks.

mod support;

use support::ReferenceArgon2Host;
use tm_gpu::{BatchRequest, GpuBackend, GpuHashBackend};

const SELF_TEST_SALT: &str = "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc";
const SELF_TEST_KEY: &str =
    "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f";
const SELF_TEST_DIFFICULTY: u32 = 8;
const SELF_TEST_DIGEST: &str = "2PKfnaEX2s+Yf/Drzi92D8HJ+B6K+FppyT7g5glp2knIMlFGWhnyOb9r1QIPf0GaVUEw8KumqQZ/pK2dkNTDxA";

fn run_self_test(gpu_first_blocks: bool) -> Option<String> {
    let _guard = support::gpu_lock();
    let device = support::first_gpu_or_skip("gpu self-test")?;
    let mut backend = GpuHashBackend::new(GpuBackend::new(device));
    let passwords = vec![SELF_TEST_KEY.to_owned()];
    let mut request = BatchRequest::new(&passwords, SELF_TEST_SALT, SELF_TEST_DIFFICULTY);
    request.target_pattern = "SELFTEST-NO-MATCH";
    request.allow_xuni = false;
    request.gpu_first_blocks = gpu_first_blocks;
    let outcome = backend
        .run_batch(&request, &ReferenceArgon2Host)
        .expect("self-test batch runs");
    assert_eq!(outcome.attempts, 1);
    assert!(outcome.matches.is_empty());
    assert_eq!(outcome.gpu_first_blocks, gpu_first_blocks);
    outcome.hash
}

#[test]
fn reference_vector_matches_with_cpu_first_blocks() {
    let Some(hash) = run_self_test(false) else {
        return;
    };
    assert_eq!(hash, SELF_TEST_DIGEST);
}

#[test]
fn reference_vector_matches_with_gpu_first_blocks() {
    let Some(hash) = run_self_test(true) else {
        return;
    };
    assert_eq!(hash, SELF_TEST_DIGEST);
}

/// The two first-block paths must agree with each other as well as with the vector — a
/// device-side Blake2b that drifted would otherwise only be caught at difficulty 8.
#[test]
fn both_first_block_paths_agree_across_difficulties() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("first-block agreement") else {
        return;
    };
    let mut backend = GpuHashBackend::new(GpuBackend::new(device));
    let passwords: Vec<String> = (0..16)
        .map(|index| format!("{index:064x}"))
        .collect();
    for difficulty in [8u32, 64, 1024] {
        let mut cpu = BatchRequest::new(&passwords, SELF_TEST_SALT, difficulty);
        cpu.collect_digests = true;
        let cpu_digests = backend
            .run_batch(&cpu, &ReferenceArgon2Host)
            .expect("cpu first-blocks batch runs")
            .digests;

        let mut gpu = cpu.clone();
        gpu.gpu_first_blocks = true;
        let gpu_digests = backend
            .run_batch(&gpu, &ReferenceArgon2Host)
            .expect("gpu first-blocks batch runs")
            .digests;

        assert_eq!(cpu_digests, gpu_digests, "difficulty {difficulty}");
        assert_eq!(cpu_digests.len(), passwords.len());
    }
}
