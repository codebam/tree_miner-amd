//! Throughput probe. Ignored by default: it takes tens of seconds and its number depends on
//! what else is using the card.
//!
//! Run with `./rs cargo test -p tm-gpu --release --test gpu_hashrate -- --ignored --nocapture`.

mod support;

use support::ReferenceArgon2Host;
use tm_gpu::{BatchRequest, GpuBackend, GpuHashBackend};

#[test]
#[ignore = "throughput measurement, not a correctness check"]
fn hashrate_at_difficulty_60000() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("hashrate") else {
        return;
    };
    let difficulty = 60_000;
    let mut backend = GpuBackend::new(device);
    backend.activate().expect("device activates");
    let plan = backend
        .plan_batch_size(difficulty, 0)
        .expect("batch size can be planned");
    let batch_size = plan.selected_batch_size;
    assert!(batch_size > 0, "no batch fits at difficulty {difficulty}");

    let mut hash_backend = GpuHashBackend::new(backend);
    let passwords: Vec<String> = (0..batch_size)
        .map(|index| format!("{index:064x}"))
        .collect();
    let request = BatchRequest::new(
        &passwords,
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        difficulty,
    );

    // One warm-up batch pays for the pool allocation and the first kernel load.
    hash_backend
        .run_batch(&request, &ReferenceArgon2Host)
        .expect("warm-up batch runs");

    let rounds = 5;
    let mut total_hashes = 0usize;
    let start = std::time::Instant::now();
    for _ in 0..rounds {
        let outcome = hash_backend
            .run_batch(&request, &ReferenceArgon2Host)
            .expect("batch runs");
        total_hashes += outcome.attempts;
    }
    let seconds = start.elapsed().as_secs_f64();
    eprintln!(
        "difficulty {difficulty}: batch {batch_size}, {total_hashes} hashes in {seconds:.2}s = \
         {:.0} h/s",
        total_hashes as f64 / seconds
    );
}
