//! Port of `src/hashapi/HashApiTuning.cpp`, including the `TREEMINER_GPU_HIP` branch that
//! is a compile-time `#if` in C++ and a runtime `GpuRuntimeKind` here.

use tm_core::batch::{
    effective_reserve_bytes, estimate_memory_batch_limit, recommended_batch_size,
    DEFAULT_MEMORY_RESERVE_BYTES, HIP_RESERVE_FLOOR_BYTES,
};
use tm_core::{select_batch_size, GpuRuntimeKind};

const GIB: usize = 1024 * 1024 * 1024;
const CUDA: GpuRuntimeKind = GpuRuntimeKind::Cuda;
const HIP: GpuRuntimeKind = GpuRuntimeKind::Hip;

#[test]
fn the_cuda_path_keeps_the_hundred_mebibyte_reserve() {
    assert_eq!(DEFAULT_MEMORY_RESERVE_BYTES, 100 * 1024 * 1024);
    for free in [512 * 1024 * 1024, 8 * GIB, 24 * GIB] {
        assert_eq!(
            effective_reserve_bytes(CUDA, free, DEFAULT_MEMORY_RESERVE_BYTES),
            DEFAULT_MEMORY_RESERVE_BYTES,
            "free {free}"
        );
    }
    // An explicit reserve is honoured as given, never shrunk.
    assert_eq!(effective_reserve_bytes(CUDA, 24 * GIB, 4 * GIB), 4 * GIB);
}

#[test]
fn the_hip_path_takes_the_largest_of_reserve_one_gibibyte_and_a_sixteenth() {
    assert_eq!(HIP_RESERVE_FLOOR_BYTES, GIB);
    // Small card: the 1 GiB floor wins over both the default reserve and free/16.
    assert_eq!(
        effective_reserve_bytes(HIP, 8 * GIB, DEFAULT_MEMORY_RESERVE_BYTES),
        GIB
    );
    // 24 GiB card: the proportional sixteenth (1.5 GiB) wins over the floor.
    assert_eq!(
        effective_reserve_bytes(HIP, 24 * GIB, DEFAULT_MEMORY_RESERVE_BYTES),
        24 * GIB / 16
    );
    // A caller-supplied reserve larger than both still wins.
    assert_eq!(effective_reserve_bytes(HIP, 24 * GIB, 4 * GIB), 4 * GIB);
    // Exactly at the crossover: free/16 == 1 GiB at 16 GiB free.
    assert_eq!(
        effective_reserve_bytes(HIP, 16 * GIB, DEFAULT_MEMORY_RESERVE_BYTES),
        GIB
    );
}

/// The measured gfx1100 cliff: at difficulty 60000 on a 24 GiB card, batch 410 held
/// 3.1 kH/s and batch 415 collapsed to 0.5 kH/s because ROCm silently backed the pool with
/// host GTT memory instead of failing the allocation. The 100 MiB CUDA cushion cannot see
/// that edge; the proportional HIP reserve must stay under 410.
#[test]
fn the_gfx1100_performance_cliff_regression() {
    let free = 24 * GIB;
    let hip = select_batch_size(HIP, free, 60000, 0);
    assert!(
        hip.selected_batch_size <= 410,
        "HIP selected {} — past the measured cliff",
        hip.selected_batch_size
    );
    assert_eq!(hip.selected_batch_size, 392);

    let cuda = select_batch_size(CUDA, free, 60000, 0);
    assert!(
        cuda.selected_batch_size > 410,
        "the CUDA reserve is supposed to overshoot the cliff"
    );
    assert!(cuda.selected_batch_size >= 415);

    // Difficulty 60000 has no tuned ceiling, so the memory limit is what gets selected.
    assert_eq!(hip.tuned_batch_size, 0);
    assert!(!hip.tuned_default_applied);
    assert!(!hip.explicit_limit_applied);
    assert_eq!(hip.selected_batch_size, hip.memory_limited_batch_size);
}

#[test]
fn the_tuned_ceilings_step_at_one_eight_and_sixty_four() {
    assert_eq!(recommended_batch_size(0), 2048);
    assert_eq!(recommended_batch_size(1), 2048);
    assert_eq!(recommended_batch_size(2), 4096);
    assert_eq!(recommended_batch_size(8), 4096);
    assert_eq!(recommended_batch_size(9), 3072);
    assert_eq!(recommended_batch_size(64), 3072);
    // Past 64 memory is the binding constraint, so there is no tuned ceiling at all.
    assert_eq!(recommended_batch_size(65), 0);
    assert_eq!(recommended_batch_size(42069), 0);
}

#[test]
fn the_tuned_ceiling_is_applied_when_memory_is_not_the_constraint() {
    let free = 24 * GIB;
    for (difficulty, expected) in [(1u32, 2048usize), (8, 4096), (64, 3072)] {
        let decision = select_batch_size(CUDA, free, difficulty, 0);
        assert_eq!(decision.tuned_batch_size, expected, "m={difficulty}");
        assert_eq!(decision.selected_batch_size, expected, "m={difficulty}");
        assert!(decision.tuned_default_applied);
        assert!(!decision.explicit_limit_applied);
        assert!(decision.memory_limited_batch_size > expected);
    }

    // At 65 the ceiling disappears and the memory limit takes over.
    let decision = select_batch_size(CUDA, free, 65, 0);
    assert_eq!(decision.tuned_batch_size, 0);
    assert!(!decision.tuned_default_applied);
    assert_eq!(decision.selected_batch_size, decision.memory_limited_batch_size);
    assert_eq!(decision.selected_batch_size, 385205);
}

#[test]
fn the_tuned_ceiling_never_raises_the_batch_above_the_memory_limit() {
    // 120 MiB free at difficulty 8: the tuned 4096 is past what fits in the 20 MiB left
    // over after the reserve.
    let decision = select_batch_size(CUDA, 120 * 1024 * 1024, 8, 0);
    assert_eq!(decision.tuned_batch_size, 4096);
    assert!(decision.memory_limited_batch_size < 4096);
    assert_eq!(decision.selected_batch_size, decision.memory_limited_batch_size);
    assert!(decision.tuned_default_applied);
}

#[test]
fn an_explicit_maximum_wins_over_the_tuned_ceiling() {
    let free = 24 * GIB;
    let decision = select_batch_size(CUDA, free, 8, 512);
    assert_eq!(decision.selected_batch_size, 512);
    assert!(decision.explicit_limit_applied);
    // The tuned path is not even consulted once an explicit limit is in play.
    assert_eq!(decision.tuned_batch_size, 0);
    assert!(!decision.tuned_default_applied);

    // An explicit maximum above the tuned ceiling still wins — it is a cap, not a clamp
    // towards the tuned value.
    let decision = select_batch_size(CUDA, free, 8, 8192);
    assert_eq!(decision.selected_batch_size, 8192);
    assert!(decision.explicit_limit_applied);

    // But it can never exceed what fits in memory.
    let decision = select_batch_size(HIP, 2 * GIB, 42069, 8192);
    assert_eq!(decision.selected_batch_size, decision.memory_limited_batch_size);
    assert!(decision.selected_batch_size < 8192);
}

#[test]
fn free_memory_at_or_below_the_reserve_yields_zero() {
    // CUDA: below the 100 MiB reserve.
    let decision = select_batch_size(CUDA, 50 * 1024 * 1024, 8, 0);
    assert_eq!(decision.selected_batch_size, 0);
    assert_eq!(decision.memory_limited_batch_size, 0);
    assert!(!decision.explicit_limit_applied);
    assert!(!decision.tuned_default_applied);
    // Exactly equal to the reserve is also zero — the comparison is `<=`.
    assert_eq!(
        estimate_memory_batch_limit(CUDA, DEFAULT_MEMORY_RESERVE_BYTES, 8, DEFAULT_MEMORY_RESERVE_BYTES),
        0
    );
    assert_eq!(estimate_memory_batch_limit(CUDA, 0, 8, DEFAULT_MEMORY_RESERVE_BYTES), 0);

    // HIP: 1 GiB free is exactly the floor, so nothing is left over.
    assert_eq!(select_batch_size(HIP, GIB, 8, 0).selected_batch_size, 0);
    // An explicit maximum cannot resurrect a batch that does not fit.
    assert_eq!(select_batch_size(HIP, GIB, 8, 4096).selected_batch_size, 0);
}

#[test]
fn difficulty_zero_yields_zero_rather_than_dividing_by_it() {
    assert_eq!(
        estimate_memory_batch_limit(CUDA, 24 * GIB, 0, DEFAULT_MEMORY_RESERVE_BYTES),
        0
    );
    assert_eq!(select_batch_size(CUDA, 24 * GIB, 0, 0).selected_batch_size, 0);
    assert_eq!(select_batch_size(HIP, 24 * GIB, 0, 4096).selected_batch_size, 0);
}

#[test]
fn the_limit_carries_the_one_tenth_percent_allocator_overhead() {
    // bytes_per_attempt is difficulty * 1024 * 1.001, so the limit sits just under the
    // naive KiB division — dropping the overhead would over-allocate by a block per batch.
    let free = 8 * GIB;
    let naive = (free - GIB) / (1024 * 1024);
    let limit = estimate_memory_batch_limit(HIP, free, 1024, DEFAULT_MEMORY_RESERVE_BYTES);
    assert!(limit < naive, "{limit} should be under the naive {naive}");
    assert_eq!(limit, 7160);
}

#[test]
fn the_hip_reserve_only_changes_the_result_when_memory_binds() {
    // At difficulty 8 both runtimes are held by the tuned ceiling, so the reserve is
    // invisible; the divergence is a memory-bound property, not a global one.
    let free = 24 * GIB;
    assert_eq!(
        select_batch_size(CUDA, free, 8, 0).selected_batch_size,
        select_batch_size(HIP, free, 8, 0).selected_batch_size
    );
    assert_ne!(
        select_batch_size(CUDA, free, 60000, 0).selected_batch_size,
        select_batch_size(HIP, free, 60000, 0).selected_batch_size
    );
}
