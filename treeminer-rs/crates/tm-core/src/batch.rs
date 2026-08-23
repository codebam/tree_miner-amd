//! GPU batch sizing. Port of `src/hashapi/HashApiTuning.cpp`, including the ROCm-specific
//! VRAM cushion: ROCm satisfies an over-large device allocation from host (GTT) memory
//! instead of failing it, so a batch sized to the last byte of VRAM allocates fine and then
//! runs every kernel across PCIe (measured on gfx1100 at difficulty 60000: batch 410 held
//! 3.1 kH/s, batch 415 collapsed to 0.5 kH/s).

pub const DEFAULT_MEMORY_RESERVE_BYTES: usize = 100 * 1024 * 1024;
pub const HIP_RESERVE_FLOOR_BYTES: usize = 1024 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BatchSizeDecision {
    pub memory_limited_batch_size: usize,
    pub tuned_batch_size: usize,
    pub selected_batch_size: usize,
    pub explicit_limit_applied: bool,
    pub tuned_default_applied: bool,
}

/// Which runtime the pool is allocated from — HIP needs the larger cushion described above.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuRuntimeKind {
    Cuda,
    Hip,
}

pub fn effective_reserve_bytes(
    runtime: GpuRuntimeKind,
    free_memory_bytes: usize,
    reserve_bytes: usize,
) -> usize {
    match runtime {
        GpuRuntimeKind::Cuda => reserve_bytes,
        GpuRuntimeKind::Hip => reserve_bytes
            .max(HIP_RESERVE_FLOOR_BYTES)
            .max(free_memory_bytes / 16),
    }
}

pub fn estimate_memory_batch_limit(
    runtime: GpuRuntimeKind,
    free_memory_bytes: usize,
    difficulty: u32,
    reserve_bytes: usize,
) -> usize {
    let reserve = effective_reserve_bytes(runtime, free_memory_bytes, reserve_bytes);
    if difficulty == 0 || free_memory_bytes <= reserve {
        return 0;
    }
    let available = (free_memory_bytes - reserve) as f64;
    let bytes_per_attempt = difficulty as f64 * 1024.0 * 1.001;
    (available / bytes_per_attempt) as usize
}

/// Hand-tuned batch ceilings for the low difficulties where memory is not the binding
/// constraint. 0 means "no tuned ceiling, use the memory limit".
pub fn recommended_batch_size(difficulty: u32) -> usize {
    match difficulty {
        0..=1 => 2048,
        2..=8 => 4096,
        9..=64 => 3072,
        _ => 0,
    }
}

pub fn select_batch_size(
    runtime: GpuRuntimeKind,
    free_memory_bytes: usize,
    difficulty: u32,
    explicit_max_batch_size: usize,
) -> BatchSizeDecision {
    let mut decision = BatchSizeDecision {
        memory_limited_batch_size: estimate_memory_batch_limit(
            runtime,
            free_memory_bytes,
            difficulty,
            DEFAULT_MEMORY_RESERVE_BYTES,
        ),
        ..Default::default()
    };
    if decision.memory_limited_batch_size == 0 {
        return decision;
    }
    if explicit_max_batch_size > 0 {
        decision.selected_batch_size = decision
            .memory_limited_batch_size
            .min(explicit_max_batch_size);
        decision.explicit_limit_applied = true;
        return decision;
    }
    decision.tuned_batch_size = recommended_batch_size(difficulty);
    if decision.tuned_batch_size > 0 {
        decision.selected_batch_size = decision
            .memory_limited_batch_size
            .min(decision.tuned_batch_size);
        decision.tuned_default_applied = true;
        return decision;
    }
    decision.selected_batch_size = decision.memory_limited_batch_size;
    decision
}
