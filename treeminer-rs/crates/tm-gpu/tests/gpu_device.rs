//! Device enumeration, memory reporting, pool reuse and the failure modes an operator can
//! hit by asking for a batch the card cannot hold.

mod support;

use support::ReferenceArgon2Host;
use tm_gpu::error::GpuError;
use tm_gpu::{Argon2Shape, BatchRequest, GpuBackend, GpuHashBackend, TelemetrySession};

#[test]
fn enumeration_reports_a_usable_device() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("enumeration") else {
        return;
    };
    assert!(!device.name().is_empty());
    assert!(device.pci_bus_id().contains(':'));
    assert!(device.bus_id() >= 0, "bus id should parse out of the PCI id");

    let total = device.total_memory_bytes();
    assert!(
        total >= 1 << 30,
        "a mining GPU should report at least 1 GiB, got {total}"
    );
    let free = device.free_memory_bytes().expect("free memory is readable");
    assert!(free > 0 && free <= total, "free {free} of total {total}");
    assert!(device.full_name().contains(" GB"));
    eprintln!(
        "device {}: {} ({} free of {} bytes)",
        device.index(),
        device.full_name(),
        free,
        total
    );
}

#[test]
fn a_batch_far_too_large_for_vram_fails_as_an_error() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("oversized batch") else {
        return;
    };
    let total = device.total_memory_bytes();
    let mut backend = GpuBackend::new(device);
    backend.activate().expect("device activates");

    // Difficulty 60000 is ~60 MiB per job; ask for twice the card's memory worth of jobs.
    let shape = Argon2Shape::for_difficulty(60_000);
    let batch_size = (2 * total / shape.job_bytes()) + 1;
    let error = backend
        .init(&shape, batch_size)
        .expect_err("an impossible pool must fail, not silently fall back to host memory");
    // The point is that the driver refused — not which driver it was. Asserting the HIP
    // variant made this test vendor-specific, and on CUDA it failed on a correct refusal
    // (`cuMemAlloc failed: out of memory`), which is the behaviour it exists to demand.
    assert!(
        matches!(error, GpuError::Hip { .. } | GpuError::Cuda { .. }),
        "expected a driver allocation error, got {error}"
    );
    assert!(backend.runner().is_none(), "a failed init must leave no pool");
}

#[test]
fn batch_planning_releases_the_pool_before_measuring() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("batch planning") else {
        return;
    };
    let mut backend = GpuBackend::new(device);
    backend.activate().expect("device activates");

    let plan = backend
        .plan_batch_size(60_000, 0)
        .expect("planning works on an idle device");
    assert!(plan.selected_batch_size > 0);

    let held = plan.selected_batch_size.min(8);
    backend
        .init(&Argon2Shape::for_difficulty(60_000), held)
        .expect("the planned batch fits");
    let with_pool = backend.free_memory_bytes().expect("free memory is readable");

    // Planning again must hand back the pool first, or its bytes are counted as used and
    // every difficulty change ratchets the batch size down.
    let replanned = backend.plan_batch_size(60_000, 0).expect("planning works");
    assert!(backend.runner().is_none(), "planning must release the pool");
    let without_pool = backend.free_memory_bytes().expect("free memory is readable");
    assert!(
        without_pool > with_pool,
        "releasing a {held}-job pool should free VRAM: {with_pool} -> {without_pool}"
    );
    assert!(replanned.selected_batch_size > 0);

    let capped = backend.plan_batch_size(60_000, 4).expect("planning works");
    assert_eq!(capped.selected_batch_size, 4);
    assert!(capped.explicit_limit_applied);
}

#[test]
fn the_pool_is_reused_for_a_smaller_difficulty_and_rebuilt_for_a_larger_one() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("pool reuse") else {
        return;
    };
    let mut backend = GpuBackend::new(device);
    backend.activate().expect("device activates");

    let large = Argon2Shape::for_difficulty(1024);
    backend.init(&large, 8).expect("initial pool allocates");
    let large_pointer = backend.runner().map(std::ptr::from_ref);

    backend
        .init(&Argon2Shape::for_difficulty(64), 8)
        .expect("a smaller shape reuses the pool");
    assert_eq!(
        backend.runner().map(std::ptr::from_ref),
        large_pointer,
        "a shape that fits must reuse the existing runner"
    );
    assert_eq!(backend.runner().map(|runner| runner.shape().memory_cost), Some(64));

    backend
        .init(&Argon2Shape::for_difficulty(4096), 8)
        .expect("a larger shape reallocates");
    assert_eq!(
        backend.runner().map(|runner| runner.shape().memory_cost),
        Some(4096)
    );
}

#[test]
fn an_empty_or_malformed_request_is_rejected_without_touching_the_device() {
    let _guard = support::gpu_lock();
    let Some(device) = support::first_gpu_or_skip("request validation") else {
        return;
    };
    let mut backend = GpuHashBackend::new(GpuBackend::new(device));
    let passwords = vec!["00".repeat(32)];

    let empty: Vec<String> = Vec::new();
    let error = backend
        .run_batch(
            &BatchRequest::new(&empty, "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc", 8),
            &ReferenceArgon2Host,
        )
        .expect_err("an empty batch is rejected");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");

    let error = backend
        .run_batch(
            &BatchRequest::new(&passwords, "not-hex", 8),
            &ReferenceArgon2Host,
        )
        .expect_err("a non-hex salt is rejected");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");

    let error = backend
        .run_batch(
            &BatchRequest::new(&passwords, "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc", 0),
            &ReferenceArgon2Host,
        )
        .expect_err("difficulty zero is rejected");
    assert!(matches!(error, GpuError::Invalid(_)), "{error}");
}

/// Telemetry is optional: it must report either real numbers or nothing, never fail.
#[test]
fn telemetry_reports_or_degrades() {
    let _guard = support::gpu_lock();
    let session = TelemetrySession::new();
    let (index, bus) = match support::first_gpu_or_skip("telemetry") {
        Some(device) => (device.index(), device.bus_id()),
        None => return,
    };
    let telemetry = session.query(index, bus);
    if session.available() {
        assert_eq!(session.source_name(), "ROCm SMI");
        eprintln!(
            "telemetry: power={:?} mW utilisation={:?} %",
            telemetry.power_milliwatts, telemetry.utilization_percent
        );
        if let Some(utilization) = telemetry.utilization_percent {
            assert!(utilization <= 100);
        }
    } else {
        eprintln!("ROCm SMI unavailable; telemetry degraded to none");
        assert_eq!(telemetry, Default::default());
    }
}
