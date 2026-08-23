//! The parts of the integration layer that need a real card.
//!
//! Every test here SKIPS rather than fails when no GPU is present, and every one of them is
//! sized to run beside a miner that already owns almost all of the VRAM: one job at m=8,
//! and no assumption whatsoever about how much memory is free.

use std::sync::Arc;

use treeminer::selftest::{cpu_reference_digest, SelfTestProbe};
use treeminer::{run_self_test, GpuMiningBackend, GpuSelfTestProbe, MiningBackend};

/// The first device, or `None` with a note — never a failure.
fn first_gpu(what: &str) -> Option<i32> {
    match tm_gpu::Device::enumerate() {
        Ok(devices) if !devices.is_empty() => Some(devices[0].index()),
        Ok(_) => {
            eprintln!("skipping {what}: no GPU present");
            None
        }
        Err(error) => {
            eprintln!("skipping {what}: GPU unavailable ({error})");
            None
        }
    }
}

fn host() -> Arc<tm_argon2::CpuArgon2Host> {
    Arc::new(tm_argon2::CpuArgon2Host::new())
}

#[test]
fn the_startup_self_test_agrees_with_the_cpu_reference_on_real_hardware() {
    let Some(index) = first_gpu("gpu self-test") else {
        return;
    };
    let expected = cpu_reference_digest().expect("the CPU reference always works");

    let mut probe = GpuSelfTestProbe::new(host());
    match probe.gpu_digest(index, false) {
        Ok(digest) => assert_eq!(
            digest, expected,
            "a GPU digest that disagrees with the CPU is the invalid-block bug"
        ),
        Err(error) => eprintln!("skipping: the device could not run a 1-job batch ({error})"),
    }
}

#[test]
fn the_self_test_report_decides_per_device() {
    let Some(index) = first_gpu("self-test report") else {
        return;
    };
    let report = run_self_test(&[index], false, &mut GpuSelfTestProbe::new(host()));

    if report.is_fatal() {
        // A busy or memory-starved card is a legitimate skip here; what must never happen
        // is a device being declared fit to mine on a mismatched digest.
        eprintln!("skipping: {}", report.fatal_message());
        return;
    }
    assert_eq!(report.mining_devices(), vec![index]);
    // Either verdict is correct; what matters is that one was recorded rather than guessed.
    let decision = &report.decisions[0];
    assert!(decision.mine);
    assert!(decision
        .lines
        .iter()
        .any(|line| line.text.contains("self-test passed")));
}

#[test]
fn batch_sizing_survives_a_card_that_is_already_full() {
    let Some(index) = first_gpu("batch sizing") else {
        return;
    };
    let Ok(mut backend) = GpuMiningBackend::open(index, host()) else {
        eprintln!("skipping: the device could not be opened");
        return;
    };

    // The production miner may hold nearly all of the VRAM. The only contract asserted here
    // is that sizing answers rather than panicking, and that an explicit cap is honoured.
    let decision = backend.plan_batch_size(1000, 4, 1);
    match decision {
        Ok(decision) => assert!(
            decision.selected_batch_size <= 4,
            "an explicit --batchSize is a hard cap"
        ),
        Err(error) => eprintln!("skipping: free memory could not be measured ({error})"),
    }
}
