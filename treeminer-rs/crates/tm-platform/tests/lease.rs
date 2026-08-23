//! Lease lifecycle: start, expiry, early termination, and the mining context it produces.

mod common;

use common::*;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use tm_platform::clock::TestClock;
use tm_platform::coordinator::MiningMode;
use tm_platform::lease::{LeaseError, LeaseManager};
use tm_platform::PlatformState;

fn manager() -> (LeaseManager, Arc<TestClock>) {
    let clock = Arc::new(TestClock::new(NOW));
    (LeaseManager::new(clock.clone()), clock)
}

#[test]
fn a_fresh_manager_has_no_lease() {
    let (leases, _) = manager();
    assert!(leases.lease().is_none());
    assert!(!leases.has_active_lease());
    // "No lease" reads as expired, matching the C++ — the watchdog relies on it.
    assert!(leases.is_expired());
    assert_eq!(leases.remaining_seconds(), 0);
    assert!(leases.end_lease().is_none());
}

#[test]
fn start_and_end() {
    let (leases, _) = manager();
    leases
        .start_lease("L-1", "C-1", CONSUMER_ADDRESS, "a1b2c3d4e5f6a7b8", 600)
        .unwrap();
    let lease = leases.lease().unwrap();
    assert_eq!(lease.lease_id, "L-1");
    assert_eq!(lease.consumer_id, "C-1");
    assert_eq!(lease.consumer_address, CONSUMER_ADDRESS);
    assert_eq!(lease.blocks_found, 0);
    assert!(leases.has_active_lease());
    assert_eq!(leases.remaining_seconds(), 600);

    let ended = leases.end_lease().unwrap();
    assert_eq!(ended.lease_id, "L-1");
    assert!(!leases.has_active_lease());
    assert!(leases.end_lease().is_none(), "ending twice is a no-op");
}

#[test]
fn a_second_lease_cannot_overwrite_a_live_one() {
    let (leases, _) = manager();
    leases.start_lease("L-1", "C-1", CONSUMER_ADDRESS, "", 600).unwrap();
    assert_eq!(
        leases.start_lease("L-2", "C-2", SELF_ADDRESS, "", 600),
        Err(LeaseError::AlreadyLeased("L-1".into()))
    );
    assert_eq!(leases.lease().unwrap().lease_id, "L-1");
}

#[test]
fn expiry_counts_down_and_then_expires() {
    let (leases, clock) = manager();
    leases.start_lease("L-1", "C-1", CONSUMER_ADDRESS, "", 600).unwrap();

    clock.advance(Duration::from_secs(599));
    assert_eq!(leases.remaining_seconds(), 1);
    assert!(!leases.is_expired());
    assert!(leases.has_active_lease());

    // Expiry is inclusive at the boundary: at exactly `duration_sec` the lease is over.
    clock.advance(Duration::from_secs(1));
    assert_eq!(leases.remaining_seconds(), 0);
    assert!(leases.is_expired());
    assert!(!leases.has_active_lease());

    clock.advance(Duration::from_secs(100_000));
    assert_eq!(leases.remaining_seconds(), 0, "remaining never goes negative");
    // The record survives expiry until something ends it — the watchdog is what reports
    // and clears it.
    assert!(leases.lease().is_some());
}

/// The wall clock moving must not end or extend a paid lease.
#[test]
fn a_wall_clock_step_does_not_move_the_expiry() {
    let (leases, clock) = manager();
    leases.start_lease("L-1", "C-1", CONSUMER_ADDRESS, "", 600).unwrap();
    clock.set_epoch(NOW + 10_000);
    assert!(!leases.is_expired());
    clock.set_epoch(NOW - 10_000);
    assert!(!leases.is_expired());
    clock.advance(Duration::from_secs(600));
    assert!(leases.is_expired());
}

/// Duration 0 means "no expiry", as in the C++.
#[test]
fn a_zero_duration_lease_never_expires() {
    let (leases, clock) = manager();
    leases.start_lease("L-1", "C-1", CONSUMER_ADDRESS, "", 0).unwrap();
    clock.advance(Duration::from_secs(86_400 * 365));
    assert!(!leases.is_expired());
    assert!(leases.has_active_lease());
}

#[test]
fn blocks_are_counted_against_the_lease_only() {
    let (leases, _) = manager();
    leases.record_block(); // no lease: ignored, not a panic
    leases.start_lease("L-1", "C-1", CONSUMER_ADDRESS, "", 600).unwrap();
    leases.record_block();
    leases.record_block();
    assert_eq!(leases.lease().unwrap().blocks_found, 2);
    leases.end_lease();
    leases.record_block();
    assert!(leases.lease().is_none());
}

#[test]
fn the_mining_context_follows_the_lease() {
    let (leases, _) = manager();
    let idle = leases.to_mining_context(SELF_ADDRESS);
    assert_eq!(idle.mode, MiningMode::SelfMining);
    assert_eq!(idle.address, SELF_ADDRESS);

    leases
        .start_lease("L-1", "C-1", CONSUMER_ADDRESS, "a1b2c3d4e5f6a7b8", 600)
        .unwrap();
    let leased = leases.to_mining_context(SELF_ADDRESS);
    assert_eq!(leased.mode, MiningMode::PlatformMining);
    assert_eq!(leased.address, CONSUMER_ADDRESS);
    assert_eq!(leased.prefix, "a1b2c3d4e5f6a7b8");
    assert_eq!(leased.consumer_id, "C-1");
    assert_eq!(leased.lease_id, "L-1");
}

// --- Through the manager ---

/// The watchdog is what turns an expired lease back into an available rig, and it puts
/// mining back on the operator's own address.
#[test]
fn the_watchdog_ends_an_expired_lease() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 600)));
    assert_eq!(harness.manager.state(), PlatformState::Mining);

    harness.clock.advance(Duration::from_secs(599));
    harness.manager.watchdog_tick();
    assert_eq!(harness.manager.state(), PlatformState::Mining);

    harness.clock.advance(Duration::from_secs(1));
    harness.manager.watchdog_tick();
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert!(harness.manager.leases().lease().is_none());
    let ctx = harness.coordinator.context();
    assert_eq!(ctx.mode, MiningMode::SelfMining);
    assert_eq!(ctx.address, SELF_ADDRESS);

    // COMPLETED is passed through on the way, so the platform sees the full transition.
    let states: Vec<String> = harness
        .transport
        .published_on("status")
        .iter()
        .map(|s| s["state"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(states, ["LEASED", "MINING", "COMPLETED", "AVAILABLE"]);
}

/// Early termination by the platform, before the lease runs out.
#[test]
fn early_release_ends_the_lease_before_expiry() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    harness.clock.advance(Duration::from_secs(10));
    harness.deliver(
        "task",
        &harness.sign(&json!({ "command": "release", "lease_id": "lease-1" })),
    );
    assert_eq!(harness.manager.state(), PlatformState::Available);
    assert!(harness.manager.leases().lease().is_none());
    assert_eq!(harness.coordinator.context().mode, MiningMode::SelfMining);

    // And the rig can take a new lease straight away.
    harness.deliver("task", &harness.sign(&assign_task("lease-2", 3600)));
    assert_eq!(harness.manager.state(), PlatformState::Mining);
    assert_eq!(harness.manager.leases().lease().unwrap().lease_id, "lease-2");
}

/// A full cycle: assign, find blocks, expire, take another lease.
#[test]
fn lease_cycle_repeats() {
    let harness = Harness::new(Some(SECRET));
    for round in 0..3 {
        let id = format!("lease-{round}");
        harness.deliver("task", &harness.sign(&assign_task(&id, 600)));
        assert_eq!(harness.manager.state(), PlatformState::Mining, "round {round}");
        harness
            .manager
            .on_block_found("00000a", "beef", CONSUMER_ADDRESS, 1, 1.0);
        assert_eq!(harness.manager.leases().lease().unwrap().blocks_found, 1);
        harness.clock.advance(Duration::from_secs(600));
        harness.manager.watchdog_tick();
        assert_eq!(harness.manager.state(), PlatformState::Available, "round {round}");
    }
}

/// A find after the lease ends is reported as self-mined, never attributed to the lease
/// that just closed.
#[test]
fn a_find_after_release_is_not_attributed_to_the_lease() {
    let harness = Harness::new(Some(SECRET));
    harness.deliver("task", &harness.sign(&assign_task("lease-1", 3600)));
    harness.deliver("task", &harness.sign(&json!({ "command": "release" })));
    harness.transport.clear();
    harness
        .manager
        .on_block_found("00000a", "beef", SELF_ADDRESS, 1, 1.0);
    let blocks = harness.transport.published_on("block");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["lease_id"], "");
}
