//! Port of `tests/unit/submit/test_drain_scheduler.cpp` — pure ordering + adaptive pacing.

mod common;

use common::record;
use tm_core::FindKind;
use tm_submit::drain::{DifficultyTrend, DrainConfig, DrainScheduler, XuniWindowState};

fn window(open: bool, ms_until_close: i64) -> XuniWindowState {
    XuniWindowState {
        open,
        ms_until_close,
    }
}

const CLOSED: XuniWindowState = XuniWindowState {
    open: false,
    ms_until_close: 0,
};

#[test]
fn empty_backlog_selects_nothing() {
    let sched = DrainScheduler::default();
    assert!(sched
        .select_next(&[], DifficultyTrend::Unknown, CLOSED)
        .is_none());
}

#[test]
fn oldest_xen11_first_by_default() {
    let sched = DrainScheduler::default();
    let v = vec![
        record(1, FindKind::Xen11, 100_000),
        record(2, FindKind::Xen11, 90_000),
        record(3, FindKind::Xen11, 95_000),
    ];
    // Journal order (oldest) wins whenever the trend is not rising.
    for trend in [
        DifficultyTrend::Unknown,
        DifficultyTrend::Falling,
        DifficultyTrend::Flat,
    ] {
        assert_eq!(sched.select_next(&v, trend, CLOSED).map(|r| r.id), Some(1));
    }
}

#[test]
fn rising_difficulty_drains_ascending_m_first() {
    let sched = DrainScheduler::default();
    let v = vec![
        record(1, FindKind::Xen11, 100_000),
        record(2, FindKind::Xen11, 90_000),
        record(3, FindKind::Xen11, 95_000),
    ];
    // Lowest m: closest to the rising floor.
    assert_eq!(
        sched
            .select_next(&v, DifficultyTrend::Rising, CLOSED)
            .map(|r| r.id),
        Some(2)
    );
}

#[test]
fn rising_difficulty_ties_break_oldest_first() {
    let sched = DrainScheduler::default();
    let v = vec![
        record(1, FindKind::Xen11, 90_000),
        record(2, FindKind::Xen11, 90_000),
    ];
    assert_eq!(
        sched
            .select_next(&v, DifficultyTrend::Rising, CLOSED)
            .map(|r| r.id),
        Some(1)
    );
}

#[test]
fn xuni_never_selected_while_the_window_is_closed() {
    let sched = DrainScheduler::default();
    let mut v = vec![record(1, FindKind::Xuni, 100_000)];
    assert!(sched
        .select_next(&v, DifficultyTrend::Unknown, CLOSED)
        .is_none());
    // ...but a XEN11 in the same backlog still drains.
    v.push(record(2, FindKind::Xen11, 100_000));
    assert_eq!(
        sched
            .select_next(&v, DifficultyTrend::Unknown, CLOSED)
            .map(|r| r.id),
        Some(2)
    );
}

#[test]
fn xuni_preempts_xen11_near_the_window_end() {
    let sched = DrainScheduler::default();
    let v = vec![
        record(1, FindKind::Xen11, 100_000),
        record(2, FindKind::Xuni, 100_000),
        record(3, FindKind::Xuni, 100_000),
    ];
    // 60 s left: inside the default 120 s preemption threshold. Oldest XUNI, ahead of XEN11.
    assert_eq!(
        sched
            .select_next(&v, DifficultyTrend::Unknown, window(true, 60_000))
            .map(|r| r.id),
        Some(2)
    );
}

#[test]
fn xuni_yields_to_xen11_while_the_window_end_is_far() {
    let sched = DrainScheduler::default();
    let v = vec![
        record(1, FindKind::Xen11, 100_000),
        record(2, FindKind::Xuni, 100_000),
    ];
    // 8 minutes of window left: the XEN11 backlog goes first.
    assert_eq!(
        sched
            .select_next(&v, DifficultyTrend::Unknown, window(true, 480_000))
            .map(|r| r.id),
        Some(1)
    );
    // With no XEN11 left, the XUNI drains even far from the end.
    let only_xuni = vec![record(2, FindKind::Xuni, 100_000)];
    assert_eq!(
        sched
            .select_next(&only_xuni, DifficultyTrend::Unknown, window(true, 480_000))
            .map(|r| r.id),
        Some(2)
    );
}

#[test]
fn select_next_is_pure_and_repeatable() {
    let sched = DrainScheduler::default();
    let v = vec![
        record(1, FindKind::Xen11, 100_000),
        record(2, FindKind::Xen11, 90_000),
    ];
    let a = sched.select_next(&v, DifficultyTrend::Rising, CLOSED);
    let b = sched.select_next(&v, DifficultyTrend::Rising, CLOSED);
    assert!(std::ptr::eq(a.expect("some"), b.expect("some")));
}

#[test]
fn rate_starts_at_1_and_doubles_per_healthy_round_trip_up_to_the_ceiling() {
    let mut s = DrainScheduler::default(); // start 1, max 4
    assert_eq!(s.rate_per_second(), 1.0);
    assert_eq!(s.submit_interval_ms(), 1000);
    s.on_healthy_round_trip();
    assert_eq!(s.rate_per_second(), 2.0);
    s.on_healthy_round_trip();
    assert_eq!(s.rate_per_second(), 4.0);
    s.on_healthy_round_trip();
    assert_eq!(s.rate_per_second(), 4.0); // ceiling: the drain_rate config
    assert_eq!(s.submit_interval_ms(), 250);
}

#[test]
fn throttle_halves_the_rate_down_to_the_floor() {
    let mut s = DrainScheduler::default();
    s.on_healthy_round_trip();
    s.on_healthy_round_trip(); // 4/s
    s.on_throttle();
    assert_eq!(s.rate_per_second(), 2.0);
    s.on_throttle();
    s.on_throttle();
    s.on_throttle();
    assert_eq!(s.rate_per_second(), 0.25); // floor
    assert_eq!(s.submit_interval_ms(), 4000);
}

#[test]
fn breaker_close_resets_to_the_start_rate() {
    let mut s = DrainScheduler::default();
    s.on_healthy_round_trip();
    s.on_healthy_round_trip();
    assert_eq!(s.rate_per_second(), 4.0);
    s.on_breaker_close();
    assert_eq!(s.rate_per_second(), 1.0);
}

#[test]
fn ceiling_is_configurable() {
    let mut s = DrainScheduler::new(DrainConfig {
        max_rate_per_s: 8.0,
        ..DrainConfig::default()
    });
    for _ in 0..6 {
        s.on_healthy_round_trip();
    }
    assert_eq!(s.rate_per_second(), 8.0);
}
