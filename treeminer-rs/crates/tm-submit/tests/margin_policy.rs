//! Port of `tests/unit/submit/test_margin_policy.cpp`.
//!
//! The ramp is grounded in the server's own adjustment rule: `manage_difficulty2.py` moves
//! difficulty at most +1000 KiB per 300 s tick, so one step of headroom per adjustment
//! period is the exact worst-case rise. These tests pin that arithmetic, the "healthy costs
//! nothing" guarantee, and the ceiling.

use tm_submit::margin::{compute_margin, MarginConfig, MarginInputs, MarginMode};

fn auto_config() -> MarginConfig {
    MarginConfig {
        mode: MarginMode::Auto,
        margin_kib: 1000,
        max_kib: 5000,
        adjust_period_ms: 300_000, // 5 minutes
    }
}

fn healthy() -> MarginInputs {
    MarginInputs::default()
}

fn outage(ms: i64) -> MarginInputs {
    MarginInputs {
        breaker_open: true,
        outage_ms: ms,
        backlog: 0,
    }
}

#[test]
fn off_mode_never_adds_headroom() {
    let cfg = MarginConfig::default(); // default is Off
    assert_eq!(cfg.mode, MarginMode::Off);
    assert_eq!(compute_margin(&cfg, &healthy()), 0);
    assert_eq!(compute_margin(&cfg, &outage(3_600_000)), 0);
    let backlogged = MarginInputs {
        backlog: 500,
        ..healthy()
    };
    assert_eq!(compute_margin(&cfg, &backlogged), 0);
}

#[test]
fn fixed_mode_is_constant_regardless_of_health() {
    let cfg = MarginConfig {
        mode: MarginMode::Fixed,
        margin_kib: 2500,
        ..MarginConfig::default()
    };
    assert_eq!(compute_margin(&cfg, &healthy()), 2500);
    assert_eq!(compute_margin(&cfg, &outage(9_999_999)), 2500);
}

#[test]
fn fixed_mode_is_not_clamped_by_the_auto_ceiling() {
    // max_kib governs the auto ramp only. An operator who writes an explicit constant gets
    // that constant; silently mining at a different cost would be worse.
    let cfg = MarginConfig {
        mode: MarginMode::Fixed,
        margin_kib: 9000,
        max_kib: 5000,
        ..MarginConfig::default()
    };
    assert_eq!(compute_margin(&cfg, &healthy()), 9000);
}

#[test]
fn auto_mode_costs_nothing_while_healthy_and_drained() {
    assert_eq!(compute_margin(&auto_config(), &healthy()), 0);
}

#[test]
fn auto_mode_buys_one_step_the_moment_the_breaker_opens() {
    // A find made right now cannot be submitted right now, so it is already exposed —
    // headroom applies immediately, not after the first adjustment period elapses.
    assert_eq!(compute_margin(&auto_config(), &outage(0)), 1000);
    assert_eq!(compute_margin(&auto_config(), &outage(1)), 1000);
}

#[test]
fn auto_ramp_adds_one_step_per_difficulty_adjustment_period() {
    let cfg = auto_config();
    assert_eq!(compute_margin(&cfg, &outage(299_999)), 1000); // < 1 period
    assert_eq!(compute_margin(&cfg, &outage(300_000)), 2000); // exactly 1 period
    assert_eq!(compute_margin(&cfg, &outage(600_000)), 3000); // 2 periods
    assert_eq!(compute_margin(&cfg, &outage(900_000)), 4000); // 3 periods
}

#[test]
fn auto_ramp_stops_at_the_ceiling() {
    let cfg = auto_config(); // max 5000
    assert_eq!(compute_margin(&cfg, &outage(1_200_000)), 5000); // 4 periods -> 5000
    assert_eq!(compute_margin(&cfg, &outage(1_500_000)), 5000); // would be 6000, capped
    assert_eq!(compute_margin(&cfg, &outage(86_400_000)), 5000); // a full day, still capped
}

#[test]
fn auto_mode_holds_headroom_while_a_backlog_drains_after_recovery() {
    // The server is reachable again but finds are still queued: difficulty may have climbed
    // during the outage, so newly mined finds still need headroom until the journal is clear.
    let mut recovering = MarginInputs {
        backlog: 1,
        ..healthy()
    };
    assert_eq!(compute_margin(&auto_config(), &recovering), 1000);
    // Outage duration does not inflate the margin once the breaker has closed — only the
    // backlog keeps it alive, at one step.
    recovering.backlog = 10_000;
    assert_eq!(compute_margin(&auto_config(), &recovering), 1000);
}

#[test]
fn auto_mode_returns_to_zero_once_the_backlog_clears() {
    assert_eq!(compute_margin(&auto_config(), &healthy()), 0);
}

#[test]
fn step_size_is_configurable_and_multiplies_through_the_ramp() {
    let cfg = MarginConfig {
        margin_kib: 500,
        max_kib: 100_000,
        ..auto_config()
    };
    assert_eq!(compute_margin(&cfg, &outage(0)), 500);
    assert_eq!(compute_margin(&cfg, &outage(300_000)), 1000);
    assert_eq!(compute_margin(&cfg, &outage(1_500_000)), 3000);
}

#[test]
fn zero_step_size_yields_no_headroom_even_while_degraded() {
    let cfg = MarginConfig {
        margin_kib: 0,
        ..auto_config()
    };
    assert_eq!(compute_margin(&cfg, &outage(600_000)), 0);
}

#[test]
fn a_huge_step_cannot_overflow_the_returned_headroom() {
    let cfg = MarginConfig {
        margin_kib: 4_000_000_000, // absurd, but must not wrap
        max_kib: 5000,
        ..auto_config()
    };
    assert_eq!(compute_margin(&cfg, &outage(3_000_000)), 5000);
}

#[test]
fn a_zero_adjust_period_does_not_divide_by_zero() {
    let cfg = MarginConfig {
        adjust_period_ms: 0,
        ..auto_config()
    };
    assert_eq!(compute_margin(&cfg, &outage(600_000)), 1000); // one step, no ramp
}

#[test]
fn margin_mode_parses_the_documented_spellings() {
    assert_eq!(MarginMode::parse("off"), Some(MarginMode::Off));
    assert_eq!(MarginMode::parse("AUTO"), Some(MarginMode::Auto));
    assert_eq!(MarginMode::parse("Fixed"), Some(MarginMode::Fixed));
    assert_eq!(MarginMode::parse("adaptive"), Some(MarginMode::Auto));
    assert_eq!(MarginMode::parse("none"), Some(MarginMode::Off));
    assert_eq!(MarginMode::parse("static"), Some(MarginMode::Fixed));
}

#[test]
fn an_unknown_margin_mode_is_rejected_never_defaulted() {
    // Silently falling back would mine at a memory cost the operator did not ask for.
    assert!(MarginMode::parse("aggressive").is_none());
    assert!(MarginMode::parse("").is_none());
    assert!(MarginMode::parse("1000").is_none());
}

#[test]
fn mode_round_trips_through_as_str() {
    assert_eq!(MarginMode::Off.as_str(), "off");
    assert_eq!(MarginMode::Fixed.as_str(), "fixed");
    assert_eq!(MarginMode::Auto.as_str(), "auto");
    assert_eq!(
        MarginMode::parse(MarginMode::Auto.as_str()),
        Some(MarginMode::Auto)
    );
}
