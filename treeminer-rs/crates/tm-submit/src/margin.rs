//! Difficulty-margin policy. Port of `src/treeminer/MarginPolicy.h` +
//! `src/submit/MarginPolicy.cpp`.
//!
//! WHY HEADROOM EXISTS
//! The server rejects a submission only when the m baked into the hash is strictly BELOW the
//! difficulty current at submit time, so a find is not bound to the difficulty it was mined
//! at: it stays valid for as long as network difficulty has not climbed past its m. Mining
//! at `difficulty + margin` buys exactly that tolerance.
//!
//! WHY THE STEP IS 1000 KiB PER 300 s
//! `manage_difficulty2.py` re-evaluates every 300 s and moves by at most +1000 KiB per tick,
//! so one step of headroom per adjustment period is the exact worst case, not a guess.
//!
//! WHY IT IS NOT ALWAYS ON
//! m IS the Argon2 memory cost, so headroom is paid for in hashrate. Auto mode holds the
//! margin at zero whenever the server is reachable and the journal is drained — no
//! healthy-state tax — and buys insurance only while finds are actually exposed.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginMode {
    /// Never add headroom (default — byte-for-byte the pre-margin behaviour).
    Off,
    /// Always add `margin_kib`, healthy or not.
    Fixed,
    /// Zero while healthy; ramps while the breaker is open or a backlog exists.
    Auto,
}

impl MarginMode {
    pub fn as_str(self) -> &'static str {
        match self {
            MarginMode::Off => "off",
            MarginMode::Fixed => "fixed",
            MarginMode::Auto => "auto",
        }
    }

    /// `"off" | "fixed" | "auto"` (case-insensitive, with the documented synonyms). `None`
    /// on anything else, so a typo in config.txt is reported rather than silently mining at
    /// an unintended memory cost.
    pub fn parse(text: &str) -> Option<Self> {
        match text.to_ascii_lowercase().as_str() {
            "off" | "none" | "disabled" => Some(MarginMode::Off),
            "fixed" | "static" | "constant" => Some(MarginMode::Fixed),
            "auto" | "adaptive" => Some(MarginMode::Auto),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MarginConfig {
    pub mode: MarginMode,
    /// Fixed mode: the constant headroom in KiB. Auto mode: the size of ONE escalation step.
    pub margin_kib: u32,
    /// Auto mode only: ceiling on the ramp. Fixed mode is taken at face value.
    pub max_kib: u32,
    /// Server difficulty adjustment period (`manage_difficulty2.py`: 300 s).
    pub adjust_period_ms: i64,
}

impl Default for MarginConfig {
    fn default() -> Self {
        Self {
            mode: MarginMode::Off,
            margin_kib: 1000,
            max_kib: 5000,
            adjust_period_ms: 300_000,
        }
    }
}

/// Observations the policy reacts to. All supplied by the caller; the policy owns no state.
#[derive(Debug, Clone, Copy, Default)]
pub struct MarginInputs {
    /// The `/verify` path is down (submissions cannot land right now).
    pub breaker_open: bool,
    /// How long it has been down; ignored unless `breaker_open`.
    pub outage_ms: i64,
    /// Finds journaled but not yet terminal (the at-risk population).
    pub backlog: usize,
}

/// Headroom in KiB to add to the mined memory cost.
///
/// Auto: 0 when healthy (breaker closed AND backlog empty). Otherwise one step immediately —
/// a find made right now is already at risk — plus one further step per elapsed adjustment
/// period of outage, capped at `max_kib`.
pub fn compute_margin(cfg: &MarginConfig, input: &MarginInputs) -> u32 {
    match cfg.mode {
        MarginMode::Off => 0,
        MarginMode::Fixed => cfg.margin_kib,
        MarginMode::Auto => {
            // Healthy and drained: no headroom, no hashrate tax. This is the common case and
            // it must cost exactly nothing.
            if !input.breaker_open && input.backlog == 0 {
                return 0;
            }
            let mut steps: u64 = 1;
            if input.breaker_open && cfg.adjust_period_ms > 0 && input.outage_ms > 0 {
                steps += (input.outage_ms / cfg.adjust_period_ms) as u64;
            }
            let margin = u64::from(cfg.margin_kib).saturating_mul(steps);
            margin.min(u64::from(cfg.max_kib)) as u32
        }
    }
}
