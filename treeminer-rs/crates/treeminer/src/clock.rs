//! The process-wide local UTC offset, captured once at startup.
//!
//! WHY THIS EXISTS AT ALL
//! The `time` crate refuses `UtcOffset::current_local_offset()` once the process has more
//! than one thread: on Unix the lookup goes through `localtime_r`, which reads the `TZ`
//! environment variable, and another thread calling `setenv` concurrently is undefined
//! behaviour. That refusal is unconditional and permanent — there is no later point in the
//! miner's life at which the offset can be obtained.
//!
//! So the offset is read here, from `main` before any thread is spawned, and cached. Every
//! later consumer (log lines, the status ticker, the TUI clock) reads the cached value and
//! gets local wall-clock time, which is what an operator staring at a rig actually wants.
//! If the lookup fails we fall back to UTC rather than guessing.

use std::sync::OnceLock;

use time::{OffsetDateTime, UtcOffset};

static LOCAL_OFFSET: OnceLock<UtcOffset> = OnceLock::new();

/// Capture the local UTC offset. **Call from `main` while the process is still
/// single-threaded**; every call after the first returns the already-captured value, so a
/// late call cannot corrupt the cached offset.
///
/// Returns [`UtcOffset::UTC`] if the platform lookup is unavailable or refused.
pub fn init_local_offset() -> UtcOffset {
    *LOCAL_OFFSET.get_or_init(|| UtcOffset::current_local_offset().unwrap_or(UtcOffset::UTC))
}

/// The captured offset, or UTC when [`init_local_offset`] was never called. Safe to call
/// from any thread.
pub fn local_offset() -> UtcOffset {
    LOCAL_OFFSET.get().copied().unwrap_or(UtcOffset::UTC)
}

/// True once [`init_local_offset`] has run.
pub fn local_offset_is_initialised() -> bool {
    LOCAL_OFFSET.get().is_some()
}

/// Now, in the captured local zone.
pub fn now_local() -> OffsetDateTime {
    OffsetDateTime::now_utc().to_offset(local_offset())
}

/// `HH:MM:SS` in local time — the ticker/console clock format.
pub fn clock_hms() -> String {
    let now = now_local();
    format!("{:02}:{:02}:{:02}", now.hour(), now.minute(), now.second())
}

/// `MM-DD HH:MM` in local time — the file logger's line prefix.
pub fn log_timestamp() -> String {
    let now = now_local();
    format!(
        "{:02}-{:02} {:02}:{:02}",
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offset_is_available_and_stable() {
        let first = init_local_offset();
        let second = init_local_offset();
        assert_eq!(first, second);
        assert_eq!(local_offset(), first);
        assert!(local_offset_is_initialised());
    }

    #[test]
    fn formats_use_the_captured_offset() {
        init_local_offset();
        let now = now_local();
        assert_eq!(now.offset(), local_offset());
        assert_eq!(clock_hms().len(), 8);
        assert_eq!(log_timestamp().len(), 11);
    }
}
