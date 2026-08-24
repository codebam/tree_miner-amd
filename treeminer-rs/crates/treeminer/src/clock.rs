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

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
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

// --- the server's clock -------------------------------------------------------------
//
// WHY A SECOND NOTION OF TIME EXISTS
// The XUNI window is checked by the SERVER, against the SERVER's own clock
// (`gpage.py:36-40`, `datetime.now()`), so every decision the miner makes about that window
// must be expressed in the server's frame — never in the operator's local zone. The two
// differ by a whole hour in most places, which happens to be invisible to a :55-:05 window,
// but by :30 or :45 in IST, NPT and ACST, where a local-clock gate mines XUNI the submitter
// will not send and sleeps through the window that was actually open.
//
// `tm_submit::SubmissionManager` learns the offset from the HTTP `Date` header of every
// response and exposes it as `server_clock_offset_ms()`. It is published here so the mining
// threads read the same number without holding a handle to the submitter.
//
// FALLBACK: until a response has been seen (and in `--test-fixed-diff` runs, which have no
// submitter at all) the offset is unknown and treated as zero, i.e. plain UTC. That is the
// correct fallback and not merely a convenient one: `xuni_window_at` is evaluated on UTC
// epoch milliseconds, so an unknown offset leaves the miner and the submitter agreeing
// exactly, which is the property this whole mechanism exists to guarantee. A learned offset
// then moves both of them together.

static SERVER_OFFSET_MS: AtomicI64 = AtomicI64::new(0);
static SERVER_OFFSET_KNOWN: AtomicBool = AtomicBool::new(false);

/// Publish the server-clock offset (server wall clock - local wall clock, ms). `None`
/// clears it back to "unknown", which reads as plain UTC.
pub fn set_server_offset_ms(offset: Option<i64>) {
    match offset {
        Some(ms) => {
            SERVER_OFFSET_MS.store(ms, Ordering::Relaxed);
            SERVER_OFFSET_KNOWN.store(true, Ordering::Release);
        }
        None => SERVER_OFFSET_KNOWN.store(false, Ordering::Release),
    }
}

/// The published offset, or `None` while no response has been observed.
pub fn server_offset_ms() -> Option<i64> {
    if SERVER_OFFSET_KNOWN.load(Ordering::Acquire) {
        Some(SERVER_OFFSET_MS.load(Ordering::Relaxed))
    } else {
        None
    }
}

/// Now, in the server's frame, as epoch milliseconds. Plain UTC until an offset is known.
pub fn now_server_ms() -> i64 {
    tm_submit::clocktime::now_wall_ms() + server_offset_ms().unwrap_or(0)
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
    fn the_server_offset_starts_unknown_and_round_trips() {
        // Serialised with the other offset test only by being distinct globals; these two
        // statics are touched nowhere else in the test binary.
        set_server_offset_ms(None);
        assert_eq!(server_offset_ms(), None, "unknown until a response is seen");
        let utc_now = tm_submit::clocktime::now_wall_ms();
        assert!(
            (now_server_ms() - utc_now).abs() < 5_000,
            "an unknown offset must read as plain UTC, so miner and submitter agree exactly"
        );

        set_server_offset_ms(Some(-90_000));
        assert_eq!(server_offset_ms(), Some(-90_000));
        let corrected = now_server_ms();
        assert!(
            (corrected - (tm_submit::clocktime::now_wall_ms() - 90_000)).abs() < 5_000,
            "a learned offset must move the miner's clock, not just the submitter's"
        );

        set_server_offset_ms(None);
        assert_eq!(server_offset_ms(), None);
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
