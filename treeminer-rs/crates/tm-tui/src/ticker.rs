//! The one-line status ticker used in `logs` mode. Port of the progress-line block in
//! `src/main.cpp` (the `ConsoleLog::progress(stream.str())` call site).
//!
//! Shape is load-bearing: operators read this line at a glance and scripts grep it, so the
//! field order, separators and rounding are kept exactly as the C++ emitted them. Optional
//! segments stay absent rather than showing zero.

use crate::snapshot::TickerSnapshot;

const SEP: &str = "  \u{2022}  ";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const RESET: &str = "\x1b[0m";

/// Render the ticker. `color` adds the ANSI runs the C++ used for finds and breaker state;
/// the leading erase-line sequence is always present because the line is redrawn in place.
pub fn format_ticker(snapshot: &TickerSnapshot, color: bool) -> String {
    let paint = |code: &str, text: String| -> String {
        if color {
            format!("{code}{text}{RESET}")
        } else {
            text
        }
    };

    let mut out = String::from("\x1b[2K\r");
    let rate = (snapshot.gpu_hashrate + snapshot.cpu_hashrate) / 1000.0;
    out.push_str(&format!("{rate:.1} kH/s"));
    out.push_str(SEP);
    out.push_str(&format!(
        "{} GPU{}",
        snapshot.active_gpus,
        if snapshot.active_gpus == 1 { "" } else { "s" }
    ));
    out.push_str(SEP);
    out.push_str(&format_hashes(snapshot.total_hashes));
    out.push_str(SEP);
    out.push_str(&format_uptime(snapshot.uptime_seconds));

    if snapshot.stream_count > snapshot.active_gpus {
        out.push_str(SEP);
        out.push_str(&format!("{} streams", snapshot.stream_count));
    }
    if snapshot.cpu_workers > 0 {
        out.push_str(SEP);
        if snapshot.cpu_paused_for_difficulty {
            out.push_str("CPU idle");
        } else {
            out.push_str(&format!("CPU {}", snapshot.cpu_workers));
        }
    }
    if snapshot.superblocks > 0 {
        out.push_str(SEP);
        out.push_str(&paint(RED, format!("{} super", snapshot.superblocks)));
    }
    if snapshot.normal_blocks > 0 {
        out.push_str(SEP);
        out.push_str(&paint(
            GREEN,
            format!(
                "{} block{}",
                snapshot.normal_blocks,
                if snapshot.normal_blocks == 1 { "" } else { "s" }
            ),
        ));
    }
    if snapshot.xuni_blocks > 0 {
        out.push_str(SEP);
        out.push_str(&paint(YELLOW, format!("{} XUNI", snapshot.xuni_blocks)));
    }
    let queued = snapshot.queued_xnm + snapshot.queued_xuni;
    if queued > 0 {
        out.push_str(SEP);
        out.push_str(&format!("{queued} queued"));
    }
    if snapshot.accepted_unconfirmed > 0 {
        out.push_str(SEP);
        out.push_str(&format!("{} confirming", snapshot.accepted_unconfirmed));
    }
    if snapshot.confirmed > 0 {
        out.push_str(SEP);
        out.push_str(&format!("{} confirmed", snapshot.confirmed));
    }
    // Breaker state, not the live outage clock: HalfOpen must stay visible even while a
    // probe is in flight, otherwise the line flickers between DOWN and nothing.
    if snapshot.breaker_half_open {
        out.push_str(SEP);
        out.push_str(&paint(YELLOW, "net PROBE".to_string()));
    } else if snapshot.pool_down {
        out.push_str(SEP);
        out.push_str(&paint(RED, "pool DOWN".to_string()));
        if snapshot.outage_ms > 0 {
            let seconds = snapshot.outage_ms / 1000;
            out.push_str(&paint(RED, format!(" {}m{}s", seconds / 60, seconds % 60)));
        }
    }

    out.push_str(SEP);
    if snapshot.margin_kib > 0 {
        out.push_str(&format!("m {} (+{})", snapshot.difficulty, snapshot.margin_kib));
    } else {
        out.push_str(&format!("diff {}", snapshot.difficulty));
    }
    out.push_str(SEP);
    out.push_str(&snapshot.console_url);
    out
}

/// `1.5M hashes` / `12.3K hashes` / `42 hashes`.
fn format_hashes(total: u64) -> String {
    if total >= 1_000_000 {
        format!("{:.1}M hashes", total as f64 / 1_000_000.0)
    } else if total >= 1_000 {
        format!("{:.1}K hashes", total as f64 / 1_000.0)
    } else {
        format!("{total} hashes")
    }
}

/// `mm:ss`, prefixed with `h:` once an hour has passed.
fn format_uptime(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console::strip_ansi;

    fn plain(snapshot: &TickerSnapshot) -> String {
        strip_ansi(&format_ticker(snapshot, true))
    }

    #[test]
    fn zero_gpus_and_zero_hashrate_still_render_every_required_field() {
        let snapshot = TickerSnapshot {
            console_url: "http://127.0.0.1:8080/".into(),
            ..Default::default()
        };
        assert_eq!(
            plain(&snapshot),
            "0.0 kH/s  \u{2022}  0 GPUs  \u{2022}  0 hashes  \u{2022}  00:00  \u{2022}  diff 0  \u{2022}  http://127.0.0.1:8080/"
        );
    }

    #[test]
    fn singular_gpu_and_block_wording() {
        let snapshot = TickerSnapshot {
            gpu_hashrate: 12_345.0,
            active_gpus: 1,
            stream_count: 1,
            total_hashes: 1_500,
            uptime_seconds: 65,
            normal_blocks: 1,
            difficulty: 1_800_000,
            console_url: "http://192.168.1.10:8080/".into(),
            ..Default::default()
        };
        assert_eq!(
            plain(&snapshot),
            "12.3 kH/s  \u{2022}  1 GPU  \u{2022}  1.5K hashes  \u{2022}  01:05  \u{2022}  1 block  \u{2022}  diff 1800000  \u{2022}  http://192.168.1.10:8080/"
        );
    }

    #[test]
    fn full_line_with_every_optional_segment() {
        let snapshot = TickerSnapshot {
            gpu_hashrate: 900_000.0,
            cpu_hashrate: 100_000.0,
            active_gpus: 2,
            stream_count: 4,
            total_hashes: 12_300_000,
            uptime_seconds: 3 * 3600 + 4 * 60 + 5,
            cpu_workers: 8,
            cpu_paused_for_difficulty: false,
            superblocks: 1,
            normal_blocks: 7,
            xuni_blocks: 3,
            queued_xnm: 2,
            queued_xuni: 1,
            accepted_unconfirmed: 4,
            confirmed: 9,
            breaker_half_open: false,
            pool_down: true,
            outage_ms: 125_000,
            margin_kib: 512,
            difficulty: 1_800_000,
            console_url: "http://10.0.0.5:8080/".into(),
        };
        assert_eq!(
            plain(&snapshot),
            "1000.0 kH/s  \u{2022}  2 GPUs  \u{2022}  12.3M hashes  \u{2022}  3:04:05  \u{2022}  4 streams  \
\u{2022}  CPU 8  \u{2022}  1 super  \u{2022}  7 blocks  \u{2022}  3 XUNI  \u{2022}  3 queued  \
\u{2022}  4 confirming  \u{2022}  9 confirmed  \u{2022}  pool DOWN 2m5s  \u{2022}  m 1800000 (+512)  \
\u{2022}  http://10.0.0.5:8080/"
        );
    }

    #[test]
    fn half_open_breaker_wins_over_pool_down() {
        let snapshot = TickerSnapshot {
            breaker_half_open: true,
            pool_down: true,
            outage_ms: 60_000,
            ..Default::default()
        };
        let line = plain(&snapshot);
        assert!(line.contains("net PROBE"), "{line}");
        assert!(!line.contains("pool DOWN"), "{line}");
    }

    #[test]
    fn cpu_workers_paused_by_difficulty_read_as_idle() {
        let snapshot = TickerSnapshot { cpu_workers: 4, cpu_paused_for_difficulty: true, ..Default::default() };
        assert!(plain(&snapshot).contains("CPU idle"));
        let running = TickerSnapshot { cpu_workers: 4, ..Default::default() };
        assert!(plain(&running).contains("CPU 4"));
    }

    #[test]
    fn streams_segment_hidden_when_one_stream_per_gpu() {
        let snapshot = TickerSnapshot { active_gpus: 2, stream_count: 2, ..Default::default() };
        assert!(!plain(&snapshot).contains("streams"));
    }

    #[test]
    fn line_begins_with_the_erase_sequence_and_colours_only_when_asked() {
        let snapshot = TickerSnapshot { superblocks: 1, ..Default::default() };
        let colored = format_ticker(&snapshot, true);
        assert!(colored.starts_with("\x1b[2K\r"));
        assert!(colored.contains(RED));
        let mono = format_ticker(&snapshot, false);
        assert!(mono.starts_with("\x1b[2K\r"));
        assert!(!mono.contains(RED));
    }
}
