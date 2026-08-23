//! Pure time helpers shared by the submitter. Ported from the anonymous namespace of
//! `src/submit/SubmissionManager.cpp` so the two implementations produce byte-identical
//! timestamps (the journal stores them as text and compares them lexicographically).

use crate::drain::XuniWindowState;

const MS_PER_HOUR: i64 = 3600 * 1000;
const XUNI_OPEN_BEFORE_HOUR_MS: i64 = 5 * 60 * 1000; // :55
const XUNI_OPEN_AFTER_HOUR_MS: i64 = 5 * 60 * 1000; // :05

/// Howard Hinnant's civil-date algorithms (public-domain formulation), as in the C++ miner.
fn days_from_civil(mut y: i64, m: u32, d: u32) -> i64 {
    y -= i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (i64::from(m) + if m > 2 { -3 } else { 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(mut z: i64) -> (i64, u32, u32) {
    z += 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = i64::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    y += i64::from(m <= 2);
    (y, m, d)
}

pub fn floor_div(a: i64, b: i64) -> i64 {
    let mut q = a / b;
    if (a % b != 0) && ((a < 0) != (b < 0)) {
        q -= 1;
    }
    q
}

/// `1970-01-01T00:00:00Z`-style ISO-8601 UTC, second resolution.
pub fn iso_utc(epoch_ms: i64) -> String {
    let days = floor_div(epoch_ms, 86_400_000);
    let mut rem = epoch_ms - days * 86_400_000;
    let (y, mo, da) = civil_from_days(days);
    let h = rem / 3_600_000;
    rem %= 3_600_000;
    let mi = rem / 60_000;
    rem %= 60_000;
    let s = rem / 1000;
    format!("{y:04}-{mo:02}-{da:02}T{h:02}:{mi:02}:{s:02}Z")
}

fn month_from_name(mon: &str) -> u32 {
    const NAMES: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    NAMES.iter().position(|n| *n == mon).map_or(0, |i| i as u32 + 1)
}

/// Parse an IMF-fixdate HTTP `Date` header: `"Sun, 06 Nov 1994 08:49:37 GMT"`.
pub fn parse_http_date_ms(date_header: &str) -> Option<i64> {
    let rest = match date_header.find(',') {
        Some(comma) => &date_header[comma + 1..],
        None => date_header,
    };
    let mut fields = rest.split_ascii_whitespace();
    let day: u32 = fields.next()?.parse().ok()?;
    let mon = fields.next()?;
    let year: i64 = fields.next()?.parse().ok()?;
    let hms = fields.next()?;
    let tz = fields.next()?;
    if tz != "GMT" && tz != "UTC" {
        return None;
    }
    let month = month_from_name(mon);
    if month == 0 || !(1..=31).contains(&day) || year < 1970 {
        return None;
    }
    let b = hms.as_bytes();
    if b.len() != 8 || b[2] != b':' || b[5] != b':' {
        return None;
    }
    for i in [0usize, 1, 3, 4, 6, 7] {
        if !b[i].is_ascii_digit() {
            return None;
        }
    }
    let h = i64::from((b[0] - b'0') * 10 + (b[1] - b'0'));
    let mi = i64::from((b[3] - b'0') * 10 + (b[4] - b'0'));
    let s = i64::from((b[6] - b'0') * 10 + (b[7] - b'0'));
    if h > 23 || mi > 59 || s > 60 {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400_000 + (h * 3600 + mi * 60 + s) * 1000)
}

/// The XUNI :55-:05 window as seen at the given (server) wall time.
pub fn xuni_window_at(server_epoch_ms: i64) -> XuniWindowState {
    let into_hour = server_epoch_ms - floor_div(server_epoch_ms, MS_PER_HOUR) * MS_PER_HOUR;
    if into_hour >= MS_PER_HOUR - XUNI_OPEN_BEFORE_HOUR_MS {
        // :55 .. :59 — closes at :05 past the NEXT hour.
        XuniWindowState {
            open: true,
            ms_until_close: (MS_PER_HOUR - into_hour) + XUNI_OPEN_AFTER_HOUR_MS,
        }
    } else if into_hour < XUNI_OPEN_AFTER_HOUR_MS {
        // :00 .. :04 — closes at :05.
        XuniWindowState {
            open: true,
            ms_until_close: XUNI_OPEN_AFTER_HOUR_MS - into_hour,
        }
    } else {
        XuniWindowState::default()
    }
}

pub fn now_monotonic_ms() -> i64 {
    use std::sync::OnceLock;
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    let start = START.get_or_init(std::time::Instant::now);
    start.elapsed().as_millis() as i64
}

pub fn now_wall_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}
