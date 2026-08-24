//! Self-confirmation against the public leaderboard: an *independent* answer to "did our
//! blocks actually land in the ledger?", separate from what `/verify` claimed.
//!
//! `GET https://xenblocks.io/v1/leaderboard` is undocumented and, as far as the audit of the
//! three reference miners found, used by no XenBlocks client in existence. Two properties
//! earn it a place here:
//!
//!   1. **It is a different, healthier route.** It is served by the TLS vhost on 443, not
//!      the port-80 API the miner otherwise depends on. Measured on this box: three
//!      consecutive `GET http://xenblocks.io/difficulty`, `/get_balance/<addr>` and
//!      `/get_super_blocks/<addr>` requests all timed out at 12 s while
//!      `https://xenblocks.io/v1/leaderboard` answered 200 in 0.19-1.2 s every time. So the
//!      per-account port-80 endpoints are *not* a usable fallback; this one is.
//!   2. **It carries per-account `blocks` and `superBlocks`.** A 200 from `/verify` can lie
//!      (see [`crate::classifier`]); a rising `blocks` count under our own account cannot.
//!
//! What this module is NOT: it is a diagnostic, never a submission path. Nothing here may
//! block or slow the drain loop, no failure here means anything about a find's fate, and a
//! failure is inert by construction — one request, no internal retry, and a cooldown that
//! makes a poll loop physically unable to storm the endpoint.
//!
//! ## Absence is the normal case
//!
//! The response carries the top 500 accounts only. A small miner is simply not in it. That
//! is [`AccountStanding::Unranked`] — not an error, not evidence that submissions are
//! failing, and deliberately a distinct variant from [`AccountStanding::Unavailable`] so no
//! caller can conflate "we are small" with "we could not ask".
//!
//! ## Case
//!
//! The reference server lowercases the account before storing it
//! (`repos/xenminer/gpage.py:382-384`) while our configured address is EIP-55 mixed case,
//! and the leaderboard as observed live returns mixed case for all 500 entries. Every
//! comparison here is therefore ASCII-case-insensitive; an exact match would silently never
//! find us, which reads identically to "not ranked" and would be the worst possible failure
//! mode for a confirmation signal.

use std::io::Read;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde_json::Value;

/// Path of the leaderboard on the TLS vhost.
pub const LEADERBOARD_PATH: &str = "/v1/leaderboard";

/// Hard cap on the body we will read and hand to the JSON parser.
///
/// The live response is ~114 KB for 500 miners. 1 MiB is ~9x that: enough headroom for the
/// list to grow or gain fields, small enough that a hostile or broken endpoint streaming
/// forever cannot exhaust memory on a mining box.
pub const MAX_LEADERBOARD_BYTES: usize = 1024 * 1024;

/// Floor on the interval between two fetches through one [`LeaderboardClient`].
///
/// The leaderboard moves on the order of minutes and this is a diagnostic; polling it
/// faster buys nothing and costs the operator (and the shared endpoint) bandwidth. Enforced
/// in the client, not merely documented, so a mis-wired UI refresh cannot turn into a
/// request storm.
pub const MIN_POLL_INTERVAL: Duration = Duration::from_secs(300);

/// Total per-request timeout. Generous relative to the 0.2-1.2 s observed live, because the
/// only cost of waiting is one background thread — but bounded, because this must never be
/// the thing that wedges a diagnostic thread forever.
pub const LEADERBOARD_TIMEOUT: Duration = Duration::from_secs(15);

/// Build the leaderboard URL from an arbitrary `rpc` base, or `None` when we cannot honestly
/// guess one.
///
/// Same conservatism as [`crate::http::derive_health_probe_url`], and for the same reason:
/// an operator who pointed the miner at their own server must not have us fabricate a URL
/// they never deployed — and here the failure would be worse than a wasted request, because
/// a *foreign* leaderboard would answer 200 with real-looking data about someone else's
/// network. Rules:
///   * only `http`/`https` bases are rewritten, and
///   * a base naming an explicit port is NOT rewritten: it identifies one specific service,
///     and 443 of that host is a different service.
///
/// The derived URL is always `https`, even from an `http` base. TLS on 443 is the route
/// whose health motivates this module at all; downgrading it to port 80 would land back on
/// the vhost we are trying to route around. The request carries no credentials.
pub fn derive_leaderboard_url(rpc_link: &str) -> Option<String> {
    let (scheme, rest) = rpc_link.trim().split_once("://")?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    // Authority ends at the first '/', '?' or '#'.
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    // Strip userinfo; a bare '@' with no host falls out as empty below.
    let host = authority.rsplit_once('@').map_or(authority, |(_, h)| h);
    if host.is_empty() {
        return None;
    }
    if let Some(close) = host.find(']') {
        // IPv6 literal: a port can only appear after the closing bracket.
        if !host.starts_with('[') || host[close + 1..].starts_with(':') {
            return None;
        }
    } else if host.contains(':') {
        return None; // explicit port
    }
    Some(format!("https://{host}{LEADERBOARD_PATH}"))
}

/// One account's row. Every numeric is [`Option`]: a field the server did not send, or sent
/// in a shape we do not understand, is `None` — never a fabricated `0`, which would read as
/// "confirmed zero blocks" and is exactly the lie this module exists to detect.
#[derive(Debug, Clone, PartialEq)]
pub struct LeaderboardEntry {
    /// As returned by the server, original case preserved. Compare with
    /// [`str::eq_ignore_ascii_case`], never `==`.
    pub account: String,
    pub rank: Option<u64>,
    /// The ledger's block count for this account — the self-confirmation signal.
    pub blocks: Option<u64>,
    pub super_blocks: Option<u64>,
    pub hash_rate: Option<f64>,
    /// Absent or `null` for ~15% of live entries; not an error.
    pub sol_address: Option<String>,
}

/// A parsed leaderboard response. Unknown top-level and per-entry fields are ignored, so a
/// server-side addition cannot break parsing.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Leaderboard {
    /// Network difficulty. Arrives as a JSON *string* live (`"difficulty":"100"`) while
    /// `blocks` arrives as a number — hence the lenient numeric handling throughout.
    pub difficulty: Option<u64>,
    pub total_blocks: Option<u64>,
    pub total_miners: Option<u64>,
    pub total_hash_rate: Option<f64>,
    /// In server order (rank ascending, as observed). Entries lacking an `account` string
    /// are dropped rather than failing the whole parse.
    pub entries: Vec<LeaderboardEntry>,
}

/// Where a given account stands. Three outcomes, modelled explicitly, because collapsing any
/// two of them loses the only information the caller actually wants.
#[derive(Debug, Clone, PartialEq)]
pub enum AccountStanding {
    /// We are in the listing. `blocks`/`super_blocks` are the ledger's own count.
    Ranked(LeaderboardEntry),
    /// The listing was fetched and parsed, and we are not in it. **Normal** for a small
    /// miner: the response only carries the top accounts. Says nothing about whether our
    /// submissions are landing.
    Unranked {
        /// How many accounts the listing carried (the top-N cutoff, whatever N is today).
        listed: usize,
        /// `blocks` of the last listed account: the bar we would have to clear to appear.
        cutoff_blocks: Option<u64>,
    },
    /// We could not ask. Network failure, non-200, oversized or malformed body, or the
    /// cooldown. Carries the reason for display; means nothing about the ledger.
    Unavailable(String),
}

impl AccountStanding {
    /// The row, when ranked. `None` for both other variants — do not use this to test for
    /// trouble; match the variant.
    pub fn entry(&self) -> Option<&LeaderboardEntry> {
        match self {
            Self::Ranked(entry) => Some(entry),
            _ => None,
        }
    }

    /// True only for [`AccountStanding::Unavailable`]. `Unranked` is a successful answer.
    pub fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable(_))
    }
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LeaderboardError {
    /// The request never produced a response (connect error, timeout, TLS, DNS).
    #[error("transport failure: {0}")]
    Transport(String),
    #[error("HTTP {0}")]
    Status(u16),
    #[error("response exceeds the {0}-byte parse cap")]
    TooLarge(usize),
    #[error("malformed leaderboard body: {0}")]
    Malformed(String),
    /// The cooldown has not elapsed. Not a failure of anything; ask again later.
    #[error("throttled: next poll allowed in {}s", .0.as_secs())]
    Throttled(Duration),
}

/// JSON number *or* numeric string to `u64`. The endpoint mixes both shapes (`difficulty` is
/// a string, `blocks` a number) and has no stated contract, so accept either everywhere
/// rather than pinning a shape it never promised. Floats are accepted because the large
/// token totals arrive in exponent form; non-finite and negative values are rejected.
fn lenient_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(n) => n.as_u64().or_else(|| {
            let f = n.as_f64()?;
            (f.is_finite() && f >= 0.0 && f <= u64::MAX as f64).then_some(f as u64)
        }),
        Value::String(s) => {
            let s = s.trim();
            s.parse::<u64>().ok().or_else(|| {
                let f = s.parse::<f64>().ok()?;
                (f.is_finite() && f >= 0.0 && f <= u64::MAX as f64).then_some(f as u64)
            })
        }
        _ => None,
    }
}

/// As [`lenient_u64`], for fields that are genuinely fractional (`hashRate`).
fn lenient_f64(value: &Value) -> Option<f64> {
    let f = match value {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse::<f64>().ok()?,
        _ => return None,
    };
    f.is_finite().then_some(f)
}

fn field_u64(object: &Value, key: &str) -> Option<u64> {
    lenient_u64(object.get(key)?)
}

/// Read at most `cap` bytes, and fail rather than truncate.
///
/// Truncating would hand the parser a prefix of valid JSON, which fails as "malformed" and
/// would have us blame the server for our own cap. We read one byte past the cap purely to
/// detect the overrun.
pub fn read_capped<R: Read>(reader: R, cap: usize) -> Result<String, LeaderboardError> {
    let mut buffer = Vec::new();
    reader
        .take(cap as u64 + 1)
        .read_to_end(&mut buffer)
        .map_err(|e| LeaderboardError::Transport(e.to_string()))?;
    if buffer.len() > cap {
        return Err(LeaderboardError::TooLarge(cap));
    }
    String::from_utf8(buffer).map_err(|_| LeaderboardError::Malformed("body is not UTF-8".into()))
}

impl Leaderboard {
    /// Parse a response body. Tolerant of unknown fields and of numerics arriving as strings;
    /// strict about exactly one thing — `miners` must be present and be an array, because
    /// without it the body is not a leaderboard at all and an empty `entries` would read as
    /// "nobody is mining".
    pub fn parse(body: &str) -> Result<Self, LeaderboardError> {
        let root: Value =
            serde_json::from_str(body).map_err(|e| LeaderboardError::Malformed(e.to_string()))?;
        let miners = root
            .get("miners")
            .and_then(Value::as_array)
            .ok_or_else(|| LeaderboardError::Malformed("no `miners` array".into()))?;
        let entries = miners
            .iter()
            .filter_map(|m| {
                Some(LeaderboardEntry {
                    account: m.get("account")?.as_str()?.to_string(),
                    rank: field_u64(m, "rank"),
                    blocks: field_u64(m, "blocks"),
                    super_blocks: field_u64(m, "superBlocks"),
                    hash_rate: m.get("hashRate").and_then(lenient_f64),
                    sol_address: m
                        .get("solAddress")
                        .and_then(Value::as_str)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string),
                })
            })
            .collect();
        Ok(Self {
            difficulty: field_u64(&root, "difficulty"),
            total_blocks: field_u64(&root, "totalBlocks"),
            total_miners: field_u64(&root, "totalMiners"),
            total_hash_rate: root.get("totalHashRate").and_then(lenient_f64),
            entries,
        })
    }

    /// Locate an account, case-insensitively. Surrounding whitespace on the needle is
    /// tolerated because config values arrive from a file an operator edits by hand.
    pub fn find(&self, account: &str) -> Option<&LeaderboardEntry> {
        let needle = account.trim();
        if needle.is_empty() {
            return None;
        }
        self.entries
            .iter()
            .find(|e| e.account.eq_ignore_ascii_case(needle))
    }

    /// [`Self::find`], widened into the three-way answer. Never yields
    /// [`AccountStanding::Unavailable`] — by the time you hold a `Leaderboard`, the fetch
    /// succeeded.
    pub fn standing(&self, account: &str) -> AccountStanding {
        match self.find(account) {
            Some(entry) => AccountStanding::Ranked(entry.clone()),
            None => AccountStanding::Unranked {
                listed: self.entries.len(),
                cutoff_blocks: self.entries.last().and_then(|e| e.blocks),
            },
        }
    }
}

/// A single-purpose HTTP client for the leaderboard.
///
/// Deliberately separate from [`crate::http::HttpTransport`]: it has its own connection pool
/// and its own timeout so that nothing it does can consume a slot or a moment belonging to
/// the submit path. `reqwest`'s *blocking* client, matching the rest of this crate — call it
/// from a dedicated thread, never from inside an async runtime.
pub struct LeaderboardClient {
    client: reqwest::blocking::Client,
    url: String,
    min_interval: Duration,
    /// Earliest instant at which the next request may be issued. Stamped *before* the
    /// request goes out, so a slow request cannot be joined by a second one.
    next_allowed: Mutex<Option<Instant>>,
}

impl LeaderboardClient {
    /// Point at an explicit URL. Use [`Self::for_rpc`] unless the operator named one.
    pub fn new(url: impl Into<String>) -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(LEADERBOARD_TIMEOUT)
                .build()?,
            url: url.into(),
            min_interval: MIN_POLL_INTERVAL,
            next_allowed: Mutex::new(None),
        })
    }

    /// Derive the URL from the configured `rpc` base. `Ok(None)` means we declined to guess
    /// — see [`derive_leaderboard_url`] — and the caller should simply not offer the
    /// feature, which is not a fault of any kind.
    pub fn for_rpc(rpc_link: &str) -> Result<Option<Self>, reqwest::Error> {
        match derive_leaderboard_url(rpc_link) {
            None => Ok(None),
            Some(url) => Self::new(url).map(Some),
        }
    }

    /// Override the cooldown. Intended for tests; production should keep
    /// [`MIN_POLL_INTERVAL`].
    pub fn with_min_interval(mut self, interval: Duration) -> Self {
        self.min_interval = interval;
        self
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    /// One request. No retry, ever: a failure is a failure until the next scheduled poll.
    pub fn fetch(&self) -> Result<Leaderboard, LeaderboardError> {
        {
            let mut next = self.next_allowed.lock();
            if let Some(at) = *next {
                let now = Instant::now();
                if now < at {
                    return Err(LeaderboardError::Throttled(at - now));
                }
            }
            *next = Some(Instant::now() + self.min_interval);
        }
        let response = self
            .client
            .get(&self.url)
            .timeout(LEADERBOARD_TIMEOUT)
            .send()
            .map_err(|e| LeaderboardError::Transport(e.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(LeaderboardError::Status(status.as_u16()));
        }
        let body = read_capped(response, MAX_LEADERBOARD_BYTES)?;
        Leaderboard::parse(&body)
    }

    /// Fetch and locate, collapsing every failure into
    /// [`AccountStanding::Unavailable`] — the whole point being that this call cannot fail
    /// in a way the caller has to handle.
    pub fn standing(&self, account: &str) -> AccountStanding {
        match self.fetch() {
            Ok(board) => board.standing(account),
            Err(e) => AccountStanding::Unavailable(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::derive_leaderboard_url as d;
    use super::*;

    #[test]
    fn derives_the_tls_leaderboard_from_a_plain_base() {
        assert_eq!(
            d("http://xenblocks.io").as_deref(),
            Some("https://xenblocks.io/v1/leaderboard")
        );
        assert_eq!(
            d("https://xenblocks.io/").as_deref(),
            Some("https://xenblocks.io/v1/leaderboard")
        );
        assert_eq!(
            d("http://xenblocks.io/api?x=1#f").as_deref(),
            Some("https://xenblocks.io/v1/leaderboard")
        );
        assert_eq!(
            d("http://user:pw@xenblocks.io").as_deref(),
            Some("https://xenblocks.io/v1/leaderboard")
        );
        assert_eq!(
            d("http://[2001:db8::1]").as_deref(),
            Some("https://[2001:db8::1]/v1/leaderboard")
        );
    }

    #[test]
    fn declines_to_guess_for_an_operators_own_server() {
        // An explicit port names one service; 443 of that host is a different one, and a
        // foreign leaderboard would answer 200 with plausible data about someone else.
        assert_eq!(d("http://localhost:8080"), None);
        assert_eq!(d("http://10.0.0.5:4000/verify"), None);
        assert_eq!(d("https://xenblocks.io:443"), None);
        assert_eq!(d("http://[2001:db8::1]:9000"), None);
        assert_eq!(d("ftp://xenblocks.io"), None);
        assert_eq!(d("xenblocks.io"), None);
        assert_eq!(d("http://"), None);
        assert_eq!(d("http:///v1/leaderboard"), None);
        assert_eq!(d(""), None);
    }

    #[test]
    fn for_rpc_declines_without_building_a_client() {
        assert!(LeaderboardClient::for_rpc("http://localhost:8080")
            .expect("client build")
            .is_none());
        let client = LeaderboardClient::for_rpc("http://xenblocks.io")
            .expect("client build")
            .expect("derivable");
        assert_eq!(client.url(), "https://xenblocks.io/v1/leaderboard");
    }

    #[test]
    fn lenient_numerics_accept_both_json_shapes() {
        assert_eq!(lenient_u64(&serde_json::json!(4790829)), Some(4790829));
        assert_eq!(lenient_u64(&serde_json::json!("9100")), Some(9100));
        assert_eq!(lenient_u64(&serde_json::json!(" 42 ")), Some(42));
        assert_eq!(lenient_u64(&serde_json::json!(5.806e21)), None); // > u64::MAX
        assert_eq!(lenient_u64(&serde_json::json!(1.0e3)), Some(1000));
        assert_eq!(lenient_u64(&serde_json::json!(-1)), None);
        assert_eq!(lenient_u64(&serde_json::json!(null)), None);
        assert_eq!(lenient_u64(&serde_json::json!("abc")), None);
        assert_eq!(lenient_f64(&serde_json::json!("100000.0")), Some(100000.0));
        assert_eq!(lenient_f64(&serde_json::json!(true)), None);
    }

    #[test]
    fn read_capped_rejects_rather_than_truncates() {
        let body = [b'x'; 100];
        assert_eq!(read_capped(&body[..], 100).expect("at cap").len(), 100);
        assert_eq!(
            read_capped(&body[..], 99),
            Err(LeaderboardError::TooLarge(99))
        );
    }
}
