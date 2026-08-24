//! `Transport` over `reqwest`'s blocking client. Port of `src/submit/HttpTransport.{h,cpp}`
//! (which wrapped cpr).
//!
//! This is the only module in the crate that touches the network; everything above it is
//! pure and unit-testable.

use std::time::Duration;

use tm_core::FoundPayload;

use crate::transport::{Transport, TransportResult};

/// Port of the XenBlocks block-explorer API. Verified live: `GET /total_blocks` there
/// answers 200 with a 27-byte `{"total_blocks":N}` body, and stayed up across every sample
/// in which port 80 `/difficulty` timed out.
pub const HEALTH_PROBE_PORT: u16 = 4447;
/// Cheapest 200 on [`HEALTH_PROBE_PORT`]. Carries no difficulty — the manager harvests that
/// separately, and treats a failed harvest as a non-event.
pub const HEALTH_PROBE_PATH: &str = "/total_blocks";

/// Build the breaker's health-probe URL from an arbitrary `rpc` base, or `None` when we
/// cannot honestly guess one (the manager then probes `/difficulty` as before).
///
/// Rules, all deliberately conservative — a wrong guess here costs a wasted request on every
/// probe of an outage:
///   * only `http`/`https` bases are rewritten; anything else is left alone,
///   * a base that already names an explicit port is NOT rewritten. The operator pointed at
///     one specific service; silently probing a different port of their host would be us
///     inventing infrastructure they never deployed,
///   * the probe itself is issued over plain `http` even for an `https` base: port 4447 is
///     a plaintext service on the reference deployment, and the probe carries no secrets
///     (a `GET` with no credentials, whose only trusted output is "the host answered").
pub fn derive_health_probe_url(rpc_link: &str) -> Option<String> {
    let (scheme, rest) = rpc_link.trim().split_once("://")?;
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return None;
    }
    // Authority ends at the first '/', '?' or '#'.
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default();
    // Strip userinfo; a bare '@' with no host is malformed and falls out as empty below.
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
    Some(format!("http://{host}:{HEALTH_PROBE_PORT}{HEALTH_PROBE_PATH}"))
}

pub struct HttpTransport {
    client: reqwest::blocking::Client,
    rpc: String,
    worker: String,
    submit_timeout: Duration,
    get_timeout: Duration,
    /// `None` when the `rpc` base gave us nothing safe to derive; see
    /// [`derive_health_probe_url`].
    health_probe_url: Option<String>,
}

impl HttpTransport {
    /// `rpc_link` e.g. `"http://xenblocks.io"`; `worker` is the machine id sent in the
    /// `/verify` payload. Timeouts are hard totals per request.
    pub fn new(
        rpc_link: &str,
        worker: &str,
        submit_timeout_ms: u64,
        get_timeout_ms: u64,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_millis(submit_timeout_ms.max(get_timeout_ms)))
                .build()?,
            rpc: rpc_link.trim_end_matches('/').to_string(),
            worker: worker.to_string(),
            submit_timeout: Duration::from_millis(submit_timeout_ms),
            get_timeout: Duration::from_millis(get_timeout_ms),
            health_probe_url: derive_health_probe_url(rpc_link),
        })
    }

    fn finish(response: Result<reqwest::blocking::Response, reqwest::Error>) -> TransportResult {
        let response = match response {
            Ok(r) => r,
            Err(e) => return TransportResult::failed(format!("transport failure: {e}")),
        };
        let http_status = response.status().as_u16() as i32;
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        };
        let retry_after = header("retry-after");
        let date_header = header("date");
        // A body we cannot read is not a body: the classifier treats it as inconclusive and
        // retries, which is the correct answer for a truncated response.
        let body = response.text().unwrap_or_default();
        TransportResult {
            transport_ok: true,
            http_status,
            body,
            retry_after,
            date_header,
            error: String::new(),
        }
    }
}

impl Transport for HttpTransport {
    fn submit(&self, payload: &FoundPayload) -> TransportResult {
        // Field-for-field the upstream /verify payload: attempts and hashes_per_second are
        // transmitted as strings.
        let body = serde_json::json!({
            "hash_to_verify": payload.hash_to_verify,
            "key": payload.key,
            "account": payload.account,
            "attempts": payload.attempts.to_string(),
            "hashes_per_second": format!("{:.2}", payload.hashes_per_second),
            "worker": if self.worker.is_empty() { &payload.worker } else { &self.worker },
        });
        Self::finish(
            self.client
                .post(format!("{}/verify", self.rpc))
                .timeout(self.submit_timeout)
                .json(&body)
                .send(),
        )
    }

    fn confirm(&self, key: &str) -> TransportResult {
        // key is 64-hex — URL-safe by construction, no escaping needed.
        Self::finish(
            self.client
                .get(format!("{}/get_block?key={key}", self.rpc))
                .timeout(self.get_timeout)
                .send(),
        )
    }

    fn difficulty(&self) -> TransportResult {
        Self::finish(
            self.client
                .get(format!("{}/difficulty", self.rpc))
                .timeout(self.get_timeout)
                .send(),
        )
    }

    fn health_probe(&self) -> Option<TransportResult> {
        let url = self.health_probe_url.as_ref()?;
        Some(Self::finish(
            self.client.get(url).timeout(self.get_timeout).send(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::derive_health_probe_url as d;

    #[test]
    fn derives_the_explorer_port_from_a_plain_base() {
        assert_eq!(
            d("http://xenblocks.io").as_deref(),
            Some("http://xenblocks.io:4447/total_blocks")
        );
        // Trailing slash, path, query and fragment all stop at the authority.
        assert_eq!(
            d("https://xenblocks.io/").as_deref(),
            Some("http://xenblocks.io:4447/total_blocks")
        );
        assert_eq!(
            d("http://xenblocks.io/api/v1?x=1#f").as_deref(),
            Some("http://xenblocks.io:4447/total_blocks")
        );
        assert_eq!(
            d("http://user:pw@xenblocks.io").as_deref(),
            Some("http://xenblocks.io:4447/total_blocks")
        );
        assert_eq!(
            d("http://[2001:db8::1]").as_deref(),
            Some("http://[2001:db8::1]:4447/total_blocks")
        );
    }

    #[test]
    fn declines_to_guess_when_the_base_is_not_a_plain_default_port_host() {
        assert_eq!(d("http://localhost:8080"), None); // explicit port: operator's choice
        assert_eq!(d("https://xenblocks.io:443"), None);
        assert_eq!(d("http://[2001:db8::1]:9000"), None);
        assert_eq!(d("ftp://xenblocks.io"), None);
        assert_eq!(d("xenblocks.io"), None); // no scheme
        assert_eq!(d("http://"), None);
        assert_eq!(d("http:///difficulty"), None);
        assert_eq!(d(""), None);
    }
}
