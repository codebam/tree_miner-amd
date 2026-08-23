//! `Transport` over `reqwest`'s blocking client. Port of `src/submit/HttpTransport.{h,cpp}`
//! (which wrapped cpr).
//!
//! This is the only module in the crate that touches the network; everything above it is
//! pure and unit-testable.

use std::time::Duration;

use tm_core::FoundPayload;

use crate::transport::{Transport, TransportResult};

pub struct HttpTransport {
    client: reqwest::blocking::Client,
    rpc: String,
    worker: String,
    submit_timeout: Duration,
    get_timeout: Duration,
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
}
