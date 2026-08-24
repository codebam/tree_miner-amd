//! Transport boundary for the submission layer. Port of `src/submit/ITransport.h`.
//!
//! An X1 migration replaces only the adapter behind this trait, never the journal or the
//! state machine.

use tm_core::FoundPayload;

/// Outcome of one transport round-trip. `transport_ok == false` means the request never
/// produced an HTTP response (connect error, timeout, DNS failure); `http_status`/`body` are
/// meaningless in that case and `error` describes the failure.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TransportResult {
    pub transport_ok: bool,
    pub http_status: i32,
    pub body: String,
    /// Raw `Retry-After` header, when present.
    pub retry_after: Option<String>,
    /// Raw HTTP `Date` header, for the server-clock offset.
    pub date_header: Option<String>,
    pub error: String,
}

impl TransportResult {
    pub fn ok(http_status: i32, body: impl Into<String>) -> Self {
        Self {
            transport_ok: true,
            http_status,
            body: body.into(),
            ..Self::default()
        }
    }

    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            transport_ok: false,
            error: error.into(),
            ..Self::default()
        }
    }
}

pub trait Transport {
    /// `POST /verify` with the immutable payload. Applies hard timeouts internally and never
    /// panics: every failure comes back as `transport_ok == false`.
    fn submit(&self, payload: &FoundPayload) -> TransportResult;

    /// `GET /get_block?key=<key>` — the confirmation lookup for `AcceptedUnconfirmed`
    /// (200 with the stored row, 404 when absent).
    fn confirm(&self, key: &str) -> TransportResult;

    /// `GET /difficulty` — difficulty observation (`{"difficulty": "<N>"}`, a JSON
    /// *string*). Also the breaker's FALLBACK health probe when the transport has no
    /// dedicated health route.
    fn difficulty(&self) -> TransportResult;

    /// Optional dedicated liveness route for the circuit-breaker probe, separate from
    /// `difficulty()`.
    ///
    /// `None` — the default — means "I have no route distinct from `difficulty()`", and the
    /// breaker probes `difficulty()` alone. Returning `Some` tells the manager that a
    /// failure here is worth a second opinion from `difficulty()` before it counts as an
    /// outage, and that a success here may need a separate difficulty harvest because the
    /// body need not carry one.
    ///
    /// Why this exists: on the live network `GET /difficulty` on port 80 is by far the
    /// flakiest route the miner touches (6 of 14 sampled requests timed out at 12 s while
    /// ports 4445/4447 answered every time). Probing it alone made the breaker open on
    /// ordinary server flakiness rather than on a genuine outage.
    fn health_probe(&self) -> Option<TransportResult> {
        None
    }
}

impl<T: Transport + ?Sized> Transport for &T {
    fn submit(&self, payload: &FoundPayload) -> TransportResult {
        (**self).submit(payload)
    }
    fn confirm(&self, key: &str) -> TransportResult {
        (**self).confirm(key)
    }
    fn difficulty(&self) -> TransportResult {
        (**self).difficulty()
    }
    fn health_probe(&self) -> Option<TransportResult> {
        (**self).health_probe()
    }
}

impl<T: Transport + ?Sized> Transport for std::sync::Arc<T> {
    fn submit(&self, payload: &FoundPayload) -> TransportResult {
        (**self).submit(payload)
    }
    fn confirm(&self, key: &str) -> TransportResult {
        (**self).confirm(key)
    }
    fn difficulty(&self) -> TransportResult {
        (**self).difficulty()
    }
    fn health_probe(&self) -> Option<TransportResult> {
        (**self).health_probe()
    }
}
