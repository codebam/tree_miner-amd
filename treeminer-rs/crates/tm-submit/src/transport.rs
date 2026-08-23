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

    /// `GET /difficulty` — breaker probe plus difficulty observation
    /// (`{"difficulty": "<N>"}`, a JSON *string*).
    fn difficulty(&self) -> TransportResult;
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
}
