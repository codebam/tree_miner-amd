//! Outage-proof submission: the journal drain, its circuit breaker, and the response truth
//! table that decides what a server answer really means.
//!
//! Port of `src/submit/` from the C++ miner. The invariant the whole crate exists for:
//! **nothing is ever silently dropped**. Every path — a lying 200, a difficulty rejection, a
//! missed XUNI window, an unknown 4xx, a dead network — ends in a status persisted through
//! [`JournalAccess`], and everything non-terminal comes back for another attempt.
//!
//! The journal itself is behind the narrow [`JournalAccess`] trait rather than a concrete
//! type, so the SQLite store, a test fake, and any future backend are interchangeable.

pub mod breaker;
pub mod classifier;
pub mod clocktime;
pub mod drain;
pub mod http;
pub mod journal;
pub mod manager;
pub mod margin;
pub mod transport;

pub use breaker::{BreakerConfig, BreakerState, CircuitBreaker};
pub use classifier::{
    classify, extract_json_field, extract_json_message, parse_difficulty_hint,
    parse_retry_after_seconds, TRANSPORT_ERROR,
};
pub use clocktime::{iso_utc, parse_http_date_ms, xuni_window_at};
pub use drain::{DifficultyTrend, DrainConfig, DrainScheduler, XuniWindowState};
pub use http::HttpTransport;
pub use journal::{JournalAccess, JournalCounts, JournalError, JournalResult};
pub use manager::{
    Config, ConfirmBodyCheck, Metrics, StepResult, SubmissionManager,
};
pub use margin::{compute_margin, MarginConfig, MarginInputs, MarginMode};
pub use transport::{Transport, TransportResult};
