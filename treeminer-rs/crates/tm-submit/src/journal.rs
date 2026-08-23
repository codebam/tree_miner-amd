//! The narrow slice of the journal contract the submitter needs.
//!
//! Deliberately NOT `tm_journal`'s concrete type: the submitter is written against this
//! trait so the SQLite journal, the in-memory test fake, and any future store are
//! interchangeable, and so the two crates can be developed independently. The integrator
//! implements `JournalAccess` for the real journal (a blanket impl over `&J` is provided,
//! so `Arc<J>`/`&J` both work).
//!
//! Every method returns `Result`: a journal that cannot be written is the one failure the
//! submitter must not paper over. `SubmissionManager` treats any `Err` as fatal, halts the
//! drain loop and fires its fatal callback — this replaces the C++ exception boundary
//! (`SubmissionManager.h`, security finding 8).

use tm_core::{Classification, FindKind, FindRecord};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct JournalError(String);

impl JournalError {
    pub fn new(what: impl Into<String>) -> Self {
        Self(what.into())
    }
}

/// Counters the submitter uses for the auto-margin backlog estimate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct JournalCounts {
    pub pending: usize,
    pub parked: usize,
    pub parked_difficulty: usize,
    pub parked_xuni: usize,
    pub quarantined: usize,
    pub acked_total: usize,
    pub dead_total: usize,
    pub accepted_unconfirmed: usize,
    pub permanently_invalid: usize,
    pub queued_xen11: usize,
    pub queued_xuni: usize,
}

pub type JournalResult<T> = Result<T, JournalError>;

/// `&self` throughout: the real journal owns a connection behind its own lock, and the
/// manager shares it with the mining threads.
pub trait JournalAccess {
    /// Oldest-first eligible work of one kind. Eligibility: status `Pending` and
    /// `next_attempt_at` null or <= `now_utc`. The LIMIT applies AFTER the kind filter —
    /// that is the whole point of the method: a single mixed slice lets either kind starve
    /// the other (a closed-window XUNI backlog hides a XEN11, a deep XEN11 backlog hides a
    /// XUNI whose window is closing).
    fn fetch_eligible_of_kind(
        &self,
        kind: FindKind,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>>;

    /// Oldest-first `AcceptedUnconfirmed` rows whose `next_attempt_at` is null or <=
    /// `now_utc`, so `/get_block` confirmation can be retried after a transient failure.
    fn fetch_awaiting_confirmation(
        &self,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>>;

    /// Persist the outcome of one attempt (status, reason, attempt bookkeeping, backoff
    /// time, http status/response, confirmation timestamp for `Acked`).
    fn record_attempt(
        &self,
        id: i64,
        classification: &Classification,
        http_status: Option<i32>,
        response_body: &str,
        next_attempt_at: Option<&str>,
        now_utc: &str,
    ) -> JournalResult<()>;

    /// `ParkedDifficulty` -> `Pending` for records with m >= `current_difficulty`. Returns
    /// the number un-parked.
    fn unpark_for_difficulty(&self, current_difficulty: u32) -> JournalResult<usize>;

    /// `ParkedXuniWindow` -> `Pending` for XUNI whose window budget remains; increments
    /// `xuni_windows_tried`; records exceeding `max_windows` go to `Dead`.
    fn unpark_xuni_for_window(&self, max_windows: i32) -> JournalResult<usize>;

    /// Difficulty observation log.
    fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> JournalResult<()>;

    fn counts(&self) -> JournalResult<JournalCounts>;
}

impl<J: JournalAccess + ?Sized> JournalAccess for &J {
    fn fetch_eligible_of_kind(
        &self,
        kind: FindKind,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        (**self).fetch_eligible_of_kind(kind, now_utc, limit)
    }
    fn fetch_awaiting_confirmation(
        &self,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        (**self).fetch_awaiting_confirmation(now_utc, limit)
    }
    fn record_attempt(
        &self,
        id: i64,
        classification: &Classification,
        http_status: Option<i32>,
        response_body: &str,
        next_attempt_at: Option<&str>,
        now_utc: &str,
    ) -> JournalResult<()> {
        (**self).record_attempt(id, classification, http_status, response_body, next_attempt_at, now_utc)
    }
    fn unpark_for_difficulty(&self, current_difficulty: u32) -> JournalResult<usize> {
        (**self).unpark_for_difficulty(current_difficulty)
    }
    fn unpark_xuni_for_window(&self, max_windows: i32) -> JournalResult<usize> {
        (**self).unpark_xuni_for_window(max_windows)
    }
    fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> JournalResult<()> {
        (**self).record_difficulty(difficulty, at_utc)
    }
    fn counts(&self) -> JournalResult<JournalCounts> {
        (**self).counts()
    }
}

impl<J: JournalAccess + ?Sized> JournalAccess for std::sync::Arc<J> {
    fn fetch_eligible_of_kind(
        &self,
        kind: FindKind,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        (**self).fetch_eligible_of_kind(kind, now_utc, limit)
    }
    fn fetch_awaiting_confirmation(
        &self,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        (**self).fetch_awaiting_confirmation(now_utc, limit)
    }
    fn record_attempt(
        &self,
        id: i64,
        classification: &Classification,
        http_status: Option<i32>,
        response_body: &str,
        next_attempt_at: Option<&str>,
        now_utc: &str,
    ) -> JournalResult<()> {
        (**self).record_attempt(id, classification, http_status, response_body, next_attempt_at, now_utc)
    }
    fn unpark_for_difficulty(&self, current_difficulty: u32) -> JournalResult<usize> {
        (**self).unpark_for_difficulty(current_difficulty)
    }
    fn unpark_xuni_for_window(&self, max_windows: i32) -> JournalResult<usize> {
        (**self).unpark_xuni_for_window(max_windows)
    }
    fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> JournalResult<()> {
        (**self).record_difficulty(difficulty, at_utc)
    }
    fn counts(&self) -> JournalResult<JournalCounts> {
        (**self).counts()
    }
}
