//! The adapter between the durable journal and the submitter's narrow view of it.
//!
//! `tm_submit::JournalAccess` and `tm_journal::Journal` are deliberately different traits —
//! the submitter must be usable against a fake store, and the journal must be usable without
//! the submitter. Both are foreign here, so the bridge is a newtype; the blanket impls in
//! `tm-submit` then make `Arc<JournalBridge>` a `JournalAccess` too.
//!
//! Every error crosses over intact. The submitter treats any journal error as fatal and
//! halts its drain, which is the correct response to a store that cannot record outcomes:
//! continuing would re-submit finds it can no longer remember the state of.

use std::sync::Arc;

use tm_core::{Classification, FindKind, FindRecord};
use tm_journal::Journal;
use tm_submit::{JournalAccess, JournalCounts, JournalError, JournalResult};

pub struct JournalBridge {
    journal: Arc<dyn Journal + Send + Sync>,
}

impl std::fmt::Debug for JournalBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JournalBridge").finish()
    }
}

impl JournalBridge {
    pub fn new(journal: Arc<dyn Journal + Send + Sync>) -> Self {
        Self { journal }
    }

    pub fn journal(&self) -> &Arc<dyn Journal + Send + Sync> {
        &self.journal
    }
}

fn cross(error: tm_journal::JournalError) -> JournalError {
    JournalError::new(error.to_string())
}

impl JournalAccess for JournalBridge {
    fn fetch_eligible_of_kind(
        &self,
        kind: FindKind,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        self.journal
            .fetch_eligible_of_kind(kind, now_utc, limit)
            .map_err(cross)
    }

    fn fetch_awaiting_confirmation(
        &self,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        self.journal
            .fetch_awaiting_confirmation(now_utc, limit)
            .map_err(cross)
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
        self.journal
            .record_attempt(
                id,
                classification,
                http_status,
                response_body,
                next_attempt_at,
                now_utc,
            )
            .map_err(cross)
    }

    fn unpark_for_difficulty(&self, current_difficulty: u32) -> JournalResult<usize> {
        self.journal
            .unpark_for_difficulty(current_difficulty)
            .map_err(cross)
    }

    fn unpark_xuni_for_window(&self, max_windows: i32) -> JournalResult<usize> {
        self.journal
            .unpark_xuni_for_window(max_windows)
            .map_err(cross)
    }

    fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> JournalResult<()> {
        self.journal
            .record_difficulty(difficulty, at_utc)
            .map_err(cross)
    }

    fn counts(&self) -> JournalResult<JournalCounts> {
        let counts = self.journal.counts().map_err(cross)?;
        Ok(JournalCounts {
            pending: counts.pending,
            parked: counts.parked,
            parked_difficulty: counts.parked_difficulty,
            parked_xuni: counts.parked_xuni,
            quarantined: counts.quarantined,
            acked_total: counts.acked_total,
            dead_total: counts.dead_total,
            accepted_unconfirmed: counts.accepted_unconfirmed,
            permanently_invalid: counts.permanently_invalid,
            queued_xen11: counts.queued_xen11,
            queued_xuni: counts.queued_xuni,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{FailingJournal, RecordingJournal};

    #[test]
    fn counts_cross_the_boundary_field_for_field() {
        let journal = Arc::new(RecordingJournal::default());
        journal
            .append(&crate::testsupport::pending_record(1, 1000, FindKind::Xen11).payload)
            .expect("append");
        let bridge = JournalBridge::new(journal);

        let counts = bridge.counts().expect("counts");
        assert_eq!(counts.pending, 1);
        assert_eq!(counts.queued_xen11, 1);
    }

    #[test]
    fn a_journal_error_reaches_the_submitter_as_fatal() {
        let bridge = Arc::new(JournalBridge::new(Arc::new(FailingJournal::new("disk gone"))));
        let error = bridge.counts().expect_err("must surface");
        assert!(error.to_string().contains("disk gone"));
    }
}
