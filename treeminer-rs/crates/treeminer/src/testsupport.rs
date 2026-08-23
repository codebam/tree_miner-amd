//! Fakes the mining loop is exercised against.
//!
//! Compiled unconditionally so the integration tests in `tests/` can reach them; nothing in
//! the shipping binary refers to this module. The GPU fake is what lets the money-critical
//! behaviours — one payload per find at the batch's own memory cost, a XUNI kept across a
//! closing window, release-before-resize — be tested on a machine whose only card is busy
//! mining.

use std::sync::atomic::{AtomicI64, Ordering};

use parking_lot::Mutex;
use tm_core::batch::BatchSizeDecision;
use tm_core::{Classification, FindKind, FindRecord, FindStatus, FoundPayload};
use tm_gpu::{BatchOutcome, BatchRequest, GpuMatch};
use tm_journal::{Counts, Journal, JournalError, RecoveryStats, Result as JournalResult};

use crate::backend::{DeviceFacts, MiningBackend};

/// A journal that keeps every append in memory.
#[derive(Debug, Default)]
pub struct RecordingJournal {
    appended: Mutex<Vec<FoundPayload>>,
    next_id: AtomicI64,
}

impl RecordingJournal {
    pub fn appended(&self) -> Vec<FoundPayload> {
        self.appended.lock().clone()
    }
}

impl Journal for RecordingJournal {
    fn append(&self, payload: &FoundPayload) -> JournalResult<i64> {
        self.appended.lock().push(payload.clone());
        Ok(self.next_id.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn fetch_eligible(&self, _now_utc: &str, _limit: usize) -> JournalResult<Vec<FindRecord>> {
        Ok(Vec::new())
    }

    fn fetch_eligible_of_kind(
        &self,
        _kind: FindKind,
        _now_utc: &str,
        _limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        Ok(Vec::new())
    }

    fn fetch_awaiting_confirmation(
        &self,
        _now_utc: &str,
        _limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        Ok(Vec::new())
    }

    fn get_by_id(&self, _id: i64) -> JournalResult<Option<FindRecord>> {
        Ok(None)
    }

    fn record_attempt(
        &self,
        _id: i64,
        _classification: &Classification,
        _http_status: Option<i32>,
        _response_body: &str,
        _next_attempt_at: Option<&str>,
        _now_utc: &str,
    ) -> JournalResult<()> {
        Ok(())
    }

    fn unpark_for_difficulty(&self, _current_difficulty: u32) -> JournalResult<usize> {
        Ok(0)
    }

    fn unpark_xuni_for_window(&self, _max_windows: i32) -> JournalResult<usize> {
        Ok(0)
    }

    fn recover_on_startup(&self) -> JournalResult<RecoveryStats> {
        Ok(RecoveryStats::default())
    }

    fn record_difficulty(&self, _difficulty: u32, _at_utc: &str) -> JournalResult<()> {
        Ok(())
    }

    fn last_known_difficulty(&self) -> JournalResult<Option<u32>> {
        Ok(None)
    }

    fn counts(&self) -> JournalResult<Counts> {
        let pending = self.appended.lock().len();
        Ok(Counts {
            pending,
            queued_xen11: pending,
            ..Counts::default()
        })
    }
}

/// A journal whose every write fails — the disk-is-broken case.
#[derive(Debug)]
pub struct FailingJournal {
    reason: String,
}

impl FailingJournal {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }

    fn fail<T>(&self) -> JournalResult<T> {
        Err(JournalError::Contract(self.reason.clone()))
    }
}

impl Journal for FailingJournal {
    fn append(&self, _payload: &FoundPayload) -> JournalResult<i64> {
        self.fail()
    }

    fn fetch_eligible(&self, _now_utc: &str, _limit: usize) -> JournalResult<Vec<FindRecord>> {
        self.fail()
    }

    fn fetch_eligible_of_kind(
        &self,
        _kind: FindKind,
        _now_utc: &str,
        _limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        self.fail()
    }

    fn fetch_awaiting_confirmation(
        &self,
        _now_utc: &str,
        _limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        self.fail()
    }

    fn get_by_id(&self, _id: i64) -> JournalResult<Option<FindRecord>> {
        self.fail()
    }

    fn record_attempt(
        &self,
        _id: i64,
        _classification: &Classification,
        _http_status: Option<i32>,
        _response_body: &str,
        _next_attempt_at: Option<&str>,
        _now_utc: &str,
    ) -> JournalResult<()> {
        self.fail()
    }

    fn unpark_for_difficulty(&self, _current_difficulty: u32) -> JournalResult<usize> {
        self.fail()
    }

    fn unpark_xuni_for_window(&self, _max_windows: i32) -> JournalResult<usize> {
        self.fail()
    }

    fn recover_on_startup(&self) -> JournalResult<RecoveryStats> {
        self.fail()
    }

    fn record_difficulty(&self, _difficulty: u32, _at_utc: &str) -> JournalResult<()> {
        self.fail()
    }

    fn last_known_difficulty(&self) -> JournalResult<Option<u32>> {
        self.fail()
    }

    fn counts(&self) -> JournalResult<Counts> {
        self.fail()
    }
}

/// A record the fake journal can hand back; used by the submission wiring tests.
pub fn pending_record(id: i64, memory_cost: u32, kind: FindKind) -> FindRecord {
    FindRecord {
        id,
        status: FindStatus::Pending,
        ..FindRecord::new(FoundPayload {
            key: format!("{id:064x}"),
            hash_to_verify: format!("$argon2id$v=19$m={memory_cost},t=1,p=1$c2FsdA$ZGlnZXN0"),
            account: "0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc".to_owned(),
            kind,
            memory_cost,
            worker: "worker-1".to_owned(),
            attempts: 1,
            hashes_per_second: 1.0,
            found_at_utc: "2026-01-01T00:00:00Z".to_owned(),
        })
    }
}

/// What the fake GPU was asked to do, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendEvent {
    Release,
    Plan {
        difficulty: u32,
    },
    Run {
        difficulty: u32,
        batch_size: usize,
        gpu_first_blocks: bool,
        allow_xuni: bool,
    },
}

enum Scripted {
    Batch(Vec<GpuMatch>),
    Failure(String),
}

/// A GPU that runs whatever the test scripted. Exhausting the script fails the batch, so a
/// test can never spin the mining loop forever.
pub struct FakeBackend {
    facts: DeviceFacts,
    batch_size: usize,
    events: Mutex<Vec<BackendEvent>>,
    script: Mutex<std::collections::VecDeque<Scripted>>,
    on_complete: Mutex<Option<Box<dyn Fn() + Send>>>,
}

impl std::fmt::Debug for FakeBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeBackend")
            .field("device", &self.facts)
            .field("events", &self.events.lock())
            .finish()
    }
}

impl FakeBackend {
    /// `batch_size` is what `plan_batch_size` will select; 0 models an allocation failure.
    pub fn new(batch_size: usize) -> Self {
        Self {
            facts: DeviceFacts {
                index: 0,
                name: "Fake GPU".to_owned(),
                bus_id: 3,
                total_memory_bytes: 24 * 1024 * 1024 * 1024,
            },
            batch_size,
            events: Mutex::new(Vec::new()),
            script: Mutex::new(std::collections::VecDeque::new()),
            on_complete: Mutex::new(None),
        }
    }

    pub fn with_index(mut self, index: i32) -> Self {
        self.facts.index = index;
        self
    }

    /// Queue one successful batch returning `matches`.
    pub fn push_batch(&mut self, matches: Vec<GpuMatch>) {
        self.script.lock().push_back(Scripted::Batch(matches));
    }

    /// Queue one failing batch.
    pub fn push_failure(&mut self, error: impl Into<String>) {
        self.script
            .lock()
            .push_back(Scripted::Failure(error.into()));
    }

    /// Run after each scripted batch is handed back — the hook tests use to move the
    /// difficulty or close the XUNI window at exactly the wrong moment.
    pub fn on_batch_complete(&mut self, callback: impl Fn() + Send + 'static) {
        *self.on_complete.lock() = Some(Box::new(callback));
    }

    pub fn events(&self) -> Vec<BackendEvent> {
        self.events.lock().clone()
    }
}

impl MiningBackend for FakeBackend {
    fn device(&self) -> DeviceFacts {
        self.facts.clone()
    }

    fn release_buffers(&mut self) {
        self.events.lock().push(BackendEvent::Release);
    }

    fn plan_batch_size(
        &mut self,
        difficulty: u32,
        explicit_max_batch_size: usize,
        _streams_per_device: usize,
    ) -> Result<BatchSizeDecision, String> {
        self.events.lock().push(BackendEvent::Plan { difficulty });
        let selected = if explicit_max_batch_size > 0 {
            self.batch_size.min(explicit_max_batch_size)
        } else {
            self.batch_size
        };
        Ok(BatchSizeDecision {
            memory_limited_batch_size: self.batch_size,
            tuned_batch_size: self.batch_size,
            selected_batch_size: selected,
            explicit_limit_applied: explicit_max_batch_size > 0,
            tuned_default_applied: false,
        })
    }

    fn run_batch(&mut self, request: &BatchRequest<'_>) -> Result<BatchOutcome, String> {
        self.events.lock().push(BackendEvent::Run {
            difficulty: request.difficulty,
            batch_size: request.passwords.len(),
            gpu_first_blocks: request.gpu_first_blocks,
            allow_xuni: request.allow_xuni,
        });
        let scripted = self.script.lock().pop_front();
        let outcome = match scripted {
            Some(Scripted::Batch(matches)) => Ok(BatchOutcome {
                attempts: request.passwords.len(),
                gpu_first_blocks: request.gpu_first_blocks,
                matches,
                ..BatchOutcome::default()
            }),
            Some(Scripted::Failure(error)) => Err(error),
            None => Err("no scripted batch remains".to_owned()),
        };
        if let Some(callback) = self.on_complete.lock().as_ref() {
            callback();
        }
        outcome
    }
}
