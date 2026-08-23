//! Shared test doubles: the in-memory journal fake (port of `tests/unit/submit/FakeJournal.h`),
//! a scripted transport, and controllable clocks.
#![allow(dead_code)]

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use tm_core::{Classification, FindKind, FindRecord, FindStatus, FoundPayload};
use tm_submit::breaker::Clock;
use tm_submit::journal::{JournalAccess, JournalCounts, JournalError, JournalResult};
use tm_submit::transport::{Transport, TransportResult};

pub fn payload(key: &str, kind: FindKind, m: u32) -> FoundPayload {
    FoundPayload {
        key: key.to_string(),
        hash_to_verify: format!("$argon2id$v=19$m={m},t=1,p=1$saltsalt$XEN11digest"),
        account: "0x1111111111111111111111111111111111111111".to_string(),
        kind,
        memory_cost: m,
        worker: "w1".to_string(),
        attempts: 1000,
        hashes_per_second: 1234.5,
        found_at_utc: "2026-01-01T00:00:00Z".to_string(),
    }
}

pub fn record(id: i64, kind: FindKind, m: u32) -> FindRecord {
    let mut r = FindRecord::new(payload(&format!("k{id}"), kind, m));
    r.id = id;
    r
}

/// The realistic `/get_block` 200 body for a record: the reference server returns the stored
/// row itself, so key and hash_to_verify must be THIS record's values. The manager validates
/// that, so a fixture with a placeholder key would be rejected exactly like a fabricated 200.
pub fn block_row(p: &FoundPayload) -> String {
    format!(
        r#"{{"account": "{}", "block_id": 7, "created_at": "2026-01-01 00:00:00", "hash_to_verify": "{}", "key": "{}"}}"#,
        p.account, p.hash_to_verify, p.key
    )
}

pub const OK_200: &str = r#"{"message": "Hash verified successfully and block saved."}"#;
pub const DUP_400: &str = r#"{"message": "Block already exists, continue"}"#;

#[derive(Debug, Clone)]
pub struct AttemptLog {
    pub id: i64,
    pub classification: Classification,
    pub http_status: Option<i32>,
    pub body: String,
    pub next_attempt_at: Option<String>,
}

#[derive(Default)]
struct FakeState {
    records: Vec<FindRecord>,
    next_id: i64,
    attempts_recorded: Vec<AttemptLog>,
    unpark_difficulty_calls: Vec<u32>,
    unpark_xuni_calls: i32,
    difficulty_log: Vec<(u32, String)>,
}

#[derive(Default)]
pub struct FakeJournal {
    state: Mutex<FakeState>,
    /// When set, `record_attempt` fails like a broken SQLite volume would.
    pub fail_record_attempt: Mutex<bool>,
    pub throw_count: Mutex<i32>,
}

impl FakeJournal {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(FakeState {
                next_id: 1,
                ..FakeState::default()
            }),
            ..Self::default()
        }
    }

    pub fn throwing() -> Self {
        let j = Self::new();
        *j.fail_record_attempt.lock().expect("lock") = true;
        j
    }

    pub fn append(&self, payload: FoundPayload) -> i64 {
        let mut s = self.state.lock().expect("lock");
        if let Some(existing) = s.records.iter().find(|r| r.payload.key == payload.key) {
            return existing.id; // idempotent local capture
        }
        let mut r = FindRecord::new(payload);
        r.id = s.next_id;
        s.next_id += 1;
        let id = r.id;
        s.records.push(r);
        id
    }

    pub fn record(&self, id: i64) -> FindRecord {
        self.state
            .lock()
            .expect("lock")
            .records
            .iter()
            .find(|r| r.id == id)
            .cloned()
            .expect("record exists")
    }

    pub fn set_status(&self, id: i64, status: FindStatus) {
        let mut s = self.state.lock().expect("lock");
        if let Some(r) = s.records.iter_mut().find(|r| r.id == id) {
            r.status = status;
        }
    }

    pub fn attempts_recorded(&self) -> Vec<AttemptLog> {
        self.state.lock().expect("lock").attempts_recorded.clone()
    }

    pub fn unpark_difficulty_calls(&self) -> Vec<u32> {
        self.state.lock().expect("lock").unpark_difficulty_calls.clone()
    }

    pub fn unpark_xuni_calls(&self) -> i32 {
        self.state.lock().expect("lock").unpark_xuni_calls
    }

    pub fn difficulty_log(&self) -> Vec<(u32, String)> {
        self.state.lock().expect("lock").difficulty_log.clone()
    }

    pub fn throw_count(&self) -> i32 {
        *self.throw_count.lock().expect("lock")
    }
}

impl JournalAccess for FakeJournal {
    fn fetch_eligible_of_kind(
        &self,
        kind: FindKind,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        let s = self.state.lock().expect("lock");
        let mut out = Vec::new();
        for r in &s.records {
            // records are id-ordered == oldest-first
            if r.status != FindStatus::Pending || r.payload.kind != kind {
                continue;
            }
            if r.next_attempt_at.as_deref().is_some_and(|t| t > now_utc) {
                continue; // ISO-8601 sorts lexicographically
            }
            out.push(r.clone());
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
    }

    fn fetch_awaiting_confirmation(
        &self,
        now_utc: &str,
        limit: usize,
    ) -> JournalResult<Vec<FindRecord>> {
        let s = self.state.lock().expect("lock");
        let mut out = Vec::new();
        for r in &s.records {
            if r.status != FindStatus::AcceptedUnconfirmed {
                continue;
            }
            if r.next_attempt_at.as_deref().is_some_and(|t| t > now_utc) {
                continue;
            }
            out.push(r.clone());
            if out.len() >= limit {
                break;
            }
        }
        Ok(out)
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
        if *self.fail_record_attempt.lock().expect("lock") {
            *self.throw_count.lock().expect("lock") += 1;
            return Err(JournalError::new("disk I/O error (simulated)"));
        }
        let mut s = self.state.lock().expect("lock");
        if let Some(r) = s.records.iter_mut().find(|r| r.id == id) {
            r.status = classification.next_status;
            r.status_reason = classification.reason.clone();
            r.attempt_count += 1;
            r.next_attempt_at = next_attempt_at.map(str::to_string);
            r.last_attempt_at = Some(now_utc.to_string());
            r.last_http_status = http_status;
            r.last_response = response_body.to_string();
            if classification.next_status == FindStatus::Acked {
                r.confirmed_at = Some(now_utc.to_string());
            }
        }
        s.attempts_recorded.push(AttemptLog {
            id,
            classification: classification.clone(),
            http_status,
            body: response_body.to_string(),
            next_attempt_at: next_attempt_at.map(str::to_string),
        });
        Ok(())
    }

    fn unpark_for_difficulty(&self, current_difficulty: u32) -> JournalResult<usize> {
        let mut s = self.state.lock().expect("lock");
        let mut n = 0;
        for r in s.records.iter_mut() {
            if r.status == FindStatus::ParkedDifficulty
                && r.payload.memory_cost >= current_difficulty
            {
                r.status = FindStatus::Pending;
                r.next_attempt_at = None;
                n += 1;
            }
        }
        s.unpark_difficulty_calls.push(current_difficulty);
        Ok(n)
    }

    fn unpark_xuni_for_window(&self, max_windows: i32) -> JournalResult<usize> {
        let mut s = self.state.lock().expect("lock");
        let mut n = 0;
        for r in s.records.iter_mut() {
            if r.status != FindStatus::ParkedXuniWindow {
                continue;
            }
            if r.xuni_windows_tried >= max_windows {
                r.status = FindStatus::Dead;
                continue;
            }
            r.xuni_windows_tried += 1;
            r.status = FindStatus::Pending;
            r.next_attempt_at = None;
            n += 1;
        }
        s.unpark_xuni_calls += 1;
        Ok(n)
    }

    fn record_difficulty(&self, difficulty: u32, at_utc: &str) -> JournalResult<()> {
        self.state
            .lock()
            .expect("lock")
            .difficulty_log
            .push((difficulty, at_utc.to_string()));
        Ok(())
    }

    fn counts(&self) -> JournalResult<JournalCounts> {
        let s = self.state.lock().expect("lock");
        let mut c = JournalCounts::default();
        for r in &s.records {
            if matches!(
                r.status,
                FindStatus::Pending | FindStatus::AcceptedUnconfirmed
            ) {
                if r.payload.kind == FindKind::Xen11 {
                    c.queued_xen11 += 1;
                } else {
                    c.queued_xuni += 1;
                }
            }
            match r.status {
                FindStatus::Pending => c.pending += 1,
                FindStatus::ParkedDifficulty => {
                    c.parked += 1;
                    c.parked_difficulty += 1;
                }
                FindStatus::ParkedXuniWindow => {
                    c.parked += 1;
                    c.parked_xuni += 1;
                }
                FindStatus::Quarantined => c.quarantined += 1,
                FindStatus::Acked => c.acked_total += 1,
                FindStatus::Dead => c.dead_total += 1,
                FindStatus::AcceptedUnconfirmed => c.accepted_unconfirmed += 1,
                FindStatus::PermanentlyInvalid => c.permanently_invalid += 1,
                FindStatus::Submitting => {}
            }
        }
        Ok(c)
    }
}

pub fn ok(status: i32, body: &str) -> TransportResult {
    TransportResult::ok(status, body)
}

pub fn down() -> TransportResult {
    TransportResult::failed("connect refused")
}

#[derive(Default)]
struct TransportState {
    submit_queue: Vec<TransportResult>,
    confirm_queue: Vec<TransportResult>,
    difficulty_queue: Vec<TransportResult>,
    submitted_keys: Vec<String>,
    confirmed_keys: Vec<String>,
    difficulty_calls: i32,
}

/// Scripted transport: each call pops the head of its queue, and an empty queue means the
/// host is down (exactly like the C++ fake).
#[derive(Default)]
pub struct FakeTransport {
    state: Mutex<TransportState>,
}

impl FakeTransport {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn push_submit(&self, r: TransportResult) {
        self.state.lock().expect("lock").submit_queue.push(r);
    }
    pub fn push_confirm(&self, r: TransportResult) {
        self.state.lock().expect("lock").confirm_queue.push(r);
    }
    pub fn push_difficulty(&self, r: TransportResult) {
        self.state.lock().expect("lock").difficulty_queue.push(r);
    }
    pub fn submitted_keys(&self) -> Vec<String> {
        self.state.lock().expect("lock").submitted_keys.clone()
    }
    pub fn confirmed_keys(&self) -> Vec<String> {
        self.state.lock().expect("lock").confirmed_keys.clone()
    }
    pub fn difficulty_calls(&self) -> i32 {
        self.state.lock().expect("lock").difficulty_calls
    }
}

fn take(queue: &mut Vec<TransportResult>) -> TransportResult {
    if queue.is_empty() {
        return down();
    }
    queue.remove(0)
}

impl Transport for FakeTransport {
    fn submit(&self, payload: &FoundPayload) -> TransportResult {
        let mut s = self.state.lock().expect("lock");
        s.submitted_keys.push(payload.key.clone());
        take(&mut s.submit_queue)
    }
    fn confirm(&self, key: &str) -> TransportResult {
        let mut s = self.state.lock().expect("lock");
        s.confirmed_keys.push(key.to_string());
        take(&mut s.confirm_queue)
    }
    fn difficulty(&self) -> TransportResult {
        let mut s = self.state.lock().expect("lock");
        s.difficulty_calls += 1;
        take(&mut s.difficulty_queue)
    }
}

/// Controllable monotonic + wall clocks. Default wall is 2026-01-01T00:00:00Z (minute 0, so
/// the XUNI window is OPEN).
pub struct Clocks {
    pub mono: Arc<AtomicI64>,
    pub wall: Arc<AtomicI64>,
}

impl Default for Clocks {
    fn default() -> Self {
        Self {
            mono: Arc::new(AtomicI64::new(0)),
            wall: Arc::new(AtomicI64::new(1_767_225_600_000)),
        }
    }
}

impl Clocks {
    pub fn advance(&self, ms: i64) {
        self.mono.fetch_add(ms, Ordering::SeqCst);
        self.wall.fetch_add(ms, Ordering::SeqCst);
    }
    pub fn set_wall(&self, epoch_ms: i64) {
        self.wall.store(epoch_ms, Ordering::SeqCst);
    }
    pub fn wall_now(&self) -> i64 {
        self.wall.load(Ordering::SeqCst)
    }
    pub fn mono_clock(&self) -> Clock {
        let c = Arc::clone(&self.mono);
        Arc::new(move || c.load(Ordering::SeqCst))
    }
    pub fn wall_clock(&self) -> Clock {
        let c = Arc::clone(&self.wall);
        Arc::new(move || c.load(Ordering::SeqCst))
    }
}
