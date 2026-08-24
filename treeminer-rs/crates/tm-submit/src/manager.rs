//! The single loop that drains the durable journal to the server. Port of
//! `src/submit/SubmissionManager.{h,cpp}`.
//!
//! One scheduling step (`run_once`, also driven directly by unit tests):
//!   breaker Open -> `GET /difficulty` probe when due (records difficulty, tracks the server
//!                   clock offset from the HTTP `Date` header); success arms HalfOpen.
//!   otherwise    -> fetch eligible per kind -> `DrainScheduler` picks a record (XUNI window
//!                   and difficulty-trend aware) -> breaker admission -> `transport.submit`
//!                   -> classify -> optional `/get_block` confirmation -> journal
//!                   `record_attempt` -> breaker/pacing updates -> difficulty hints out.
//!
//! Confirmation-aware acks: a 200 or a duplicate is `AcceptedUnconfirmed` until
//! `GET /get_block?key=` finds the row. A 404 after a 200 is the server's lying-200 (its
//! insert retries were exhausted): the record goes back to `Pending` and is resubmitted. If
//! the lookup itself is unavailable the record stays `AcceptedUnconfirmed` with a backed-off
//! `next_attempt_at` and is re-driven later — never silently presented as confirmed.
//!
//! A confirmation 200 is only trusted when its BODY proves it: the protocol runs over
//! plaintext HTTP, so any intermediary (captive portal, transparent proxy, hostile MITM)
//! could answer 200 to everything and permanently suppress resubmission of real finds. We
//! require a JSON body whose `key` is byte-equal to the record's key and, when present, a
//! `hash_to_verify` byte-equal to the record's immutable hash.
//!
//! Fatal-error boundary: any journal error halts the drain loop exactly once, fires the
//! fatal callback and leaves the manager inert. A submission layer that cannot touch its
//! journal must not spin. (In C++ this was a `catch (...)` around the step; in Rust the
//! journal returns `Result` and this is the `?` propagation target.)

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use tm_core::{Classification, FindKind, FindRecord, FindStatus};

use crate::breaker::{BreakerConfig, BreakerState, CircuitBreaker, Clock, Jitter};
use crate::classifier::{classify, extract_json_field, parse_difficulty_hint, parse_retry_after_seconds, TRANSPORT_ERROR};
use crate::clocktime::{iso_utc, now_monotonic_ms, now_wall_ms, xuni_window_at};
use crate::drain::{DifficultyTrend, DrainConfig, DrainScheduler};
use crate::journal::{JournalAccess, JournalError, JournalResult};
use crate::margin::{compute_margin, MarginConfig, MarginInputs, MarginMode};
use crate::transport::{Transport, TransportResult};

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub fetch_limit: usize,
    /// `AcceptedUnconfirmed` retry batch per step.
    pub confirm_fetch_limit: usize,
    /// Per-record: `base * 2^attempts`, capped.
    pub backoff_base_ms: i64,
    pub backoff_cap_ms: i64,
    /// Window budget before `Dead`.
    pub xuni_max_windows: i32,
    /// Thread wakeup granularity.
    pub idle_poll_ms: i64,
    pub breaker: BreakerConfig,
    pub drain: DrainConfig,
    /// Difficulty headroom baked into newly mined hashes. Default `Off`: the miner behaves
    /// exactly as it did before margins existed until an operator asks for insurance.
    pub margin: MarginConfig,
    /// How often the auto ramp is re-evaluated. Auto mode reads journal counts, so this is
    /// deliberately coarse — the ramp moves on a 300 s scale, not a 250 ms one.
    pub margin_eval_interval_ms: i64,
    /// Difficulty-transition quiesce: how long `/verify` stays paused after the observed
    /// difficulty CHANGES. Finds keep being journaled throughout — the only thing that
    /// pauses is the network round-trip. `0` disables it; values above
    /// [`QUIESCE_MAX_MS`] are clamped down to it.
    ///
    /// Why: difficulty steps every 300 s, and the miner and the server do not step at the
    /// same instant. Submitting across that boundary races — the operator's 20:12 logs show
    /// a burst of 401s that were purely the transition, each one costing a round-trip, a
    /// park and an un-park. Waiting a few seconds converts the whole burst into finds that
    /// simply sit in the journal a moment longer, which is what the journal is for.
    /// (`repos/xnminer-linux/core/supervisor.py:274-289` defers for `max(5, sample)` s for
    /// the same reason.)
    pub difficulty_quiesce_ms: i64,
}

/// Hard ceiling on [`Config::difficulty_quiesce_ms`]. A quiesce is a deliberate stall of the
/// one path that gets finds paid for, so it is bounded twice over: the deadline is absolute
/// (a monotonic timestamp computed once, never extended by the passage of time) and the
/// configured duration cannot exceed this no matter what an operator types. A stuck quiesce
/// is therefore not expressible.
pub const QUIESCE_MAX_MS: i64 = 60_000;

impl Default for Config {
    fn default() -> Self {
        Self {
            fetch_limit: 16,
            confirm_fetch_limit: 4,
            backoff_base_ms: 2000,
            backoff_cap_ms: 300_000,
            xuni_max_windows: 3,
            idle_poll_ms: 250,
            breaker: BreakerConfig::default(),
            drain: DrainConfig::default(),
            margin: MarginConfig::default(),
            margin_eval_interval_ms: 5000,
            difficulty_quiesce_ms: 5000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepResult {
    /// Nothing due (no eligible work / pacing gate / probe not due).
    Idle,
    /// Open: issued a `/difficulty` probe.
    Probed,
    /// Issued a `/verify` attempt (result recorded in the journal).
    Submitted,
    /// Eligible work exists but the breaker refused admission.
    BreakerBlocked,
    /// No submission was due, but confirmation retries were driven.
    ConfirmRetried,
    /// Submissions are paused for a difficulty transition. Distinct from `Idle` so the
    /// operator console (and the tests) can tell "nothing to do" from "deliberately
    /// holding". Confirmation retries still ran; only `/verify` is held.
    Quiescing,
}

/// How a `/get_block` 200 body relates to the record it is supposed to confirm. Anything but
/// `Confirmed` keeps the record `AcceptedUnconfirmed` with the normal backoff — never
/// `Acked` (the 200 proved nothing) and never `Pending` (only a real 404 proves the row is
/// absent; resubmitting on a garbled body would double-spend drain budget against a server
/// that may genuinely hold the row).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfirmBodyCheck {
    /// JSON object; `key` byte-equal; `hash_to_verify` absent or byte-equal.
    Confirmed,
    /// Not a JSON object, or no scalar `key` field — an untrusted 200.
    Malformed,
    /// Describes some OTHER row: not a confirmation of ours.
    KeyMismatch,
    /// Our key but a different stored hash — serious, logged at error level.
    HashMismatch,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Metrics {
    pub submitted: u64,
    /// `/verify` attempts for records tried before.
    pub resubmitted: u64,
    pub acked: u64,
    pub accepted_unconfirmed: u64,
    pub reconciled_via_get_block: u64,
    pub confirmation_retries: u64,
    pub lying_200_detected: u64,
    pub parked_difficulty: u64,
    pub parked_xuni: u64,
    pub quarantined: u64,
    pub permanently_invalid: u64,
    pub transport_failures: u64,
    pub probes: u64,
    /// Headroom ramp steps taken.
    pub margin_changes: u64,
    /// Difficulty transitions that armed a submission quiesce.
    pub difficulty_quiesces: u64,
    /// `/get_block` 200s whose body failed validation: malformed, wrong key, mismatched hash.
    pub confirm_body_rejected: u64,
    /// Journal failures caught at the step boundary; >0 means the loop halted.
    pub thread_loop_exceptions: u64,
}

pub type DifficultyCallback = Arc<dyn Fn(u32) + Send + Sync>;
pub type OutcomeCallback = Arc<dyn Fn(&FindRecord, &Classification, Option<i32>) + Send + Sync>;
pub type NetworkStateCallback = Arc<dyn Fn(BreakerState) + Send + Sync>;
/// Fired at most once when a journal failure halted the drain loop. Must not call `stop()`
/// on this manager (self-join deadlock) and should hand off anything heavy to another thread.
pub type FatalCallback = Arc<dyn Fn(&str) + Send + Sync>;
pub type MarginCallback = Arc<dyn Fn(u32) + Send + Sync>;

#[derive(Default)]
struct Callbacks {
    difficulty_hint: Option<DifficultyCallback>,
    outcome: Option<OutcomeCallback>,
    network_state: Option<NetworkStateCallback>,
    fatal: Option<FatalCallback>,
    margin: Option<MarginCallback>,
}

/// Deferred so no callback runs while the step lock is held.
enum Event {
    DifficultyHint(u32),
    Outcome(Box<FindRecord>, Box<Classification>, Option<i32>),
    NetworkState(BreakerState),
    Margin(u32),
}

struct StepState {
    breaker: CircuitBreaker,
    scheduler: DrainScheduler,
    next_submit_allowed_ms: i64,
    last_window_open: bool,
    last_margin_eval_ms: i64,
    margin_eval_started: bool,
}

#[derive(Default)]
struct Shared {
    last_difficulty: Option<u32>,
    trend: Option<DifficultyTrend>,
    server_offset_ms: Option<i64>,
    metrics: Metrics,
}

pub struct SubmissionManager<J: JournalAccess, T: Transport> {
    journal: J,
    transport: T,
    cfg: Config,
    mono: Clock,
    wall: Clock,
    step: Mutex<StepState>,
    shared: Mutex<Shared>,
    callbacks: Mutex<Callbacks>,
    margin_kib: AtomicU32,
    /// 0 = the `/verify` path is not open.
    outage_started_ms: AtomicI64,
    /// Latched the instant the breaker leaves Open, so a recovery log can still report how
    /// long the pool was down after the live clock has been reset.
    last_outage_span_ms: AtomicI64,
    /// Monotonic deadline before which `/verify` is held for a difficulty transition; 0 =
    /// not quiescing. An absolute instant rather than a countdown, and written from
    /// `observe_difficulty` (which the difficulty poller calls on its own thread), so the
    /// drain loop never blocks on it — it reads one atomic and returns.
    quiesce_until_ms: AtomicI64,
    fatal: AtomicBool,
    running: AtomicBool,
    /// Only used to make `notify_find_appended` cheap; the loop also polls.
    wake: (std::sync::Mutex<u64>, std::sync::Condvar),
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
    breaker_state_cache: AtomicU64,
}

impl<J: JournalAccess, T: Transport> SubmissionManager<J, T> {
    pub fn new(journal: J, transport: T) -> Self {
        Self::with_config(journal, transport, Config::default(), None, None, None)
    }

    /// `monotonic`/`wall` default to the system clocks; tests inject deterministic ones.
    pub fn with_config(
        journal: J,
        transport: T,
        cfg: Config,
        monotonic: Option<Clock>,
        wall: Option<Clock>,
        jitter: Option<Jitter>,
    ) -> Self {
        let mono: Clock = monotonic.unwrap_or_else(|| Arc::new(now_monotonic_ms));
        let wall: Clock = wall.unwrap_or_else(|| Arc::new(now_wall_ms));
        let breaker = CircuitBreaker::new(cfg.breaker, Arc::clone(&mono), jitter);
        Self {
            journal,
            transport,
            cfg,
            mono,
            wall,
            step: Mutex::new(StepState {
                breaker,
                scheduler: DrainScheduler::new(cfg.drain),
                next_submit_allowed_ms: 0,
                last_window_open: false,
                last_margin_eval_ms: 0,
                margin_eval_started: false,
            }),
            shared: Mutex::new(Shared::default()),
            callbacks: Mutex::new(Callbacks::default()),
            margin_kib: AtomicU32::new(0),
            outage_started_ms: AtomicI64::new(0),
            last_outage_span_ms: AtomicI64::new(0),
            quiesce_until_ms: AtomicI64::new(0),
            fatal: AtomicBool::new(false),
            running: AtomicBool::new(false),
            wake: (std::sync::Mutex::new(0), std::sync::Condvar::new()),
            thread: Mutex::new(None),
            breaker_state_cache: AtomicU64::new(0),
        }
    }

    // --- callbacks ---

    pub fn set_difficulty_hint_callback(&self, cb: DifficultyCallback) {
        self.callbacks.lock().difficulty_hint = Some(cb);
    }
    pub fn set_outcome_callback(&self, cb: OutcomeCallback) {
        self.callbacks.lock().outcome = Some(cb);
    }
    pub fn set_network_state_callback(&self, cb: NetworkStateCallback) {
        self.callbacks.lock().network_state = Some(cb);
    }
    pub fn set_fatal_callback(&self, cb: FatalCallback) {
        self.callbacks.lock().fatal = Some(cb);
    }
    pub fn set_margin_callback(&self, cb: MarginCallback) {
        self.callbacks.lock().margin = Some(cb);
    }

    // --- observable state ---

    pub fn metrics(&self) -> Metrics {
        self.shared.lock().metrics
    }

    pub fn breaker_state(&self) -> BreakerState {
        match self.breaker_state_cache.load(Ordering::Relaxed) {
            1 => BreakerState::Open,
            2 => BreakerState::HalfOpen,
            _ => BreakerState::Closed,
        }
    }

    pub fn drain_rate_per_second(&self) -> f64 {
        self.step.lock().scheduler.rate_per_second()
    }

    pub fn difficulty_trend(&self) -> DifficultyTrend {
        self.shared.lock().trend.unwrap_or(DifficultyTrend::Unknown)
    }

    pub fn last_observed_difficulty(&self) -> Option<u32> {
        self.shared.lock().last_difficulty
    }

    /// Offset = server wall clock - local wall clock, from HTTP `Date` headers. Unknown
    /// until the first dated response.
    pub fn server_clock_offset_ms(&self) -> Option<i64> {
        self.shared.lock().server_offset_ms
    }

    /// Headroom in KiB currently in effect. Safe to read from any thread.
    pub fn margin_in_effect(&self) -> u32 {
        self.margin_kib.load(Ordering::Relaxed)
    }

    /// Milliseconds the `/verify` path has been Open, or 0 when it is not.
    pub fn outage_duration_ms(&self) -> i64 {
        let started = self.outage_started_ms.load(Ordering::Relaxed);
        if started == 0 {
            return 0;
        }
        ((self.mono)() - started).max(0)
    }

    /// Span of the most recent Open period; survives HalfOpen/Closed.
    pub fn last_outage_span_ms(&self) -> i64 {
        self.last_outage_span_ms.load(Ordering::Relaxed)
    }

    /// Milliseconds of difficulty-transition quiesce still to run, or 0 when submissions are
    /// flowing. Bounded above by [`QUIESCE_MAX_MS`] by construction.
    pub fn quiesce_remaining_ms(&self) -> i64 {
        let until = self.quiesce_until_ms.load(Ordering::Relaxed);
        if until == 0 {
            return 0;
        }
        (until - (self.mono)()).max(0)
    }

    /// Arm the difficulty-transition quiesce. Idempotent and monotonic: concurrent observers
    /// can only ever agree on the LATEST deadline, and every candidate deadline is
    /// `now + <= QUIESCE_MAX_MS`, so no sequence of calls can push submissions out further
    /// than one clamped quiesce from the last transition.
    fn arm_quiesce(&self) {
        let hold = self.cfg.difficulty_quiesce_ms.clamp(0, QUIESCE_MAX_MS);
        if hold == 0 {
            return; // disabled
        }
        let until = (self.mono)() + hold;
        let previous = self.quiesce_until_ms.fetch_max(until, Ordering::Relaxed);
        if previous >= until {
            return; // an in-flight quiesce already covers this transition
        }
        self.shared.lock().metrics.difficulty_quiesces += 1;
        tracing::info!(
            hold_ms = hold,
            "difficulty transition — pausing submissions briefly; finds keep queueing"
        );
    }

    // --- difficulty integration ---

    /// Called by the difficulty poller too, so trend tracking sees every sample.
    pub fn observe_difficulty(&self, difficulty: u32) -> JournalResult<usize> {
        let (decreased, first_observation, changed) = {
            let mut shared = self.shared.lock();
            let previous = shared.last_difficulty;
            let mut decreased = false;
            let mut first = false;
            let changed = previous.is_some_and(|prev| prev != difficulty);
            match previous {
                Some(prev) => {
                    shared.trend = Some(if difficulty > prev {
                        DifficultyTrend::Rising
                    } else if difficulty < prev {
                        decreased = true;
                        DifficultyTrend::Falling
                    } else {
                        DifficultyTrend::Flat
                    });
                }
                None => first = true,
            }
            shared.last_difficulty = Some(difficulty);
            (decreased, first, changed)
        };
        // Only a real transition quiesces. The FIRST observation of a process must not: it
        // carries no information about the server stepping, and stalling a just-started
        // miner's backlog for nothing is the opposite of the point.
        if changed {
            self.arm_quiesce();
        }
        // A falling floor re-qualifies parked finds with m >= current.
        //
        // The first observation of a process must un-park too, even though there is no trend
        // to compare against: a restart begins with no last_difficulty, so gating purely on a
        // strict decrease left finds parked that were already valid again — they would wait
        // for some LATER decrease that may never come while difficulty trends upward. The
        // update is bounded to ParkedDifficulty rows with m >= current, so a no-op is free.
        if decreased || first_observation {
            let unparked = self.journal.unpark_for_difficulty(difficulty)?;
            tracing::info!(difficulty, unparked, "difficulty observation un-parked finds");
            return Ok(unparked);
        }
        Ok(0)
    }

    // --- threading ---

    pub fn notify_find_appended(&self) {
        let (lock, cv) = &self.wake;
        if let Ok(mut guard) = lock.lock() {
            *guard += 1;
        }
        cv.notify_all();
    }

    pub fn stop(&self) {
        // No early-out on "already not running": after a fatal error the loop clears
        // `running` itself, but the thread handle is still joinable.
        self.running.store(false, Ordering::SeqCst);
        self.notify_find_appended();
        let handle = self.thread.lock().take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }

    // --- the scheduling step ---

    /// One scheduling step. Never fails outward: this is the fatal-error boundary. After a
    /// fatal journal failure it is inert and returns `Idle`.
    pub fn run_once(&self) -> StepResult {
        if self.fatal.load(Ordering::SeqCst) {
            return StepResult::Idle;
        }
        let mut events: Vec<Event> = Vec::new();
        let outcome = {
            let mut step = self.step.lock();
            let r = self.run_step(&mut step, &mut events);
            self.breaker_state_cache.store(
                match step.breaker.state() {
                    BreakerState::Closed => 0,
                    BreakerState::Open => 1,
                    BreakerState::HalfOpen => 2,
                },
                Ordering::Relaxed,
            );
            r
        };
        match outcome {
            Ok(result) => {
                self.fire(events);
                result
            }
            Err(e) => {
                self.handle_fatal(&format!("submission step: {e}"));
                StepResult::Idle
            }
        }
    }

    fn fire(&self, events: Vec<Event>) {
        let (hint_cb, outcome_cb, net_cb, margin_cb) = {
            let cbs = self.callbacks.lock();
            (
                cbs.difficulty_hint.clone(),
                cbs.outcome.clone(),
                cbs.network_state.clone(),
                cbs.margin.clone(),
            )
        };
        for event in events {
            match event {
                Event::DifficultyHint(d) => {
                    if let Some(cb) = &hint_cb {
                        cb(d);
                    }
                }
                Event::Outcome(record, classification, http) => {
                    if let Some(cb) = &outcome_cb {
                        cb(&record, &classification, http);
                    }
                }
                Event::NetworkState(state) => {
                    if let Some(cb) = &net_cb {
                        cb(state);
                    }
                }
                Event::Margin(kib) => {
                    if let Some(cb) = &margin_cb {
                        cb(kib);
                    }
                }
            }
        }
    }

    fn handle_fatal(&self, what: &str) {
        if self.fatal.swap(true, Ordering::SeqCst) {
            return; // first failure wins
        }
        // A submission layer that cannot touch its journal must not spin: every further step
        // would fail the same way while looking "alive" to the operator.
        self.running.store(false, Ordering::SeqCst);
        self.shared.lock().metrics.thread_loop_exceptions += 1;
        tracing::error!(
            "FATAL | submission loop halted — journal or step failure, finds are still \
             durable on disk but will NOT drain until restart | {what}"
        );
        let cb = self.callbacks.lock().fatal.clone();
        if let Some(cb) = cb {
            cb(what);
        }
        self.notify_find_appended();
    }

    fn run_step(&self, step: &mut StepState, events: &mut Vec<Event>) -> JournalResult<StepResult> {
        // Headroom is re-evaluated first so the outage clock advances even on steps that do
        // no network work at all (an Open breaker whose probe is not yet due still ages it).
        self.update_margin(step, events)?;
        if step.breaker.state() == BreakerState::Open {
            // Open: probes only — no /verify traffic and no confirmation lookups either
            // (they target the same host; hammering /get_block during an outage helps nobody).
            return self.probe_step(step, events);
        }
        let submit_result = self.submit_step(step, events)?;
        // Confirmation retries run after the normal drain step and outside the drain-rate
        // budget, so a backlog of unconfirmed acks can never starve fresh submissions.
        let confirm_result = self.confirm_step(step, events)?;
        if matches!(submit_result, StepResult::Idle | StepResult::Quiescing)
            && confirm_result != StepResult::Idle
        {
            return Ok(confirm_result);
        }
        Ok(submit_result)
    }

    fn update_margin(&self, step: &mut StepState, events: &mut Vec<Event>) -> JournalResult<()> {
        let now = (self.mono)();

        // Outage clock: the /verify path being Open is what puts finds at risk. Tracked here
        // rather than in CircuitBreaker so the breaker stays a pure state machine.
        let open = step.breaker.state() == BreakerState::Open;
        if open {
            if self.outage_started_ms.load(Ordering::Relaxed) == 0 {
                self.outage_started_ms.store(now, Ordering::Relaxed);
            }
        } else {
            let started = self.outage_started_ms.load(Ordering::Relaxed);
            if started != 0 {
                self.last_outage_span_ms
                    .store((now - started).max(0), Ordering::Relaxed);
            }
            self.outage_started_ms.store(0, Ordering::Relaxed);
        }

        if self.cfg.margin.mode == MarginMode::Off {
            return Ok(()); // never touch the mine loop when the feature is off
        }

        if step.margin_eval_started
            && (now - step.last_margin_eval_ms) < self.cfg.margin_eval_interval_ms
        {
            return Ok(());
        }
        step.last_margin_eval_ms = now;
        step.margin_eval_started = true;

        let mut input = MarginInputs {
            breaker_open: open,
            outage_ms: if open {
                now - self.outage_started_ms.load(Ordering::Relaxed)
            } else {
                0
            },
            backlog: 0,
        };
        if self.cfg.margin.mode == MarginMode::Auto {
            // Backlog = everything journaled that has not reached a terminal state. Parked
            // and unconfirmed records count: they are finds we still owe the operator.
            let c = self.journal.counts()?;
            input.backlog = c.pending + c.parked + c.accepted_unconfirmed + c.quarantined;
        }

        let next = compute_margin(&self.cfg.margin, &input);
        if next == self.margin_kib.load(Ordering::Relaxed) {
            return Ok(());
        }
        self.margin_kib.store(next, Ordering::Relaxed);
        self.shared.lock().metrics.margin_changes += 1;
        events.push(Event::Margin(next));
        Ok(())
    }

    fn track_server_date(&self, r: &TransportResult) {
        if !r.transport_ok {
            return;
        }
        let Some(header) = r.date_header.as_deref() else {
            return;
        };
        if let Some(server_ms) = crate::clocktime::parse_http_date_ms(header) {
            self.shared.lock().server_offset_ms = Some(server_ms - (self.wall)());
        }
    }

    /// `Ok(true)` when the body actually carried a difficulty we could record. The health
    /// probe's body may legitimately carry none, which is why this reports rather than
    /// silently no-ops.
    fn handle_difficulty_body(&self, body: &str, events: &mut Vec<Event>) -> JournalResult<bool> {
        // The reference server answers {"difficulty": "<N>"} — a JSON string. The
        // leaderboard route embeds the same field in a much larger object, and
        // `extract_json_field` scans top-level keys, so both parse here.
        let Some(field) = extract_json_field(body, "difficulty") else {
            return Ok(false);
        };
        // Reuse the bounded digit parser rather than a second integer parse.
        let Some(d) = parse_difficulty_hint(&format!("m={field}")) else {
            return Ok(false);
        };
        self.observe_difficulty(d)?;
        self.journal.record_difficulty(d, &iso_utc((self.wall)()))?;
        events.push(Event::DifficultyHint(d));
        Ok(true)
    }

    fn backoff_time_iso(&self, attempt_count: i32, retry_after_s: Option<i64>) -> String {
        let mut delay = self.cfg.backoff_base_ms;
        match retry_after_s {
            Some(secs) => delay = secs * 1000,
            None => {
                let mut i = 0;
                while i < attempt_count && delay < self.cfg.backoff_cap_ms {
                    delay *= 2;
                    i += 1;
                }
            }
        }
        iso_utc((self.wall)() + delay.min(self.cfg.backoff_cap_ms))
    }

    /// Does a `/get_block` 200 body actually describe `record`?
    ///
    /// The stored row is `{block_id, hash_to_verify, key, account, created_at}`. Over
    /// plaintext HTTP a captive portal / transparent proxy / MITM can answer 200 to
    /// anything, so the body must positively identify OUR row: a JSON object with a scalar
    /// `key` byte-equal to ours (keys are lowercase 64-hex we generated ourselves and echo
    /// verbatim, so byte equality is the right test — normalizing would only widen what an
    /// attacker may return), and, when the row carries `hash_to_verify`, a byte-equal hash
    /// (the key is derived from the hash, so a mismatch means the server holds a DIFFERENT
    /// find under our key — credit theft or corruption, never ackable).
    pub fn confirmation_matches(record: &FindRecord, body: &str) -> ConfirmBodyCheck {
        let Some(key) = extract_json_field(body, "key") else {
            return ConfirmBodyCheck::Malformed;
        };
        if key != record.payload.key {
            return ConfirmBodyCheck::KeyMismatch;
        }
        if let Some(hash) = extract_json_field(body, "hash_to_verify") {
            if hash != record.payload.hash_to_verify {
                return ConfirmBodyCheck::HashMismatch;
            }
        }
        ConfirmBodyCheck::Confirmed
    }

    /// A probe response only counts as "the host is alive" when it is a real, non-empty 200.
    /// Same bar as everywhere else in this crate: an empty body is indistinguishable from a
    /// proxy failure.
    fn probe_response_healthy(r: &TransportResult) -> bool {
        r.transport_ok && r.http_status == 200 && !r.body.trim().is_empty()
    }

    /// Breaker health probe.
    ///
    /// The route matters. Measured on the live network, `GET /difficulty` on port 80 timed
    /// out on 6 of 14 requests while the explorer port answered every one — so probing
    /// `/difficulty` alone made the breaker open on ordinary flakiness of the single least
    /// reliable endpoint we touch, not on a real outage. A transport that offers
    /// [`Transport::health_probe`] is asked there FIRST; `/difficulty` remains the fallback,
    /// so the breaker only stays Open when BOTH routes are down.
    ///
    /// Difficulty is still harvested whenever it can be: the dedicated health route's body
    /// carries none, so a successful probe follows up with `/difficulty` opportunistically.
    /// That follow-up cannot un-heal the probe — health has already been proven, and a
    /// difficulty we failed to read is a thing the poller and the next 401 hint both supply.
    fn probe_step(&self, step: &mut StepState, events: &mut Vec<Event>) -> JournalResult<StepResult> {
        if !step.breaker.probe_due() {
            return Ok(StepResult::Idle);
        }
        self.shared.lock().metrics.probes += 1;

        let mut healthy = false;
        let mut difficulty_seen = false;
        let mut route = "/difficulty";
        if let Some(h) = self.transport.health_probe() {
            self.track_server_date(&h);
            if Self::probe_response_healthy(&h) {
                healthy = true;
                route = "health";
                difficulty_seen = self.handle_difficulty_body(&h.body, events)?;
            }
        }
        if !healthy || !difficulty_seen {
            let d = self.transport.difficulty();
            self.track_server_date(&d);
            if Self::probe_response_healthy(&d) {
                healthy = true;
                self.handle_difficulty_body(&d.body, events)?;
            }
        }

        if healthy {
            step.breaker.on_probe_success(); // HalfOpen: next step admits one real submission
            tracing::info!(route, "submissions probing — probe succeeded, one test submit next");
        } else {
            step.breaker.on_probe_failure();
        }
        events.push(Event::NetworkState(step.breaker.state()));
        Ok(StepResult::Probed)
    }

    fn submit_step(&self, step: &mut StepState, events: &mut Vec<Event>) -> JournalResult<StepResult> {
        let now_mono = (self.mono)();
        if now_mono < step.next_submit_allowed_ms {
            return Ok(StepResult::Idle);
        }

        // XUNI window, on the server's clock when we know the offset.
        let offset = self.shared.lock().server_offset_ms.unwrap_or(0);
        let window = xuni_window_at((self.wall)() + offset);
        if window.open && !step.last_window_open {
            // A window just opened: parked XUNI with remaining budget become Pending again.
            self.journal.unpark_xuni_for_window(self.cfg.xuni_max_windows)?;
        }
        step.last_window_open = window.open;

        // Difficulty-transition quiesce. Deliberately AFTER the window bookkeeping above, so
        // a quiesce that straddles :55 or :05 still records the transition and still un-parks
        // XUNI — holding submissions must never also hold the schedule that decides what is
        // submittable. Returning here is a plain early return: no sleeping, no lock held
        // across a wait, so the drain thread keeps polling at its normal cadence and picks
        // straight back up when the deadline passes. The breaker is untouched (a quiesce is
        // not a failure) and so is the pacing gate, which simply finds itself already due.
        if now_mono < self.quiesce_until_ms.load(Ordering::Relaxed) {
            return Ok(StepResult::Quiescing);
        }

        // Fetch per kind rather than taking one mixed oldest-first slice. A single LIMITed
        // slice lets either kind starve the other, and both directions are reachable:
        //   * XUNI is journaled Pending whenever it is found, including outside a window, but
        //     is not submittable then. `fetch_limit` such records ahead of a XEN11 backlog
        //     produced a slice with nothing selectable in it — the drain reported Idle and no
        //     XEN11 was submitted until the next window, against a healthy server.
        //   * Symmetrically, a XEN11 backlog deeper than `fetch_limit` could hide a XUNI
        //     whose window is closing — the one record that genuinely cannot wait.
        let now_iso = iso_utc((self.wall)());
        let mut eligible =
            self.journal
                .fetch_eligible_of_kind(FindKind::Xen11, &now_iso, self.cfg.fetch_limit)?;
        let mut xuni_pressure = false;
        if window.open {
            // Only worth asking while the window is open: outside it XUNI is never selectable.
            let xuni =
                self.journal
                    .fetch_eligible_of_kind(FindKind::Xuni, &now_iso, self.cfg.fetch_limit)?;
            xuni_pressure = !xuni.is_empty();
            eligible.extend(xuni);
        }
        step.breaker.set_xuni_pressure(xuni_pressure);
        if eligible.is_empty() {
            return Ok(StepResult::Idle);
        }

        let trend = self.difficulty_trend();
        let Some(rec) = step.scheduler.select_next(&eligible, trend, window).cloned() else {
            return Ok(StepResult::Idle);
        };
        if !step.breaker.try_admit() {
            return Ok(StepResult::BreakerBlocked);
        }

        let was_closed = step.breaker.state() == BreakerState::Closed;

        let res = self.transport.submit(&rec.payload);
        self.track_server_date(&res);
        let status = if res.transport_ok {
            res.http_status
        } else {
            TRANSPORT_ERROR
        };
        let mut c = classify(status, &res.body, rec.payload.kind, res.retry_after.as_deref());

        // Difficulty hint from a 401 body: update the cache without waiting for the poller.
        if let Some(hint) = c.server_difficulty_hint {
            self.observe_difficulty(hint)?;
            self.journal.record_difficulty(hint, &iso_utc((self.wall)()))?;
            events.push(Event::DifficultyHint(hint));
        }

        // Confirmation-aware acks: 200 and duplicates are only Acked once /get_block agrees —
        // and "agrees" means the 200 BODY describes this record, not merely that some
        // intermediary answered 200 over plaintext HTTP.
        let mut next_attempt: Option<String> = None;
        if c.needs_lookup_confirmation {
            let conf = self.transport.confirm(&rec.payload.key);
            self.track_server_date(&conf);
            let http_200 = conf.transport_ok && conf.http_status == 200;
            let body_check = if http_200 {
                Self::confirmation_matches(&rec, &conf.body)
            } else {
                ConfirmBodyCheck::Malformed // unused unless status == 200
            };
            if http_200 && body_check == ConfirmBodyCheck::Confirmed {
                c.next_status = FindStatus::Acked;
                c.needs_lookup_confirmation = false;
                c.reason += "; confirmed via /get_block (body matches key)";
                self.shared.lock().metrics.reconciled_via_get_block += 1;
            } else if http_200 {
                // A 200 whose body does not prove our row: stay AcceptedUnconfirmed with the
                // normal backoff and let the confirm step re-ask later.
                if body_check == ConfirmBodyCheck::HashMismatch {
                    tracing::error!(
                        key = %rec.payload.key,
                        "hash_to_verify MISMATCH on /get_block — server row differs from our \
                         immutable find; NOT acking (possible interception or corruption)"
                    );
                }
                c.reason += "; ";
                c.reason += confirm_reject_reason(body_check);
                next_attempt = Some(self.backoff_time_iso(rec.attempt_count, None));
                self.shared.lock().metrics.confirm_body_rejected += 1;
            } else if conf.transport_ok && conf.http_status == 404 {
                // The lying-200: the server said saved, the lookup says absent. Resubmit —
                // replay is idempotent thanks to the server's UNIQUE key.
                c.next_status = FindStatus::Pending;
                c.needs_lookup_confirmation = false;
                c.reason += "; /get_block says ABSENT — server 200 was not durable, resubmitting";
                next_attempt = Some(self.backoff_time_iso(rec.attempt_count, None));
                self.shared.lock().metrics.lying_200_detected += 1;
            } else {
                // Lookup unavailable: remain AcceptedUnconfirmed with a backed-off
                // next_attempt_at so the confirm step re-drives it later — never presented
                // as confirmed.
                c.reason += "; /get_block unavailable, remaining unconfirmed";
                next_attempt = Some(self.backoff_time_iso(rec.attempt_count, None));
            }
        }

        if c.next_status == FindStatus::Pending && next_attempt.is_none() {
            let retry_after_s = if status == 429 {
                res.retry_after.as_deref().and_then(parse_retry_after_seconds)
            } else {
                None
            };
            next_attempt = Some(self.backoff_time_iso(rec.attempt_count, retry_after_s));
        }

        log_submission(&rec, &c, &res);

        let http_status = if res.transport_ok {
            Some(res.http_status)
        } else {
            None
        };
        self.journal.record_attempt(
            rec.id,
            &c,
            http_status,
            &res.body,
            next_attempt.as_deref(),
            &iso_utc((self.wall)()),
        )?;

        // Breaker + adaptive pacing.
        let transport_failure = !res.transport_ok
            || res.http_status >= 500
            || res.http_status == 408
            || res.http_status == 425
            || res.body.trim().is_empty();
        let accepted = matches!(
            c.next_status,
            FindStatus::Acked | FindStatus::AcceptedUnconfirmed
        );
        if accepted {
            step.breaker.on_verify_success();
            if was_closed {
                step.scheduler.on_healthy_round_trip();
            } else {
                step.scheduler.on_breaker_close(); // recovery drain restarts at 1/s
            }
        } else if transport_failure {
            step.breaker.on_verify_transport_failure();
            step.scheduler.on_throttle();
        } else if res.http_status == 429 {
            step.breaker.on_verify_inconclusive();
            step.scheduler.on_throttle();
        } else {
            // Conclusive non-success (parked/quarantined/invalid): the round-trip itself was
            // healthy.
            step.breaker.on_verify_inconclusive();
            step.scheduler.on_healthy_round_trip();
        }
        events.push(Event::NetworkState(step.breaker.state()));

        step.next_submit_allowed_ms = now_mono + step.scheduler.submit_interval_ms();

        {
            let mut shared = self.shared.lock();
            shared.metrics.submitted += 1;
            if rec.attempt_count > 0 {
                shared.metrics.resubmitted += 1;
            }
            if !res.transport_ok {
                shared.metrics.transport_failures += 1;
            }
            match c.next_status {
                FindStatus::Acked => shared.metrics.acked += 1,
                FindStatus::AcceptedUnconfirmed => shared.metrics.accepted_unconfirmed += 1,
                FindStatus::ParkedDifficulty => shared.metrics.parked_difficulty += 1,
                FindStatus::ParkedXuniWindow => shared.metrics.parked_xuni += 1,
                FindStatus::Quarantined => shared.metrics.quarantined += 1,
                FindStatus::PermanentlyInvalid => shared.metrics.permanently_invalid += 1,
                _ => {}
            }
        }
        events.push(Event::Outcome(Box::new(rec), Box::new(c), http_status));
        Ok(StepResult::Submitted)
    }

    fn confirm_step(&self, step: &mut StepState, events: &mut Vec<Event>) -> JournalResult<StepResult> {
        // Re-drive AcceptedUnconfirmed rows whose confirmation lookup previously failed,
        // honoring the persisted next_attempt_at. Skipped while the breaker is Open —
        // including when this very step opened it.
        if step.breaker.state() == BreakerState::Open {
            return Ok(StepResult::Idle);
        }
        let batch = self
            .journal
            .fetch_awaiting_confirmation(&iso_utc((self.wall)()), self.cfg.confirm_fetch_limit)?;
        if batch.is_empty() {
            return Ok(StepResult::Idle);
        }
        let mut any = false;
        for rec in batch {
            let conf = self.transport.confirm(&rec.payload.key);
            self.track_server_date(&conf);

            let http_200 = conf.transport_ok && conf.http_status == 200;
            let body_check = if http_200 {
                Self::confirmation_matches(&rec, &conf.body)
            } else {
                ConfirmBodyCheck::Malformed // unused unless status == 200
            };
            let mut next_attempt: Option<String> = None;
            let c = if http_200 && body_check == ConfirmBodyCheck::Confirmed {
                let mut shared = self.shared.lock();
                shared.metrics.acked += 1;
                shared.metrics.reconciled_via_get_block += 1;
                Classification {
                    next_status: FindStatus::Acked,
                    server_difficulty_hint: None,
                    needs_lookup_confirmation: false,
                    reason: "confirmed via /get_block (retry, body matches key)".to_string(),
                }
            } else if http_200 {
                // Same rule as the initial confirmation: an unproven 200 keeps the record
                // AcceptedUnconfirmed with per-record backoff — never Acked, never demoted.
                if body_check == ConfirmBodyCheck::HashMismatch {
                    tracing::error!(
                        key = %rec.payload.key,
                        "hash_to_verify MISMATCH on /get_block — server row differs from our \
                         immutable find; NOT acking (possible interception or corruption)"
                    );
                }
                next_attempt = Some(self.backoff_time_iso(rec.attempt_count, None));
                self.shared.lock().metrics.confirm_body_rejected += 1;
                Classification {
                    next_status: FindStatus::AcceptedUnconfirmed,
                    server_difficulty_hint: None,
                    needs_lookup_confirmation: false,
                    reason: confirm_reject_reason(body_check).to_string(),
                }
            } else if conf.transport_ok && conf.http_status == 404 {
                // The lying-200, caught on retry: the row never became durable server-side.
                next_attempt = Some(self.backoff_time_iso(rec.attempt_count, None));
                self.shared.lock().metrics.lying_200_detected += 1;
                Classification {
                    next_status: FindStatus::Pending,
                    server_difficulty_hint: None,
                    needs_lookup_confirmation: false,
                    reason: "/get_block says ABSENT — server 200 was not durable, resubmitting"
                        .to_string(),
                }
            } else {
                // Still unavailable (transport failure, 5xx, unexpected schema): stay
                // AcceptedUnconfirmed and push the retry out with per-record backoff.
                next_attempt = Some(self.backoff_time_iso(rec.attempt_count, None));
                Classification {
                    next_status: FindStatus::AcceptedUnconfirmed,
                    server_difficulty_hint: None,
                    needs_lookup_confirmation: false,
                    reason: "/get_block unavailable, remaining unconfirmed (retry backoff)"
                        .to_string(),
                }
            };

            let http_status = if conf.transport_ok {
                Some(conf.http_status)
            } else {
                None
            };
            self.journal.record_attempt(
                rec.id,
                &c,
                http_status,
                &conf.body,
                next_attempt.as_deref(),
                &iso_utc((self.wall)()),
            )?;
            tracing::info!(
                id = rec.id,
                attempt = rec.attempt_count + 1,
                status = c.next_status.as_str(),
                reason = %c.reason,
                "confirmation lookup"
            );
            self.shared.lock().metrics.confirmation_retries += 1;
            let transport_ok = conf.transport_ok;
            events.push(Event::Outcome(Box::new(rec), Box::new(c), http_status));
            any = true;
            if !transport_ok {
                break; // the host looks down — don't hammer it with the rest of the batch
            }
        }
        Ok(if any {
            StepResult::ConfirmRetried
        } else {
            StepResult::Idle
        })
    }
}

impl<J, T> SubmissionManager<J, T>
where
    J: JournalAccess + Send + Sync + 'static,
    T: Transport + Send + Sync + 'static,
{
    /// Start the drain thread. Idempotent; pair with `stop()`.
    pub fn start(self: &Arc<Self>) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let me = Arc::clone(self);
        let handle = std::thread::spawn(move || me.thread_loop());
        *self.thread.lock() = Some(handle);
    }

    fn thread_loop(&self) {
        while self.running.load(Ordering::SeqCst) {
            let r = self.run_once();
            let wait_ms = if r == StepResult::Idle {
                self.cfg.idle_poll_ms
            } else {
                self.cfg.idle_poll_ms.min(50)
            };
            let (lock, cv) = &self.wake;
            let Ok(guard) = lock.lock() else { return };
            let _unused = cv
                .wait_timeout(guard, std::time::Duration::from_millis(wait_ms.max(0) as u64));
        }
    }
}

impl<J: JournalAccess, T: Transport> Drop for SubmissionManager<J, T> {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        self.notify_find_appended(); // don't make the join wait out a full idle poll
        let handle = self.thread.lock().take();
        if let Some(h) = handle {
            let _ = h.join();
        }
    }
}

/// One reason fragment per rejected-body case, appended to the record's `status_reason` so
/// the journal shows WHY a 200 was not trusted.
fn confirm_reject_reason(check: ConfirmBodyCheck) -> &'static str {
    match check {
        ConfirmBodyCheck::KeyMismatch => {
            "/get_block 200 body describes a different key — not a confirmation of this \
             record, remaining unconfirmed"
        }
        ConfirmBodyCheck::HashMismatch => {
            "/get_block 200 has our key but a DIFFERENT hash_to_verify — refusing to ack, \
             remaining unconfirmed"
        }
        _ => {
            "/get_block 200 body malformed or missing key — untrusted, remaining unconfirmed"
        }
    }
}

fn log_submission(rec: &FindRecord, c: &Classification, res: &TransportResult) {
    let http = if res.transport_ok {
        format!("HTTP {}", res.http_status)
    } else {
        "network unavailable".to_string()
    };
    // The reason ("why parked/rejected/resubmitting") is the thread an operator pulls during
    // an outage, so it stays on the record and in the log for anything but a clean ack.
    match c.next_status {
        FindStatus::Acked => tracing::info!(
            id = rec.id,
            kind = rec.payload.kind.as_str(),
            m = rec.payload.memory_cost,
            %http,
            "submission confirmed"
        ),
        FindStatus::PermanentlyInvalid | FindStatus::Quarantined => tracing::error!(
            id = rec.id,
            kind = rec.payload.kind.as_str(),
            m = rec.payload.memory_cost,
            %http,
            reason = %c.reason,
            "submission rejected"
        ),
        _ => tracing::warn!(
            id = rec.id,
            kind = rec.payload.kind.as_str(),
            m = rec.payload.memory_cost,
            status = c.next_status.as_str(),
            %http,
            reason = %c.reason,
            "submission not acked"
        ),
    }
}

/// Convenience for implementors: turn any error into the submitter's fatal journal error.
impl From<std::io::Error> for JournalError {
    fn from(e: std::io::Error) -> Self {
        JournalError::new(e.to_string())
    }
}
