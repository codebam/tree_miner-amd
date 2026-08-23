//! Journal-first find capture. Port of the `submitCallback` lambda in `src/main.cpp`.
//!
//! TWO INVARIANTS LIVE HERE, AND THEY ARE WHY THIS PROJECT EXISTS.
//!
//! 1. **The payload is built once, from the parameters the batch actually used.** Upstream
//!    re-hashed with the *current* difficulty at submit time, so a find made just before a
//!    difficulty tick was silently dropped or submitted with an `m=` it never paid. The
//!    memory cost travels with the find from the moment it is discovered and is never
//!    recomputed.
//!
//! 2. **Durability precedes the network.** Every find is appended to the journal before any
//!    HTTP attempt. If the journal write fails the find goes to an append-only fsync'd
//!    fallback sink whose failure domain is deliberately disjoint from SQLite's, and the
//!    next boot imports it back. If *both* fail, the miner is destroying every future find
//!    it makes, so that is declared fatal rather than logged and shrugged off.

use std::sync::Arc;

use tm_core::{FindKind, FoundPayload};
use tm_journal::{FallbackSink, Journal};
use tm_tui::{Console, FileLogger, Level};

use crate::state::{classify_find, FindClass, MiningState};

/// One find, as the producing loop knows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Find {
    /// The 40 hex characters of the address that was mined for (no `0x`).
    pub hexsalt: String,
    /// The Argon2 password: the server's dedupe key.
    pub key: String,
    /// Bare unpadded base64 digest — not a PHC string.
    pub digest: String,
    /// The `m` THIS batch ran at. Never the global difficulty.
    pub memory_cost: u32,
    pub attempts: u64,
    pub hashes_per_second: f64,
    /// `"GPU"` or `"CPU"`, for per-backend accounting.
    pub source: String,
}

/// Where a find ended up.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capture {
    /// In the journal, with its row id.
    Journaled(i64),
    /// In the fallback sink; the next boot imports it.
    Fallback,
    /// Nowhere. Fatal: declared as such before this is returned.
    Lost,
}

impl Capture {
    pub fn is_durable(self) -> bool {
        !matches!(self, Capture::Lost)
    }
}

/// Called after a find is durably journaled, so the submitter can wake its drain loop
/// instead of waiting out an idle poll.
pub type FindNotifier = Arc<dyn Fn() + Send + Sync>;

/// Called after a find reaches durable storage — journal OR fallback sink — with the
/// payload as it will be submitted.
///
/// Platform mode reports finds to the broker through this. STRICTLY after durable capture,
/// and never for a find that reached neither path: MQTT can block or race shutdown, and
/// advertising a block the disk never received has the platform (and its consumer)
/// accounting for value that no longer exists.
pub type FindObserver = Arc<dyn Fn(&Find, &FoundPayload) + Send + Sync>;

/// The journal-first sink every producer (GPU loop, CPU sidecar) hands its finds to.
pub struct FindSink {
    journal: Arc<dyn Journal + Send + Sync>,
    fallback: FallbackSink,
    state: Arc<MiningState>,
    machine_id: String,
    logger: Option<Arc<FileLogger>>,
    notifier: Option<FindNotifier>,
    observer: Option<FindObserver>,
    /// Injected so tests do not depend on the wall clock.
    now_utc: Box<dyn Fn() -> String + Send + Sync>,
}

impl std::fmt::Debug for FindSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FindSink")
            .field("fallback", &self.fallback.path())
            .field("machine_id", &self.machine_id)
            .finish()
    }
}

impl FindSink {
    pub fn new(
        journal: Arc<dyn Journal + Send + Sync>,
        fallback: FallbackSink,
        state: Arc<MiningState>,
        machine_id: impl Into<String>,
    ) -> Self {
        Self {
            journal,
            fallback,
            state,
            machine_id: machine_id.into(),
            logger: None,
            notifier: None,
            observer: None,
            now_utc: Box::new(|| {
                tm_submit::iso_utc(tm_submit::clocktime::now_wall_ms())
            }),
        }
    }

    pub fn with_logger(mut self, logger: Arc<FileLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    pub fn with_notifier(mut self, notifier: FindNotifier) -> Self {
        self.notifier = Some(notifier);
        self
    }

    pub fn with_observer(mut self, observer: FindObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Fixed timestamps for tests.
    pub fn with_clock(mut self, clock: impl Fn() -> String + Send + Sync + 'static) -> Self {
        self.now_utc = Box::new(clock);
        self
    }

    /// Build the immutable payload for a find. Public because it is the single place the
    /// `m=` is fixed, and the tests assert on it directly.
    pub fn payload_for(&self, find: &Find) -> Result<FoundPayload, String> {
        let hash_to_verify = tm_core::assemble_phc(find.memory_cost, &find.hexsalt, &find.digest)
            .map_err(|error| format!("cannot encode find: {error}"))?;
        Ok(FoundPayload {
            key: find.key.clone(),
            hash_to_verify,
            account: format!("0x{}", find.hexsalt),
            kind: if find.digest.contains("XEN11") {
                FindKind::Xen11
            } else {
                FindKind::Xuni
            },
            memory_cost: find.memory_cost,
            worker: self.machine_id.clone(),
            attempts: find.attempts,
            hashes_per_second: find.hashes_per_second,
            found_at_utc: (self.now_utc)(),
        })
    }

    /// Capture one find. Returns only after it is durable (or after the fatal state has
    /// been declared).
    pub fn record(&self, find: &Find) -> Capture {
        let payload = match self.payload_for(find) {
            Ok(payload) => payload,
            Err(error) => {
                // Not a durability failure: the find itself is unrepresentable, which can
                // only come from a malformed address, and no amount of retrying fixes it.
                Console::global().event(Level::Error, "FIND", &error);
                self.log(&format!("found DROPPED source={} {error}", find.source));
                return Capture::Lost;
            }
        };

        let capture = self.persist(&payload);
        self.account(find, &payload, capture);
        if let (Capture::Journaled(_), Some(notifier)) = (capture, &self.notifier) {
            notifier();
        }
        // Journal or fallback both count as durable — the sink is imported back on the next
        // boot — but `Lost` does not, and must not be reported anywhere.
        if let (true, Some(observer)) = (capture.is_durable(), &self.observer) {
            observer(find, &payload);
        }
        capture
    }

    /// Journal, then fallback sink, then fatal.
    fn persist(&self, payload: &FoundPayload) -> Capture {
        let error = match self.journal.append(payload) {
            Ok(id) => {
                self.refresh_queue_counts();
                return Capture::Journaled(id);
            }
            Err(error) => error.to_string(),
        };

        if self.fallback.append(payload) {
            Console::global().event(
                Level::Warn,
                "JOURNAL",
                &format!(
                    "write failed; find captured in fallback sink | {} | {error}",
                    self.fallback.path().display()
                ),
            );
            self.log(&format!("JOURNAL WRITE FAILED, fallback sink OK error={error}"));
            return Capture::Fallback;
        }

        Console::global().event(
            Level::Error,
            "JOURNAL",
            &format!(
                "write failed AND fallback sink failed — find at risk; HALTING miner \
                 (exit nonzero for supervisor restart) | {error}"
            ),
        );
        self.log(&format!(
            "JOURNAL WRITE FAILED, fallback sink FAILED — FATAL, stopping miner error={error}"
        ));
        self.state.declare_fatal_durability_failure(&format!(
            "journal append and fallback sink both failed: {error}"
        ));
        Capture::Lost
    }

    /// Counters, log line and console line. The lifetime counters mean "finds this run that
    /// still exist", so a find that reached neither durability path is not counted: the
    /// status line must not claim value the disk never received.
    fn account(&self, find: &Find, payload: &FoundPayload, capture: Capture) {
        let class = classify_find(&find.digest);
        if capture.is_durable() {
            self.state.record_find_class(class);
        }

        let id = match capture {
            Capture::Journaled(id) => id.to_string(),
            Capture::Fallback => "fallback".to_owned(),
            Capture::Lost => "none".to_owned(),
        };
        self.log(&format!(
            "found id={id} source={} kind={} mined_m={} {}",
            find.source,
            payload.kind.as_str(),
            payload.memory_cost,
            if matches!(capture, Capture::Journaled(_)) {
                "journaled"
            } else {
                "NOT-JOURNALED"
            }
        ));

        let mut message = format!(
            "#{id}  \u{2022}  {}  \u{2022}  m={}",
            find.source, payload.memory_cost
        );
        if class == FindClass::Superblock {
            message.push_str("  \u{2022}  SUPERBLOCK");
        }
        message.push_str(match capture {
            Capture::Journaled(_) => "  \u{2022}  saved locally  \u{2022}  queued",
            Capture::Fallback => "  \u{2022}  saved to fallback",
            // Must not read as "handled": nothing durable holds this find and the fatal
            // state declared above is about to take the miner down.
            Capture::Lost => "  \u{2022}  SAVE FAILED  \u{2022}  stopping",
        });
        Console::global().event(
            match capture {
                Capture::Journaled(_) => Level::Found,
                Capture::Fallback => Level::Warn,
                Capture::Lost => Level::Error,
            },
            payload.kind.as_str(),
            &message,
        );
    }

    /// Refresh the queued gauges from the journal. A failure here is not worth reacting to:
    /// the durable append already succeeded and the next outcome refreshes the display.
    fn refresh_queue_counts(&self) {
        if let Ok(counts) = self.journal.counts() {
            self.state
                .set_queued(counts.queued_xen11 as u64, counts.queued_xuni as u64);
        }
    }

    fn log(&self, message: &str) {
        if let Some(logger) = &self.logger {
            let _ = logger.log(message);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{FailingJournal, RecordingJournal};

    fn find(memory_cost: u32, digest: &str) -> Find {
        Find {
            hexsalt: "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc".to_owned(),
            key: "52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f".to_owned(),
            digest: digest.to_owned(),
            memory_cost,
            attempts: 42,
            hashes_per_second: 1234.5,
            source: "GPU".to_owned(),
        }
    }

    fn sink_with(
        journal: Arc<dyn Journal + Send + Sync>,
        state: Arc<MiningState>,
        dir: &std::path::Path,
    ) -> FindSink {
        FindSink::new(
            journal,
            FallbackSink::new(dir.join("fallback.jsonl")),
            state,
            "worker-1",
        )
        .with_clock(|| "2026-01-01T00:00:00Z".to_owned())
    }

    #[test]
    fn the_payload_carries_the_batch_s_own_memory_cost() {
        let dir = tempfile::tempdir().expect("tempdir");
        let journal = Arc::new(RecordingJournal::default());
        let state = Arc::new(MiningState::for_test(1000));
        let sink = sink_with(journal.clone(), state, dir.path());

        assert_eq!(sink.record(&find(1000, "abcXEN11def")), Capture::Journaled(1));

        let appended = journal.appended();
        assert_eq!(appended.len(), 1);
        assert_eq!(appended[0].memory_cost, 1000);
        assert!(appended[0].hash_to_verify.starts_with("$argon2id$v=19$m=1000,t=1,p=1$"));
        assert!(appended[0].hash_to_verify.ends_with("abcXEN11def"));
        assert_eq!(appended[0].kind, FindKind::Xen11);
        assert_eq!(appended[0].account, "0xe4bb184781bbc9c7004e8dafd4a9b49d203bc9bc");
        assert_eq!(appended[0].worker, "worker-1");
    }

    #[test]
    fn a_difficulty_change_after_the_batch_cannot_rewrite_the_payload() {
        let dir = tempfile::tempdir().expect("tempdir");
        let journal = Arc::new(RecordingJournal::default());
        let state = Arc::new(MiningState::for_test(1000));
        let sink = sink_with(journal.clone(), Arc::clone(&state), dir.path());

        // The batch ran at m=1000; the network moved on before the find was recorded.
        let found = find(1000, "abcXEN11def");
        state.set_difficulty(2000);
        sink.record(&found);

        let appended = journal.appended();
        assert_eq!(appended.len(), 1, "exactly one payload per find");
        assert_eq!(
            appended[0].memory_cost, 1000,
            "the PHC must carry the m the batch actually paid, not the new difficulty"
        );
        assert!(appended[0].hash_to_verify.contains("m=1000,"));
    }

    #[test]
    fn a_journal_failure_routes_to_the_fallback_sink_without_losing_the_find() {
        let dir = tempfile::tempdir().expect("tempdir");
        let journal = Arc::new(FailingJournal::new("disk I/O error"));
        let state = Arc::new(MiningState::for_test(1000));
        let sink = sink_with(journal, Arc::clone(&state), dir.path());

        assert_eq!(sink.record(&find(1000, "abcXEN11def")), Capture::Fallback);
        assert!(!state.fatal_durability_failure(), "the sink took it; not fatal");

        let sunk = std::fs::read_to_string(dir.path().join("fallback.jsonl")).expect("sink file");
        assert!(sunk.contains("m=1000,"));
        assert_eq!(state.super_blocks() + state.normal_blocks(), 1);
    }

    #[test]
    fn losing_both_durability_paths_is_fatal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let journal = Arc::new(FailingJournal::new("disk I/O error"));
        let state = Arc::new(MiningState::for_test(1000));
        // A sink path inside a file (not a directory) cannot be appended to.
        let blocked = dir.path().join("not-a-dir/fallback.jsonl");
        let sink = FindSink::new(
            journal,
            FallbackSink::new(blocked),
            Arc::clone(&state),
            "worker-1",
        );

        assert_eq!(sink.record(&find(1000, "abcXEN11def")), Capture::Lost);
        assert!(state.fatal_durability_failure());
        assert!(!state.is_running());
        assert_eq!(
            state.normal_blocks() + state.super_blocks() + state.xuni_blocks(),
            0,
            "a find that reached no durable store must not be counted"
        );
    }

    #[test]
    fn a_xuni_find_is_journaled_as_xuni() {
        let dir = tempfile::tempdir().expect("tempdir");
        let journal = Arc::new(RecordingJournal::default());
        let state = Arc::new(MiningState::for_test(1000));
        let sink = sink_with(journal.clone(), Arc::clone(&state), dir.path());

        sink.record(&find(1000, "abcXUNIdef"));

        assert_eq!(journal.appended()[0].kind, FindKind::Xuni);
        assert_eq!(state.xuni_blocks(), 1);
    }

    #[test]
    fn the_submitter_is_woken_only_for_journaled_finds() {
        let dir = tempfile::tempdir().expect("tempdir");
        let woken = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = Arc::clone(&woken);
        let state = Arc::new(MiningState::for_test(1000));

        let good = sink_with(
            Arc::new(RecordingJournal::default()),
            Arc::clone(&state),
            dir.path(),
        )
        .with_notifier(Arc::new(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }));
        good.record(&find(1000, "abcXEN11def"));
        assert_eq!(woken.load(std::sync::atomic::Ordering::SeqCst), 1);

        let counter = Arc::clone(&woken);
        let bad = sink_with(
            Arc::new(FailingJournal::new("boom")),
            Arc::clone(&state),
            dir.path(),
        )
        .with_notifier(Arc::new(move || {
            counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }));
        bad.record(&find(1000, "abcXEN11def"));
        assert_eq!(
            woken.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "nothing to drain when the journal never took it"
        );
    }
}
