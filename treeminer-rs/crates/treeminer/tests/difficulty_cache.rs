//! The three behaviours `difficulty.cache` exists for: it seeds the boot difficulty, every
//! successful poll rewrites it atomically, and a failed poll leaves the miner hashing at the
//! cached value while the endpoint is reported down after the failure threshold.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tm_core::FoundPayload;
use tm_submit::{Transport, TransportResult};
use treeminer::{
    load_cached_difficulty, persist_difficulty, seed_initial_difficulty, DifficultyPoller,
    DifficultyShared, PollOutcome, FALLBACK_DIFFICULTY, POOL_DOWN_FAILURE_THRESHOLD,
};

/// Scripted `/difficulty` source. Each entry is one round-trip; the last one repeats.
struct FakeSource {
    responses: Vec<TransportResult>,
    calls: AtomicUsize,
}

impl FakeSource {
    fn new(responses: Vec<TransportResult>) -> Self {
        Self { responses, calls: AtomicUsize::new(0) }
    }

    fn ok(values: &[u32]) -> Self {
        Self::new(
            values
                .iter()
                .map(|v| TransportResult::ok(200, format!("{{\"difficulty\": \"{v}\"}}")))
                .collect(),
        )
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Acquire)
    }
}

impl Transport for FakeSource {
    fn submit(&self, _payload: &FoundPayload) -> TransportResult {
        unreachable!("the difficulty poller never submits")
    }
    fn confirm(&self, _key: &str) -> TransportResult {
        unreachable!("the difficulty poller never confirms")
    }
    fn difficulty(&self) -> TransportResult {
        let index = self.calls.fetch_add(1, Ordering::AcqRel);
        self.responses[index.min(self.responses.len() - 1)].clone()
    }
}

struct Cache {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

impl Cache {
    fn new() -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("difficulty.cache");
        Self { _dir: dir, path }
    }

    fn write(&self, text: &str) {
        std::fs::write(&self.path, text).expect("write cache");
    }

    fn read(&self) -> String {
        std::fs::read_to_string(&self.path).unwrap_or_default()
    }

    fn tmp_exists(&self) -> bool {
        let mut tmp = self.path.as_os_str().to_os_string();
        tmp.push(".tmp");
        PathBuf::from(tmp).exists()
    }
}

// ------------------------------------------------- 1. the cache seeds the boot difficulty

#[test]
fn boot_seeds_from_the_cache_when_it_holds_a_plausible_value() {
    let cache = Cache::new();
    cache.write("5000\n");
    let (difficulty, note) = seed_initial_difficulty(&cache.path);
    assert_eq!(difficulty, 5000);
    assert_eq!(note.as_deref(), Some("seeded from cache | current=5000"));
}

/// Without a cache the miner starts at 42069 — roughly 50x the real difficulty, which is
/// exactly the wasted work the cache exists to prevent.
#[test]
fn boot_falls_back_to_42069_without_a_cache() {
    let cache = Cache::new();
    let (difficulty, note) = seed_initial_difficulty(&cache.path);
    assert_eq!(difficulty, FALLBACK_DIFFICULTY);
    assert!(note.is_none());
}

#[test]
fn a_garbage_or_out_of_range_cache_is_treated_as_absent() {
    let cache = Cache::new();
    for text in ["", "not a number", "0", "-5", "100000001"] {
        cache.write(text);
        assert_eq!(load_cached_difficulty(&cache.path), None, "{text:?}");
        assert_eq!(seed_initial_difficulty(&cache.path).0, FALLBACK_DIFFICULTY);
    }
    cache.write("1");
    assert_eq!(load_cached_difficulty(&cache.path), Some(1));
    cache.write("100000000");
    assert_eq!(load_cached_difficulty(&cache.path), Some(100_000_000));
}

// ------------------------------------------- 2. every successful poll rewrites the cache

#[test]
fn a_successful_poll_rewrites_the_cache_atomically() {
    let cache = Cache::new();
    cache.write("100\n");
    let shared = Arc::new(DifficultyShared::new(100));
    let mut poller =
        DifficultyPoller::new(FakeSource::ok(&[6000, 6000, 7000]), &cache.path, Arc::clone(&shared));

    assert_eq!(
        poller.poll_once(),
        PollOutcome::Updated { difficulty: 6000, changed: true }
    );
    assert_eq!(shared.difficulty(), 6000);
    assert_eq!(cache.read(), "6000\n");
    // Write-then-rename: no temporary file is ever left behind.
    assert!(!cache.tmp_exists());

    // An unchanged value still rewrites the cache — a restart must not lose freshness.
    assert_eq!(
        poller.poll_once(),
        PollOutcome::Updated { difficulty: 6000, changed: false }
    );
    assert_eq!(cache.read(), "6000\n");

    assert_eq!(
        poller.poll_once(),
        PollOutcome::Updated { difficulty: 7000, changed: true }
    );
    assert_eq!(cache.read(), "7000\n");
    assert_eq!(load_cached_difficulty(&cache.path), Some(7000));
}

#[test]
fn the_observer_sees_every_sample_changed_or_not() {
    let cache = Cache::new();
    let seen = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let recorder = Arc::clone(&seen);
    let mut poller = DifficultyPoller::new(
        FakeSource::ok(&[900, 900, 950]),
        &cache.path,
        Arc::new(DifficultyShared::new(900)),
    )
    .with_observer(move |value| recorder.lock().push(value));

    for _ in 0..3 {
        poller.poll_once();
    }
    assert_eq!(*seen.lock(), vec![900, 900, 950]);
}

#[test]
fn persist_is_atomic_and_leaves_no_temporary_behind() {
    let cache = Cache::new();
    persist_difficulty(&cache.path, 1234);
    assert_eq!(cache.read(), "1234\n");
    assert!(!cache.tmp_exists());
    persist_difficulty(&cache.path, 5678);
    assert_eq!(cache.read(), "5678\n");
}

// ------------- 3. a failed poll keeps mining on the cached value, then reports DOWN

#[test]
fn a_failed_poll_keeps_the_cached_difficulty_and_reports_down_after_the_threshold() {
    let cache = Cache::new();
    cache.write("4321\n");
    let (seeded, _) = seed_initial_difficulty(&cache.path);
    let shared = Arc::new(DifficultyShared::new(seeded));
    let mut poller = DifficultyPoller::new(
        FakeSource::new(vec![TransportResult::failed("Could not resolve host")]),
        &cache.path,
        Arc::clone(&shared),
    );

    for attempt in 1..POOL_DOWN_FAILURE_THRESHOLD {
        let outcome = poller.poll_once();
        assert!(matches!(
            outcome,
            PollOutcome::Failed { consecutive_failures, .. } if consecutive_failures == attempt
        ));
        // Below the threshold a single miss stays quiet: mining is unaffected.
        assert!(!shared.endpoint_down());
        assert_eq!(shared.difficulty(), 4321);
    }

    let outcome = poller.poll_once();
    assert!(matches!(
        outcome,
        PollOutcome::Failed { consecutive_failures, .. }
            if consecutive_failures == POOL_DOWN_FAILURE_THRESHOLD
    ));
    assert!(shared.endpoint_down());
    // The whole point: mining continues at the cached value, and the cache is untouched.
    assert_eq!(shared.difficulty(), 4321);
    assert_eq!(cache.read(), "4321\n");
}

#[test]
fn an_http_error_is_a_failure_with_the_cpp_wording() {
    let cache = Cache::new();
    let mut poller = DifficultyPoller::new(
        FakeSource::new(vec![TransportResult::ok(503, "upstream down")]),
        &cache.path,
        Arc::new(DifficultyShared::new(1000)),
    );
    let PollOutcome::Failed { error, .. } = poller.poll_once() else {
        panic!("a 503 must not count as a sample");
    };
    assert_eq!(error, "Error: Failed to get the difficulty: HTTP status code 503");
    assert_eq!(cache.read(), "");
}

#[test]
fn an_unparseable_body_is_a_failure_not_a_zero_difficulty() {
    let cache = Cache::new();
    let mut poller = DifficultyPoller::new(
        FakeSource::new(vec![TransportResult::ok(200, "<html>captive portal</html>")]),
        &cache.path,
        Arc::new(DifficultyShared::new(1000)),
    );
    let PollOutcome::Failed { error, .. } = poller.poll_once() else {
        panic!("a non-JSON body must not count as a sample");
    };
    assert!(error.starts_with("JSON parsing error:"), "{error}");
}

#[test]
fn recovery_clears_the_down_flag_and_resets_the_failure_run() {
    let cache = Cache::new();
    let mut responses = vec![TransportResult::failed("timeout"); POOL_DOWN_FAILURE_THRESHOLD as usize];
    responses.push(TransportResult::ok(200, "{\"difficulty\": \"2048\"}"));
    let shared = Arc::new(DifficultyShared::new(1000));
    let source = FakeSource::new(responses);
    let mut poller = DifficultyPoller::new(source, &cache.path, Arc::clone(&shared));

    for _ in 0..POOL_DOWN_FAILURE_THRESHOLD {
        poller.poll_once();
    }
    assert!(shared.endpoint_down());

    assert_eq!(
        poller.poll_once(),
        PollOutcome::Updated { difficulty: 2048, changed: true }
    );
    assert!(!shared.endpoint_down());
    assert_eq!(poller.consecutive_failures(), 0);
    assert_eq!(cache.read(), "2048\n");
}

#[test]
fn the_background_handle_polls_and_stops_cleanly() {
    let cache = Cache::new();
    cache.write("777\n");
    let source = Arc::new(FakeSource::ok(&[888]));
    let handle = treeminer::DifficultyHandle::spawn(
        Arc::clone(&source),
        cache.path.clone(),
        seed_initial_difficulty(&cache.path).0,
        std::time::Duration::from_millis(20),
    );
    assert_eq!(handle.difficulty(), 888);
    assert!(!handle.endpoint_down());
    handle.stop();
    assert!(source.calls() >= 1);
}
