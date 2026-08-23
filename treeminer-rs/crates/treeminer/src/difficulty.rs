//! Network difficulty: the poller, the shared value the miner reads, and the on-disk cache
//! that survives a restart during an outage. Port of `src/DifficultyManager.{h,cpp}`.
//!
//! WHY THE CACHE EXISTS
//! `m` is the Argon2 memory cost, so mining at the wrong difficulty is not a cosmetic
//! problem: too low and every find is rejected with a 401, too high and hashrate is thrown
//! away. When the endpoint is unreachable the miner has no way to learn the real value, and
//! the hardcoded 42069 fallback is roughly 50x the real difficulty — a restart during an
//! outage would burn the whole outage's worth of work. So every successful poll writes the
//! value to `difficulty.cache` (write-then-rename, because a torn write is a garbage
//! difficulty), and boot seeds from it.
//!
//! Poll failures never stop mining. The miner keeps hashing at the cached value; the
//! endpoint is only declared DOWN after [`POOL_DOWN_FAILURE_THRESHOLD`] consecutive
//! failures, which keeps a single transient miss out of the operator's face.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tm_submit::{extract_json_field, Transport};
use tm_tui::{Console, Level};

/// Cache file name, relative to the working directory — matches the C++ constant.
pub const DIFFICULTY_CACHE_FILE: &str = "difficulty.cache";

/// Consecutive poll failures before the endpoint is declared down.
pub const POOL_DOWN_FAILURE_THRESHOLD: u32 = 3;

/// Poll interval (`updateDifficultyPeriodically`).
pub const POLL_INTERVAL: Duration = Duration::from_secs(10);

/// HTTP timeout for `GET /difficulty`.
pub const DIFFICULTY_TIMEOUT_MS: u64 = 5000;

/// Difficulty the C++ falls back to when there is no cache and no successful poll yet.
pub const FALLBACK_DIFFICULTY: u32 = 42069;

/// Accepted range for a cached value. Anything outside it is treated as no cache at all.
const MIN_DIFFICULTY: i64 = 1;
const MAX_DIFFICULTY: i64 = 100_000_000;

/// The difficulty the mining loop reads and the endpoint-health flag the dashboard shows.
/// Shared by `Arc`; every field is atomic so no reader ever blocks the poller.
#[derive(Debug)]
pub struct DifficultyShared {
    difficulty: AtomicU32,
    endpoint_down: AtomicBool,
}

impl Default for DifficultyShared {
    fn default() -> Self {
        Self::new(FALLBACK_DIFFICULTY)
    }
}

impl DifficultyShared {
    pub fn new(initial: u32) -> Self {
        Self { difficulty: AtomicU32::new(initial), endpoint_down: AtomicBool::new(false) }
    }

    pub fn difficulty(&self) -> u32 {
        self.difficulty.load(Ordering::Acquire)
    }

    pub fn set_difficulty(&self, value: u32) {
        self.difficulty.store(value, Ordering::Release);
    }

    pub fn endpoint_down(&self) -> bool {
        self.endpoint_down.load(Ordering::Acquire)
    }
}

/// What one poll did, so callers (and tests) can assert on it without reading the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollOutcome {
    /// A successful sample. `changed` is false when the server repeated the same value.
    Updated { difficulty: u32, changed: bool },
    /// The sample failed; the shared difficulty is untouched and mining continues on it.
    Failed { consecutive_failures: u32, error: String },
}

/// Read `difficulty.cache`. Returns `None` when absent, unreadable, unparseable, or out of
/// the sane range — all of which the C++ collapsed into "0", i.e. "no cache".
pub fn load_cached_difficulty(path: &Path) -> Option<u32> {
    let text = fs::read_to_string(path).ok()?;
    let value = crate::config::stoi(&text)?;
    if !(MIN_DIFFICULTY..=MAX_DIFFICULTY).contains(&value) {
        return None;
    }
    Some(value as u32)
}

/// Write `difficulty.cache` atomically: full write to `<file>.tmp`, then rename over the
/// real path. Best-effort — a cache we could not write is not worth failing a poll over.
pub fn persist_difficulty(path: &Path, difficulty: u32) {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    if fs::write(&tmp, format!("{difficulty}\n")).is_err() {
        let _ = fs::remove_file(&tmp);
        return;
    }
    if fs::rename(&tmp, path).is_err() {
        let _ = fs::remove_file(&tmp);
    }
}

/// Starting difficulty for this run: the cache when it holds a plausible value, otherwise
/// the 42069 fallback. The returned message is the C++ `DIFFICULTY seeded from cache` line.
pub fn seed_initial_difficulty(cache_path: &Path) -> (u32, Option<String>) {
    match load_cached_difficulty(cache_path) {
        Some(cached) => (cached, Some(format!("seeded from cache | current={cached}"))),
        None => (FALLBACK_DIFFICULTY, None),
    }
}

type Observer = Box<dyn Fn(u32) + Send + Sync>;

/// The poll loop's state. Drive it one step at a time with [`DifficultyPoller::poll_once`]
/// or hand it to [`DifficultyPoller::run`].
pub struct DifficultyPoller<T> {
    transport: T,
    cache_path: PathBuf,
    shared: Arc<DifficultyShared>,
    observer: Option<Observer>,
    consecutive_failures: u32,
}

impl<T: Transport> DifficultyPoller<T> {
    pub fn new(transport: T, cache_path: impl Into<PathBuf>, shared: Arc<DifficultyShared>) -> Self {
        Self {
            transport,
            cache_path: cache_path.into(),
            shared,
            observer: None,
            consecutive_failures: 0,
        }
    }

    /// Fired on every successful sample, changed or not, so the submission manager's trend
    /// tracking and difficulty unparking see the poller's observations too.
    pub fn with_observer<F: Fn(u32) + Send + Sync + 'static>(mut self, observer: F) -> Self {
        self.observer = Some(Box::new(observer));
        self
    }

    pub fn shared(&self) -> Arc<DifficultyShared> {
        Arc::clone(&self.shared)
    }

    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }

    /// One `GET /difficulty` round-trip and everything that follows from it.
    pub fn poll_once(&mut self) -> PollOutcome {
        let sample = match fetch_difficulty(&self.transport) {
            Ok(value) => value,
            Err(error) => {
                self.consecutive_failures += 1;
                if self.consecutive_failures == 1 {
                    // Keep the cause (DNS/timeout/HTTP) on the first failure — it is the
                    // operator's first clue whether the pool, the link, or DNS is at fault.
                    Console::global().event(
                        Level::Warn,
                        "NETWORK",
                        &format!(
                            "difficulty poll failed; using cached value and retrying — {error}"
                        ),
                    );
                }
                if self.consecutive_failures == POOL_DOWN_FAILURE_THRESHOLD {
                    self.shared.endpoint_down.store(true, Ordering::Release);
                    Console::global().event(
                        Level::Error,
                        "NETWORK",
                        &format!(
                            "difficulty endpoint DOWN after {} failures — mining continues on cached difficulty; polling every 10s",
                            self.consecutive_failures
                        ),
                    );
                }
                return PollOutcome::Failed {
                    consecutive_failures: self.consecutive_failures,
                    error,
                };
            }
        };

        let changed = self.shared.difficulty() != sample;
        if changed {
            self.shared.set_difficulty(sample);
        }
        if let Some(observer) = &self.observer {
            observer(sample);
        }
        persist_difficulty(&self.cache_path, sample);

        if self.consecutive_failures > 0 {
            let failures = self.consecutive_failures;
            self.consecutive_failures = 0;
            if self.shared.endpoint_down.swap(false, Ordering::AcqRel) {
                Console::global().event(
                    Level::Ok,
                    "NETWORK",
                    &format!("difficulty restored | current={sample}"),
                );
            } else {
                Console::global().event(
                    Level::Info,
                    "NETWORK",
                    &format!("difficulty poll restored | prior_failures={failures}"),
                );
            }
        }

        PollOutcome::Updated { difficulty: sample, changed }
    }

    /// Poll until `running` clears. The sleep is chopped up so shutdown does not have to
    /// wait out a full interval.
    pub fn run(&mut self, running: &AtomicBool, interval: Duration) {
        while running.load(Ordering::Acquire) {
            self.poll_once();
            let deadline = Instant::now() + interval;
            while running.load(Ordering::Acquire) && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(100).min(interval));
            }
        }
    }
}

/// `GET /difficulty` and parse `{"difficulty": "<N>"}` — note the value is a JSON *string*.
/// Error strings mirror the C++ exception messages so operator reports stay comparable.
fn fetch_difficulty<T: Transport>(transport: &T) -> Result<u32, String> {
    let response = transport.difficulty();
    if !response.transport_ok {
        return Err(format!("Error: {}", response.error));
    }
    if response.http_status != 200 {
        return Err(format!(
            "Error: Failed to get the difficulty: HTTP status code {}",
            response.http_status
        ));
    }
    let field = extract_json_field(&response.body, "difficulty")
        .ok_or_else(|| "JSON parsing error: no 'difficulty' field in the response".to_string())?;
    let value = crate::config::stoi(&field)
        .ok_or_else(|| format!("Error: difficulty '{field}' is not a number"))?;
    if !(MIN_DIFFICULTY..=MAX_DIFFICULTY).contains(&value) {
        return Err(format!("Error: difficulty '{field}' is out of range"));
    }
    Ok(value as u32)
}

/// Owning handle for the background poller. Holding it keeps the thread alive; dropping it
/// stops and joins the thread, so no poller outlives the miner it belongs to.
pub struct DifficultyHandle {
    shared: Arc<DifficultyShared>,
    running: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
}

impl DifficultyHandle {
    /// Seed from the cache, do the first poll on the calling thread (as the C++ did, so a
    /// reachable server has corrected the difficulty before the first batch is sized), then
    /// keep polling in the background.
    pub fn spawn<T: Transport + Send + 'static>(
        transport: T,
        cache_path: impl Into<PathBuf>,
        initial_difficulty: u32,
        interval: Duration,
    ) -> Self {
        let shared = Arc::new(DifficultyShared::new(initial_difficulty));
        let running = Arc::new(AtomicBool::new(true));
        let mut poller =
            DifficultyPoller::new(transport, cache_path, Arc::clone(&shared));
        poller.poll_once();

        let thread_running = Arc::clone(&running);
        let join = thread::Builder::new()
            .name("difficulty".into())
            .spawn(move || poller.run(&thread_running, interval))
            .ok();
        Self { shared, running, join }
    }

    pub fn shared(&self) -> Arc<DifficultyShared> {
        Arc::clone(&self.shared)
    }

    pub fn difficulty(&self) -> u32 {
        self.shared.difficulty()
    }

    pub fn endpoint_down(&self) -> bool {
        self.shared.endpoint_down()
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

impl Drop for DifficultyHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}
