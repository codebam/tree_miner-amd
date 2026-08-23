//! Rotating file logger. Port of `src/Logger.{h,cpp}`.
//!
//! Two files alternate — `<base>0.txt` and `<base>1.txt` — so disk use is bounded at twice
//! the size limit regardless of how long the miner runs. The C++ version appended to the
//! current file on startup and only counted bytes it wrote itself; both are preserved so a
//! restart does not silently reset the rotation point.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use parking_lot::Mutex;

use crate::console::{now_local, Console};

/// The default the C++ miner used: `log0.txt` / `log1.txt`, 1 MiB each.
pub const DEFAULT_MAX_FILE_SIZE: u64 = 1024 * 1024;

pub struct FileLogger {
    base: PathBuf,
    max_file_size: u64,
    state: Mutex<LogState>,
}

struct LogState {
    file: BufWriter<File>,
    current_size: u64,
    index: u8,
}

impl FileLogger {
    /// `base` is a path prefix, not a directory: `log` yields `log0.txt`. Any parent
    /// directory in the prefix is created, so `log/miner` works out of the box.
    pub fn new(base: impl AsRef<Path>, max_file_size: u64) -> std::io::Result<Self> {
        let base = base.as_ref().to_path_buf();
        if let Some(parent) = base.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let path = file_path(&base, 0);
        let current_size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            base,
            max_file_size,
            state: Mutex::new(LogState { file: BufWriter::new(file), current_size, index: 0 }),
        })
    }

    /// Path of the file currently being written.
    pub fn current_path(&self) -> PathBuf {
        file_path(&self.base, self.state.lock().index)
    }

    /// Append one timestamped line, rotating first if it would cross the size limit.
    pub fn log(&self, message: &str) -> std::io::Result<()> {
        self.log_with_timestamp(&timestamp(), message)
    }

    /// Same, with the timestamp supplied — used by tests so output is deterministic.
    pub fn log_with_timestamp(&self, timestamp: &str, message: &str) -> std::io::Result<()> {
        let line = format!("{timestamp} {message}");
        let mut state = self.state.lock();
        if state.current_size + line.len() as u64 >= self.max_file_size {
            self.rotate(&mut state)?;
        }
        state.file.write_all(line.as_bytes())?;
        state.file.write_all(b"\n")?;
        // The C++ logger used std::endl; operators tail these files during an outage, so the
        // flush per line is deliberate.
        state.file.flush()?;
        state.current_size += line.len() as u64 + 1;
        Ok(())
    }

    fn rotate(&self, state: &mut LogState) -> std::io::Result<()> {
        state.file.flush()?;
        state.index = (state.index + 1) % 2;
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(file_path(&self.base, state.index))?;
        state.file = BufWriter::new(file);
        state.current_size = 0;
        Ok(())
    }
}

fn file_path(base: &Path, index: u8) -> PathBuf {
    let mut name = base.as_os_str().to_os_string();
    name.push(format!("{index}.txt"));
    PathBuf::from(name)
}

/// `MM-DD HH:MM`, as in the C++ logger.
fn timestamp() -> String {
    let now = now_local();
    format!(
        "{:02}-{:02} {:02}:{:02}",
        u8::from(now.month()),
        now.day(),
        now.hour(),
        now.minute()
    )
}

/// Plain console output, routed through the single console writer (and therefore to the
/// TUI's event pane when one is attached). Port of `Logger::logToConsole`.
pub fn log_to_console(message: &str) {
    Console::global().line(message);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_the_parent_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().join("log").join("miner");
        let logger = FileLogger::new(&base, DEFAULT_MAX_FILE_SIZE).expect("logger");
        logger.log("hello").expect("write");
        assert!(dir.path().join("log").is_dir());
        let body = std::fs::read_to_string(dir.path().join("log/miner0.txt")).expect("read");
        assert!(body.ends_with(" hello\n"), "unexpected body: {body:?}");
    }

    #[test]
    fn rotates_between_two_files_at_the_size_limit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().join("log");
        // Each line is "ts message\n"; the limit admits a couple of them before rotating.
        let logger = FileLogger::new(&base, 40).expect("logger");
        for i in 0..12 {
            logger
                .log_with_timestamp("01-02 03:04", &format!("line{i}"))
                .expect("write");
        }
        let zero = std::fs::read_to_string(dir.path().join("log0.txt")).expect("read 0");
        let one = std::fs::read_to_string(dir.path().join("log1.txt")).expect("read 1");
        assert!(zero.len() < 40, "log0 exceeded the limit: {}", zero.len());
        assert!(one.len() < 40, "log1 exceeded the limit: {}", one.len());
        // The newest line is always in whichever file is current; nothing is lost silently.
        let newest = if logger.current_path().ends_with("log0.txt") { &zero } else { &one };
        assert!(newest.contains("line11"), "newest file missing the last line: {newest:?}");
        // Rotation really happened rather than one file growing without bound.
        assert!(!one.is_empty());
    }

    #[test]
    fn resumes_the_existing_file_size_on_restart() {
        let dir = tempfile::tempdir().expect("tempdir");
        let base = dir.path().join("log");
        {
            let logger = FileLogger::new(&base, 1000).expect("logger");
            logger.log_with_timestamp("01-02 03:04", "first").expect("write");
        }
        let logger = FileLogger::new(&base, 1000).expect("reopen");
        logger.log_with_timestamp("01-02 03:04", "second").expect("write");
        let body = std::fs::read_to_string(dir.path().join("log0.txt")).expect("read");
        assert_eq!(body, "01-02 03:04 first\n01-02 03:04 second\n");
    }

    #[test]
    fn timestamp_has_the_cpp_shape() {
        let stamp = timestamp();
        assert_eq!(stamp.len(), 11, "unexpected timestamp {stamp:?}");
        assert_eq!(stamp.as_bytes()[2], b'-');
        assert_eq!(stamp.as_bytes()[5], b' ');
        assert_eq!(stamp.as_bytes()[8], b':');
    }
}
