//! Serialised terminal output. Port of `src/ConsoleLog.h`.
//!
//! The C++ version segfaulted inside libc when a background thread wrote to `std::cout`
//! while the TUI thread was redrawing the alternate screen: two unsynchronised streams into
//! the same FILE. It was fixed there by convention — every call site had to remember to take
//! `ConsoleLog::mutex()`.
//!
//! Here the fix is structural. The `Stdout` handle lives *inside* the mutex, is private to
//! this module, and is never handed out. There is no way to obtain a writer without holding
//! the lock, so the racing-writer bug cannot be reintroduced by a new call site. Everything
//! that wants the terminal — events, the ticker, plain lines, the whole ratatui frame — goes
//! through [`Console`], and each call emits its text under one lock acquisition.

use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// Redirected output cannot redraw a single line, so the ticker is downsampled to a plain
/// snapshot at this interval instead of one line per batch.
const PLAIN_PROGRESS_INTERVAL: Duration = Duration::from_secs(30);

/// Offset applied to every timestamp this crate formats, in whole seconds east of UTC.
///
/// The lookup that produces it (`localtime_r`) is only legal while the process is
/// single-threaded, so this crate cannot perform it: the miner captures the offset in
/// `main` and pushes it here with [`set_time_offset`]. Zero (UTC) until it does, which is
/// also the right answer when the platform lookup was refused.
static TIME_OFFSET_SECONDS: AtomicI32 = AtomicI32::new(0);

/// Set the offset every console and log timestamp is rendered in. Call once from `main`,
/// before any thread that logs is spawned.
pub fn set_time_offset(offset: time::UtcOffset) {
    TIME_OFFSET_SECONDS.store(offset.whole_seconds(), Ordering::Release);
}

/// The offset in effect, or UTC when none was supplied.
pub fn time_offset() -> time::UtcOffset {
    time::UtcOffset::from_whole_seconds(TIME_OFFSET_SECONDS.load(Ordering::Acquire))
        .unwrap_or(time::UtcOffset::UTC)
}

/// Now, in the offset [`set_time_offset`] installed.
pub fn now_local() -> time::OffsetDateTime {
    time::OffsetDateTime::now_utc().to_offset(time_offset())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    Debug,
    Info,
    Found,
    Ok,
    Retry,
    Park,
    Warn,
    Error,
}

impl Level {
    pub fn name(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Found => "FOUND",
            Level::Ok => "OK",
            Level::Retry => "RETRY",
            Level::Park => "PARK",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }

    pub fn color(self) -> &'static str {
        match self {
            Level::Debug => "\x1b[90m",
            Level::Info => "\x1b[34m",
            Level::Found => "\x1b[35m",
            Level::Ok => "\x1b[32m",
            Level::Retry | Level::Park | Level::Warn => "\x1b[33m",
            Level::Error => "\x1b[31m",
        }
    }
}

/// Called for every event so the TUI can mirror it into its event pane.
pub type EventForwarder = Arc<dyn Fn(Level, &str, &str) + Send + Sync>;
/// Called instead of writing a plain line, for the same reason.
pub type LineSink = Arc<dyn Fn(&str) + Send + Sync>;

/// stdout plus the state that must not be observed mid-write. Private: the only way to
/// touch it is through a `Console` method, which means through the mutex.
struct Inner {
    out: std::io::Stdout,
    last_plain_progress: Option<Instant>,
}

pub struct Console {
    inner: Mutex<Inner>,
    /// When the TUI owns the tty, ordinary writes are dropped (and forwarded instead)
    /// rather than spliced into a frame.
    tui_owns_stdout: AtomicBool,
    forwarder: Mutex<Option<EventForwarder>>,
    line_sink: Mutex<Option<LineSink>>,
    interactive: bool,
    color: bool,
}

/// True when stdout is a terminal *and* `TERM` is set, matching
/// `ConsoleLog::interactiveTerminal()`.
pub fn stdout_is_interactive() -> bool {
    std::env::var_os("TERM").is_some() && std::io::stdout().is_terminal()
}

impl Console {
    fn new() -> Self {
        let interactive = stdout_is_interactive();
        Self {
            inner: Mutex::new(Inner { out: std::io::stdout(), last_plain_progress: None }),
            tui_owns_stdout: AtomicBool::new(false),
            forwarder: Mutex::new(None),
            line_sink: Mutex::new(None),
            interactive,
            color: interactive && std::env::var_os("NO_COLOR").is_none(),
        }
    }

    /// The process-wide console. One instance, so one writer.
    pub fn global() -> &'static Console {
        static CONSOLE: OnceLock<Console> = OnceLock::new();
        CONSOLE.get_or_init(Console::new)
    }

    pub fn is_interactive(&self) -> bool {
        self.interactive
    }

    pub fn color_enabled(&self) -> bool {
        self.color
    }

    pub fn tui_owns_stdout(&self) -> bool {
        self.tui_owns_stdout.load(Ordering::Acquire)
    }

    pub fn set_tui_owns_stdout(&self, owns: bool) {
        self.tui_owns_stdout.store(owns, Ordering::Release);
    }

    pub fn set_event_forwarder(&self, forwarder: EventForwarder) {
        *self.forwarder.lock() = Some(forwarder);
    }

    pub fn clear_event_forwarder(&self) {
        *self.forwarder.lock() = None;
    }

    pub fn set_line_sink(&self, sink: LineSink) {
        *self.line_sink.lock() = Some(sink);
    }

    pub fn clear_line_sink(&self) {
        *self.line_sink.lock() = None;
    }

    /// A structured event line. Forwarded to the TUI (if attached) and written to stdout
    /// unless the TUI owns the screen.
    pub fn event(&self, level: Level, component: &str, message: &str) {
        if !self.tui_owns_stdout() {
            let text = format_event_line(&clock_hms(), level, component, message, self.color);
            let mut inner = self.inner.lock();
            if self.interactive {
                let _ = inner.out.write_all(b"\x1b[2K\r");
            }
            let _ = inner.out.write_all(text.as_bytes());
            let _ = inner.out.write_all(b"\n");
            let _ = inner.out.flush();
        }
        // Forwarding happens outside the stdout lock: the TUI's event queue has its own.
        let forwarder = self.forwarder.lock().clone();
        if let Some(forwarder) = forwarder {
            forwarder(level, component, message);
        }
    }

    /// The one-line status ticker. Rewrites the current line on a tty; downsamples to a
    /// plain line elsewhere.
    pub fn progress(&self, message: &str) {
        if self.tui_owns_stdout() {
            return;
        }
        let mut inner = self.inner.lock();
        // Re-checked under the lock: the TUI may have claimed stdout while we waited.
        if self.tui_owns_stdout() {
            return;
        }
        if self.interactive {
            let _ = inner.out.write_all(message.as_bytes());
        } else {
            let now = Instant::now();
            if let Some(last) = inner.last_plain_progress {
                if now.duration_since(last) < PLAIN_PROGRESS_INTERVAL {
                    return;
                }
            }
            inner.last_plain_progress = Some(now);
            let _ = inner.out.write_all(strip_ansi(message).as_bytes());
            let _ = inner.out.write_all(b"\n");
        }
        let _ = inner.out.flush();
    }

    /// A plain console line. Goes to the line sink when one is attached (the TUI's event
    /// pane), otherwise to stdout on a cleared line.
    pub fn line(&self, message: &str) {
        let sink = self.line_sink.lock().clone();
        if let Some(sink) = sink {
            sink(message);
            return;
        }
        let mut inner = self.inner.lock();
        if self.interactive {
            let _ = inner.out.write_all(b"\x1b[2K\r");
        }
        let _ = inner.out.write_all(message.as_bytes());
        if !message.ends_with('\n') {
            let _ = inner.out.write_all(b"\n");
        }
        let _ = inner.out.flush();
    }

    /// Raw bytes — a whole rendered frame or a terminal control sequence — emitted under
    /// the same lock as everything else.
    pub fn write_raw(&self, bytes: &[u8]) {
        let mut inner = self.inner.lock();
        let _ = inner.out.write_all(bytes);
        let _ = inner.out.flush();
    }

    /// Last-resort write for the panic hook: a panic can unwind while this thread already
    /// holds the console lock, and blocking there would hang the process instead of
    /// restoring the terminal. Losing interleaving in that path is acceptable; losing the
    /// cursor is not.
    pub fn write_raw_best_effort(&self, bytes: &[u8]) {
        match self.inner.try_lock() {
            Some(mut inner) => {
                let _ = inner.out.write_all(bytes);
                let _ = inner.out.flush();
            }
            None => {
                let mut out = std::io::stdout();
                let _ = out.write_all(bytes);
                let _ = out.flush();
            }
        }
    }
}

/// `{hh:mm:ss}  {LEVEL:<6}{component:<12}{message}`, with optional ANSI styling.
pub fn format_event_line(
    timestamp: &str,
    level: Level,
    component: &str,
    message: &str,
    color: bool,
) -> String {
    if color {
        format!(
            "\x1b[2m{timestamp}\x1b[0m  {}{:<6}\x1b[0m\x1b[36m{:<12}\x1b[0m{message}",
            level.color(),
            level.name(),
            component
        )
    } else {
        format!("{timestamp}  {:<6}{:<12}{message}", level.name(), component)
    }
}

/// Drop CSI escape sequences and carriage returns so redirected logs stay readable.
pub fn strip_ansi(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = String::with_capacity(value.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\r' => i += 1,
            0x1b if i + 1 < bytes.len() && bytes[i + 1] == b'[' => {
                i += 2;
                while i < bytes.len() && !bytes[i].is_ascii_alphabetic() {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1;
                }
            }
            _ => {
                // Copy one whole UTF-8 character, not one byte.
                let start = i;
                i += 1;
                while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                    i += 1;
                }
                out.push_str(&value[start..i]);
            }
        }
    }
    out
}

pub(crate) fn clock_hms() -> String {
    let now = now_local();
    format!("{:02}:{:02}:{:02}", now.hour(), now.minute(), now.second())
}

/// `std::io::Write` adapter that buffers a whole frame and hands it to the console in one
/// locked write. ratatui's backend wants a `Write`; giving it stdout directly would put a
/// second writer on the terminal, which is the bug this module exists to prevent.
pub struct ConsoleWriter {
    buffer: Vec<u8>,
}

impl ConsoleWriter {
    pub fn new() -> Self {
        Self { buffer: Vec::with_capacity(8192) }
    }
}

impl Default for ConsoleWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl Write for ConsoleWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.buffer.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if !self.buffer.is_empty() {
            Console::global().write_raw(&self.buffer);
            self.buffer.clear();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_csi_sequences_and_carriage_returns() {
        assert_eq!(strip_ansi("\x1b[2K\rhello"), "hello");
        assert_eq!(strip_ansi("\x1b[31mred\x1b[0m"), "red");
        assert_eq!(strip_ansi("plain"), "plain");
        assert_eq!(strip_ansi("a\x1b[1;32mb"), "ab");
    }

    #[test]
    fn strip_ansi_keeps_multibyte_characters_intact() {
        assert_eq!(strip_ansi("12.0 kH/s  •  1 GPU"), "12.0 kH/s  •  1 GPU");
    }

    #[test]
    fn event_line_pads_level_and_component() {
        assert_eq!(
            format_event_line("01:02:03", Level::Ok, "submit", "stored", false),
            "01:02:03  OK    submit      stored"
        );
        let colored = format_event_line("01:02:03", Level::Error, "gpu", "boom", true);
        assert!(colored.contains("\x1b[31m"));
        assert_eq!(strip_ansi(&colored), "01:02:03  ERROR gpu         boom");
    }

    #[test]
    fn console_writer_buffers_until_flush() {
        let mut writer = ConsoleWriter::new();
        writer.write_all(b"abc").expect("buffered write cannot fail");
        assert_eq!(writer.buffer, b"abc");
    }
}
