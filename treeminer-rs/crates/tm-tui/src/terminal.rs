//! The alternate-screen operator UI. Port of `src/TerminalUi.{h,cpp}`, rendered with
//! ratatui instead of hand-assembled escape sequences.
//!
//! Two hard-won behaviours from the C++ are preserved:
//!   * the UI never writes to stdout itself — every frame goes through [`Console`], which is
//!     the single writer (see `console.rs`);
//!   * a render failure must never affect mining, so the render thread swallows errors and
//!     tries again on the next tick.
//!
//! The terminal is restored on `stop()`, on `Drop`, and from a panic hook, because leaving
//! an operator in the alternate screen with a hidden cursor after a crash is how a
//! recoverable fault turns into "the box is wedged".

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::{Condvar, Mutex};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{Frame, Terminal};

use crate::console::{strip_ansi, Console, ConsoleWriter};
use crate::snapshot::{MinerSnapshot, NetworkState};

const MAX_EVENTS: usize = 100;
const TICK: Duration = Duration::from_millis(500);
const ENTER_ALTERNATE: &[u8] = b"\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H";
const LEAVE_ALTERNATE: &[u8] = b"\x1b[?25h\x1b[?1049l";

/// The integrator supplies this; it is called once per frame.
pub type SnapshotProvider = Arc<dyn Fn() -> MinerSnapshot + Send + Sync>;

/// Set while any TUI holds the screen, so the panic hook knows whether to restore it.
static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Restore the terminal from a panic. Installed once; chains to the previous hook so
/// backtraces and test harness reporting still happen.
pub fn install_panic_hook() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if TUI_ACTIVE.swap(false, Ordering::AcqRel) {
                Console::global().set_tui_owns_stdout(false);
                Console::global().write_raw_best_effort(LEAVE_ALTERNATE);
            }
            previous(info);
        }));
    });
}

struct Events {
    queue: Mutex<VecDeque<String>>,
    wake: Condvar,
}

pub struct TerminalUi {
    provider: SnapshotProvider,
    events: Arc<Events>,
    running: Arc<AtomicBool>,
    stop_requested: Arc<AtomicBool>,
    frame: Arc<AtomicU64>,
    thread: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl TerminalUi {
    pub fn new(provider: SnapshotProvider) -> Self {
        Self {
            provider,
            events: Arc::new(Events { queue: Mutex::new(VecDeque::new()), wake: Condvar::new() }),
            running: Arc::new(AtomicBool::new(false)),
            stop_requested: Arc::new(AtomicBool::new(false)),
            frame: Arc::new(AtomicU64::new(0)),
            thread: Mutex::new(None),
        }
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Take over the screen and start rendering. Idempotent; fails only if the terminal
    /// cannot be sized, which means the caller ignored the display-mode check.
    pub fn start(&self) -> std::io::Result<()> {
        if self.running.swap(true, Ordering::AcqRel) {
            return Ok(());
        }
        install_panic_hook();
        self.stop_requested.store(false, Ordering::Release);

        let mut terminal = Terminal::new(ratatui::backend::CrosstermBackend::new(ConsoleWriter::new()))
            .inspect_err(|_| self.running.store(false, Ordering::Release))?;

        Console::global().set_tui_owns_stdout(true);
        TUI_ACTIVE.store(true, Ordering::Release);
        Console::global().write_raw(ENTER_ALTERNATE);

        let provider = Arc::clone(&self.provider);
        let events = Arc::clone(&self.events);
        let stop_requested = Arc::clone(&self.stop_requested);
        let frame_counter = Arc::clone(&self.frame);
        let handle = std::thread::Builder::new()
            .name("treeminer-tui".into())
            .spawn(move || {
                while !stop_requested.load(Ordering::Acquire) {
                    let snapshot = provider();
                    let pending: Vec<String> = events.queue.lock().iter().cloned().collect();
                    // A display failure must never affect mining.
                    let _ = terminal.draw(|f| render(f, &snapshot, &pending));
                    frame_counter.fetch_add(1, Ordering::Relaxed);
                    let mut queue = events.queue.lock();
                    events.wake.wait_for(&mut queue, TICK);
                }
            })?;
        *self.thread.lock() = Some(handle);
        Ok(())
    }

    /// Stop rendering and hand the screen back. Safe to call repeatedly.
    pub fn stop(&self) {
        if !self.running.load(Ordering::Acquire) {
            return;
        }
        self.stop_requested.store(true, Ordering::Release);
        self.events.wake.notify_all();
        if let Some(handle) = self.thread.lock().take() {
            let _ = handle.join();
        }
        self.running.store(false, Ordering::Release);
        Console::global().set_tui_owns_stdout(false);
        if TUI_ACTIVE.swap(false, Ordering::AcqRel) {
            Console::global().write_raw(LEAVE_ALTERNATE);
        }
    }

    /// Queue a line for the events pane. ANSI is stripped and the ticker is dropped: it
    /// redraws in place and would otherwise fill the pane with duplicates.
    pub fn post_event(&self, message: &str) {
        let clean = strip_ansi(message).trim().to_string();
        if clean.is_empty() || clean.starts_with("Mining:") {
            return;
        }
        {
            let mut queue = self.events.queue.lock();
            queue.push_back(format!("{}  {clean}", crate::console::clock_hms()));
            while queue.len() > MAX_EVENTS {
                queue.pop_front();
            }
        }
        self.events.wake.notify_one();
    }

    /// Frames drawn so far; lets the integrator confirm the thread is alive.
    pub fn frames_rendered(&self) -> u64 {
        self.frame.load(Ordering::Relaxed)
    }
}

impl Drop for TerminalUi {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Draw one frame. Public so it can be exercised against a `TestBackend` without a tty.
pub fn render(frame: &mut Frame, snapshot: &MinerSnapshot, events: &[String]) {
    let area = frame.area();
    let gpu_rows = snapshot.gpus.len().max(1) as u16 + 2;
    let chunks = Layout::vertical([
        Constraint::Length(2),
        Constraint::Length(5),
        Constraint::Length(5),
        Constraint::Length(gpu_rows),
        Constraint::Min(3),
        Constraint::Length(1),
    ])
    .split(area);

    render_header(frame, chunks[0], snapshot);
    render_engine(frame, chunks[1], snapshot);
    render_delivery(frame, chunks[2], snapshot);
    render_compute(frame, chunks[3], snapshot);
    render_events(frame, chunks[4], events);
    render_footer(frame, chunks[5], snapshot);
}

fn render_header(frame: &mut Frame, area: Rect, snapshot: &MinerSnapshot) {
    let text = vec![
        Line::from(Span::styled(
            "  HASHHEAD // TREEMINER",
            Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            format!("  TERMINAL OPS :: {}", snapshot.identity.name),
            Style::default().add_modifier(Modifier::DIM),
        )),
    ];
    frame.render_widget(Paragraph::new(text), area);
}

fn panel(title: &str) -> Block<'_> {
    Block::default().borders(Borders::ALL).title(format!(" {title} "))
}

fn render_engine(frame: &mut Frame, area: Rect, snapshot: &MinerSnapshot) {
    let block = panel("ENGINE");
    let inner = block.inner(area).width as usize;
    let engine = &snapshot.engine;
    let text = vec![
        Line::from(Span::styled(
            columns(&["RATE", "WORK RATE", "DIFFICULTY", "UPTIME"], inner),
            Style::default().add_modifier(Modifier::DIM),
        )),
        Line::from(columns(
            &[
                &rate(engine.total_hashrate()),
                &format!("{:.1} M-units/s", engine.work_rate_m_units()),
                &engine.difficulty.to_string(),
                &duration(engine.uptime_seconds),
            ],
            inner,
        )),
        Line::from(columns(
            &[
                &format!("GPU  {} / {} streams", rate(engine.gpu_hashrate), engine.gpu_streams),
                &format!("CPU  {} / {} workers", rate(engine.cpu_hashrate), engine.cpu_workers),
            ],
            inner,
        )),
    ];
    frame.render_widget(Paragraph::new(text).block(block), area);
}

fn render_delivery(frame: &mut Frame, area: Rect, snapshot: &MinerSnapshot) {
    let block = panel("DELIVERY");
    let inner = block.inner(area).width as usize;
    let delivery = &snapshot.delivery;
    let network_style = Style::default().fg(match delivery.network {
        NetworkState::Online => Color::Green,
        NetworkState::Probing => Color::Cyan,
        NetworkState::Offline => Color::Yellow,
    });
    let finds = &snapshot.finds;
    let text = vec![
        Line::from(Span::styled(
            columns(&["NETWORK", "LAST UPLINK", "Q_XNM", "Q_XUNI"], inner),
            Style::default().add_modifier(Modifier::DIM),
        )),
        Line::from(Span::styled(
            columns(
                &[
                    delivery.network.as_str(),
                    &delivery.last_submission,
                    &delivery.queued_xnm.to_string(),
                    &delivery.queued_xuni.to_string(),
                ],
                inner,
            ),
            network_style,
        )),
        Line::from(columns(
            &[
                &format!("SESSION XNM  {}", finds.xnm),
                &format!("SESSION XUNI  {}", finds.xuni),
                &format!("SUPER  {}", finds.superblocks),
                &format!("REJECTED  {}", finds.rejected),
            ],
            inner,
        )),
    ];
    frame.render_widget(Paragraph::new(text).block(block), area);
}

fn render_compute(frame: &mut Frame, area: Rect, snapshot: &MinerSnapshot) {
    let block = panel("COMPUTE");
    let text: Vec<Line> = if snapshot.gpus.is_empty() {
        vec![Line::from("  No active GPU telemetry")]
    } else {
        snapshot
            .gpus
            .iter()
            .map(|gpu| {
                Line::from(format!(
                    "  GPU {} / S{}  {}  |  {}  |  VRAM {:.1}%",
                    gpu.index,
                    gpu.stream + 1,
                    gpu.name,
                    rate(gpu.hashrate),
                    gpu.memory_used_percent
                ))
            })
            .collect()
    };
    frame.render_widget(Paragraph::new(text).block(block), area);
}

fn render_events(frame: &mut Frame, area: Rect, events: &[String]) {
    let block = panel("EVENTS");
    let rows = block.inner(area).height as usize;
    let start = events.len().saturating_sub(rows);
    let text: Vec<Line> = events[start..]
        .iter()
        .map(|event| Line::from(Span::styled(format!(" {event}"), event_style(event))))
        .collect();
    frame.render_widget(Paragraph::new(text).block(block), area);
}

fn render_footer(frame: &mut Frame, area: Rect, snapshot: &MinerSnapshot) {
    let footer = format!(
        " Ctrl-C stop  |  {}  |  {}  |  {}",
        snapshot.console_url, snapshot.identity.name, snapshot.identity.address
    );
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            fit(&footer, area.width as usize),
            Style::default().add_modifier(Modifier::DIM),
        ))),
        area,
    );
}

fn event_style(event: &str) -> Style {
    if event.contains("CONFIRMED") {
        Style::default().fg(Color::Green)
    } else if event.contains("RETRY") || event.contains("PARKED") {
        Style::default().fg(Color::Yellow)
    } else if event.contains("FAILED") || event.contains("REJECTED") {
        Style::default().fg(Color::Red)
    } else {
        Style::default()
    }
}

/// Pad to `width`, or truncate with an ellipsis. Byte-oriented like the C++ original, but
/// truncating on a character boundary so a multi-byte name cannot split.
fn fit(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        let mut out = String::from(value);
        out.extend(std::iter::repeat_n(' ', width - value.chars().count()));
        return out;
    }
    if width <= 3 {
        return value.chars().take(width).collect();
    }
    let mut out: String = value.chars().take(width - 3).collect();
    out.push_str("...");
    out
}

/// Lay values out in equal columns across `width`, each indented by two spaces. The last
/// column absorbs the rounding remainder.
fn columns(values: &[&str], width: usize) -> String {
    if values.is_empty() {
        return " ".repeat(width);
    }
    let column_width = width / values.len();
    let mut result = String::new();
    for (i, value) in values.iter().enumerate() {
        let available = if i + 1 == values.len() {
            width.saturating_sub(result.chars().count())
        } else {
            column_width
        };
        result.push_str(&fit(&format!("  {value}"), available));
    }
    result
}

/// `MH/s` above a million hashes per second, `kH/s` below.
fn rate(hashes: f64) -> String {
    if hashes >= 1_000_000.0 {
        format!("{:.2} MH/s", hashes / 1_000_000.0)
    } else {
        format!("{:.2} kH/s", hashes / 1_000.0)
    }
}

/// `Xh MMm SSs`, dropping the hour segment while it is zero.
fn duration(seconds: i64) -> String {
    let seconds = seconds.max(0);
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;
    if hours > 0 {
        format!("{hours}h {minutes:02}m {secs:02}s")
    } else {
        format!("{minutes:02}m {secs:02}s")
    }
}

/// Read one row of a rendered buffer back as text. Test helper, also useful to the
/// integrator when snapshotting frames.
pub fn buffer_row(buffer: &Buffer, y: u16) -> String {
    (0..buffer.area.width)
        .map(|x| buffer[(x, y)].symbol())
        .collect::<String>()
        .trim_end()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::snapshot::{DeliveryStats, EngineStats, FindCounts, GpuStats, Identity};
    use ratatui::backend::TestBackend;

    fn sample() -> MinerSnapshot {
        MinerSnapshot {
            identity: Identity {
                name: "rig-01".into(),
                address: "0x1111111111111111111111111111111111111111".into(),
            },
            engine: EngineStats {
                gpu_hashrate: 1_800_000.0,
                cpu_hashrate: 200_000.0,
                gpu_streams: 4,
                cpu_workers: 8,
                difficulty: 1_800_000,
                uptime_seconds: 3 * 3600 + 4 * 60 + 5,
            },
            finds: FindCounts { xnm: 3, xuni: 2, superblocks: 1, rejected: 0 },
            delivery: DeliveryStats {
                network: NetworkState::Online,
                last_submission: "12s ago".into(),
                queued_xnm: 2,
                queued_xuni: 1,
            },
            gpus: vec![GpuStats {
                index: 0,
                stream: 0,
                name: "RX 7900 XTX".into(),
                hashrate: 1_800_000.0,
                memory_used_percent: 42.5,
            }],
            console_url: "http://10.0.0.5:8080/".into(),
        }
    }

    fn draw(snapshot: &MinerSnapshot, events: &[String]) -> Buffer {
        let mut terminal = Terminal::new(TestBackend::new(96, 24)).expect("test backend");
        terminal.draw(|f| render(f, snapshot, events)).expect("draw");
        terminal.backend().buffer().clone()
    }

    #[test]
    fn renders_a_known_snapshot() {
        let buffer = draw(&sample(), &["01:02:03  submit  XNM CONFIRMED".to_string()]);
        assert_eq!(buffer_row(&buffer, 0), "  HASHHEAD // TREEMINER");
        assert_eq!(buffer_row(&buffer, 1), "  TERMINAL OPS :: rig-01");
        assert!(buffer_row(&buffer, 2).contains("ENGINE"));
        // Rate is the GPU+CPU sum; uptime crosses an hour so the hour segment shows.
        let values = buffer_row(&buffer, 4);
        assert!(values.contains("2.00 MH/s"), "{values}");
        assert!(values.contains("3600000.0 M-units/s"), "{values}");
        assert!(values.contains("1800000"), "{values}");
        assert!(values.contains("3h 04m 05s"), "{values}");
        let engines = buffer_row(&buffer, 5);
        assert!(engines.contains("GPU  1.80 MH/s / 4 streams"), "{engines}");
        assert!(engines.contains("CPU  200.00 kH/s / 8 workers"), "{engines}");

        let delivery = buffer_row(&buffer, 9);
        assert!(delivery.contains("online"), "{delivery}");
        assert!(delivery.contains("12s ago"), "{delivery}");
        let finds = buffer_row(&buffer, 10);
        assert!(finds.contains("SESSION XNM  3"), "{finds}");
        assert!(finds.contains("SUPER  1"), "{finds}");

        let gpu = buffer_row(&buffer, 13);
        assert!(gpu.contains("GPU 0 / S1  RX 7900 XTX  |  1.80 MH/s  |  VRAM 42.5%"), "{gpu}");

        let event = buffer_row(&buffer, 16);
        assert!(event.contains("01:02:03  submit  XNM CONFIRMED"), "{event}");

        let footer = buffer_row(&buffer, 23);
        assert!(footer.starts_with(" Ctrl-C stop  |  http://10.0.0.5:8080/  |  rig-01  |  0x1111"), "{footer}");
    }

    #[test]
    fn renders_with_no_gpus_and_no_events() {
        let mut snapshot = sample();
        snapshot.gpus.clear();
        snapshot.engine = EngineStats::default();
        let buffer = draw(&snapshot, &[]);
        assert!(buffer_row(&buffer, 13).contains("No active GPU telemetry"));
        assert!(buffer_row(&buffer, 4).contains("0.00 kH/s"));
        assert!(buffer_row(&buffer, 4).contains("00m 00s"));
    }

    #[test]
    fn events_pane_shows_only_the_newest_lines_that_fit() {
        let events: Vec<String> = (0..40).map(|i| format!("00:00:0{i}  evt{i}")).collect();
        let buffer = draw(&sample(), &events);
        // The pane is 6 rows tall in a 24-row terminal; the last event must be visible.
        let rendered: String = (16..22).map(|y| buffer_row(&buffer, y)).collect();
        assert!(rendered.contains("evt39"), "{rendered}");
        assert!(!rendered.contains("evt0 "), "{rendered}");
    }

    #[test]
    fn event_colours_follow_the_keyword() {
        assert_eq!(event_style("XNM CONFIRMED").fg, Some(Color::Green));
        assert_eq!(event_style("submit RETRY in 4s").fg, Some(Color::Yellow));
        assert_eq!(event_style("PARKED difficulty").fg, Some(Color::Yellow));
        assert_eq!(event_style("upload FAILED").fg, Some(Color::Red));
        assert_eq!(event_style("hello").fg, None);
    }

    #[test]
    fn post_event_drops_the_ticker_and_bounds_the_queue() {
        let ui = TerminalUi::new(Arc::new(MinerSnapshot::default));
        ui.post_event("\x1b[2K\rMining: 12 kH/s");
        ui.post_event("   ");
        assert!(ui.events.queue.lock().is_empty());
        for i in 0..(MAX_EVENTS + 25) {
            ui.post_event(&format!("event {i}"));
        }
        let queue = ui.events.queue.lock();
        assert_eq!(queue.len(), MAX_EVENTS);
        assert!(queue.back().expect("newest").ends_with(&format!("event {}", MAX_EVENTS + 24)));
        assert!(queue.front().expect("oldest").ends_with("event 25"));
    }

    #[test]
    fn fit_pads_and_truncates() {
        assert_eq!(fit("ab", 5), "ab   ");
        assert_eq!(fit("abcdef", 5), "ab...");
        assert_eq!(fit("abcdef", 3), "abc");
    }

    #[test]
    fn columns_fill_the_whole_width() {
        let line = columns(&["a", "b", "c"], 30);
        assert_eq!(line.chars().count(), 30);
        assert!(line.starts_with("  a"));
    }
}
