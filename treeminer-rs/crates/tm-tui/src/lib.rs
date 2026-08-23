//! Operator-facing output for TreeMiner: display-mode resolution, the serialised console,
//! the rotating file logger, the `logs`-mode status ticker, and the alternate-screen TUI.
//!
//! Rust port of `src/TerminalUi.{h,cpp}`, `src/ConsoleLog.h`, `src/Logger.{h,cpp}` and the
//! display-mode logic in `src/main.cpp`.
//!
//! The crate owns no mining state. The integrator supplies a [`MinerSnapshot`] per frame and
//! a [`TickerSnapshot`] per ticker update, so nothing here has to reach across crates.

pub mod console;
pub mod display;
pub mod logger;
pub mod snapshot;
pub mod terminal;
pub mod ticker;
pub mod shutdown;

pub use console::{now_local, set_time_offset, time_offset, Console, ConsoleWriter, Level};
pub use display::{
    resolve_display_mode, resolve_prompt_selection, tui_forbidden, DisplayDecision, DisplayEnv,
    DisplayMode, PROMPT_TEXT,
};
pub use logger::{log_to_console, FileLogger, DEFAULT_MAX_FILE_SIZE};
pub use shutdown::Shutdown;
pub use snapshot::{
    DeliveryStats, EngineStats, FindCounts, GpuStats, Identity, MinerSnapshot, NetworkState,
    TickerSnapshot,
};
pub use terminal::{install_panic_hook, render, SnapshotProvider, TerminalUi};
pub use ticker::format_ticker;
