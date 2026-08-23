//! Display-mode resolution. Port of `tuiForbidden()` / `resolveDisplayMode()` from
//! `src/main.cpp`.
//!
//! The decision is a pure function of three inputs (requested mode, environment, tty-ness)
//! so it can be tested without a terminal. The C++ version reached straight into `getenv`
//! and `isatty` from inside the decision, which made the systemd fallback untestable — and
//! that fallback is load-bearing: the alternate-screen UI hangs or races when stdout is not
//! an operator tty.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayMode {
    /// Scrolling log lines plus the one-line status ticker.
    Logs,
    /// Alternate-screen ratatui UI.
    Terminal,
    /// Ask the operator once on stdin, then behave as the chosen mode.
    Prompt,
}

impl DisplayMode {
    pub fn as_str(self) -> &'static str {
        match self {
            DisplayMode::Logs => "logs",
            DisplayMode::Terminal => "terminal",
            DisplayMode::Prompt => "prompt",
        }
    }

    /// `None` for anything the C++ CLI would have rejected.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "logs" => Some(DisplayMode::Logs),
            "terminal" => Some(DisplayMode::Terminal),
            "prompt" => Some(DisplayMode::Prompt),
            _ => None,
        }
    }
}

impl fmt::Display for DisplayMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The three environment facts the decision depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DisplayEnv {
    pub stdout_is_tty: bool,
    /// `TREEMINER_NO_TUI` is set (to anything, including empty — matches `getenv != nullptr`).
    pub no_tui_env: bool,
    /// `INVOCATION_ID` is set: we are a systemd unit.
    pub invocation_id_env: bool,
}

impl DisplayEnv {
    /// Reads the real process environment. `TERM` unset counts as non-interactive, matching
    /// `ConsoleLog::interactiveTerminal()`.
    pub fn from_process() -> Self {
        Self {
            stdout_is_tty: crate::console::stdout_is_interactive(),
            no_tui_env: std::env::var_os("TREEMINER_NO_TUI").is_some(),
            invocation_id_env: std::env::var_os("INVOCATION_ID").is_some(),
        }
    }
}

/// The TUI is for an interactive operator tty only.
pub fn tui_forbidden(env: &DisplayEnv) -> bool {
    env.no_tui_env || env.invocation_id_env || !env.stdout_is_tty
}

/// Outcome of resolving a requested mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayDecision {
    pub mode: DisplayMode,
    /// Operator-visible reason for a downgrade, written to stderr by the binary.
    pub warning: Option<String>,
    /// True when the binary must still ask on stdin and feed the answer to
    /// [`resolve_prompt_selection`].
    pub prompt_required: bool,
}

/// Resolve a requested mode against the environment.
///
/// `Prompt` survives resolution only when the TUI is allowed; otherwise it degrades to
/// `Logs` without asking, because there is nobody at the other end of stdin.
pub fn resolve_display_mode(requested: DisplayMode, env: &DisplayEnv) -> DisplayDecision {
    if tui_forbidden(env) && requested != DisplayMode::Logs {
        return DisplayDecision {
            mode: DisplayMode::Logs,
            warning: Some(format!(
                "Display '{requested}' is disabled for service/non-interactive runs; using logs."
            )),
            prompt_required: false,
        };
    }
    DisplayDecision {
        mode: requested,
        warning: None,
        prompt_required: requested == DisplayMode::Prompt,
    }
}

/// Map the operator's answer to the prompt. Anything but `2` means the presentation
/// terminal, matching the C++ default-on-empty-input behaviour.
pub fn resolve_prompt_selection(selection: &str, env: &DisplayEnv) -> DisplayMode {
    let chosen = if selection.trim() == "2" {
        DisplayMode::Logs
    } else {
        DisplayMode::Terminal
    };
    if tui_forbidden(env) && chosen != DisplayMode::Logs {
        return DisplayMode::Logs;
    }
    chosen
}

/// The menu the binary prints before reading stdin, kept here so its wording travels with
/// the decision it belongs to.
pub const PROMPT_TEXT: &str = "\nTreeMiner display\n  1. Presentation terminal\n  2. Scrolling logs\nSelect [1]: ";

#[cfg(test)]
mod tests {
    use super::*;

    fn env(tty: bool, no_tui: bool, systemd: bool) -> DisplayEnv {
        DisplayEnv { stdout_is_tty: tty, no_tui_env: no_tui, invocation_id_env: systemd }
    }

    #[test]
    fn parses_only_the_three_known_modes() {
        assert_eq!(DisplayMode::parse("logs"), Some(DisplayMode::Logs));
        assert_eq!(DisplayMode::parse("terminal"), Some(DisplayMode::Terminal));
        assert_eq!(DisplayMode::parse("prompt"), Some(DisplayMode::Prompt));
        assert_eq!(DisplayMode::parse("Terminal"), None);
        assert_eq!(DisplayMode::parse(""), None);
    }

    #[test]
    fn forbidden_table() {
        assert!(!tui_forbidden(&env(true, false, false)));
        assert!(tui_forbidden(&env(false, false, false)));
        assert!(tui_forbidden(&env(true, true, false)));
        assert!(tui_forbidden(&env(true, false, true)));
        assert!(tui_forbidden(&env(false, true, true)));
    }

    #[test]
    fn resolution_table_over_tty_env_and_requested_mode() {
        let modes = [DisplayMode::Logs, DisplayMode::Terminal, DisplayMode::Prompt];
        for tty in [false, true] {
            for no_tui in [false, true] {
                for systemd in [false, true] {
                    let e = env(tty, no_tui, systemd);
                    let allowed = tty && !no_tui && !systemd;
                    for requested in modes {
                        let got = resolve_display_mode(requested, &e);
                        let expected = if allowed || requested == DisplayMode::Logs {
                            requested
                        } else {
                            DisplayMode::Logs
                        };
                        assert_eq!(got.mode, expected, "tty={tty} no_tui={no_tui} systemd={systemd} req={requested}");
                        assert_eq!(got.warning.is_some(), expected != requested);
                        assert_eq!(got.prompt_required, expected == DisplayMode::Prompt);
                    }
                }
            }
        }
    }

    #[test]
    fn logs_is_never_downgraded_or_warned_about() {
        let d = resolve_display_mode(DisplayMode::Logs, &env(false, true, true));
        assert_eq!(d.mode, DisplayMode::Logs);
        assert!(d.warning.is_none());
        assert!(!d.prompt_required);
    }

    #[test]
    fn prompt_selection_defaults_to_terminal_and_still_respects_forbidden() {
        let interactive = env(true, false, false);
        assert_eq!(resolve_prompt_selection("", &interactive), DisplayMode::Terminal);
        assert_eq!(resolve_prompt_selection("1", &interactive), DisplayMode::Terminal);
        assert_eq!(resolve_prompt_selection("2", &interactive), DisplayMode::Logs);
        assert_eq!(resolve_prompt_selection(" 2 \n", &interactive), DisplayMode::Logs);
        // A prompt answered while the TUI is forbidden (e.g. stdin piped in a unit file).
        assert_eq!(resolve_prompt_selection("1", &env(true, true, false)), DisplayMode::Logs);
    }
}
