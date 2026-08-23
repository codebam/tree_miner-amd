//! The operator console boundary. Port of the `std::cin` / `std::cout` calls that
//! `AppConfig` and `resolveDisplayMode` made directly.
//!
//! The C++ read stdin from inside the config loader, which made first-run setup and the
//! invalid-address recovery path impossible to test. Routing it through a trait keeps the
//! behaviour identical while letting the tests drive both.

use std::io::{self, BufRead, Write};

pub trait Prompter {
    /// Print `prompt` (no newline) and read one line of input. `Ok(None)` on EOF.
    fn prompt_line(&mut self, prompt: &str) -> io::Result<Option<String>>;
    /// Print an operator-visible notice (the C++ `std::cout` messages around the prompts).
    fn notify(&mut self, message: &str);
}

/// Real stdin/stdout.
#[derive(Debug, Default)]
pub struct StdioPrompter;

impl Prompter for StdioPrompter {
    fn prompt_line(&mut self, prompt: &str) -> io::Result<Option<String>> {
        print!("{prompt}");
        io::stdout().flush()?;
        let mut line = String::new();
        if io::stdin().lock().read_line(&mut line)? == 0 {
            return Ok(None);
        }
        Ok(Some(line.trim_end_matches(['\n', '\r']).to_string()))
    }

    fn notify(&mut self, message: &str) {
        println!("{message}");
    }
}

/// Fails every prompt. Used when the caller has already decided that nothing may block on
/// stdin (`--minerAddr` + `--totalDevFee` supplied, service runs).
#[derive(Debug, Default)]
pub struct NoPrompter;

impl Prompter for NoPrompter {
    fn prompt_line(&mut self, _prompt: &str) -> io::Result<Option<String>> {
        Ok(None)
    }
    fn notify(&mut self, _message: &str) {}
}

/// Scripted answers, for tests.
#[derive(Debug, Default)]
pub struct ScriptedPrompter {
    answers: Vec<String>,
    pub prompts: Vec<String>,
    pub notices: Vec<String>,
}

impl ScriptedPrompter {
    pub fn new<I, S>(answers: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut answers: Vec<String> = answers.into_iter().map(Into::into).collect();
        answers.reverse();
        Self { answers, prompts: Vec::new(), notices: Vec::new() }
    }
}

impl Prompter for ScriptedPrompter {
    fn prompt_line(&mut self, prompt: &str) -> io::Result<Option<String>> {
        self.prompts.push(prompt.to_string());
        Ok(self.answers.pop())
    }

    fn notify(&mut self, message: &str) {
        self.notices.push(message.to_string());
    }
}
