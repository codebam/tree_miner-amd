//! `config.txt` reader/writer. Port of `src/ConfigManager.{h,cpp}`.
//!
//! The format is deliberately dumb — `key=value`, one per line, split on the FIRST `=` —
//! because operators edit this file by hand and existing deployments already have one. A
//! missing file is not an error: the C++ `ifstream` simply reads nothing, and callers rely
//! on "absent key" and "empty value" being the same thing.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Default config file name, matching `CONFIG_FILENAME` in `MiningCommon.h`.
pub const CONFIG_FILENAME: &str = "config.txt";

#[derive(Debug, Clone)]
pub struct ConfigManager {
    path: PathBuf,
    values: BTreeMap<String, String>,
}

impl ConfigManager {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into(), values: BTreeMap::new() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Read the file into memory. A missing or unreadable file leaves the map untouched,
    /// exactly as the C++ `std::ifstream` did.
    pub fn load(&mut self) {
        let Ok(text) = fs::read_to_string(&self.path) else {
            return;
        };
        for line in text.lines() {
            let Some(delimiter) = line.find('=') else { continue };
            let key = trim(&line[..delimiter]);
            let value = trim(&line[delimiter + 1..]);
            self.values.insert(key.to_string(), value.to_string());
        }
    }

    /// `""` for an absent key — the C++ `getConfigValue` contract, which every caller's
    /// "empty means fall back to the default" branch depends on.
    pub fn get(&self, key: &str) -> &str {
        self.values.get(key).map(String::as_str).unwrap_or("")
    }

    pub fn set(&mut self, key: &str, value: impl Into<String>) {
        self.values.insert(key.to_string(), value.into());
    }

    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Rewrite the whole file from the in-memory map. Keys are written in sorted order; the
    /// C++ used an `unordered_map` and had no stable order at all, so nothing can depend on
    /// it and a deterministic order is strictly better for diffing a deployed config.
    pub fn save(&self) -> io::Result<()> {
        let mut out = String::new();
        for (key, value) in &self.values {
            out.push_str(key);
            out.push('=');
            out.push_str(value);
            out.push('\n');
        }
        fs::write(&self.path, out)
    }
}

/// C++ `ConfigManager::trim`: spaces and tabs only. `\r` is deliberately NOT trimmed, so a
/// CRLF config behaves here exactly as it does in the C++ miner.
fn trim(text: &str) -> &str {
    let begin = match text.find(|c| c != ' ' && c != '\t') {
        Some(index) => index,
        None => return "",
    };
    let end = text.rfind(|c| c != ' ' && c != '\t').unwrap_or(begin);
    &text[begin..=end]
}

/// `std::stoi` semantics: skip leading whitespace, optional sign, then take as many digits
/// as parse. `None` where `stoi` would have thrown (`invalid_argument`/`out_of_range`).
///
/// This matters because `config.txt` is hand-edited: `devfee_permillage=10 ` and
/// `difficulty_margin=1000 # kib` both parsed in the C++ miner and must keep parsing.
pub fn stoi(text: &str) -> Option<i64> {
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() && (bytes[index] as char).is_whitespace() {
        index += 1;
    }
    let start = index;
    if index < bytes.len() && (bytes[index] == b'+' || bytes[index] == b'-') {
        index += 1;
    }
    let digits_start = index;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index == digits_start {
        return None;
    }
    text[start..index].parse::<i64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_only_spaces_and_tabs() {
        assert_eq!(trim("  a b  "), "a b");
        assert_eq!(trim("\t x\t"), "x");
        assert_eq!(trim("   "), "");
        assert_eq!(trim("a\r"), "a\r");
    }

    #[test]
    fn splits_on_the_first_equals_only() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.txt");
        std::fs::write(&path, "rpc = http://host/a=b\nnoequals\n = empty key\n").expect("write");
        let mut config = ConfigManager::new(&path);
        config.load();
        assert_eq!(config.get("rpc"), "http://host/a=b");
        assert_eq!(config.get("noequals"), "");
        assert_eq!(config.get(""), "empty key");
    }

    #[test]
    fn missing_file_loads_as_empty() {
        let mut config = ConfigManager::new("/nonexistent/treeminer/config.txt");
        config.load();
        assert_eq!(config.get("account_address"), "");
    }

    #[test]
    fn round_trips_through_save() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.txt");
        let mut config = ConfigManager::new(&path);
        config.set("account_address", "0xabc");
        config.set("devfee_permillage", "10");
        config.save().expect("save");
        let mut reloaded = ConfigManager::new(&path);
        reloaded.load();
        assert_eq!(reloaded.get("account_address"), "0xabc");
        assert_eq!(reloaded.get("devfee_permillage"), "10");
    }

    #[test]
    fn stoi_matches_cpp_prefix_parsing() {
        assert_eq!(stoi("42"), Some(42));
        assert_eq!(stoi("  -7 "), Some(-7));
        assert_eq!(stoi("12abc"), Some(12));
        assert_eq!(stoi("abc"), None);
        assert_eq!(stoi(""), None);
        assert_eq!(stoi("+5"), Some(5));
    }
}
