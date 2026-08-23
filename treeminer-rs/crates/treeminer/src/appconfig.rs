//! Miner identity persisted in `config.txt`. Port of `src/AppConfig.{h,cpp}`.
//!
//! Two load modes exist for a reason: [`AppConfig::load`] is the interactive first-run /
//! repair path (it will ask the operator), while [`AppConfig::try_load`] is what runs when
//! the address and fee came from the command line and nothing may block on stdin.

use std::io;
use std::path::{Path, PathBuf};

use tm_core::is_valid_ethereum_address;

use crate::config::{stoi, ConfigManager};
use crate::prompt::Prompter;

pub const RED_INVALID_ACCOUNT: &str = "The account address in the configuration file is invalid.";
pub const RED_INVALID_DEVFEE: &str = "The devfee permillage in the configuration file is invalid. ";

#[derive(Debug, thiserror::Error)]
pub enum AppConfigError {
    #[error("{0}")]
    Io(#[from] io::Error),
    /// Operator input ended before a valid answer arrived. The C++ looped forever on EOF;
    /// failing closed is the only sane behaviour for a service with no tty.
    #[error("no operator input available for: {0}")]
    NoInput(String),
}

#[derive(Debug, Clone)]
pub struct AppConfig {
    config_file_name: PathBuf,
    account_address: String,
    eco_dev_addr: String,
    devfee_permillage: i32,
}

impl AppConfig {
    pub fn new(config_file_name: impl Into<PathBuf>) -> Self {
        Self {
            config_file_name: config_file_name.into(),
            account_address: String::new(),
            eco_dev_addr: String::new(),
            devfee_permillage: 0,
        }
    }

    pub fn path(&self) -> &Path {
        &self.config_file_name
    }

    pub fn account_address(&self) -> &str {
        &self.account_address
    }

    pub fn eco_dev_addr(&self) -> &str {
        &self.eco_dev_addr
    }

    pub fn devfee_permillage(&self) -> i32 {
        self.devfee_permillage
    }

    pub fn set_account_address(&mut self, address: impl Into<String>) {
        self.account_address = address.into();
    }

    pub fn set_eco_dev_addr(&mut self, address: impl Into<String>) {
        self.eco_dev_addr = address.into();
    }

    pub fn set_devfee_permillage(&mut self, devfee: i32) {
        self.devfee_permillage = devfee;
    }

    /// Interactive load. Reads `config.txt` if it exists, repairs anything invalid by
    /// asking, and always writes the address and fee back.
    pub fn load(&mut self, prompter: &mut dyn Prompter) -> Result<(), AppConfigError> {
        let mut config = ConfigManager::new(&self.config_file_name);
        let exists = self.config_file_name.is_file();
        if exists {
            config.load();
            self.account_address = config.get("account_address").to_string();
            if !valid_eip55(&self.account_address, prompter) {
                prompter.notify(RED_INVALID_ACCOUNT);
                prompter
                    .notify("Please enter a valid EIP-55 account address (Ethereum address) again.");
                self.account_address = prompt_for_eip55_address(prompter)?;
            }
            self.eco_dev_addr = config.get("ecodev_address").to_string();
            if !self.eco_dev_addr.is_empty() && !valid_eip55(&self.eco_dev_addr, prompter) {
                self.eco_dev_addr.clear();
            }
            match stoi(config.get("devfee_permillage")).filter(|v| is_valid_devfee_i64(*v)) {
                Some(value) => self.devfee_permillage = value as i32,
                None => {
                    prompter.notify(RED_INVALID_DEVFEE);
                    prompter.notify("Please enter a valid devfee per thousand (range 0 - 1000) again.");
                    self.devfee_permillage = prompt_for_devfee_permillage(prompter)?;
                }
            }
        } else {
            prompter.notify(
                "Welcome! It looks like this is your first time running the application. Let's set up the necessary configurations.",
            );
            self.account_address = prompt_for_eip55_address(prompter)?;
            self.devfee_permillage = prompt_for_devfee_permillage(prompter)?;
            prompter.notify(
                "All set! Your configurations are saved and the application is ready to use.",
            );
        }

        config.set("account_address", self.account_address.clone());
        config.set("devfee_permillage", self.devfee_permillage.to_string());
        config.save()?;
        Ok(())
    }

    /// Non-interactive load: take whatever is there, validate nothing. Matches the C++
    /// `tryLoad`, which is used when the CLI already supplied the identity.
    pub fn try_load(&mut self) {
        if !self.config_file_name.is_file() {
            return;
        }
        let mut config = ConfigManager::new(&self.config_file_name);
        config.load();
        self.account_address = config.get("account_address").to_string();
        self.eco_dev_addr = config.get("ecodev_address").to_string();
        if let Some(value) = stoi(config.get("devfee_permillage")) {
            self.devfee_permillage = value.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }

    /// `--saveConfig`: write the current console inputs back to the file. Unlike [`load`],
    /// this also persists `ecodev_address`.
    pub fn save(&self) -> Result<(), AppConfigError> {
        let mut config = ConfigManager::new(&self.config_file_name);
        config.set("account_address", self.account_address.clone());
        config.set("ecodev_address", self.eco_dev_addr.clone());
        config.set("devfee_permillage", self.devfee_permillage.to_string());
        config.save()?;
        Ok(())
    }
}

pub fn is_valid_devfee(devfee: i32) -> bool {
    (0..=1000).contains(&devfee)
}

fn is_valid_devfee_i64(devfee: i64) -> bool {
    (0..=1000).contains(&devfee)
}

fn valid_eip55(address: &str, prompter: &mut dyn Prompter) -> bool {
    if is_valid_ethereum_address(address) {
        true
    } else {
        prompter.notify("Invalid Ethereum address");
        false
    }
}

fn prompt_for_eip55_address(prompter: &mut dyn Prompter) -> Result<String, AppConfigError> {
    loop {
        let Some(answer) = prompter.prompt_line("Enter valid EIP-55 account address: ")? else {
            return Err(AppConfigError::NoInput("EIP-55 account address".into()));
        };
        let answer = answer.trim().to_string();
        if valid_eip55(&answer, prompter) {
            return Ok(answer);
        }
    }
}

fn prompt_for_devfee_permillage(prompter: &mut dyn Prompter) -> Result<i32, AppConfigError> {
    loop {
        let Some(answer) = prompter.prompt_line("Enter devfee per thousand (0-1000): ")? else {
            return Err(AppConfigError::NoInput("devfee per thousand".into()));
        };
        if let Some(value) = stoi(&answer) {
            if is_valid_devfee_i64(value) {
                return Ok(value as i32);
            }
        }
    }
}
