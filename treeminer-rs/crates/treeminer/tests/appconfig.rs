//! `config.txt` identity handling: first-run setup, repairing invalid stored values, and
//! the non-interactive `try_load` used when the command line already supplied an identity.

use treeminer::{AppConfig, ConfigManager, NoPrompter, ScriptedPrompter};

fn address(seed: u8) -> String {
    let body: String = (0..40).map(|i| char::from(b'a' + ((seed + i as u8) % 6))).collect();
    tm_core::to_checksum_address(&format!("0x{body}")).expect("checksum")
}

#[test]
fn first_run_asks_for_an_identity_and_writes_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.txt");
    let miner = address(0);

    let mut prompter = ScriptedPrompter::new([miner.clone(), "10".to_string()]);
    let mut config = AppConfig::new(&path);
    config.load(&mut prompter).expect("load");

    assert_eq!(config.account_address(), miner);
    assert_eq!(config.devfee_permillage(), 10);
    assert!(prompter.notices.iter().any(|n| n.starts_with("Welcome!")));

    let mut written = ConfigManager::new(&path);
    written.load();
    assert_eq!(written.get("account_address"), miner);
    assert_eq!(written.get("devfee_permillage"), "10");
}

/// An address the operator cannot spend to is worse than no miner at all, so a stored value
/// that fails the EIP-55 checksum is re-asked rather than used.
#[test]
fn an_invalid_stored_address_is_re_prompted_until_valid() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.txt");
    let good = address(3);
    std::fs::write(&path, "account_address=0xdeadbeef\ndevfee_permillage=5\n").expect("write");

    let mut prompter = ScriptedPrompter::new(["still-not-an-address".to_string(), good.clone()]);
    let mut config = AppConfig::new(&path);
    config.load(&mut prompter).expect("load");

    assert_eq!(config.account_address(), good);
    assert_eq!(config.devfee_permillage(), 5);
    assert_eq!(prompter.prompts.len(), 2);
}

#[test]
fn an_out_of_range_devfee_is_re_prompted() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.txt");
    let good = address(2);
    std::fs::write(&path, format!("account_address={good}\ndevfee_permillage=5000\n"))
        .expect("write");

    let mut prompter = ScriptedPrompter::new(["1001".to_string(), "250".to_string()]);
    let mut config = AppConfig::new(&path);
    config.load(&mut prompter).expect("load");
    assert_eq!(config.devfee_permillage(), 250);
}

/// An invalid ecosystem address is dropped, not fatal: the miner still has a valid reward
/// address of its own and must keep hashing.
#[test]
fn an_invalid_ecosystem_address_is_blanked() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.txt");
    let good = address(2);
    std::fs::write(
        &path,
        format!("account_address={good}\necodev_address=0xnope\ndevfee_permillage=5\n"),
    )
    .expect("write");

    let mut config = AppConfig::new(&path);
    config.load(&mut NoPrompter).expect("load");
    assert_eq!(config.eco_dev_addr(), "");
}

#[test]
fn try_load_never_prompts_and_never_writes() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.txt");
    std::fs::write(&path, "account_address=0xdeadbeef\ndevfee_permillage=notanumber\n")
        .expect("write");

    let mut config = AppConfig::new(&path);
    config.try_load();
    assert_eq!(config.account_address(), "0xdeadbeef");
    assert_eq!(config.devfee_permillage(), 0);
    assert_eq!(
        std::fs::read_to_string(&path).expect("read"),
        "account_address=0xdeadbeef\ndevfee_permillage=notanumber\n"
    );
}

#[test]
fn try_load_on_a_missing_file_leaves_the_defaults() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = AppConfig::new(dir.path().join("absent.txt"));
    config.try_load();
    assert_eq!(config.account_address(), "");
    assert_eq!(config.devfee_permillage(), 0);
}

/// A service run with no tty must fail closed rather than block forever on stdin, which is
/// what the C++ `do { cin >> ... } while(!valid)` loop did.
#[test]
fn load_fails_closed_when_there_is_nobody_to_ask() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = AppConfig::new(dir.path().join("config.txt"));
    let error = config.load(&mut NoPrompter).expect_err("must fail");
    assert!(error.to_string().contains("no operator input"), "{error}");
}

#[test]
fn save_persists_all_three_identity_keys() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("config.txt");
    let mut config = AppConfig::new(&path);
    config.set_account_address(address(1));
    config.set_eco_dev_addr(address(4));
    config.set_devfee_permillage(33);
    config.save().expect("save");

    let mut written = ConfigManager::new(&path);
    written.load();
    assert_eq!(written.get("account_address"), address(1));
    assert_eq!(written.get("ecodev_address"), address(4));
    assert_eq!(written.get("devfee_permillage"), "33");
}
