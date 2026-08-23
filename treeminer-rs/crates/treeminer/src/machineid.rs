//! Stable per-rig worker identity. Port of `src/MachineIDGetter.{h,cpp}` plus
//! `getMachineId()` from `src/main.cpp`.
//!
//! The id must be stable across restarts (the server groups a rig's submissions by it) and
//! must differ between two miners on the same box using different GPUs — hence the device
//! list is folded in. The facts are gathered into [`MachineFacts`] so the derivation can be
//! tested without pretending to be a different machine.

use std::collections::BTreeSet;
use std::fs;

use sha2::{Digest, Sha256};
use tm_argon2::keygen::RandomHexKeyGenerator;

/// Everything the C++ `MachineIDGetter::getMachineId()` reads on Linux.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MachineFacts {
    pub hostname: String,
    pub user: String,
    /// Verbatim `/proc/cpuinfo`, empty when unreadable.
    pub cpu_info: String,
    /// Uppercase `AA:BB:CC:DD:EE:FF`, one per link-layer interface. Order is irrelevant —
    /// [`machine_identity`] sorts them, which is what makes the id survive an interface
    /// enumeration order change across reboots.
    pub mac_addresses: Vec<String>,
}

impl MachineFacts {
    pub fn from_system() -> Self {
        Self {
            hostname: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
            user: std::env::var("USER").unwrap_or_else(|_| "unknown".to_string()),
            cpu_info: fs::read_to_string("/proc/cpuinfo").unwrap_or_default(),
            mac_addresses: read_mac_addresses(),
        }
    }
}

/// Link-layer addresses of every interface, in `/sys` order. The C++ walked `getifaddrs`
/// with `SIOCGIFHWADDR`; `/sys/class/net/*/address` is the same kernel data without an
/// ioctl, and loopback's all-zero address is included by both.
fn read_mac_addresses() -> Vec<String> {
    let Ok(entries) = fs::read_dir("/sys/class/net") else {
        return Vec::new();
    };
    let mut addresses = Vec::new();
    for entry in entries.flatten() {
        let Ok(text) = fs::read_to_string(entry.path().join("address")) else {
            continue;
        };
        let mac = text.trim().to_ascii_uppercase();
        if mac.len() == 17 {
            addresses.push(mac);
        }
    }
    addresses
}

/// The raw identity string: `hostname_user_<cpuinfo><sorted concatenated MACs>`.
pub fn machine_identity(facts: &MachineFacts) -> String {
    let mut macs = facts.mac_addresses.clone();
    macs.sort();
    format!("{}_{}_{}{}", facts.hostname, facts.user, facts.cpu_info, macs.concat())
}

/// `sha256(identity + device_info)` truncated to 16 hex characters, as the server expects.
pub fn derive_machine_id(identity: &str, device_info: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(identity.as_bytes());
    hasher.update(device_info.as_bytes());
    hex::encode(hasher.finalize())[..16].to_string()
}

/// The C++ `oss_usedDevices` string: every selected index followed by a comma (`"0,1,"`).
pub fn device_info_text(devices: &BTreeSet<i32>) -> String {
    let mut text = String::new();
    for device in devices {
        text.push_str(&device.to_string());
        text.push(',');
    }
    text
}

/// Full `getMachineId()`: derive from the real machine, falling back to a random id when
/// the machine has no identity at all (containers with no `/proc`, no NICs). A random id is
/// better than a shared empty-string id, which would merge unrelated rigs server-side.
pub fn machine_id_for_devices(devices: &BTreeSet<i32>) -> String {
    let identity = machine_identity(&MachineFacts::from_system());
    if identity.is_empty() {
        let mut generator = RandomHexKeyGenerator::new("", 64);
        return derive_machine_id(&generator.next_random_key(), "");
    }
    derive_machine_id(&identity, &device_info_text(devices))
}

/// Parse `--device=1,2,7`. Port of `parseDeviceList()`: out-of-range and unparseable
/// entries are skipped, and an empty result means "use every device".
pub fn parse_device_list(device_list_text: &str, device_count: i32) -> BTreeSet<i32> {
    let mut devices = BTreeSet::new();
    for item in device_list_text.split(',') {
        let Some(value) = crate::config::stoi(item) else { continue };
        let Ok(device_id) = i32::try_from(value) else { continue };
        if device_id < 0 || device_id >= device_count {
            continue;
        }
        devices.insert(device_id);
    }
    if devices.is_empty() {
        devices.extend(0..device_count);
    }
    devices
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> MachineFacts {
        MachineFacts {
            hostname: "rig-01".into(),
            user: "miner".into(),
            cpu_info: "model name : Ryzen\n".into(),
            mac_addresses: vec!["AA:BB:CC:DD:EE:FF".into(), "00:00:00:00:00:00".into()],
        }
    }

    #[test]
    fn identity_is_independent_of_interface_enumeration_order() {
        let mut reordered = facts();
        reordered.mac_addresses.reverse();
        assert_eq!(machine_identity(&facts()), machine_identity(&reordered));
    }

    #[test]
    fn id_is_sixteen_hex_characters() {
        let id = derive_machine_id(&machine_identity(&facts()), "0,");
        assert_eq!(id.len(), 16);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn id_is_stable_for_the_same_device_list_and_differs_across_lists() {
        let identity = machine_identity(&facts());
        let one: BTreeSet<i32> = [0, 1].into_iter().collect();
        let other: BTreeSet<i32> = [0, 2].into_iter().collect();

        let first = derive_machine_id(&identity, &device_info_text(&one));
        let again = derive_machine_id(&identity, &device_info_text(&one));
        let different = derive_machine_id(&identity, &device_info_text(&other));

        assert_eq!(first, again);
        assert_ne!(first, different);
    }

    #[test]
    fn a_different_machine_yields_a_different_id() {
        let mut other = facts();
        other.hostname = "rig-02".into();
        let devices: BTreeSet<i32> = [0].into_iter().collect();
        assert_ne!(
            derive_machine_id(&machine_identity(&facts()), &device_info_text(&devices)),
            derive_machine_id(&machine_identity(&other), &device_info_text(&devices))
        );
    }

    #[test]
    fn device_info_text_matches_the_cpp_trailing_comma_form() {
        assert_eq!(device_info_text(&[0, 1, 2].into_iter().collect()), "0,1,2,");
        assert_eq!(device_info_text(&BTreeSet::new()), "");
    }

    #[test]
    fn device_list_skips_bad_entries_and_defaults_to_all() {
        assert_eq!(parse_device_list("1,2,7", 4), [1, 2].into_iter().collect());
        assert_eq!(parse_device_list("", 3), [0, 1, 2].into_iter().collect());
        assert_eq!(parse_device_list("abc,-1,9", 2), [0, 1].into_iter().collect());
        assert_eq!(parse_device_list(" 1 , 0 ", 4), [0, 1].into_iter().collect());
    }
}
