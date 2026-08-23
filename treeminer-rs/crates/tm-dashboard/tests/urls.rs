//! Advertised-URL rules. Port of the C++ `test_dashboard_contract.py` expectations, which
//! exist because the miner once told operators to open `http://0.0.0.0:42069`.

use std::net::IpAddr;

use tm_dashboard::{
    advertised_addresses, console_url, format_dashboard_url, is_loopback_dashboard_bind,
    is_valid_dashboard_bind, ready_message, StaticInterfaces,
};

fn ifaces(addrs: &[&str]) -> StaticInterfaces {
    StaticInterfaces(
        addrs
            .iter()
            .map(|a| a.parse::<IpAddr>().expect("test address"))
            .collect(),
    )
}

#[test]
fn bind_must_be_an_ip_literal() {
    assert!(is_valid_dashboard_bind("0.0.0.0"));
    assert!(is_valid_dashboard_bind("127.0.0.1"));
    assert!(is_valid_dashboard_bind("::"));
    assert!(is_valid_dashboard_bind("fd00::1"));
    assert!(!is_valid_dashboard_bind("localhost"));
    assert!(!is_valid_dashboard_bind("example.com"));
    assert!(!is_valid_dashboard_bind(""));
    assert!(!is_valid_dashboard_bind("0.0.0.0:42069"));
}

#[test]
fn loopback_binds_are_recognised() {
    assert!(is_loopback_dashboard_bind("127.0.0.1"));
    assert!(is_loopback_dashboard_bind("::1"));
    assert!(!is_loopback_dashboard_bind("0.0.0.0"));
    assert!(!is_loopback_dashboard_bind("192.168.1.5"));
}

#[test]
fn ipv6_urls_are_bracketed_and_ipv4_urls_are_not() {
    assert_eq!(
        format_dashboard_url("192.168.1.5", 42069),
        "http://192.168.1.5:42069"
    );
    assert_eq!(
        format_dashboard_url("fd12:3456::1", 42069),
        "http://[fd12:3456::1]:42069"
    );
    assert_eq!(format_dashboard_url("::1", 8080), "http://[::1]:8080");
}

#[test]
fn wildcard_bind_advertises_interface_addresses_not_the_wildcard() {
    let hosts = advertised_addresses("0.0.0.0", &ifaces(&["192.168.1.5", "fd00::5"]));
    assert_eq!(hosts, vec!["192.168.1.5".to_string(), "fd00::5".to_string()]);
    assert!(!hosts.iter().any(|h| h == "0.0.0.0"));
}

#[test]
fn wildcard_bind_skips_loopback_and_link_local_interfaces() {
    let hosts = advertised_addresses(
        "0.0.0.0",
        &ifaces(&["127.0.0.1", "169.254.7.7", "fe80::1", "10.0.0.9"]),
    );
    assert_eq!(hosts, vec!["10.0.0.9".to_string()]);
}

#[test]
fn wildcard_bind_orders_ipv4_before_ipv6_and_dedupes() {
    let hosts = advertised_addresses(
        "::",
        &ifaces(&["fd00::5", "10.0.0.9", "fd00::5", "10.0.0.9"]),
    );
    assert_eq!(hosts, vec!["10.0.0.9".to_string(), "fd00::5".to_string()]);
}

#[test]
fn wildcard_bind_without_usable_interfaces_falls_back_to_loopback() {
    assert_eq!(advertised_addresses("0.0.0.0", &ifaces(&[])), vec!["127.0.0.1".to_string()]);
    assert_eq!(advertised_addresses("::", &ifaces(&[])), vec!["::1".to_string()]);
}

#[test]
fn explicit_bind_advertises_itself_and_ignores_interfaces() {
    let discovered = ifaces(&["192.168.1.5"]);
    assert_eq!(
        advertised_addresses("10.1.2.3", &discovered),
        vec!["10.1.2.3".to_string()]
    );
    assert_eq!(
        advertised_addresses("127.0.0.1", &discovered),
        vec!["127.0.0.1".to_string()]
    );
    assert_eq!(
        advertised_addresses("fd00::9", &discovered),
        vec!["fd00::9".to_string()]
    );
    // Link-local: reachable by nobody worth telling, and not loopback either.
    assert!(advertised_addresses("169.254.4.4", &discovered).is_empty());
}

#[test]
fn console_url_prefers_the_first_reachable_address() {
    assert_eq!(
        console_url("0.0.0.0", 42069, &ifaces(&["192.168.1.5", "fd00::5"])),
        "http://192.168.1.5:42069"
    );
    assert_eq!(
        console_url("::", 42069, &ifaces(&["fd00::5"])),
        "http://[fd00::5]:42069"
    );
    // Nothing advertisable: never emit the wildcard, fall back to loopback.
    assert_eq!(
        console_url("169.254.4.4", 42069, &ifaces(&[])),
        "http://127.0.0.1:42069"
    );
}

#[test]
fn ready_message_for_a_wildcard_bind_lists_lan_urls() {
    let message = ready_message("0.0.0.0", 42069, &ifaces(&["192.168.1.5", "fd00::5"]));
    assert_eq!(
        message,
        "Dashboard listening on all interfaces, port 42069\n  \
         open  http://192.168.1.5:42069\n  open  http://[fd00::5]:42069\n"
    );
    assert!(!message.contains("http://0.0.0.0"));
}

#[test]
fn ready_message_for_loopback_says_this_machine_only() {
    assert_eq!(
        ready_message("127.0.0.1", 42069, &ifaces(&["192.168.1.5"])),
        "Dashboard ready — open http://127.0.0.1:42069 (this machine only)\n"
    );
    assert_eq!(
        ready_message("::1", 8080, &ifaces(&[])),
        "Dashboard ready — open http://[::1]:8080 (this machine only)\n"
    );
}

#[test]
fn ready_message_for_an_explicit_bind_names_that_address() {
    assert_eq!(
        ready_message("192.168.1.5", 42069, &ifaces(&["10.0.0.9"])),
        "Dashboard listening on 192.168.1.5:42069\n  open  http://192.168.1.5:42069\n"
    );
}
