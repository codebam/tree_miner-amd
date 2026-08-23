//! Bind validation and the URLs an operator can actually type into a browser.
//!
//! Port of the address half of `src/LocalServer.cpp`. The console binds `0.0.0.0` by
//! default so a rig can be reached from the LAN, which means the bind string is useless
//! as a URL: the startup banner must advertise interface addresses instead. Getting this
//! wrong is what `CHANGES-FROM-UPSTREAM.md` records as the "open http://0.0.0.0:42069"
//! bug, so the rules below are kept literal.

use std::fmt::Write as _;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket};

/// Default listen address: every interface, so the console is LAN-reachable.
pub const DEFAULT_BIND: &str = "0.0.0.0";
/// Default console port.
pub const DEFAULT_PORT: u16 = 42069;

/// `isValidDashboardBind`: an IP literal, never a hostname.
pub fn is_valid_dashboard_bind(address: &str) -> bool {
    address.parse::<IpAddr>().is_ok()
}

/// `isLoopbackDashboardBind`.
pub fn is_loopback_dashboard_bind(address: &str) -> bool {
    address == "127.0.0.1" || address == "::1"
}

fn is_wildcard_bind(address: &str) -> bool {
    address == "0.0.0.0" || address == "::"
}

/// `isUnusableAdvertisedHost`: wildcards, loopback and link-local addresses are never
/// worth printing — nobody else on the LAN can reach them.
fn is_unusable_advertised_host(host: &str) -> bool {
    if host.is_empty() || is_wildcard_bind(host) || is_loopback_dashboard_bind(host) {
        return true;
    }
    match host.parse::<IpAddr>() {
        // The C++ matches the "169.254."/"fe80:" text prefixes; matching the real
        // link-local ranges covers the same addresses and the fe80::/10 tail as well.
        Ok(IpAddr::V4(v4)) => v4.is_link_local() || v4.is_loopback() || v4.is_unspecified(),
        Ok(IpAddr::V6(v6)) => {
            is_ipv6_link_local(&v6) || v6.is_loopback() || v6.is_unspecified()
        }
        Err(_) => false,
    }
}

fn is_ipv6_link_local(addr: &Ipv6Addr) -> bool {
    addr.segments()[0] & 0xffc0 == 0xfe80
}

/// The addresses of this host, as the advertised-URL list wants them.
///
/// The C++ walks `getifaddrs`; doing that from Rust would need `unsafe`, which this
/// workspace confines to `tm-gpu`. Instead the default implementation asks the routing
/// table which source address would be used to reach off-box, which is precisely the
/// address an operator on the LAN can reach back on. Injectable so the URL rules can be
/// tested without depending on the host's interfaces.
pub trait InterfaceSource: Send + Sync {
    fn local_addresses(&self) -> Vec<IpAddr>;
}

/// Routing-table probe. No packets are sent: `connect` on a UDP socket only fixes the
/// local end, which is what `local_addr` then reports.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemInterfaces;

impl SystemInterfaces {
    fn probe(local: SocketAddr, remote: SocketAddr) -> Option<IpAddr> {
        let socket = UdpSocket::bind(local).ok()?;
        socket.connect(remote).ok()?;
        Some(socket.local_addr().ok()?.ip())
    }
}

impl InterfaceSource for SystemInterfaces {
    fn local_addresses(&self) -> Vec<IpAddr> {
        // Documentation-range destinations (RFC 5737 / RFC 3849): routable enough for the
        // kernel to pick an interface, never actually contacted.
        let v4 = Self::probe(
            SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
            SocketAddr::from((Ipv4Addr::new(192, 0, 2, 1), 9)),
        );
        let v6 = Self::probe(
            SocketAddr::from((Ipv6Addr::UNSPECIFIED, 0)),
            SocketAddr::from((
                Ipv6Addr::new(0x2001, 0x0db8, 0, 0, 0, 0, 0, 1),
                9,
            )),
        );
        v4.into_iter().chain(v6).collect()
    }
}

/// Fixed list, for tests and for callers that already know their addresses.
#[derive(Debug, Clone, Default)]
pub struct StaticInterfaces(pub Vec<IpAddr>);

impl InterfaceSource for StaticInterfaces {
    fn local_addresses(&self) -> Vec<IpAddr> {
        self.0.clone()
    }
}

/// `formatDashboardUrl`: IPv6 literals must be bracketed or the port parses as part of
/// the address.
pub fn format_dashboard_url(host: &str, port: u16) -> String {
    if host.contains(':') {
        format!("http://[{host}]:{port}")
    } else {
        format!("http://{host}:{port}")
    }
}

/// `dashboardAdvertisedAddresses`. Never returns the wildcard itself.
pub fn advertised_addresses(bind_address: &str, interfaces: &dyn InterfaceSource) -> Vec<String> {
    if !is_wildcard_bind(bind_address) {
        if !is_unusable_advertised_host(bind_address) {
            return vec![bind_address.to_string()];
        }
        if is_loopback_dashboard_bind(bind_address) {
            return vec![bind_address.to_string()];
        }
        return Vec::new();
    }

    let mut hosts: Vec<String> = Vec::new();
    // IPv4 first, then IPv6: the C++ concatenates the two lists in that order and the
    // first entry becomes the "open this" URL.
    let discovered = interfaces.local_addresses();
    for family_is_v4 in [true, false] {
        for addr in discovered.iter().filter(|a| a.is_ipv4() == family_is_v4) {
            let text = addr.to_string();
            if is_unusable_advertised_host(&text) || hosts.contains(&text) {
                continue;
            }
            hosts.push(text);
        }
    }
    if hosts.is_empty() {
        hosts.push(if bind_address == "::" { "::1" } else { "127.0.0.1" }.to_string());
    }
    hosts
}

/// `getConsoleUrl`: the single URL shown in the status line and in `console.open`.
pub fn console_url(bind_address: &str, port: u16, interfaces: &dyn InterfaceSource) -> String {
    let hosts = advertised_addresses(bind_address, interfaces);
    let host = hosts.first().map(String::as_str).unwrap_or("127.0.0.1");
    format_dashboard_url(host, port)
}

/// `dashboardReadyMessage`: the startup banner, verbatim in wording and layout because
/// operators grep for it and `main.cpp` writes it to `dashboard.url`.
pub fn ready_message(bind_address: &str, port: u16, interfaces: &dyn InterfaceSource) -> String {
    let hosts = advertised_addresses(bind_address, interfaces);
    let mut message = String::new();
    if is_loopback_dashboard_bind(bind_address) {
        let _ = writeln!(
            message,
            "Dashboard ready — open {} (this machine only)",
            format_dashboard_url(bind_address, port)
        );
        return message;
    }
    if is_wildcard_bind(bind_address) {
        let _ = writeln!(
            message,
            "Dashboard listening on all interfaces, port {port}"
        );
    } else {
        let _ = writeln!(message, "Dashboard listening on {bind_address}:{port}");
    }
    for host in &hosts {
        let _ = writeln!(message, "  open  {}", format_dashboard_url(host, port));
    }
    message
}
