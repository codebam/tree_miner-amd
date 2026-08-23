//! Read-only operator console for TreeMiner.
//!
//! Rust port of `src/LocalServer.cpp` (Crow → axum), `src/DashboardPage.h` and
//! `src/StatReporter.cpp`. Three things are load-bearing and must not drift:
//!
//! 1. **Read-only.** Only GET routes, only reported state. No control endpoint, no key,
//!    seed or config material; the sole identity in any payload is the public reward
//!    address. The default bind is `0.0.0.0`, so any write route here would be a write
//!    route for the whole LAN.
//! 2. **Field names.** The embedded page and third-party fleet dashboards parse these
//!    payloads. [`json`] spells every key out literally for that reason.
//! 3. **Advertised URLs.** The banner prints addresses an operator can actually open,
//!    never the wildcard bind, with IPv6 literals bracketed. See [`url`].
//!
//! The mining loop lives in another crate, so this one reads a [`StatsSnapshot`] through
//! the [`StatsSource`] trait instead of touching mining state.

pub mod json;
pub mod page;
pub mod server;
pub mod stats;
pub mod url;

pub use json::{
    platform_payload, rig_payload, stat_upload_payload, stats_payload, REPORTED_VERSION,
    STATS_CACHE_SECONDS,
};
pub use page::PAGE;
pub use server::{
    router, DashboardConfig, DashboardError, DashboardServer, HASHFIELD_ASSET,
};
pub use stats::{
    GpuStat, GpuTelemetry, JournalCounts, LastSubmissionState, NetworkState, PlatformLease,
    PlatformStatus, SharedStats, StatsSnapshot, StatsSource, SubmissionMetrics, TreeminerStats,
    POWER_UNAVAILABLE,
};
pub use url::{
    advertised_addresses, console_url, format_dashboard_url, is_loopback_dashboard_bind,
    is_valid_dashboard_bind, ready_message, InterfaceSource, StaticInterfaces, SystemInterfaces,
    DEFAULT_BIND, DEFAULT_PORT,
};
