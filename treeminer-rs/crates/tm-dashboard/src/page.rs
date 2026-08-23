//! The embedded console page. Port of `src/DashboardPage.h`, whose whole content was one
//! raw string literal; here it is the same bytes in a real `.html` file so editors and
//! linters can work on it.

/// The single-page console. It polls `/api/rig` every 2.5s and needs nothing else — no
/// external script, style or font, because a rig LAN is often offline.
pub const PAGE: &str = include_str!("../assets/dashboard.html");

#[cfg(test)]
mod tests {
    use super::PAGE;

    /// The JSON keys the page reads. They are the wire contract `tests/routes.rs` pins, so
    /// they are allowed to keep their historical spelling even where it names a vendor.
    const WIRE_KEYS: &[&str] = &[
        "engine.cuda_streams",
        "engine.gpu_devices",
        "engine.running",
        "engine.fatal_durability_failure",
        "engine.uptime_seconds",
        "engine.difficulty",
        "engine.cpu_workers",
        "engine.cpu_hashrate",
        "engine.gpu_hashrate",
        "engine.total_hashrate",
        "finds.xnm",
        "finds.xuni",
        "finds.super",
        "finds.rejected",
        "delivery.network",
        "delivery.last_submission_age_seconds",
        "delivery.last_submission",
        "delivery.queued_xnm",
        "delivery.queued_xuni",
        "delivery.queued_total",
        "identity.name",
        "identity.machine_id",
        "identity.address",
    ];

    #[test]
    fn page_is_self_contained_and_polls_the_rig_endpoint() {
        assert!(PAGE.starts_with("<!doctype html>"));
        assert!(PAGE.trim_end().ends_with("</html>"));
        assert!(PAGE.contains("fetch('/api/rig'"));
        // Operator language the C++ dashboard contract test pins.
        assert!(PAGE.contains("UPSTREAM OFFLINE"));
        assert!(PAGE.contains("Mining continues"));
        assert!(PAGE.contains("Q_XNM") && PAGE.contains("Q_XUNI"));
    }

    #[test]
    fn page_loads_no_third_party_origin() {
        for marker in ["src=\"http", "href=\"http", "//cdn.", "googleapis"] {
            assert!(!PAGE.contains(marker), "page references {marker}");
        }
    }

    #[test]
    fn page_reads_every_field_the_rig_payload_carries() {
        for key in WIRE_KEYS {
            assert!(PAGE.contains(key), "the page never reads `{key}`");
        }
    }

    /// The miner runs AMD kernels today and NVIDIA is untested, so no visible string may
    /// name a vendor. The JSON keys above are exempt: renaming them would break the
    /// third-party dashboards `tests/routes.rs` protects.
    #[test]
    fn page_shows_no_vendor_specific_wording() {
        let mut visible = PAGE.to_string();
        for key in WIRE_KEYS {
            visible = visible.replace(key, "");
        }
        let lower = visible.to_lowercase();
        for word in [
            "cuda", "nvidia", "geforce", "nvml", "radeon", "rocm", "hip", "opencl",
        ] {
            assert!(
                !lower.contains(word),
                "page shows vendor-specific wording `{word}` outside a wire key"
            );
        }
        // ...and it still says the vendor-neutral thing in that label's place.
        assert!(PAGE.contains("GPU stream telemetry"));
        assert!(PAGE.contains(">GPU streams<"));
    }

    /// A submission outcome is only ever labelled with an age when there is one, and an
    /// old one is marked stale rather than left looking current.
    #[test]
    fn submission_age_rendering_handles_the_no_submission_case() {
        assert!(PAGE.contains("delivery.last_submission_age_seconds"));
        // Null age (nothing submitted) and the "none" state both fall back to the bare label.
        assert!(PAGE.contains("submissionAge != null && submissionState !== 'NONE'"));
        assert!(PAGE.contains("const STALE_SUBMISSION_SECONDS = 300;"));
        assert!(PAGE.contains(
            "$('last-submission').classList.toggle('stale', aged && Number(submissionAge) >= STALE_SUBMISSION_SECONDS);"
        ));
        // `ago()` must round down and never render a bare number of seconds as minutes.
        assert!(PAGE.contains("if (s < 60) return `${s}s`;"));
    }

    /// The three states the page must not swallow: an unpersistable disk, a stopped
    /// engine, and server-rejected finds.
    #[test]
    fn page_surfaces_the_failure_states_the_payload_reports() {
        assert!(PAGE.contains("DURABILITY FAILURE"));
        assert!(PAGE.contains("$('durability').classList.toggle('show', !!engine.fatal_durability_failure);"));
        assert!(PAGE.contains("engine.running === false"));
        assert!(PAGE.contains("engine stopped"));
        assert!(PAGE.contains("text('rejected', compact(finds.rejected));"));
        // Probing is not the same claim as offline.
        assert!(PAGE.contains("UPSTREAM PROBING"));
    }

    /// Nothing on the page may hardcode the console port: `--dashboard-port` moves it.
    #[test]
    fn page_does_not_hardcode_the_console_port() {
        assert!(PAGE.contains("LOCAL_OPS::${location.port"));
        // The literal only survives as the pre-render placeholder in the markup.
        assert_eq!(PAGE.matches("42069").count(), 1);
    }
}
