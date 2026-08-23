//! The embedded console page. Port of `src/DashboardPage.h`, whose whole content was one
//! raw string literal; here it is the same bytes in a real `.html` file so editors and
//! linters can work on it.

/// The single-page console. It polls `/api/rig` every 2.5s and needs nothing else — no
/// external script, style or font, because a rig LAN is often offline.
pub const PAGE: &str = include_str!("../assets/dashboard.html");

#[cfg(test)]
mod tests {
    use super::PAGE;

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
}
