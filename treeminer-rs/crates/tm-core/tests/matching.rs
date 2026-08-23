//! Port of `src/hashapi/HashApiMatching.cpp` and `MiningCommon::isWithinXuniWindow`.
//! These thresholds decide what is payable, so the boundary cases are pinned exactly.

use tm_core::{has_xuni_match, is_superblock_hash, is_within_xuni_window_at};

#[test]
fn superblock_threshold_sits_at_exactly_fifty_uppercase() {
    assert!(!is_superblock_hash(&"A".repeat(49)));
    assert!(is_superblock_hash(&"A".repeat(50)));
    assert!(is_superblock_hash(&"A".repeat(51)));
}

#[test]
fn superblock_counts_only_uppercase_letters() {
    // Digits and '+'/'/' are in the base64 alphabet but are not uppercase letters.
    assert!(!is_superblock_hash(&"1".repeat(200)));
    assert!(!is_superblock_hash(&"a".repeat(200)));
    assert!(!is_superblock_hash(&"+/".repeat(200)));
    // 49 uppercase letters padded with non-letters stays below the threshold.
    assert!(!is_superblock_hash(&format!("{}{}", "Z".repeat(49), "z9+/".repeat(20))));
    assert!(is_superblock_hash(&format!("{}{}", "Z".repeat(50), "z9+/".repeat(20))));
    assert!(!is_superblock_hash(""));
}

#[test]
fn xuni_match_requires_a_trailing_digit() {
    assert!(has_xuni_match("XUNI0"));
    assert!(has_xuni_match("XUNI9"));
    // A bare XUNI, or one followed by a letter or the end of the string, is not a match.
    assert!(!has_xuni_match("XUNI"));
    assert!(!has_xuni_match("XUNIX"));
    assert!(!has_xuni_match("XUNIa"));
    assert!(!has_xuni_match("XUNI+"));
    // Case sensitive.
    assert!(!has_xuni_match("xuni1"));
    assert!(!has_xuni_match("Xuni1"));
    assert!(!has_xuni_match(""));
}

#[test]
fn xuni_matches_at_any_offset() {
    assert!(has_xuni_match("XUNI5tail"));
    assert!(has_xuni_match("head$XUNI5"));
    assert!(has_xuni_match("head$XUNI5$tail"));
    assert!(has_xuni_match(&format!("{}XUNI7", "a".repeat(120))));
}

#[test]
fn xuni_keeps_scanning_past_a_non_digit_occurrence() {
    // The C++ loop restarts the search at offset+1 rather than giving up on the first hit;
    // a digest can carry a dud "XUNIX" before the real one.
    assert!(has_xuni_match("XUNIXXUNI4"));
    assert!(has_xuni_match("XUNI-XUNI-XUNI8"));
    assert!(!has_xuni_match("XUNIXUNIXUNI"));
    // Overlapping prefixes: "XUNIXUNI3" hits on the second occurrence at offset 4.
    assert!(has_xuni_match("XUNIXUNI3"));
}

#[test]
fn xuni_window_boundaries_are_minute_five_and_fifty_five() {
    // Open through :04, shut from :05.
    assert!(is_within_xuni_window_at(0));
    assert!(is_within_xuni_window_at(4));
    assert!(!is_within_xuni_window_at(5));
    // Shut through :54, open from :55.
    assert!(!is_within_xuni_window_at(54));
    assert!(is_within_xuni_window_at(55));
    assert!(is_within_xuni_window_at(59));
}

#[test]
fn the_xuni_window_is_open_for_ten_minutes_an_hour() {
    let open = (0..60).filter(|m| is_within_xuni_window_at(*m)).count();
    assert_eq!(open, 10);
}
