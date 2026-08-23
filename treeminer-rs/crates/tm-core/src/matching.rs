//! Find-detection rules. Port of `src/hashapi/HashApiMatching.cpp` — the thresholds here
//! decide what counts as a payable find, so they must not drift from the C++ miner.

/// A superblock is a digest carrying at least 50 uppercase letters.
pub fn is_superblock_hash(hash: &str) -> bool {
    hash.chars().filter(|c| c.is_ascii_uppercase()).count() >= 50
}

/// XUNI matches are `XUNI` followed by a digit, anywhere in the digest.
pub fn has_xuni_match(hash: &str) -> bool {
    let bytes = hash.as_bytes();
    bytes
        .windows(5)
        .any(|window| &window[..4] == b"XUNI" && window[4].is_ascii_digit())
}

/// True while the XUNI window is open: the :55-:05 span around the top of the hour, in
/// local time (matching `MiningCommon::isWithinXuniWindow`).
pub fn is_within_xuni_window_at(minute: u32) -> bool {
    !(5..55).contains(&minute)
}
