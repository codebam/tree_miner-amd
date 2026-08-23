//! Port of `src/EthereumAddressValidator.cpp`. A typo that survives validation binds every
//! find to an address nobody controls, so the checksum is required, not advisory.

use tm_core::address::keccak256_hex;
use tm_core::{is_valid_ethereum_address, salt_hex_for_address, to_checksum_address};

/// The EIP-55 specification's own examples.
const MIXED_CASE: &str = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";
const ALL_CAPS: &str = "0x52908400098527886E0F7030069857D2E4169EE7";
const ALL_LOWER: &str = "0xde709f2102306220921060314715629080e2fb77";

#[test]
fn keccak256_matches_the_known_empty_string_digest() {
    assert_eq!(
        keccak256_hex(""),
        "C5D2460186F7233C927E7DB2DCC703C0E500B653CA82273B7BFAD8045D85A470"
    );
}

#[test]
fn a_correctly_checksummed_address_is_accepted() {
    for address in [MIXED_CASE, ALL_CAPS, ALL_LOWER] {
        assert!(is_valid_ethereum_address(address), "rejected {address}");
        assert_eq!(to_checksum_address(address).as_deref(), Some(address));
    }
}

#[test]
fn flipping_one_letters_case_fails_the_checksum() {
    // 'A' at index 3 of the body is uppercase in the checksummed form; lowercasing just
    // that one character must be rejected.
    let mut bytes = MIXED_CASE.as_bytes().to_vec();
    assert_eq!(bytes[4], b'A');
    bytes[4] = b'a';
    let flipped = String::from_utf8(bytes).expect("ascii");
    assert_ne!(flipped, MIXED_CASE);
    assert!(!is_valid_ethereum_address(&flipped));

    // And the reverse direction: uppercasing a character that should be lowercase.
    let mut bytes = MIXED_CASE.as_bytes().to_vec();
    assert_eq!(bytes[2], b'5');
    bytes[3] = bytes[3].to_ascii_uppercase();
    let flipped = String::from_utf8(bytes).expect("ascii");
    assert_ne!(flipped, MIXED_CASE);
    assert!(!is_valid_ethereum_address(&flipped));
}

#[test]
fn an_all_lowercase_form_of_a_mixed_case_address_is_rejected() {
    // This is the common paste error: correct hex, no checksum information at all.
    assert!(!is_valid_ethereum_address(&MIXED_CASE.to_ascii_lowercase()));
    assert!(!is_valid_ethereum_address(&format!("0x{}", MIXED_CASE[2..].to_ascii_uppercase())));
}

#[test]
fn wrong_length_missing_prefix_and_non_hex_are_rejected() {
    let body = &MIXED_CASE[2..];
    // 39 and 41 body characters.
    assert!(!is_valid_ethereum_address(&format!("0x{}", &body[..39])));
    assert!(!is_valid_ethereum_address(&format!("0x{body}0")));
    // No 0x prefix, and a wrong prefix.
    assert!(!is_valid_ethereum_address(body));
    assert!(!is_valid_ethereum_address(&format!("0X{body}")));
    assert!(!is_valid_ethereum_address(&format!("xx{body}")));
    // Non-hex character inside the body.
    assert!(!is_valid_ethereum_address(&format!("0x{}z", &body[..39])));
    assert!(!is_valid_ethereum_address(""));
    assert!(!is_valid_ethereum_address("0x"));

    for bad in [body, "", "0x", &format!("0x{body}0")] {
        assert_eq!(to_checksum_address(bad), None, "accepted {bad}");
    }
}

#[test]
fn checksumming_is_idempotent_and_case_insensitive_on_input() {
    let expected = to_checksum_address(MIXED_CASE).expect("valid");
    assert_eq!(
        to_checksum_address(&MIXED_CASE.to_ascii_lowercase()).as_deref(),
        Some(expected.as_str())
    );
    assert_eq!(
        to_checksum_address(&format!("0x{}", MIXED_CASE[2..].to_ascii_uppercase())).as_deref(),
        Some(expected.as_str())
    );
    // The `0x` prefix itself is matched literally, as the C++ regex does: `0X` is not it.
    assert_eq!(to_checksum_address(&MIXED_CASE.to_ascii_uppercase()), None);
    assert_eq!(to_checksum_address(&expected).as_deref(), Some(expected.as_str()));
}

#[test]
fn the_salt_is_the_forty_hex_characters_without_the_prefix() {
    assert_eq!(
        salt_hex_for_address(MIXED_CASE),
        Some("5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed")
    );
    assert_eq!(salt_hex_for_address(MIXED_CASE).map(str::len), Some(40));
    // Same rejections as the validator, minus the checksum itself.
    assert_eq!(salt_hex_for_address(&MIXED_CASE[2..]), None);
    assert_eq!(salt_hex_for_address("0xabc"), None);
    assert_eq!(salt_hex_for_address(&format!("0x{}", "z".repeat(40))), None);
}
