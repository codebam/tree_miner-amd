//! Port of the encoding half of `src/hashapi/HashApiEncoding.cpp` and
//! `src/treeminer/PhcAssembler.h`. The output of these functions goes into the PHC string
//! the server verifies, so a one-character drift is an invalid find.

use tm_core::encoding::{
    base64_encode, base64_encoded_len, hex_to_bytes, phc_digest, HexError,
};
use tm_core::assemble_phc;

/// The exact alphabet from `kBase64Chars`, in order.
const CPP_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[test]
fn base64_uses_the_cpp_alphabet_in_order() {
    // Six-bit values 0..64 packed three-per-four-chars reproduce the alphabet exactly.
    let mut bytes = Vec::new();
    for group in 0..16u8 {
        let a = group * 4;
        bytes.push(a << 2 | (a + 1) >> 4);
        bytes.push(((a + 1) & 0x0f) << 4 | (a + 2) >> 2);
        bytes.push(((a + 2) & 0x03) << 6 | (a + 3));
    }
    assert_eq!(base64_encode(&bytes), CPP_ALPHABET);
}

#[test]
fn base64_emits_no_padding_for_any_remainder() {
    // remainder 0: whole groups only.
    assert_eq!(base64_encode(b"abc"), "YWJj");
    assert_eq!(base64_encode(b"abcdef"), "YWJjZGVm");
    // remainder 1: two characters, no "==".
    assert_eq!(base64_encode(b"a"), "YQ");
    assert_eq!(base64_encode(b"abcd"), "YWJjZA");
    // remainder 2: three characters, no "=".
    assert_eq!(base64_encode(b"ab"), "YWI");
    assert_eq!(base64_encode(b"abcde"), "YWJjZGU");
    // empty input encodes to nothing at all.
    assert_eq!(base64_encode(b""), "");

    for encoded in [
        base64_encode(b"a"),
        base64_encode(b"ab"),
        base64_encode(b"abc"),
    ] {
        assert!(!encoded.contains('='), "padding leaked: {encoded}");
    }
}

#[test]
fn base64_high_bit_bytes_use_the_last_two_alphabet_slots() {
    // 0xfb 0xff 0xfe covers the '+' and '/' code points that a URL-safe alphabet would move.
    assert_eq!(base64_encode(&[0xfb, 0xff, 0xfe]), "+//+");
}

#[test]
fn base64_encoded_len_agrees_with_the_encoder_for_every_remainder() {
    for len in 0..=32usize {
        let bytes = vec![0xa5u8; len];
        assert_eq!(base64_encoded_len(len), base64_encode(&bytes).len(), "len {len}");
    }
    // The 64-byte digest the miner actually encodes: 21 full groups plus one leftover byte.
    assert_eq!(base64_encoded_len(64), 86);
}

#[test]
fn hex_round_trips_in_both_cases() {
    assert_eq!(hex_to_bytes(""), Ok(Vec::new()));
    assert_eq!(hex_to_bytes("00ff10"), Ok(vec![0x00, 0xff, 0x10]));
    // The C++ nibble() accepts a-f and A-F alike.
    assert_eq!(hex_to_bytes("DeAdBeEf"), hex_to_bytes("deadbeef"));

    let address = "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc";
    let bytes = hex_to_bytes(address).expect("valid hex");
    assert_eq!(bytes.len(), 20);
    assert_eq!(hex::encode(&bytes), address);
}

#[test]
fn hex_rejects_odd_length_and_non_hex() {
    assert_eq!(hex_to_bytes("abc"), Err(HexError::OddLength));
    assert_eq!(hex_to_bytes("zz"), Err(HexError::NonHexCharacter));
    // A '0x' prefix is not stripped here — that belongs to the caller, so 'x' is non-hex.
    assert_eq!(hex_to_bytes("0xff"), Err(HexError::NonHexCharacter));
    assert_eq!(hex_to_bytes("ab  cd"), Err(HexError::NonHexCharacter));
    // Odd length is checked before the character scan, as in the C++ hexToBytes.
    assert_eq!(hex_to_bytes("zzz"), Err(HexError::OddLength));
}

#[test]
fn assemble_phc_matches_the_port_md_reference_vector() {
    let phc = assemble_phc(
        8,
        "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc",
        "2PKfnaEX2s+Yf/Drzi92D8HJ+B6K+FppyT7g5glp2knIMlFGWhnyOb9r1QIPf0GaVUEw8KumqQZ/pK2dkNTDxA",
    )
    .expect("valid salt");
    assert_eq!(
        phc,
        "$argon2id$v=19$m=8,t=1,p=1$5LsYR4G7yccATo2v1Km0nSA7ybw$\
         2PKfnaEX2s+Yf/Drzi92D8HJ+B6K+FppyT7g5glp2knIMlFGWhnyOb9r1QIPf0GaVUEw8KumqQZ/pK2dkNTDxA"
    );
}

#[test]
fn assemble_phc_shape_is_six_dollar_separated_fields() {
    let phc = assemble_phc(60000, "e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc", "DIGEST").unwrap();
    let fields: Vec<&str> = phc.split('$').collect();
    // A leading '$' makes the first field empty, exactly as the C++ string does.
    assert_eq!(fields.len(), 6);
    assert_eq!(fields[0], "");
    assert_eq!(fields[1], "argon2id");
    assert_eq!(fields[2], "v=19");
    // The memory cost baked in is the batch's, not a global — the stale-difficulty fix.
    assert_eq!(fields[3], "m=60000,t=1,p=1");
    assert_eq!(fields[4], "5LsYR4G7yccATo2v1Km0nSA7ybw");
    assert_eq!(fields[5], "DIGEST");
}

#[test]
fn assemble_phc_propagates_a_bad_salt_instead_of_panicking() {
    assert_eq!(assemble_phc(8, "abc", "D"), Err(HexError::OddLength));
    assert_eq!(assemble_phc(8, "zz", "D"), Err(HexError::NonHexCharacter));
}

#[test]
fn phc_digest_takes_everything_after_the_last_dollar() {
    let phc = "$argon2id$v=19$m=8,t=1,p=1$5LsYR4G7yccATo2v1Km0nSA7ybw$DIGEST";
    assert_eq!(phc_digest(phc), Some("DIGEST"));
    // A digest containing '+' and '/' survives; only '$' separates.
    assert_eq!(phc_digest("$a$b$c+d/e"), Some("c+d/e"));
    // No '$' at all, or nothing after the last one, is not a digest.
    assert_eq!(phc_digest("nodollars"), None);
    assert_eq!(phc_digest("$argon2id$"), None);
    assert_eq!(phc_digest(""), None);
}
