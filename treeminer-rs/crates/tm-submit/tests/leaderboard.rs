//! Leaderboard self-confirmation: parsing a real captured response, the case-insensitive
//! match that the whole feature hinges on, and every way the answer can be "we could not
//! ask" without that being mistaken for "our blocks are missing".
//!
//! `tests/fixtures/leaderboard.json` is a genuine `GET https://xenblocks.io/v1/leaderboard`
//! body, trimmed from 500 miners to five (ranks 1, 2, 47, 499, 500) — the real 114 KB has no
//! business in the repo, but every value below is the server's own.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;

use tm_submit::leaderboard::{read_capped, MAX_LEADERBOARD_BYTES};
use tm_submit::{AccountStanding, Leaderboard, LeaderboardClient, LeaderboardError};

const CAPTURED: &str = include_str!("fixtures/leaderboard.json");
/// Rank 1 as the server spells it: EIP-55 mixed case.
const TOP_ACCOUNT: &str = "0x18C1c90101aA2D04B62a8Fa80fb8D9a574362079";

fn board() -> Leaderboard {
    Leaderboard::parse(CAPTURED).expect("captured response parses")
}

#[test]
fn parses_the_captured_live_response() {
    let board = board();
    // `difficulty` arrives as a JSON *string* while `blocks` is a number; both land as u64.
    assert_eq!(board.difficulty, Some(100));
    assert_eq!(board.total_blocks, Some(93_803_933));
    assert_eq!(board.total_miners, Some(91));
    assert_eq!(board.total_hash_rate, Some(28.0));
    assert_eq!(board.entries.len(), 5);

    let top = &board.entries[0];
    assert_eq!(top.account, TOP_ACCOUNT);
    assert_eq!(top.rank, Some(1));
    assert_eq!(top.blocks, Some(4_790_829));
    assert_eq!(top.super_blocks, Some(5806));
    assert_eq!(top.hash_rate, Some(100_000.0));
    assert_eq!(
        top.sol_address.as_deref(),
        Some("93Hd48EXeid4gsHoVSXf33ZwcRe1tHh2KwfMH96i5Jux")
    );

    // ~15% of live entries carry no Solana address. Not an error, not a parse failure.
    let no_sol = &board.entries[2];
    assert_eq!(no_sol.rank, Some(47));
    assert_eq!(no_sol.sol_address, None);

    // The xnm/xuni/xblk token totals are 1e21-scale floats we deliberately do not model;
    // their presence must not disturb anything above.
    assert_eq!(board.entries[4].rank, Some(500));
}

#[test]
fn finds_the_account_whatever_the_case() {
    let board = board();
    // The server lowercases on storage (gpage.py:382-384) while our config is EIP-55, so an
    // exact match would silently never find us — and would look exactly like "not ranked".
    for spelling in [
        TOP_ACCOUNT,
        &TOP_ACCOUNT.to_lowercase(),
        &TOP_ACCOUNT.to_uppercase().replace("0X", "0x"),
        &format!("  {TOP_ACCOUNT}  "), // hand-edited config file
    ] {
        let standing = board.standing(spelling);
        let entry = standing
            .entry()
            .unwrap_or_else(|| panic!("{spelling} must match rank 1"));
        assert_eq!(entry.blocks, Some(4_790_829));
        assert!(!standing.is_unavailable());
    }
    // Case-insensitivity must not become "any hex address matches": one digit off is a miss.
    assert!(board
        .find("0x18C1c90101aA2D04B62a8Fa80fb8D9a574362078")
        .is_none());
    assert!(board.find("").is_none());
    assert!(board.find("   ").is_none());
}

#[test]
fn an_account_outside_the_listing_is_unranked_not_a_failure() {
    let board = board();
    match board.standing("0x0000000000000000000000000000000000000001") {
        AccountStanding::Unranked {
            listed,
            cutoff_blocks,
        } => {
            assert_eq!(listed, 5);
            // The bar to appear: the last listed account's block count.
            assert_eq!(cutoff_blocks, Some(23_912));
        }
        other => panic!("a small miner is Unranked, got {other:?}"),
    }
    // The distinction that matters: this is NOT the "we could not ask" variant.
    assert!(!board
        .standing("0x0000000000000000000000000000000000000001")
        .is_unavailable());
}

#[test]
fn tolerates_response_shape_drift() {
    // Unknown top-level and per-entry fields, numerics as strings, a null solAddress, and a
    // missing field that must land as None rather than a fabricated 0.
    let drifted = r#"{
        "difficulty": 9100,
        "schemaVersion": "2",
        "miners": [
            {"account":"0xAbC","blocks":"1234","superBlocks":"7","rank":"1",
             "hashRate":"250.5","solAddress":null,"epoch":{"n":3}},
            {"account":"0xDeF","rank":2}
        ],
        "totalBlocks": "93803933",
        "somethingNew": [1,2,3]
    }"#;
    let board = Leaderboard::parse(drifted).expect("drifted shape still parses");
    assert_eq!(board.difficulty, Some(9100)); // number this time, string in the capture
    assert_eq!(board.total_blocks, Some(93_803_933));
    assert_eq!(board.total_miners, None);
    let first = board.find("0xabc").expect("case-insensitive hit");
    assert_eq!(first.blocks, Some(1234));
    assert_eq!(first.super_blocks, Some(7));
    assert_eq!(first.hash_rate, Some(250.5));
    assert_eq!(first.sol_address, None);
    let second = board.find("0xdef").expect("second entry");
    assert_eq!(second.blocks, None, "absent must not become a confirmed 0");

    // An entry with no usable account is dropped, not fatal.
    let board = Leaderboard::parse(r#"{"miners":[{"blocks":1},{"account":"0x1"}]}"#).expect("ok");
    assert_eq!(board.entries.len(), 1);
}

#[test]
fn malformed_and_truncated_bodies_are_errors_not_empty_leaderboards() {
    // Truncated mid-array: the single most likely real failure, and it must NOT read as
    // "the leaderboard is empty".
    let truncated = &CAPTURED[..CAPTURED.len() / 2];
    assert!(matches!(
        Leaderboard::parse(truncated),
        Err(LeaderboardError::Malformed(_))
    ));
    for body in [
        "",
        "   ",
        "not json",
        "[]",
        "null",
        r#"{"miners":{}}"#,
        "{}",
    ] {
        assert!(
            matches!(
                Leaderboard::parse(body),
                Err(LeaderboardError::Malformed(_))
            ),
            "{body:?} must be Malformed, never an empty board"
        );
    }
    // An honestly empty listing is different, and is allowed.
    assert_eq!(
        Leaderboard::parse(r#"{"miners":[]}"#)
            .expect("empty list")
            .entries
            .len(),
        0
    );
}

#[test]
fn the_body_cap_refuses_rather_than_truncating() {
    let oversized = vec![b'{'; MAX_LEADERBOARD_BYTES + 1];
    assert_eq!(
        read_capped(&oversized[..], MAX_LEADERBOARD_BYTES),
        Err(LeaderboardError::TooLarge(MAX_LEADERBOARD_BYTES))
    );
    // Exactly at the cap is fine, and the live 114 KB body is nowhere near it.
    assert!(read_capped(&oversized[..MAX_LEADERBOARD_BYTES], MAX_LEADERBOARD_BYTES).is_ok());
    assert!(CAPTURED.len() < MAX_LEADERBOARD_BYTES);
    // A non-UTF-8 body is malformed, not a panic.
    assert!(matches!(
        read_capped(&[0xff, 0xfe][..], 16),
        Err(LeaderboardError::Malformed(_))
    ));
}

// ---------------------------------------------------------------------------------------
// Client-level: against a real socket. `for_rpc` derives an https URL we cannot serve
// locally, so these drive `LeaderboardClient::new` with an explicit http URL; the derivation
// itself is covered by the unit tests in src/leaderboard.rs.
// ---------------------------------------------------------------------------------------

struct Serving {
    hits: AtomicU32,
    status: AtomicU32,
}

fn serve() -> (SocketAddr, Arc<Serving>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    drop(listener);
    let state = Arc::new(Serving {
        hits: AtomicU32::new(0),
        status: AtomicU32::new(200),
    });
    let served = Arc::clone(&state);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async move {
            let app = Router::new()
                .route(
                    "/v1/leaderboard",
                    get(
                        |axum::extract::State(s): axum::extract::State<Arc<Serving>>| async move {
                            s.hits.fetch_add(1, Ordering::SeqCst);
                            let code = StatusCode::from_u16(s.status.load(Ordering::SeqCst) as u16)
                                .expect("status");
                            (code, CAPTURED).into_response()
                        },
                    ),
                )
                .with_state(served);
            let listener = tokio::net::TcpListener::bind(addr).await.expect("bind");
            axum::serve(listener, app).await.expect("serve");
        });
    });
    for _ in 0..200 {
        if std::net::TcpStream::connect(addr).is_ok() {
            return (addr, state);
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("mock leaderboard never became ready");
}

#[test]
fn client_fetches_locates_and_refuses_to_storm() {
    let (addr, state) = serve();
    let client = LeaderboardClient::new(format!("http://{addr}/v1/leaderboard"))
        .expect("client")
        .with_min_interval(Duration::from_secs(600));

    assert!(matches!(
        client.standing(&TOP_ACCOUNT.to_lowercase()),
        AccountStanding::Ranked(_)
    ));
    assert_eq!(state.hits.load(Ordering::SeqCst), 1);

    // A second call inside the cooldown must not reach the network — and must be reported as
    // Unavailable (we could not ask), never as Unranked (we asked and are not listed).
    let throttled = client.standing(TOP_ACCOUNT);
    assert!(throttled.is_unavailable(), "got {throttled:?}");
    assert!(matches!(
        client.fetch(),
        Err(LeaderboardError::Throttled(_))
    ));
    assert_eq!(state.hits.load(Ordering::SeqCst), 1, "no second request");
}

#[test]
fn every_failure_is_inert_and_unavailable() {
    let (addr, state) = serve();
    state.status.store(503, Ordering::SeqCst);
    let client = LeaderboardClient::new(format!("http://{addr}/v1/leaderboard"))
        .expect("client")
        .with_min_interval(Duration::ZERO);
    assert!(matches!(client.fetch(), Err(LeaderboardError::Status(503))));
    assert!(client.standing(TOP_ACCOUNT).is_unavailable());

    // Nothing listening at all: a transport failure, still just Unavailable.
    let dead = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
    let dead_addr = dead.local_addr().expect("addr");
    drop(dead);
    let client = LeaderboardClient::new(format!("http://{dead_addr}/v1/leaderboard"))
        .expect("client")
        .with_min_interval(Duration::ZERO);
    assert!(matches!(
        client.fetch(),
        Err(LeaderboardError::Transport(_))
    ));
    assert!(client.standing(TOP_ACCOUNT).is_unavailable());
}
