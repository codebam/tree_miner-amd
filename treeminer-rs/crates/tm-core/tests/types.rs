//! Port of `src/treeminer/Types.h`. The status strings are persisted in the journal, so a
//! rename silently strands every row already on disk — round-tripping every variant here is
//! what keeps that from happening quietly.

use tm_core::{FindKind, FindRecord, FindStatus, FoundPayload};

const ALL_STATUSES: [FindStatus; 9] = [
    FindStatus::Pending,
    FindStatus::Submitting,
    FindStatus::AcceptedUnconfirmed,
    FindStatus::Acked,
    FindStatus::ParkedDifficulty,
    FindStatus::ParkedXuniWindow,
    FindStatus::Quarantined,
    FindStatus::Dead,
    FindStatus::PermanentlyInvalid,
];

#[test]
fn every_status_round_trips_through_its_string() {
    for status in ALL_STATUSES {
        assert_eq!(FindStatus::parse(status.as_str()), Some(status));
    }
    assert_eq!(FindStatus::Pending.as_str(), "Pending");
    assert_eq!(FindStatus::AcceptedUnconfirmed.as_str(), "AcceptedUnconfirmed");
    assert_eq!(FindStatus::PermanentlyInvalid.as_str(), "PermanentlyInvalid");
}

#[test]
fn unknown_status_text_parses_to_none_rather_than_a_default() {
    // Defaulting an unrecognised row to Pending would resubmit finds the server already
    // rejected, so parse must fail loudly.
    for text in ["", "pending", "PENDING", "Acked ", "NotAStatus"] {
        assert_eq!(FindStatus::parse(text), None, "accepted {text:?}");
    }
}

#[test]
fn every_kind_round_trips_and_unknown_text_is_none() {
    for kind in [FindKind::Xen11, FindKind::Xuni] {
        assert_eq!(FindKind::parse(kind.as_str()), Some(kind));
    }
    assert_eq!(FindKind::Xen11.as_str(), "XEN11");
    assert_eq!(FindKind::Xuni.as_str(), "XUNI");
    for text in ["", "xen11", "Xuni", "XEN", "XUNI1"] {
        assert_eq!(FindKind::parse(text), None, "accepted {text:?}");
    }
}

#[test]
fn exactly_the_three_terminal_statuses_are_terminal() {
    let terminal: Vec<FindStatus> = ALL_STATUSES
        .into_iter()
        .filter(|status| status.is_terminal())
        .collect();
    assert_eq!(
        terminal,
        vec![
            FindStatus::Acked,
            FindStatus::Dead,
            FindStatus::PermanentlyInvalid
        ]
    );
    // Spelled out per variant so adding a state forces a decision here.
    assert!(!FindStatus::Pending.is_terminal());
    assert!(!FindStatus::Submitting.is_terminal());
    assert!(!FindStatus::AcceptedUnconfirmed.is_terminal());
    assert!(FindStatus::Acked.is_terminal());
    assert!(!FindStatus::ParkedDifficulty.is_terminal());
    assert!(!FindStatus::ParkedXuniWindow.is_terminal());
    assert!(!FindStatus::Quarantined.is_terminal());
    assert!(FindStatus::Dead.is_terminal());
    assert!(FindStatus::PermanentlyInvalid.is_terminal());
}

#[test]
fn a_new_record_starts_pending_and_unattempted() {
    let payload = FoundPayload {
        key: "a".repeat(64),
        hash_to_verify: "$argon2id$v=19$m=8,t=1,p=1$salt$digest".to_string(),
        account: "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed".to_string(),
        kind: FindKind::Xen11,
        memory_cost: 8,
        worker: "worker-0".to_string(),
        attempts: 12,
        hashes_per_second: 3100.0,
        found_at_utc: "2026-08-23T00:00:00Z".to_string(),
    };
    let record = FindRecord::new(payload.clone());
    assert_eq!(record.payload, payload);
    assert_eq!(record.status, FindStatus::Pending);
    assert!(!record.status.is_terminal());
    assert_eq!(record.id, -1);
    assert_eq!(record.attempt_count, 0);
    assert_eq!(record.xuni_windows_tried, 0);
    assert_eq!(record.next_attempt_at, None);
    assert_eq!(record.last_http_status, None);
    assert_eq!(record.confirmed_at, None);
}
