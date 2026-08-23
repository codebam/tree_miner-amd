//! Lease lifecycle. Port of `LeaseManager.{h,cpp}`.
//!
//! Expiry is measured on the monotonic clock, as in the C++ (`steady_clock`): a lease is
//! paid for in wall-clock seconds but must not be shortened or extended by the system
//! clock moving.

use crate::clock::Clock;
use crate::coordinator::{MiningContext, MiningMode};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseInfo {
    pub lease_id: String,
    pub consumer_id: String,
    /// Ethereum address to mine for.
    pub consumer_address: String,
    /// Hex prefix for key generation.
    pub prefix: String,
    pub duration_sec: i64,
    /// Monotonic reading at `start_lease`.
    pub started_at: Duration,
    pub blocks_found: i64,
}

/// Why a lease could not be started.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LeaseError {
    #[error("already in lease {0}")]
    AlreadyLeased(String),
}

pub struct LeaseManager {
    current: Mutex<Option<LeaseInfo>>,
    clock: Arc<dyn Clock>,
}

impl std::fmt::Debug for LeaseManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LeaseManager")
            .field("current", &*self.current.lock())
            .finish_non_exhaustive()
    }
}

impl LeaseManager {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            current: Mutex::new(None),
            clock,
        }
    }

    /// Start a lease. Fails if one is already active — the C++ refuses to overwrite a
    /// live lease, so a second `assign_task` cannot silently repoint the rig.
    pub fn start_lease(
        &self,
        lease_id: &str,
        consumer_id: &str,
        consumer_address: &str,
        prefix: &str,
        duration_sec: i64,
    ) -> Result<(), LeaseError> {
        let mut current = self.current.lock();
        if let Some(existing) = current.as_ref() {
            return Err(LeaseError::AlreadyLeased(existing.lease_id.clone()));
        }
        *current = Some(LeaseInfo {
            lease_id: lease_id.to_string(),
            consumer_id: consumer_id.to_string(),
            consumer_address: consumer_address.to_string(),
            prefix: prefix.to_string(),
            duration_sec,
            started_at: self.clock.monotonic(),
            blocks_found: 0,
        });
        Ok(())
    }

    /// End the current lease. Returns the lease that ended, or `None` if there was none.
    pub fn end_lease(&self) -> Option<LeaseInfo> {
        self.current.lock().take()
    }

    /// True iff a lease exists and has not expired.
    pub fn has_active_lease(&self) -> bool {
        let current = self.current.lock();
        current.is_some() && !self.is_expired_locked(current.as_ref())
    }

    /// True when there is no lease at all, or the lease has run out. Matches the C++,
    /// where "no lease" reads as expired.
    pub fn is_expired(&self) -> bool {
        let current = self.current.lock();
        self.is_expired_locked(current.as_ref())
    }

    pub fn record_block(&self) {
        if let Some(lease) = self.current.lock().as_mut() {
            lease.blocks_found += 1;
        }
    }

    pub fn lease(&self) -> Option<LeaseInfo> {
        self.current.lock().clone()
    }

    /// Seconds left on the lease; 0 when there is none or it has expired.
    pub fn remaining_seconds(&self) -> i64 {
        let current = self.current.lock();
        let Some(lease) = current.as_ref() else {
            return 0;
        };
        let elapsed = self.elapsed_secs(lease);
        (lease.duration_sec - elapsed).max(0)
    }

    /// The mining context this lease implies, or the self-mining context when idle.
    pub fn to_mining_context(&self, self_address: &str) -> MiningContext {
        match self.current.lock().as_ref() {
            None => MiningContext {
                mode: MiningMode::SelfMining,
                address: self_address.to_string(),
                ..MiningContext::default()
            },
            Some(lease) => MiningContext {
                mode: MiningMode::PlatformMining,
                address: lease.consumer_address.clone(),
                prefix: lease.prefix.clone(),
                consumer_id: lease.consumer_id.clone(),
                lease_id: lease.lease_id.clone(),
            },
        }
    }

    fn elapsed_secs(&self, lease: &LeaseInfo) -> i64 {
        self.clock
            .monotonic()
            .saturating_sub(lease.started_at)
            .as_secs() as i64
    }

    fn is_expired_locked(&self, lease: Option<&LeaseInfo>) -> bool {
        let Some(lease) = lease else {
            return true;
        };
        if lease.duration_sec <= 0 {
            return false; // 0 means no expiry, as in the C++
        }
        self.elapsed_secs(lease) >= lease.duration_sec
    }
}
