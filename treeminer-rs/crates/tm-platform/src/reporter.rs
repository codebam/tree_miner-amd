//! Worker/GPU telemetry payloads. Port of `WorkerReporter.{h,cpp}`.
//!
//! Every payload is built as a typed struct from [`crate::proto`] and serialised once, so
//! the shapes on the wire are the ones the schema documents rather than an ad-hoc JSON
//! literal per call site.

use crate::clock::Clock;
use crate::proto::{topic, BlockFound, GpuInfo, Heartbeat, OfflineNotice, Register, StatusUpdate};
use crate::transport::{build_topic, Transport, TransportError};
use std::sync::Arc;

/// A snapshot of live mining statistics, gathered by the caller so the reporter itself
/// touches no global state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkerStats {
    pub total_hashrate: f32,
    pub active_gpus: i64,
    pub accepted_blocks: i64,
    pub difficulty: i64,
    pub uptime_sec: i64,
    /// Current self-mining identity, echoed back so the platform dashboard can show what
    /// a `set_config` actually took effect as.
    pub address: String,
    pub prefix: String,
    pub block_pattern: String,
}

pub struct WorkerReporter<T: Transport> {
    transport: T,
    worker_id: String,
    clock: Arc<dyn Clock>,
}

impl<T: Transport> WorkerReporter<T> {
    pub fn new(transport: T, worker_id: impl Into<String>, clock: Arc<dyn Clock>) -> Self {
        Self {
            transport,
            worker_id: worker_id.into(),
            clock,
        }
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn transport(&self) -> &T {
        &self.transport
    }

    pub fn send_registration(
        &self,
        eth_address: &str,
        gpus: &[GpuInfo],
    ) -> Result<(), TransportError> {
        let payload = Register {
            worker_id: self.worker_id.clone(),
            eth_address: eth_address.to_string(),
            gpu_count: gpus.len() as i64,
            total_memory_gb: gpus.iter().map(|g| g.memory_gb).sum(),
            gpus: gpus.to_vec(),
            version: crate::proto::WORKER_VERSION.to_string(),
            timestamp: self.clock.now_epoch_s(),
        };
        self.publish(topic::REGISTER, &payload)
    }

    pub fn send_heartbeat(&self, stats: &WorkerStats) -> Result<(), TransportError> {
        let payload = Heartbeat {
            worker_id: self.worker_id.clone(),
            hashrate: stats.total_hashrate as f64,
            active_gpus: stats.active_gpus,
            accepted_blocks: stats.accepted_blocks,
            difficulty: stats.difficulty,
            address: stats.address.clone(),
            prefix: stats.prefix.clone(),
            block_pattern: stats.block_pattern.clone(),
            uptime_sec: stats.uptime_sec,
            timestamp: self.clock.now_epoch_s(),
        };
        self.publish(topic::HEARTBEAT, &payload)
    }

    pub fn send_status_update(
        &self,
        state: &str,
        lease_id: &str,
        detail: &str,
    ) -> Result<(), TransportError> {
        let payload = StatusUpdate {
            worker_id: self.worker_id.clone(),
            state: state.to_string(),
            lease_id: lease_id.to_string(),
            detail: detail.to_string(),
            timestamp: self.clock.now_epoch_s(),
        };
        self.publish(topic::STATUS, &payload)
    }

    /// The graceful-disconnect twin of the Last Will. Uses the `status` field, not
    /// `state` — see [`OfflineNotice`].
    pub fn send_offline(&self) -> Result<(), TransportError> {
        let payload = OfflineNotice::new(self.worker_id.clone(), self.clock.now_epoch_s());
        self.publish(topic::STATUS, &payload)
    }

    /// Report a find. An empty `lease_id` means it was mined for the operator, not a
    /// consumer.
    pub fn send_block_found(
        &self,
        lease_id: &str,
        hash: &str,
        key: &str,
        account: &str,
        attempts: u64,
        hashrate: f32,
    ) -> Result<(), TransportError> {
        let payload = BlockFound {
            worker_id: self.worker_id.clone(),
            lease_id: lease_id.to_string(),
            hash: hash.to_string(),
            key: key.to_string(),
            account: account.to_string(),
            attempts,
            // A string with two decimals, as the schema types it and the C++'s
            // `setprecision(2)` produces.
            hashrate: format!("{hashrate:.2}"),
            timestamp: self.clock.now_epoch_s(),
        };
        self.publish(topic::BLOCK, &payload)
    }

    fn publish<P: serde::Serialize>(
        &self,
        suffix: &str,
        payload: &P,
    ) -> Result<(), TransportError> {
        // Serialisation of these types cannot fail (no maps with non-string keys, no
        // non-finite floats reach here), but a miner must not die on telemetry either way.
        let Ok(body) = serde_json::to_string(payload) else {
            return Err(TransportError::Rejected);
        };
        self.transport
            .publish(&build_topic(&self.worker_id, suffix), &body)
    }
}
