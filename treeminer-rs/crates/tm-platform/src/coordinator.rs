//! The mining context platform commands mutate. Port of `MiningCoordinator.{h,cpp}` and
//! the `MiningIdentityConfig` snapshot it works alongside.
//!
//! The C++ is a singleton behind a `shared_mutex`. Here it is an ordinary value shared as
//! an `Arc`: the binary owns one and hands clones to whoever needs it, which keeps the
//! mining hot path's read cheap without a global.

use parking_lot::RwLock;
use std::sync::atomic::{AtomicI64, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MiningMode {
    #[default]
    SelfMining,
    PlatformMining,
}

impl MiningMode {
    /// The spelling `/platform/status` uses.
    pub fn as_str(self) -> &'static str {
        match self {
            MiningMode::SelfMining => "self",
            MiningMode::PlatformMining => "platform",
        }
    }
}

/// Who the miner is currently working for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MiningContext {
    pub mode: MiningMode,
    /// Target mining address — the operator's own, or the lease consumer's.
    pub address: String,
    /// Hex prefix for key generation (16 chars for a platform lease).
    pub prefix: String,
    pub consumer_id: String,
    pub lease_id: String,
}

/// Remotely-updatable mining identity, read by the mining loop as one immutable snapshot
/// so a batch cannot observe a mixture of values while a `set_config` lands.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MiningIdentity {
    pub user_address: String,
    pub self_mining_prefix: String,
    pub test_block_pattern: String,
}

/// The shared mining state a platform command may change.
#[derive(Debug, Default)]
pub struct MiningCoordinator {
    context: RwLock<MiningContext>,
    identity: RwLock<MiningIdentity>,
    difficulty: AtomicI64,
}

impl MiningCoordinator {
    pub fn new(identity: MiningIdentity, difficulty: i64) -> Self {
        Self {
            context: RwLock::new(MiningContext {
                mode: MiningMode::SelfMining,
                address: identity.user_address.clone(),
                ..MiningContext::default()
            }),
            identity: RwLock::new(identity),
            difficulty: AtomicI64::new(difficulty),
        }
    }

    pub fn context(&self) -> MiningContext {
        self.context.read().clone()
    }

    pub fn mode(&self) -> MiningMode {
        self.context.read().mode
    }

    pub fn is_self_mining(&self) -> bool {
        self.mode() == MiningMode::SelfMining
    }

    pub fn is_platform_mining(&self) -> bool {
        self.mode() == MiningMode::PlatformMining
    }

    pub fn update_context(&self, ctx: MiningContext) {
        *self.context.write() = ctx;
    }

    pub fn switch_to_self_mining(&self) {
        let address = self.identity.read().user_address.clone();
        *self.context.write() = MiningContext {
            mode: MiningMode::SelfMining,
            address,
            ..MiningContext::default()
        };
    }

    pub fn switch_to_platform_mining(
        &self,
        address: &str,
        prefix: &str,
        consumer_id: &str,
        lease_id: &str,
    ) {
        *self.context.write() = MiningContext {
            mode: MiningMode::PlatformMining,
            address: address.to_string(),
            prefix: prefix.to_string(),
            consumer_id: consumer_id.to_string(),
            lease_id: lease_id.to_string(),
        };
    }

    pub fn identity(&self) -> MiningIdentity {
        self.identity.read().clone()
    }

    /// Redirects every future block reward. Only ever called from a signature-verified
    /// `set_config`; the caller logs it loudly.
    pub fn set_user_address(&self, address: &str) {
        self.identity.write().user_address = address.to_string();
    }

    pub fn set_self_mining_prefix(&self, prefix: &str) {
        self.identity.write().self_mining_prefix = prefix.to_string();
    }

    pub fn set_test_block_pattern(&self, pattern: &str) {
        self.identity.write().test_block_pattern = pattern.to_string();
    }

    pub fn difficulty(&self) -> i64 {
        self.difficulty.load(Ordering::Relaxed)
    }

    pub fn set_difficulty(&self, difficulty: i64) {
        self.difficulty.store(difficulty, Ordering::Relaxed);
    }
}
