//! Cooperative shutdown flag.
//!
//! The C++ miner crashed on Ctrl-C and on systemd's SIGINT because the handler did real
//! teardown work — allocating, freeing, touching CUDA — from async-signal context. The fix
//! there, and the contract here, is that a signal handler only ever flips this flag; every
//! thread notices it on its next loop iteration and unwinds normally.
//!
//! This type is deliberately the whole API: it is `Clone` + `Send` + `Sync` and contains
//! nothing but an atomic, so a handler registered by the binary crate can hold a copy and
//! still be async-signal-safe.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Shutdown {
    running: Arc<AtomicBool>,
}

impl Default for Shutdown {
    fn default() -> Self {
        Self::new()
    }
}

impl Shutdown {
    pub fn new() -> Self {
        Self { running: Arc::new(AtomicBool::new(true)) }
    }

    /// The only call a signal handler may make. Store-only: no allocation, no locks.
    pub fn request_stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub fn is_stopping(&self) -> bool {
        !self.is_running()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_running_and_latches_once_stopped() {
        let shutdown = Shutdown::new();
        assert!(shutdown.is_running());
        let clone = shutdown.clone();
        clone.request_stop();
        assert!(shutdown.is_stopping());
        // Idempotent: a second signal must not resurrect the flag.
        clone.request_stop();
        assert!(!shutdown.is_running());
    }
}
