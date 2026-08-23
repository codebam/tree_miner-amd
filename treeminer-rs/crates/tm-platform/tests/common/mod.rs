//! Shared test scaffolding: a broker-free transport and a signer.

#![allow(dead_code)]

use parking_lot::Mutex;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tm_platform::clock::{Clock, TestClock};
use tm_platform::coordinator::{MiningCoordinator, MiningIdentity};
use tm_platform::manager::{PlatformConfig, PlatformManager};
use tm_platform::secret::Secret;
use tm_platform::transport::{Transport, TransportError};

/// A transport that records what would have gone to the broker.
#[derive(Debug, Default)]
pub struct FakeTransport {
    pub published: Mutex<Vec<(String, String)>>,
    pub subscribed: Mutex<Vec<String>>,
    connected: AtomicBool,
}

impl FakeTransport {
    pub fn connected() -> Arc<Self> {
        let t = Arc::new(Self::default());
        t.connected.store(true, Ordering::SeqCst);
        t
    }

    pub fn set_connected(&self, value: bool) {
        self.connected.store(value, Ordering::SeqCst);
    }

    /// Every payload published on the given topic suffix, parsed as JSON.
    pub fn published_on(&self, suffix: &str) -> Vec<Value> {
        self.published
            .lock()
            .iter()
            .filter(|(topic, _)| topic.ends_with(&format!("/{suffix}")))
            .filter_map(|(_, body)| serde_json::from_str(body).ok())
            .collect()
    }

    pub fn clear(&self) {
        self.published.lock().clear();
    }
}

impl Transport for FakeTransport {
    fn publish(&self, topic: &str, payload: &str) -> Result<(), TransportError> {
        if !self.is_connected() {
            return Err(TransportError::NotConnected);
        }
        self.published
            .lock()
            .push((topic.to_string(), payload.to_string()));
        Ok(())
    }

    fn subscribe(&self, topic: &str) -> Result<(), TransportError> {
        self.subscribed.lock().push(topic.to_string());
        Ok(())
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
}

pub const WORKER: &str = "rig-01";
pub const SECRET: &str = "correct horse battery staple";
pub const NOW: i64 = 1_700_000_000;
/// A checksummed address; the handler requires EIP-55, not just 40 hex characters.
pub const CONSUMER_ADDRESS: &str = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed";
pub const SELF_ADDRESS: &str = "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359";

pub struct Harness {
    pub manager: Arc<PlatformManager<Arc<FakeTransport>>>,
    pub transport: Arc<FakeTransport>,
    pub clock: Arc<TestClock>,
    pub coordinator: Arc<MiningCoordinator>,
    nonce_counter: Mutex<u64>,
    signed: bool,
}

impl Harness {
    pub fn new(secret: Option<&str>) -> Self {
        let transport = FakeTransport::connected();
        let clock = Arc::new(TestClock::new(NOW));
        let coordinator = Arc::new(MiningCoordinator::new(
            MiningIdentity {
                user_address: SELF_ADDRESS.into(),
                ..Default::default()
            },
            8,
        ));
        let mut config = PlatformConfig::new(WORKER, SELF_ADDRESS);
        config.command_secret = secret.map(Secret::new);
        let manager = Arc::new(PlatformManager::new(
            config,
            Arc::clone(&transport),
            Arc::clone(&coordinator),
            clock.clone(),
        ));
        manager.announce();
        transport.clear();
        Self {
            manager,
            transport,
            clock,
            coordinator,
            nonce_counter: Mutex::new(0),
            signed: secret.is_some(),
        }
    }

    /// Signed with the harness secret, addressed to this worker, valid for 60s.
    pub fn sign(&self, msg: &Value) -> Value {
        self.sign_as(msg, WORKER, SECRET)
    }

    pub fn sign_as(&self, msg: &Value, worker: &str, secret: &str) -> Value {
        let n = {
            let mut c = self.nonce_counter.lock();
            *c += 1;
            *c
        };
        let now = self.clock.now_epoch_s();
        tm_platform::envelope::sign_command(
            msg,
            secret,
            worker,
            &format!("cmd-{n}"),
            &format!("{n:032x}"),
            now,
            now + 60,
        )
    }

    /// Feed one message through the intake path the broker would use.
    pub fn deliver(&self, suffix: &str, msg: &Value) {
        let topic = format!("xenminer/{WORKER}/{suffix}");
        self.manager
            .enqueue_command(&topic, msg.to_string().as_bytes());
        self.manager.dispatch_pending();
    }

    pub fn deliver_raw(&self, suffix: &str, payload: &[u8]) {
        let topic = format!("xenminer/{WORKER}/{suffix}");
        self.manager.enqueue_command(&topic, payload);
        self.manager.dispatch_pending();
    }

    /// Drive the manager into MINING with a valid lease.
    pub fn assign(&self, lease_id: &str, duration_sec: i64) {
        let msg = assign_task(lease_id, duration_sec);
        let msg = if self.signed { self.sign(&msg) } else { msg };
        self.deliver("task", &msg);
    }

}

pub fn assign_task(lease_id: &str, duration_sec: i64) -> Value {
    json!({
        "command": "assign_task",
        "lease_id": lease_id,
        "consumer_id": "consumer-1",
        "consumer_address": CONSUMER_ADDRESS,
        "prefix": "a1b2c3d4e5f6a7b8",
        "duration_sec": duration_sec,
    })
}
