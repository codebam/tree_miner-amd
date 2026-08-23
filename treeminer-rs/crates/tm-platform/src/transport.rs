//! Broker connection, topics and reconnect. Port of `MqttClient.{h,cpp}`, on rumqttc
//! instead of paho.
//!
//! The rest of the crate talks to a [`Transport`], not to rumqttc, so the manager's
//! command handling is testable without a broker of any kind.

use crate::backoff::ReconnectBackoff;
use crate::proto::topic;
use crate::secret::{redact_url, Secret};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Full topic for a suffix: `xenminer/{worker_id}/{suffix}`.
pub fn build_topic(worker_id: &str, suffix: &str) -> String {
    format!("xenminer/{worker_id}/{suffix}")
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TransportError {
    #[error("not connected to the broker")]
    NotConnected,
    #[error("broker request queue is full")]
    QueueFull,
    #[error("broker rejected the request")]
    Rejected,
}

/// What the manager needs from a broker: publish, subscribe, and whether the link is up.
///
/// Publishing must never block the caller — a heartbeat is best-effort telemetry and the
/// mining loop cannot wait on a dead socket — so implementations return
/// [`TransportError::QueueFull`] rather than applying backpressure.
pub trait Transport: Send + Sync {
    fn publish(&self, topic: &str, payload: &str) -> Result<(), TransportError>;
    fn subscribe(&self, topic: &str) -> Result<(), TransportError>;
    fn is_connected(&self) -> bool;
}

impl<T: Transport + ?Sized> Transport for Arc<T> {
    fn publish(&self, topic: &str, payload: &str) -> Result<(), TransportError> {
        (**self).publish(topic, payload)
    }
    fn subscribe(&self, topic: &str) -> Result<(), TransportError> {
        (**self).subscribe(topic)
    }
    fn is_connected(&self) -> bool {
        (**self).is_connected()
    }
}

// --- Broker URI ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BrokerUri {
    pub host: String,
    pub port: u16,
    /// True for `ssl://`, `mqtts://` and `tls://`.
    pub secure: bool,
    /// Credentials embedded in the URI, if any. Kept out of every log line.
    pub userinfo: Option<(String, Secret)>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum UriError {
    /// The URI itself is never echoed: it may carry `user:pass@`.
    #[error("broker URI has no scheme (expected tcp:// or ssl://)")]
    NoScheme,
    #[error("unsupported broker URI scheme '{0}'")]
    UnsupportedScheme(String),
    #[error("broker URI has no host")]
    NoHost,
    #[error("broker URI has an invalid port")]
    BadPort,
}

impl BrokerUri {
    /// Parse `scheme://[user:pass@]host[:port]`.
    ///
    /// Written by hand rather than using `MqttOptions::parse_url` so that the TLS decision
    /// is ours and visible, and so a parse failure cannot echo the URI (and with it any
    /// embedded password) into an error message.
    pub fn parse(uri: &str) -> Result<Self, UriError> {
        let (scheme, rest) = uri.split_once("://").ok_or(UriError::NoScheme)?;
        let secure = match scheme.to_ascii_lowercase().as_str() {
            "tcp" | "mqtt" => false,
            "ssl" | "mqtts" | "tls" => true,
            other => return Err(UriError::UnsupportedScheme(other.to_string())),
        };

        let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
        let authority = &rest[..authority_end];

        let (userinfo, hostport) = match authority.rfind('@') {
            Some(at) => {
                let (user, pass) = authority[..at].split_once(':').unwrap_or((&authority[..at], ""));
                (
                    Some((user.to_string(), Secret::new(pass))),
                    &authority[at + 1..],
                )
            }
            None => (None, authority),
        };

        let (host, port) = match hostport.rsplit_once(':') {
            Some((host, port)) => (
                host,
                port.parse::<u16>().map_err(|_| UriError::BadPort)?,
            ),
            None => (hostport, if secure { 8883 } else { 1883 }),
        };
        if host.is_empty() {
            return Err(UriError::NoHost);
        }
        if port == 0 {
            return Err(UriError::BadPort);
        }

        Ok(Self {
            host: host.to_string(),
            port,
            secure,
            userinfo,
        })
    }
}

// --- Configuration ---

/// Everything needed to open a broker connection.
#[derive(Debug, Clone)]
pub struct BrokerConfig {
    /// `tcp://host:1883` or `ssl://host:8883`.
    pub uri: String,
    pub worker_id: String,
    /// Broker username/password, or `None` for an anonymous broker.
    pub credentials: Option<(String, Secret)>,
    /// CA bundle used to verify the broker certificate. `None` uses the system store.
    pub tls_ca_file: Option<std::path::PathBuf>,
    /// Mutual-TLS client identity: PEM certificate + PEM private key.
    pub tls_client_cert: Option<(std::path::PathBuf, std::path::PathBuf)>,
    pub keepalive: Duration,
    pub backoff: ReconnectBackoff,
}

impl BrokerConfig {
    pub fn new(uri: impl Into<String>, worker_id: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            worker_id: worker_id.into(),
            credentials: None,
            tls_ca_file: None,
            tls_client_cert: None,
            keepalive: Duration::from_secs(KEEPALIVE_INTERVAL_SEC),
            backoff: ReconnectBackoff::default(),
        }
    }

    /// The URI as it may appear in a log line.
    pub fn safe_uri(&self) -> String {
        redact_url(&self.uri)
    }
}

pub const KEEPALIVE_INTERVAL_SEC: u64 = 60;
pub const QOS: rumqttc::QoS = rumqttc::QoS::AtLeastOnce;
/// Outgoing request queue. Deep enough for a burst of telemetry, shallow enough that a
/// dead broker cannot accumulate unbounded memory.
const REQUEST_CAPACITY: usize = 64;

#[derive(Debug, thiserror::Error)]
pub enum ConnectError {
    #[error(transparent)]
    Uri(#[from] UriError),
    #[error("could not read TLS material from {0}")]
    TlsFile(std::path::PathBuf),
    #[error("no usable trust anchors for TLS (set a CA file)")]
    NoTrustAnchors,
}

// --- rumqttc-backed transport ---

/// Callback for an inbound publish. Runs on the connection pump thread, so it must not
/// block: the manager's implementation only enqueues.
pub type OnMessage = Arc<dyn Fn(&str, &[u8]) + Send + Sync>;

pub struct MqttTransport {
    client: rumqttc::Client,
    connected: Arc<AtomicBool>,
    /// Every subscription ever made, replayed after each reconnect because the session is
    /// clean (`clean_session = true`, matching the C++) and the broker forgets them.
    subscriptions: Arc<Mutex<Vec<String>>>,
    worker_id: String,
    safe_uri: String,
    stop: Arc<AtomicBool>,
    pump: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl std::fmt::Debug for MqttTransport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MqttTransport")
            .field("uri", &self.safe_uri)
            .field("worker_id", &self.worker_id)
            .field("connected", &self.is_connected())
            .finish_non_exhaustive()
    }
}

impl MqttTransport {
    /// Open the connection and start the pump thread that keeps it alive.
    ///
    /// The Last Will and Testament is registered here so that the platform learns about an
    /// ungraceful death, exactly as the C++ does — same topic, same payload shape.
    pub fn start(config: BrokerConfig, on_message: OnMessage) -> Result<Arc<Self>, ConnectError> {
        let uri = BrokerUri::parse(&config.uri)?;
        let safe_uri = config.safe_uri();

        let mut opts = rumqttc::MqttOptions::new(
            format!("xenminer_{}", config.worker_id),
            uri.host.clone(),
            uri.port,
        );
        opts.set_keep_alive(config.keepalive);
        opts.set_clean_session(true);
        // Bound what the network can make the MQTT codec allocate. The command dispatcher
        // caps payloads again, but a packet larger than this is refused before it is even
        // assembled.
        opts.set_max_packet_size(crate::envelope::MAX_PAYLOAD_BYTES + 4096, 256 * 1024);

        // Explicit credentials win over anything embedded in the URI; the URI form is
        // supported because operators use it, not because it is a good idea.
        if let Some((user, pass)) = config.credentials.clone().or(uri.userinfo.clone()) {
            opts.set_credentials(user, pass.expose().to_string());
        }

        if uri.secure {
            opts.set_transport(rumqttc::Transport::tls_with_config(tls_config(&config)?));
        } else {
            // A plaintext broker carries payout-affecting commands in the clear. The
            // envelope signature still stops forgery, but anyone on the path reads the
            // fleet's telemetry, so say so once.
            tracing::warn!(
                broker = %safe_uri,
                "PLAINTEXT broker connection — anyone on the path can read platform traffic. \
                 Use an ssl:// URI in production."
            );
        }

        let lwt = rumqttc::LastWill::new(
            build_topic(&config.worker_id, topic::STATUS),
            serde_json::to_vec(&crate::proto::OfflineNotice::new(
                config.worker_id.clone(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0),
            ))
            .unwrap_or_default(),
            QOS,
            true,
        );
        opts.set_last_will(lwt);

        let (client, connection) = rumqttc::Client::new(opts, REQUEST_CAPACITY);

        let transport = Arc::new(Self {
            client: client.clone(),
            connected: Arc::new(AtomicBool::new(false)),
            subscriptions: Arc::new(Mutex::new(Vec::new())),
            worker_id: config.worker_id.clone(),
            safe_uri: safe_uri.clone(),
            stop: Arc::new(AtomicBool::new(false)),
            pump: Mutex::new(None),
        });

        let pump = {
            let connected = Arc::clone(&transport.connected);
            let subscriptions = Arc::clone(&transport.subscriptions);
            let stop = Arc::clone(&transport.stop);
            let mut backoff = config.backoff.clone();
            std::thread::Builder::new()
                .name("tm-mqtt".into())
                .spawn(move || {
                    pump_loop(
                        connection,
                        client,
                        connected,
                        subscriptions,
                        stop,
                        &mut backoff,
                        on_message,
                        &safe_uri,
                    )
                })
                .ok()
        };
        *transport.pump.lock() = pump;

        Ok(transport)
    }

    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    pub fn topic_for(&self, suffix: &str) -> String {
        build_topic(&self.worker_id, suffix)
    }

    /// Stop the pump and disconnect. Idempotent.
    pub fn shutdown(&self) {
        if self.stop.swap(true, Ordering::SeqCst) {
            return;
        }
        let _ = self.client.try_disconnect();
        if let Some(handle) = self.pump.lock().take() {
            let _ = handle.join();
        }
        self.connected.store(false, Ordering::SeqCst);
    }
}

impl Drop for MqttTransport {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Transport for MqttTransport {
    fn publish(&self, topic: &str, payload: &str) -> Result<(), TransportError> {
        if !self.is_connected() {
            return Err(TransportError::NotConnected);
        }
        self.client
            .try_publish(topic, QOS, false, payload.as_bytes())
            .map_err(|_| TransportError::QueueFull)
    }

    fn subscribe(&self, topic: &str) -> Result<(), TransportError> {
        // Remembered even if the immediate request fails: the pump replays the list on
        // every (re)connect, so a subscribe issued while the link is down still lands.
        {
            let mut subs = self.subscriptions.lock();
            if !subs.iter().any(|t| t == topic) {
                subs.push(topic.to_string());
            }
        }
        if !self.is_connected() {
            return Err(TransportError::NotConnected);
        }
        self.client
            .try_subscribe(topic, QOS)
            .map_err(|_| TransportError::QueueFull)
    }

    fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }
}

#[allow(clippy::too_many_arguments)]
fn pump_loop(
    mut connection: rumqttc::Connection,
    client: rumqttc::Client,
    connected: Arc<AtomicBool>,
    subscriptions: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    backoff: &mut ReconnectBackoff,
    on_message: OnMessage,
    safe_uri: &str,
) {
    for event in connection.iter() {
        if stop.load(Ordering::SeqCst) {
            break;
        }
        match event {
            Ok(rumqttc::Event::Incoming(rumqttc::Packet::ConnAck(ack))) => {
                if ack.code == rumqttc::ConnectReturnCode::Success {
                    connected.store(true, Ordering::SeqCst);
                    backoff.reset();
                    tracing::info!(broker = %safe_uri, "MQTT connected");
                    for topic in subscriptions.lock().iter() {
                        let _ = client.try_subscribe(topic.as_str(), QOS);
                    }
                } else {
                    // A refused CONNACK is a credential or authorisation problem. The code
                    // is not secret; the credentials that produced it are, and are not here.
                    connected.store(false, Ordering::SeqCst);
                    tracing::error!(broker = %safe_uri, code = ?ack.code, "broker refused the connection");
                }
            }
            Ok(rumqttc::Event::Incoming(rumqttc::Packet::Publish(publish))) => {
                on_message(&publish.topic, &publish.payload);
            }
            Ok(_) => {}
            Err(err) => {
                connected.store(false, Ordering::SeqCst);
                if stop.load(Ordering::SeqCst) {
                    break;
                }
                let delay = backoff.next_delay();
                tracing::warn!(
                    broker = %safe_uri,
                    error = %err,
                    retry_in_ms = delay.as_millis(),
                    "MQTT connection lost"
                );
                // Sleep in slices so shutdown does not have to wait out a 30s backoff.
                let deadline = std::time::Instant::now() + delay;
                while std::time::Instant::now() < deadline {
                    if stop.load(Ordering::SeqCst) {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(20).min(delay));
                }
            }
        }
    }
    connected.store(false, Ordering::SeqCst);
}

/// TLS configuration for an `ssl://` broker.
///
/// Server certificate verification is ALWAYS on and there is no knob to turn it off: TLS
/// without verification is indistinguishable from plaintext to an active attacker, which
/// is the realistic broker threat. rustls verifies the hostname as part of the same check.
fn tls_config(config: &BrokerConfig) -> Result<rumqttc::TlsConfiguration, ConnectError> {
    let client_auth = match &config.tls_client_cert {
        Some((cert, key)) => {
            let cert_pem =
                std::fs::read(cert).map_err(|_| ConnectError::TlsFile(cert.clone()))?;
            let key_pem = std::fs::read(key).map_err(|_| ConnectError::TlsFile(key.clone()))?;
            Some((cert_pem, key_pem))
        }
        None => None,
    };

    match &config.tls_ca_file {
        Some(path) => {
            let ca = std::fs::read(path).map_err(|_| ConnectError::TlsFile(path.clone()))?;
            Ok(rumqttc::TlsConfiguration::Simple {
                ca,
                alpn: None,
                client_auth,
            })
        }
        None => {
            // rumqttc's `TlsConfiguration::default()` loads the platform trust store and
            // *panics* if it cannot. A miner must not abort because a container image
            // shipped without ca-certificates, so the panic is converted into an error the
            // operator can act on.
            let config = std::panic::catch_unwind(rumqttc::TlsConfiguration::default)
                .map_err(|_| ConnectError::NoTrustAnchors)?;
            if client_auth.is_some() {
                // Client certificates need the `Simple` shape, which needs an explicit CA.
                return Err(ConnectError::NoTrustAnchors);
            }
            Ok(config)
        }
    }
}
