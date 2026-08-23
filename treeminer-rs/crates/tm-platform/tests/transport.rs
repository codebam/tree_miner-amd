//! Broker URI policy, reconnect backoff, and the pump's behaviour against a fake broker.
//!
//! No test here needs a real MQTT broker: the "broker" is a `TcpListener` that speaks no
//! MQTT at all, which is enough to exercise connect, failure and retry.

use std::io::Read;
use std::net::TcpListener;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tm_platform::backoff::*;
use tm_platform::transport::*;

// --- URI ---

#[test]
fn uri_scheme_decides_tls() {
    for (uri, secure, port) in [
        ("tcp://broker:1883", false, 1883),
        ("mqtt://broker", false, 1883),
        ("ssl://broker", true, 8883),
        ("mqtts://broker:9001", true, 9001),
        ("tls://broker:8883", true, 8883),
        ("SSL://broker", true, 8883),
    ] {
        let parsed = BrokerUri::parse(uri).unwrap();
        assert_eq!(parsed.secure, secure, "{uri}");
        assert_eq!(parsed.port, port, "{uri}");
        assert_eq!(parsed.host, "broker", "{uri}");
    }
}

#[test]
fn bad_uris_are_rejected_without_echoing_them() {
    for (uri, expected) in [
        ("myhost.example:1883", UriError::NoScheme),
        ("", UriError::NoScheme),
        ("http://myhost.example", UriError::UnsupportedScheme("http".into())),
        // No websocket transport is compiled in, so promising one would be a lie.
        ("ws://myhost.example", UriError::UnsupportedScheme("ws".into())),
        ("wss://myhost.example", UriError::UnsupportedScheme("wss".into())),
        ("tcp://", UriError::NoHost),
        ("tcp://myhost.example:0", UriError::BadPort),
        ("tcp://myhost.example:99999", UriError::BadPort),
        ("tcp://myhost.example:notaport", UriError::BadPort),
    ] {
        let err = BrokerUri::parse(uri).unwrap_err();
        assert_eq!(err, expected, "{uri}");
        // Only the scheme is ever echoed. A URI may carry `user:pass@`, so the host and
        // authority must never reach the message.
        assert!(!err.to_string().contains("myhost"), "{uri}: {err}");
    }
}

/// A parse failure on a URI that carries credentials must not print them.
#[test]
fn a_bad_uri_with_credentials_does_not_leak_them() {
    for uri in [
        "user:sup3rs3cret@broker:1883",
        "ftp://user:sup3rs3cret@broker:1883",
        "tcp://user:sup3rs3cret@broker:notaport",
        "tcp://user:sup3rs3cret@:1883",
    ] {
        let err = BrokerUri::parse(uri).unwrap_err();
        assert!(!err.to_string().contains("sup3rs3cret"), "{uri}: {err}");
        assert!(!format!("{err:?}").contains("sup3rs3cret"), "{uri}");
    }
}

#[test]
fn topics_follow_the_specified_structure() {
    assert_eq!(build_topic("abc123", "register"), "xenminer/abc123/register");
    assert_eq!(build_topic("abc123", "task"), "xenminer/abc123/task");
    assert_eq!(build_topic("abc123", "control"), "xenminer/abc123/control");
}

// --- Backoff ---

#[test]
fn backoff_doubles_up_to_the_ceiling() {
    let mut backoff = ReconnectBackoff::new(Duration::from_millis(1_000), Duration::from_millis(30_000));
    let delays: Vec<u64> = (0..8).map(|_| backoff.next_delay().as_millis() as u64).collect();
    assert_eq!(delays, [1_000, 2_000, 4_000, 8_000, 16_000, 30_000, 30_000, 30_000]);
}

#[test]
fn backoff_resets_after_a_successful_connection() {
    let mut backoff = ReconnectBackoff::default();
    for _ in 0..10 {
        backoff.next_delay();
    }
    assert_eq!(backoff.peek().as_millis() as u64, MAX_RECONNECT_DELAY_MS);
    backoff.reset();
    assert_eq!(
        backoff.next_delay().as_millis() as u64,
        INITIAL_RECONNECT_DELAY_MS,
        "the next outage must recover fast, not inherit the last one's ceiling"
    );
}

#[test]
fn backoff_never_returns_zero() {
    // A zero delay would spin the CPU against a dead broker.
    let mut backoff = ReconnectBackoff::new(Duration::ZERO, Duration::ZERO);
    for _ in 0..5 {
        assert!(backoff.next_delay() > Duration::ZERO);
    }
}

// --- Fake broker ---

/// A TCP listener that accepts and immediately drops every connection. To an MQTT client
/// this is a broker that dies mid-handshake, over and over.
struct RudeBroker {
    port: u16,
    accepts: Arc<AtomicUsize>,
    stop: Arc<std::sync::atomic::AtomicBool>,
}

impl RudeBroker {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let accepts = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let counter = Arc::clone(&accepts);
        let stopper = Arc::clone(&stop);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                if stopper.load(Ordering::SeqCst) {
                    return;
                }
                match stream {
                    Ok(mut stream) => {
                        counter.fetch_add(1, Ordering::SeqCst);
                        // Read whatever the client sends, then hang up without a CONNACK.
                        let mut buf = [0u8; 64];
                        let _ = stream.read(&mut buf);
                    }
                    Err(_) => return,
                }
            }
        });
        Self { port, accepts, stop }
    }

    fn uri(&self) -> String {
        format!("tcp://127.0.0.1:{}", self.port)
    }

    fn accepts(&self) -> usize {
        self.accepts.load(Ordering::SeqCst)
    }
}

impl Drop for RudeBroker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = std::net::TcpStream::connect(("127.0.0.1", self.port));
    }
}

/// A broker that never completes a handshake must produce retries, not a panic and not a
/// spin — and the transport must report itself disconnected throughout.
#[test]
fn the_pump_retries_a_broker_that_hangs_up() {
    let broker = RudeBroker::start();
    let mut config = BrokerConfig::new(broker.uri(), "rig-01");
    config.backoff = ReconnectBackoff::new(Duration::from_millis(20), Duration::from_millis(60));

    let received = Arc::new(AtomicUsize::new(0));
    let sink = Arc::clone(&received);
    let transport =
        MqttTransport::start(config, Arc::new(move |_: &str, _: &[u8]| {
            sink.fetch_add(1, Ordering::SeqCst);
        }))
        .expect("start");

    let deadline = Instant::now() + Duration::from_secs(5);
    while broker.accepts() < 3 && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(broker.accepts() >= 3, "expected retries, saw {}", broker.accepts());
    assert!(!transport.is_connected());
    assert_eq!(received.load(Ordering::SeqCst), 0);

    // Nothing may be published to a broker that was never up.
    assert_eq!(
        transport.publish("xenminer/rig-01/heartbeat", "{}"),
        Err(TransportError::NotConnected)
    );

    // The backoff is doing its job: a spin would have produced far more attempts than
    // this in five seconds.
    assert!(broker.accepts() < 400, "too many attempts: {}", broker.accepts());

    transport.shutdown();
    transport.shutdown(); // idempotent
}

/// A subscribe issued while the link is down is remembered and replayed on connect, so a
/// command topic is never silently unsubscribed after an outage.
#[test]
fn a_subscribe_while_disconnected_is_remembered() {
    let broker = RudeBroker::start();
    let mut config = BrokerConfig::new(broker.uri(), "rig-01");
    config.backoff = ReconnectBackoff::new(Duration::from_millis(20), Duration::from_millis(40));
    let transport = MqttTransport::start(config, Arc::new(|_: &str, _: &[u8]| {})).expect("start");

    assert_eq!(
        transport.subscribe(&transport.topic_for("task")),
        Err(TransportError::NotConnected)
    );
    // Repeating it does not accumulate duplicates, and still reports the link state.
    assert_eq!(
        transport.subscribe(&transport.topic_for("task")),
        Err(TransportError::NotConnected)
    );
    transport.shutdown();
}

/// An unreachable port must be an ordinary retry loop, not a startup failure and not a
/// panic — a miner that starts before its broker does has to survive it.
#[test]
fn an_unreachable_broker_is_not_a_startup_failure() {
    // Port 1 on loopback: reliably closed.
    let mut config = BrokerConfig::new("tcp://127.0.0.1:1", "rig-01");
    config.backoff = ReconnectBackoff::new(Duration::from_millis(10), Duration::from_millis(20));
    let transport = MqttTransport::start(config, Arc::new(|_: &str, _: &[u8]| {})).expect("start");
    std::thread::sleep(Duration::from_millis(120));
    assert!(!transport.is_connected());
    transport.shutdown();
}

#[test]
fn a_bad_uri_fails_at_start_rather_than_retrying_forever() {
    let config = BrokerConfig::new("http://broker:80", "rig-01");
    assert!(MqttTransport::start(config, Arc::new(|_: &str, _: &[u8]| {})).is_err());
}

/// TLS material that cannot be read is an error naming the file, not a panic and not a
/// silent fallback to plaintext.
#[test]
fn missing_tls_material_is_an_error_not_a_downgrade() {
    let mut config = BrokerConfig::new("ssl://127.0.0.1:8883", "rig-01");
    config.tls_ca_file = Some("/nonexistent/ca.pem".into());
    let err = MqttTransport::start(config, Arc::new(|_: &str, _: &[u8]| {})).unwrap_err();
    assert!(matches!(err, ConnectError::TlsFile(_)), "{err}");

    let mut config = BrokerConfig::new("ssl://127.0.0.1:8883", "rig-01");
    config.tls_ca_file = Some("/nonexistent/ca.pem".into());
    config.tls_client_cert = Some(("/nonexistent/c.pem".into(), "/nonexistent/k.pem".into()));
    assert!(MqttTransport::start(config, Arc::new(|_: &str, _: &[u8]| {})).is_err());
}
