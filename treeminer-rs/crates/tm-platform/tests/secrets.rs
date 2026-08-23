//! Secret handling: redaction, credential ingestion, and URL scrubbing.
//!
//! The point of every test here is that a secret cannot escape into a log line, a `Debug`
//! dump, or an error message — the three places operators actually read.

use tm_platform::secret::*;
use tm_platform::transport::{BrokerConfig, BrokerUri};

const PASSWORD: &str = "hunter2-swordfish-correcthorse";

#[test]
fn debug_and_display_are_redacted() {
    let secret = Secret::new(PASSWORD);
    assert_eq!(format!("{secret:?}"), REDACTED);
    assert_eq!(format!("{secret}"), REDACTED);
    assert!(!format!("{secret:?}").contains("hunter2"));
    // Only the explicit accessor yields the value.
    assert_eq!(secret.expose(), PASSWORD);
}

/// A `#[derive(Debug)]` struct holding a secret must not leak it — this is the case a
/// stray `tracing::debug!(?config)` hits.
#[test]
fn a_struct_holding_a_secret_does_not_leak_it_in_debug() {
    #[derive(Debug)]
    #[allow(dead_code)]
    struct Config {
        broker: String,
        password: Secret,
    }
    let config = Config {
        broker: "ssl://broker:8883".into(),
        password: Secret::new(PASSWORD),
    };
    let rendered = format!("{config:#?}");
    assert!(!rendered.contains(PASSWORD), "{rendered}");
    assert!(rendered.contains(REDACTED));
}

/// The same for the crate's own config types, which are the ones that will actually be
/// dumped by the binary.
#[test]
fn broker_config_debug_does_not_leak_credentials() {
    let mut config = BrokerConfig::new("ssl://broker:8883", "rig-01");
    config.credentials = Some(("worker".into(), Secret::new(PASSWORD)));
    let rendered = format!("{config:#?}");
    assert!(!rendered.contains(PASSWORD), "{rendered}");

    let uri = BrokerUri::parse(&format!("ssl://worker:{PASSWORD}@broker:8883")).unwrap();
    let rendered = format!("{uri:?}");
    assert!(!rendered.contains(PASSWORD), "{rendered}");
}

/// The error path: a failure while handling credentials must name the variable and nothing
/// else. This is the case where leaks usually happen — "invalid password: hunter2".
#[test]
fn credential_errors_name_the_variable_and_no_value() {
    for error in [
        CredentialError::Missing(ENV_COMMAND_SECRET),
        CredentialError::Empty(ENV_MQTT_PASSWORD),
        CredentialError::NotUtf8(ENV_MQTT_USERNAME),
        CredentialError::AnonymousNotOptedIn,
    ] {
        let display = error.to_string();
        let debug = format!("{error:?}");
        assert!(!display.contains(PASSWORD), "{display}");
        assert!(!debug.contains(PASSWORD), "{debug}");
    }
    assert!(CredentialError::Missing(ENV_COMMAND_SECRET)
        .to_string()
        .contains("TREEMINER_PLATFORM_COMMAND_SECRET"));
    assert!(CredentialError::Empty(ENV_MQTT_PASSWORD)
        .to_string()
        .contains("TREEMINER_MQTT_PASSWORD"));

    // And the assembled credentials do not leak through Debug either.
    let creds = PlatformCredentials {
        command_secret: Secret::new(PASSWORD),
        broker_auth: Some(("worker".into(), Secret::new(PASSWORD))),
    };
    assert!(!format!("{creds:#?}").contains(PASSWORD));
}

#[test]
fn url_redaction() {
    let cases = [
        ("mqtt://user:pass@host:1883", "mqtt://<redacted>@host:1883"),
        ("ssl://user:pass@host:8883/path", "ssl://<redacted>@host:8883/path"),
        ("ssl://user@host:8883", "ssl://<redacted>@host:8883"),
        // Multiple '@' in the userinfo: the last one ends it.
        ("mqtt://user:p@ss@host:1883", "mqtt://<redacted>@host:1883"),
        // A '@' after the authority belongs to the path, not to userinfo.
        ("mqtt://host:1883/a@b", "mqtt://host:1883/a@b"),
        ("tcp://host:1883", "tcp://host:1883"),
        ("host:1883", "host:1883"),
        // Even something that is not a URL must not pass a password through.
        ("user:pass@host", "<redacted>@host"),
        ("", ""),
    ];
    for (input, expected) in cases {
        assert_eq!(redact_url(input), expected, "{input}");
    }
}

#[test]
fn broker_config_logs_a_redacted_uri() {
    let config = BrokerConfig::new(format!("ssl://worker:{PASSWORD}@broker:8883"), "rig-01");
    assert_eq!(config.safe_uri(), "ssl://<redacted>@broker:8883");
    assert!(!config.safe_uri().contains(PASSWORD));
}

/// Credentials embedded in the URI are still usable, but they never survive into anything
/// printable.
#[test]
fn uri_credentials_are_parsed_into_a_secret() {
    let uri = BrokerUri::parse(&format!("ssl://worker:{PASSWORD}@broker:8883")).unwrap();
    let (user, pass) = uri.userinfo.as_ref().unwrap();
    assert_eq!(user, "worker");
    assert_eq!(pass.expose(), PASSWORD);
    assert!(uri.secure);
    assert_eq!(uri.host, "broker");
    assert_eq!(uri.port, 8883);
}

/// The environment is the only source. These run in one test because the process
/// environment is global state and separate `#[test]`s would race each other.
#[test]
fn credentials_come_from_the_environment() {
    let vars = [
        ENV_COMMAND_SECRET,
        ENV_MQTT_USERNAME,
        ENV_MQTT_PASSWORD,
        ENV_MQTT_ANONYMOUS,
        ENV_WORKER_ID,
    ];
    let saved: Vec<_> = vars.iter().map(|v| (*v, std::env::var(v).ok())).collect();
    let clear = || {
        for var in vars {
            std::env::remove_var(var);
        }
    };

    clear();
    assert_eq!(
        PlatformCredentials::from_env().unwrap_err(),
        CredentialError::Missing(ENV_COMMAND_SECRET)
    );

    // An empty variable is a distinct, named mistake from an unset one.
    std::env::set_var(ENV_COMMAND_SECRET, "");
    assert_eq!(
        PlatformCredentials::from_env().unwrap_err(),
        CredentialError::Empty(ENV_COMMAND_SECRET)
    );

    // A secret but no broker credentials: refuse rather than connect anonymously by
    // accident.
    std::env::set_var(ENV_COMMAND_SECRET, PASSWORD);
    assert_eq!(
        PlatformCredentials::from_env().unwrap_err(),
        CredentialError::AnonymousNotOptedIn
    );

    // A username without a password names the missing half.
    std::env::set_var(ENV_MQTT_USERNAME, "worker");
    assert_eq!(
        PlatformCredentials::from_env().unwrap_err(),
        CredentialError::Missing(ENV_MQTT_PASSWORD)
    );

    std::env::set_var(ENV_MQTT_PASSWORD, PASSWORD);
    let creds = PlatformCredentials::from_env().unwrap();
    assert_eq!(creds.command_secret.expose(), PASSWORD);
    let (user, pass) = creds.broker_auth.as_ref().unwrap();
    assert_eq!(user, "worker");
    assert_eq!(pass.expose(), PASSWORD);

    // Explicit anonymous opt-in.
    std::env::remove_var(ENV_MQTT_USERNAME);
    std::env::remove_var(ENV_MQTT_PASSWORD);
    std::env::set_var(ENV_MQTT_ANONYMOUS, "1");
    let creds = PlatformCredentials::from_env().unwrap();
    assert!(creds.broker_auth.is_none());

    // Worker id from the environment, empty treated as unset.
    assert_eq!(PlatformCredentials::worker_id_from_env(), None);
    std::env::set_var(ENV_WORKER_ID, "");
    assert_eq!(PlatformCredentials::worker_id_from_env(), None);
    std::env::set_var(ENV_WORKER_ID, "rig-07");
    assert_eq!(
        PlatformCredentials::worker_id_from_env().as_deref(),
        Some("rig-07")
    );

    clear();
    for (var, value) in saved {
        if let Some(value) = value {
            std::env::set_var(var, value);
        }
    }
}
