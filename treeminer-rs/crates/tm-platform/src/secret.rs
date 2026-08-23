//! Secret handling for platform mode: the redacting [`Secret`] newtype, credential
//! ingestion from the environment, and URL redaction for logs.
//!
//! # Why the environment, and only the environment
//!
//! Broker credentials and the platform command secret are read from environment
//! variables. They are deliberately **not** command-line flags — a flag puts the secret in
//! `/proc/<pid>/cmdline`, which is world-readable on Linux — and deliberately not in
//! `config.txt`, which is the wrong place for a token and is what the C++ did
//! (`platform_command_secret`).
//!
//! | variable | meaning |
//! | --- | --- |
//! | `TREEMINER_PLATFORM_COMMAND_SECRET` | shared secret for the HMAC-SHA256 command envelope. Without it no command can be authenticated and the miner refuses every mutating command. |
//! | `TREEMINER_MQTT_USERNAME` | MQTT broker username (`CONNECT` username field) |
//! | `TREEMINER_MQTT_PASSWORD` | MQTT broker password |
//! | `TREEMINER_MQTT_ANONYMOUS` | set to `1`/`true` to connect with no broker credentials at all. Required to be explicit: an anonymous broker is a deployment decision, not a default to fall into silently. |
//! | `TREEMINER_WORKER_ID` | worker id, if `--worker-id` was not given. The flag wins. |
//!
//! Non-secret settings (`--mqtt-broker`, `--worker-id`) keep their flags.

use std::env;

/// The environment variable holding the HMAC shared secret for command envelopes.
pub const ENV_COMMAND_SECRET: &str = "TREEMINER_PLATFORM_COMMAND_SECRET";
/// The environment variable holding the MQTT broker username.
pub const ENV_MQTT_USERNAME: &str = "TREEMINER_MQTT_USERNAME";
/// The environment variable holding the MQTT broker password.
pub const ENV_MQTT_PASSWORD: &str = "TREEMINER_MQTT_PASSWORD";
/// Opt in to connecting without broker credentials.
pub const ENV_MQTT_ANONYMOUS: &str = "TREEMINER_MQTT_ANONYMOUS";
/// Fallback source for the worker id; `--worker-id` takes precedence.
pub const ENV_WORKER_ID: &str = "TREEMINER_WORKER_ID";

/// What a redacted secret renders as, everywhere.
pub const REDACTED: &str = "<redacted>";

/// A string that must never reach a log, a `Debug` dump, or an error message.
///
/// The only way to read the contents is [`Secret::expose`], which is greppable. `Debug`
/// and `Display` both print [`REDACTED`], so a stray `{:?}` on a struct that contains one
/// — a config dump, a `#[derive(Debug)]` error variant — cannot leak it.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The secret material. Named to make every read site obvious in review.
    pub fn expose(&self) -> &str {
        &self.0
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(REDACTED)
    }
}

impl std::fmt::Display for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Failure to assemble platform credentials. Every variant names the *variable*, never a
/// value: an error message is a log line, and a log line is not a place for a secret.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CredentialError {
    #[error("{0} is not set — platform mode needs it (see the tm-platform crate docs)")]
    Missing(&'static str),
    #[error("{0} is set but empty")]
    Empty(&'static str),
    #[error("{0} is not valid UTF-8")]
    NotUtf8(&'static str),
    #[error(
        "{ENV_MQTT_USERNAME}/{ENV_MQTT_PASSWORD} are unset and {ENV_MQTT_ANONYMOUS} is not \
         set — refusing to connect anonymously by accident"
    )]
    AnonymousNotOptedIn,
}

/// Read one required environment variable, distinguishing unset from empty so the operator
/// is told which mistake they made.
fn required_env(var: &'static str) -> Result<String, CredentialError> {
    match env::var(var) {
        Ok(v) if v.is_empty() => Err(CredentialError::Empty(var)),
        Ok(v) => Ok(v),
        Err(env::VarError::NotPresent) => Err(CredentialError::Missing(var)),
        Err(env::VarError::NotUnicode(_)) => Err(CredentialError::NotUtf8(var)),
    }
}

fn env_flag(var: &str) -> bool {
    matches!(
        env::var(var).unwrap_or_default().trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Everything secret that platform mode needs, sourced from the environment.
#[derive(Debug, Clone)]
pub struct PlatformCredentials {
    /// Shared secret for the command envelope.
    pub command_secret: Secret,
    /// Broker username/password, or `None` when the operator opted into anonymous.
    pub broker_auth: Option<(String, Secret)>,
}

impl PlatformCredentials {
    /// Read credentials from the process environment.
    ///
    /// Every missing or empty variable is a hard error naming the variable, because
    /// starting platform mode without a command secret means the miner is reachable by
    /// anyone who can publish to the broker.
    pub fn from_env() -> Result<Self, CredentialError> {
        let command_secret = Secret::new(required_env(ENV_COMMAND_SECRET)?);

        let anonymous = env_flag(ENV_MQTT_ANONYMOUS);
        let username = env::var(ENV_MQTT_USERNAME).ok().filter(|v| !v.is_empty());
        let broker_auth = match (anonymous, username) {
            (_, Some(username)) => {
                // A username without a password is a misconfiguration, not an anonymous
                // connection: say which half is missing.
                Some((username, Secret::new(required_env(ENV_MQTT_PASSWORD)?)))
            }
            (true, None) => None,
            (false, None) => return Err(CredentialError::AnonymousNotOptedIn),
        };

        Ok(Self {
            command_secret,
            broker_auth,
        })
    }

    /// The worker id from the environment, for callers that have no `--worker-id`. Not a
    /// secret; here only because it shares the namespace.
    pub fn worker_id_from_env() -> Option<String> {
        env::var(ENV_WORKER_ID).ok().filter(|v| !v.is_empty())
    }
}

/// A broker URL with any embedded userinfo replaced by [`REDACTED`].
///
/// `mqtt://user:pass@host:1883` is a legal way to pass credentials and operators do it, so
/// every log line and error that mentions the broker URL goes through this first. Anything
/// that does not parse as `scheme://…` is returned unchanged — there is no userinfo to
/// hide in it — except that a bare `@` still triggers redaction, so a malformed URL cannot
/// smuggle a password past this.
pub fn redact_url(url: &str) -> String {
    let Some((scheme, rest)) = url.split_once("://") else {
        return match url.split_once('@') {
            Some((_, after)) => format!("{REDACTED}@{after}"),
            None => url.to_string(),
        };
    };
    // Userinfo, if present, is everything before the LAST '@' of the authority, and the
    // authority ends at the first '/', '?' or '#'.
    let authority_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let (authority, tail) = rest.split_at(authority_end);
    match authority.rfind('@') {
        Some(at) => format!("{scheme}://{REDACTED}@{}{tail}", &authority[at + 1..]),
        None => url.to_string(),
    }
}
