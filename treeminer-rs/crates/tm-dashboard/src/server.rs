//! The read-only operator console. Port of the Crow app in `src/LocalServer.cpp`.
//!
//! Every route is a GET that reports state. There is no control endpoint, no key or seed
//! material in any payload, and the only identity exposed is the public reward address —
//! this is what makes binding `0.0.0.0` by default defensible.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use parking_lot::Mutex;
use tokio::net::TcpListener;

use crate::json::{
    platform_payload, rig_payload, stats_payload, STATS_CACHE_SECONDS,
};
use crate::page::PAGE;
use crate::stats::StatsSource;
use crate::url::{ready_message, InterfaceSource, SystemInterfaces, DEFAULT_BIND, DEFAULT_PORT};

/// Relative location of the optional background image, as the C++ resolves it.
pub const HASHFIELD_ASSET: &str = "res/dashboard/hashfield.webp";

#[derive(Debug, thiserror::Error)]
pub enum DashboardError {
    #[error("dashboard bind address must be an IP literal, got `{0}`")]
    InvalidBind(String),
    #[error("dashboard could not listen on {addr}: {source}")]
    Listen {
        addr: SocketAddr,
        #[source]
        source: std::io::Error,
    },
    #[error("dashboard server stopped: {0}")]
    Serve(#[source] std::io::Error),
}

/// Operator-supplied console settings. `--dashboard-bind` / `--dashboard-port` in the CLI
/// crate map onto `bind` and `port`.
#[derive(Clone)]
pub struct DashboardConfig {
    pub bind: String,
    pub port: u16,
    /// Directory the `hashfield.webp` asset is resolved against (the C++ uses the process
    /// working directory).
    pub asset_root: PathBuf,
    /// How advertised URLs are discovered. Swappable so tests are not at the mercy of the
    /// host's interfaces.
    pub interfaces: Arc<dyn InterfaceSource>,
}

impl std::fmt::Debug for DashboardConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DashboardConfig")
            .field("bind", &self.bind)
            .field("port", &self.port)
            .field("asset_root", &self.asset_root)
            .finish_non_exhaustive()
    }
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND.to_string(),
            port: DEFAULT_PORT,
            asset_root: PathBuf::from("."),
            interfaces: Arc::new(SystemInterfaces),
        }
    }
}

impl DashboardConfig {
    pub fn new(bind: impl Into<String>, port: u16) -> Self {
        Self {
            bind: bind.into(),
            port,
            ..Self::default()
        }
    }

    fn socket_addr(&self) -> Result<SocketAddr, DashboardError> {
        let ip: IpAddr = self
            .bind
            .parse()
            .map_err(|_| DashboardError::InvalidBind(self.bind.clone()))?;
        Ok(SocketAddr::new(ip, self.port))
    }

    /// The startup banner: reachable URLs, never the wildcard bind.
    pub fn ready_message(&self) -> String {
        ready_message(&self.bind, self.port, self.interfaces.as_ref())
    }
}

struct AppState {
    source: Arc<dyn StatsSource>,
    config: DashboardConfig,
    /// `/stats` is polled by fleet tooling; the C++ caches the rendered body for two
    /// seconds so a polling storm cannot turn into a snapshot storm.
    stats_cache: Mutex<Option<(Instant, String)>>,
}

impl AppState {
    fn cached_stats(&self) -> String {
        let mut cache = self.stats_cache.lock();
        let fresh = cache
            .as_ref()
            .is_some_and(|(at, _)| at.elapsed() < Duration::from_secs(STATS_CACHE_SECONDS));
        if !fresh {
            let body = stats_payload(
                &self.source.snapshot(),
                &self.config.bind,
                self.config.port,
                self.config.interfaces.as_ref(),
            )
            .to_string();
            *cache = Some((Instant::now(), body));
        }
        cache
            .as_ref()
            .map(|(_, body)| body.clone())
            .unwrap_or_default()
    }
}

fn json_response(body: String) -> Response {
    (
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
}

async fn healthz() -> Response {
    json_response(r#"{"ok":true}"#.to_string())
}

async fn stats(State(state): State<Arc<AppState>>) -> Response {
    json_response(state.cached_stats())
}

async fn rig(State(state): State<Arc<AppState>>) -> Response {
    json_response(
        rig_payload(
            &state.source.snapshot(),
            &state.config.bind,
            state.config.port,
            state.config.interfaces.as_ref(),
        )
        .to_string(),
    )
}

async fn platform(State(state): State<Arc<AppState>>) -> Response {
    json_response(platform_payload(&state.source.snapshot()).to_string())
}

async fn index() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        PAGE,
    )
        .into_response()
}

async fn hashfield(State(state): State<Arc<AppState>>) -> Response {
    let path: &Path = &state.config.asset_root.join(HASHFIELD_ASSET);
    match tokio::fs::read(path).await {
        Ok(bytes) => (
            [
                (header::CONTENT_TYPE, "image/webp"),
                (header::CACHE_CONTROL, "public, max-age=86400"),
            ],
            Body::from(bytes),
        )
            .into_response(),
        // The page styles around a missing background rather than failing, so an absent
        // asset is a 404 with a plain body, exactly as the C++ answers.
        Err(_) => (StatusCode::NOT_FOUND, "asset unavailable").into_response(),
    }
}

/// Build the router without binding a socket — useful for in-process tests.
pub fn router(source: Arc<dyn StatsSource>, config: DashboardConfig) -> Router {
    let state = Arc::new(AppState {
        source,
        config,
        stats_cache: Mutex::new(None),
    });
    Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/stats", get(stats))
        .route("/api/v1/status", get(stats))
        .route("/api/rig", get(rig))
        .route("/platform/status", get(platform))
        .route("/assets/hashfield.webp", get(hashfield))
        .with_state(state)
}

/// A console that has claimed its port but is not yet serving. Binding separately is what
/// lets a caller (or a test on port 0) learn the real address before the accept loop runs,
/// and it surfaces "port already in use" as an error instead of a late thread death.
pub struct DashboardServer {
    listener: TcpListener,
    router: Router,
    local_addr: SocketAddr,
    config: DashboardConfig,
}

impl DashboardServer {
    pub async fn bind(
        config: DashboardConfig,
        source: Arc<dyn StatsSource>,
    ) -> Result<Self, DashboardError> {
        let addr = config.socket_addr()?;
        let listener = TcpListener::bind(addr)
            .await
            .map_err(|source| DashboardError::Listen { addr, source })?;
        let local_addr = listener
            .local_addr()
            .map_err(|source| DashboardError::Listen { addr, source })?;
        // Port 0 means "any free port"; every URL this server prints or serves must name
        // the port it actually got, not the request.
        let mut config = config;
        config.port = local_addr.port();
        let router = router(source, config.clone());
        Ok(Self {
            listener,
            router,
            local_addr,
            config,
        })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Banner reflecting the port actually bound, so `--dashboard-port 0` still prints a
    /// URL an operator can use.
    pub fn ready_message(&self) -> String {
        self.config.ready_message()
    }

    pub async fn serve(self) -> Result<(), DashboardError> {
        axum::serve(self.listener, self.router)
            .await
            .map_err(DashboardError::Serve)
    }

    pub async fn serve_with_shutdown<F>(self, shutdown: F) -> Result<(), DashboardError>
    where
        F: std::future::Future<Output = ()> + Send + 'static,
    {
        axum::serve(self.listener, self.router)
            .with_graceful_shutdown(shutdown)
            .await
            .map_err(DashboardError::Serve)
    }
}
