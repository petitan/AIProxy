use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::DefaultBodyLimit;
use axum::routing::{delete, get, post};
use axum::Router;
use tower::limit::ConcurrencyLimitLayer;
use tower_http::catch_panic::CatchPanicLayer;
use tower_http::timeout::TimeoutLayer;

use crate::config::{BackendType, ProxyConfig};
use crate::metrics::Metrics;
use crate::rate_limiter::{CircuitBreaker, RateLimiter};
use crate::routes;
use crate::vram::coordinator::VramCoordinator;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ProxyConfig>,
    pub client: reqwest::Client,
    pub stats: Arc<RequestStats>,
    pub started_at: Instant,
    pub vram: Arc<VramCoordinator>,
    pub rate_limiter: Arc<RateLimiter>,
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub metrics: Arc<Metrics>,
}

/// Simple request counter
pub struct RequestStats {
    total_requests: AtomicU64,
}

impl Default for RequestStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestStats {
    pub fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
        }
    }

    pub fn increment_requests(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            total_requests: self.total_requests.load(Ordering::Relaxed),
        }
    }
}

pub struct StatsSnapshot {
    pub total_requests: u64,
}

pub fn build_router(config: ProxyConfig) -> Router {
    // Extract GPU config
    let (total_vram, overhead, permanent) = extract_gpu_config(&config);

    // Find the Ollama backend URL (if any)
    let ollama_url = config
        .backends
        .values()
        .find(|b| b.backend_type == BackendType::Local && b.format == crate::config::ApiFormat::OpenAI)
        .map(|b| b.base_url.as_str());

    let vram = VramCoordinator::new(total_vram, overhead, permanent, ollama_url);

    tracing::info!(
        total_vram_mb = total_vram,
        overhead_mb = overhead,
        permanent_mb = permanent,
        budget_mb = total_vram.saturating_sub(overhead + permanent),
        "VRAM coordinator initialized"
    );

    // Connect timeout prevents hanging on unreachable backends.
    // Global request timeout is handled by TimeoutLayer (safety net above per-backend timeouts).
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("Failed to build HTTP client");

    // Collect local models for warmup (before config moves into Arc)
    let warmup_models: Vec<(String, u64)> = config
        .models
        .values()
        .filter(|m| {
            config.backends.get(&m.backend)
                .map(|b| b.backend_type == BackendType::Local)
                .unwrap_or(false)
        })
        .filter_map(|m| {
            m.estimated_vram_mb.map(|vram| (m.name.clone(), vram))
        })
        .collect();

    // Sync with Ollama on startup (discover already-loaded models), then warmup
    let startup_client = client.clone();
    let vram_arc = Arc::new(vram);
    let vram_startup = vram_arc.clone();
    tokio::spawn(async move {
        // Phase 1: Discover already-loaded models
        vram_startup.sync_with_ollama(&startup_client).await;

        // Phase 2: Warmup — preload all local models into VRAM
        if warmup_models.is_empty() {
            return;
        }
        tracing::info!(models = warmup_models.len(), "Starting model warmup");
        for (model_name, estimated_vram) in &warmup_models {
            let result = vram_startup
                .acquire(
                    &startup_client,
                    model_name,
                    crate::vram::coordinator::Priority::Low,
                    *estimated_vram,
                )
                .await;
            match result {
                crate::vram::coordinator::AcquireResult::Acquired => {
                    // Demote to Idle so warmup doesn't block eviction
                    vram_startup.demote_to_idle(model_name);
                    tracing::info!(model = %model_name, "Warmup complete");
                }
                _ => {
                    tracing::warn!(model = %model_name, "Warmup skipped (insufficient VRAM)");
                }
            }
        }
        tracing::info!("All model warmups finished");
    });

    // Spawn reservation reaper (expires timed-out reservations).
    // The AbortHandle is dropped here — the reaper runs until the tokio runtime shuts down,
    // which is the desired behavior (graceful shutdown aborts all spawned tasks).
    let _reaper_handle = vram_arc.spawn_reservation_reaper();

    // Spawn periodic Ollama sync — keeps coordinator slots in step with Ollama's actually
    // loaded models (prunes phantom slots Ollama unloaded on its own keep_alive timeout).
    // Without this the coordinator only synced once at startup and over-booked VRAM.
    let _ollama_sync_handle = vram_arc.spawn_ollama_sync(client.clone());

    let rate_limiter = RateLimiter::new(&config.rate_limits);
    let circuit_breaker = CircuitBreaker::new(
        config.circuit_breaker.failure_threshold,
        config.circuit_breaker.recovery_timeout_secs,
        config.backends.keys().map(|s| s.as_str()),
    );

    // Resolve API key from env var at startup (not per-request)
    // Safety: config.validate() already enforces fail-closed — if api_key_env is set,
    // the env var MUST exist and be non-empty, or startup is aborted.
    let auth_state = crate::middleware::AuthState {
        expected_key: config.proxy.api_key_env.as_ref().map(|env_var| {
            let key = std::env::var(env_var)
                .expect("BUG: validate() should have caught missing env var");
            tracing::info!(env_var = %env_var, "API key auth enabled");
            Arc::new(key)
        }),
    };

    // Extract hardening config before moving config into Arc
    let max_body_bytes = (config.proxy.max_body_size_mb * 1024 * 1024) as usize;
    let request_timeout = std::time::Duration::from_secs(config.proxy.request_timeout_secs);
    let max_connections = config.proxy.max_connections;

    tracing::info!(
        max_body_mb = config.proxy.max_body_size_mb,
        request_timeout_secs = config.proxy.request_timeout_secs,
        max_connections = max_connections,
        "Production hardening enabled"
    );

    let state = AppState {
        config: Arc::new(config),
        client,
        stats: Arc::new(RequestStats::new()),
        started_at: Instant::now(),
        vram: vram_arc,
        rate_limiter: Arc::new(rate_limiter),
        circuit_breaker: Arc::new(circuit_breaker),
        metrics: Arc::new(Metrics::new()),
    };

    // Protected routes (require auth if configured)
    let protected = Router::new()
        .route("/v1/chat/completions", post(routes::chat::chat_completions))
        .route("/v1/embeddings", post(routes::embeddings::create_embeddings))
        .route("/v1/models", get(routes::models::list_models))
        .route("/v1/vram/reserve", post(routes::vram::reserve_vram))
        .route("/v1/vram/reserve/{id}", delete(routes::vram::release_reservation))
        .route("/v1/vram/reservations", get(routes::vram::list_reservations))
        .route("/status", get(routes::health::status))
        .route("/metrics", get(routes::health::metrics))
        .layer(axum::middleware::from_fn_with_state(
            auth_state,
            crate::middleware::auth_guard,
        ));

    // Public routes (no auth)
    let public = Router::new()
        .route("/health", get(routes::health::health));

    protected
        .merge(public)
        .layer(axum::middleware::from_fn(crate::middleware::request_logger))
        .with_state(state)
        // Production hardening layers (outermost = last applied)
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .layer(TimeoutLayer::with_status_code(axum::http::StatusCode::GATEWAY_TIMEOUT, request_timeout))
        .layer(ConcurrencyLimitLayer::new(max_connections))
        .layer(CatchPanicLayer::new())
}

/// Extract GPU config from the TOML config
fn extract_gpu_config(config: &ProxyConfig) -> (u64, u64, u64) {
    let gpu = match &config.gpu {
        Some(gpu) => gpu,
        None => return (0, 0, 0),
    };

    let mut total_vram: u64 = 0;
    let mut overhead: u64 = 0;
    let mut permanent: u64 = 0;

    for (key, value) in &gpu.slots {
        if key.starts_with("slot_") {
            // GPU slot: {"total_vram_mb": N, "overhead_mb": M}
            if let Some(t) = value.get("total_vram_mb").and_then(|v| v.as_u64()) {
                total_vram += t;
            }
            if let Some(o) = value.get("overhead_mb").and_then(|v| v.as_u64()) {
                overhead += o;
            }
        } else if key.starts_with("permanent") || key == "permanent" {
            // Permanent reservations
            if let Some(obj) = value.as_object() {
                for (_name, res) in obj {
                    if let Some(v) = res.get("estimated_vram_mb").and_then(|v| v.as_u64()) {
                        permanent += v;
                    }
                }
            } else if let Some(v) = value.get("estimated_vram_mb").and_then(|v| v.as_u64()) {
                permanent += v;
            }
        }
    }

    (total_vram, overhead, permanent)
}
