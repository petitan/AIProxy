use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use axum::body::Body;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::Json;
use http_body::Frame;
use serde_json::Value;

use super::common;
use crate::backends;
use crate::backends::TokenUsage;
use crate::config::{BackendConfig, BackendType, ModelConfig};
use crate::error::ProxyError;
use crate::rate_limiter::{RateLimitResult, RateLimiter};
use crate::server::AppState;
use crate::vram::coordinator::{AcquireResult, Priority, VramCoordinator};

/// Active route after VRAM acquire + possible fallback resolution.
/// Replaces the fragile 6-tuple with named fields.
struct ActiveRoute {
    model_name: String,
    model_key: String,
    model_config: ModelConfig,
    backend_config: BackendConfig,
    backend_name: String,
    is_local: bool,
    estimated_vram: u64,
}

/// POST /v1/chat/completions
pub async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Response, ProxyError> {
    // ── Required field validation ──
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::BadRequest("Missing 'model' field".into()))?
        .to_string();

    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProxyError::BadRequest("Missing 'messages' array".into()))?;

    if messages.is_empty() {
        return Err(ProxyError::BadRequest("'messages' array is empty".into()));
    }

    for (i, msg) in messages.iter().enumerate() {
        if msg.get("role").and_then(|v| v.as_str()).is_none() {
            return Err(ProxyError::BadRequest(format!(
                "Message at index {} is missing 'role' field",
                i
            )));
        }
    }

    // ── Optional field validation ──
    if let Some(n) = body.get("n").and_then(|v| v.as_u64()) {
        if n > 1 {
            return Err(ProxyError::BadRequest(
                "'n' > 1 is not supported (proxy does not support multiple completions)".into(),
            ));
        }
    }

    if let Some(temp) = body.get("temperature").and_then(|v| v.as_f64()) {
        if !(0.0..=2.0).contains(&temp) {
            return Err(ProxyError::BadRequest(format!(
                "'temperature' must be between 0.0 and 2.0, got {}",
                temp
            )));
        }
    }

    if let Some(top_p) = body.get("top_p").and_then(|v| v.as_f64()) {
        if !(0.0..=1.0).contains(&top_p) {
            return Err(ProxyError::BadRequest(format!(
                "'top_p' must be between 0.0 and 1.0, got {}",
                top_p
            )));
        }
    }

    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // ── Model resolution ──
    let (model_key, model_config, backend_config) = state
        .config
        .resolve_model(&model_name)
        .ok_or_else(|| ProxyError::UnknownModel(model_name.clone()))?;

    // ── Privacy check (case-insensitive to prevent bypass) ──
    let privacy_header = &state.config.routing.privacy_header;
    let local_only = headers
        .get(privacy_header)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("local_only"))
        .unwrap_or(false);

    if local_only && backend_config.backend_type == BackendType::Remote {
        return Err(ProxyError::PrivacyViolation(model_name));
    }

    // ── Priority: header override > model default > Medium ──
    let priority_header = &state.config.routing.priority_header;
    let priority = headers
        .get(priority_header)
        .and_then(|v| v.to_str().ok())
        .map(Priority::from_str)
        .or_else(|| model_config.default_priority.as_deref().map(Priority::from_str))
        .unwrap_or(Priority::Medium);

    // Clone to owned values for the rest of the handler
    let original_backend_name = model_config.backend.clone();
    let model_key_owned = model_key.to_string();
    let model_config = model_config.clone();
    let backend_config = backend_config.clone();
    let estimated_vram = model_config.estimated_vram_mb.unwrap_or(0);

    let start = Instant::now();

    tracing::info!(
        model = %model_name,
        stream = stream,
        backend = %original_backend_name,
        messages = messages.len(),
        priority = ?priority,
        local_only = local_only,
        "Chat completion request"
    );

    state.stats.increment_requests();

    // ── Circuit breaker check on ORIGINAL backend (BEFORE VRAM acquire to avoid wasted acquire+demote) ──
    let (allowed, probe_original) = state.circuit_breaker.allow_request(&original_backend_name);
    if !allowed {
        tracing::warn!(
            backend = %original_backend_name,
            model = %model_name,
            "Circuit breaker open — rejecting request"
        );
        state.metrics.record_circuit_breaker_trip(&original_backend_name);
        return Err(ProxyError::BackendUnavailable(format!(
            "Backend '{}' circuit breaker is open (too many consecutive failures)",
            original_backend_name
        )));
    }

    // ── VRAM acquire (local backends only) — with fallback ──
    let is_local = backend_config.backend_type == BackendType::Local;
    let active = if is_local && estimated_vram > 0 {
        match state
            .vram
            .acquire(&state.client, &model_name, priority, estimated_vram)
            .await
        {
            AcquireResult::Acquired => ActiveRoute {
                model_name: model_name.clone(),
                model_key: model_key_owned.clone(),
                model_config: model_config.clone(),
                backend_config: backend_config.clone(),
                backend_name: original_backend_name.clone(),
                is_local: true,
                estimated_vram,
            },
            AcquireResult::InsufficientVram { needed_mb, available_mb }
                if !local_only =>
            {
                try_fallback(
                    &state, &model_config, &model_key_owned, &model_name,
                    local_only, needed_mb, available_mb,
                )?
            }
            AcquireResult::EvictionFailed(_) if !local_only => {
                let budget = state.vram.budget_summary();
                try_fallback(
                    &state, &model_config, &model_key_owned, &model_name,
                    local_only, estimated_vram, budget.available_mb,
                )?
            }
            AcquireResult::InsufficientVram { needed_mb, available_mb } => {
                return Err(ProxyError::BackendUnavailable(format!(
                    "Insufficient VRAM: need {} MB, only {} MB available (local_only, no fallback)",
                    needed_mb, available_mb
                )));
            }
            AcquireResult::EvictionFailed(e) => {
                return Err(ProxyError::BackendUnavailable(format!(
                    "VRAM eviction failed: {} (local_only, no fallback)",
                    e
                )));
            }
        }
    } else {
        ActiveRoute {
            model_name: model_name.clone(),
            model_key: model_key_owned.clone(),
            model_config: model_config.clone(),
            backend_config: backend_config.clone(),
            backend_name: original_backend_name.clone(),
            is_local,
            estimated_vram,
        }
    };

    // ── Circuit breaker check on ACTIVE backend + resolve the half-open probe token ──
    // If a VRAM fallback changed the backend, the original backend's probe (if any) is
    // abandoned — release it as a failure so that circuit isn't left stuck half-open.
    let active_probe = if active.backend_name != original_backend_name {
        if probe_original.is_some() {
            state.circuit_breaker.record_failure(&original_backend_name, probe_original);
        }
        let (allowed, probe) = state.circuit_breaker.allow_request(&active.backend_name);
        if !allowed {
            tracing::warn!(
                backend = %active.backend_name,
                model = %active.model_name,
                "Circuit breaker open on fallback backend — rejecting request"
            );
            state.metrics.record_circuit_breaker_trip(&active.backend_name);
            if active.is_local && active.estimated_vram > 0 {
                state.vram.demote_to_idle(&active.model_name);
            }
            return Err(ProxyError::BackendUnavailable(format!(
                "Backend '{}' circuit breaker is open (too many consecutive failures)",
                active.backend_name
            )));
        }
        probe
    } else {
        probe_original
    };

    // ── Inject model defaults (extra_body) from the ACTIVE model config ──
    active.model_config.apply_extra_body(&mut body);

    // If we fell back to a different model, update the model name in the body
    if active.model_name != model_name {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("model".to_string(), Value::String(active.model_name.clone()));
        }
    }

    // ── Estimate tokens for the TPM pre-reservation (max_tokens, or a default) ──
    let request_max_tokens = backends::resolve_max_tokens(&body);

    // ── Rate limit check (after CB + VRAM; pre-reserves the estimate so concurrent
    //    requests can't burst past the TPM budget — reconciled to real usage below) ──
    if let RateLimitResult::Denied { retry_after_secs } =
        state.rate_limiter.check(&active.model_key, request_max_tokens)
    {
        // Release a half-open probe (the request never reached the backend) + VRAM. A None
        // probe means the circuit was Closed — a rate limit is not a backend failure, so
        // we must NOT record one in that case.
        if active_probe.is_some() {
            state.circuit_breaker.record_failure(&active.backend_name, active_probe);
        }
        if active.is_local && active.estimated_vram > 0 {
            state.vram.demote_to_idle(&active.model_name);
        }
        tracing::warn!(
            model = %active.model_name,
            retry_after_secs = retry_after_secs,
            "Rate limit exceeded"
        );
        return Err(ProxyError::RateLimited { retry_after_secs });
    }

    // ── Forward to backend ──
    // Shared cell the stream generator writes the real usage into; the response body's
    // Drop guard reads it to reconcile the TPM reservation (so it settles even on abort).
    let actual_tokens = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let stream_ctx = backends::StreamCtx {
        circuit_breaker: state.circuit_breaker.clone(),
        backend_name: active.backend_name.clone(),
        probe: active_probe,
        actual_tokens: actual_tokens.clone(),
    };
    let retry_result =
        backends::dispatch_chat(&state.client, &active.backend_config, body, stream, &stream_ctx)
            .await;
    let result = retry_result.result;

    let latency_ms = start.elapsed().as_millis() as u64;

    // ── Circuit breaker feedback (on the ACTIVE backend). For a streaming 2xx the outcome
    //    is recorded by the stream generator at stream end (it carries the same probe). ──
    common::record_cb_outcome(
        &state.circuit_breaker,
        &active.backend_name,
        retry_result.failed_attempts,
        &result,
        stream,
        active_probe,
    );

    // ── Reconcile the TPM reservation against the real usage ──
    match &result {
        Ok(resp) if resp.status().is_success() => {
            if !stream {
                // Non-streaming: exact token count from the response extension.
                if let Some(usage) = resp.extensions().get::<TokenUsage>() {
                    state
                        .rate_limiter
                        .reconcile_tokens(&active.model_key, request_max_tokens, usage.total_tokens);
                    state.metrics.record_tokens(
                        &active.model_name,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                    );
                }
                // else: usage unknown → leave the conservative reservation in place.
            }
            // Streaming 2xx: the generator reconciles at stream end.
        }
        _ => {
            // Non-2xx or transport error → refund the full reservation (the request did
            // not consume the estimated tokens).
            state
                .rate_limiter
                .reconcile_tokens(&active.model_key, request_max_tokens, 0);
        }
    }

    // ── Metrics: request + backend (using ACTIVE names) ──
    common::record_request_metrics(
        &state.metrics,
        &active.model_name,
        &active.backend_name,
        latency_ms,
        &result,
        stream,
    );

    // ── Streaming 2xx: wrap the body in a guard that, on stream end OR client disconnect
    //    (Drop), reconciles the TPM reservation to the real usage the generator published
    //    and demotes VRAM (local). This settles the reservation even if the stream is
    //    aborted mid-flight (the generator dropped without reconciling). All backends are
    //    wrapped — remote streams reserve TPM too. ──
    if stream {
        match result {
            Ok(resp) if resp.status().is_success() => {
                let (parts, body) = resp.into_parts();
                let vram = (active.is_local && active.estimated_vram > 0)
                    .then(|| (state.vram.clone(), active.model_name.clone()));
                let wrapped = StreamEndGuard::new(
                    body,
                    state.rate_limiter.clone(),
                    active.model_key.clone(),
                    request_max_tokens,
                    actual_tokens.clone(),
                    vram,
                );
                tracing::info!(
                    model = %active.model_name,
                    backend = %active.backend_name,
                    stream = true,
                    status = parts.status.as_u16(),
                    latency_ms = latency_ms,
                    "Chat completion streaming started"
                );
                return Ok(Response::from_parts(parts, Body::new(wrapped)));
            }
            other => {
                // Streaming error / non-2xx: no stream body to guard. TPM already refunded
                // above; demote VRAM now (local).
                if active.is_local && active.estimated_vram > 0 {
                    state.vram.demote_to_idle(&active.model_name);
                }
                match &other {
                    Ok(resp) => tracing::info!(
                        model = %active.model_name, backend = %active.backend_name,
                        stream = true, status = resp.status().as_u16(), latency_ms = latency_ms,
                        "Chat completion completed"
                    ),
                    Err(e) => tracing::warn!(
                        model = %active.model_name, backend = %active.backend_name,
                        stream = true, latency_ms = latency_ms, error = %e,
                        "Chat completion failed (streaming)"
                    ),
                }
                return other;
            }
        }
    }

    // ── Non-streaming: demote to idle after completion (local backends only) ──
    if active.is_local && active.estimated_vram > 0 {
        state.vram.demote_to_idle(&active.model_name);
    }

    match &result {
        Ok(resp) => {
            tracing::info!(
                model = %active.model_name,
                original_model = %model_name,
                backend = %active.backend_name,
                stream = stream,
                status = resp.status().as_u16(),
                latency_ms = latency_ms,
                "Chat completion completed"
            );
        }
        Err(e) => {
            tracing::warn!(
                model = %active.model_name,
                original_model = %model_name,
                backend = %active.backend_name,
                stream = stream,
                latency_ms = latency_ms,
                error = %e,
                "Chat completion failed"
            );
        }
    }

    result
}

/// Result of a successful fallback resolution
struct FallbackResult {
    model_name: String,
    model_key: String,
    model_config: ModelConfig,
    backend_config: BackendConfig,
    backend_name: String,
}

/// Try to fall back to a remote model when local VRAM is insufficient.
/// Returns the full fallback model config, backend config, and names,
/// or an error if fallback is not possible.
fn try_fallback(
    state: &AppState,
    model_config: &ModelConfig,
    original_model_key: &str,
    original_model: &str,
    local_only: bool,
    needed_mb: u64,
    available_mb: u64,
) -> Result<ActiveRoute, ProxyError> {
    if local_only {
        return Err(ProxyError::BackendUnavailable(format!(
            "Insufficient VRAM: need {} MB, only {} MB available (local_only, no fallback)",
            needed_mb, available_mb
        )));
    }

    // Check if model has explicit fallback configured
    let fb = match &model_config.fallback {
        Some(key) => resolve_fallback(state, key, original_model, needed_mb, available_mb)?,
        None => {
            // No explicit fallback — check default_fallback policy
            if state.config.routing.default_fallback == "remote" {
                // Auto-select first available remote model
                auto_select_remote_fallback(state, original_model_key, original_model, needed_mb, available_mb)?
            } else {
                // "deny" or unknown policy
                return Err(ProxyError::BackendUnavailable(format!(
                    "Insufficient VRAM: need {} MB, only {} MB available (no fallback configured)",
                    needed_mb, available_mb
                )));
            }
        }
    };

    // Warn if fallback is also local (no VRAM acquire will happen for it)
    if fb.backend_config.backend_type == BackendType::Local {
        tracing::warn!(
            original_model = %original_model,
            fallback_model = %fb.model_name,
            "Fallback model is also local — no VRAM acquire for fallback, OOM risk"
        );
    }

    let fb_is_local = fb.backend_config.backend_type == BackendType::Local;
    let fb_vram = fb.model_config.estimated_vram_mb.unwrap_or(0);

    Ok(ActiveRoute {
        model_name: fb.model_name,
        model_key: fb.model_key,
        model_config: fb.model_config,
        backend_config: fb.backend_config,
        backend_name: fb.backend_name,
        is_local: fb_is_local,
        estimated_vram: fb_vram,
    })
}

/// Resolve an explicit fallback model by config key
fn resolve_fallback(
    state: &AppState,
    fallback_key: &str,
    original_model: &str,
    needed_mb: u64,
    available_mb: u64,
) -> Result<FallbackResult, ProxyError> {
    let (_, fallback_model, fallback_backend) = state
        .config
        .resolve_model_by_key(fallback_key)
        .ok_or_else(|| {
            ProxyError::BackendNotConfigured(format!(
                "Fallback model '{}' not found",
                fallback_key
            ))
        })?;

    tracing::warn!(
        original_model = %original_model,
        fallback_model = %fallback_model.name,
        fallback_backend = %fallback_model.backend,
        needed_mb = needed_mb,
        available_mb = available_mb,
        "VRAM insufficient — falling back"
    );

    Ok(FallbackResult {
        model_name: fallback_model.name.clone(),
        model_key: fallback_key.to_string(),
        model_config: fallback_model.clone(),
        backend_config: fallback_backend.clone(),
        backend_name: fallback_model.backend.clone(),
    })
}

/// Auto-select a remote model when default_fallback == "remote" and no explicit fallback is set.
fn auto_select_remote_fallback(
    state: &AppState,
    original_model_key: &str,
    original_model: &str,
    needed_mb: u64,
    available_mb: u64,
) -> Result<FallbackResult, ProxyError> {
    // Deterministic selection: iterate model keys in sorted order. `config.models` is a
    // HashMap whose iteration order is randomized per process, so without sorting the
    // auto-selected fallback model could differ across restarts — an unpredictable choice
    // for anyone relying on default_fallback=remote.
    let mut keys: Vec<&String> = state.config.models.keys().collect();
    keys.sort();
    for key in keys {
        if key == original_model_key {
            continue;
        }
        let model = &state.config.models[key];
        if let Some(backend) = state.config.backends.get(&model.backend) {
            if backend.backend_type == BackendType::Remote {
                tracing::warn!(
                    original_model = %original_model,
                    fallback_model = %model.name,
                    fallback_backend = %model.backend,
                    needed_mb = needed_mb,
                    available_mb = available_mb,
                    "VRAM insufficient — auto-fallback to remote (default_fallback=remote)"
                );
                return Ok(FallbackResult {
                    model_name: model.name.clone(),
                    model_key: key.clone(),
                    model_config: model.clone(),
                    backend_config: backend.clone(),
                    backend_name: model.backend.clone(),
                });
            }
        }
    }

    Err(ProxyError::BackendUnavailable(format!(
        "Insufficient VRAM: need {} MB, only {} MB available (default_fallback=remote but no remote model configured)",
        needed_mb, available_mb
    )))
}

/// Body wrapper for streaming responses. When the stream ends OR the client disconnects
/// (Drop), it reconciles the TPM reservation to the real usage the generator published into
/// `actual_tokens` (0 if the stream aborted before producing usage → full refund), and
/// demotes the model to Idle priority (local backends). Doing this in Drop — not in the
/// generator — guarantees both settle even when the generator is dropped mid-flight.
struct StreamEndGuard {
    inner: Body,
    rate_limiter: Arc<RateLimiter>,
    model_key: String,
    reserved_tokens: u64,
    actual_tokens: Arc<AtomicU64>,
    /// `Some((coordinator, model_name))` for local backends that acquired VRAM.
    vram: Option<(Arc<VramCoordinator>, String)>,
}

impl StreamEndGuard {
    fn new(
        inner: Body,
        rate_limiter: Arc<RateLimiter>,
        model_key: String,
        reserved_tokens: u64,
        actual_tokens: Arc<AtomicU64>,
        vram: Option<(Arc<VramCoordinator>, String)>,
    ) -> Self {
        Self { inner, rate_limiter, model_key, reserved_tokens, actual_tokens, vram }
    }
}

impl http_body::Body for StreamEndGuard {
    type Data = bytes::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        // SAFETY: we only pin-project to `inner`, which is Unpin (axum::body::Body is Unpin)
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for StreamEndGuard {
    fn drop(&mut self) {
        let actual = self.actual_tokens.load(Ordering::Relaxed);
        self.rate_limiter
            .reconcile_tokens(&self.model_key, self.reserved_tokens, actual);
        if let Some((vram, model)) = &self.vram {
            vram.demote_to_idle(model);
        }
    }
}
