use std::time::Instant;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::Json;
use serde_json::Value;

use super::common;
use crate::backends;
use crate::backends::TokenUsage;
use crate::config::BackendType;
use crate::error::ProxyError;
use crate::rate_limiter::RateLimitResult;
use crate::server::AppState;
use crate::vram::coordinator::{AcquireResult, Priority};

/// POST /v1/embeddings
pub async fn create_embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Response, ProxyError> {
    let model_name = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::BadRequest("Missing 'model' field".into()))?
        .to_string();

    let (model_key, model_config, backend_config) = state
        .config
        .resolve_model(&model_name)
        .ok_or_else(|| ProxyError::UnknownModel(model_name.to_string()))?;

    // ── Privacy check ──
    let privacy_header = &state.config.routing.privacy_header;
    let local_only = headers
        .get(privacy_header)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("local_only"))
        .unwrap_or(false);

    if local_only && backend_config.backend_type == BackendType::Remote {
        return Err(ProxyError::PrivacyViolation(model_name));
    }

    let backend_name = model_config.backend.clone();
    let backend_config = backend_config.clone();
    let estimated_vram = model_config.estimated_vram_mb.unwrap_or(0);
    let priority_header = &state.config.routing.priority_header;
    let priority = headers
        .get(priority_header)
        .and_then(|v| v.to_str().ok())
        .map(Priority::from_str)
        .or_else(|| model_config.default_priority.as_deref().map(Priority::from_str))
        .unwrap_or(Priority::Medium);

    let start = Instant::now();

    tracing::info!(
        model = %model_name,
        backend = %backend_name,
        "Embeddings request"
    );

    state.stats.increment_requests();

    // ── Circuit breaker check (BEFORE VRAM acquire) ──
    if !state.circuit_breaker.allow_request(&backend_name) {
        tracing::warn!(
            backend = %backend_name,
            model = %model_name,
            "Circuit breaker open — rejecting embeddings request"
        );
        state.metrics.record_circuit_breaker_trip(&backend_name);
        return Err(ProxyError::BackendUnavailable(format!(
            "Backend '{}' circuit breaker is open (too many consecutive failures)",
            backend_name
        )));
    }

    // ── VRAM acquire (local backends only) ──
    let is_local = backend_config.backend_type == BackendType::Local;
    if is_local && estimated_vram > 0 {
        match state
            .vram
            .acquire(&state.client, &model_name, priority, estimated_vram)
            .await
        {
            AcquireResult::Acquired => {}
            AcquireResult::InsufficientVram {
                needed_mb,
                available_mb,
            } => {
                return Err(ProxyError::BackendUnavailable(format!(
                    "Insufficient VRAM: need {} MB, only {} MB available",
                    needed_mb, available_mb
                )));
            }
            AcquireResult::EvictionFailed(e) => {
                return Err(ProxyError::BackendUnavailable(format!(
                    "VRAM eviction failed: {}",
                    e
                )));
            }
        }
    }

    // ── Rate limit check (after CB + VRAM, so no phantom consumption) ──
    if let RateLimitResult::Denied { retry_after_secs } = state.rate_limiter.check(model_key) {
        // Demote VRAM if we already acquired
        if is_local && estimated_vram > 0 {
            state.vram.demote_to_idle(&model_name);
        }
        tracing::warn!(
            model = %model_name,
            retry_after_secs = retry_after_secs,
            "Rate limit exceeded"
        );
        return Err(ProxyError::RateLimited { retry_after_secs });
    }

    // ── Inject model defaults (extra_body) ──
    model_config.apply_extra_body(&mut body);

    let retry_result = backends::dispatch_embeddings(&state.client, &backend_config, body).await;
    let result = retry_result.result;

    let latency_ms = start.elapsed().as_millis() as u64;

    // ── Circuit breaker feedback ──
    common::record_cb_outcome(
        &state.circuit_breaker,
        &backend_name,
        retry_result.failed_attempts,
        &result,
        false, // embeddings never stream
    );

    // ── Token usage + metrics reporting ──
    if let Ok(ref resp) = result {
        if let Some(usage) = resp.extensions().get::<TokenUsage>() {
            state.rate_limiter.report_tokens(model_key, usage.total_tokens);
            state.metrics.record_tokens(&model_name, usage.prompt_tokens, usage.completion_tokens);
        }
    }

    // ── Metrics: request + backend ──
    common::record_request_metrics(
        &state.metrics,
        &model_name,
        &backend_name,
        latency_ms,
        &result,
        false,
    );

    // ── Demote to idle after completion (local backends only) ──
    if is_local && estimated_vram > 0 {
        state.vram.demote_to_idle(&model_name);
    }

    match &result {
        Ok(resp) => {
            tracing::info!(
                model = %model_name,
                backend = %backend_name,
                status = resp.status().as_u16(),
                latency_ms = latency_ms,
                "Embeddings completed"
            );
        }
        Err(e) => {
            tracing::warn!(
                model = %model_name,
                backend = %backend_name,
                latency_ms = latency_ms,
                error = %e,
                "Embeddings failed"
            );
        }
    }

    result
}
