//! Shared request-handling helpers used by both the chat and embeddings routes.
//! These capture the parts of the two handlers that must stay identical — circuit
//! breaker feedback and metrics recording — so a fix lands in one place.

use axum::response::Response;

use crate::error::ProxyError;
use crate::metrics::Metrics;
use crate::rate_limiter::CircuitBreaker;

/// Record circuit-breaker feedback for a completed request.
///
/// Each failed retry attempt counts as a separate failure, then the final result is
/// recorded: success on 2xx, failure on 5xx or connection error. **4xx is neutral** —
/// it does not reset `consecutive_failures`, preventing CB bypass via crafted 400s.
///
/// For **streaming** requests a 2xx here only means the upstream returned headers — the
/// body may still fail mid-stream (read error, chunk timeout, empty completion). So the
/// final success/failure is NOT recorded here for a streaming 2xx; the stream generator
/// records it at stream end instead. Retry failures and non-2xx are still recorded here.
pub fn record_cb_outcome(
    cb: &CircuitBreaker,
    backend_name: &str,
    failed_attempts: u32,
    result: &Result<Response, ProxyError>,
    stream: bool,
) {
    for _ in 0..failed_attempts {
        cb.record_failure(backend_name);
    }
    match result {
        // Non-streaming 2xx: success now. Streaming 2xx is intentionally NOT matched here —
        // it falls through to the neutral arm, and the stream generator records the outcome
        // at stream end (see doc comment).
        Ok(resp) if resp.status().is_success() && !stream => cb.record_success(backend_name),
        Ok(resp) if resp.status().is_server_error() => cb.record_failure(backend_name),
        Err(_) => cb.record_failure(backend_name),
        _ => {} // 4xx, or streaming 2xx (deferred): neutral
    }
}

/// Record request + backend metrics for a completed request.
pub fn record_request_metrics(
    metrics: &Metrics,
    model_name: &str,
    backend_name: &str,
    latency_ms: u64,
    result: &Result<Response, ProxyError>,
    stream: bool,
) {
    let is_error = result.as_ref().map_or(true, |r| !r.status().is_success());
    metrics.record_request(model_name, latency_ms, is_error, stream);
    let (status_code, is_conn_err) = match result {
        Ok(resp) => (Some(resp.status().as_u16()), false),
        Err(_) => (None, true),
    };
    metrics.record_backend_request(backend_name, status_code, is_conn_err);
}
