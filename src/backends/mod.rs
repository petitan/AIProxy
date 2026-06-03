pub mod anthropic;
pub mod ollama;

use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use reqwest::Client;
use serde_json::{json, Value};

use crate::config::{ApiFormat, BackendConfig};
use crate::error::ProxyError;
use crate::rate_limiter::CircuitBreaker;

/// Default `max_tokens` when a request omits it — see DEFAULT_MAX_TOKENS below.
///
/// Resolve the effective `max_tokens` estimate from an OpenAI request body: `max_tokens`,
/// then `max_completion_tokens`, else the default. Shared so the TPM-reservation estimate
/// (routes) and the backend conversions agree on one value.
pub fn resolve_max_tokens(body: &Value) -> u64 {
    body.get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

/// Context threaded into a streaming response so the generator can record the request's
/// outcome at stream end — which is AFTER the handler has returned. Carries the circuit
/// breaker (+ half-open probe token) for CB feedback, and an `actual_tokens` cell the
/// generator writes the real usage into; the streaming response body's Drop guard
/// (routes::chat) reads it to reconcile the TPM reservation — so the reservation is settled
/// even if the stream fails or the client disconnects (the generator is dropped mid-flight).
#[derive(Clone)]
pub struct StreamCtx {
    pub circuit_breaker: Arc<CircuitBreaker>,
    pub backend_name: String,
    pub probe: Option<u64>,
    /// Real token usage observed by the generator at clean completion (0 if the stream
    /// failed or was aborted before producing usage). Shared with the body Drop guard.
    pub actual_tokens: Arc<AtomicU64>,
}

/// Default `max_tokens` when a request omits it — shared across backends and the
/// streaming TPM estimator so the value never diverges between call sites.
pub const DEFAULT_MAX_TOKENS: u64 = 4096;

/// Token usage extracted from backend responses (non-streaming only).
/// Attached as a response extension for post-request token rate limiting.
#[derive(Debug, Clone)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

/// Build a masked error response for a failing backend.
///
/// The caller is responsible for logging the raw upstream body (which may contain
/// sensitive details) — this only constructs the client-facing response, exposing
/// the status code but never the upstream body. Falls back to `502 Bad Gateway`
/// if the status code is somehow invalid.
pub fn masked_error_response(status: StatusCode) -> Response {
    let error_json = json!({
        "error": {
            "message": format!("Backend error ({})", status),
            "type": "backend_error",
        }
    });
    let bytes = serde_json::to_vec(&error_json).unwrap_or_else(|_| b"{}".to_vec());
    Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY))
        .header("content-type", "application/json; charset=utf-8")
        .body(Body::from(bytes))
        .expect("valid error response builder")
}

/// Dispatch a chat completion request to the appropriate backend.
/// Non-streaming requests are retried on transient errors (429, 500, 502, 503, 529).
/// Returns RetryResult with failed_attempts count for circuit breaker feedback.
pub async fn dispatch_chat(
    client: &Client,
    backend: &BackendConfig,
    body: serde_json::Value,
    stream: bool,
    ctx: &StreamCtx,
) -> RetryResult {
    let max_retries = if stream { 0 } else { backend.effective_max_retries() };

    with_retry(max_retries, &backend.base_url, || async {
        match backend.format {
            ApiFormat::OpenAI => {
                ollama::chat_completions(client, backend, body.clone(), stream, ctx.clone()).await
            }
            ApiFormat::Anthropic => {
                anthropic::chat_completions(client, backend, body.clone(), stream, ctx.clone()).await
            }
        }
    })
    .await
}

/// Dispatch an embeddings request (only OpenAI-compatible backends).
/// Returns RetryResult with failed_attempts count for circuit breaker feedback.
pub async fn dispatch_embeddings(
    client: &Client,
    backend: &BackendConfig,
    body: serde_json::Value,
) -> RetryResult {
    let max_retries = backend.effective_max_retries();

    with_retry(max_retries, &backend.base_url, || async {
        match backend.format {
            ApiFormat::OpenAI => ollama::embeddings(client, backend, body.clone()).await,
            ApiFormat::Anthropic => Err(ProxyError::BadRequest(
                "Anthropic does not support embeddings endpoint".to_string(),
            )),
        }
    }).await
}

/// Transient HTTP status codes worth retrying
fn is_retryable_status(status: StatusCode) -> bool {
    matches!(
        status.as_u16(),
        429 | 500 | 502 | 503 | 529
    )
}

/// Check if a response is retryable (transient error status)
fn should_retry(result: &Result<Response, ProxyError>) -> bool {
    match result {
        Ok(resp) => is_retryable_status(resp.status()),
        Err(ProxyError::BackendUnavailable(_)) => true, // connection errors, timeouts
        _ => false,
    }
}

/// Result of a retried request, including how many failed attempts occurred.
pub struct RetryResult {
    pub result: Result<Response, ProxyError>,
    /// Number of failed attempts (0 = first attempt succeeded or no retries configured)
    pub failed_attempts: u32,
}

/// Retry wrapper with exponential backoff (1s, 2s, 4s, ...)
async fn with_retry<F, Fut>(
    max_retries: u32,
    backend_url: &str,
    f: F,
) -> RetryResult
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<Response, ProxyError>>,
{
    let mut last_result = f().await;
    let mut failed_attempts: u32 = 0;

    for attempt in 1..=max_retries {
        if !should_retry(&last_result) {
            return RetryResult { result: last_result, failed_attempts };
        }

        failed_attempts += 1;

        let delay = std::time::Duration::from_secs(1 << (attempt - 1).min(3)); // 1s, 2s, 4s, cap at 8s

        let status_info = match &last_result {
            Ok(resp) => format!("HTTP {}", resp.status()),
            Err(e) => format!("{}", e),
        };

        tracing::warn!(
            backend = %backend_url,
            attempt = attempt,
            max_retries = max_retries,
            error = %status_info,
            delay_ms = delay.as_millis() as u64,
            "Retrying request"
        );

        tokio::time::sleep(delay).await;
        last_result = f().await;
    }

    // If we exhausted all retries, the last attempt also failed
    if should_retry(&last_result) {
        failed_attempts += 1;
    }

    RetryResult { result: last_result, failed_attempts }
}

