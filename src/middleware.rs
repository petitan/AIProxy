use std::sync::Arc;
use std::time::Instant;

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;

/// Auth middleware state — holds the expected API key (resolved from env at startup).
#[derive(Clone)]
pub struct AuthState {
    /// None = auth disabled
    pub expected_key: Option<Arc<String>>,
}

/// Auth middleware: validates `Authorization: Bearer <key>` against configured API key.
/// Returns 401 on missing/invalid token.
pub async fn auth_guard(
    axum::extract::State(auth): axum::extract::State<AuthState>,
    request: Request,
    next: Next,
) -> Response {
    let expected = match &auth.expected_key {
        Some(key) => key,
        None => return next.run(request).await, // auth disabled
    };

    let auth_header = request
        .headers()
        .get("authorization")
        .and_then(|v| v.to_str().ok());

    match auth_header {
        Some(header) if header.starts_with("Bearer ") => {
            let token = &header[7..];
            // Constant-time comparison to prevent timing attacks
            if token.as_bytes().ct_eq(expected.as_bytes()).into() {
                next.run(request).await
            } else {
                tracing::warn!("Invalid API key");
                unauthorized("Invalid API key")
            }
        }
        Some(_) => {
            tracing::warn!("Malformed Authorization header");
            unauthorized("Authorization header must use Bearer scheme")
        }
        None => {
            tracing::warn!("Missing Authorization header");
            unauthorized("Missing Authorization header")
        }
    }
}

fn unauthorized(message: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "authentication_error",
        }
    });
    (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
}

/// Request logging middleware: request_id, method, path, status, latency_ms
pub async fn request_logger(request: Request, next: Next) -> Response {
    let request_id = uuid::Uuid::new_v4().to_string();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let start = Instant::now();

    let mut response = next.run(request).await;

    let latency_ms = start.elapsed().as_millis() as u64;
    let status = response.status().as_u16();

    // Inject X-Request-Id header for client-side correlation
    response.headers_mut().insert(
        "x-request-id",
        request_id.parse().unwrap_or_else(|_| "unknown".parse().unwrap()),
    );

    tracing::info!(
        request_id = %request_id,
        method = %method,
        path = %path,
        status = status,
        latency_ms = latency_ms,
        "Request completed"
    );

    response
}
