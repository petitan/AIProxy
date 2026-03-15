//! Integration tests — test API endpoints via axum Router without starting a server.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use serde_json::{json, Value};
use tower::ServiceExt;

/// Build a test router from inline TOML config (no real backends needed for validation tests)
fn test_router() -> axum::Router {
    let toml_str = r#"
[proxy]
listen = "127.0.0.1:19999"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[backends.anthropic]
type = "remote"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"

[models.qwen]
name = "qwen:7b"
backend = "ollama"
estimated_vram_mb = 6400

[models.claude]
name = "claude-sonnet"
backend = "anthropic"
"#;
    let config: ai_proxy::config::ProxyConfig = toml::from_str(toml_str).unwrap();
    ai_proxy::server::build_router(config)
}

/// Helper: send a request and get (status, body JSON)
async fn send(router: axum::Router, req: Request<Body>) -> (StatusCode, Value) {
    let response = router.oneshot(req).await.unwrap();
    let status = response.status();
    let body_bytes = response.into_body().collect().await.unwrap().to_bytes();
    let body: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
    (status, body)
}

// ─── GET /health ───

#[tokio::test]
async fn test_health_returns_json() {
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    // Status might be ok or degraded depending on backend availability
    assert!(status == StatusCode::OK);
    assert!(body.get("status").is_some());
    assert_eq!(body["service"], "ai-proxy");
    assert!(body.get("backends").unwrap().is_array());
}

// ─── GET /v1/models ───

#[tokio::test]
async fn test_models_list() {
    let req = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
    let models = body["data"].as_array().unwrap();
    assert_eq!(models.len(), 2);
    // Check model IDs exist
    let model_ids: Vec<&str> = models.iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert!(model_ids.contains(&"qwen:7b"));
    assert!(model_ids.contains(&"claude-sonnet"));
}

// ─── POST /v1/chat/completions — validation ───

#[tokio::test]
async fn test_chat_missing_model() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"messages": [{"role": "user", "content": "hi"}]}).to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("model"));
}

#[tokio::test]
async fn test_chat_missing_messages() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(json!({"model": "qwen:7b"}).to_string()))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("messages"));
}

#[tokio::test]
async fn test_chat_empty_messages() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "qwen:7b", "messages": []}).to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("empty"));
}

#[tokio::test]
async fn test_chat_message_missing_role() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "qwen:7b", "messages": [{"content": "hi"}]}).to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("role"));
}

#[tokio::test]
async fn test_chat_unknown_model() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "nonexistent", "messages": [{"role": "user", "content": "hi"}]})
                .to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found");
}

#[tokio::test]
async fn test_chat_temperature_out_of_range() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "qwen:7b",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 3.0
            })
            .to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("temperature"));
}

#[tokio::test]
async fn test_chat_top_p_out_of_range() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "qwen:7b",
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 1.5
            })
            .to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("top_p"));
}

#[tokio::test]
async fn test_chat_n_greater_than_one() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({
                "model": "qwen:7b",
                "messages": [{"role": "user", "content": "hi"}],
                "n": 3
            })
            .to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("n"));
}

// ─── POST /v1/embeddings — validation ───

#[tokio::test]
async fn test_embeddings_missing_model() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(json!({"input": "hello"}).to_string()))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("model"));
}

#[tokio::test]
async fn test_embeddings_unknown_model() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header("content-type", "application/json")
        .body(Body::from(
            json!({"model": "nonexistent", "input": "hello"}).to_string(),
        ))
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["type"], "not_found");
}

// ─── GET /status ───

#[tokio::test]
async fn test_status_returns_detailed_info() {
    let req = Request::builder()
        .uri("/status")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ok");
    assert!(body.get("version").is_some());
    assert!(body.get("uptime_seconds").is_some());
    assert!(body.get("backends").unwrap().is_array());
    assert!(body.get("models").unwrap().is_array());
    assert!(body.get("vram").is_some());
    assert!(body.get("stats").is_some());
}

// ─── 404 for unknown routes ───

#[tokio::test]
async fn test_unknown_route_404() {
    let req = Request::builder()
        .uri("/v1/nonexistent")
        .body(Body::empty())
        .unwrap();
    let response = test_router().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ─── X-Request-Id header ───

#[tokio::test]
async fn test_request_id_header_present() {
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let response = test_router().oneshot(req).await.unwrap();
    assert!(response.headers().get("x-request-id").is_some());
    // Should be a valid UUID
    let request_id = response.headers().get("x-request-id").unwrap().to_str().unwrap();
    assert_eq!(request_id.len(), 36); // UUID v4 format: 8-4-4-4-12
}

// ─── Auth middleware ───

/// Build a router with auth enabled (api_key_env resolved from env var)
fn test_router_with_auth() -> axum::Router {
    // Set a test API key env var
    std::env::set_var("TEST_PROXY_KEY", "secret-test-key-123");

    let toml_str = r#"
[proxy]
listen = "127.0.0.1:19999"
api_key_env = "TEST_PROXY_KEY"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[models.qwen]
name = "qwen:7b"
backend = "ollama"
"#;
    let config: ai_proxy::config::ProxyConfig = toml::from_str(toml_str).unwrap();
    ai_proxy::server::build_router(config)
}

#[tokio::test]
async fn test_auth_health_no_key_required() {
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router_with_auth(), req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["service"], "ai-proxy");
}

#[tokio::test]
async fn test_auth_missing_header_returns_401() {
    let req = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router_with_auth(), req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(body["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn test_auth_invalid_key_returns_401() {
    let req = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Bearer wrong-key")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router_with_auth(), req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Invalid"));
}

#[tokio::test]
async fn test_auth_valid_key_passes() {
    std::env::set_var("TEST_PROXY_KEY", "secret-test-key-123");
    let req = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Bearer secret-test-key-123")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router_with_auth(), req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["object"], "list");
}

#[tokio::test]
async fn test_auth_malformed_header_returns_401() {
    let req = Request::builder()
        .uri("/v1/models")
        .header("authorization", "Basic dXNlcjpwYXNz")
        .body(Body::empty())
        .unwrap();
    let (status, body) = send(test_router_with_auth(), req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Bearer"));
}

#[tokio::test]
async fn test_auth_status_endpoint_protected() {
    let req = Request::builder()
        .uri("/status")
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(test_router_with_auth(), req).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_no_auth_configured_allows_all() {
    // test_router() has no api_key_env → auth disabled
    let req = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let (status, _) = send(test_router(), req).await;
    assert_eq!(status, StatusCode::OK);
}

// ─── Body size limit ───

#[tokio::test]
async fn test_oversized_body_rejected() {
    // Default limit is 10 MB — send 11 MB
    let huge_body = "x".repeat(11 * 1024 * 1024);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(huge_body))
        .unwrap();
    let response = test_router().oneshot(req).await.unwrap();
    assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
}
