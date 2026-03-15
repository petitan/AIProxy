use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::config::BackendType;
use crate::server::AppState;

/// Health check timeout — fast, don't block load balancers
const HEALTH_CHECK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// GET /health — readiness check with backend status
/// Checks run in parallel to minimize latency.
pub async fn health(State(state): State<AppState>) -> Json<Value> {
    // Spawn all checks in parallel
    let mut handles = Vec::new();
    for (name, backend) in &state.config.backends {
        let name = name.clone();
        let client = state.client.clone();
        let backend = backend.clone();
        handles.push(tokio::spawn(async move {
            let status = match backend.backend_type {
                BackendType::Local => check_local_backend(&client, &backend.base_url).await,
                BackendType::Remote => check_remote_backend(&backend),
            };
            (name, status)
        }));
    }

    let mut backend_statuses: Vec<Value> = Vec::new();
    let mut all_ok = true;

    for handle in handles {
        let (name, status) = match handle.await {
            Ok((name, status)) => (name, status),
            Err(_) => ("unknown".to_string(), "error"), // Fix #2: JoinError → error status
        };

        if status != "ok" {
            all_ok = false;
        }

        backend_statuses.push(json!({
            "name": name,
            "status": status,
        }));
    }

    Json(json!({
        "status": if all_ok { "ok" } else { "degraded" },
        "service": "ai-proxy",
        "version": env!("CARGO_PKG_VERSION"),
        "backends": backend_statuses,
    }))
}

/// Check local backend (Ollama): GET /api/tags with short timeout
async fn check_local_backend(client: &reqwest::Client, base_url: &str) -> &'static str {
    let url = format!("{}/api/tags", base_url.trim_end_matches('/'));
    match client.get(&url).timeout(HEALTH_CHECK_TIMEOUT).send().await {
        Ok(resp) if resp.status().is_success() => "ok",
        Ok(_) => "unavailable",
        Err(_) => "unavailable",
    }
}

/// Check remote backend: API key env var exists (don't call the API)
fn check_remote_backend(backend: &crate::config::BackendConfig) -> &'static str {
    match &backend.api_key_env {
        Some(env_var) => {
            if std::env::var(env_var).is_ok() {
                "ok"
            } else {
                "misconfigured"
            }
        }
        None => "ok",
    }
}

/// GET /metrics — per-model and per-backend request metrics
pub async fn metrics(State(state): State<AppState>) -> Json<Value> {
    Json(state.metrics.to_json())
}

/// GET /status — detailed proxy status
pub async fn status(State(state): State<AppState>) -> Json<Value> {
    let backends: Vec<Value> = state
        .config
        .backends
        .iter()
        .map(|(name, b)| {
            json!({
                "name": name,
                "type": format!("{:?}", b.backend_type).to_lowercase(),
                "format": format!("{:?}", b.format).to_lowercase(),
                "circuit": state.circuit_breaker.state(name),
            })
        })
        .collect();

    let models: Vec<Value> = state
        .config
        .models
        .values()
        .map(|m| {
            let mut model = json!({
                "name": m.name,
                "backend": m.backend,
            });
            if let Some(vram) = m.estimated_vram_mb {
                model["estimated_vram_mb"] = json!(vram);
            }
            if let Some(ref prio) = m.default_priority {
                model["default_priority"] = json!(prio);
            }
            model
        })
        .collect();

    let stats = state.stats.snapshot();

    // VRAM coordinator status
    let budget = state.vram.budget_summary();
    let loaded_models: Vec<Value> = state
        .vram
        .slot_status()
        .iter()
        .map(|slot| {
            json!({
                "model": slot.model,
                "priority": format!("{:?}", slot.priority),
                "estimated_vram_mb": slot.estimated_vram_mb,
            })
        })
        .collect();

    let mut vram_status = json!({
        "budget": budget,
        "loaded_models": loaded_models,
    });

    if let Some(gpu) = state.vram.gpu_info() {
        vram_status["gpu"] = json!({
            "name": gpu.gpu_name,
            "total_mb": gpu.total_mb,
            "used_mb": gpu.used_mb,
            "free_mb": gpu.free_mb,
        });
    }

    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": state.started_at.elapsed().as_secs(),
        "backends": backends,
        "models": models,
        "vram": vram_status,
        "stats": {
            "total_requests": stats.total_requests,
        }
    }))
}
