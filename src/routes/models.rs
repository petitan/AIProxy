use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::server::AppState;

/// GET /v1/models — list all configured models
pub async fn list_models(State(state): State<AppState>) -> Json<Value> {
    let models: Vec<Value> = state
        .config
        .models
        .values()
        .map(|m| {
            json!({
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": m.backend,
            })
        })
        .collect();

    Json(json!({
        "object": "list",
        "data": models,
    }))
}
