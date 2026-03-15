use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::server::AppState;
use crate::vram::coordinator::{Priority, ReserveError};

#[derive(Debug, Deserialize)]
pub struct ReserveRequest {
    pub required_free_mb: u64,
    #[serde(default = "default_priority")]
    pub priority: String,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_priority() -> String {
    "high".to_string()
}

fn default_timeout() -> u64 {
    300
}

/// POST /v1/vram/reserve
pub async fn reserve_vram(
    State(state): State<AppState>,
    Json(req): Json<ReserveRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if req.required_free_mb == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": "required_free_mb must be greater than 0",
                    "type": "invalid_request"
                }
            })),
        ));
    }

    // Max reservation timeout: 1 hour (prevents DoS via eternal reservations)
    const MAX_TIMEOUT_SECS: u64 = 3600;
    if req.timeout_secs > MAX_TIMEOUT_SECS {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("timeout_secs ({}) exceeds maximum allowed ({})", req.timeout_secs, MAX_TIMEOUT_SECS),
                    "type": "invalid_request",
                    "max_timeout_secs": MAX_TIMEOUT_SECS,
                }
            })),
        ));
    }

    // Upper bound: can't reserve more than the total VRAM budget
    let max_reservable = state.vram.budget_summary().budget_mb;
    if req.required_free_mb > max_reservable {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!(
                        "required_free_mb ({}) exceeds total VRAM budget ({} MB)",
                        req.required_free_mb, max_reservable
                    ),
                    "type": "invalid_request",
                    "max_reservable_mb": max_reservable,
                }
            })),
        ));
    }

    let priority = Priority::from_str(&req.priority);
    let reason = req.reason.unwrap_or_default();
    let timeout = Duration::from_secs(req.timeout_secs);

    match state
        .vram
        .reserve(&state.client, req.required_free_mb, priority, reason, timeout)
        .await
    {
        Ok(result) => Ok(Json(json!({
            "reservation_id": result.reservation_id,
            "granted_mb": result.granted_mb,
            "evicted_models": result.evicted_models,
            "expires_at": result.expires_at.to_rfc3339(),
            "budget": result.budget,
        }))),
        Err(ReserveError::InsufficientVram { required_mb, available_mb }) => Err((
            StatusCode::CONFLICT,
            Json(json!({
                "error": {
                    "message": format!("Insufficient VRAM: need {} MB, only {} MB available", required_mb, available_mb),
                    "type": "insufficient_vram",
                    "required_mb": required_mb,
                    "available_mb": available_mb,
                }
            })),
        )),
    }
}

/// DELETE /v1/vram/reserve/:id
pub async fn release_reservation(
    State(state): State<AppState>,
    Path(reservation_id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    match state.vram.release_reservation(&reservation_id) {
        Some(freed_mb) => Ok(Json(json!({
            "reservation_id": reservation_id,
            "released_mb": freed_mb,
            "budget": state.vram.budget_summary(),
        }))),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": {
                    "message": format!("Reservation '{}' not found", reservation_id),
                    "type": "not_found"
                }
            })),
        )),
    }
}

/// GET /v1/vram/reservations
pub async fn list_reservations(
    State(state): State<AppState>,
) -> Json<Value> {
    let reservations: Vec<Value> = state
        .vram
        .active_reservations()
        .iter()
        .map(|r| {
            let remaining = (r.expires_at - chrono::Utc::now()).num_seconds().max(0);
            json!({
                "id": r.id,
                "reserved_mb": r.reserved_mb,
                "reason": r.reason,
                "priority": format!("{:?}", r.priority),
                "expires_at": r.expires_at.to_rfc3339(),
                "remaining_secs": remaining,
            })
        })
        .collect();

    Json(json!({
        "reservations": reservations,
        "budget": state.vram.budget_summary(),
    }))
}
