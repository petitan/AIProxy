use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("Unknown model: {0}")]
    UnknownModel(String),

    #[error("Backend not configured: {0}")]
    BackendNotConfigured(String),

    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("Request to backend failed: {0}")]
    BackendRequest(#[from] reqwest::Error),

    #[error("Invalid request: {0}")]
    BadRequest(String),

    #[error("Format conversion error: {0}")]
    FormatConversion(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Privacy violation: model {0} requires remote backend but local_only requested")]
    PrivacyViolation(String),

    #[error("Rate limit exceeded: retry after {retry_after_secs:.1}s")]
    RateLimited { retry_after_secs: f64 },
}

impl IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let retry_after = match &self {
            ProxyError::RateLimited { retry_after_secs } => Some(retry_after_secs.ceil() as u64),
            _ => None,
        };

        let status = match &self {
            ProxyError::UnknownModel(_) => StatusCode::NOT_FOUND,
            ProxyError::BackendNotConfigured(_) => StatusCode::BAD_GATEWAY,
            ProxyError::BackendUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ProxyError::BackendRequest(_) => StatusCode::BAD_GATEWAY,
            ProxyError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ProxyError::FormatConversion(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::Config(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::PrivacyViolation(_) => StatusCode::FORBIDDEN,
            ProxyError::RateLimited { .. } => StatusCode::TOO_MANY_REQUESTS,
        };

        let body = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": error_type(&self),
            }
        });

        let mut response = (status, axum::Json(body)).into_response();

        if let Some(secs) = retry_after {
            if let Ok(val) = secs.to_string().parse() {
                response.headers_mut().insert("retry-after", val);
            }
        }

        response
    }
}

fn error_type(err: &ProxyError) -> &'static str {
    match err {
        ProxyError::UnknownModel(_) => "not_found",
        ProxyError::BackendNotConfigured(_) => "configuration_error",
        ProxyError::BackendUnavailable(_) => "service_unavailable",
        ProxyError::BackendRequest(_) => "backend_error",
        ProxyError::BadRequest(_) => "invalid_request",
        ProxyError::FormatConversion(_) => "conversion_error",
        ProxyError::Config(_) => "configuration_error",
        ProxyError::PrivacyViolation(_) => "privacy_violation",
        ProxyError::RateLimited { .. } => "rate_limit_exceeded",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    /// Helper: convert ProxyError to axum Response and extract status + body
    fn error_to_parts(err: ProxyError) -> (StatusCode, serde_json::Value) {
        let response = err.into_response();
        let status = response.status();
        // We can't easily extract the body from an axum Response in sync context,
        // so we test status codes and error_type() separately
        (status, serde_json::Value::Null)
    }

    #[test]
    fn test_unknown_model_status() {
        let err = ProxyError::UnknownModel("gpt-5".into());
        assert_eq!(error_type(&err), "not_found");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn test_backend_not_configured_status() {
        let err = ProxyError::BackendNotConfigured("ollama".into());
        assert_eq!(error_type(&err), "configuration_error");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn test_backend_unavailable_status() {
        let err = ProxyError::BackendUnavailable("connection refused".into());
        assert_eq!(error_type(&err), "service_unavailable");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn test_bad_request_status() {
        let err = ProxyError::BadRequest("missing model field".into());
        assert_eq!(error_type(&err), "invalid_request");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn test_format_conversion_status() {
        let err = ProxyError::FormatConversion("invalid tool schema".into());
        assert_eq!(error_type(&err), "conversion_error");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_config_error_status() {
        let err = ProxyError::Config("missing backend".into());
        assert_eq!(error_type(&err), "configuration_error");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn test_privacy_violation_status() {
        let err = ProxyError::PrivacyViolation("claude-sonnet".into());
        assert_eq!(error_type(&err), "privacy_violation");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn test_rate_limited_status() {
        let err = ProxyError::RateLimited {
            retry_after_secs: 5.3,
        };
        assert_eq!(error_type(&err), "rate_limit_exceeded");
        let (status, _) = error_to_parts(err);
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    }

    #[test]
    fn test_error_display_messages() {
        assert_eq!(
            ProxyError::UnknownModel("gpt-5".into()).to_string(),
            "Unknown model: gpt-5"
        );
        assert_eq!(
            ProxyError::BackendUnavailable("timeout".into()).to_string(),
            "Backend unavailable: timeout"
        );
        assert_eq!(
            ProxyError::PrivacyViolation("claude".into()).to_string(),
            "Privacy violation: model claude requires remote backend but local_only requested"
        );
        assert_eq!(
            ProxyError::BadRequest("no model".into()).to_string(),
            "Invalid request: no model"
        );
    }
}
