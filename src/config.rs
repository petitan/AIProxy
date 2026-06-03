use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::error::ProxyError;

#[derive(Debug, Clone, Deserialize)]
pub struct ProxyConfig {
    pub proxy: ProxySettings,
    pub backends: HashMap<String, BackendConfig>,
    pub models: HashMap<String, ModelConfig>,
    #[serde(default)]
    pub gpu: Option<GpuConfig>,
    #[serde(default)]
    pub routing: RoutingConfig,
    #[serde(default)]
    pub rate_limits: HashMap<String, RateLimitConfig>,
    #[serde(default)]
    pub circuit_breaker: CircuitBreakerConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProxySettings {
    #[serde(default = "default_listen")]
    pub listen: String,
    // Note: `log_level` is intentionally NOT a field here. It must be read before this struct
    // is built (logging is set up before config validation), so main.rs reads it via a raw
    // TOML preview. Serde ignores the unknown `log_level` key, so configs keep working.
    /// Environment variable name containing the proxy API key.
    /// If set, all endpoints (except /health) require `Authorization: Bearer <key>`.
    /// If unset, auth is disabled.
    #[serde(default)]
    pub api_key_env: Option<String>,
    /// Max request body size in MB (default: 10)
    #[serde(default = "default_max_body_size_mb")]
    pub max_body_size_mb: u64,
    /// Global request timeout in seconds — safety net above per-backend timeouts (default: 600)
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Max concurrent connections (default: 1024)
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,
}

fn default_listen() -> String {
    "127.0.0.1:8800".to_string()
}

fn default_max_body_size_mb() -> u64 {
    10
}
fn default_request_timeout_secs() -> u64 {
    600
}
fn default_max_connections() -> usize {
    1024
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackendConfig {
    #[serde(rename = "type")]
    pub backend_type: BackendType,
    pub base_url: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default = "default_format")]
    pub format: ApiFormat,
    /// Request timeout in seconds (default: 300 for local, 60 for remote)
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Max retries on transient errors (default: 0 for local, 2 for remote)
    #[serde(default)]
    pub max_retries: Option<u32>,
    /// Max seconds to wait for a single streaming chunk (default: 60). Must be shorter than
    /// `timeout_secs`: it bounds the gap *between* chunks, so a stalled backend frees the slot
    /// quickly instead of holding it for the whole (300s) request timeout.
    #[serde(default)]
    pub stream_chunk_timeout_secs: Option<u64>,
}

impl BackendConfig {
    /// Effective timeout: explicit config > type-based default
    pub fn effective_timeout(&self) -> std::time::Duration {
        let secs = self.timeout_secs.unwrap_or(match self.backend_type {
            BackendType::Local => 300,
            BackendType::Remote => 60,
        });
        std::time::Duration::from_secs(secs)
    }

    /// Effective per-chunk streaming timeout: explicit config > 60s default.
    /// Type-independent: the inter-chunk gap should be short regardless of backend.
    pub fn effective_stream_chunk_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.stream_chunk_timeout_secs.unwrap_or(60))
    }

    /// Effective max retries: explicit config > type-based default
    pub fn effective_max_retries(&self) -> u32 {
        self.max_retries.unwrap_or(match self.backend_type {
            BackendType::Local => 0,   // Local backends: no retry (Ollama manages itself)
            BackendType::Remote => 2,  // Remote APIs: retry on transient errors
        })
    }
}

fn default_format() -> ApiFormat {
    ApiFormat::OpenAI
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    Local,
    Remote,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ApiFormat {
    OpenAI,
    Anthropic,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub backend: String,
    #[serde(default)]
    pub estimated_vram_mb: Option<u64>,
    #[serde(default)]
    pub default_priority: Option<String>,
    /// Fallback model key (e.g. "claude_haiku") used when local VRAM is insufficient
    /// and routing.default_fallback = "remote".
    #[serde(default)]
    pub fallback: Option<String>,
    /// Extra fields injected into the request body before forwarding.
    /// Client-sent values take priority (extra_body only fills missing keys).
    #[serde(default)]
    pub extra_body: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct GpuConfig {
    #[serde(default)]
    pub slots: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RoutingConfig {
    #[serde(default = "default_fallback")]
    pub default_fallback: String,
    #[serde(default = "default_privacy_header")]
    pub privacy_header: String,
    #[serde(default = "default_priority_header")]
    pub priority_header: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit (default: 5)
    #[serde(default = "default_cb_failure_threshold")]
    pub failure_threshold: u32,
    /// Seconds to wait before allowing a probe request (default: 30)
    #[serde(default = "default_cb_recovery_timeout")]
    pub recovery_timeout_secs: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: default_cb_failure_threshold(),
            recovery_timeout_secs: default_cb_recovery_timeout(),
        }
    }
}

fn default_cb_failure_threshold() -> u32 {
    5
}
fn default_cb_recovery_timeout() -> u64 {
    30
}

impl Default for RoutingConfig {
    fn default() -> Self {
        Self {
            default_fallback: default_fallback(),
            privacy_header: default_privacy_header(),
            priority_header: default_priority_header(),
        }
    }
}

fn default_fallback() -> String {
    "deny".to_string()
}
fn default_privacy_header() -> String {
    "X-Privacy".to_string()
}
fn default_priority_header() -> String {
    "X-Priority".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitConfig {
    #[serde(default)]
    pub requests_per_minute: Option<u32>,
    #[serde(default)]
    pub tokens_per_minute: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    pub daily_budget_usd: Option<f64>,
}

impl ModelConfig {
    /// Merge extra_body defaults into a request body.
    /// Client-sent values take priority — extra_body only fills missing keys.
    pub fn apply_extra_body(&self, body: &mut serde_json::Value) {
        if let Some(ref extra) = self.extra_body {
            if let Some(obj) = body.as_object_mut() {
                for (key, value) in extra {
                    if !obj.contains_key(key) {
                        tracing::debug!(model = %self.name, key = %key, value = %value, "Injecting extra_body default");
                        obj.insert(key.clone(), value.clone());
                    }
                }
            }
        }
    }
}

impl ProxyConfig {
    pub fn load(path: &Path) -> Result<Self, ProxyError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ProxyError::Config(format!("Failed to read config {}: {}", path.display(), e)))?;
        let config: ProxyConfig = toml::from_str(&content)
            .map_err(|e| ProxyError::Config(format!("Failed to parse config: {}", e)))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ProxyError> {
        // Model → backend reference check
        for (model_key, model) in &self.models {
            if !self.backends.contains_key(&model.backend) {
                return Err(ProxyError::Config(format!(
                    "Model '{}' references unknown backend '{}'",
                    model_key, model.backend
                )));
            }
            // Fallback model reference check
            if let Some(ref fallback_key) = model.fallback {
                if !self.models.contains_key(fallback_key) {
                    return Err(ProxyError::Config(format!(
                        "Model '{}' has fallback '{}' which doesn't exist in [models]",
                        model_key, fallback_key
                    )));
                }
            }
        }

        // Transitive fallback cycle detection (A→B→C→A)
        for start_key in self.models.keys() {
            let mut visited = std::collections::HashSet::new();
            visited.insert(start_key.as_str());
            let mut current = start_key.as_str();
            while let Some(model) = self.models.get(current) {
                match &model.fallback {
                    Some(fallback_key) => {
                        if !visited.insert(fallback_key.as_str()) {
                            return Err(ProxyError::Config(format!(
                                "Circular fallback chain detected: model '{}' eventually leads back to '{}'",
                                start_key, fallback_key
                            )));
                        }
                        current = fallback_key;
                    }
                    None => break,
                }
            }
        }

        // Backend URL validation
        for (name, backend) in &self.backends {
            if !backend.base_url.starts_with("http://") && !backend.base_url.starts_with("https://") {
                return Err(ProxyError::Config(format!(
                    "Backend '{}' has invalid base_url '{}' — must start with http:// or https://",
                    name, backend.base_url
                )));
            }
        }

        // Auth fail-closed: if api_key_env is configured, the env var MUST exist and be non-empty
        if let Some(ref env_var) = self.proxy.api_key_env {
            match std::env::var(env_var) {
                Ok(val) if !val.is_empty() => {} // OK
                Ok(_) => {
                    return Err(ProxyError::Config(format!(
                        "api_key_env '{}' is configured but the environment variable is empty — \
                         refusing to start without auth",
                        env_var
                    )));
                }
                Err(_) => {
                    return Err(ProxyError::Config(format!(
                        "api_key_env '{}' is configured but the environment variable is not set — \
                         refusing to start without auth",
                        env_var
                    )));
                }
            }
        }

        Ok(())
    }

    /// Find model config by the model name that clients send
    pub fn resolve_model(&self, model_name: &str) -> Option<(&str, &ModelConfig, &BackendConfig)> {
        for (key, model) in &self.models {
            if model.name == model_name {
                if let Some(backend) = self.backends.get(&model.backend) {
                    return Some((key, model, backend));
                }
            }
        }
        None
    }

    /// Find model config by its config key (e.g. "claude_haiku"), not by model name.
    /// Used for fallback resolution where the config key is stored, not the model name.
    pub fn resolve_model_by_key<'a>(&'a self, key: &'a str) -> Option<(&'a str, &'a ModelConfig, &'a BackendConfig)> {
        let model = self.models.get(key)?;
        let backend = self.backends.get(&model.backend)?;
        Some((key, model, backend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_toml() -> &'static str {
        r#"
[proxy]
listen = "127.0.0.1:9999"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[backends.anthropic]
type = "remote"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
timeout_secs = 30
max_retries = 3

[models.qwen]
name = "qwen:7b"
backend = "ollama"
estimated_vram_mb = 6400
default_priority = "high"

[models.claude]
name = "claude-sonnet"
backend = "anthropic"
"#
    }

    #[test]
    fn test_parse_config() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();
        assert_eq!(config.proxy.listen, "127.0.0.1:9999");
        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.models.len(), 2);
    }

    #[test]
    fn test_backend_types() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();
        assert_eq!(config.backends["ollama"].backend_type, BackendType::Local);
        assert_eq!(config.backends["anthropic"].backend_type, BackendType::Remote);
    }

    #[test]
    fn test_effective_timeout_defaults() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();
        // Ollama: no timeout_secs → default 300 (local)
        assert_eq!(config.backends["ollama"].effective_timeout().as_secs(), 300);
        // Anthropic: explicit timeout_secs = 30
        assert_eq!(config.backends["anthropic"].effective_timeout().as_secs(), 30);
    }

    #[test]
    fn test_effective_stream_chunk_timeout() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();
        // No stream_chunk_timeout_secs configured → 60s default (type-independent)
        assert_eq!(config.backends["ollama"].effective_stream_chunk_timeout().as_secs(), 60);
        assert_eq!(config.backends["anthropic"].effective_stream_chunk_timeout().as_secs(), 60);

        // Explicit value honored
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"
stream_chunk_timeout_secs = 15

[models.test]
name = "test-model"
backend = "ollama"
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.backends["ollama"].effective_stream_chunk_timeout().as_secs(), 15);
    }

    #[test]
    fn test_effective_max_retries() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();
        // Ollama: no max_retries → default 0 (local)
        assert_eq!(config.backends["ollama"].effective_max_retries(), 0);
        // Anthropic: explicit max_retries = 3
        assert_eq!(config.backends["anthropic"].effective_max_retries(), 3);
    }

    #[test]
    fn test_resolve_model() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();

        let (key, model, backend) = config.resolve_model("qwen:7b").unwrap();
        assert_eq!(key, "qwen");
        assert_eq!(model.name, "qwen:7b");
        assert_eq!(backend.backend_type, BackendType::Local);

        let (_, model, backend) = config.resolve_model("claude-sonnet").unwrap();
        assert_eq!(model.backend, "anthropic");
        assert_eq!(backend.backend_type, BackendType::Remote);

        assert!(config.resolve_model("nonexistent").is_none());
    }

    #[test]
    fn test_validate_unknown_backend() {
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[models.bad]
name = "bad-model"
backend = "nonexistent"
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_self_fallback_rejected() {
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[models.qwen]
name = "qwen:7b"
backend = "ollama"
fallback = "qwen"
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("Circular fallback chain detected"));
    }

    #[test]
    fn test_validate_invalid_backend_url() {
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"

[backends.bad]
type = "local"
base_url = "ftp://localhost:11434"

[models.test]
name = "test-model"
backend = "bad"
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("invalid base_url"));
    }

    #[test]
    fn test_validate_auth_fail_closed() {
        // Use an env var name that definitely doesn't exist
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"
api_key_env = "ABSOLUTELY_NONEXISTENT_VAR_12345"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[models.test]
name = "test-model"
backend = "ollama"
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("refusing to start"));
    }

    #[test]
    fn test_default_hardening_values() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();
        assert_eq!(config.proxy.max_body_size_mb, 10);
        assert_eq!(config.proxy.request_timeout_secs, 600);
        assert_eq!(config.proxy.max_connections, 1024);
    }

    #[test]
    fn test_custom_hardening_values() {
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"
max_body_size_mb = 50
request_timeout_secs = 120
max_connections = 256

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[models.test]
name = "test-model"
backend = "ollama"
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.proxy.max_body_size_mb, 50);
        assert_eq!(config.proxy.request_timeout_secs, 120);
        assert_eq!(config.proxy.max_connections, 256);
    }

    #[test]
    fn test_model_config_optional_fields() {
        let config: ProxyConfig = toml::from_str(sample_toml()).unwrap();
        // qwen has estimated_vram_mb and default_priority
        assert_eq!(config.models["qwen"].estimated_vram_mb, Some(6400));
        assert_eq!(config.models["qwen"].default_priority.as_deref(), Some("high"));
        // claude has neither
        assert_eq!(config.models["claude"].estimated_vram_mb, None);
        assert_eq!(config.models["claude"].default_priority, None);
    }

    #[test]
    fn test_extra_body_parsing() {
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[models.qwen]
name = "qwen:7b"
backend = "ollama"

[models.qwen.extra_body]
think = false
num_ctx = 16384
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        let extra = config.models["qwen"].extra_body.as_ref().unwrap();
        assert_eq!(extra["think"], serde_json::Value::Bool(false));
        assert_eq!(extra["num_ctx"], serde_json::json!(16384));
    }

    #[test]
    fn test_apply_extra_body_fills_missing() {
        let model = ModelConfig {
            name: "test".to_string(),
            backend: "ollama".to_string(),
            estimated_vram_mb: None,
            default_priority: None,
            fallback: None,
            extra_body: Some(HashMap::from([
                ("think".to_string(), serde_json::json!(false)),
                ("num_ctx".to_string(), serde_json::json!(8192)),
            ])),
        };
        let mut body = serde_json::json!({"model": "test", "messages": []});
        model.apply_extra_body(&mut body);
        assert_eq!(body["think"], serde_json::json!(false));
        assert_eq!(body["num_ctx"], serde_json::json!(8192));
    }

    #[test]
    fn test_apply_extra_body_client_wins() {
        let model = ModelConfig {
            name: "test".to_string(),
            backend: "ollama".to_string(),
            estimated_vram_mb: None,
            default_priority: None,
            fallback: None,
            extra_body: Some(HashMap::from([
                ("think".to_string(), serde_json::json!(false)),
            ])),
        };
        // Client explicitly sends think=true → should NOT be overwritten
        let mut body = serde_json::json!({"model": "test", "think": true});
        model.apply_extra_body(&mut body);
        assert_eq!(body["think"], serde_json::json!(true));
    }

    #[test]
    fn test_apply_extra_body_none() {
        let model = ModelConfig {
            name: "test".to_string(),
            backend: "ollama".to_string(),
            estimated_vram_mb: None,
            default_priority: None,
            fallback: None,
            extra_body: None,
        };
        let mut body = serde_json::json!({"model": "test"});
        model.apply_extra_body(&mut body);
        // No change
        assert_eq!(body, serde_json::json!({"model": "test"}));
    }

    #[test]
    fn test_gpu_config_parsing() {
        let toml_str = r#"
[proxy]
listen = "127.0.0.1:8800"

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"

[models.test]
name = "test-model"
backend = "ollama"

[gpu.slots.slot_0]
total_vram_mb = 24576
overhead_mb = 1024

[gpu.slots.permanent]
reranker = { estimated_vram_mb = 800 }
"#;
        let config: ProxyConfig = toml::from_str(toml_str).unwrap();
        let gpu = config.gpu.unwrap();
        assert!(gpu.slots.contains_key("slot_0"));
        assert!(gpu.slots.contains_key("permanent"));
    }
}
