use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde_json::{json, Value};

/// Per-model request metrics
struct ModelMetrics {
    total_requests: AtomicU64,
    total_errors: AtomicU64,
    total_prompt_tokens: AtomicU64,
    total_completion_tokens: AtomicU64,
    /// Latency tracking (non-streaming only — streaming latency is time-to-first-byte, less meaningful)
    latency: Mutex<LatencyTracker>,
}

impl ModelMetrics {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            latency: Mutex::new(LatencyTracker::new()),
        }
    }

    fn to_json(&self) -> Value {
        let latency = self.latency.lock();
        let latency_json = if latency.count > 0 {
            json!({
                "count": latency.count,
                "avg": latency.sum_ms / latency.count as f64,
                "min": latency.min_ms,
                "max": latency.max_ms,
                "p95_approx": latency.approx_p95(),
                // p95 is computed over the recent ring buffer only, NOT all `count`
                // requests — expose the sample size so it isn't misread as global.
                "p95_sample_size": latency.p95_sample_size(),
            })
        } else {
            json!({
                "count": 0,
                "avg": null,
                "min": null,
                "max": null,
                "p95_approx": null,
            })
        };
        json!({
            "total_requests": self.total_requests.load(Ordering::Relaxed),
            "total_errors": self.total_errors.load(Ordering::Relaxed),
            "total_prompt_tokens": self.total_prompt_tokens.load(Ordering::Relaxed),
            "total_completion_tokens": self.total_completion_tokens.load(Ordering::Relaxed),
            "latency_ms": latency_json,
        })
    }
}

/// Per-backend request metrics
struct BackendMetrics {
    total_requests: AtomicU64,
    total_errors_4xx: AtomicU64,
    total_errors_5xx: AtomicU64,
    total_errors_connection: AtomicU64,
    circuit_breaker_trips: AtomicU64,
}

impl BackendMetrics {
    fn new() -> Self {
        Self {
            total_requests: AtomicU64::new(0),
            total_errors_4xx: AtomicU64::new(0),
            total_errors_5xx: AtomicU64::new(0),
            total_errors_connection: AtomicU64::new(0),
            circuit_breaker_trips: AtomicU64::new(0),
        }
    }

    fn to_json(&self) -> Value {
        json!({
            "total_requests": self.total_requests.load(Ordering::Relaxed),
            "errors": {
                "client_4xx": self.total_errors_4xx.load(Ordering::Relaxed),
                "server_5xx": self.total_errors_5xx.load(Ordering::Relaxed),
                "connection": self.total_errors_connection.load(Ordering::Relaxed),
            },
            "circuit_breaker_trips": self.circuit_breaker_trips.load(Ordering::Relaxed),
        })
    }
}

/// Simple latency tracker — keeps running stats without storing all values.
/// Uses a fixed-size ring buffer of recent values for approximate percentile estimation.
struct LatencyTracker {
    count: u64,
    sum_ms: f64,
    min_ms: f64,
    max_ms: f64,
    /// Ring buffer of recent latencies for percentile approximation
    recent: Vec<f64>,
    recent_pos: usize,
}

const RECENT_BUFFER_SIZE: usize = 256;

impl LatencyTracker {
    fn new() -> Self {
        Self {
            count: 0,
            sum_ms: 0.0,
            min_ms: f64::MAX,
            max_ms: 0.0,
            recent: Vec::with_capacity(RECENT_BUFFER_SIZE),
            recent_pos: 0,
        }
    }

    fn record(&mut self, ms: f64) {
        self.count += 1;
        self.sum_ms += ms;
        if ms < self.min_ms {
            self.min_ms = ms;
        }
        if ms > self.max_ms {
            self.max_ms = ms;
        }

        // Ring buffer
        if self.recent.len() < RECENT_BUFFER_SIZE {
            self.recent.push(ms);
        } else {
            self.recent[self.recent_pos] = ms;
        }
        self.recent_pos = (self.recent_pos + 1) % RECENT_BUFFER_SIZE;
    }

    /// Approximate p95 from the ring buffer of recent values
    fn approx_p95(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let mut sorted = self.recent.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        // Nearest-rank on a 0-based array: index = round((n-1) * 0.95). `ceil(n*0.95)`
        // biased one position high (e.g. n=256 → 244 instead of ~243).
        let idx = (((sorted.len() - 1) as f64) * 0.95).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    /// Number of samples the p95 is computed over (the recent ring buffer size, capped
    /// at RECENT_BUFFER_SIZE). Exposed so callers don't read p95 as covering all requests.
    fn p95_sample_size(&self) -> usize {
        self.recent.len()
    }
}

/// Thread-safe metrics collector for the entire proxy.
/// Uses Arc<T> for entries so the outer Mutex is held only briefly (lookup + Arc::clone),
/// preventing nested lock contention between the HashMap lock and per-entry locks.
pub struct Metrics {
    models: Mutex<HashMap<String, Arc<ModelMetrics>>>,
    backends: Mutex<HashMap<String, Arc<BackendMetrics>>>,
    started_at: Instant,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
            backends: Mutex::new(HashMap::new()),
            started_at: Instant::now(),
        }
    }

    /// Get or create a model metrics entry (releases HashMap lock immediately)
    fn get_model(&self, model: &str) -> Arc<ModelMetrics> {
        let mut models = self.models.lock();
        Arc::clone(models.entry(model.to_string()).or_insert_with(|| Arc::new(ModelMetrics::new())))
    }

    /// Get or create a backend metrics entry (releases HashMap lock immediately)
    fn get_backend(&self, backend: &str) -> Arc<BackendMetrics> {
        let mut backends = self.backends.lock();
        Arc::clone(backends.entry(backend.to_string()).or_insert_with(|| Arc::new(BackendMetrics::new())))
    }

    /// Record a completed request for a model
    pub fn record_request(&self, model: &str, latency_ms: u64, is_error: bool, is_stream: bool) {
        let m = self.get_model(model);
        // HashMap lock released — latency lock is no longer nested
        m.total_requests.fetch_add(1, Ordering::Relaxed);
        if is_error {
            m.total_errors.fetch_add(1, Ordering::Relaxed);
        }
        if !is_stream && !is_error {
            m.latency.lock().record(latency_ms as f64);
        }
    }

    /// Record token usage for a model
    pub fn record_tokens(&self, model: &str, prompt_tokens: u64, completion_tokens: u64) {
        if prompt_tokens == 0 && completion_tokens == 0 {
            return;
        }
        let m = self.get_model(model);
        m.total_prompt_tokens.fetch_add(prompt_tokens, Ordering::Relaxed);
        m.total_completion_tokens.fetch_add(completion_tokens, Ordering::Relaxed);
    }

    /// Record a backend request outcome
    pub fn record_backend_request(&self, backend: &str, status: Option<u16>, is_connection_error: bool) {
        let b = self.get_backend(backend);
        b.total_requests.fetch_add(1, Ordering::Relaxed);
        if is_connection_error {
            b.total_errors_connection.fetch_add(1, Ordering::Relaxed);
        } else if let Some(code) = status {
            if (400..500).contains(&code) {
                b.total_errors_4xx.fetch_add(1, Ordering::Relaxed);
            } else if code >= 500 {
                b.total_errors_5xx.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Record a circuit breaker trip for a backend
    pub fn record_circuit_breaker_trip(&self, backend: &str) {
        let b = self.get_backend(backend);
        b.circuit_breaker_trips.fetch_add(1, Ordering::Relaxed);
    }

    /// Export all metrics as JSON
    pub fn to_json(&self) -> Value {
        // Snapshot Arc references under lock, then release lock before serialization
        let model_arcs: Vec<(String, Arc<ModelMetrics>)> = {
            let models = self.models.lock();
            models.iter().map(|(k, v)| (k.clone(), Arc::clone(v))).collect()
        };
        let model_metrics: HashMap<String, Value> = model_arcs
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json()))
            .collect();

        let backend_arcs: Vec<(String, Arc<BackendMetrics>)> = {
            let backends = self.backends.lock();
            backends.iter().map(|(k, v)| (k.clone(), Arc::clone(v))).collect()
        };
        let backend_metrics: HashMap<String, Value> = backend_arcs
            .iter()
            .map(|(k, v)| (k.clone(), v.to_json()))
            .collect();

        json!({
            "uptime_seconds": self.started_at.elapsed().as_secs(),
            "models": model_metrics,
            "backends": backend_metrics,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_request_basic() {
        let m = Metrics::new();
        m.record_request("qwen:7b", 150, false, false);
        m.record_request("qwen:7b", 200, false, false);
        m.record_request("qwen:7b", 0, true, false);

        let json = m.to_json();
        let model = &json["models"]["qwen:7b"];
        assert_eq!(model["total_requests"], 3);
        assert_eq!(model["total_errors"], 1);
        assert_eq!(model["latency_ms"]["count"], 2); // only non-error, non-stream
    }

    #[test]
    fn test_record_tokens() {
        let m = Metrics::new();
        m.record_tokens("claude", 100, 50);
        m.record_tokens("claude", 200, 150);

        let json = m.to_json();
        assert_eq!(json["models"]["claude"]["total_prompt_tokens"], 300);
        assert_eq!(json["models"]["claude"]["total_completion_tokens"], 200);
    }

    #[test]
    fn test_record_zero_tokens_ignored() {
        let m = Metrics::new();
        m.record_tokens("test", 0, 0);

        let json = m.to_json();
        assert!(json["models"]["test"].is_null());
    }

    #[test]
    fn test_record_backend_request() {
        let m = Metrics::new();
        m.record_backend_request("ollama", Some(200), false);
        m.record_backend_request("ollama", Some(429), false);
        m.record_backend_request("ollama", Some(502), false);
        m.record_backend_request("ollama", None, true);

        let json = m.to_json();
        let b = &json["backends"]["ollama"];
        assert_eq!(b["total_requests"], 4);
        assert_eq!(b["errors"]["client_4xx"], 1);
        assert_eq!(b["errors"]["server_5xx"], 1);
        assert_eq!(b["errors"]["connection"], 1);
    }

    #[test]
    fn test_circuit_breaker_trip_count() {
        let m = Metrics::new();
        m.record_circuit_breaker_trip("anthropic");
        m.record_circuit_breaker_trip("anthropic");

        let json = m.to_json();
        assert_eq!(json["backends"]["anthropic"]["circuit_breaker_trips"], 2);
    }

    #[test]
    fn test_latency_stats() {
        let m = Metrics::new();
        m.record_request("test", 100, false, false);
        m.record_request("test", 200, false, false);
        m.record_request("test", 300, false, false);

        let json = m.to_json();
        let lat = &json["models"]["test"]["latency_ms"];
        assert_eq!(lat["count"], 3);
        assert_eq!(lat["min"], 100.0);
        assert_eq!(lat["max"], 300.0);
        assert_eq!(lat["avg"], 200.0);
    }

    #[test]
    fn test_streaming_excluded_from_latency() {
        let m = Metrics::new();
        m.record_request("test", 50, false, true); // streaming — excluded
        m.record_request("test", 100, false, false);

        let json = m.to_json();
        let lat = &json["models"]["test"]["latency_ms"];
        assert_eq!(lat["count"], 1); // only the non-streaming one
    }

    #[test]
    fn test_empty_metrics() {
        let m = Metrics::new();
        let json = m.to_json();
        assert!(json["models"].as_object().unwrap().is_empty());
        assert!(json["backends"].as_object().unwrap().is_empty());
        assert!(json["uptime_seconds"].as_u64().is_some());
    }

    #[test]
    fn test_multiple_models_independent() {
        let m = Metrics::new();
        m.record_request("model_a", 100, false, false);
        m.record_request("model_b", 200, true, false);

        let json = m.to_json();
        assert_eq!(json["models"]["model_a"]["total_requests"], 1);
        assert_eq!(json["models"]["model_a"]["total_errors"], 0);
        assert_eq!(json["models"]["model_b"]["total_requests"], 1);
        assert_eq!(json["models"]["model_b"]["total_errors"], 1);
    }
}
