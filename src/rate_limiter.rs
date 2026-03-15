use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use crate::config::RateLimitConfig;

/// Token bucket rate limiter — per-model buckets with configurable capacity and refill rate.
/// Supports both requests-per-minute (RPM) and tokens-per-minute (TPM) limits.
pub struct RateLimiter {
    /// RPM buckets: pre-request check (1 token per request)
    rpm_buckets: HashMap<String, Arc<Mutex<TokenBucket>>>,
    default_rpm: Option<Arc<Mutex<TokenBucket>>>,
    /// TPM buckets: pre-request admission check + post-request deduction
    tpm_buckets: HashMap<String, Arc<Mutex<TokenBucket>>>,
    default_tpm: Option<Arc<Mutex<TokenBucket>>>,
}

struct TokenBucket {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: u32) -> Self {
        let cap = capacity as f64;
        Self {
            tokens: cap, // start full
            capacity: cap,
            refill_rate: cap / 60.0,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume `amount` tokens. Returns Ok(()) or Err(retry_after_secs).
    fn try_acquire(&mut self, amount: f64) -> Result<(), f64> {
        self.refill();
        if self.tokens >= amount {
            self.tokens -= amount;
            Ok(())
        } else {
            let deficit = amount - self.tokens;
            let retry_after = deficit / self.refill_rate;
            Err(retry_after)
        }
    }

    /// Check if there's at least 1 token available (for admission check).
    /// Does NOT consume — just peeks after refill.
    fn has_capacity(&mut self) -> Result<(), f64> {
        self.refill();
        if self.tokens >= 1.0 {
            Ok(())
        } else {
            let deficit = 1.0 - self.tokens;
            Err(deficit / self.refill_rate)
        }
    }

    /// Consume tokens without checking (post-request deduction).
    /// Can go negative — subsequent requests will be denied until refilled.
    /// Returns the remaining token count after consumption.
    fn consume(&mut self, amount: f64) -> f64 {
        self.refill();
        self.tokens -= amount;
        self.tokens
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }
}

/// Result of a rate limit check
pub enum RateLimitResult {
    /// Request is allowed
    Allowed,
    /// Request is denied — retry after N seconds
    Denied { retry_after_secs: f64 },
}

impl RateLimiter {
    /// Build rate limiter from config. Creates RPM and TPM buckets for models that have them.
    pub fn new(rate_limits: &HashMap<String, RateLimitConfig>) -> Self {
        let mut rpm_buckets = HashMap::new();
        let mut default_rpm = None;
        let mut tpm_buckets = HashMap::new();
        let mut default_tpm = None;

        for (key, config) in rate_limits {
            // RPM bucket
            if let Some(rpm) = config.requests_per_minute {
                if rpm > 0 {
                    let bucket = Arc::new(Mutex::new(TokenBucket::new(rpm)));
                    if key == "default" {
                        default_rpm = Some(bucket);
                    } else {
                        rpm_buckets.insert(key.clone(), bucket);
                    }
                }
            }

            // TPM bucket
            if let Some(tpm) = config.tokens_per_minute {
                if tpm > 0 {
                    let bucket = Arc::new(Mutex::new(TokenBucket::new(tpm)));
                    if key == "default" {
                        default_tpm = Some(bucket);
                    } else {
                        tpm_buckets.insert(key.clone(), bucket);
                    }
                }
            }
        }

        let rpm_count = rpm_buckets.len() + if default_rpm.is_some() { 1 } else { 0 };
        let tpm_count = tpm_buckets.len() + if default_tpm.is_some() { 1 } else { 0 };

        if rpm_count > 0 || tpm_count > 0 {
            tracing::info!(
                rpm_buckets = rpm_count,
                tpm_buckets = tpm_count,
                "Rate limiter initialized"
            );
        }

        Self {
            rpm_buckets,
            default_rpm,
            tpm_buckets,
            default_tpm,
        }
    }

    /// Pre-request check: verify both RPM and TPM limits.
    /// RPM consumes 1 token per request — but ONLY if TPM also passes (atomic check).
    /// TPM only checks admission (tokens > 0), actual deduction happens post-request.
    pub fn check(&self, model_key: &str) -> RateLimitResult {
        let rpm_bucket = self
            .rpm_buckets
            .get(model_key)
            .or(self.default_rpm.as_ref());

        let tpm_bucket = self
            .tpm_buckets
            .get(model_key)
            .or(self.default_tpm.as_ref());

        // Phase 1: Check TPM admission first (non-consuming check)
        if let Some(bucket) = tpm_bucket {
            let mut bucket = bucket.lock();
            if let Err(retry_after_secs) = bucket.has_capacity() {
                return RateLimitResult::Denied { retry_after_secs };
            }
        }

        // Phase 2: Check + consume RPM (only if TPM passed)
        if let Some(bucket) = rpm_bucket {
            let mut bucket = bucket.lock();
            if let Err(retry_after_secs) = bucket.try_acquire(1.0) {
                return RateLimitResult::Denied { retry_after_secs };
            }
        }

        RateLimitResult::Allowed
    }

    /// Post-request: report actual token usage for TPM accounting.
    /// Called after a successful response with known token count.
    pub fn report_tokens(&self, model_key: &str, tokens: u64) {
        if tokens == 0 {
            return;
        }

        let tpm_bucket = self
            .tpm_buckets
            .get(model_key)
            .or(self.default_tpm.as_ref());

        if let Some(bucket) = tpm_bucket {
            let mut bucket = bucket.lock();
            let remaining = bucket.consume(tokens as f64);
            tracing::debug!(
                model = %model_key,
                tokens = tokens,
                remaining = remaining as i64,
                "Token usage reported"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Circuit Breaker — per-backend failure tracking
// ═══════════════════════════════════════════════════════════════════

/// Circuit breaker states:
/// - Closed: normal operation, requests pass through
/// - Open: backend is considered down, requests are rejected immediately
/// - HalfOpen: one probe request is allowed to test if backend recovered
#[derive(Debug, Clone, Copy, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

struct BackendCircuit {
    state: CircuitState,
    consecutive_failures: u32,
    failure_threshold: u32,
    last_failure: Option<Instant>,
    recovery_timeout: std::time::Duration,
}

impl BackendCircuit {
    fn new(failure_threshold: u32, recovery_timeout_secs: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            failure_threshold,
            last_failure: None,
            recovery_timeout: std::time::Duration::from_secs(recovery_timeout_secs),
        }
    }

    /// Check if a request should be allowed through
    fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if recovery timeout has elapsed → transition to HalfOpen
                if let Some(last) = self.last_failure {
                    if last.elapsed() >= self.recovery_timeout {
                        self.state = CircuitState::HalfOpen;
                        tracing::info!("Circuit breaker → HalfOpen (probe allowed)");
                        true
                    } else {
                        false
                    }
                } else {
                    // No last_failure recorded (shouldn't happen), allow
                    true
                }
            }
            CircuitState::HalfOpen => {
                // Only one probe request allowed; subsequent requests are rejected
                // The probe will resolve via record_success or record_failure
                false
            }
        }
    }

    /// Record a successful response → close the circuit
    fn record_success(&mut self) {
        if self.state != CircuitState::Closed {
            tracing::info!(
                previous_state = ?self.state,
                "Circuit breaker → Closed (backend recovered)"
            );
        }
        self.state = CircuitState::Closed;
        self.consecutive_failures = 0;
    }

    /// Record a failure → potentially open the circuit
    fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        self.last_failure = Some(Instant::now());

        match self.state {
            CircuitState::HalfOpen => {
                // Probe failed → back to Open
                self.state = CircuitState::Open;
                tracing::warn!("Circuit breaker → Open (probe failed)");
            }
            CircuitState::Closed => {
                if self.consecutive_failures >= self.failure_threshold {
                    self.state = CircuitState::Open;
                    tracing::warn!(
                        failures = self.consecutive_failures,
                        threshold = self.failure_threshold,
                        "Circuit breaker → Open (threshold reached)"
                    );
                }
            }
            CircuitState::Open => {
                // Already open, no state change
            }
        }
    }
}

/// Thread-safe circuit breaker for all backends
pub struct CircuitBreaker {
    circuits: Mutex<HashMap<String, BackendCircuit>>,
    failure_threshold: u32,
    recovery_timeout_secs: u64,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, recovery_timeout_secs: u64) -> Self {
        tracing::info!(
            failure_threshold = failure_threshold,
            recovery_timeout_secs = recovery_timeout_secs,
            "Circuit breaker initialized"
        );
        Self {
            circuits: Mutex::new(HashMap::new()),
            failure_threshold,
            recovery_timeout_secs,
        }
    }

    /// Check if a request to this backend should be allowed.
    /// Returns true if allowed, false if circuit is open.
    pub fn allow_request(&self, backend_name: &str) -> bool {
        let mut circuits = self.circuits.lock();
        let circuit = circuits
            .entry(backend_name.to_string())
            .or_insert_with(|| BackendCircuit::new(self.failure_threshold, self.recovery_timeout_secs));
        circuit.allow_request()
    }

    /// Record a successful response for a backend
    pub fn record_success(&self, backend_name: &str) {
        let mut circuits = self.circuits.lock();
        if let Some(circuit) = circuits.get_mut(backend_name) {
            circuit.record_success();
        }
    }

    /// Record a failure for a backend
    pub fn record_failure(&self, backend_name: &str) {
        let mut circuits = self.circuits.lock();
        let circuit = circuits
            .entry(backend_name.to_string())
            .or_insert_with(|| BackendCircuit::new(self.failure_threshold, self.recovery_timeout_secs));
        circuit.record_failure();
    }

    /// Get current state of a backend's circuit (for /status endpoint)
    pub fn state(&self, backend_name: &str) -> &'static str {
        let circuits = self.circuits.lock();
        match circuits.get(backend_name) {
            Some(c) => match c.state {
                CircuitState::Closed => "closed",
                CircuitState::Open => "open",
                CircuitState::HalfOpen => "half_open",
            },
            None => "closed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── RPM tests ─────────────────────────────────────────────────

    #[test]
    fn test_no_limits_configured() {
        let limiter = RateLimiter::new(&HashMap::new());
        assert!(matches!(limiter.check("anything"), RateLimitResult::Allowed));
    }

    #[test]
    fn test_bucket_allows_within_capacity() {
        let mut limits = HashMap::new();
        limits.insert(
            "claude".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(10),
                tokens_per_minute: None,
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        for _ in 0..10 {
            assert!(matches!(limiter.check("claude"), RateLimitResult::Allowed));
        }
    }

    #[test]
    fn test_bucket_denies_over_capacity() {
        let mut limits = HashMap::new();
        limits.insert(
            "claude".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(3),
                tokens_per_minute: None,
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        for _ in 0..3 {
            assert!(matches!(limiter.check("claude"), RateLimitResult::Allowed));
        }

        match limiter.check("claude") {
            RateLimitResult::Denied { retry_after_secs } => {
                assert!(retry_after_secs > 0.0);
                assert!(retry_after_secs <= 20.0);
            }
            _ => panic!("Expected Denied"),
        }
    }

    #[test]
    fn test_default_fallback() {
        let mut limits = HashMap::new();
        limits.insert(
            "default".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(5),
                tokens_per_minute: None,
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        for _ in 0..5 {
            assert!(matches!(
                limiter.check("unknown_model"),
                RateLimitResult::Allowed
            ));
        }

        assert!(matches!(
            limiter.check("unknown_model"),
            RateLimitResult::Denied { .. }
        ));
    }

    #[test]
    fn test_model_specific_overrides_default() {
        let mut limits = HashMap::new();
        limits.insert(
            "default".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(2),
                tokens_per_minute: None,
                daily_budget_usd: None,
            },
        );
        limits.insert(
            "claude".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(100),
                tokens_per_minute: None,
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        for _ in 0..50 {
            assert!(matches!(limiter.check("claude"), RateLimitResult::Allowed));
        }

        assert!(matches!(
            limiter.check("other_model"),
            RateLimitResult::Allowed
        ));
        assert!(matches!(
            limiter.check("other_model"),
            RateLimitResult::Allowed
        ));
        assert!(matches!(
            limiter.check("other_model"),
            RateLimitResult::Denied { .. }
        ));
    }

    #[test]
    fn test_zero_rpm_ignored() {
        let mut limits = HashMap::new();
        limits.insert(
            "claude".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(0),
                tokens_per_minute: None,
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);
        assert!(matches!(limiter.check("claude"), RateLimitResult::Allowed));
    }

    #[test]
    fn test_retry_after_is_reasonable() {
        let mut limits = HashMap::new();
        limits.insert(
            "test".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(60),
                tokens_per_minute: None,
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        for _ in 0..60 {
            limiter.check("test");
        }

        match limiter.check("test") {
            RateLimitResult::Denied { retry_after_secs } => {
                assert!(retry_after_secs > 0.0);
                assert!(retry_after_secs <= 2.0);
            }
            _ => panic!("Expected Denied"),
        }
    }

    #[test]
    fn test_no_rpm_configured_allows_all() {
        let mut limits = HashMap::new();
        limits.insert(
            "claude".to_string(),
            RateLimitConfig {
                requests_per_minute: None,
                tokens_per_minute: None,
                daily_budget_usd: Some(10.0),
            },
        );
        let limiter = RateLimiter::new(&limits);
        for _ in 0..100 {
            assert!(matches!(limiter.check("claude"), RateLimitResult::Allowed));
        }
    }

    // ─── TPM tests ─────────────────────────────────────────────────

    #[test]
    fn test_tpm_report_tokens_deducts() {
        let mut limits = HashMap::new();
        limits.insert(
            "claude".to_string(),
            RateLimitConfig {
                requests_per_minute: None,
                tokens_per_minute: Some(1000),
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        // Should be allowed initially
        assert!(matches!(limiter.check("claude"), RateLimitResult::Allowed));

        // Consume most of the budget
        limiter.report_tokens("claude", 999);
        // Still allowed (1 token left)
        assert!(matches!(limiter.check("claude"), RateLimitResult::Allowed));

        // Consume more → goes negative
        limiter.report_tokens("claude", 100);
        // Now should be denied
        assert!(matches!(
            limiter.check("claude"),
            RateLimitResult::Denied { .. }
        ));
    }

    #[test]
    fn test_tpm_default_fallback() {
        let mut limits = HashMap::new();
        limits.insert(
            "default".to_string(),
            RateLimitConfig {
                requests_per_minute: None,
                tokens_per_minute: Some(500),
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        // Unknown model uses default TPM bucket
        limiter.report_tokens("unknown", 600);
        assert!(matches!(
            limiter.check("unknown"),
            RateLimitResult::Denied { .. }
        ));
    }

    #[test]
    fn test_tpm_zero_report_ignored() {
        let mut limits = HashMap::new();
        limits.insert(
            "test".to_string(),
            RateLimitConfig {
                requests_per_minute: None,
                tokens_per_minute: Some(100),
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        // Reporting 0 tokens should be a no-op
        limiter.report_tokens("test", 0);
        assert!(matches!(limiter.check("test"), RateLimitResult::Allowed));
    }

    #[test]
    fn test_rpm_and_tpm_combined() {
        let mut limits = HashMap::new();
        limits.insert(
            "test".to_string(),
            RateLimitConfig {
                requests_per_minute: Some(100),
                tokens_per_minute: Some(500),
                daily_budget_usd: None,
            },
        );
        let limiter = RateLimiter::new(&limits);

        // RPM has plenty of room, but TPM gets exhausted
        assert!(matches!(limiter.check("test"), RateLimitResult::Allowed));
        limiter.report_tokens("test", 600);

        // Should be denied by TPM even though RPM has capacity
        assert!(matches!(
            limiter.check("test"),
            RateLimitResult::Denied { .. }
        ));
    }

    // ─── Circuit breaker tests ─────────────────────────────────────

    #[test]
    fn test_circuit_breaker_closed_by_default() {
        let cb = CircuitBreaker::new(3, 30);
        assert!(cb.allow_request("backend1"));
        assert_eq!(cb.state("backend1"), "closed");
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new(3, 30);

        cb.record_failure("backend1");
        assert!(cb.allow_request("backend1")); // 1 failure, still closed
        cb.record_failure("backend1");
        assert!(cb.allow_request("backend1")); // 2 failures, still closed
        cb.record_failure("backend1");
        // 3 failures = threshold → open
        assert!(!cb.allow_request("backend1"));
        assert_eq!(cb.state("backend1"), "open");
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let cb = CircuitBreaker::new(3, 30);

        cb.record_failure("backend1");
        cb.record_failure("backend1");
        // 2 failures, then success resets
        cb.record_success("backend1");
        assert_eq!(cb.state("backend1"), "closed");
        assert!(cb.allow_request("backend1"));

        // Need 3 consecutive failures again
        cb.record_failure("backend1");
        cb.record_failure("backend1");
        assert!(cb.allow_request("backend1")); // still closed (only 2)
    }

    #[test]
    fn test_circuit_breaker_half_open_recovery() {
        let cb = CircuitBreaker::new(2, 0); // 0 second recovery for testing

        cb.record_failure("b1");
        cb.record_failure("b1");
        assert_eq!(cb.state("b1"), "open");

        // With 0s recovery timeout, should immediately transition to half-open
        assert!(cb.allow_request("b1")); // transitions to HalfOpen, allows probe
        assert_eq!(cb.state("b1"), "half_open");

        // Second request while half-open should be denied
        assert!(!cb.allow_request("b1"));

        // Probe success → closed
        cb.record_success("b1");
        assert_eq!(cb.state("b1"), "closed");
        assert!(cb.allow_request("b1"));
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let cb = CircuitBreaker::new(2, 0);

        cb.record_failure("b1");
        cb.record_failure("b1");
        assert!(cb.allow_request("b1")); // → HalfOpen

        // Probe fails → back to Open
        cb.record_failure("b1");
        assert_eq!(cb.state("b1"), "open");
    }

    #[test]
    fn test_circuit_breaker_independent_backends() {
        let cb = CircuitBreaker::new(2, 30);

        cb.record_failure("backend_a");
        cb.record_failure("backend_a");
        assert!(!cb.allow_request("backend_a")); // open

        // backend_b should be unaffected
        assert!(cb.allow_request("backend_b"));
        assert_eq!(cb.state("backend_b"), "closed");
    }
}
