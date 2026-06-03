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
    /// `amount` may be negative (refund of an over-reservation).
    /// Returns the remaining token count after consumption.
    ///
    /// Clamped to `[-capacity, capacity]`: the lower bound caps the worst-case
    /// lockout after a bogus large token report at ~60s of refill (instead of
    /// hours); the upper bound prevents a refund from pushing the bucket above
    /// its capacity.
    fn consume(&mut self, amount: f64) -> f64 {
        self.refill();
        self.tokens = (self.tokens - amount).clamp(-self.capacity, self.capacity);
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

    /// Pre-request check: verify RPM, then admit + **pre-reserve** the estimated TPM.
    ///
    /// `estimated_tokens` (the request's max_tokens, or a default) is pre-deducted from
    /// the TPM bucket at admission so concurrent requests can't all slip through while the
    /// budget is ~0 (the old non-consuming admission check allowed exactly that burst
    /// bypass). The reservation is corrected to the real usage post-request via
    /// `reconcile_tokens`. Admission is gated on `has_capacity` (>= 1 token) rather than
    /// `try_acquire(estimated)`, so a large estimate can't permanently deny a model whose
    /// budget is smaller than the estimate.
    ///
    /// Locks are taken one at a time (never RPM and TPM simultaneously) — no deadlock.
    pub fn check(&self, model_key: &str, estimated_tokens: u64) -> RateLimitResult {
        let rpm_bucket = self
            .rpm_buckets
            .get(model_key)
            .or(self.default_rpm.as_ref());

        let tpm_bucket = self
            .tpm_buckets
            .get(model_key)
            .or(self.default_tpm.as_ref());

        // Phase 1: check + consume RPM (1 token).
        if let Some(bucket) = rpm_bucket {
            let mut b = bucket.lock();
            if let Err(retry_after_secs) = b.try_acquire(1.0) {
                return RateLimitResult::Denied { retry_after_secs };
            }
        }

        // Phase 2: TPM admission gate, then pre-reserve the estimate.
        if let Some(bucket) = tpm_bucket {
            let mut b = bucket.lock();
            if let Err(retry_after_secs) = b.has_capacity() {
                drop(b);
                // Refund the RPM token we already took for this now-denied request.
                if let Some(rb) = rpm_bucket {
                    rb.lock().consume(-1.0);
                }
                return RateLimitResult::Denied { retry_after_secs };
            }
            b.consume(estimated_tokens as f64);
        }

        RateLimitResult::Allowed
    }

    /// Reconcile a pre-reserved TPM estimate against the real usage. `delta = actual -
    /// reserved` is applied: a positive delta deducts the overflow, a negative delta
    /// refunds the unused reservation. Call exactly once per admitted request (with
    /// `actual = 0` to fully refund a request that failed before producing usage).
    pub fn reconcile_tokens(&self, model_key: &str, reserved: u64, actual: u64) {
        let delta = actual as i64 - reserved as i64;
        if delta == 0 {
            return;
        }
        let tpm_bucket = self
            .tpm_buckets
            .get(model_key)
            .or(self.default_tpm.as_ref());
        if let Some(bucket) = tpm_bucket {
            let remaining = bucket.lock().consume(delta as f64);
            tracing::debug!(
                model = %model_key,
                reserved = reserved,
                actual = actual,
                remaining = remaining as i64,
                "TPM reservation reconciled"
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
    /// Monotonic counter incremented each time the circuit enters HalfOpen, identifying
    /// the current probe. `record_*` only transitions state when the outcome carries the
    /// matching probe token — so a request that started BEFORE the circuit opened can't
    /// close the circuit by completing during the half-open window.
    probe_generation: u64,
    active_probe: Option<u64>,
    /// When the current probe was issued. If a probe is never resolved (e.g. a streaming
    /// probe whose client disconnects → the generator is dropped before recording an
    /// outcome), HalfOpen would otherwise be stuck forever. After `recovery_timeout` the
    /// probe is considered abandoned and a fresh one is issued.
    probe_started: Option<Instant>,
}

impl BackendCircuit {
    fn new(failure_threshold: u32, recovery_timeout_secs: u64) -> Self {
        Self {
            state: CircuitState::Closed,
            consecutive_failures: 0,
            failure_threshold,
            last_failure: None,
            recovery_timeout: std::time::Duration::from_secs(recovery_timeout_secs),
            probe_generation: 0,
            active_probe: None,
            probe_started: None,
        }
    }

    /// Issue a fresh probe (entering or re-entering HalfOpen) and return its token.
    fn issue_probe(&mut self) -> (bool, Option<u64>) {
        self.state = CircuitState::HalfOpen;
        self.probe_generation += 1;
        self.active_probe = Some(self.probe_generation);
        self.probe_started = Some(Instant::now());
        tracing::info!("Circuit breaker → HalfOpen (probe allowed)");
        (true, self.active_probe)
    }

    /// Check if a request should be allowed. Returns `(allowed, probe)` — `probe` is
    /// `Some(token)` only when this request is the half-open probe; the caller must pass
    /// that token back to `record_success`/`record_failure` so only the probe's own
    /// outcome moves the circuit.
    fn allow_request(&mut self) -> (bool, Option<u64>) {
        match self.state {
            CircuitState::Closed => (true, None),
            CircuitState::Open => {
                // Check if recovery timeout has elapsed → transition to HalfOpen
                if let Some(last) = self.last_failure {
                    if last.elapsed() >= self.recovery_timeout {
                        self.issue_probe()
                    } else {
                        (false, None)
                    }
                } else {
                    // No last_failure recorded (shouldn't happen), allow
                    (true, None)
                }
            }
            CircuitState::HalfOpen => {
                // Normally only one probe is in flight. But if the outstanding probe was
                // never resolved (its request's outcome never recorded — e.g. a streaming
                // probe whose client disconnected, dropping the generator), reissue a probe
                // after recovery_timeout instead of staying stuck HalfOpen forever.
                match self.probe_started {
                    Some(started) if started.elapsed() >= self.recovery_timeout => {
                        tracing::warn!("Circuit breaker: half-open probe abandoned — reissuing");
                        self.issue_probe()
                    }
                    _ => (false, None),
                }
            }
        }
    }

    /// Record a successful response. `probe` is the token from `allow_request` (None for a
    /// non-probe request). In HalfOpen only the probe's own success closes the circuit.
    fn record_success(&mut self, probe: Option<u64>) {
        match self.state {
            CircuitState::Closed => {
                self.consecutive_failures = 0;
            }
            CircuitState::HalfOpen => {
                if probe.is_some() && probe == self.active_probe {
                    tracing::info!("Circuit breaker → Closed (probe succeeded)");
                    self.state = CircuitState::Closed;
                    self.consecutive_failures = 0;
                    self.active_probe = None;
                    self.probe_started = None;
                }
                // else: a request that started before the circuit opened just succeeded —
                // ignore it; only the probe decides recovery.
            }
            CircuitState::Open => {
                // Stray success while open (raced with opening) — ignore.
            }
        }
    }

    /// Record a failure. `probe` is the token from `allow_request`. In HalfOpen only the
    /// probe's own failure reopens the circuit; a stray old request's failure is ignored.
    fn record_failure(&mut self, probe: Option<u64>) {
        self.last_failure = Some(Instant::now());

        match self.state {
            CircuitState::HalfOpen => {
                if probe.is_some() && probe == self.active_probe {
                    self.state = CircuitState::Open;
                    self.active_probe = None;
                    self.probe_started = None;
                    tracing::warn!("Circuit breaker → Open (probe failed)");
                }
            }
            CircuitState::Closed => {
                self.consecutive_failures += 1;
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

/// Thread-safe circuit breaker for all backends. Circuits are pre-populated per
/// configured backend at construction; the per-backend threshold/timeout live in each
/// `BackendCircuit`, so this struct holds no global copies.
pub struct CircuitBreaker {
    circuits: Mutex<HashMap<String, BackendCircuit>>,
}

impl CircuitBreaker {
    /// Create a circuit breaker, pre-populating one circuit per known backend name.
    /// Pre-population (rather than lazily inserting on first use) bounds the `circuits`
    /// map to the configured backends — a stray/unknown backend name can no longer grow
    /// it without bound.
    pub fn new<'a>(
        failure_threshold: u32,
        recovery_timeout_secs: u64,
        backend_names: impl Iterator<Item = &'a str>,
    ) -> Self {
        let circuits: HashMap<String, BackendCircuit> = backend_names
            .map(|name| {
                (
                    name.to_string(),
                    BackendCircuit::new(failure_threshold, recovery_timeout_secs),
                )
            })
            .collect();
        tracing::info!(
            failure_threshold = failure_threshold,
            recovery_timeout_secs = recovery_timeout_secs,
            backends = circuits.len(),
            "Circuit breaker initialized"
        );
        Self {
            circuits: Mutex::new(circuits),
        }
    }

    /// Check if a request to this backend should be allowed. Returns `(allowed, probe)`:
    /// `probe` is `Some(token)` when this request is the half-open probe and must be passed
    /// back to `record_success`/`record_failure`. An unknown backend name (not pre-populated
    /// from config) is allowed fail-safe and logged — it must not silently create a new
    /// circuit (that's what let the map grow unbounded).
    pub fn allow_request(&self, backend_name: &str) -> (bool, Option<u64>) {
        let mut circuits = self.circuits.lock();
        match circuits.get_mut(backend_name) {
            Some(circuit) => circuit.allow_request(),
            None => {
                tracing::warn!(backend = %backend_name, "Circuit breaker: unknown backend — allowing (fail-safe)");
                (true, None)
            }
        }
    }

    /// Record a successful response for a backend. `probe` is the token from `allow_request`
    /// (None for a non-probe request).
    pub fn record_success(&self, backend_name: &str, probe: Option<u64>) {
        let mut circuits = self.circuits.lock();
        if let Some(circuit) = circuits.get_mut(backend_name) {
            circuit.record_success(probe);
        }
    }

    /// Record a failure for a backend. `probe` is the token from `allow_request`.
    pub fn record_failure(&self, backend_name: &str, probe: Option<u64>) {
        let mut circuits = self.circuits.lock();
        match circuits.get_mut(backend_name) {
            Some(circuit) => circuit.record_failure(probe),
            None => {
                tracing::warn!(backend = %backend_name, "Circuit breaker: failure for unknown backend — ignored");
            }
        }
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
        assert!(matches!(limiter.check("anything", 0), RateLimitResult::Allowed));
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
            assert!(matches!(limiter.check("claude", 0), RateLimitResult::Allowed));
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
            assert!(matches!(limiter.check("claude", 0), RateLimitResult::Allowed));
        }

        match limiter.check("claude", 0) {
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
                limiter.check("unknown_model", 0),
                RateLimitResult::Allowed
            ));
        }

        assert!(matches!(
            limiter.check("unknown_model", 0),
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
            assert!(matches!(limiter.check("claude", 0), RateLimitResult::Allowed));
        }

        assert!(matches!(
            limiter.check("other_model", 0),
            RateLimitResult::Allowed
        ));
        assert!(matches!(
            limiter.check("other_model", 0),
            RateLimitResult::Allowed
        ));
        assert!(matches!(
            limiter.check("other_model", 0),
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
        assert!(matches!(limiter.check("claude", 0), RateLimitResult::Allowed));
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
            limiter.check("test", 0);
        }

        match limiter.check("test", 0) {
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
            assert!(matches!(limiter.check("claude", 0), RateLimitResult::Allowed));
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
        assert!(matches!(limiter.check("claude", 0), RateLimitResult::Allowed));

        // Consume most of the budget
        limiter.reconcile_tokens("claude", 0, 999);
        // Still allowed (1 token left)
        assert!(matches!(limiter.check("claude", 0), RateLimitResult::Allowed));

        // Consume more → goes negative
        limiter.reconcile_tokens("claude", 0, 100);
        // Now should be denied
        assert!(matches!(
            limiter.check("claude", 0),
            RateLimitResult::Denied { .. }
        ));
    }

    #[test]
    fn test_consume_clamps_to_capacity_bounds() {
        let mut b = TokenBucket::new(1000);
        // A bogus huge report must not drive the bucket arbitrarily negative — it's
        // clamped at -capacity so recovery takes at most ~60s, not hours.
        b.consume(1_000_000.0);
        assert_eq!(b.tokens, -1000.0);
        // A refund (negative amount) is clamped to +capacity, not above.
        b.consume(-1_000_000.0);
        assert_eq!(b.tokens, 1000.0);
    }

    fn tpm_limiter(tpm: u32) -> RateLimiter {
        let mut limits = HashMap::new();
        limits.insert(
            "claude".to_string(),
            RateLimitConfig {
                requests_per_minute: None,
                tokens_per_minute: Some(tpm),
                daily_budget_usd: None,
            },
        );
        RateLimiter::new(&limits)
    }

    #[test]
    fn test_tpm_pre_reservation_blocks_burst() {
        let limiter = tpm_limiter(1000);
        // Each admitted request pre-reserves 800 — so the budget depletes immediately and
        // a third concurrent request is denied (the old non-consuming admission check let
        // them all slip through at ~0 budget).
        assert!(matches!(limiter.check("claude", 800), RateLimitResult::Allowed)); // 1000→200
        assert!(matches!(limiter.check("claude", 800), RateLimitResult::Allowed)); // 200→-600
        assert!(matches!(limiter.check("claude", 800), RateLimitResult::Denied { .. })); // <1 → denied
    }

    #[test]
    fn test_tpm_reconcile_refund_restores_budget() {
        let limiter = tpm_limiter(1000);
        assert!(matches!(limiter.check("claude", 800), RateLimitResult::Allowed)); // 200
        assert!(matches!(limiter.check("claude", 800), RateLimitResult::Allowed)); // -600
        assert!(matches!(limiter.check("claude", 800), RateLimitResult::Denied { .. }));
        // Reconcile one reservation down to 0 actual → refunds 800 (tokens -600 → 200).
        limiter.reconcile_tokens("claude", 800, 0);
        assert!(matches!(limiter.check("claude", 800), RateLimitResult::Allowed));
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
        limiter.reconcile_tokens("unknown", 0, 600);
        assert!(matches!(
            limiter.check("unknown", 0),
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
        limiter.reconcile_tokens("test", 0, 0);
        assert!(matches!(limiter.check("test", 0), RateLimitResult::Allowed));
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
        assert!(matches!(limiter.check("test", 0), RateLimitResult::Allowed));
        limiter.reconcile_tokens("test", 0, 600);

        // Should be denied by TPM even though RPM has capacity
        assert!(matches!(
            limiter.check("test", 0),
            RateLimitResult::Denied { .. }
        ));
    }

    // ─── Circuit breaker tests ─────────────────────────────────────

    #[test]
    fn test_circuit_breaker_closed_by_default() {
        let cb = CircuitBreaker::new(3, 30, ["backend1"].into_iter());
        assert!(cb.allow_request("backend1").0);
        assert_eq!(cb.state("backend1"), "closed");
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let cb = CircuitBreaker::new(3, 30, ["backend1"].into_iter());

        cb.record_failure("backend1", None);
        assert!(cb.allow_request("backend1").0); // 1 failure, still closed
        cb.record_failure("backend1", None);
        assert!(cb.allow_request("backend1").0); // 2 failures, still closed
        cb.record_failure("backend1", None);
        // 3 failures = threshold → open
        assert!(!cb.allow_request("backend1").0);
        assert_eq!(cb.state("backend1"), "open");
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let cb = CircuitBreaker::new(3, 30, ["backend1"].into_iter());

        cb.record_failure("backend1", None);
        cb.record_failure("backend1", None);
        // 2 failures, then success resets the count (still closed)
        cb.record_success("backend1", None);
        assert_eq!(cb.state("backend1"), "closed");
        assert!(cb.allow_request("backend1").0);

        // Need 3 consecutive failures again
        cb.record_failure("backend1", None);
        cb.record_failure("backend1", None);
        assert!(cb.allow_request("backend1").0); // still closed (only 2)
    }

    #[test]
    fn test_circuit_breaker_half_open_recovery() {
        let cb = CircuitBreaker::new(2, 0, ["b1"].into_iter()); // 0 second recovery for testing

        cb.record_failure("b1", None);
        cb.record_failure("b1", None);
        assert_eq!(cb.state("b1"), "open");

        // With 0s recovery timeout, should immediately transition to half-open
        let (allowed, probe) = cb.allow_request("b1");
        assert!(allowed);
        assert!(probe.is_some());
        assert_eq!(cb.state("b1"), "half_open");
        // (A second request within recovery_timeout is denied; not asserted here because the
        // test uses a 0s timeout, under which the abandoned-probe escape reissues instead —
        // see test_circuit_breaker_half_open_reissues_abandoned_probe.)

        // Probe success (with the probe token) → closed
        cb.record_success("b1", probe);
        assert_eq!(cb.state("b1"), "closed");
        assert!(cb.allow_request("b1").0);
    }

    #[test]
    fn test_circuit_breaker_half_open_reissues_abandoned_probe() {
        // A probe whose outcome is never recorded (e.g. a streaming probe whose client
        // disconnected, dropping the generator) must not leave the circuit stuck HalfOpen.
        // With 0s recovery the next request reissues a fresh probe.
        let cb = CircuitBreaker::new(2, 0, ["b1"].into_iter());
        cb.record_failure("b1", None);
        cb.record_failure("b1", None);
        let (a1, probe1) = cb.allow_request("b1"); // → HalfOpen, probe1
        assert!(a1);
        assert_eq!(cb.state("b1"), "half_open");

        // probe1 is never resolved → next request reissues a new probe (not stuck).
        let (a2, probe2) = cb.allow_request("b1");
        assert!(a2);
        assert!(probe2.is_some());
        assert_ne!(probe1, probe2);

        // A late, stale probe1 outcome is ignored; only the current probe2 resolves it.
        cb.record_success("b1", probe1);
        assert_eq!(cb.state("b1"), "half_open");
        cb.record_success("b1", probe2);
        assert_eq!(cb.state("b1"), "closed");
    }

    #[test]
    fn test_circuit_breaker_half_open_failure() {
        let cb = CircuitBreaker::new(2, 0, ["b1"].into_iter());

        cb.record_failure("b1", None);
        cb.record_failure("b1", None);
        let (allowed, probe) = cb.allow_request("b1"); // → HalfOpen
        assert!(allowed);

        // Probe fails (with the probe token) → back to Open
        cb.record_failure("b1", probe);
        assert_eq!(cb.state("b1"), "open");
    }

    #[test]
    fn test_circuit_breaker_half_open_ignores_stale_success() {
        // A request that started BEFORE the circuit opened (probe = None) must NOT close
        // the circuit while it's half-open — only the probe's own outcome decides.
        let cb = CircuitBreaker::new(2, 0, ["b1"].into_iter());

        cb.record_failure("b1", None);
        cb.record_failure("b1", None);
        let (allowed, probe) = cb.allow_request("b1"); // → HalfOpen, probe issued
        assert!(allowed);
        assert_eq!(cb.state("b1"), "half_open");

        // Stale (non-probe) success → ignored, stays half-open
        cb.record_success("b1", None);
        assert_eq!(cb.state("b1"), "half_open");

        // The real probe success closes it
        cb.record_success("b1", probe);
        assert_eq!(cb.state("b1"), "closed");
    }

    #[test]
    fn test_circuit_breaker_independent_backends() {
        let cb = CircuitBreaker::new(2, 30, ["backend_a", "backend_b"].into_iter());

        cb.record_failure("backend_a", None);
        cb.record_failure("backend_a", None);
        assert!(!cb.allow_request("backend_a").0); // open

        // backend_b should be unaffected
        assert!(cb.allow_request("backend_b").0);
        assert_eq!(cb.state("backend_b"), "closed");
    }
}
