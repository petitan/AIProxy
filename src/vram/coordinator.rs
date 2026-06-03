use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use reqwest::Client;
use serde::Serialize;
use tokio::task::AbortHandle;
use uuid::Uuid;

use super::monitor::GpuMonitor;
use super::ollama::OllamaLifecycle;

/// How often the background reaper scans for and removes expired reservations.
const RESERVATION_REAPER_INTERVAL_SECS: u64 = 30;
/// Fallback reservation lifetime, used only if the requested timeout can't be converted to a
/// chrono duration (practically unreachable — the request timeout is range-validated upstream).
const RESERVATION_FALLBACK_TTL_SECS: i64 = 300;

/// Priority levels for VRAM allocation (higher = harder to evict)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum Priority {
    Idle = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

impl Priority {
    // `from_str` here is infallible (&str -> Self with a Medium default), not the fallible
    // `std::str::FromStr` trait — the name is intentional for call-site readability.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => Priority::Critical,
            "high" => Priority::High,
            "medium" => Priority::Medium,
            "low" => Priority::Low,
            "idle" => Priority::Idle,
            _ => Priority::Medium,
        }
    }
}

/// A model slot currently occupying VRAM
#[derive(Debug, Clone, Serialize)]
pub struct VramSlot {
    pub model: String,
    pub priority: Priority,
    pub estimated_vram_mb: u64,
    #[serde(skip)]
    #[allow(dead_code)]
    pub loaded_at: Instant,
    #[serde(skip)]
    pub last_used: Instant,
}

/// Result of an acquire attempt
#[derive(Debug)]
pub enum AcquireResult {
    /// Model is already loaded or was loaded successfully
    Acquired,
    /// Not enough VRAM even after eviction
    InsufficientVram {
        needed_mb: u64,
        available_mb: u64,
    },
    /// Eviction failed
    #[allow(dead_code)]
    EvictionFailed(String),
}

/// External VRAM reservation (for processes outside the proxy, e.g. DokLing).
///
/// Reservations block VRAM from being used by model acquire() calls.
/// The external process is responsible for releasing the reservation when done
/// (via DELETE /v1/vram/reserve/{id}), or it expires after `timeout_secs`.
#[derive(Debug, Clone, Serialize)]
pub struct Reservation {
    pub id: String,
    pub reserved_mb: u64,
    pub reason: String,
    pub priority: Priority,
    pub expires_at: DateTime<Utc>,
    #[serde(skip)]
    #[allow(dead_code)]
    pub created_at: Instant,
}

/// Result of a reservation attempt
#[derive(Debug, Serialize)]
pub struct ReservationResult {
    pub reservation_id: String,
    pub granted_mb: u64,
    pub evicted_models: Vec<String>,
    pub expires_at: DateTime<Utc>,
    pub budget: BudgetSummary,
}

/// Combined VRAM state under a single lock.
///
/// Lock ordering: always acquire `state` as a single unit.
/// This eliminates deadlock risk from nested locks on separate collections.
struct VramState {
    slots: HashMap<String, VramSlot>,
    reservations: HashMap<String, Reservation>,
}

impl VramState {
    fn new() -> Self {
        Self {
            slots: HashMap::new(),
            reservations: HashMap::new(),
        }
    }

    /// Available VRAM budget computed from locked state.
    /// Call this while holding the lock to get a consistent view.
    fn available_budget(&self, total_vram_mb: u64, overhead_mb: u64, permanent_mb: u64) -> u64 {
        let used_by_models: u64 = self.slots.values().map(|s| s.estimated_vram_mb).sum();
        let used_by_reservations: u64 = self.reservations.values().map(|r| r.reserved_mb).sum();
        let budget = total_vram_mb.saturating_sub(overhead_mb + permanent_mb);
        budget.saturating_sub(used_by_models + used_by_reservations)
    }
}

/// VRAM Coordinator — manages GPU memory allocation across models and external reservations.
///
/// **Reservation semantics:** An active reservation blocks the reserved VRAM from being
/// used by model acquire() calls. The eviction logic only evicts model slots, never
/// reservations — a reservation is a hard guarantee to the external process.
/// This means a large reservation can cause acquire() to reject requests until
/// the reservation is released or expires.
pub struct VramCoordinator {
    /// Combined state: model slots + external reservations (single lock)
    state: Arc<RwLock<VramState>>,
    /// GPU monitor for real VRAM queries
    monitor: GpuMonitor,
    /// Total VRAM budget (from config)
    total_vram_mb: u64,
    /// Overhead (system, Xwayland, etc.)
    overhead_mb: u64,
    /// Permanent reservations (e.g. reranker in Flask)
    permanent_mb: u64,
    /// Ollama lifecycle manager (None if no Ollama backend)
    ollama: Option<OllamaLifecycle>,
}

impl VramCoordinator {
    pub fn new(
        total_vram_mb: u64,
        overhead_mb: u64,
        permanent_mb: u64,
        ollama_base_url: Option<&str>,
    ) -> Self {
        let monitor = GpuMonitor::new(0);

        // Log real GPU info if available
        if let Some(info) = monitor.memory_info() {
            tracing::info!(
                gpu = %info.gpu_name,
                total_mb = info.total_mb,
                used_mb = info.used_mb,
                free_mb = info.free_mb,
                "GPU detected"
            );
        }

        let ollama = ollama_base_url.map(OllamaLifecycle::new);

        Self {
            state: Arc::new(RwLock::new(VramState::new())),
            monitor,
            total_vram_mb,
            overhead_mb,
            permanent_mb,
            ollama,
        }
    }

    /// Available VRAM budget (total - overhead - permanent - loaded models - reservations)
    pub fn available_budget_mb(&self) -> u64 {
        self.state.read().available_budget(self.total_vram_mb, self.overhead_mb, self.permanent_mb)
    }

    /// Try to acquire VRAM for a model. Evicts lower-priority models if needed.
    ///
    /// Uses optimistic slot registration to prevent TOCTOU races:
    /// the slot is registered BEFORE releasing the lock for the async HTTP call,
    /// so concurrent requests see the reserved VRAM as "used".
    pub async fn acquire(
        &self,
        client: &Client,
        model: &str,
        priority: Priority,
        estimated_vram_mb: u64,
    ) -> AcquireResult {
        // Phase 1: Check state and optimistically reserve under a single write lock
        let needs_load = {
            let mut state = self.state.write();

            // Already loaded? Just update priority and last_used
            if let Some(slot) = state.slots.get_mut(model) {
                if priority > slot.priority {
                    slot.priority = priority;
                }
                slot.last_used = Instant::now();
                return AcquireResult::Acquired;
            }

            // Calculate available budget while holding the lock (atomic read of both collections)
            let available = state.available_budget(self.total_vram_mb, self.overhead_mb, self.permanent_mb);

            if available >= estimated_vram_mb {
                // Enough space — register slot NOW (optimistic reservation)
                let now = Instant::now();
                state.slots.insert(model.to_string(), VramSlot {
                    model: model.to_string(),
                    priority,
                    estimated_vram_mb,
                    loaded_at: now,
                    last_used: now,
                });
                true // needs HTTP preload
            } else {
                false // needs eviction first
            }
        };
        // Lock released here

        if needs_load {
            // Phase 2: HTTP preload (slot already reserved, no race possible)
            // Note: preload failure is non-fatal — Ollama lazy-loads models on first
            // generation request. We keep the slot registered to prevent TOCTOU races.
            if let Err(e) = self.load_model(client, model).await {
                tracing::warn!(model = %model, error = %e, "Failed to preload model (Ollama will lazy-load on first request)");
            }
            tracing::info!(
                model = %model,
                priority = ?priority,
                vram_mb = estimated_vram_mb,
                "Model registered in VRAM coordinator"
            );
            return AcquireResult::Acquired;
        }

        // Phase 3: Need to evict — evict_for_space computes needed space under lock
        match self.evict_for_space(client, estimated_vram_mb, priority).await {
            Ok(freed) => {
                tracing::info!(
                    model = %model,
                    needed_mb = estimated_vram_mb,
                    freed_mb = freed,
                    "Evicted models to make space"
                );
            }
            Err(e) => {
                // Check if we have enough space despite partial eviction failure
                let available = self.available_budget_mb();
                if available < estimated_vram_mb {
                    tracing::warn!(
                        model = %model,
                        needed_mb = estimated_vram_mb,
                        available_mb = available,
                        error = %e,
                        "Cannot acquire VRAM"
                    );
                    return AcquireResult::InsufficientVram {
                        needed_mb: estimated_vram_mb,
                        available_mb: available,
                    };
                }
            }
        }

        // Phase 4: After eviction, atomically verify budget + register slot
        // Another request may have claimed the freed space between eviction and now
        {
            let mut state = self.state.write();
            let available = state.available_budget(self.total_vram_mb, self.overhead_mb, self.permanent_mb);

            if available < estimated_vram_mb {
                tracing::warn!(
                    model = %model,
                    needed_mb = estimated_vram_mb,
                    available_mb = available,
                    "VRAM claimed by another request after eviction"
                );
                return AcquireResult::InsufficientVram {
                    needed_mb: estimated_vram_mb,
                    available_mb: available,
                };
            }

            // Register optimistically under lock
            let now = Instant::now();
            state.slots.insert(model.to_string(), VramSlot {
                model: model.to_string(),
                priority,
                estimated_vram_mb,
                loaded_at: now,
                last_used: now,
            });
        }
        // Lock released — safe to do async preload
        if let Err(e) = self.load_model(client, model).await {
            tracing::warn!(model = %model, error = %e, "Failed to preload after eviction (Ollama will lazy-load on first request)");
        }
        tracing::info!(
            model = %model,
            priority = ?priority,
            vram_mb = estimated_vram_mb,
            "Model registered in VRAM coordinator (after eviction)"
        );
        AcquireResult::Acquired
    }

    /// Release a model from VRAM tracking (and unload from Ollama).
    /// Currently only exercised by tests — production uses `demote_to_idle` + eviction.
    #[cfg(test)]
    pub async fn release(&self, client: &Client, model: &str) {
        let existed = self.state.write().slots.remove(model).is_some();
        if existed {
            if let Err(e) = self.unload_model(client, model).await {
                tracing::warn!(model = %model, error = %e, "Failed to unload model");
            }
        }
    }

    /// Demote a model to IDLE priority (stays in VRAM but evictable)
    pub fn demote_to_idle(&self, model: &str) {
        let mut state = self.state.write();
        if let Some(slot) = state.slots.get_mut(model) {
            slot.priority = Priority::Idle;
            tracing::debug!(model = %model, "Model demoted to IDLE");
        }
    }

    /// Sync coordinator state with Ollama's actual loaded models
    pub async fn sync_with_ollama(&self, client: &Client) {
        let ollama = match &self.ollama {
            Some(o) => o,
            None => return,
        };

        match ollama.loaded_models(client).await {
            Ok(loaded) => {
                let mut state = self.state.write();
                for model in &loaded {
                    let full_name = &model.name;
                    if !state.slots.contains_key(full_name) {
                        let vram_mb = model.size_vram / (1024 * 1024);
                        tracing::info!(
                            model = %full_name,
                            vram_mb = vram_mb,
                            "Discovered untracked model in Ollama, registering as IDLE"
                        );
                        state.slots.insert(
                            full_name.clone(),
                            VramSlot {
                                model: full_name.clone(),
                                priority: Priority::Idle,
                                estimated_vram_mb: vram_mb,
                                loaded_at: Instant::now(),
                                last_used: Instant::now(),
                            },
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to sync with Ollama");
            }
        }
    }

    /// Get current slot status (for /status endpoint)
    pub fn slot_status(&self) -> Vec<VramSlot> {
        self.state.read().slots.values().cloned().collect()
    }

    /// Get GPU memory info (for /status endpoint)
    pub fn gpu_info(&self) -> Option<super::monitor::GpuMemoryInfo> {
        self.monitor.memory_info()
    }

    // ── Reservations ─────────────────────────────────────────────

    /// Reserve VRAM for an external process (e.g. DokLing model loading).
    ///
    /// Evicts lower-priority Ollama models if needed to free the requested space.
    /// The reservation blocks the VRAM from being claimed by subsequent acquire() calls
    /// until it is explicitly released (DELETE) or expires after `timeout`.
    ///
    /// **Important:** While a reservation is active, acquire() calls that would need
    /// the reserved VRAM will be rejected with InsufficientVram. The caller must
    /// release the reservation when done to restore normal model loading.
    pub async fn reserve(
        &self,
        client: &Client,
        required_free_mb: u64,
        priority: Priority,
        reason: String,
        timeout: Duration,
    ) -> Result<ReservationResult, ReserveError> {
        // Phase 1: Evict if needed (evict_for_space computes deficit under lock)
        let available = self.available_budget_mb();
        let mut evicted_models = Vec::new();

        if available < required_free_mb {
            // Snapshot model names before eviction
            let before: Vec<String> = self.state.read().slots.keys().cloned().collect();

            match self.evict_for_space(client, required_free_mb, priority).await {
                Ok(_freed) => {}
                Err(_) => {
                    let available = self.available_budget_mb();
                    if available < required_free_mb {
                        return Err(ReserveError::InsufficientVram {
                            required_mb: required_free_mb,
                            available_mb: available,
                        });
                    }
                }
            }

            // Collect evicted model names
            let after: std::collections::HashSet<String> =
                self.state.read().slots.keys().cloned().collect();
            evicted_models = before.into_iter().filter(|m| !after.contains(m)).collect();
        }

        // Phase 2: Atomically verify budget + register reservation under single lock
        let id = format!("res_{}", Uuid::new_v4().as_simple());
        let expires_at = Utc::now()
            + chrono::Duration::from_std(timeout)
                .unwrap_or(chrono::Duration::seconds(RESERVATION_FALLBACK_TTL_SECS));

        {
            let mut state = self.state.write();
            let available = state.available_budget(
                self.total_vram_mb,
                self.overhead_mb,
                self.permanent_mb,
            );
            if available < required_free_mb {
                return Err(ReserveError::InsufficientVram {
                    required_mb: required_free_mb,
                    available_mb: available,
                });
            }

            state.reservations.insert(
                id.clone(),
                Reservation {
                    id: id.clone(),
                    reserved_mb: required_free_mb,
                    reason: reason.clone(),
                    priority,
                    expires_at,
                    created_at: Instant::now(),
                },
            );
        }
        // Lock released

        tracing::info!(
            reservation_id = %id,
            reserved_mb = required_free_mb,
            reason = %reason,
            expires_at = %expires_at,
            "VRAM reservation created"
        );

        Ok(ReservationResult {
            reservation_id: id,
            granted_mb: required_free_mb,
            evicted_models,
            expires_at,
            budget: self.budget_summary(),
        })
    }

    /// Release a reservation by ID. Returns the freed MB, or None if not found.
    pub fn release_reservation(&self, reservation_id: &str) -> Option<u64> {
        let removed = self.state.write().reservations.remove(reservation_id);
        if let Some(ref res) = removed {
            tracing::info!(
                reservation_id = %reservation_id,
                freed_mb = res.reserved_mb,
                "VRAM reservation released"
            );
        }
        removed.map(|r| r.reserved_mb)
    }

    /// Get active reservations (for /status and GET /v1/vram/reservations)
    pub fn active_reservations(&self) -> Vec<Reservation> {
        self.state.read().reservations.values().cloned().collect()
    }

    /// Remove expired reservations. Called periodically by the reaper task.
    pub fn reap_expired(&self) -> Vec<String> {
        let now = Utc::now();
        let mut state = self.state.write();
        let expired: Vec<String> = state
            .reservations
            .iter()
            .filter(|(_, r)| r.expires_at <= now)
            .map(|(id, _)| id.clone())
            .collect();

        for id in &expired {
            if let Some(res) = state.reservations.remove(id) {
                tracing::warn!(
                    reservation_id = %id,
                    reserved_mb = res.reserved_mb,
                    reason = %res.reason,
                    "VRAM reservation expired — releasing"
                );
            }
        }
        expired
    }

    /// Spawn background task that reaps expired reservations every 30 seconds.
    /// Returns an AbortHandle that can be used to stop the reaper on shutdown.
    pub fn spawn_reservation_reaper(self: &Arc<Self>) -> AbortHandle {
        let coordinator = Arc::clone(self);
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(RESERVATION_REAPER_INTERVAL_SECS));
            loop {
                interval.tick().await;
                let expired = coordinator.reap_expired();
                if !expired.is_empty() {
                    tracing::info!(count = expired.len(), "Reaped expired reservations");
                }
            }
        });
        handle.abort_handle()
    }

    /// Get budget summary
    pub fn budget_summary(&self) -> BudgetSummary {
        let state = self.state.read();
        let used_by_models: u64 = state.slots.values().map(|s| s.estimated_vram_mb).sum();
        let reserved: u64 = state.reservations.values().map(|r| r.reserved_mb).sum();
        let budget = self.total_vram_mb.saturating_sub(self.overhead_mb + self.permanent_mb);
        BudgetSummary {
            total_mb: self.total_vram_mb,
            overhead_mb: self.overhead_mb,
            permanent_mb: self.permanent_mb,
            budget_mb: budget,
            used_by_models_mb: used_by_models,
            reserved_mb: reserved,
            // Backward compat: used_mb = total used (models + reservations)
            used_mb: used_by_models + reserved,
            available_mb: budget.saturating_sub(used_by_models + reserved),
        }
    }

    // ── Private ──────────────────────────────────────────────────

    #[cfg(test)]
    fn register_slot(&self, model: &str, priority: Priority, estimated_vram_mb: u64) {
        let now = Instant::now();
        self.state.write().slots.insert(
            model.to_string(),
            VramSlot {
                model: model.to_string(),
                priority,
                estimated_vram_mb,
                loaded_at: now,
                last_used: now,
            },
        );
        tracing::info!(
            model = %model,
            priority = ?priority,
            vram_mb = estimated_vram_mb,
            "Model registered in VRAM coordinator"
        );
    }

    /// Evict models to free at least `required_free_mb` of VRAM.
    /// Computes the actual deficit under lock to prevent TOCTOU races.
    /// Only evicts model slots with priority STRICTLY LOWER than `requesting_priority`.
    /// Never evicts reservations — those are hard guarantees to external processes.
    async fn evict_for_space(
        &self,
        client: &Client,
        required_free_mb: u64,
        requesting_priority: Priority,
    ) -> Result<u64, String> {
        // Compute needed space and collect candidates atomically under a single lock
        let (needed_mb, candidates): (u64, Vec<(String, u64, Priority, Instant)>) = {
            let state = self.state.read();
            let available = state.available_budget(self.total_vram_mb, self.overhead_mb, self.permanent_mb);
            let needed = required_free_mb.saturating_sub(available);
            if needed == 0 {
                return Ok(0); // Another request freed space in the meantime
            }
            let mut candidates: Vec<_> = state
                .slots
                .values()
                .filter(|s| s.priority < requesting_priority)
                .map(|s| (s.model.clone(), s.estimated_vram_mb, s.priority, s.last_used))
                .collect();
            // Sort: lowest priority first, then oldest last_used (LRU within same priority)
            candidates.sort_by(|a, b| a.2.cmp(&b.2).then(a.3.cmp(&b.3)));
            (needed, candidates)
        };

        if candidates.is_empty() {
            return Err("No evictable models (all have equal or higher priority)".into());
        }

        let mut freed: u64 = 0;
        for (model, vram_mb, priority, _last_used) in &candidates {
            if freed >= needed_mb {
                break;
            }
            tracing::info!(
                model = %model,
                priority = ?priority,
                vram_mb = vram_mb,
                "Evicting model"
            );
            // Unload from Ollama — if this fails, still remove the slot.
            // The model is either offline (VRAM is free) or unreachable (can't manage it anyway).
            // NOT removing the slot on unload failure causes eviction deadlock.
            if let Err(e) = self.unload_model(client, model).await {
                tracing::warn!(model = %model, error = %e, "Failed to unload during eviction (removing slot anyway)");
            }
            // Remove from slots — check result to avoid double-counting
            // (another concurrent eviction may have already removed this model)
            if self.state.write().slots.remove(model.as_str()).is_some() {
                freed += vram_mb;
            }
        }

        if freed >= needed_mb {
            Ok(freed)
        } else {
            Err(format!(
                "Could only free {} MB, needed {} MB",
                freed, needed_mb
            ))
        }
    }

    async fn load_model(&self, client: &Client, model: &str) -> Result<(), String> {
        if let Some(ref ollama) = self.ollama {
            ollama.preload(client, model).await
        } else {
            Ok(()) // No Ollama — nothing to preload
        }
    }

    async fn unload_model(&self, client: &Client, model: &str) -> Result<(), String> {
        if let Some(ref ollama) = self.ollama {
            ollama.unload(client, model).await
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BudgetSummary {
    pub total_mb: u64,
    pub overhead_mb: u64,
    pub permanent_mb: u64,
    pub budget_mb: u64,
    pub used_by_models_mb: u64,
    pub reserved_mb: u64,
    /// Backward compat: total used VRAM (models + reservations)
    pub used_mb: u64,
    pub available_mb: u64,
}

/// Error from a reservation attempt
#[derive(Debug)]
pub enum ReserveError {
    InsufficientVram {
        required_mb: u64,
        available_mb: u64,
    },
}

impl std::fmt::Display for ReserveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReserveError::InsufficientVram {
                required_mb,
                available_mb,
            } => {
                write!(
                    f,
                    "Insufficient VRAM: need {} MB, only {} MB available",
                    required_mb, available_mb
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coordinator(total: u64, overhead: u64, permanent: u64) -> VramCoordinator {
        VramCoordinator::new(total, overhead, permanent, None)
    }

    #[test]
    fn test_budget_calculation() {
        let coord = make_coordinator(24576, 1024, 800);
        let budget = coord.budget_summary();
        assert_eq!(budget.total_mb, 24576);
        assert_eq!(budget.overhead_mb, 1024);
        assert_eq!(budget.permanent_mb, 800);
        assert_eq!(budget.budget_mb, 22752); // 24576 - 1024 - 800
        assert_eq!(budget.used_by_models_mb, 0);
        assert_eq!(budget.used_mb, 0);
        assert_eq!(budget.available_mb, 22752);
    }

    #[test]
    fn test_budget_with_loaded_model() {
        let coord = make_coordinator(24576, 1024, 800);
        coord.register_slot("model-a", Priority::High, 6400);
        let budget = coord.budget_summary();
        assert_eq!(budget.used_by_models_mb, 6400);
        assert_eq!(budget.used_mb, 6400);
        assert_eq!(budget.available_mb, 22752 - 6400);
    }

    #[test]
    fn test_budget_saturating_sub() {
        // Budget smaller than overhead — should not underflow
        let coord = make_coordinator(500, 1000, 0);
        let budget = coord.budget_summary();
        assert_eq!(budget.budget_mb, 0);
        assert_eq!(budget.available_mb, 0);
    }

    #[test]
    fn test_register_and_slot_status() {
        let coord = make_coordinator(24576, 0, 0);
        assert_eq!(coord.slot_status().len(), 0);

        coord.register_slot("model-a", Priority::High, 6400);
        let slots = coord.slot_status();
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].model, "model-a");
        assert_eq!(slots[0].priority, Priority::High);
        assert_eq!(slots[0].estimated_vram_mb, 6400);
    }

    #[test]
    fn test_demote_to_idle() {
        let coord = make_coordinator(24576, 0, 0);
        coord.register_slot("model-a", Priority::High, 6400);

        coord.demote_to_idle("model-a");
        let slots = coord.slot_status();
        assert_eq!(slots[0].priority, Priority::Idle);
    }

    #[test]
    fn test_demote_nonexistent_model() {
        let coord = make_coordinator(24576, 0, 0);
        // Should not panic
        coord.demote_to_idle("nonexistent");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(Priority::Idle < Priority::Low);
        assert!(Priority::Low < Priority::Medium);
        assert!(Priority::Medium < Priority::High);
        assert!(Priority::High < Priority::Critical);
    }

    #[test]
    fn test_priority_from_str() {
        assert_eq!(Priority::from_str("critical"), Priority::Critical);
        assert_eq!(Priority::from_str("HIGH"), Priority::High);
        assert_eq!(Priority::from_str("medium"), Priority::Medium);
        assert_eq!(Priority::from_str("low"), Priority::Low);
        assert_eq!(Priority::from_str("idle"), Priority::Idle);
        assert_eq!(Priority::from_str("unknown"), Priority::Medium); // default
    }

    #[tokio::test]
    async fn test_acquire_already_loaded() {
        let coord = make_coordinator(24576, 0, 0);
        let client = reqwest::Client::new();
        coord.register_slot("model-a", Priority::Low, 6400);

        // Acquire same model → should upgrade priority
        let result = coord.acquire(&client, "model-a", Priority::High, 6400).await;
        assert!(matches!(result, AcquireResult::Acquired));

        let slots = coord.slot_status();
        assert_eq!(slots[0].priority, Priority::High);
    }

    #[tokio::test]
    async fn test_acquire_enough_budget() {
        let coord = make_coordinator(24576, 0, 0);
        let client = reqwest::Client::new();

        let result = coord.acquire(&client, "model-a", Priority::Medium, 6400).await;
        assert!(matches!(result, AcquireResult::Acquired));

        let budget = coord.budget_summary();
        assert_eq!(budget.used_by_models_mb, 6400);
    }

    #[tokio::test]
    async fn test_acquire_insufficient_vram() {
        let coord = make_coordinator(8000, 1000, 0);
        let client = reqwest::Client::new();

        // Fill up budget
        coord.register_slot("model-a", Priority::Critical, 7000);

        // Try to acquire — not enough space, can't evict Critical
        let result = coord.acquire(&client, "model-b", Priority::Critical, 5000).await;
        assert!(matches!(result, AcquireResult::InsufficientVram { .. }));
    }

    #[tokio::test]
    async fn test_acquire_evicts_lower_priority() {
        let coord = make_coordinator(16000, 0, 0);
        let client = reqwest::Client::new();

        // Load a low-priority model
        coord.register_slot("model-low", Priority::Idle, 10000);

        // Acquire high-priority model — should evict model-low (no Ollama, so unload is no-op)
        let result = coord.acquire(&client, "model-high", Priority::High, 10000).await;
        assert!(matches!(result, AcquireResult::Acquired));

        let slots = coord.slot_status();
        assert_eq!(slots.len(), 1);
        assert_eq!(slots[0].model, "model-high");
    }

    #[tokio::test]
    async fn test_acquire_does_not_evict_equal_priority() {
        let coord = make_coordinator(16000, 0, 0);
        let client = reqwest::Client::new();

        // Load a medium-priority model
        coord.register_slot("model-a", Priority::Medium, 10000);

        // Try to acquire another medium — should NOT evict (only strictly lower)
        let result = coord.acquire(&client, "model-b", Priority::Medium, 10000).await;
        assert!(matches!(result, AcquireResult::InsufficientVram { .. }));
    }

    #[tokio::test]
    async fn test_release() {
        let coord = make_coordinator(24576, 0, 0);
        let client = reqwest::Client::new();

        coord.register_slot("model-a", Priority::High, 6400);
        assert_eq!(coord.slot_status().len(), 1);

        coord.release(&client, "model-a").await;
        assert_eq!(coord.slot_status().len(), 0);
        assert_eq!(coord.budget_summary().used_by_models_mb, 0);
    }

    #[tokio::test]
    async fn test_reservation_blocks_budget() {
        let coord = make_coordinator(16000, 0, 0);
        let client = reqwest::Client::new();

        // Reserve 10 GB
        let result = coord
            .reserve(&client, 10000, Priority::High, "test".into(), Duration::from_secs(60))
            .await;
        assert!(result.is_ok());
        let res = result.unwrap();

        // Budget should reflect reservation
        let budget = coord.budget_summary();
        assert_eq!(budget.reserved_mb, 10000);
        assert_eq!(budget.available_mb, 6000);

        // Acquiring a model that needs more than 6 GB should fail
        let acq = coord.acquire(&client, "big-model", Priority::Medium, 8000).await;
        assert!(matches!(acq, AcquireResult::InsufficientVram { .. }));

        // Release reservation
        let freed = coord.release_reservation(&res.reservation_id);
        assert_eq!(freed, Some(10000));

        // Now the model fits
        let acq = coord.acquire(&client, "big-model", Priority::Medium, 8000).await;
        assert!(matches!(acq, AcquireResult::Acquired));
    }

    #[test]
    fn test_reap_expired() {
        let coord = make_coordinator(24576, 0, 0);

        // Insert an already-expired reservation
        {
            let mut state = coord.state.write();
            state.reservations.insert(
                "res_expired".to_string(),
                Reservation {
                    id: "res_expired".to_string(),
                    reserved_mb: 5000,
                    reason: "test".to_string(),
                    priority: Priority::High,
                    expires_at: Utc::now() - chrono::Duration::seconds(10),
                    created_at: Instant::now(),
                },
            );
        }

        assert_eq!(coord.active_reservations().len(), 1);
        let expired = coord.reap_expired();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0], "res_expired");
        assert_eq!(coord.active_reservations().len(), 0);
    }

    #[test]
    fn test_reservation_id_uniqueness() {
        // Generate multiple IDs and verify they're unique
        let ids: std::collections::HashSet<String> = (0..100)
            .map(|_| format!("res_{}", Uuid::new_v4().as_simple()))
            .collect();
        assert_eq!(ids.len(), 100);
    }
}
