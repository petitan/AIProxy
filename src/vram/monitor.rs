use serde::Serialize;

/// GPU memory information
#[derive(Debug, Clone, Serialize)]
pub struct GpuMemoryInfo {
    pub total_mb: u64,
    pub used_mb: u64,
    pub free_mb: u64,
    pub gpu_name: String,
}

/// GPU monitor — wraps nvml-wrapper for real VRAM queries
pub struct GpuMonitor {
    nvml: Option<nvml_wrapper::Nvml>,
    device_index: u32,
}

impl GpuMonitor {
    /// Initialize GPU monitor. Returns Ok even if NVML is unavailable (fallback mode).
    pub fn new(device_index: u32) -> Self {
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(nvml) => {
                tracing::info!("NVML initialized successfully");
                Some(nvml)
            }
            Err(e) => {
                tracing::warn!(error = %e, "NVML init failed — running without GPU monitoring");
                None
            }
        };
        Self { nvml, device_index }
    }

    /// Query real VRAM usage. Returns None if NVML is unavailable.
    pub fn memory_info(&self) -> Option<GpuMemoryInfo> {
        let nvml = self.nvml.as_ref()?;
        let device = nvml.device_by_index(self.device_index).ok()?;
        let mem = device.memory_info().ok()?;
        let name = device.name().unwrap_or_else(|_| "Unknown GPU".into());

        Some(GpuMemoryInfo {
            total_mb: mem.total / (1024 * 1024),
            used_mb: mem.used / (1024 * 1024),
            free_mb: mem.free / (1024 * 1024),
            gpu_name: name,
        })
    }

    /// Get available VRAM in MB (real if NVML available, None otherwise)
    #[allow(dead_code)]
    pub fn available_vram_mb(&self) -> Option<u64> {
        self.memory_info().map(|m| m.free_mb)
    }

    /// Check if NVML is available
    #[allow(dead_code)]
    pub fn is_available(&self) -> bool {
        self.nvml.is_some()
    }
}
