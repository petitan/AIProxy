use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

/// Information about a model currently loaded in Ollama
#[derive(Debug, Clone, Deserialize)]
pub struct OllamaLoadedModel {
    pub name: String,
    #[allow(dead_code)]
    pub size: u64,
    #[serde(default)]
    pub size_vram: u64,
}

#[derive(Debug, Deserialize)]
struct PsResponse {
    models: Vec<OllamaLoadedModel>,
}

/// Ollama model lifecycle management.
/// The proxy is the ONLY component that sends keep_alive directives.
pub struct OllamaLifecycle {
    base_url: String,
}

impl OllamaLifecycle {
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Query which models are currently loaded in Ollama
    pub async fn loaded_models(&self, client: &Client) -> Result<Vec<OllamaLoadedModel>, String> {
        let url = format!("{}/api/ps", self.base_url);
        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Ollama ps request failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!("Ollama ps returned {}", resp.status()));
        }

        let ps: PsResponse = resp
            .json()
            .await
            .map_err(|e| format!("Failed to parse Ollama ps response: {}", e))?;

        Ok(ps.models)
    }

    /// Preload a model into VRAM (keep_alive: "24h")
    pub async fn preload(&self, client: &Client, model: &str) -> Result<(), String> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": model,
            "keep_alive": "24h",
            "prompt": "",
        });

        tracing::info!(model = %model, "Preloading model into VRAM");

        let resp = client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(120))
            .send()
            .await
            .map_err(|e| format!("Ollama preload failed: {}", e))?;

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_default();
            return Err(format!("Ollama preload returned error: {}", error));
        }

        // Consume the response body (Ollama streams even empty responses)
        let _ = resp.bytes().await;

        tracing::info!(model = %model, "Model preloaded");
        Ok(())
    }

    /// Unload a model from VRAM (keep_alive: "0")
    pub async fn unload(&self, client: &Client, model: &str) -> Result<(), String> {
        let url = format!("{}/api/generate", self.base_url);
        let body = json!({
            "model": model,
            "keep_alive": "0",
            "prompt": "",
        });

        tracing::info!(model = %model, "Unloading model from VRAM");

        let resp = client
            .post(&url)
            .json(&body)
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .await
            .map_err(|e| format!("Ollama unload failed: {}", e))?;

        if !resp.status().is_success() {
            let error = resp.text().await.unwrap_or_default();
            return Err(format!("Ollama unload returned error: {}", error));
        }

        let _ = resp.bytes().await;

        tracing::info!(model = %model, "Model unloaded");
        Ok(())
    }

    /// Check if a specific model is currently loaded
    #[allow(dead_code)]
    pub async fn is_loaded(&self, client: &Client, model: &str) -> bool {
        match self.loaded_models(client).await {
            Ok(models) => models.iter().any(|m| m.name == model),
            Err(_) => false,
        }
    }
}
