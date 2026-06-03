use std::path::PathBuf;

use clap::Parser;
use tracing_subscriber::EnvFilter;

mod backends;
mod config;
mod error;
mod metrics;
mod middleware;
mod rate_limiter;
mod routes;
mod server;
mod vram;

use config::ProxyConfig;

#[derive(Parser)]
#[command(name = "ai-proxy", version, about = "AI Proxy — unified routing for AI backends")]
struct Cli {
    /// Path to config file
    #[arg(short, long, default_value = "ai-proxy.toml")]
    config: PathBuf,

    /// Log format: "json" for production (journald/loki), "pretty" for development
    #[arg(long, default_value = "pretty")]
    log_format: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Read the config file ONCE. We need log_level before initializing tracing, then the
    // fully-validated config afterwards — both come from this single read (no second read,
    // no TOCTOU window where the file could change between the two).
    let config_content = std::fs::read_to_string(&cli.config)?;
    let config_preview: toml::Value = config_content.parse()?;
    let config_log_level = config_preview
        .get("proxy")
        .and_then(|p| p.get("log_level"))
        .and_then(|v| v.as_str())
        .unwrap_or("info");

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(config_log_level));

    match cli.log_format.as_str() {
        "json" => {
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(env_filter)
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false)
                .init();
        }
        _ => {
            tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .init();
        }
    }

    tracing::info!("AI Proxy v{}", env!("CARGO_PKG_VERSION"));

    let config = ProxyConfig::from_toml_str(&config_content)?;

    tracing::info!(
        listen = %config.proxy.listen,
        backends = config.backends.len(),
        models = config.models.len(),
        "Configuration loaded"
    );

    for (name, backend) in &config.backends {
        tracing::info!(
            name = %name,
            backend_type = ?backend.backend_type,
            url = %backend.base_url,
            timeout_secs = backend.effective_timeout().as_secs(),
            max_retries = backend.effective_max_retries(),
            "Backend registered"
        );
    }

    for model in config.models.values() {
        tracing::info!(
            model = %model.name,
            backend = %model.backend,
            "Model route registered"
        );
    }

    let listen_addr = config.proxy.listen.clone();
    let router = server::build_router(config);

    let listener = tokio::net::TcpListener::bind(&listen_addr).await?;
    tracing::info!(addr = %listen_addr, "AI Proxy listening");

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    tracing::info!("AI Proxy shut down gracefully");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Ctrl+C received"),
        _ = terminate => tracing::info!("SIGTERM received"),
    }
}
