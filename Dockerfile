# AIProxy — Rust multi-stage build.
# Build-context Gitea-ról jön (BuildKit-clone), futtatás Portainer Stack-ben.

# ─── Build stage ─────────────────────────────────────────────
FROM rust:1-slim-bookworm AS builder
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --bin ai-proxy

# ─── Runtime stage ───────────────────────────────────────────
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates libssl3 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/ai-proxy /usr/local/bin/ai-proxy
COPY ai-proxy.toml /app/ai-proxy.toml
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8800
ENTRYPOINT ["/app/docker-entrypoint.sh"]
