# AIProxy

Unified routing proxy for local and remote AI backends. OpenAI-compatible API that transparently routes between **Ollama** (local GPU) and **Anthropic** (remote API), with automatic format conversion, VRAM management, rate limiting, and authentication.

```
Client (OpenAI SDK)  ──>  AIProxy  ──>  Ollama (local, OpenAI format)
                                   ──>  Anthropic (remote, converted)
```

## Quick Start

### Build

```bash
cargo build --release
```

### Configure

```toml
# ai-proxy.toml — minimal working config

[proxy]
listen = "127.0.0.1:8800"
api_key_env = "AI_PROXY_KEY"   # Optional: omit to disable auth

[backends.ollama]
type = "local"
base_url = "http://localhost:11434"
format = "openai"

[backends.anthropic]
type = "remote"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
format = "anthropic"

[models.qwen]
name = "qwen3.5:4b"
backend = "ollama"
estimated_vram_mb = 6400
default_priority = "high"

[models.claude_sonnet]
name = "claude-sonnet-4-20250514"
backend = "anthropic"
```

### Run

```bash
# Without auth
./target/release/ai-proxy --config ai-proxy.toml

# With auth
AI_PROXY_KEY="your-secret-key" ./target/release/ai-proxy --config ai-proxy.toml

# JSON logging (production)
AI_PROXY_KEY="your-secret-key" ./target/release/ai-proxy --config ai-proxy.toml --log-format json
```

### Use

```bash
# Chat completion
curl http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Streaming
curl -N http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:4b",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'

# List models
curl http://localhost:8800/v1/models -H "Authorization: Bearer your-secret-key"

# Health check (no auth required)
curl http://localhost:8800/health
```

Works with any OpenAI-compatible SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="your-secret-key"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

---

## Model Fallback

When a local model can't load (VRAM full), AIProxy automatically routes to a configured fallback model — typically a remote API.

### Setup

```toml
[models.qwen]
name = "qwen3.5:4b"
backend = "ollama"
estimated_vram_mb = 6400
fallback = "claude_haiku"       # <-- fallback model key

[models.claude_haiku]
name = "claude-haiku-4-5-20251001"
backend = "anthropic"
```

### What happens

1. Client sends `"model": "qwen3.5:4b"`
2. AIProxy tries to acquire VRAM for qwen on Ollama
3. VRAM is full, can't evict (all models are equal or higher priority)
4. AIProxy transparently routes to `claude_haiku` instead
5. Client gets a response from Claude — same OpenAI format, no client-side changes

The fallback model's `extra_body` defaults and backend config are used (not the original model's). Self-referencing fallbacks (A → A) are rejected at config validation.

### Per-model defaults with extra_body

Inject default parameters per model — useful for models that need specific settings:

```toml
[models.qwen]
name = "qwen3.5:4b"
backend = "ollama"
extra_body = { "num_ctx" = 8192, "temperature" = 0.7 }
```

Client-sent values always take priority. `extra_body` only fills missing keys.

---

## VRAM Coordinator

Priority-based GPU memory management for local (Ollama) models. Tracks which models are loaded, evicts lower-priority ones when needed, and supports external VRAM reservations.

### Setup

```toml
[gpu.slots.slot_0]
total_vram_mb = 24576    # Your GPU's total VRAM
overhead_mb = 1024       # OS/display overhead

[gpu.slots.permanent]
reranker = { estimated_vram_mb = 800 }  # Always reserved (e.g. Flask reranker)
```

VRAM budget = `total_vram_mb - overhead_mb - permanent_reservations`

### Priority levels

| Priority | Evictable by | Typical use |
|----------|--------------|-------------|
| `idle` | Any higher priority | Loaded but not in use |
| `low` | Medium+ | Background tasks |
| `medium` | High+ | Normal requests (default) |
| `high` | Critical only | Priority requests |
| `critical` | Never | Must stay loaded |

Active requests run at their configured priority. After completion, models are demoted to `idle` — they stay loaded but are first in line for eviction.

### How eviction works

When a model needs VRAM and there isn't enough:

1. Find models with **strictly lower** priority
2. Sort: lowest priority first, then **least recently used** (LRU) within same priority
3. Evict (unload from Ollama) until enough space is free
4. Atomically verify budget + register new model under a single lock (prevents TOCTOU race)

If eviction fails (Ollama unreachable), the slot is still removed — the model is offline anyway, and keeping the slot would deadlock future evictions.

### Streaming safety

Streaming responses are wrapped in `DemoteOnDropBody`. While the stream is active, the model stays at its request priority (protected from eviction). When the stream ends or the connection drops, the model is automatically demoted to `idle`.

### External VRAM reservations (DocLing, custom models)

External processes that need GPU memory can reserve VRAM through the proxy. While a reservation is active, AIProxy won't allocate that VRAM to Ollama models.

**Typical workflow:**

```bash
# 1. Reserve VRAM before loading your model
RESPONSE=$(curl -s -X POST http://localhost:8800/v1/vram/reserve \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "required_free_mb": 4000,
    "priority": "high",
    "reason": "DocLing layout model",
    "timeout_secs": 600
  }')

# Response:
# {
#   "reservation_id": "res_a1b2c3...",
#   "granted_mb": 4000,
#   "evicted_models": ["qwen3.5:4b"],   <-- evicted to make space
#   "expires_at": "2026-03-11T15:10:00Z",
#   "budget": { "total_mb": 24576, "available_mb": 12352, ... }
# }

RESERVATION_ID=$(echo $RESPONSE | jq -r .reservation_id)

# 2. Run your process (uses the reserved GPU memory)
# ...

# 3. Release reservation when done
curl -s -X DELETE "http://localhost:8800/v1/vram/reserve/$RESERVATION_ID" \
  -H "Authorization: Bearer $KEY"
```

**From Python (PeTitanWeb Flask):**

```python
import requests

PROXY = "http://localhost:8800"
HEADERS = {"Authorization": "Bearer your-key"}

# Reserve
resp = requests.post(f"{PROXY}/v1/vram/reserve", headers=HEADERS, json={
    "required_free_mb": 4000,
    "priority": "high",
    "reason": "DocLing layout model",
    "timeout_secs": 600,
})
reservation_id = resp.json()["reservation_id"]

# ... run DocLing ...

# Release
requests.delete(f"{PROXY}/v1/vram/reserve/{reservation_id}", headers=HEADERS)
```

**What happens if the caller crashes:** the reservation auto-expires after `timeout_secs` (checked every 30s). No manual cleanup needed.

**Validation:** `required_free_mb` must be > 0 and can't exceed the total VRAM budget. Absurd values are rejected immediately instead of evicting everything first.

### Startup sync

On startup, the coordinator queries Ollama's `/api/ps` to discover already-loaded models and registers them as `idle`. This prevents the proxy from trying to load models into already-occupied VRAM.

---

## Rate Limiting

Per-model request throttling with two independent token buckets: **RPM** (requests per minute) and **TPM** (tokens per minute).

### Setup

```toml
# Per-model limit
[rate_limits.claude_sonnet]
requests_per_minute = 50
tokens_per_minute = 100000

# Default fallback (applies to models without specific limits)
[rate_limits.default]
requests_per_minute = 60
tokens_per_minute = 200000
```

Models without any configured limit (and no `default`) are unlimited.

### What happens when limited

```bash
# Client gets 429 with Retry-After header:
HTTP/1.1 429 Too Many Requests
Retry-After: 2.5

{
  "error": {
    "message": "Rate limit exceeded. Retry after 2.50 seconds",
    "type": "rate_limit_exceeded"
  }
}
```

The `Retry-After` value is precise — it's the exact seconds until the next token refills.

### How RPM + TPM work together

1. **TPM check first** (non-consuming) — "is there capacity?"
2. **RPM consume** (only if TPM passed) — deducts 1 request token

This atomic ordering prevents wasting an RPM token when TPM would deny the request anyway.

**Token accounting:**
- Non-streaming: actual `prompt_tokens + completion_tokens` from the response is deducted from TPM
- Streaming: `max_tokens` from the request body is **pre-deducted** (actual count unknown during streaming), preventing TPM bypass

### What to do when users hit rate limits

```bash
# Check current metrics to see if limits are too tight
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/metrics | \
  jq '.models["claude-sonnet-4-20250514"]'
```

If `total_requests` is high but `total_errors` is low, the limit is working as intended. If legitimate traffic is getting throttled, increase the limits in config and restart.

---

## Circuit Breaker

Per-backend circuit breaker that stops sending requests to a failing backend, giving it time to recover.

### Setup

```toml
[circuit_breaker]
failure_threshold = 5        # Open after 5 consecutive failures
recovery_timeout_secs = 30   # Wait 30s before trying again
```

### States

```
            success
  ┌──────────────────────┐
  │                      │
  ▼        5 failures    │     30s timeout      1 probe
Closed  ──────────────> Open ──────────────> HalfOpen
  ▲                      ▲                      │
  │                      └───── failure ────────┘
  │                                             │
  └──────────── success ────────────────────────┘
```

| State | What happens to requests |
|-------|--------------------------|
| **Closed** | Normal — requests go through, consecutive failures counted |
| **Open** | All requests immediately rejected with 503 |
| **HalfOpen** | One probe request allowed — success → Closed, failure → Open |

### What counts as failure/success

- **Failure:** 5xx server errors, connection errors/timeouts. Each retry attempt counts separately.
- **Success:** 2xx responses, 4xx client errors (backend is reachable, client sent bad data)

### How to diagnose a tripped circuit breaker

```bash
# Check backend states
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/status | \
  jq '.backends[] | {name, circuit}'

# Check trip count history
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/metrics | \
  jq '.backends.ollama.circuit_breaker_trips'
```

If a circuit is open: the backend is down. Check the backend directly (e.g. `curl http://localhost:11434/api/tags`). The circuit will auto-recover after `recovery_timeout_secs` once the backend is back.

The circuit breaker is checked **before** VRAM acquire — if Ollama is down, AIProxy won't waste time evicting models for a request that would fail anyway.

---

## Anthropic Format Conversion

AIProxy automatically converts between OpenAI and Anthropic message formats. Clients always use the OpenAI SDK — the proxy handles the translation.

| Feature | Support |
|---------|---------|
| Text messages | Full |
| System messages | Extracted to Anthropic `system` field |
| Tool definitions | `function` -> `tool`, `parameters` -> `input_schema` |
| Tool calls (response) | `tool_use` -> OpenAI `tool_calls` |
| Tool results | `tool` role -> `user` with `tool_result` content |
| tool_choice | `auto`/`none`/`required`(`any`)/specific tool |
| Streaming (SSE) | Full (message_start, text_delta, tool_use, message_delta) |
| Images (base64) | Converted to Anthropic image content blocks |
| Thinking blocks | Filtered out (not forwarded to client) |
| response_format | Appended as system prompt instruction |
| stop sequences | String or array, normalized to `stop_sequences` |
| Usage tracking | `input_tokens`/`output_tokens` -> `prompt_tokens`/`completion_tokens` |
| Consecutive role merge | Auto-merged for Anthropic's alternating role requirement |

**Not proxied** (Anthropic doesn't support): `frequency_penalty`, `presence_penalty`, `logprobs`, `seed`, `logit_bias`

---

## Authentication

Optional Bearer token authentication.

```toml
[proxy]
api_key_env = "AI_PROXY_KEY"   # Env var name, not the key itself
```

- If set: all endpoints except `/health` require `Authorization: Bearer <key>`
- If unset or env var empty: auth is disabled
- Uses **constant-time comparison** (`subtle::ConstantTimeEq`) — resistant to timing attacks
- Fail-closed: if the env var is configured but missing at startup, the proxy refuses to start

---

## Retry Logic

Automatic retry with exponential backoff for transient errors.

| Error | Retried? |
|-------|----------|
| 429 Too Many Requests | Yes |
| 500 Internal Server Error | Yes |
| 502 Bad Gateway | Yes |
| 503 Service Unavailable | Yes |
| 529 Site Overloaded | Yes |
| Connection errors / timeouts | Yes |
| Client errors (4xx) | No |
| Streaming requests | Never (body not replayable) |

**Backoff**: 1s, 2s, 4s (capped at 8s). Default retries: 0 (local), 2 (remote).

Each failed retry attempt is recorded as a separate circuit breaker failure — a request that fails 3 times records 3 failures, not 1.

---

## Monitoring & Operations

### Metrics endpoint

```bash
# Full metrics
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/metrics | jq
```

```json
{
  "uptime_seconds": 3600,
  "models": {
    "claude-sonnet-4-20250514": {
      "total_requests": 150,
      "total_errors": 3,
      "total_prompt_tokens": 50000,
      "total_completion_tokens": 25000,
      "latency_ms": {
        "count": 147,
        "avg": 1234.5,
        "min": 200.0,
        "max": 8500.0,
        "p95_approx": 5200.0
      }
    }
  },
  "backends": {
    "anthropic": {
      "total_requests": 200,
      "errors": { "client_4xx": 5, "server_5xx": 2, "connection": 1 },
      "circuit_breaker_trips": 0
    }
  }
}
```

Latency is tracked for non-streaming requests only. When no requests have been recorded, latency fields are `null` (not `0.0` or `Infinity`). p95 is approximate — based on a ring buffer of the last 256 values.

### Useful monitoring queries

```bash
# Is everything healthy?
curl -s http://localhost:8800/health | jq .status

# Which models are loaded in VRAM?
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/status | jq '.vram.loaded_models'

# Is any circuit breaker open?
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/status | \
  jq '.backends[] | select(.circuit != "closed")'

# Which model has the highest error rate?
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/metrics | \
  jq '[.models | to_entries[] | {model: .key, error_pct: ((.value.total_errors / .value.total_requests) * 100)}] | sort_by(-.error_pct)'

# Active VRAM reservations
curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/v1/vram/reservations | jq

# Watch VRAM in real-time
watch -n5 'curl -s -H "Authorization: Bearer $KEY" http://localhost:8800/status | jq .vram'
```

### Request tracing

Every response includes an `X-Request-Id` header (UUID v4). The same ID is logged server-side — match client errors to server logs:

```json
{"timestamp":"...","level":"INFO","fields":{"request_id":"550e8400-...","method":"POST","path":"/v1/chat/completions","status":200,"latency_ms":1234}}
```

### Health check integration

`/health` is unauthenticated — safe for load balancers and orchestrators.

```yaml
# Kubernetes
livenessProbe:
  httpGet:
    path: /health
    port: 8800
  initialDelaySeconds: 5
  periodSeconds: 10
```

```bash
# Simple watchdog
while true; do
  curl -sf http://localhost:8800/health > /dev/null || systemctl restart ai-proxy
  sleep 30
done
```

Response: `200` with `"status": "ok"` (all backends healthy) or `"status": "degraded"` (at least one backend down).

### Logging

```bash
RUST_LOG=debug    # VRAM demote events, extra_body injection
RUST_LOG=info     # Requests, startup, backend registration (default)
RUST_LOG=warn     # Rate limits, auth failures, retries, circuit breaker
RUST_LOG=error    # Stream errors, backend failures
```

Use `--log-format json` for production (structured, parseable by journald/loki/elasticsearch).
Use `--log-format pretty` for development (human-readable, colored).

---

## Production Deployment

### Systemd

```ini
# /etc/systemd/system/ai-proxy.service
[Unit]
Description=AIProxy — unified AI backend routing proxy
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=ai-proxy
Group=ai-proxy
ExecStart=/usr/local/bin/ai-proxy --config /etc/ai-proxy/ai-proxy.toml --log-format json
Restart=on-failure
RestartSec=5

# Environment
Environment=RUST_LOG=info
Environment=AI_PROXY_KEY=your-production-key
Environment=ANTHROPIC_API_KEY=sk-ant-...
# Or: EnvironmentFile=/etc/ai-proxy/env

# Graceful shutdown
KillSignal=SIGTERM
TimeoutStopSec=30

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadOnlyPaths=/etc/ai-proxy

# GPU access (required for VRAM monitoring)
SupplementaryGroups=video render

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp target/release/ai-proxy /usr/local/bin/
sudo mkdir -p /etc/ai-proxy && sudo cp ai-proxy.toml /etc/ai-proxy/
sudo systemctl daemon-reload && sudo systemctl enable --now ai-proxy
sudo journalctl -u ai-proxy -f   # logs
```

### Graceful shutdown

AIProxy handles `SIGTERM` and `Ctrl+C`:

1. Stops accepting new connections
2. Drains active requests (including in-progress streams)
3. `DemoteOnDropBody` fires — VRAM slots are released
4. Process exits cleanly

```bash
kill -SIGTERM $(pgrep ai-proxy)
# NEVER use SIGKILL — active streams break, VRAM state may become stale
```

### Reverse proxy (nginx)

```nginx
upstream ai-proxy {
    server 127.0.0.1:8800;
}

server {
    listen 443 ssl;
    server_name ai.example.com;

    ssl_certificate     /etc/ssl/ai.example.com.pem;
    ssl_certificate_key /etc/ssl/ai.example.com.key;

    proxy_buffering off;         # Required for SSE streaming
    proxy_cache off;
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://ai-proxy;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }

    location = /health {
        proxy_pass http://ai-proxy;
    }
}
```

### Docker

```dockerfile
FROM rust:1.83 AS builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/ai-proxy /usr/local/bin/
COPY ai-proxy.toml /etc/ai-proxy/

EXPOSE 8800
CMD ["ai-proxy", "--config", "/etc/ai-proxy/ai-proxy.toml", "--log-format", "json"]
```

```bash
docker build -t ai-proxy .
docker run -d -p 8800:8800 \
  -e AI_PROXY_KEY=your-key \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  --gpus all \    # Required for VRAM monitoring via nvml
  ai-proxy
```

---

## Reference

### CLI

```
USAGE: ai-proxy [OPTIONS]

OPTIONS:
  -c, --config <PATH>       Path to TOML config file [default: ai-proxy.toml]
      --log-format <FORMAT>  "json" or "pretty" [default: pretty]
  -h, --help                 Print help
  -V, --version              Print version
```

### API Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/v1/chat/completions` | Yes | Chat completion (streaming + non-streaming) |
| POST | `/v1/embeddings` | Yes | Embeddings (OpenAI-format backends only) |
| GET | `/v1/models` | Yes | List configured models |
| GET | `/status` | Yes | Proxy status: backends, models, VRAM, circuit breaker states |
| GET | `/metrics` | Yes | Per-model latency/tokens/errors, per-backend error breakdown |
| POST | `/v1/vram/reserve` | Yes | Reserve VRAM for external process |
| DELETE | `/v1/vram/reserve/{id}` | Yes | Release VRAM reservation |
| GET | `/v1/vram/reservations` | Yes | List active VRAM reservations |
| GET | `/health` | No | Readiness check for load balancers |

### Request validation (`/v1/chat/completions`)

| Field | Rule |
|-------|------|
| `model` | Required, must exist in config |
| `messages` | Required, non-empty array, each must have `role` |
| `temperature` | 0.0 to 2.0 |
| `top_p` | 0.0 to 1.0 |
| `n` | Only 1 supported |

### Error responses

All errors return OpenAI-compatible JSON:

```json
{ "error": { "message": "Human readable message", "type": "error_type" } }
```

| Error | HTTP Status | Type |
|-------|-------------|------|
| Unknown model | 404 | `not_found` |
| Invalid request | 400 | `invalid_request` |
| Rate limited | 429 | `rate_limit_exceeded` |
| Auth failed | 401 | `authentication_error` |
| Backend unavailable / CB open | 503 | `service_unavailable` |
| Backend error | 502 | `backend_error` |
| Format conversion | 500 | `conversion_error` |
| Config error | 500 | `configuration_error` |
| Privacy violation | 403 | `privacy_violation` |

Backend error bodies are **never forwarded** to clients — only the status code is exposed. This prevents leaking internal backend details.

### Configuration reference

<details>
<summary>Full config with all options</summary>

```toml
[proxy]
listen = "127.0.0.1:8800"              # Bind address
log_level = "info"                      # trace/debug/info/warn/error
api_key_env = "AI_PROXY_KEY"            # Env var for API key (omit to disable auth)
max_body_size_mb = 10                   # Max request body
request_timeout_secs = 600              # Global safety timeout
max_connections = 1024                  # Max concurrent connections

[backends.ollama]
type = "local"                          # "local" or "remote"
base_url = "http://localhost:11434"     # Must start with http:// or https://
format = "openai"                       # "openai" or "anthropic"
# api_key_env = "..."                   # Env var for backend API key
# timeout_secs = 300                    # Default: 300 (local), 60 (remote)
# max_retries = 0                       # Default: 0 (local), 2 (remote)

[backends.anthropic]
type = "remote"
base_url = "https://api.anthropic.com"
api_key_env = "ANTHROPIC_API_KEY"
format = "anthropic"
timeout_secs = 30
max_retries = 3

[models.qwen]
name = "qwen3.5:4b"                    # Model name (what clients send)
backend = "ollama"                      # Backend config key
estimated_vram_mb = 6400                # VRAM estimate (enables VRAM coordination)
default_priority = "high"               # idle/low/medium/high/critical
fallback = "claude_haiku"               # Fallback model key (on VRAM failure)
extra_body = { "num_ctx" = 8192 }       # Default request body fields

[models.claude_sonnet]
name = "claude-sonnet-4-20250514"
backend = "anthropic"

[models.claude_haiku]
name = "claude-haiku-4-5-20251001"
backend = "anthropic"

[gpu.slots.slot_0]
total_vram_mb = 24576                   # Total GPU VRAM
overhead_mb = 1024                      # OS/system overhead

[gpu.slots.permanent]
reranker = { estimated_vram_mb = 800 }  # Permanently reserved VRAM

[rate_limits.claude_sonnet]
requests_per_minute = 50                # RPM limit
tokens_per_minute = 100000              # TPM limit

[rate_limits.default]                   # Fallback for models without specific limits
requests_per_minute = 60
tokens_per_minute = 200000

[circuit_breaker]
failure_threshold = 5                   # Consecutive failures → open
recovery_timeout_secs = 30              # Wait before probe request

[routing]
default_fallback = "deny"               # "deny" or "remote"
privacy_header = "X-Privacy"            # Header for local-only enforcement
priority_header = "X-Priority"          # Header for priority override
```

</details>

### Project structure

```
src/
├── main.rs                 # CLI entry, logging, signal handling
├── lib.rs                  # Module exports
├── config.rs               # TOML config parsing & validation
├── error.rs                # Error types & HTTP status mapping
├── server.rs               # Router, AppState, GPU config extraction
├── middleware.rs            # Auth guard, request logger
├── rate_limiter.rs          # Token bucket rate limiter (RPM+TPM) + circuit breaker
├── metrics.rs              # Per-model latency/token/error tracking, per-backend errors
├── backends/
│   ├── mod.rs              # Dispatch, retry logic (RetryResult), streaming helpers
│   ├── anthropic.rs        # OpenAI <-> Anthropic conversion
│   └── ollama.rs           # OpenAI-compatible dispatch
├── routes/
│   ├── chat.rs             # /v1/chat/completions + VRAM lifecycle + fallback
│   ├── embeddings.rs       # /v1/embeddings
│   ├── health.rs           # /health, /status, /metrics
│   ├── models.rs           # /v1/models
│   └── vram.rs             # /v1/vram/reserve, /v1/vram/reservations
└── vram/
    ├── coordinator.rs       # VRAM allocation, eviction (LRU), reservations
    ├── monitor.rs           # GPU monitoring (NVIDIA nvml)
    └── ollama.rs            # Ollama model lifecycle (preload/unload)

tests/
└── integration.rs          # API endpoint integration tests
```

### Tests

```bash
cargo test    # 112 tests, ~0.01s
```

| Module | Tests | Coverage |
|--------|-------|----------|
| `anthropic.rs` | 19 | Request/response/SSE conversion, tools, images, role merging |
| `coordinator.rs` | 16 | Budget, slots, priority, acquire, eviction, reservations, reap |
| `config.rs` | 14 | Parsing, resolve, defaults, GPU config, fallback validation, extra_body |
| `rate_limiter.rs` | 18 | RPM/TPM capacity, combined checks, circuit breaker states |
| `metrics.rs` | 9 | Request/token/backend recording, latency stats, edge cases |
| `error.rs` | 9 | Status codes, error types, display messages |
| `integration.rs` | 23 | Health, models, status, validation, auth, request-id, body limits |

## License

MIT
