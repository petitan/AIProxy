#!/bin/bash
# AIProxy konténer-entrypoint.
#   - Kötelező env-változó (OLLAMA_URL) ellenőrzés (silent fallback tilos)
#   - [backends.ollama] base_url runtime csere (a Rust kód nem env-overridable
#     a base_url-re; csak ezt az egy mezőt cseréljük szakasz-specifikus sed-del,
#     az [backends.anthropic] base_url ÉRINTETLEN marad)
#   - exec ai-proxy (PID 1)

set -euo pipefail

log() { printf '[entrypoint] %s\n' "$*" >&2; }

# ─── Kötelező env-változók ───────────────────────────────────
: "${OLLAMA_URL:?A külső Ollama URL kötelező — pl. http://<host>:11434}"

# ─── Ollama base_url csere (csak a [backends.ollama] szekcióban) ───
log "Ollama backend base_url = $OLLAMA_URL"
sed -i '/^\[backends\.ollama\]/,/^\[/{s|^base_url[[:space:]]*=.*|base_url = "'"$OLLAMA_URL"'"|}' /app/ai-proxy.toml

# ─── Indítás ────────────────────────────────────────────────
log "ai-proxy indítás"
exec ai-proxy --config /app/ai-proxy.toml --log-format json
