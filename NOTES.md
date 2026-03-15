# AIProxy — Tervezési feljegyzések

## 1. Per-request vezérlés HTTP headerekkel

A kliens a request headerben jelez a proxy-nak, a proxy értelmezi és cselekszik.
Ez nem külön protokoll — az OpenAI API kiterjesztése standard HTTP headerekkel.

### Headerek

| Header | Értékek | Hatás |
|--------|---------|-------|
| `X-Privacy` | `local_only` | Csak lokális backend (Ollama/vLLM). Ha local nem elérhető → 503, NEM remote fallback. |
| `X-Priority` | `high`, `medium`, `low` | VRAM acquire prioritás override. |
| `X-Think` | `true`, `false` | Thinking/reasoning engedélyezés. `false` → Ollama-nak `think: false` injection. |

Ha nincs header → a modell config default-jai érvényesek.

### Példa

```bash
# Privát adat, thinking kikapcsolva
curl http://localhost:8800/v1/chat/completions \
  -H "X-Privacy: local_only" \
  -H "X-Think: false" \
  -d '{"model": "qwen3.5:4b", "messages": [...]}'

# Default viselkedés (config dönt)
curl http://localhost:8800/v1/chat/completions \
  -d '{"model": "qwen3.5:4b", "messages": [...]}'
```

## 2. Per-model default-ok config-ban

A TOML config-ban modell-specifikus default paraméterek adhatók meg.
A proxy forwarding előtt merge-öli a request body-ba.
A kliens header **felülírja** a config default-ot per-request szinten.

### Config minta

```toml
[models.qwen3_5_4b]
name = "qwen3.5:4b"
backend = "ollama"
estimated_vram_mb = 5700
default_priority = "high"

# Ollama-specifikus default-ok — a proxy injektálja forwarding előtt
[models.qwen3_5_4b.extra_body]
think = false
num_ctx = 16384
```

### Prioritási sorrend

1. **Kliens header** (per-request) — legmagasabb prioritás
2. **Model config `extra_body`** (per-model default) — ha nincs header
3. **Backend default** — ha nincs sem header, sem model config

### Példa: thinking vezérlés

| Kliens header | Model config | Eredmény |
|---------------|-------------|----------|
| `X-Think: false` | `think = true` | `think: false` (header nyer) |
| `X-Think: true` | `think = false` | `think: true` (header nyer) |
| nincs header | `think = false` | `think: false` (config default) |
| nincs header | nincs config | nincs injection (backend default) |

## 3. Ismert probléma: Qwen3.5 thinking + üres content

Az Ollama `/v1/chat/completions` endpoint a Qwen3.5 thinking-et a `reasoning` mezőbe teszi,
a `content`-et üresre hagyja. Ha a kliens JSON structured output-ot vár, üres választ kap.

**Megoldás:** `think = false` a model config-ban (extra_body) VAGY kliens `X-Think: false` header.
A proxy az Ollama body-ba injektálja a `think: false`-t forwarding előtt.

## 4. Jövőbeli headerek (tervezett)

| Header | Cél |
|--------|-----|
| `X-Fallback` | `allow` / `deny` — remote fallback engedélyezés |
| `X-Cost-Limit` | Per-request cost limit (token budget) |
| `X-Timeout` | Per-request timeout override |
