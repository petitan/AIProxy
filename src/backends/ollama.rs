use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use reqwest::Client;
use serde_json::{json, Value};
use std::collections::HashMap;

use crate::config::BackendConfig;
use crate::error::ProxyError;

/// Forward chat completions to Ollama using the **native** `/api/chat` endpoint.
///
/// Why native instead of `/v1/chat/completions`?
/// Ollama's OpenAI-compat layer (v0.17.x) ignores the `think` parameter in streaming mode.
/// The native API correctly supports `think: false`, which is required for Qwen3.5.
///
/// This function:
/// 1. Converts the OpenAI-format request body → Ollama native format
/// 2. POSTs to `/api/chat`
/// 3. Converts the response back to OpenAI format (both streaming and non-streaming)
pub async fn chat_completions(
    client: &Client,
    backend: &BackendConfig,
    body: Value,
    stream: bool,
) -> Result<Response, ProxyError> {
    let url = format!("{}/api/chat", backend.base_url.trim_end_matches('/'));
    let native_body = openai_to_native(body, stream);

    tracing::debug!(url = %url, stream = stream, "Forwarding chat completion to Ollama native API");

    let resp = client
        .post(&url)
        .timeout(backend.effective_timeout())
        .json(&native_body)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                ProxyError::BackendUnavailable(format!(
                    "{}: request timed out after {}s",
                    backend.base_url,
                    backend.effective_timeout().as_secs()
                ))
            } else {
                ProxyError::BackendUnavailable(format!("{}: {}", backend.base_url, e))
            }
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let error_body = resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %error_body, "Ollama backend returned error");
        // Mask internal error details — log the full body but only expose status to client
        return Ok(super::masked_error_response(status));
    }

    if stream {
        Ok(native_stream_to_openai_sse(resp, backend.effective_stream_chunk_timeout()))
    } else {
        let bytes = resp.bytes().await?;
        let native: Value = serde_json::from_slice(&bytes).map_err(|e| {
            tracing::error!(error = %e, "Failed to parse Ollama response JSON");
            ProxyError::FormatConversion(format!("Failed to parse Ollama response: {}", e))
        })?;

        // Extract token usage before converting. Missing usage is logged (not silently
        // zeroed) — a 0 here under-counts TPM and could let the rate limit be bypassed.
        if native.get("eval_count").is_none() {
            tracing::warn!(
                model = native.get("model").and_then(|v| v.as_str()).unwrap_or("unknown"),
                "Ollama response has no token usage (eval_count) — TPM accounting may undercount"
            );
        }
        let prompt_tokens = native.get("prompt_eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let completion_tokens = native.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0);

        let openai_resp = native_to_openai_response(&native);
        let out = serde_json::to_vec(&openai_resp).unwrap_or_else(|_| b"{}".to_vec());
        let mut response = Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/json; charset=utf-8")
            .body(Body::from(out))
            .expect("valid JSON response builder");
        response.extensions_mut().insert(super::TokenUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        });
        Ok(response)
    }
}

/// Forward embeddings to an OpenAI-compatible backend.
/// Embeddings stay on `/v1/embeddings` — no native API conversion needed.
pub async fn embeddings(
    client: &Client,
    backend: &BackendConfig,
    body: Value,
) -> Result<Response, ProxyError> {
    let url = format!("{}/v1/embeddings", backend.base_url.trim_end_matches('/'));

    tracing::debug!(url = %url, "Forwarding embeddings to OpenAI-compatible backend");

    let resp = client
        .post(&url)
        .timeout(backend.effective_timeout())
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                ProxyError::BackendUnavailable(format!(
                    "{}: embeddings timed out after {}s",
                    backend.base_url,
                    backend.effective_timeout().as_secs()
                ))
            } else {
                ProxyError::BackendUnavailable(format!("{}: {}", backend.base_url, e))
            }
        })?;

    let status = resp.status();
    if !status.is_success() {
        tracing::warn!(status = %status, "Embeddings backend error");
    }

    let bytes = resp.bytes().await?;
    Ok(Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY))
        .header("content-type", "application/json; charset=utf-8")
        .body(Body::from(bytes))
        .expect("valid embeddings response builder"))
}

// ─── Request conversion: OpenAI → Ollama native ───────────────────────────

/// Convert an OpenAI `/v1/chat/completions` request body to Ollama native `/api/chat` format.
///
/// Key differences:
/// - `temperature`, `top_p`, `seed`, `stop`, etc. → `options.{param}`
/// - `max_tokens` → `options.num_predict`
/// - `think` stays top-level (this is why we use native API!)
/// - `response_format: {type: "json_object"}` → `format: "json"`
fn openai_to_native(body: Value, stream: bool) -> Value {
    let obj = body.as_object();

    let mut native = json!({
        "stream": stream,
    });
    let n = native.as_object_mut().unwrap();

    // Direct top-level mappings (except messages — needs image conversion)
    for key in &["model", "think", "keep_alive"] {
        if let Some(val) = obj.and_then(|o| o.get(*key)) {
            n.insert(key.to_string(), val.clone());
        }
    }

    // Convert messages: OpenAI content arrays → Ollama native (string + images)
    if let Some(messages) = obj.and_then(|o| o.get("messages")).and_then(|v| v.as_array()) {
        // tool_call_id -> function name térkép (a tool-eredmény üzenetekhez kell,
        // mert Ollama natív a `tool_name` mezőt várja, az OpenAI csak tool_call_id-t ad).
        let mut tc_names: HashMap<String, String> = HashMap::new();
        for msg in messages {
            if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tcs {
                    if let (Some(id), Some(name)) = (
                        tc.get("id").and_then(|v| v.as_str()),
                        tc.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str()),
                    ) {
                        tc_names.insert(id.to_string(), name.to_string());
                    }
                }
            }
        }
        let native_messages: Vec<Value> =
            messages.iter().map(|msg| convert_message(msg, &tc_names)).collect();
        n.insert("messages".to_string(), json!(native_messages));
    }

    // Tools forwardolása: az Ollama natív /api/chat ugyanabban az OpenAI-alakban
    // fogadja a `tools` tömböt. Eddig hiányzott → a modell sosem kapta meg az
    // eszközöket, ezért nem emittált tool_call-t.
    if let Some(tools) = obj.and_then(|o| o.get("tools")) {
        n.insert("tools".to_string(), tools.clone());
    }

    // Build options from OpenAI params + any existing options object
    let mut options = obj
        .and_then(|o| o.get("options"))
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();

    // Map OpenAI parameter names → Ollama option names
    let direct_opts = [
        "temperature", "top_p", "seed", "stop",
        "frequency_penalty", "presence_penalty",
        "num_ctx",
    ];
    for key in &direct_opts {
        if let Some(val) = obj.and_then(|o| o.get(*key)) {
            options.entry(key.to_string()).or_insert_with(|| val.clone());
        }
    }

    // max_tokens → num_predict
    if let Some(val) = obj.and_then(|o| o.get("max_tokens")) {
        options.entry("num_predict".to_string()).or_insert_with(|| val.clone());
    }

    if !options.is_empty() {
        n.insert("options".to_string(), Value::Object(options));
    }

    // response_format → format
    if let Some(rf) = obj.and_then(|o| o.get("response_format")) {
        if rf.get("type").and_then(|t| t.as_str()) == Some("json_object") {
            n.insert("format".to_string(), json!("json"));
        }
    }

    native
}

// ─── Response conversion: Ollama native → OpenAI ──────────────────────────

/// Convert a single Ollama native tool_call to OpenAI shape.
///
/// Ollama native: `{"id":"call_xxx","function":{"index":0,"name":...,"arguments":{obj}}}`
/// OpenAI: `{"id","type":"function","function":{"name","arguments":"<json string>"}}`.
///
/// - `id`: prefer Ollama's own id (random, globally unique → no multi-turn collisions);
///   fall back to a positional `call_{i}` only if absent.
/// - `arguments`: Ollama gives a JSON object → serialized to the OpenAI JSON string.
/// - `include_index`: streaming deltas require a tool_call `index`; non-streaming omits it.
fn native_tool_call_to_openai(tc: &Value, i: usize, include_index: bool) -> Value {
    let func = tc.get("function").cloned().unwrap_or_else(|| json!({}));
    let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
    let args = func.get("arguments").cloned().unwrap_or_else(|| json!({}));
    let args_str = if let Some(s) = args.as_str() {
        s.to_string()
    } else {
        serde_json::to_string(&args).unwrap_or_else(|_| "{}".to_string())
    };
    let id = tc
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("call_{}", i));

    let mut obj = serde_json::Map::new();
    if include_index {
        let idx = func.get("index").and_then(|v| v.as_u64()).unwrap_or(i as u64);
        obj.insert("index".to_string(), json!(idx));
    }
    obj.insert("id".to_string(), json!(id));
    obj.insert("type".to_string(), json!("function"));
    obj.insert("function".to_string(), json!({"name": name, "arguments": args_str}));
    Value::Object(obj)
}

/// Ollama natív assistant-üzenet → OpenAI alak (non-streaming). Visszaadja az átalakított
/// üzenetet és hogy volt-e tool_call.
fn convert_native_message_to_openai(message: &Value) -> (Value, bool) {
    let mut out = message.clone();
    let mut has_tc = false;
    if let Some(tcs) = message.get("tool_calls").and_then(|v| v.as_array()) {
        if !tcs.is_empty() {
            has_tc = true;
            let converted: Vec<Value> = tcs
                .iter()
                .enumerate()
                .map(|(i, tc)| native_tool_call_to_openai(tc, i, false))
                .collect();
            if let Some(o) = out.as_object_mut() {
                o.insert("tool_calls".to_string(), json!(converted));
                // OpenAI: ha csak tool_call van, a content legyen null.
                let empty = o.get("content").and_then(|c| c.as_str()).map(|s| s.is_empty()).unwrap_or(true);
                if empty {
                    o.insert("content".to_string(), Value::Null);
                }
            }
        }
    }
    (out, has_tc)
}

fn native_to_openai_response(native: &Value) -> Value {
    let model = native.get("model").and_then(|v| v.as_str()).unwrap_or("unknown");
    let raw_message = native.get("message").cloned().unwrap_or(json!({"role": "assistant", "content": ""}));
    let (message, has_tool_calls) = convert_native_message_to_openai(&raw_message);
    let done_reason = native.get("done_reason").and_then(|v| v.as_str()).unwrap_or("stop");

    let prompt_tokens = native.get("prompt_eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let completion_tokens = native.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
    let created = unix_timestamp();

    let finish_reason = if has_tool_calls {
        "tool_calls"
    } else {
        match done_reason {
            "length" => "length",
            _ => "stop",
        }
    };

    json!({
        "id": format!("chatcmpl-{}", created),
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    })
}

// ─── Streaming conversion: Ollama NDJSON → OpenAI SSE ─────────────────────

/// Convert Ollama native streaming response (NDJSON) to OpenAI SSE format.
///
/// Ollama native streams newline-delimited JSON:
///   {"model":"...","message":{"role":"assistant","content":"tok"},"done":false}\n
///   {"model":"...","message":{"role":"assistant","content":""},"done":true,...}\n
///
/// We convert each line to OpenAI SSE:
///   data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"tok"}}]}\n\n
///   data: {"id":"...","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n
///   data: [DONE]\n\n
fn native_stream_to_openai_sse(
    upstream: reqwest::Response,
    chunk_timeout: std::time::Duration,
) -> Response {
    let byte_stream = upstream.bytes_stream();
    let chat_id = format!("chatcmpl-{}", unix_timestamp());
    let created = unix_timestamp();

    let sse_stream = async_stream::stream! {
        use futures::StreamExt;
        let mut byte_stream = Box::pin(byte_stream);
        let mut buffer = Vec::new();
        let mut first_chunk = true;
        let mut model_name = String::from("unknown");
        // Ollama's `done_reason` is "stop" even when the turn ended on a tool call, so we
        // track tool emission ourselves to set finish_reason="tool_calls" on the final chunk.
        let mut has_tool_calls = false;

        loop {
            match tokio::time::timeout(chunk_timeout, byte_stream.next()).await {
                Ok(Some(Ok(bytes))) => {
                    buffer.extend_from_slice(&bytes);

                    // Process complete NDJSON lines
                    while let Some(newline_pos) = buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = buffer.drain(..=newline_pos).collect();
                        let line = String::from_utf8_lossy(&line_bytes);
                        let line = line.trim();

                        if line.is_empty() {
                            continue;
                        }

                        let native: Value = match serde_json::from_str(line) {
                            Ok(v) => v,
                            Err(e) => {
                                tracing::warn!(error = %e, line = %line, "Failed to parse Ollama NDJSON line");
                                continue;
                            }
                        };

                        if let Some(m) = native.get("model").and_then(|v| v.as_str()) {
                            model_name = m.to_string();
                        }

                        let done = native.get("done").and_then(|v| v.as_bool()).unwrap_or(false);

                        if done {
                            // Final chunk: finish_reason + [DONE]
                            let done_reason = native.get("done_reason")
                                .and_then(|v| v.as_str())
                                .unwrap_or("stop");
                            let finish_reason = if has_tool_calls {
                                "tool_calls"
                            } else {
                                match done_reason {
                                    "length" => "length",
                                    _ => "stop",
                                }
                            };

                            let chunk = json!({
                                "id": &chat_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": &model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason
                                }]
                            });

                            yield Ok::<_, std::io::Error>(
                                bytes::Bytes::from(format!("data: {}\n\n", chunk))
                            );
                            yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                            return;
                        }

                        // Content chunk
                        let content = native.get("message")
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str())
                            .unwrap_or("");

                        let mut delta = serde_json::Map::new();
                        if first_chunk {
                            delta.insert("role".to_string(), json!("assistant"));
                            first_chunk = false;
                        }
                        if !content.is_empty() {
                            delta.insert("content".to_string(), json!(content));
                        }

                        // Tool calls — Ollama emits them complete (full arguments) in a single
                        // non-done chunk, not as incremental deltas. Forward as OpenAI tool_call
                        // deltas (with index) so streaming agents receive the function call.
                        if let Some(tcs) = native
                            .get("message")
                            .and_then(|m| m.get("tool_calls"))
                            .and_then(|v| v.as_array())
                        {
                            if !tcs.is_empty() {
                                let converted: Vec<Value> = tcs
                                    .iter()
                                    .enumerate()
                                    .map(|(i, tc)| native_tool_call_to_openai(tc, i, true))
                                    .collect();
                                delta.insert("tool_calls".to_string(), json!(converted));
                                has_tool_calls = true;
                            }
                        }

                        // Skip empty chunks (no role, no content, no tool_calls)
                        if delta.is_empty() {
                            continue;
                        }

                        let chunk = json!({
                            "id": &chat_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": &model_name,
                            "choices": [{
                                "index": 0,
                                "delta": Value::Object(delta),
                                "finish_reason": null
                            }]
                        });

                        yield Ok::<_, std::io::Error>(
                            bytes::Bytes::from(format!("data: {}\n\n", chunk))
                        );
                    }
                }
                Ok(Some(Err(e))) => {
                    tracing::error!(error = %e, "Error reading Ollama native stream");
                    yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                    break;
                }
                Ok(None) => {
                    // Stream ended — process remaining buffer
                    if !buffer.is_empty() {
                        let remaining = String::from_utf8_lossy(&buffer).trim().to_string();
                        if let Ok(native) = serde_json::from_str::<Value>(&remaining) {
                            if native.get("done").and_then(|v| v.as_bool()) == Some(true) {
                                let done_reason = native.get("done_reason")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("stop");
                                let finish_reason = if has_tool_calls {
                                    "tool_calls"
                                } else {
                                    match done_reason {
                                        "length" => "length",
                                        _ => "stop",
                                    }
                                };
                                let chunk = json!({
                                    "id": &chat_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": &model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish_reason
                                    }]
                                });
                                yield Ok::<_, std::io::Error>(
                                    bytes::Bytes::from(format!("data: {}\n\n", chunk))
                                );
                            }
                        }
                    }
                    yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                    break;
                }
                Err(_elapsed) => {
                    tracing::error!(
                        timeout_secs = chunk_timeout.as_secs(),
                        "Ollama native stream chunk timed out"
                    );
                    yield Ok(bytes::Bytes::from("data: [DONE]\n\n"));
                    break;
                }
            }
        }
    };

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(sse_stream))
        .unwrap()
}

/// Convert a single OpenAI message to Ollama native format.
///
/// OpenAI allows `content` to be either a string or an array of content parts:
///   [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:image/png;base64,ABC"}}]
///
/// Ollama native expects:
///   {"role":"user", "content":"...", "images":["ABC"]}
fn convert_message(msg: &Value, tc_names: &HashMap<String, String>) -> Value {
    let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
    let content = msg.get("content");

    // Tool-eredmény üzenet: OpenAI {role:tool, tool_call_id, content}
    //   -> Ollama natív {role:tool, content, tool_name}
    if role == "tool" {
        let text = content.and_then(|c| c.as_str()).unwrap_or("");
        let mut m = json!({"role": "tool", "content": text});
        if let Some(name) = msg
            .get("tool_call_id")
            .and_then(|v| v.as_str())
            .and_then(|id| tc_names.get(id))
        {
            m["tool_name"] = json!(name);
        }
        return m;
    }

    // Assistant tool-hívásokkal: OpenAI tool_calls (arguments JSON-string)
    //   -> Ollama natív tool_calls (arguments objektum, id/type nélkül)
    if role == "assistant" {
        if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            let native_tcs: Vec<Value> = tcs
                .iter()
                .map(|tc| {
                    let func = tc.get("function").cloned().unwrap_or_else(|| json!({}));
                    let name = func.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    let args_obj = match func.get("arguments") {
                        Some(Value::String(s)) => {
                            serde_json::from_str::<Value>(s).unwrap_or_else(|_| json!({}))
                        }
                        Some(v) => v.clone(),
                        None => json!({}),
                    };
                    json!({"function": {"name": name, "arguments": args_obj}})
                })
                .collect();
            let text = content.and_then(|c| c.as_str()).unwrap_or("");
            return json!({"role": "assistant", "content": text, "tool_calls": native_tcs});
        }
    }

    // If content is a string (or missing), pass through as-is
    let content_array = match content.and_then(|c| c.as_array()) {
        Some(arr) => arr,
        None => return msg.clone(),
    };

    // Extract text parts and image base64 data from content array
    let mut text_parts: Vec<String> = Vec::new();
    let mut images: Vec<String> = Vec::new();

    for part in content_array {
        let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match part_type {
            "text" => {
                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "image_url" => {
                if let Some(url) = part
                    .get("image_url")
                    .and_then(|iu| iu.get("url"))
                    .and_then(|u| u.as_str())
                {
                    // Strip "data:image/...;base64," prefix if present
                    let base64_data = if let Some(pos) = url.find(";base64,") {
                        &url[pos + 8..]
                    } else {
                        url
                    };
                    images.push(base64_data.to_string());
                }
            }
            _ => {
                tracing::debug!(part_type = %part_type, "Skipping unknown content part type");
            }
        }
    }

    let mut native_msg = json!({
        "role": msg.get("role").and_then(|r| r.as_str()).unwrap_or("user"),
        "content": text_parts.join("\n"),
    });

    if !images.is_empty() {
        native_msg["images"] = json!(images);
    }

    native_msg
}

fn unix_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
