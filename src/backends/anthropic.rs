use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use reqwest::Client;
use serde_json::{json, Value};

use crate::config::BackendConfig;
use crate::error::ProxyError;

const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u64 = 4096;

// ═══════════════════════════════════════════════════════════════════
//  OpenAI → Anthropic REQUEST conversion
// ═══════════════════════════════════════════════════════════════════

fn convert_request(openai_body: &Value) -> Result<Value, ProxyError> {
    let model = openai_body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ProxyError::BadRequest("Missing 'model' field".into()))?;

    let messages = openai_body
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| ProxyError::BadRequest("Missing 'messages' array".into()))?;

    let mut system_parts: Vec<String> = Vec::new();
    let mut anthropic_messages: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        match role {
            "system" => {
                if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                    system_parts.push(content.to_string());
                }
            }
            "assistant" => {
                anthropic_messages.push(convert_assistant_message(msg)?);
            }
            "tool" => {
                // OpenAI tool result → Anthropic user message with tool_result content block
                anthropic_messages.push(convert_tool_result_message(msg)?);
            }
            // "user" and anything else
            _ => {
                anthropic_messages.push(convert_user_message(msg)?);
            }
        }
    }

    // Anthropic requires alternating user/assistant. Consecutive same-role messages
    // need merging. Tool results (mapped to "user") may collide with the next user message.
    let anthropic_messages = merge_consecutive_roles(anthropic_messages);

    let max_tokens = openai_body
        .get("max_tokens")
        .or_else(|| openai_body.get("max_completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_MAX_TOKENS);

    let stream = openai_body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let mut request = json!({
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": max_tokens,
        "stream": stream,
    });

    if !system_parts.is_empty() {
        request["system"] = Value::String(system_parts.join("\n\n"));
    }

    // Tools definition: OpenAI → Anthropic
    if let Some(tools) = openai_body.get("tools").and_then(|v| v.as_array()) {
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .filter_map(|t| convert_tool_definition(t))
            .collect();
        if !anthropic_tools.is_empty() {
            request["tools"] = Value::Array(anthropic_tools);
        }
    }

    // tool_choice
    if let Some(choice) = openai_body.get("tool_choice") {
        request["tool_choice"] = convert_tool_choice(choice);
    }

    // Forward optional parameters
    if let Some(temp) = openai_body.get("temperature") {
        request["temperature"] = temp.clone();
    }
    if let Some(top_p) = openai_body.get("top_p") {
        request["top_p"] = top_p.clone();
    }
    if let Some(top_k) = openai_body.get("top_k") {
        request["top_k"] = top_k.clone();
    }
    if let Some(stop) = openai_body.get("stop") {
        request["stop_sequences"] = match stop {
            Value::String(s) => json!([s]),
            Value::Array(_) => stop.clone(),
            _ => stop.clone(),
        };
    }

    // response_format → Anthropic system prompt instruction
    // OpenAI: {"type":"json_object"} or {"type":"json_schema","json_schema":{...}}
    // Anthropic: no native json_mode — system prompt instruction only (no prefill,
    // because prefill creates alternating-role issues and fragile string injection)
    if let Some(rf) = openai_body.get("response_format") {
        let rf_type = rf.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let json_instruction = match rf_type {
            "json_object" => {
                Some("You must respond with valid JSON only. No markdown, no code fences, no explanation — output a single JSON object.".to_string())
            }
            "json_schema" => {
                let schema = rf.get("json_schema").cloned().unwrap_or(json!({}));
                let schema_str = serde_json::to_string_pretty(&schema).unwrap_or_default();
                Some(format!(
                    "You must respond with valid JSON that conforms to this schema:\n{}\nNo markdown, no code fences, no explanation — output a single JSON object.",
                    schema_str
                ))
            }
            _ => None,
        };
        if let Some(instruction) = json_instruction {
            match request.get("system").and_then(|v| v.as_str()).map(String::from) {
                Some(existing) => {
                    request["system"] = Value::String(format!("{}\n\n{}", existing, instruction));
                }
                None => {
                    request["system"] = Value::String(instruction);
                }
            }
        }
    }

    // Parameters that Anthropic does NOT support — log warning, don't error
    if openai_body.get("frequency_penalty").is_some() {
        tracing::debug!("'frequency_penalty' is not supported by Anthropic, ignored");
    }
    if openai_body.get("presence_penalty").is_some() {
        tracing::debug!("'presence_penalty' is not supported by Anthropic, ignored");
    }
    if openai_body.get("logprobs").is_some() {
        tracing::debug!("'logprobs' is not supported by Anthropic, ignored");
    }
    if openai_body.get("seed").is_some() {
        tracing::debug!("'seed' is not supported by Anthropic, ignored");
    }
    if openai_body.get("logit_bias").is_some() {
        tracing::debug!("'logit_bias' is not supported by Anthropic, ignored");
    }

    // user → metadata.user_id (Anthropic tracking)
    if let Some(user) = openai_body.get("user").and_then(|v| v.as_str()) {
        request["metadata"] = json!({"user_id": user});
    }

    Ok(request)
}

/// OpenAI tool definition → Anthropic tool definition
/// OpenAI: {"type":"function","function":{"name":"x","description":"y","parameters":{...}}}
/// Anthropic: {"name":"x","description":"y","input_schema":{...}}
fn convert_tool_definition(openai_tool: &Value) -> Option<Value> {
    let func = openai_tool.get("function")?;
    let name = func.get("name")?.as_str()?;
    let mut tool = json!({ "name": name });

    if let Some(desc) = func.get("description") {
        tool["description"] = desc.clone();
    }

    // parameters → input_schema
    if let Some(params) = func.get("parameters") {
        tool["input_schema"] = params.clone();
    } else {
        tool["input_schema"] = json!({"type": "object", "properties": {}});
    }

    Some(tool)
}

/// OpenAI tool_choice → Anthropic tool_choice
fn convert_tool_choice(choice: &Value) -> Value {
    match choice {
        Value::String(s) => match s.as_str() {
            "none" => json!({"type": "none"}),
            "auto" => json!({"type": "auto"}),
            "required" => json!({"type": "any"}),
            _ => json!({"type": "auto"}),
        },
        Value::Object(obj) => {
            // {"type":"function","function":{"name":"x"}} → {"type":"tool","name":"x"}
            if let Some(func) = obj.get("function") {
                if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                    return json!({"type": "tool", "name": name});
                }
            }
            json!({"type": "auto"})
        }
        _ => json!({"type": "auto"}),
    }
}

/// Convert a user message (may contain text, images, etc.)
fn convert_user_message(msg: &Value) -> Result<Value, ProxyError> {
    let content = msg.get("content").cloned().unwrap_or(Value::Null);

    let anthropic_content = match &content {
        Value::String(_) => content.clone(),
        Value::Array(parts) => {
            let converted: Vec<Value> = parts
                .iter()
                .filter_map(|part| convert_content_part(part).ok())
                .collect();
            Value::Array(converted)
        }
        Value::Null => Value::String(String::new()),
        _ => content.clone(),
    };

    Ok(json!({
        "role": "user",
        "content": anthropic_content,
    }))
}

/// Convert an assistant message (may contain text + tool_calls)
fn convert_assistant_message(msg: &Value) -> Result<Value, ProxyError> {
    let mut content_blocks: Vec<Value> = Vec::new();

    // Text content
    if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            content_blocks.push(json!({"type": "text", "text": text}));
        }
    } else if let Some(parts) = msg.get("content").and_then(|v| v.as_array()) {
        for part in parts {
            if let Ok(converted) = convert_content_part(part) {
                content_blocks.push(converted);
            }
        }
    }

    // Tool calls → tool_use content blocks
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
        for tc in tool_calls {
            if let Some(block) = convert_tool_call_to_tool_use(tc) {
                content_blocks.push(block);
            }
        }
    }

    if content_blocks.is_empty() {
        content_blocks.push(json!({"type": "text", "text": ""}));
    }

    Ok(json!({
        "role": "assistant",
        "content": content_blocks,
    }))
}

/// OpenAI tool_call → Anthropic tool_use content block
/// OpenAI: {"id":"call_x","type":"function","function":{"name":"y","arguments":"{...}"}}
/// Anthropic: {"type":"tool_use","id":"call_x","name":"y","input":{...}}
fn convert_tool_call_to_tool_use(tc: &Value) -> Option<Value> {
    let id = tc.get("id")?.as_str()?;
    let func = tc.get("function")?;
    let name = func.get("name")?.as_str()?;
    let arguments = func.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
    let input: Value = serde_json::from_str(arguments).unwrap_or(json!({}));

    Some(json!({
        "type": "tool_use",
        "id": id,
        "name": name,
        "input": input,
    }))
}

/// OpenAI tool result message → Anthropic user message with tool_result block
/// OpenAI: {"role":"tool","tool_call_id":"call_x","content":"result text"}
/// Anthropic: {"role":"user","content":[{"type":"tool_result","tool_use_id":"call_x","content":"result text"}]}
fn convert_tool_result_message(msg: &Value) -> Result<Value, ProxyError> {
    let tool_use_id = msg
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let content = msg.get("content").cloned().unwrap_or(Value::String(String::new()));

    // Content can be string or structured
    let tool_result_content = match &content {
        Value::String(_) => content.clone(),
        Value::Array(parts) => {
            // Convert content parts
            let converted: Vec<Value> = parts
                .iter()
                .filter_map(|p| convert_content_part(p).ok())
                .collect();
            Value::Array(converted)
        }
        _ => content.clone(),
    };

    let mut block = json!({
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": tool_result_content,
    });

    // If the tool result indicates an error
    if msg.get("is_error").and_then(|v| v.as_bool()).unwrap_or(false) {
        block["is_error"] = json!(true);
    }

    Ok(json!({
        "role": "user",
        "content": [block],
    }))
}

fn convert_content_part(part: &Value) -> Result<Value, ProxyError> {
    let part_type = part.get("type").and_then(|v| v.as_str()).unwrap_or("text");

    match part_type {
        "text" => Ok(json!({
            "type": "text",
            "text": part.get("text").cloned().unwrap_or(Value::String(String::new())),
        })),
        "image_url" => {
            if let Some(url_obj) = part.get("image_url") {
                let url = url_obj
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if let Some(base64_data) = url.strip_prefix("data:") {
                    if let Some((media_info, data)) = base64_data.split_once(',') {
                        let media_type = media_info.trim_end_matches(";base64");
                        return Ok(json!({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            }
                        }));
                    }
                }
                Ok(json!({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url,
                    }
                }))
            } else {
                Ok(part.clone())
            }
        }
        _ => Ok(part.clone()),
    }
}

/// Anthropic requires strictly alternating user/assistant roles.
/// Merge consecutive messages with the same role into one by combining content blocks.
fn merge_consecutive_roles(messages: Vec<Value>) -> Vec<Value> {
    let mut merged: Vec<Value> = Vec::new();

    for msg in messages {
        let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("");
        let prev_role = merged
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if role == prev_role {
            // Merge into previous message
            let prev = merged.last_mut().unwrap();
            let prev_content = ensure_content_array(prev.get("content").cloned());
            let curr_content = ensure_content_array(msg.get("content").cloned());

            let mut combined = prev_content;
            combined.extend(curr_content);
            prev["content"] = Value::Array(combined);
        } else {
            merged.push(msg);
        }
    }

    merged
}

/// Normalize content to array of content blocks
fn ensure_content_array(content: Option<Value>) -> Vec<Value> {
    match content {
        Some(Value::Array(arr)) => arr,
        Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
        Some(other) => vec![other],
        None => vec![],
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Anthropic → OpenAI RESPONSE conversion (non-streaming)
// ═══════════════════════════════════════════════════════════════════

fn convert_response(anthropic_resp: &Value, model: &str) -> Value {
    let content_blocks = anthropic_resp
        .get("content")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    // Collect text (skip thinking blocks)
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<Value> = Vec::new();
    let mut tool_call_index: usize = 0;

    for block in &content_blocks {
        let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    text_parts.push(text.to_string());
                }
            }
            "tool_use" => {
                // Anthropic: {"type":"tool_use","id":"x","name":"y","input":{...}}
                // OpenAI:    {"id":"x","type":"function","function":{"name":"y","arguments":"{...}"}}
                if let (Some(id), Some(name)) = (
                    block.get("id").and_then(|v| v.as_str()),
                    block.get("name").and_then(|v| v.as_str()),
                ) {
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    let arguments = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments,
                        },
                        "index": tool_call_index,
                    }));
                    tool_call_index += 1;
                }
            }
            "thinking" => {
                // Anthropic thinking blocks — no OpenAI equivalent, skip
            }
            _ => {}
        }
    }

    let text = text_parts.join("");

    let stop_reason = anthropic_resp
        .get("stop_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");

    let finish_reason = map_stop_reason(stop_reason);

    let usage = anthropic_resp.get("usage").cloned().unwrap_or(json!({}));
    let input_tokens = usage.get("input_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
    let output_tokens = usage.get("output_tokens").and_then(|v| v.as_u64()).unwrap_or(0);

    let mut message = json!({
        "role": "assistant",
        "content": if text.is_empty() { Value::Null } else { Value::String(text) },
    });

    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls);
    }

    json!({
        "id": format!("chatcmpl-{}", uuid::Uuid::new_v4()),
        "object": "chat.completion",
        "created": chrono::Utc::now().timestamp(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    })
}

fn map_stop_reason(stop_reason: &str) -> &'static str {
    match stop_reason {
        "end_turn" | "stop_sequence" => "stop",
        "max_tokens" => "length",
        "tool_use" => "tool_calls",
        _ => "stop",
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Anthropic SSE → OpenAI SSE conversion (streaming)
// ═══════════════════════════════════════════════════════════════════

/// Mutable state carried across SSE events within a single stream
struct StreamState {
    stream_id: String,
    model: String,
    /// Sequential tool_call index (only counts tool_use blocks, NOT text blocks)
    tool_call_index: usize,
    /// OpenAI tool_call index for the CURRENT tool_use block being streamed
    current_tool_call_index: usize,
    /// Type of the current content block being streamed
    current_block_type: String,
    /// Usage tracking for stream_options.include_usage
    input_tokens: u64,
    output_tokens: u64,
}

impl StreamState {
    fn new(model: String) -> Self {
        Self {
            stream_id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            model,
            tool_call_index: 0,
            current_tool_call_index: 0,
            current_block_type: String::new(),
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    fn make_chunk(&self, choices: Value) -> String {
        let chunk = json!({
            "id": &self.stream_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": &self.model,
            "choices": choices,
        });
        format!("data: {}\n\n", chunk)
    }
}

/// Convert a single Anthropic SSE event. May produce 0 or 1 output chunks.
fn convert_sse_event(state: &mut StreamState, event_type: &str, data: &Value) -> Option<String> {
    match event_type {
        "message_start" => {
            // Initial chunk with role
            Some(state.make_chunk(json!([{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": null,
            }])))
        }

        "content_block_start" => {
            let block = data.get("content_block")?;
            let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            state.current_block_type = block_type.to_string();

            match block_type {
                "tool_use" => {
                    let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");

                    // Sequential tool_call index: only counts tool_use blocks
                    let tool_index = state.tool_call_index;
                    state.current_tool_call_index = tool_index;
                    state.tool_call_index += 1;

                    Some(state.make_chunk(json!([{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": tool_index,
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": "",
                                }
                            }]
                        },
                        "finish_reason": null,
                    }])))
                }
                "thinking" => None,
                _ => None,
            }
        }

        "content_block_delta" => {
            let delta = data.get("delta")?;
            let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");

            match delta_type {
                "text_delta" => {
                    let text = delta.get("text").and_then(|v| v.as_str())?;
                    Some(state.make_chunk(json!([{
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": null,
                    }])))
                }
                "input_json_delta" => {
                    let partial = delta.get("partial_json").and_then(|v| v.as_str())?;
                    let tool_index = state.current_tool_call_index;

                    Some(state.make_chunk(json!([{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": tool_index,
                                "function": {
                                    "arguments": partial,
                                }
                            }]
                        },
                        "finish_reason": null,
                    }])))
                }
                "thinking_delta" => None, // Skip thinking content
                _ => None,
            }
        }

        "content_block_stop" => None, // No OpenAI equivalent needed

        "message_delta" => {
            let delta = data.get("delta")?;
            let stop_reason = delta.get("stop_reason").and_then(|v| v.as_str())?;
            let finish_reason = map_stop_reason(stop_reason);

            Some(state.make_chunk(json!([{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }])))
        }

        // message_stop is handled by the outer loop to ensure correct
        // ordering: usage chunk BEFORE [DONE] (OpenAI spec)
        "message_stop" => None,

        // ping, error, etc.
        _ => None,
    }
}

// ═══════════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════════

pub async fn chat_completions(
    client: &Client,
    backend: &BackendConfig,
    body: Value,
    stream: bool,
) -> Result<Response, ProxyError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    // stream_options.include_usage — emit a final chunk with usage stats
    let include_usage = body
        .get("stream_options")
        .and_then(|so| so.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let anthropic_request = convert_request(&body)?;

    let api_key = backend
        .api_key_env
        .as_ref()
        .and_then(|env_var| std::env::var(env_var).ok())
        .ok_or_else(|| {
            ProxyError::Config(format!(
                "API key env var '{}' not set",
                backend.api_key_env.as_deref().unwrap_or("(none)")
            ))
        })?;

    let url = format!(
        "{}/v1/messages",
        backend.base_url.trim_end_matches('/')
    );

    tracing::debug!(url = %url, stream = stream, model = %model, "Forwarding to Anthropic");

    let resp = client
        .post(&url)
        .timeout(backend.effective_timeout())
        .header("x-api-key", &api_key)
        .header("anthropic-version", ANTHROPIC_API_VERSION)
        .header("content-type", "application/json; charset=utf-8")
        .json(&anthropic_request)
        .send()
        .await
        .map_err(|e| {
            if e.is_timeout() {
                ProxyError::BackendUnavailable(format!(
                    "Anthropic: request timed out after {}s",
                    backend.effective_timeout().as_secs()
                ))
            } else {
                ProxyError::BackendUnavailable(format!("Anthropic: {}", e))
            }
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let error_body = resp.text().await.unwrap_or_default();
        tracing::warn!(status = %status, body = %error_body, "Anthropic returned error");
        // Mask internal error details — log the full body but only expose status to client
        let error_json = json!({
            "error": {
                "message": format!("Backend error ({})", status),
                "type": "backend_error",
            }
        });
        return Ok(build_json_response(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            &error_json,
        ));
    }

    if stream {
        stream_anthropic_response(resp, &model, include_usage, backend.effective_timeout()).await
    } else {
        let anthropic_json: Value = resp.json().await.map_err(|e| {
            ProxyError::FormatConversion(format!("Failed to parse Anthropic response: {}", e))
        })?;

        // Extract token usage before converting
        let usage = anthropic_json.get("usage");
        let input_tokens = usage.and_then(|u| u.get("input_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);
        let output_tokens = usage.and_then(|u| u.get("output_tokens")).and_then(|v| v.as_u64()).unwrap_or(0);

        let openai_json = convert_response(&anthropic_json, &model);
        let mut response = build_json_response(StatusCode::OK, &openai_json);
        response.extensions_mut().insert(super::TokenUsage {
            prompt_tokens: input_tokens,
            completion_tokens: output_tokens,
            total_tokens: input_tokens + output_tokens,
        });
        Ok(response)
    }
}

/// Build a JSON response without panic — falls back to plain text on serialization failure
fn build_json_response(status: StatusCode, body: &Value) -> Response {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|e| {
        format!(r#"{{"error":{{"message":"Serialization failed: {}"}}}}"#, e).into_bytes()
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json; charset=utf-8")
        .body(Body::from(bytes))
        .expect("valid JSON response builder")
}

async fn stream_anthropic_response(
    resp: reqwest::Response,
    model: &str,
    include_usage: bool,
    chunk_timeout: std::time::Duration,
) -> Result<Response, ProxyError> {
    let model = model.to_string();
    let byte_stream = resp.bytes_stream();

    let converted_stream = async_stream::stream! {
        use futures::StreamExt;

        let mut state = StreamState::new(model);
        let mut raw_buf: Vec<u8> = Vec::new();
        let mut current_event_type = String::new();
        let mut byte_stream = Box::pin(byte_stream);
        let mut got_done = false;

        loop {
            let chunk_result = match tokio::time::timeout(chunk_timeout, byte_stream.next()).await {
                Ok(Some(result)) => result,
                Ok(None) => break, // Stream ended normally
                Err(_elapsed) => {
                    tracing::error!(
                        timeout_secs = chunk_timeout.as_secs(),
                        "Anthropic stream chunk timed out — no data received"
                    );
                    // Emit stop + [DONE] so the client doesn't hang
                    let stop_chunk = state.make_chunk(json!([{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]));
                    yield Ok::<_, std::io::Error>(bytes::Bytes::from(stop_chunk));
                    yield Ok::<_, std::io::Error>(bytes::Bytes::from("data: [DONE]\n\n".to_string()));
                    got_done = true;
                    break;
                }
            };

            match chunk_result {
                Ok(bytes) => {
                    raw_buf.extend_from_slice(&bytes);

                    loop {
                        let newline_pos = match raw_buf.iter().position(|&b| b == b'\n') {
                            Some(pos) => pos,
                            None => break,
                        };

                        let mut line_end = newline_pos;
                        if line_end > 0 && raw_buf[line_end - 1] == b'\r' {
                            line_end -= 1;
                        }
                        let line_bytes = &raw_buf[..line_end];

                        if let Ok(line) = std::str::from_utf8(line_bytes) {
                            if let Some(event) = line.strip_prefix("event: ") {
                                current_event_type = event.trim().to_string();
                            } else if let Some(data_str) = line.strip_prefix("data: ") {
                                if let Ok(data) = serde_json::from_str::<Value>(data_str) {
                                    // Track usage from message_start and message_delta
                                    if current_event_type == "message_start" {
                                        if let Some(usage) = data.pointer("/message/usage") {
                                            state.input_tokens = usage.get("input_tokens")
                                                .and_then(|v| v.as_u64()).unwrap_or(0);
                                        }
                                    }
                                    if current_event_type == "message_delta" {
                                        if let Some(usage) = data.get("usage") {
                                            state.output_tokens = usage.get("output_tokens")
                                                .and_then(|v| v.as_u64()).unwrap_or(0);
                                        }
                                    }

                                    if current_event_type == "message_stop" {
                                        // Emit usage chunk BEFORE [DONE] (OpenAI spec)
                                        if include_usage {
                                            let usage_json = json!({
                                                "id": &state.stream_id,
                                                "object": "chat.completion.chunk",
                                                "created": chrono::Utc::now().timestamp(),
                                                "model": &state.model,
                                                "choices": [],
                                                "usage": {
                                                    "prompt_tokens": state.input_tokens,
                                                    "completion_tokens": state.output_tokens,
                                                    "total_tokens": state.input_tokens + state.output_tokens,
                                                }
                                            });
                                            yield Ok::<_, std::io::Error>(bytes::Bytes::from(format!("data: {}\n\n", usage_json)));
                                        }
                                        yield Ok::<_, std::io::Error>(bytes::Bytes::from("data: [DONE]\n\n".to_string()));
                                        got_done = true;
                                    } else if let Some(converted) = convert_sse_event(&mut state, &current_event_type, &data) {
                                        yield Ok::<_, std::io::Error>(bytes::Bytes::from(converted));
                                    }
                                }
                            }
                        }

                        // O(1) split instead of O(n) copy
                        let _ = raw_buf.drain(..newline_pos + 1);
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Error reading Anthropic stream");
                    let error_chunk = state.make_chunk(json!([{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }]));
                    yield Ok::<_, std::io::Error>(bytes::Bytes::from(error_chunk));
                    yield Ok::<_, std::io::Error>(bytes::Bytes::from("data: [DONE]\n\n".to_string()));
                    got_done = true;
                    break;
                }
            }
        }

        // Premature stream end — no message_stop received
        if !got_done {
            tracing::warn!("Anthropic stream ended without message_stop");
            if include_usage {
                let usage_json = json!({
                    "id": &state.stream_id,
                    "object": "chat.completion.chunk",
                    "created": chrono::Utc::now().timestamp(),
                    "model": &state.model,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": state.input_tokens,
                        "completion_tokens": state.output_tokens,
                        "total_tokens": state.input_tokens + state.output_tokens,
                    }
                });
                yield Ok::<_, std::io::Error>(bytes::Bytes::from(format!("data: {}\n\n", usage_json)));
            }
            let stop_chunk = state.make_chunk(json!([{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }]));
            yield Ok::<_, std::io::Error>(bytes::Bytes::from(stop_chunk));
            yield Ok::<_, std::io::Error>(bytes::Bytes::from("data: [DONE]\n\n".to_string()));
        }
    };

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "text/event-stream; charset=utf-8")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(converted_stream))
        .expect("valid SSE response builder"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── convert_request tests ──

    #[test]
    fn test_basic_request_conversion() {
        let openai = json!({
            "model": "claude-sonnet-4-20250514",
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        });
        let result = convert_request(&openai).unwrap();
        assert_eq!(result["model"], "claude-sonnet-4-20250514");
        assert_eq!(result["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(result["stream"], false);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"], "Hello");
    }

    #[test]
    fn test_system_message_extraction() {
        let openai = json!({
            "model": "test",
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hi"}
            ]
        });
        let result = convert_request(&openai).unwrap();
        assert_eq!(result["system"], "You are helpful\n\nBe concise");
        // System messages should not appear in messages array
        assert_eq!(result["messages"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_max_tokens_variants() {
        // max_tokens
        let openai = json!({"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100});
        assert_eq!(convert_request(&openai).unwrap()["max_tokens"], 100);

        // max_completion_tokens (OpenAI alias)
        let openai = json!({"model": "test", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 200});
        assert_eq!(convert_request(&openai).unwrap()["max_tokens"], 200);
    }

    #[test]
    fn test_tool_definition_conversion() {
        let openai_tool = json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        });
        let result = convert_tool_definition(&openai_tool).unwrap();
        assert_eq!(result["name"], "get_weather");
        assert_eq!(result["description"], "Get weather");
        assert_eq!(result["input_schema"]["type"], "object");
        // Should NOT have "parameters" key
        assert!(result.get("parameters").is_none());
    }

    #[test]
    fn test_tool_choice_conversion() {
        assert_eq!(convert_tool_choice(&json!("none"))["type"], "none");
        assert_eq!(convert_tool_choice(&json!("auto"))["type"], "auto");
        assert_eq!(convert_tool_choice(&json!("required"))["type"], "any");

        let specific = json!({"type": "function", "function": {"name": "get_weather"}});
        let result = convert_tool_choice(&specific);
        assert_eq!(result["type"], "tool");
        assert_eq!(result["name"], "get_weather");
    }

    #[test]
    fn test_tool_result_message_conversion() {
        let tool_msg = json!({
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "72°F"
        });
        let result = convert_tool_result_message(&tool_msg).unwrap();
        assert_eq!(result["role"], "user");
        let content = result["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_result");
        assert_eq!(content[0]["tool_use_id"], "call_123");
        assert_eq!(content[0]["content"], "72°F");
    }

    #[test]
    fn test_assistant_message_with_tool_calls() {
        let msg = json!({
            "role": "assistant",
            "content": "Let me check",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "search", "arguments": "{\"q\":\"rust\"}"}
            }]
        });
        let result = convert_assistant_message(&msg).unwrap();
        assert_eq!(result["role"], "assistant");
        let blocks = result["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2); // text + tool_use
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["name"], "search");
        assert_eq!(blocks[1]["input"]["q"], "rust");
    }

    #[test]
    fn test_consecutive_role_merging() {
        let messages = vec![
            json!({"role": "user", "content": "First"}),
            json!({"role": "user", "content": "Second"}),
            json!({"role": "assistant", "content": "Reply"}),
        ];
        let merged = merge_consecutive_roles(messages);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0]["role"], "user");
        // Content should be merged into array
        let content = merged[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
    }

    #[test]
    fn test_response_format_json_object() {
        let openai = json!({
            "model": "test",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"}
        });
        let result = convert_request(&openai).unwrap();
        let system = result["system"].as_str().unwrap();
        assert!(system.contains("valid JSON"));
    }

    // ── convert_response tests ──

    #[test]
    fn test_basic_response_conversion() {
        let anthropic = json!({
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5}
        });
        let result = convert_response(&anthropic, "claude-test");
        assert_eq!(result["object"], "chat.completion");
        assert_eq!(result["model"], "claude-test");
        assert_eq!(result["choices"][0]["message"]["content"], "Hello!");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
        assert_eq!(result["usage"]["prompt_tokens"], 10);
        assert_eq!(result["usage"]["completion_tokens"], 5);
        assert_eq!(result["usage"]["total_tokens"], 15);
    }

    #[test]
    fn test_response_with_tool_use() {
        let anthropic = json!({
            "content": [
                {"type": "text", "text": "Checking..."},
                {"type": "tool_use", "id": "call_1", "name": "search", "input": {"q": "rust"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        });
        let result = convert_response(&anthropic, "test");
        assert_eq!(result["choices"][0]["finish_reason"], "tool_calls");
        let tool_calls = result["choices"][0]["message"]["tool_calls"].as_array().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["id"], "call_1");
        assert_eq!(tool_calls[0]["type"], "function");
        assert_eq!(tool_calls[0]["function"]["name"], "search");
        // arguments should be stringified JSON
        let args: Value = serde_json::from_str(tool_calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args["q"], "rust");
    }

    #[test]
    fn test_thinking_blocks_skipped() {
        let anthropic = json!({
            "content": [
                {"type": "thinking", "thinking": "hmm..."},
                {"type": "text", "text": "Answer"}
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });
        let result = convert_response(&anthropic, "test");
        assert_eq!(result["choices"][0]["message"]["content"], "Answer");
    }

    // ── map_stop_reason tests ──

    #[test]
    fn test_stop_reason_mapping() {
        assert_eq!(map_stop_reason("end_turn"), "stop");
        assert_eq!(map_stop_reason("stop_sequence"), "stop");
        assert_eq!(map_stop_reason("max_tokens"), "length");
        assert_eq!(map_stop_reason("tool_use"), "tool_calls");
        assert_eq!(map_stop_reason("unknown_reason"), "stop");
    }

    // ── SSE event conversion tests ──

    #[test]
    fn test_sse_message_start() {
        let mut state = StreamState::new("test-model".into());
        let data = json!({});
        let result = convert_sse_event(&mut state, "message_start", &data).unwrap();
        assert!(result.contains("assistant"));
        assert!(result.starts_with("data: "));
    }

    #[test]
    fn test_sse_text_delta() {
        let mut state = StreamState::new("test".into());
        state.current_block_type = "text".into();
        let data = json!({"delta": {"type": "text_delta", "text": "Hello"}});
        let result = convert_sse_event(&mut state, "content_block_delta", &data).unwrap();
        assert!(result.contains("Hello"));
    }

    #[test]
    fn test_sse_tool_use_indexing() {
        let mut state = StreamState::new("test".into());

        // First tool_use block → index 0
        let data = json!({"content_block": {"type": "tool_use", "id": "call_1", "name": "fn1"}});
        let result = convert_sse_event(&mut state, "content_block_start", &data).unwrap();
        assert!(result.contains("\"index\":0"));
        assert_eq!(state.tool_call_index, 1);

        // Second tool_use block → index 1
        let data = json!({"content_block": {"type": "tool_use", "id": "call_2", "name": "fn2"}});
        let result = convert_sse_event(&mut state, "content_block_start", &data).unwrap();
        assert!(result.contains("\"index\":1"));
        assert_eq!(state.tool_call_index, 2);
    }

    #[test]
    fn test_sse_message_delta_finish() {
        let mut state = StreamState::new("test".into());
        let data = json!({"delta": {"stop_reason": "end_turn"}});
        let result = convert_sse_event(&mut state, "message_delta", &data).unwrap();
        assert!(result.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn test_sse_message_stop_returns_none() {
        let mut state = StreamState::new("test".into());
        let data = json!({});
        // message_stop is handled by the outer loop, not convert_sse_event
        assert!(convert_sse_event(&mut state, "message_stop", &data).is_none());
    }

    // ── Image conversion tests ──

    #[test]
    fn test_base64_image_conversion() {
        let part = json!({
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,iVBOR..."}
        });
        let result = convert_content_part(&part).unwrap();
        assert_eq!(result["type"], "image");
        assert_eq!(result["source"]["type"], "base64");
        assert_eq!(result["source"]["media_type"], "image/png");
        assert_eq!(result["source"]["data"], "iVBOR...");
    }
}
