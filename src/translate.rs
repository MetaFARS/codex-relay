use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

use crate::{session::SessionStore, types::*};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamespaceToolName {
    pub namespace: String,
    pub name: String,
}

pub type NamespaceToolMap = HashMap<String, NamespaceToolName>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CustomToolName {
    pub name: String,
    pub argument_field: String,
}

pub type CustomToolMap = HashMap<String, CustomToolName>;

/// Convert a Responses API request + prior history into a Chat Completions request.
pub fn to_chat_request(
    req: &ResponsesRequest,
    history: Vec<ChatMessage>,
    sessions: &SessionStore,
) -> ChatRequest {
    let mut messages = history;

    // Prefer `instructions` (Codex CLI) over `system` (other clients).
    let system_text = req.instructions.as_ref().or(req.system.as_ref());
    if let Some(system) = system_text {
        if messages.is_empty() || messages[0].role != "system" {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: Some(Value::String(system.clone())),
                    reasoning_content: None,
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            );
        }
    }

    // Append new input, mapping Responses API roles to Chat Completions roles.
    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(ChatMessage {
                role: "user".into(),
                content: Some(Value::String(text.clone())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        ResponsesInput::Messages(items) => {
            // Collect call_ids already present in history (from previous_response_id).
            // This prevents creating duplicate assistant-with-tool_calls messages
            // when the input items replay function_call entries from prior output.
            let existing_call_ids: HashSet<String> = messages
                .iter()
                .flat_map(|msg| {
                    let mut ids: Vec<String> = Vec::new();
                    if let Some(tcs) = &msg.tool_calls {
                        ids.extend(tcs.iter().filter_map(|tc| {
                            tc.get("id").and_then(|v| v.as_str()).map(String::from)
                        }));
                    }
                    ids.extend(msg.tool_call_id.iter().cloned());
                    ids
                })
                .collect();

            // For function_call_output dedup, only skip if a tool response
            // already exists for the call_id (not just from assistant tool_calls).
            let existing_tool_responses: HashSet<String> = messages
                .iter()
                .filter_map(|msg| msg.tool_call_id.clone())
                .collect();

            // Process items with index so we can group consecutive function_call
            // entries into a single assistant message. Providers require all tool
            // calls from one turn to live in one message with a tool_calls array.
            let mut i = 0;
            while i < items.len() {
                let item = &items[i];
                let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");

                if matches!(item_type, "function_call" | "custom_tool_call") {
                    // Skip function_call items whose call_id already exists in history.
                    // Duplicates occur when both previous_response_id and input replay
                    // the same function_call entries from prior output.
                    let call_id = item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                    if existing_call_ids.contains(call_id) {
                        i += 1;
                        continue;
                    }
                    // Collect this and all immediately following tool call items
                    // into one assistant message with multiple tool_calls entries.
                    let mut grouped: Vec<Value> = Vec::new();
                    let mut reasoning_content: Option<String> = None;

                    while i < items.len() {
                        let cur = &items[i];
                        let cur_type = cur.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if !matches!(cur_type, "function_call" | "custom_tool_call") {
                            break;
                        }
                        let call_id = cur.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = response_function_name_for_chat(cur);
                        let args = if cur_type == "custom_tool_call" {
                            let input = cur.get("input").and_then(Value::as_str).unwrap_or("");
                            json!({ custom_argument_field(&name): input }).to_string()
                        } else {
                            cur.get("arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("{}")
                                .to_string()
                        };
                        if reasoning_content.is_none() {
                            reasoning_content = sessions.get_reasoning(call_id);
                        }
                        grouped.push(json!({
                            "id": call_id,
                            "type": "function",
                            "function": { "name": name, "arguments": args }
                        }));
                        i += 1;
                    }

                    let mut msg = ChatMessage {
                        role: "assistant".into(),
                        content: None,
                        reasoning_content,
                        tool_calls: Some(grouped),
                        tool_call_id: None,
                        name: None,
                    };
                    // Fallback: try turn-level fingerprint if call_id lookup missed
                    if msg.reasoning_content.is_none() {
                        msg.reasoning_content = sessions.get_turn_reasoning(&messages, &msg);
                    }
                    messages.push(msg);
                } else {
                    match item_type {
                        "function_call_output" | "custom_tool_call_output" => {
                            let call_id =
                                item.get("call_id").and_then(|v| v.as_str()).unwrap_or("");
                            // Skip function_call_output items if a tool response
                            // for this call_id already exists in history.
                            if existing_tool_responses.contains(call_id) {
                                i += 1;
                                continue;
                            }
                            let output = item.get("output").and_then(|v| v.as_str()).unwrap_or("");
                            messages.push(ChatMessage {
                                role: "tool".into(),
                                content: Some(Value::String(output.to_string())),
                                reasoning_content: None,
                                tool_calls: None,
                                tool_call_id: Some(call_id.to_string()),
                                name: None,
                            });
                        }
                        // Codex 0.128+ may replay reasoning items in input history.
                        // Reasoning round-trip is handled separately by the session
                        // store (call_id and turn-level fingerprint indexes), not via
                        // input items — drop these so they don't pollute as empty
                        // user messages in the catch-all branch.
                        "reasoning" => {}
                        _ => {
                            // Regular user/assistant/developer message
                            let role = item.get("role").and_then(|v| v.as_str()).unwrap_or("user");
                            let role = match role {
                                "developer" => "system",
                                other => other,
                            }
                            .to_string();
                            let mut msg = ChatMessage {
                                role,
                                content: value_to_chat_content(item.get("content")),
                                reasoning_content: None,
                                tool_calls: None,
                                tool_call_id: None,
                                name: None,
                            };
                            // For assistant messages, try to recover reasoning_content
                            // from the turn-level index (needed for thinking models like
                            // DeepSeek that require reasoning_content to be passed back).
                            if msg.role == "assistant" {
                                msg.reasoning_content =
                                    sessions.get_turn_reasoning(&messages, &msg);
                            }
                            // System/developer messages from input items must go to the
                            // front of the array. Codex sometimes interleaves them between
                            // function_call and function_call_output items, which would
                            // break the assistant→tool message ordering required by the
                            // Chat Completions API.
                            if msg.role == "system" {
                                if !messages.is_empty() && messages[0].role == "system" {
                                    messages[0] = msg; // replace existing system prompt
                                } else {
                                    messages.insert(0, msg);
                                }
                            } else {
                                messages.push(msg);
                            }
                        }
                    }
                    i += 1;
                }
            }
        }
    }

    let mapped_model = map_model_name(&req.model);
    // GLM/Zhipu only emits reasoning_content when `thinking` is explicitly
    // enabled; its default auto-thinking is suppressed by heavy agent system
    // prompts (e.g. Codex). Other providers (DeepSeek/Kimi) think by default and
    // must not receive this field, so it stays GLM-gated to preserve their
    // request shape. See GitHub issue #26.
    let enable_glm_thinking = is_glm_like_model(&req.model) || is_glm_like_model(&mapped_model);

    ChatRequest {
        model: mapped_model,
        messages,
        tools: convert_tools(&req.tools),
        temperature: req.temperature,
        max_tokens: req.max_output_tokens,
        stream_options: req.stream.then_some(ChatStreamOptions {
            include_usage: true,
        }),
        thinking: enable_glm_thinking.then(|| ChatThinking {
            kind: "enabled".into(),
        }),
        stream: req.stream,
    }
}

/// Whether a model name looks like a GLM/Zhipu reasoning model that needs the
/// explicit `thinking` switch to emit reasoning_content.
fn is_glm_like_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("glm") || m.contains("zhipu") || m.contains("bigmodel")
}

/// Map model names via `CODEX_RELAY_MODEL_MAP` env var.
/// Format: `source-model:target-model,source2:target2`
/// Example: `CODEX_RELAY_MODEL_MAP="gpt-5.4:deepseek-v4-pro,gpt-5.5:deepseek-v4-pro"`
fn map_model_name(name: &str) -> String {
    if let Ok(map_str) = std::env::var("CODEX_RELAY_MODEL_MAP") {
        for pair in map_str.split(',') {
            let mut parts = pair.splitn(2, ':');
            if let (Some(from), Some(to)) = (parts.next(), parts.next()) {
                if name == from.trim() {
                    return to.trim().to_string();
                }
            }
        }
    }
    name.to_string()
}

/// Flatten Responses-API tools into Chat Completions tools.
///
/// - `function` → keep, normalize shape
/// - `namespace` (Codex 0.128+ MCP plugin grouping) → splice in each child function
/// - `web_search`, `image_generation`, `computer`, `file_search`, … → drop;
///   non-OpenAI providers reject these built-ins.
fn convert_tools(tools: &[Value]) -> Vec<Value> {
    let denied = tool_denylist_from_env();
    convert_tools_with_denylist(tools, &denied)
}

pub fn namespace_tool_map(tools: &[Value]) -> NamespaceToolMap {
    let mut map = NamespaceToolMap::new();
    for tool in tools {
        if tool.get("type").and_then(Value::as_str) != Some("namespace") {
            continue;
        }
        let Some(namespace) = tool.get("name").and_then(Value::as_str) else {
            continue;
        };
        let Some(subs) = tool.get("tools").and_then(Value::as_array) else {
            continue;
        };
        for sub in subs {
            if sub.get("type").and_then(Value::as_str) != Some("function") {
                continue;
            }
            let Some(name) = sub.get("name").and_then(Value::as_str) else {
                continue;
            };
            let chat_name = chat_function_name_for_namespace_tool(namespace, name);
            map.insert(
                chat_name,
                NamespaceToolName {
                    namespace: namespace.to_string(),
                    name: name.to_string(),
                },
            );
        }
    }
    map
}

pub fn custom_tool_map(tools: &[Value]) -> CustomToolMap {
    tools
        .iter()
        .filter(|tool| tool.get("type").and_then(Value::as_str) == Some("custom"))
        .filter_map(|tool| {
            let name = tool.get("name").and_then(Value::as_str)?;
            Some((
                name.to_string(),
                CustomToolName {
                    name: name.to_string(),
                    argument_field: custom_argument_field(name).to_string(),
                },
            ))
        })
        .collect()
}

fn custom_argument_field(name: &str) -> &'static str {
    if name == "apply_patch" {
        "patch"
    } else {
        "input"
    }
}

pub(crate) fn custom_tool_input(arguments: &str, argument_field: &str) -> String {
    serde_json::from_str::<Value>(arguments)
        .ok()
        .and_then(|value| value.get(argument_field)?.as_str().map(String::from))
        .unwrap_or_else(|| arguments.to_string())
}

fn tool_denylist_from_env() -> HashSet<String> {
    std::env::var("CODEX_RELAY_TOOL_DENYLIST")
        .unwrap_or_default()
        .split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(String::from)
        .collect()
}

fn convert_tools_with_denylist(tools: &[Value], denied: &HashSet<String>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(tools.len());
    for tool in tools {
        match tool.get("type").and_then(Value::as_str) {
            Some("function") => {
                if !tool_is_denied(tool, None, denied) {
                    out.push(convert_tool(tool));
                }
            }
            Some("namespace") => {
                let namespace = tool.get("name").and_then(Value::as_str).unwrap_or("");
                if let Some(subs) = tool.get("tools").and_then(Value::as_array) {
                    for sub in subs {
                        if sub.get("type").and_then(Value::as_str) == Some("function") {
                            let name = sub
                                .get("name")
                                .and_then(Value::as_str)
                                .map(|name| chat_function_name_for_namespace_tool(namespace, name));
                            if !tool_is_denied(sub, name.as_deref(), denied) {
                                out.push(convert_tool_with_name(sub, name.as_deref()));
                            }
                        }
                    }
                }
            }
            Some("custom") => {
                let Some(name) = tool.get("name").and_then(Value::as_str) else {
                    continue;
                };
                if denied.contains(name) {
                    continue;
                }
                let argument_field = custom_argument_field(name);
                let description = tool
                    .get("description")
                    .and_then(Value::as_str)
                    .unwrap_or("Provide the raw custom tool input.");
                out.push(json!({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                argument_field: {"type": "string"}
                            },
                            "required": [argument_field],
                            "additionalProperties": false
                        }
                    }
                }));
            }
            _ => {}
        }
    }
    out
}

fn tool_is_denied(tool: &Value, override_name: Option<&str>, denied: &HashSet<String>) -> bool {
    if denied.is_empty() {
        return false;
    }
    let name = override_name
        .map(str::to_string)
        .or_else(|| {
            tool.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .map(String::from)
        })
        .or_else(|| tool.get("name").and_then(Value::as_str).map(String::from));
    name.is_some_and(|name| denied.contains(&name))
}

/// Responses API tool format → Chat Completions tool format.
///
/// Responses API (flat):
///   {"type":"function","name":"foo","description":"...","parameters":{...},"strict":false}
///
/// Chat Completions (nested):
///   {"type":"function","function":{"name":"foo","description":"...","parameters":{...}}}
fn convert_tool(tool: &Value) -> Value {
    convert_tool_with_name(tool, None)
}

fn convert_tool_with_name(tool: &Value, override_name: Option<&str>) -> Value {
    let Some(obj) = tool.as_object() else {
        return tool.clone();
    };
    // Already in Chat Completions format if it has a "function" sub-object.
    if obj.contains_key("function") {
        let mut tool = tool.clone();
        if let Some(name) = override_name {
            if let Some(func) = tool.get_mut("function").and_then(Value::as_object_mut) {
                func.insert("name".into(), Value::String(name.to_string()));
            }
        }
        return tool;
    }
    // Convert from Responses API flat format.
    if obj.get("type").and_then(Value::as_str) == Some("function") {
        let mut func = serde_json::Map::new();
        if let Some(name) = override_name {
            func.insert("name".into(), Value::String(name.to_string()));
        } else if let Some(v) = obj.get("name") {
            func.insert("name".into(), v.clone());
        }
        if let Some(v) = obj.get("description") {
            func.insert("description".into(), v.clone());
        }
        if let Some(v) = obj.get("parameters") {
            func.insert("parameters".into(), v.clone());
        }
        if let Some(v) = obj.get("strict") {
            func.insert("strict".into(), v.clone());
        }
        return json!({"type": "function", "function": func});
    }
    tool.clone()
}

fn response_function_name_for_chat(item: &Value) -> String {
    let name = item.get("name").and_then(Value::as_str).unwrap_or("");
    let namespace = item.get("namespace").and_then(Value::as_str).unwrap_or("");
    if namespace.is_empty() {
        name.to_string()
    } else {
        chat_function_name_for_namespace_tool(namespace, name)
    }
}

pub(crate) fn chat_function_name_for_namespace_tool(namespace: &str, name: &str) -> String {
    // Chat Completions tool names must match `^[a-zA-Z0-9_-]+$`, so `.` is not
    // accepted by strict upstreams. Decoding must use NamespaceToolMap whenever
    // request tools are available; the separator alone is not authoritative.
    format!("{namespace}-{name}")
}

/// Convert a Chat Completions response into a Responses API response.
pub fn from_chat_response(
    id: String,
    model: &str,
    chat: ChatResponse,
) -> (ResponsesResponse, Vec<ChatMessage>) {
    from_chat_response_with_tool_map(id, model, chat, &NamespaceToolMap::new())
}

pub fn from_chat_response_with_tool_map(
    id: String,
    model: &str,
    chat: ChatResponse,
    namespace_tools: &NamespaceToolMap,
) -> (ResponsesResponse, Vec<ChatMessage>) {
    from_chat_response_with_tool_maps(id, model, chat, namespace_tools, &CustomToolMap::new())
}

pub fn from_chat_response_with_tool_maps(
    id: String,
    model: &str,
    chat: ChatResponse,
    namespace_tools: &NamespaceToolMap,
    custom_tools: &CustomToolMap,
) -> (ResponsesResponse, Vec<ChatMessage>) {
    let choice = chat
        .choices
        .into_iter()
        .next()
        .unwrap_or_else(|| ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: Some(Value::String(String::new())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        });

    let usage = chat.usage.unwrap_or_default();
    tracing::debug!("cache(non-stream): {}", usage.cache_summary());
    let mut output = Vec::new();

    let text = choice.message.text_content().to_string();
    if !text.is_empty() || choice.message.tool_calls.is_none() {
        output.push(json!({
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": text,
            }],
        }));
    }

    if let Some(tool_calls) = &choice.message.tool_calls {
        for tool_call in tool_calls {
            let function = tool_call.get("function").unwrap_or(&Value::Null);
            let raw_name = function.get("name").and_then(Value::as_str).unwrap_or("");
            let (namespace, name) = response_function_name_for_responses(raw_name, namespace_tools);
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}");
            let call_id = tool_call.get("id").and_then(Value::as_str).unwrap_or("");
            let item = if let Some(custom) = custom_tools.get(raw_name) {
                json!({
                    "type": "custom_tool_call",
                    "id": format!("ctc_{}", uuid::Uuid::new_v4().simple()),
                    "call_id": call_id,
                    "name": custom.name,
                    "input": custom_tool_input(arguments, &custom.argument_field),
                    "status": "completed"
                })
            } else {
                let mut item = json!({
                    "type": "function_call",
                    "id": format!("fc_{}", uuid::Uuid::new_v4().simple()),
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed"
                });
                if let Some(namespace) = namespace {
                    item["namespace"] = Value::String(namespace);
                }
                item
            };
            output.push(item);
        }
    }

    let response = ResponsesResponse {
        id,
        object: "response",
        model: model.to_string(),
        output,
        usage: ResponsesUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            input_tokens_details: Some(InputTokensDetails {
                cached_tokens: usage.cache_hit(),
            }),
        },
    };

    (response, vec![choice.message])
}

pub(crate) fn response_function_name_for_responses(
    name: &str,
    namespace_tools: &NamespaceToolMap,
) -> (Option<String>, String) {
    if let Some(tool_name) = namespace_tools.get(name) {
        return (Some(tool_name.namespace.clone()), tool_name.name.clone());
    }
    split_mcp_function_name(name)
}

pub(crate) fn split_mcp_function_name(name: &str) -> (Option<String>, String) {
    if let Some((namespace, child)) = name.split_once('.') {
        if !namespace.is_empty() && !child.is_empty() {
            return (Some(namespace.to_string()), child.to_string());
        }
    }

    let Some(rest) = name.strip_prefix("mcp__") else {
        return (None, name.to_string());
    };
    let Some(server_end) = rest.find("__") else {
        return (None, name.to_string());
    };
    let split_at = "mcp__".len() + server_end + "__".len();
    if split_at >= name.len() {
        return (None, name.to_string());
    }
    (
        Some(name[..split_at].to_string()),
        name[split_at..].to_string(),
    )
}

/// Translate a Responses-API `content` value to its Chat Completions equivalent.
///
/// - Plain string → `Value::String`.
/// - Parts array containing only text → collapsed to `Value::String` (the
///   shape Chat Completions expects in the common text-only case, and the
///   shape session.rs's reasoning fingerprint compares against).
/// - Parts array with any non-text part (e.g. `input_image`) → kept as a
///   `Value::Array` of multimodal Chat Completions parts:
///     * `input_text` / `text`  → `{type:"text", text}`
///     * `input_image` (string) → `{type:"image_url", image_url:{url}}`
///     * `image_url`            → normalized to `{type:"image_url", image_url:{url}}`
///
///   Unknown part types pass through; the upstream may reject them and the
///   relay propagates that error as-is.
fn value_to_chat_content(v: Option<&Value>) -> Option<Value> {
    match v {
        None => None,
        Some(Value::String(s)) => Some(Value::String(s.clone())),
        Some(Value::Array(parts)) => {
            // `output_text` is what Codex replays for assistant history items;
            // treat it the same as text for the purposes of collapsing.
            let has_non_text = parts.iter().any(|p| {
                let kind = p.get("type").and_then(|t| t.as_str()).unwrap_or("");
                !matches!(kind, "input_text" | "text" | "output_text")
            });
            if !has_non_text {
                let s: String = parts
                    .iter()
                    .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                    .collect::<Vec<_>>()
                    .join("");
                Some(Value::String(s))
            } else {
                let mapped: Vec<Value> = parts.iter().map(map_content_part).collect();
                Some(Value::Array(mapped))
            }
        }
        Some(other) => Some(Value::String(other.to_string())),
    }
}

/// Reshape a single Responses-API content part into a Chat Completions one.
fn map_content_part(part: &Value) -> Value {
    let kind = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
    match kind {
        "input_text" | "text" | "output_text" => {
            let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
            json!({"type": "text", "text": text})
        }
        "input_image" => {
            // Responses API: image_url is a plain string (often a data: URL).
            // Chat Completions wants it wrapped in an object.
            let url = part.get("image_url").and_then(|u| u.as_str()).unwrap_or("");
            json!({"type": "image_url", "image_url": {"url": url}})
        }
        "image_url" => {
            // Either already-Chat-Completions-shaped (image_url is an object)
            // or a Responses-style flat url; normalize both.
            let inner = match part.get("image_url") {
                Some(Value::Object(_)) => part.get("image_url").cloned().unwrap_or(Value::Null),
                Some(Value::String(s)) => json!({"url": s}),
                _ => json!({"url": ""}),
            };
            json!({"type": "image_url", "image_url": inner})
        }
        _ => part.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn base_req(input: ResponsesInput) -> ResponsesRequest {
        ResponsesRequest {
            model: "test".into(),
            input,
            previous_response_id: None,
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
        }
    }

    #[test]
    fn test_text_input_becomes_user_message() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Text("hello".into()));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[0].text_content(), "hello");
    }

    #[test]
    fn test_system_prompt_from_instructions() {
        let sessions = SessionStore::new();
        let mut req = base_req(ResponsesInput::Text("hi".into()));
        req.instructions = Some("be helpful".into());
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].text_content(), "be helpful");
    }

    #[test]
    fn test_developer_role_mapped_to_system() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "developer", "content": "secret instructions"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].text_content(), "secret instructions");
    }

    #[test]
    fn test_function_call_grouping() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "c1", "name": "fn_a", "arguments": "{}"}),
            json!({"type": "function_call", "call_id": "c2", "name": "fn_b", "arguments": "{}"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, "assistant");
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["id"], "c1");
        assert_eq!(calls[1]["id"], "c2");
    }

    #[test]
    fn test_custom_tool_call_and_output_replay_as_chat_messages() {
        let sessions = SessionStore::new();
        let patch = "*** Begin Patch\n*** End Patch";
        let req = base_req(ResponsesInput::Messages(vec![
            json!({
                "type": "custom_tool_call",
                "call_id": "call_patch",
                "name": "apply_patch",
                "input": patch
            }),
            json!({
                "type": "custom_tool_call_output",
                "call_id": "call_patch",
                "output": "Done!"
            }),
        ]));

        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages.len(), 2);
        let call = &chat.messages[0].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call["function"]["name"], "apply_patch");
        assert_eq!(
            serde_json::from_str::<Value>(call["function"]["arguments"].as_str().unwrap()).unwrap(),
            json!({"patch": patch})
        );
        assert_eq!(chat.messages[1].role, "tool");
        assert_eq!(chat.messages[1].tool_call_id.as_deref(), Some("call_patch"));
        assert_eq!(chat.messages[1].text_content(), "Done!");
    }

    #[test]
    fn test_namespaced_function_call_replays_to_flattened_chat_name() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![json!({
            "type": "function_call",
            "call_id": "call_status",
            "namespace": "mcp__node_repl",
            "name": "status",
            "arguments": "{}"
        })]));
        let chat = to_chat_request(&req, vec![], &sessions);
        let calls = chat.messages[0].tool_calls.as_ref().unwrap();
        assert_eq!(
            calls[0]["function"]["name"].as_str(),
            Some("mcp__node_repl-status")
        );
    }

    #[test]
    fn test_from_chat_response_uses_request_tool_map_for_namespace() {
        let chat = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![json!({
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "mcp__node_repl-status",
                            "arguments": "{}"
                        }
                    })]),
                    tool_call_id: None,
                    name: None,
                },
            }],
            usage: None,
        };
        let tools = vec![json!({
            "type": "namespace",
            "name": "mcp__node_repl",
            "tools": [{"type": "function", "name": "status"}]
        })];
        let namespace_tools = namespace_tool_map(&tools);

        let (resp, _) =
            from_chat_response_with_tool_map("resp_1".into(), "test-model", chat, &namespace_tools);
        assert_eq!(resp.output.len(), 1);
        assert_eq!(resp.output[0]["type"], "function_call");
        assert_eq!(resp.output[0]["namespace"], "mcp__node_repl");
        assert_eq!(resp.output[0]["name"], "status");
        assert_eq!(resp.output[0]["call_id"], "call_status");
    }

    #[test]
    fn test_from_chat_response_preserves_hyphen_flat_tool_name() {
        let chat = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![json!({
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "foo-bar",
                            "arguments": "{}"
                        }
                    })]),
                    tool_call_id: None,
                    name: None,
                },
            }],
            usage: None,
        };

        let (resp, _) = from_chat_response("resp_1".into(), "test-model", chat);
        assert!(resp.output[0].get("namespace").is_none());
        assert_eq!(resp.output[0]["name"], "foo-bar");
    }

    #[test]
    fn test_from_chat_response_keeps_legacy_non_namespaced_tool_name() {
        let chat = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![json!({
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "mcp__node_repljs",
                            "arguments": "{}"
                        }
                    })]),
                    tool_call_id: None,
                    name: None,
                },
            }],
            usage: None,
        };

        let (resp, _) = from_chat_response("resp_1".into(), "test-model", chat);
        assert!(resp.output[0].get("namespace").is_none());
        assert_eq!(resp.output[0]["name"], "mcp__node_repljs");
    }

    #[test]
    fn test_from_chat_response_keeps_legacy_dot_namespace_split() {
        let chat = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![json!({
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "mcp__node_repl.status",
                            "arguments": "{}"
                        }
                    })]),
                    tool_call_id: None,
                    name: None,
                },
            }],
            usage: None,
        };

        let (resp, _) = from_chat_response("resp_1".into(), "test-model", chat);
        assert_eq!(resp.output[0]["namespace"], "mcp__node_repl");
        assert_eq!(resp.output[0]["name"], "status");
    }

    #[test]
    fn test_function_call_output_becomes_tool_message() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call_output", "call_id": "c1", "output": "result"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "tool");
        assert_eq!(chat.messages[0].text_content(), "result");
        assert_eq!(chat.messages[0].tool_call_id.as_deref(), Some("c1"));
    }

    #[test]
    fn test_convert_tool_flat_to_nested() {
        let flat = json!({
            "type": "function",
            "name": "my_fn",
            "description": "does stuff",
            "parameters": {"type": "object"}
        });
        let nested = convert_tool(&flat);
        assert_eq!(nested["type"], "function");
        assert_eq!(nested["function"]["name"], "my_fn");
        assert_eq!(nested["function"]["description"], "does stuff");
    }

    #[test]
    fn test_convert_tool_already_nested() {
        let already = json!({
            "type": "function",
            "function": {"name": "my_fn", "description": "does stuff"}
        });
        let result = convert_tool(&already);
        assert_eq!(result, already);
    }

    #[test]
    fn test_convert_tools_preserves_subagent_tools_without_denylist() {
        let tools = vec![
            json!({"type": "function", "name": "spawn_agent"}),
            json!({"type": "function", "name": "wait_agent"}),
        ];
        let converted = convert_tools_with_denylist(&tools, &HashSet::new());
        let names: Vec<&str> = converted
            .iter()
            .filter_map(|tool| {
                tool.get("function")
                    .and_then(|func| func.get("name"))
                    .and_then(Value::as_str)
            })
            .collect();
        assert_eq!(names, ["spawn_agent", "wait_agent"]);
    }

    #[test]
    fn test_convert_tools_denylist_filters_flat_and_namespaced_tools() {
        let tools = vec![
            json!({"type": "function", "name": "spawn_agent"}),
            json!({"type": "function", "name": "exec_command"}),
            json!({
                "type": "namespace",
                "name": "mcp__server",
                "tools": [
                    {"type": "function", "name": "blocked"},
                    {"type": "function", "name": "allowed"}
                ]
            }),
        ];
        let denied = HashSet::from(["spawn_agent".to_string(), "mcp__server-blocked".to_string()]);

        let converted = convert_tools_with_denylist(&tools, &denied);
        let names: Vec<&str> = converted
            .iter()
            .filter_map(|tool| {
                tool.get("function")
                    .and_then(|func| func.get("name"))
                    .and_then(Value::as_str)
            })
            .collect();

        assert_eq!(names, ["exec_command", "mcp__server-allowed"]);
    }

    #[test]
    fn test_to_chat_request_honors_tool_denylist_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("CODEX_RELAY_TOOL_DENYLIST", "spawn_agent, wait_agent");

        let sessions = SessionStore::new();
        let mut req = base_req(ResponsesInput::Text("hello".into()));
        req.tools = vec![
            json!({"type": "function", "name": "spawn_agent"}),
            json!({"type": "function", "name": "exec_command"}),
            json!({"type": "function", "name": "wait_agent"}),
        ];

        let chat = to_chat_request(&req, vec![], &sessions);
        let names: Vec<&str> = chat
            .tools
            .iter()
            .filter_map(|tool| {
                tool.get("function")
                    .and_then(|func| func.get("name"))
                    .and_then(Value::as_str)
            })
            .collect();

        assert_eq!(names, ["exec_command"]);
        std::env::remove_var("CODEX_RELAY_TOOL_DENYLIST");
    }

    #[test]
    fn test_value_to_text_string() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": "plain text"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].text_content(), "plain text");
    }

    /// input_image (Responses API) + input_text → Chat Completions
    /// multimodal content array with image_url wrapped in {url:...}.
    #[test]
    fn test_input_image_becomes_multimodal_content() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "what is this?"},
                {"type": "input_image", "image_url": "data:image/png;base64,AAA"}
            ]}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        let parts = chat.messages[0]
            .content
            .as_ref()
            .and_then(|v| v.as_array())
            .expect("content must be a parts array when an image is present");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "what is this?");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/png;base64,AAA");
    }

    /// Chat-Completions-style image_url passes through normalized.
    #[test]
    fn test_chat_style_image_url_passes_through() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        let parts = chat.messages[0]
            .content
            .as_ref()
            .and_then(|v| v.as_array())
            .expect("multimodal content");
        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(parts[0]["image_url"]["url"], "https://example.com/x.png");
    }

    /// Text-only content arrays must still collapse to a plain string —
    /// session.rs fingerprints assistant turns on the string form.
    #[test]
    fn test_text_only_parts_collapse_to_string() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hi"}
            ]}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert!(chat.messages[0].content.as_ref().unwrap().is_string());
        assert_eq!(chat.messages[0].text_content(), "hi");
    }

    #[test]
    fn test_value_to_text_parts_array() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": "hello "},
                {"type": "input_text", "text": "world"}
            ]}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].text_content(), "hello world");
    }

    // ── Deduplication tests ────────────────────────────────────────────

    /// When previous_response_id supplies history that already contains
    /// assistant tool_calls, function_call items in the new input with the
    /// same call_ids must be skipped to avoid duplicate tool_calls messages.
    #[test]
    fn test_skip_duplicate_function_call_from_history() {
        let sessions = SessionStore::new();

        // Simulate history from previous_response_id: assistant with tool_call
        let history = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("run command".into()),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![json!({
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "exec", "arguments": "{\"cmd\":\"ls\"}"}
                })]),
                tool_call_id: None,
                name: None,
            },
        ];

        // Input replays the same function_call + output + new user message
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "call_1", "name": "exec", "arguments": "{\"cmd\":\"ls\"}"}),
            json!({"type": "function_call_output", "call_id": "call_1", "output": "file.txt"}),
            json!({"type": "message", "role": "user", "content": "next"}),
        ]));

        let chat = to_chat_request(&req, history, &sessions);

        // Should have: user, assistant{tool_calls:[call_1]}, tool(call_1), user(next)
        // NOT: user, assistant{tool_calls:[call_1]}, assistant{tool_calls:[call_1]}, tool(call_1), user
        assert_eq!(
            chat.messages.len(),
            4,
            "should not duplicate assistant tool_calls message"
        );
        assert_eq!(chat.messages[0].role, "user");
        assert_eq!(chat.messages[1].role, "assistant");
        assert!(chat.messages[1].tool_calls.is_some());
        assert_eq!(chat.messages[1].tool_calls.as_ref().unwrap().len(), 1);
        assert_eq!(chat.messages[2].role, "tool");
        assert_eq!(chat.messages[2].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(chat.messages[3].role, "user");
    }

    /// When previous_response_id supplies history that already contains
    /// tool messages, function_call_output items in the new input with the
    /// same call_ids must be skipped.
    #[test]
    fn test_skip_duplicate_function_call_output_from_history() {
        let sessions = SessionStore::new();

        let history = vec![
            ChatMessage {
                role: "user".into(),
                content: Some("run".into()),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![json!({
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "ls", "arguments": "{}"}
                })]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some("output".into()),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: Some("call_x".into()),
                name: None,
            },
        ];

        // Input replays function_call_output + new user message
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call_output", "call_id": "call_x", "output": "output"}),
            json!({"type": "message", "role": "user", "content": "next"}),
        ]));

        let chat = to_chat_request(&req, history, &sessions);

        // Should have: user, assistant{tool_calls}, tool, user(next)
        // NOT: user, assistant{tool_calls}, tool, tool(dup), user
        assert_eq!(chat.messages.len(), 4);
        assert_eq!(chat.messages[2].role, "tool");
        assert_eq!(chat.messages[3].role, "user");
    }

    // ── System/developer interleaving tests (#4) ──────────────────────

    /// When Codex interleaves a developer/system message between
    /// function_call and function_call_output items, it must be moved to
    /// the front so the Chat Completions API sees:
    ///   assistant[tool_calls] → tool  (not assistant[tool_calls] → system → tool)
    #[test]
    fn test_system_message_between_tool_calls_moved_to_front() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "function_call", "call_id": "c1", "name": "exec", "arguments": "{}"}),
            json!({"type": "message", "role": "developer", "content": "be careful"}),
            json!({"type": "function_call_output", "call_id": "c1", "output": "done"}),
            json!({"type": "message", "role": "user", "content": "next turn"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(
            roles,
            ["system", "assistant", "tool", "user"],
            "system must be at front, assistant→tool pairing must be contiguous"
        );
    }

    /// When a system/developer message appears at the very start of input
    /// items (before any function calls), it still lands at messages[0].
    #[test]
    fn test_system_message_at_start_of_input() {
        let sessions = SessionStore::new();
        let req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "developer", "content": "rules"}),
            json!({"type": "message", "role": "user", "content": "hello"}),
        ]));
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].text_content(), "rules");
        assert_eq!(chat.messages[1].role, "user");
    }

    /// When both `instructions` and a developer input item provide a system
    /// prompt, the later one (from input items) wins.
    #[test]
    fn test_system_from_input_replaces_instructions() {
        let sessions = SessionStore::new();
        let mut req = base_req(ResponsesInput::Messages(vec![
            json!({"type": "message", "role": "user", "content": "hi"}),
            json!({"type": "message", "role": "developer", "content": "override"}),
        ]));
        req.instructions = Some("original".into());
        let chat = to_chat_request(&req, vec![], &sessions);
        assert_eq!(chat.messages[0].role, "system");
        assert_eq!(chat.messages[0].text_content(), "override");
    }

    // ── Model name mapping tests (#4) ─────────────────────────────────

    #[test]
    fn test_map_model_name_with_env() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(
            "CODEX_RELAY_MODEL_MAP",
            "gpt-5.4:deepseek-v4-pro,gpt-5.5:deepseek-v4-pro",
        );
        assert_eq!(map_model_name("gpt-5.4"), "deepseek-v4-pro");
        assert_eq!(map_model_name("gpt-5.5"), "deepseek-v4-pro");
        std::env::remove_var("CODEX_RELAY_MODEL_MAP");
    }

    #[test]
    fn test_map_model_name_no_match_passthrough() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var("CODEX_RELAY_MODEL_MAP", "gpt-5.4:deepseek-v4-pro");
        assert_eq!(map_model_name("unknown-model"), "unknown-model");
        std::env::remove_var("CODEX_RELAY_MODEL_MAP");
    }

    #[test]
    fn test_map_model_name_no_env_var() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var("CODEX_RELAY_MODEL_MAP");
        assert_eq!(map_model_name("gpt-5.4"), "gpt-5.4");
    }

    #[test]
    fn test_map_model_name_trims_whitespace() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(
            "CODEX_RELAY_MODEL_MAP",
            " gpt-5.4 : deepseek-v4-pro , gpt-5.5 : deepseek-v4-flash ",
        );
        assert_eq!(map_model_name("gpt-5.4"), "deepseek-v4-pro");
        assert_eq!(map_model_name("gpt-5.5"), "deepseek-v4-flash");
        std::env::remove_var("CODEX_RELAY_MODEL_MAP");
    }
}
