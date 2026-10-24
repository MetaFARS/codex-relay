use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;

// ── Responses API (inbound from Codex CLI) ──────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(default)]
    pub previous_response_id: Option<String>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    /// Responses API system prompt field (some clients use `system`, others `instructions`)
    #[serde(default)]
    pub system: Option<String>,
    #[serde(default)]
    pub instructions: Option<String>,
    /// Responses-API reasoning controls. Codex sends `"reasoning": null` for
    /// models it believes have no reasoning support, so this must tolerate an
    /// explicit null as well as a missing key.
    #[serde(default)]
    pub reasoning: Option<ResponsesReasoning>,
}

/// Reasoning block of a Responses request. Only `effort` maps onto Chat
/// Completions; `summary` has no counterpart there and is intentionally
/// dropped (reasoning text comes back via `reasoning_content` instead).
#[derive(Debug, Deserialize, Default)]
pub struct ResponsesReasoning {
    #[serde(default)]
    pub effort: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    /// Each item may be a user/assistant message OR a function_call_output result.
    /// Using Value here lets us handle both without a brittle fixed schema.
    Messages(Vec<Value>),
}

// Public compatibility type. The binary compiles this module directly but
// does not construct content parts itself, so it appears unused there.
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub model: String,
    pub output: Vec<Value>,
    pub usage: ResponsesUsage,
}

#[derive(Debug, Serialize, Default)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
}

#[derive(Debug, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u32,
}

// ── Chat Completions (outbound to provider) ──────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Conversation input without request-scoped `instructions`/`system`.
    /// Retain this separately so refreshing the base prompt cannot erase
    /// developer messages or carry old request instructions into a new turn.
    #[serde(skip)]
    pub retained_messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatStreamOptions>,
    /// Zhipu/GLM thinking switch. Only serialized for GLM-like models so other
    /// providers keep their existing request shape. GLM suppresses its default
    /// auto-thinking under heavy agent system prompts (e.g. Codex), so this must
    /// be sent explicitly for reasoning to be emitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ChatThinking>,
    /// Reasoning budget, forwarded verbatim from the Responses request's
    /// `reasoning.effort`. Deliberately a free-form String rather than an enum:
    /// the accepted set is the upstream's business, not the relay's, and it
    /// differs per provider (Command Code takes low|medium|high|xhigh|max and
    /// rejects none/minimal; OpenRouter and OpenAI take other sets). Passing it
    /// through unchanged means an unsupported value surfaces as the upstream's
    /// own error instead of being silently rewritten here. Use
    /// `--drop-upstream-params '["reasoning_effort"]'` to strip it for an
    /// upstream that chokes on the field entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    pub stream: bool,
}

#[derive(Debug, Serialize)]
pub struct ChatStreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Serialize)]
pub struct ChatThinking {
    #[serde(rename = "type")]
    pub kind: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    /// Either a plain string (the common case) or a multimodal content-parts
    /// array (`[{type:"text",...}, {type:"image_url",...}]`). Modeled as a raw
    /// JSON Value so both shapes pass through serde transparently. Use
    /// [`ChatMessage::text_content`] when you only care about the text.
    pub content: Option<Value>,
    /// Reasoning/thinking content emitted by models like kimi-k2.6.
    /// Must be round-tripped back when replaying tool call history.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Best-effort plain-text view of `content`. Returns `""` for missing,
    /// non-string, or multimodal payloads — callers that care about the
    /// multimodal parts should look at `content` directly.
    pub fn text_content(&self) -> &str {
        self.content.as_ref().and_then(|v| v.as_str()).unwrap_or("")
    }
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    // Prompt-cache accounting. Providers disagree on the shape:
    //   DeepSeek / packyapi: top-level prompt_cache_{hit,miss}_tokens
    //   OpenAI-compatible:   prompt_tokens_details.cached_tokens
    // Capture both; normalize via cache_hit()/cache_miss().
    #[serde(default)]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_cache_miss_tokens: Option<u32>,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

impl ChatUsage {
    /// Cached (hit) prompt tokens. Prefers the DeepSeek-style top-level field,
    /// else falls back to the OpenAI-style `prompt_tokens_details.cached_tokens`.
    pub(crate) fn cache_hit(&self) -> u32 {
        self.prompt_cache_hit_tokens
            .or_else(|| self.prompt_tokens_details.as_ref().map(|d| d.cached_tokens))
            .unwrap_or(0)
    }

    /// Non-cached (miss) prompt tokens; falls back to `prompt_tokens - hit`.
    pub(crate) fn cache_miss(&self) -> u32 {
        self.prompt_cache_miss_tokens
            .unwrap_or_else(|| self.prompt_tokens.saturating_sub(self.cache_hit()))
    }

    /// One-line prompt-cache summary for debug logging, e.g.
    /// `hit=1152 miss=51 prompt=1203 hit_rate=95.8%`.
    pub fn cache_summary(&self) -> String {
        let (hit, prompt) = (self.cache_hit(), self.prompt_tokens);
        let rate = if prompt > 0 {
            100.0 * hit as f64 / prompt as f64
        } else {
            0.0
        };
        format!(
            "hit={hit} miss={} prompt={prompt} hit_rate={rate:.1}%",
            self.cache_miss()
        )
    }
}

// ── SSE streaming types ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatStreamChunk {
    pub choices: Vec<ChatStreamChoice>,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatStreamChoice {
    #[serde(default)]
    pub index: usize,
    pub delta: ChatDelta,
    /// Upstream end-of-turn signal. Used by stream.rs to tell a provider that
    /// merely omitted `[DONE]` from a connection that died mid-generation.
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatDelta {
    #[allow(dead_code)]
    pub role: Option<String>,
    /// Streamed content. OpenAI-compatible providers send a plain string;
    /// Mistral-hosted GLM sends an array of structured blocks whose `text`
    /// parts carry the answer and `thinking` parts carry reasoning. Normalized
    /// via [`ChatDelta::content_text`] and [`ChatDelta::thinking_text`].
    #[serde(default)]
    pub content: Option<DeltaContent>,
    /// Thinking content. Providers disagree on the field name: DeepSeek/Kimi/GLM
    /// use `reasoning_content`; OpenRouter/Together-style (and some newer GLM-5
    /// deployments) use `reasoning`. Capture both and normalize via
    /// [`ChatDelta::reasoning_text`].
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

/// Upstream `delta.content` shape.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum DeltaContent {
    /// `content: "text"` — the common Chat Completions shape.
    Text(String),
    /// `content: [{type: "text"|"thinking", ...}, ...]` — Mistral-style
    /// structured blocks (Mistral-hosted GLM).
    Blocks(Vec<DeltaBlock>),
}

/// One structured block inside a `delta.content` array.
///
/// Unknown block kinds and extra fields (e.g. Mistral's `closed`) are ignored,
/// so a provider adding new block types cannot break stream deserialization.
#[derive(Debug, Deserialize)]
pub struct DeltaBlock {
    /// Block kind: `"text"` (visible output) or `"thinking"` (reasoning).
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Visible text payload of a `text` block.
    #[serde(default)]
    pub text: Option<String>,
    /// Reasoning payload of a `thinking` block. Mistral nests text items
    /// (`thinking: [{type: "text", text: "..."}]`); GLM's native API sends a
    /// plain string. Both shapes are accepted.
    #[serde(default)]
    pub thinking: Option<ThinkingPayload>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ThinkingPayload {
    /// `thinking: "reasoning text"`
    Text(String),
    /// `thinking: [{type: "text", text: "reasoning text"}, ...]`
    Items(Vec<ThinkingItem>),
}

#[derive(Debug, Deserialize)]
pub struct ThinkingItem {
    #[serde(default)]
    pub text: Option<String>,
}

impl ChatDelta {
    /// Normalized reasoning delta, preferring `reasoning_content` and falling
    /// back to the `reasoning` alias.
    pub fn reasoning_text(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
    }

    /// Visible text carried by this delta: the plain string shape, or the
    /// concatenation of `text` blocks from the structured shape. `None` when
    /// the delta carries no visible text.
    pub fn content_text(&self) -> Option<Cow<'_, str>> {
        match self.content.as_ref()? {
            DeltaContent::Text(text) => Some(Cow::Borrowed(text)),
            DeltaContent::Blocks(blocks) => {
                let mut joined = String::new();
                for block in blocks {
                    if block.kind.as_deref() == Some("text") {
                        if let Some(text) = &block.text {
                            joined.push_str(text);
                        }
                    }
                }
                (!joined.is_empty()).then_some(Cow::Owned(joined))
            }
        }
    }

    /// Reasoning text carried inside structured `thinking` blocks
    /// (Mistral-hosted GLM). `None` for plain-string content and for deltas
    /// without thinking payloads.
    pub fn thinking_text(&self) -> Option<String> {
        let blocks = match self.content.as_ref()? {
            DeltaContent::Blocks(blocks) => blocks,
            DeltaContent::Text(_) => return None,
        };
        let mut joined = String::new();
        for block in blocks {
            if block.kind.as_deref() != Some("thinking") {
                continue;
            }
            match &block.thinking {
                Some(ThinkingPayload::Text(text)) => joined.push_str(text),
                Some(ThinkingPayload::Items(items)) => {
                    for item in items {
                        if let Some(text) = &item.text {
                            joined.push_str(text);
                        }
                    }
                }
                None => {}
            }
        }
        (!joined.is_empty()).then_some(joined)
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct DeltaToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: Option<DeltaFunction>,
}

#[derive(Debug, Deserialize, Default)]
pub struct DeltaFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::{ChatDelta, ChatStreamChunk, ChatUsage};
    use serde_json::json;

    #[test]
    fn cache_summary_uses_deepseek_top_level_cache_fields() {
        let usage: ChatUsage = serde_json::from_value(json!({
            "prompt_tokens": 1203,
            "completion_tokens": 25,
            "total_tokens": 1228,
            "prompt_cache_hit_tokens": 1152,
            "prompt_cache_miss_tokens": 51
        }))
        .unwrap();

        assert_eq!(
            usage.cache_summary(),
            "hit=1152 miss=51 prompt=1203 hit_rate=95.8%"
        );
    }

    #[test]
    fn cache_summary_uses_openai_prompt_tokens_details() {
        let usage: ChatUsage = serde_json::from_value(json!({
            "prompt_tokens": 893,
            "completion_tokens": 12,
            "total_tokens": 905,
            "prompt_tokens_details": {
                "cached_tokens": 640
            }
        }))
        .unwrap();

        assert_eq!(
            usage.cache_summary(),
            "hit=640 miss=253 prompt=893 hit_rate=71.7%"
        );
    }

    #[test]
    fn cache_summary_prefers_top_level_hit_when_both_shapes_exist() {
        let usage: ChatUsage = serde_json::from_value(json!({
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_cache_hit_tokens": 25,
            "prompt_tokens_details": {
                "cached_tokens": 90
            }
        }))
        .unwrap();

        assert_eq!(
            usage.cache_summary(),
            "hit=25 miss=75 prompt=100 hit_rate=25.0%"
        );
    }

    #[test]
    fn cache_summary_handles_missing_usage_fields() {
        let usage = ChatUsage::default();

        assert_eq!(usage.cache_summary(), "hit=0 miss=0 prompt=0 hit_rate=0.0%");
    }

    #[test]
    fn delta_accepts_plain_string_content() {
        let delta: ChatDelta = serde_json::from_value(json!({
            "role": "assistant",
            "content": "hello world"
        }))
        .unwrap();

        assert_eq!(delta.content_text().as_deref(), Some("hello world"));
        assert_eq!(delta.thinking_text(), None);
    }

    #[test]
    fn delta_accepts_mistral_thinking_blocks() {
        // Exact delta shape from the Mistral-hosted GLM failure report, which
        // previously failed deserialization with
        // "invalid type: sequence, expected a string".
        let delta: ChatDelta = serde_json::from_value(json!({
            "content": [
                {
                    "type": "thinking",
                    "thinking": [{"type": "text", "text": "The"}],
                    "closed": true
                }
            ]
        }))
        .unwrap();

        assert_eq!(delta.thinking_text().as_deref(), Some("The"));
        assert_eq!(delta.content_text(), None);
    }

    #[test]
    fn delta_accepts_mistral_text_blocks() {
        let delta: ChatDelta = serde_json::from_value(json!({
            "content": [{"type": "text", "text": "visible answer"}]
        }))
        .unwrap();

        assert_eq!(delta.content_text().as_deref(), Some("visible answer"));
        assert_eq!(delta.thinking_text(), None);
    }

    #[test]
    fn delta_mixed_blocks_map_text_and_thinking_separately() {
        let delta: ChatDelta = serde_json::from_value(json!({
            "content": [
                {"type": "thinking", "thinking": [{"type": "text", "text": "ponder"}]},
                {"type": "text", "text": "vis"},
                {"type": "thinking", "thinking": [{"type": "text", "text": " more"}]},
                {"type": "text", "text": "ible"}
            ]
        }))
        .unwrap();

        assert_eq!(delta.content_text().as_deref(), Some("visible"));
        assert_eq!(delta.thinking_text().as_deref(), Some("ponder more"));
    }

    #[test]
    fn delta_thinking_block_accepts_string_payload() {
        // GLM's native API nests a plain string instead of text items.
        let delta: ChatDelta = serde_json::from_value(json!({
            "content": [{"type": "thinking", "thinking": "reasoning text"}]
        }))
        .unwrap();

        assert_eq!(delta.thinking_text().as_deref(), Some("reasoning text"));
        assert_eq!(delta.content_text(), None);
    }

    #[test]
    fn delta_ignores_unknown_block_kinds_and_extra_fields() {
        let delta: ChatDelta = serde_json::from_value(json!({
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.invalid/x.png"}},
                {"type": "text", "text": "kept", "unknown_future_field": true}
            ]
        }))
        .unwrap();

        assert_eq!(delta.content_text().as_deref(), Some("kept"));
        assert_eq!(delta.thinking_text(), None);
    }

    #[test]
    fn delta_missing_or_null_content_yields_nothing() {
        let delta: ChatDelta = serde_json::from_value(json!({"role": "assistant"})).unwrap();
        assert_eq!(delta.content_text(), None);
        assert_eq!(delta.thinking_text(), None);

        let delta: ChatDelta = serde_json::from_value(json!({"content": null})).unwrap();
        assert_eq!(delta.content_text(), None);
    }

    #[test]
    fn mistral_stream_chunk_with_structured_content_deserializes() {
        let chunk: ChatStreamChunk = serde_json::from_value(json!({
            "choices": [{
                "index": 0,
                "delta": {
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "The"}],
                            "closed": true
                        }
                    ]
                },
                "finish_reason": null
            }]
        }))
        .unwrap();

        let delta = &chunk.choices[0].delta;
        assert_eq!(delta.thinking_text().as_deref(), Some("The"));
        assert_eq!(delta.content_text(), None);
    }

    #[test]
    fn delta_tool_calls_survive_alongside_structured_content() {
        let delta: ChatDelta = serde_json::from_value(json!({
            "content": [{"type": "text", "text": ""}],
            "tool_calls": [{
                "index": 0,
                "id": "call_1",
                "function": {"name": "shell", "arguments": "{\"cmd\":\"ls\"}"}
            }]
        }))
        .unwrap();

        assert!(delta.tool_calls.is_some());
        assert_eq!(delta.tool_calls.unwrap()[0].id.as_deref(), Some("call_1"));
    }
}
