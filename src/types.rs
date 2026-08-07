use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    pub delta: ChatDelta,
    #[allow(dead_code)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatDelta {
    #[allow(dead_code)]
    pub role: Option<String>,
    pub content: Option<String>,
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

impl ChatDelta {
    /// Normalized reasoning delta, preferring `reasoning_content` and falling
    /// back to the `reasoning` alias.
    pub fn reasoning_text(&self) -> Option<&str> {
        self.reasoning_content
            .as_deref()
            .or(self.reasoning.as_deref())
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
    use super::ChatUsage;
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
}
