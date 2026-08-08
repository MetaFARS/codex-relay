use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};

use crate::types::ChatRequest;

#[derive(Clone, Debug, Default)]
pub struct UpstreamRequestConfig {
    extra_params: Map<String, Value>,
    drop_params: Vec<String>,
}

impl UpstreamRequestConfig {
    pub fn from_raw(extra_params: Option<&str>, drop_params: Option<&str>) -> Result<Self> {
        Ok(Self {
            extra_params: parse_extra_params(extra_params)?,
            drop_params: parse_drop_params(drop_params)?,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.extra_params.is_empty() && self.drop_params.is_empty()
    }

    pub fn extra_param_count(&self) -> usize {
        self.extra_params.len()
    }

    pub fn drop_param_count(&self) -> usize {
        self.drop_params.len()
    }

    pub fn request_body(&self, chat_req: &ChatRequest) -> Result<Value> {
        let mut body = serde_json::to_value(chat_req)?;
        let object = body
            .as_object_mut()
            .context("serialized chat request was not a JSON object")?;

        for key in &self.drop_params {
            object.remove(key);
        }
        for (key, value) in &self.extra_params {
            object.insert(key.clone(), value.clone());
        }
        if object
            .get("n")
            .is_some_and(|value| value.as_u64() != Some(1))
        {
            bail!("codex-relay supports only one upstream completion (n=1)");
        }

        Ok(body)
    }
}

fn parse_extra_params(raw: Option<&str>) -> Result<Map<String, Value>> {
    let Some(raw) = non_empty(raw) else {
        return Ok(Map::new());
    };
    match serde_json::from_str::<Value>(raw)? {
        Value::Object(object) => Ok(object),
        _ => bail!("--upstream-extra-params must be a JSON object"),
    }
}

fn parse_drop_params(raw: Option<&str>) -> Result<Vec<String>> {
    let Some(raw) = non_empty(raw) else {
        return Ok(Vec::new());
    };
    let values = serde_json::from_str::<Vec<String>>(raw)
        .context("--drop-upstream-params must be a JSON array of strings")?;
    if values.iter().any(|value| value.is_empty()) {
        bail!("--drop-upstream-params entries must not be empty");
    }
    Ok(values)
}

fn non_empty(raw: Option<&str>) -> Option<&str> {
    raw.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::types::ChatMessage;

    fn chat_request() -> ChatRequest {
        ChatRequest {
            model: "deepseek-v4-pro".into(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: Some(json!("hello")),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: Vec::new(),
            temperature: Some(0.2),
            max_tokens: Some(100),
            stream_options: None,
            thinking: None,
            reasoning_effort: None,
            stream: false,
        }
    }

    #[test]
    fn merges_extra_params_and_drops_top_level_params() {
        let config = UpstreamRequestConfig::from_raw(
            Some(r#"{"thinking":{"type":"disabled"},"temperature":0.7}"#),
            Some(r#"["max_tokens"]"#),
        )
        .unwrap();

        let body = config.request_body(&chat_request()).unwrap();

        assert_eq!(body["thinking"], json!({"type": "disabled"}));
        assert_eq!(body["temperature"], json!(0.7));
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["model"], json!("deepseek-v4-pro"));
    }

    #[test]
    fn rejects_non_object_extra_params() {
        assert!(UpstreamRequestConfig::from_raw(Some(r#"["thinking"]"#), None).is_err());
    }

    #[test]
    fn rejects_multiple_upstream_completions() {
        let config = UpstreamRequestConfig::from_raw(Some(r#"{"n":2}"#), None).unwrap();
        let error = config.request_body(&chat_request()).unwrap_err();
        assert!(error.to_string().contains("n=1"));
    }

    #[test]
    fn rejects_non_string_drop_params() {
        assert!(UpstreamRequestConfig::from_raw(None, Some(r#"["ok", 1]"#)).is_err());
    }
}
