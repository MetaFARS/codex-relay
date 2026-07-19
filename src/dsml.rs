//! Healing for leaked DeepSeek DSML tool-call markup.
//!
//! DeepSeek V3.2/V4 models emit tool calls as DSML markup which the provider
//! is supposed to parse into structured `tool_calls`. Intermittently —
//! especially with `tool_choice=auto` + streaming, or when a parameter value
//! contains newlines — the raw markup leaks into the assistant `content`
//! instead:
//!
//! ```text
//! <｜DSML｜tool_calls>
//! <｜DSML｜invoke name="shell">
//! <｜DSML｜parameter name="command" string="true">Get-Content 'a.csv' -TotalCount 5</｜DSML｜parameter>
//! </｜DSML｜invoke>
//! </｜DSML｜tool_calls>
//! ```
//!
//! Codex then sees plain text, no tool runs, and the turn silently stalls.
//! This module parses leaked DSML back into structured tool calls and strips
//! the markup from the visible text. The `｜` (U+FF5C) delimiters are
//! DeepSeek-internal tokens that never appear in legitimate output, so
//! healing is always on.

use serde_json::{json, Value};

use crate::types::ChatMessage;

/// Prefix common to every DSML tag.
pub const DSML_MARKER: &str = "<｜DSML｜";

const INVOKE_OPEN: &str = "<｜DSML｜invoke";
const INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const PARAM_OPEN: &str = "<｜DSML｜parameter";
const PARAM_CLOSE: &str = "</｜DSML｜parameter>";
const CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
const CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsmlToolCall {
    pub name: String,
    /// JSON-encoded arguments object, ready for Chat Completions `function.arguments`.
    pub arguments: String,
}

/// Parse leaked DSML tool-call markup out of `text`.
///
/// Returns the cleaned visible text plus the parsed calls, or `None` when no
/// complete `<｜DSML｜invoke>` block could be parsed (the text is then left
/// untouched so nothing is lost).
pub fn parse_leaked_tool_calls(text: &str) -> Option<(String, Vec<DsmlToolCall>)> {
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = text;

    while let Some(start) = rest.find(INVOKE_OPEN) {
        let header_start = start + INVOKE_OPEN.len();
        let Some(header_len) = unquoted_char(&rest[header_start..], '>') else {
            break;
        };
        let header = &rest[header_start..header_start + header_len];
        let body_start = header_start + header_len + 1;
        let Some(body_len) = rest[body_start..].find(INVOKE_CLOSE) else {
            break;
        };
        let Some(name) = attribute(header, "name") else {
            break;
        };
        let arguments = parse_parameters(&rest[body_start..body_start + body_len]);
        calls.push(DsmlToolCall {
            name: name.to_string(),
            arguments: Value::Object(arguments).to_string(),
        });
        cleaned.push_str(&rest[..start]);
        rest = &rest[body_start + body_len + INVOKE_CLOSE.len()..];
    }

    if calls.is_empty() {
        return None;
    }
    cleaned.push_str(rest);
    let cleaned = cleaned.replace(CALLS_OPEN, "").replace(CALLS_CLOSE, "");
    Some((cleaned.trim().to_string(), calls))
}

fn parse_parameters(body: &str) -> serde_json::Map<String, Value> {
    let mut arguments = serde_json::Map::new();
    let mut rest = body;
    while let Some(start) = rest.find(PARAM_OPEN) {
        let header_start = start + PARAM_OPEN.len();
        let Some(header_len) = unquoted_char(&rest[header_start..], '>') else {
            break;
        };
        let header = &rest[header_start..header_start + header_len];
        let value_start = header_start + header_len + 1;
        let Some(value_len) = rest[value_start..].find(PARAM_CLOSE) else {
            break;
        };
        let raw = &rest[value_start..value_start + value_len];
        if let Some(name) = attribute(header, "name") {
            // string="false" marks a non-string JSON value (number, bool, ...).
            let value = if attribute(header, "string") == Some("false") {
                serde_json::from_str::<Value>(raw.trim())
                    .unwrap_or_else(|_| Value::String(raw.to_string()))
            } else {
                Value::String(raw.to_string())
            };
            arguments.insert(name.to_string(), value);
        }
        rest = &rest[value_start + value_len + PARAM_CLOSE.len()..];
    }
    arguments
}

/// Byte offset of the first `needle` in `text` that is outside double quotes.
fn unquoted_char(text: &str, needle: char) -> Option<usize> {
    let mut in_quotes = false;
    for (i, c) in text.char_indices() {
        match c {
            '"' => in_quotes = !in_quotes,
            c if c == needle && !in_quotes => return Some(i),
            _ => {}
        }
    }
    None
}

/// Extract a `key="value"` attribute from a DSML tag header.
fn attribute<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let mut rest = header;
    loop {
        let start = rest.find(key)?;
        let after = &rest[start + key.len()..];
        // Guard against matching `name` inside another attribute's key.
        let preceded_ok = rest[..start].ends_with(char::is_whitespace) || start == 0;
        if preceded_ok {
            if let Some(after_eq) = after.strip_prefix("=\"") {
                let end = after_eq.find('"')?;
                return Some(&after_eq[..end]);
            }
        }
        rest = after;
    }
}

pub(crate) fn synthesize_call_id() -> String {
    format!("call_dsml_{}", uuid::Uuid::new_v4().simple())
}

/// Heal a blocking Chat Completions assistant message in place: parse leaked
/// DSML markup in its text content into structured `tool_calls` entries and
/// strip the markup from the visible text.
pub fn heal_chat_message(message: &mut ChatMessage) {
    let text = message.text_content();
    if !text.contains(DSML_MARKER) {
        return;
    }
    let Some((cleaned, calls)) = parse_leaked_tool_calls(text) else {
        return;
    };
    tracing::warn!(
        "quirk dsml_heal fired: healed {} leaked DSML tool call(s) from blocking response",
        calls.len()
    );
    message.content = if cleaned.is_empty() {
        None
    } else {
        Some(Value::String(cleaned))
    };
    let tool_calls = message.tool_calls.get_or_insert_with(Vec::new);
    for call in calls {
        tool_calls.push(json!({
            "id": synthesize_call_id(),
            "type": "function",
            "function": { "name": call.name, "arguments": call.arguments }
        }));
    }
}

/// Incremental DSML filter for streamed text content.
///
/// Feed content deltas through [`DsmlStreamFilter::push`]; it returns the text
/// that is safe to emit downstream. Text that could be (part of) a DSML marker
/// is withheld. Once a marker is confirmed the rest of the stream is buffered
/// and [`DsmlStreamFilter::finish`] returns the cleaned leftover text plus the
/// healed tool calls.
#[derive(Debug)]
pub struct DsmlStreamFilter {
    pending: String,
    in_dsml: bool,
    enabled: bool,
}

impl Default for DsmlStreamFilter {
    fn default() -> Self {
        Self::new(true)
    }
}

impl DsmlStreamFilter {
    /// A filter that heals when `enabled`, or passes all text through
    /// untouched when the `dsml_heal` quirk is disabled.
    pub fn new(enabled: bool) -> Self {
        Self {
            pending: String::new(),
            in_dsml: false,
            enabled,
        }
    }

    /// Append a content delta; returns the portion that is safe to emit now.
    pub fn push(&mut self, delta: &str) -> String {
        if !self.enabled {
            return delta.to_string();
        }
        self.pending.push_str(delta);
        if self.in_dsml {
            return String::new();
        }
        if let Some(start) = self.pending.find(DSML_MARKER) {
            self.in_dsml = true;
            let emit = self.pending[..start].to_string();
            self.pending.drain(..start);
            return emit;
        }
        // Withhold the longest tail that could still grow into the marker.
        let hold = longest_marker_prefix_suffix(&self.pending);
        let emit_len = self.pending.len() - hold;
        let emit = self.pending[..emit_len].to_string();
        self.pending.drain(..emit_len);
        emit
    }

    /// Consume the filter at end of stream. Returns any remaining visible text
    /// and the tool calls healed from buffered DSML markup.
    pub fn finish(self) -> (String, Vec<DsmlToolCall>) {
        if !self.in_dsml {
            return (self.pending, Vec::new());
        }
        match parse_leaked_tool_calls(&self.pending) {
            Some((cleaned, calls)) => (cleaned, calls),
            // Incomplete or unparseable markup: pass the raw text through so
            // nothing is silently dropped.
            None => (self.pending, Vec::new()),
        }
    }
}

/// Length in bytes of the longest suffix of `text` that is a proper prefix of
/// [`DSML_MARKER`].
fn longest_marker_prefix_suffix(text: &str) -> usize {
    let mut best = 0;
    for (i, _) in DSML_MARKER.char_indices().skip(1) {
        if text.ends_with(&DSML_MARKER[..i]) {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENVELOPE: &str = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"shell\">\n<｜DSML｜parameter name=\"command\" string=\"true\">Get-Content 'D:\\data\\a.csv' -Encoding UTF8 -TotalCount 5</｜DSML｜parameter>\n<｜DSML｜parameter name=\"context\" string=\"true\">preview csv headers</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    #[test]
    fn parses_leaked_envelope() {
        let text = format!("我来读取文件。\n{ENVELOPE}");
        let (cleaned, calls) = parse_leaked_tool_calls(&text).expect("healed");
        assert_eq!(cleaned, "我来读取文件。");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(
            args["command"],
            "Get-Content 'D:\\data\\a.csv' -Encoding UTF8 -TotalCount 5"
        );
        assert_eq!(args["context"], "preview csv headers");
    }

    #[test]
    fn parses_multiline_and_nonstring_parameters() {
        let text = "<｜DSML｜invoke name=\"bash\">\n<｜DSML｜parameter name=\"command\" string=\"true\">line one \\\nline two > out.txt</｜DSML｜parameter>\n<｜DSML｜parameter name=\"timeout\" string=\"false\">15</｜DSML｜parameter>\n</｜DSML｜invoke>";
        let (cleaned, calls) = parse_leaked_tool_calls(text).expect("healed");
        assert_eq!(cleaned, "");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["command"], "line one \\\nline two > out.txt");
        assert_eq!(args["timeout"], 15);
    }

    #[test]
    fn parses_multiple_invokes() {
        let text = format!("{ENVELOPE}\n{ENVELOPE}");
        let (_, calls) = parse_leaked_tool_calls(&text).expect("healed");
        assert_eq!(calls.len(), 2);
    }

    #[test]
    fn plain_text_is_untouched() {
        assert!(parse_leaked_tool_calls("no markup here, just a < b").is_none());
        assert!(parse_leaked_tool_calls("<｜DSML｜tool_calls>dangling").is_none());
    }

    #[test]
    fn stream_filter_passes_plain_text() {
        let mut filter = DsmlStreamFilter::default();
        assert_eq!(filter.push("hello "), "hello ");
        assert_eq!(filter.push("a < b and c > d"), "a < b and c > d");
        let (leftover, calls) = filter.finish();
        assert_eq!(leftover, "");
        assert!(calls.is_empty());
    }

    #[test]
    fn stream_filter_holds_split_marker_and_heals() {
        let mut filter = DsmlStreamFilter::default();
        let mut emitted = String::new();
        emitted.push_str(&filter.push("先看文件。<｜DS"));
        emitted.push_str(&filter.push("ML｜tool_calls>\n<｜DSML｜invoke name=\"shell\">\n<｜DSML｜parameter name=\"command\" string=\"true\">ls</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜"));
        emitted.push_str(&filter.push("tool_calls>"));
        assert_eq!(emitted, "先看文件。");
        let (leftover, calls) = filter.finish();
        assert_eq!(leftover, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn stream_filter_releases_false_marker_prefix() {
        let mut filter = DsmlStreamFilter::default();
        let first = filter.push("a <");
        let second = filter.push("b> c");
        assert_eq!(format!("{first}{second}"), "a <b> c");
    }

    #[test]
    fn disabled_stream_filter_passes_markup_through() {
        let mut filter = DsmlStreamFilter::new(false);
        assert_eq!(filter.push(ENVELOPE), ENVELOPE);
        let (leftover, calls) = filter.finish();
        assert_eq!(leftover, "");
        assert!(calls.is_empty());
    }

    #[test]
    fn stream_filter_passes_incomplete_markup_through() {
        let mut filter = DsmlStreamFilter::default();
        filter.push("<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"shell\">truncat");
        let (leftover, calls) = filter.finish();
        assert!(calls.is_empty());
        assert!(leftover.contains("truncat"), "raw text must not be lost");
    }

    #[test]
    fn heal_chat_message_moves_markup_into_tool_calls() {
        let mut message = ChatMessage {
            role: "assistant".into(),
            content: Some(Value::String(format!("我来逐步完成这个任务。\n{ENVELOPE}"))),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        heal_chat_message(&mut message);
        assert_eq!(message.text_content(), "我来逐步完成这个任务。");
        let tool_calls = message.tool_calls.as_ref().expect("healed tool_calls");
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0]["function"]["name"], "shell");
        assert!(tool_calls[0]["id"]
            .as_str()
            .unwrap()
            .starts_with("call_dsml_"));
    }
}
