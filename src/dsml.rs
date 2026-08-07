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
//!
//! # Dialects
//!
//! V4 Flash leaks a *doubled* delimiter and expresses parameters as
//! self-closing `invoke` tags whose value sits in the `string` attribute
//! (observed 2026-08-07 via Command Code):
//!
//! ```text
//! <｜｜DSML｜｜tool_calls>
//! <｜｜DSML｜｜invoke name="exec_command">
//! <｜｜DSML｜｜invoke name="cmd" string="echo alpha" />
//! </｜｜DSML｜｜invoke>
//! </｜｜DSML｜｜tool_calls>
//! ```
//!
//! Matching only the single-bar form made healing a no-op for that model: the
//! marker search missed, `parse_leaked_tool_calls` returned `None`, and the
//! raw markup reached Codex as plain text. Both delimiters are handled now,
//! and parameters are read by *shape* rather than by tag name — a self-closing
//! tag has no body, so its value can only live in an attribute.

use serde_json::{json, Value};

use crate::types::ChatMessage;

/// One delimiter flavour of the DSML markup.
///
/// Everything is spelled out per dialect rather than concatenated from a
/// prefix, so a mistyped delimiter fails to compile instead of silently
/// producing a marker that never matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DsmlDialect {
    /// Prefix common to every opening tag of this dialect.
    marker: &'static str,
    invoke_open: &'static str,
    invoke_close: &'static str,
    calls_open: &'static str,
    calls_close: &'static str,
}

/// `<｜DSML｜…>` — V3.2 / V4 Pro.
const SINGLE_BAR: DsmlDialect = DsmlDialect {
    marker: "<｜DSML｜",
    invoke_open: "<｜DSML｜invoke",
    invoke_close: "</｜DSML｜invoke>",
    calls_open: "<｜DSML｜tool_calls>",
    calls_close: "</｜DSML｜tool_calls>",
};

/// `<｜｜DSML｜｜…>` — V4 Flash.
const DOUBLE_BAR: DsmlDialect = DsmlDialect {
    marker: "<｜｜DSML｜｜",
    invoke_open: "<｜｜DSML｜｜invoke",
    invoke_close: "</｜｜DSML｜｜invoke>",
    calls_open: "<｜｜DSML｜｜tool_calls>",
    calls_close: "</｜｜DSML｜｜tool_calls>",
};

/// Double-bar first: `<｜DSML｜` is not a substring of `<｜｜DSML｜｜` (the `<`
/// is followed by two bars there), but probing the more specific form first
/// keeps that independent of the delimiters ever changing.
const DIALECTS: [DsmlDialect; 2] = [DOUBLE_BAR, SINGLE_BAR];

/// The dialect whose marker appears earliest in `text`, if any.
fn detect_dialect(text: &str) -> Option<(DsmlDialect, usize)> {
    DIALECTS
        .iter()
        .filter_map(|dialect| text.find(dialect.marker).map(|at| (*dialect, at)))
        .min_by_key(|(_, at)| *at)
}

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
    let (dialect, _) = detect_dialect(text)?;
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut rest = text;

    while let Some(start) = rest.find(dialect.invoke_open) {
        let header_start = start + dialect.invoke_open.len();
        let Some(header_len) = unquoted_char(&rest[header_start..], '>') else {
            break;
        };
        let header = &rest[header_start..header_start + header_len];
        let body_start = header_start + header_len + 1;
        let Some(body_len) = rest[body_start..].find(dialect.invoke_close) else {
            break;
        };
        let Some(name) = attribute(header, "name") else {
            break;
        };
        let arguments = parse_parameters(&rest[body_start..body_start + body_len], dialect);
        calls.push(DsmlToolCall {
            name: name.to_string(),
            arguments: Value::Object(arguments).to_string(),
        });
        cleaned.push_str(&rest[..start]);
        rest = &rest[body_start + body_len + dialect.invoke_close.len()..];
    }

    if calls.is_empty() {
        return None;
    }
    cleaned.push_str(rest);
    let cleaned = cleaned
        .replace(dialect.calls_open, "")
        .replace(dialect.calls_close, "");
    Some((cleaned.trim().to_string(), calls))
}

/// Parse the parameter tags inside one `invoke` body.
///
/// Tags are matched on the dialect marker rather than on a fixed tag name:
/// single-bar leaks name them `parameter`, double-bar leaks reuse `invoke`.
/// What actually distinguishes a value's location is the tag's *shape*:
///
/// - self-closing (`… />`) — no body exists, so the value is the `string`
///   attribute (double-bar form).
/// - with a body — the value is the body text, and `string="false"` marks it
///   as a non-string JSON literal (single-bar form).
fn parse_parameters(body: &str, dialect: DsmlDialect) -> serde_json::Map<String, Value> {
    let mut arguments = serde_json::Map::new();
    let mut rest = body;
    while let Some(start) = rest.find(dialect.marker) {
        let header_start = start + dialect.marker.len();
        let Some(header_len) = unquoted_char(&rest[header_start..], '>') else {
            break;
        };
        let header = &rest[header_start..header_start + header_len];
        let after_tag = header_start + header_len + 1;

        let Some(tag) = header.split_whitespace().next() else {
            break;
        };
        // A closing tag (`</…invoke>`) means the enclosing element ended; the
        // marker search cannot see `</` because the marker starts with `<`, so
        // this only guards against a malformed header.
        if tag.starts_with('/') {
            rest = &rest[after_tag..];
            continue;
        }

        if header.trim_end().ends_with('/') {
            if let Some(name) = attribute(header, "name") {
                let value = attribute(header, "string").unwrap_or_default();
                arguments.insert(name.to_string(), Value::String(value.to_string()));
            }
            rest = &rest[after_tag..];
            continue;
        }

        let close = format!("</{}{}>", dialect.marker.trim_start_matches('<'), tag);
        let Some(value_len) = rest[after_tag..].find(&close) else {
            break;
        };
        let raw = &rest[after_tag..after_tag + value_len];
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
        rest = &rest[after_tag + value_len + close.len()..];
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
    if detect_dialect(text).is_none() {
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
        if let Some((_, start)) = detect_dialect(&self.pending) {
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
/// any dialect's marker.
///
/// Taking the maximum across dialects matters: `<｜` is a prefix of both, and
/// withholding only the shorter one would emit the first bar of a double-bar
/// marker before the rest of it arrives, splitting the marker across chunks so
/// it never matches.
fn longest_marker_prefix_suffix(text: &str) -> usize {
    let mut best = 0;
    for dialect in DIALECTS {
        for (i, _) in dialect.marker.char_indices().skip(1) {
            if text.ends_with(&dialect.marker[..i]) {
                best = best.max(i);
            }
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
        assert!(parse_leaked_tool_calls("<｜｜DSML｜｜tool_calls>dangling").is_none());
    }

    /// Verbatim capture from Codex 0.144.5 → relay → Command Code →
    /// deepseek/deepseek-v4-flash, 2026-08-07. Doubled delimiters, and the
    /// parameter is a self-closing `invoke` carrying its value in `string`.
    const DOUBLE_BAR_ENVELOPE: &str = "<｜｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name=\"exec_command\">\n<｜｜DSML｜｜invoke name=\"cmd\" string=\"echo alpha\" />\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜tool_calls>";

    #[test]
    fn parses_double_bar_envelope_with_self_closing_parameters() {
        let text = format!("Let me run it.\n{DOUBLE_BAR_ENVELOPE}");
        let (cleaned, calls) = parse_leaked_tool_calls(&text).expect("healed");
        assert_eq!(cleaned, "Let me run it.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "exec_command");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["cmd"], "echo alpha");
    }

    #[test]
    fn double_bar_self_closing_parameter_keeps_special_characters() {
        // Quotes inside the value would end the header scan early if the `>`
        // search were not quote-aware; `/` inside it must not read as
        // self-closing on its own either.
        let text = "<｜｜DSML｜｜invoke name=\"exec_command\">\n<｜｜DSML｜｜invoke name=\"cmd\" string=\"grep -r a/b > out.txt\" />\n</｜｜DSML｜｜invoke>";
        let (_, calls) = parse_leaked_tool_calls(text).expect("healed");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["cmd"], "grep -r a/b > out.txt");
    }

    #[test]
    fn double_bar_supports_bodied_parameters_too() {
        // Not observed in the wild, but the dialect differs only in the
        // delimiter; a bodied parameter must not silently drop its value.
        let text = "<｜｜DSML｜｜invoke name=\"bash\">\n<｜｜DSML｜｜parameter name=\"command\" string=\"true\">ls -la</｜｜DSML｜｜parameter>\n<｜｜DSML｜｜parameter name=\"timeout\" string=\"false\">15</｜｜DSML｜｜parameter>\n</｜｜DSML｜｜invoke>";
        let (_, calls) = parse_leaked_tool_calls(text).expect("healed");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["command"], "ls -la");
        assert_eq!(args["timeout"], 15);
    }

    #[test]
    fn heal_chat_message_fires_on_double_bar() {
        let mut message = ChatMessage {
            role: "assistant".into(),
            content: Some(Value::String(DOUBLE_BAR_ENVELOPE.to_string())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        heal_chat_message(&mut message);
        let calls = message.tool_calls.expect("tool calls synthesized");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "exec_command");
        assert!(message.content.is_none());
    }

    #[test]
    fn stream_filter_holds_split_double_bar_marker_and_heals() {
        let mut filter = DsmlStreamFilter::default();
        let mut emitted = String::new();
        // Split inside the doubled delimiter: withholding only a single-bar
        // prefix here would leak "<｜" and desync the marker.
        emitted.push_str(&filter.push("Let me run it.<｜"));
        emitted.push_str(&filter.push("｜DSML｜｜tool_calls>\n<｜｜DSML｜｜invoke name=\"exec_command\">\n<｜｜DSML｜｜invoke name=\"cmd\" string=\"echo alpha\" />\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜"));
        emitted.push_str(&filter.push("tool_calls>"));
        assert_eq!(emitted, "Let me run it.");
        let (leftover, calls) = filter.finish();
        assert_eq!(leftover, "");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "exec_command");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["cmd"], "echo alpha");
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
