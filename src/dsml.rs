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
use std::collections::BTreeMap;

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
    let mut calls = Vec::new();
    let mut cleaned = String::new();
    let mut copied_through = 0;
    let mut search_from = 0;

    while let Some((dialect, start)) = find_next_invoke(text, search_from) {
        match parse_invoke(text, start, dialect) {
            Ok((end, call)) => {
                let removal_start = envelope_start(text, copied_through, start, dialect);
                let removal_end = envelope_end(text, end, dialect);
                cleaned.push_str(&text[copied_through..removal_start]);
                calls.push(call);
                copied_through = removal_end;
                search_from = removal_end;
            }
            Err(()) => {
                // Preserve malformed candidates byte-for-byte, but continue so
                // an unrelated valid call later in the message can still heal.
                let Some(end) = malformed_invoke_end(text, start, dialect) else {
                    break;
                };
                search_from = end;
            }
        }
    }

    if calls.is_empty() {
        return None;
    }
    cleaned.push_str(&text[copied_through..]);
    Some((cleaned.trim().to_string(), calls))
}

#[derive(Debug)]
struct ParsedTag {
    name: String,
    attrs: BTreeMap<String, String>,
    self_closing: bool,
    end: usize,
}

fn find_next_invoke(text: &str, from: usize) -> Option<(DsmlDialect, usize)> {
    DIALECTS
        .iter()
        .filter_map(|dialect| {
            let mut offset = from;
            while let Some(found) = text[offset..].find(dialect.invoke_open) {
                let start = offset + found;
                let after = start + dialect.invoke_open.len();
                if text[after..]
                    .chars()
                    .next()
                    .is_some_and(|ch| ch.is_whitespace() || ch == '>')
                {
                    return Some((*dialect, start));
                }
                offset = after;
            }
            None
        })
        .min_by_key(|(_, start)| *start)
}

fn parse_invoke(
    text: &str,
    start: usize,
    dialect: DsmlDialect,
) -> Result<(usize, DsmlToolCall), ()> {
    let invoke = parse_open_tag(text, start, dialect)?;
    if invoke.name != "invoke" || invoke.self_closing {
        return Err(());
    }
    let name = invoke
        .attrs
        .get("name")
        .filter(|name| !name.is_empty())
        .ok_or(())?;
    let mut arguments = serde_json::Map::new();
    let mut cursor = invoke.end;

    loop {
        cursor = skip_whitespace(text, cursor);
        if text[cursor..].starts_with(dialect.invoke_close) {
            let end = cursor + dialect.invoke_close.len();
            return Ok((
                end,
                DsmlToolCall {
                    name: name.clone(),
                    arguments: Value::Object(arguments).to_string(),
                },
            ));
        }
        if !text[cursor..].starts_with(dialect.marker) {
            return Err(());
        }

        let parameter = parse_open_tag(text, cursor, dialect)?;
        let parameter_name = parameter
            .attrs
            .get("name")
            .filter(|name| !name.is_empty())
            .ok_or(())?
            .clone();
        if arguments.contains_key(&parameter_name) {
            return Err(());
        }

        let value = if dialect == DOUBLE_BAR && parameter.name == "invoke" && parameter.self_closing
        {
            Value::String(parameter.attrs.get("string").ok_or(())?.clone())
        } else if parameter.name == "parameter" && !parameter.self_closing {
            let string_kind = parameter
                .attrs
                .get("string")
                .map(String::as_str)
                .ok_or(())?;
            if !matches!(string_kind, "true" | "false") {
                return Err(());
            }
            let close = format!("</{}parameter>", dialect.marker.trim_start_matches('<'));
            let tail = &text[parameter.end..];
            let close_offset = tail.find(&close).ok_or(())?;
            if tail
                .find(dialect.marker)
                .into_iter()
                .chain(tail.find(dialect.invoke_close))
                .any(|offset| offset < close_offset)
            {
                return Err(());
            }
            let close_at = parameter.end + close_offset;
            let raw = &text[parameter.end..close_at];
            cursor = close_at + close.len();
            if string_kind == "false" {
                serde_json::from_str(raw.trim()).map_err(|_| ())?
            } else {
                Value::String(raw.to_string())
            }
        } else {
            return Err(());
        };
        if parameter.self_closing {
            cursor = parameter.end;
        }
        arguments.insert(parameter_name, value);
    }
}

/// Find the balanced end of a malformed invoke so recovery never descends
/// into a nested, valid-looking call and executes it out of context.
fn malformed_invoke_end(text: &str, start: usize, dialect: DsmlDialect) -> Option<usize> {
    let outer = parse_open_tag(text, start, dialect).ok()?;
    if outer.name != "invoke" || outer.self_closing {
        return Some(outer.end);
    }
    let mut depth = 1usize;
    let mut cursor = outer.end;
    while cursor < text.len() {
        let next_open = text[cursor..]
            .find(dialect.marker)
            .map(|offset| cursor + offset);
        let next_close = text[cursor..]
            .find(dialect.invoke_close)
            .map(|offset| cursor + offset);
        match (next_open, next_close) {
            (None, Some(close)) => {
                depth -= 1;
                cursor = close + dialect.invoke_close.len();
                if depth == 0 {
                    return Some(cursor);
                }
            }
            (Some(open), Some(close)) if close < open => {
                depth -= 1;
                cursor = close + dialect.invoke_close.len();
                if depth == 0 {
                    return Some(cursor);
                }
            }
            (Some(open), _) => {
                let tag = parse_open_tag(text, open, dialect).ok()?;
                if tag.name == "invoke" && !tag.self_closing {
                    depth += 1;
                }
                cursor = tag.end;
            }
            (None, None) => return None,
        }
    }
    None
}

fn parse_open_tag(text: &str, start: usize, dialect: DsmlDialect) -> Result<ParsedTag, ()> {
    if !text[start..].starts_with(dialect.marker) {
        return Err(());
    }
    let mut cursor = start + dialect.marker.len();
    let name_start = cursor;
    while let Some(ch) = text[cursor..].chars().next() {
        if ch.is_whitespace() || matches!(ch, '/' | '>') {
            break;
        }
        cursor += ch.len_utf8();
    }
    if cursor == name_start {
        return Err(());
    }
    let name = text[name_start..cursor].to_string();
    let mut attrs = BTreeMap::new();

    loop {
        cursor = skip_whitespace(text, cursor);
        match text[cursor..].chars().next().ok_or(())? {
            '>' => {
                return Ok(ParsedTag {
                    name,
                    attrs,
                    self_closing: false,
                    end: cursor + 1,
                });
            }
            '/' => {
                cursor += 1;
                cursor = skip_whitespace(text, cursor);
                if !text[cursor..].starts_with('>') {
                    return Err(());
                }
                return Ok(ParsedTag {
                    name,
                    attrs,
                    self_closing: true,
                    end: cursor + 1,
                });
            }
            _ => {}
        }

        let key_start = cursor;
        while let Some(ch) = text[cursor..].chars().next() {
            if ch.is_whitespace() || ch == '=' {
                break;
            }
            if matches!(ch, '/' | '>') {
                return Err(());
            }
            cursor += ch.len_utf8();
        }
        if cursor == key_start {
            return Err(());
        }
        let key = text[key_start..cursor].to_string();
        cursor = skip_whitespace(text, cursor);
        if !text[cursor..].starts_with('=') {
            return Err(());
        }
        cursor += 1;
        cursor = skip_whitespace(text, cursor);
        if !text[cursor..].starts_with('"') {
            return Err(());
        }
        cursor += 1;
        let (value, end) = parse_quoted_value(text, cursor)?;
        cursor = end;
        if attrs.insert(key, value).is_some() {
            return Err(());
        }
    }
}

fn parse_quoted_value(text: &str, mut cursor: usize) -> Result<(String, usize), ()> {
    let mut value = String::new();
    loop {
        let ch = text[cursor..].chars().next().ok_or(())?;
        cursor += ch.len_utf8();
        match ch {
            '"' => return Ok((value, cursor)),
            '\\' => {
                let escaped = text[cursor..].chars().next().ok_or(())?;
                if matches!(escaped, '"' | '\\') {
                    value.push(escaped);
                    cursor += escaped.len_utf8();
                } else {
                    value.push('\\');
                }
            }
            other => value.push(other),
        }
    }
}

fn skip_whitespace(text: &str, mut cursor: usize) -> usize {
    while let Some(ch) = text[cursor..].chars().next() {
        if !ch.is_whitespace() {
            break;
        }
        cursor += ch.len_utf8();
    }
    cursor
}

fn envelope_start(
    text: &str,
    copied_through: usize,
    invoke_start: usize,
    dialect: DsmlDialect,
) -> usize {
    let prefix = &text[copied_through..invoke_start];
    prefix
        .rfind(dialect.calls_open)
        .filter(|at| prefix[at + dialect.calls_open.len()..].trim().is_empty())
        .map_or(invoke_start, |at| copied_through + at)
}

fn envelope_end(text: &str, invoke_end: usize, dialect: DsmlDialect) -> usize {
    let after_whitespace = skip_whitespace(text, invoke_end);
    if text[after_whitespace..].starts_with(dialect.calls_close) {
        after_whitespace + dialect.calls_close.len()
    } else {
        invoke_end
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
    fn double_bar_attribute_parser_handles_escapes_and_tag_text() {
        let text = "<｜｜DSML｜｜invoke name=\"exec_command\">\n<｜｜DSML｜｜invoke name=\"cmd\" string=\"printf \\\"a>b\\\" C:\\temp\\file </｜｜DSML｜｜invoke>\" />\n</｜｜DSML｜｜invoke>";
        let (_, calls) = parse_leaked_tool_calls(text).expect("healed");
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(
            args["cmd"],
            "printf \"a>b\" C:\\temp\\file </｜｜DSML｜｜invoke>"
        );
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
    fn parses_mixed_dialects_in_source_order() {
        for text in [
            format!("{ENVELOPE}\n{DOUBLE_BAR_ENVELOPE}"),
            format!("{DOUBLE_BAR_ENVELOPE}\n{ENVELOPE}"),
        ] {
            let (cleaned, calls) = parse_leaked_tool_calls(&text).expect("healed");
            assert_eq!(cleaned, "");
            assert_eq!(calls.len(), 2);
            if text.starts_with("<｜DSML｜") {
                assert_eq!(calls[0].name, "shell");
                assert_eq!(calls[1].name, "exec_command");
            } else {
                assert_eq!(calls[0].name, "exec_command");
                assert_eq!(calls[1].name, "shell");
            }
        }
    }

    #[test]
    fn malformed_marker_does_not_block_later_valid_dialect() {
        let malformed = "<｜DSML｜invoke_extra name=\"not_a_call\">raw</｜DSML｜invoke_extra>";
        let text = format!("{malformed}\n{DOUBLE_BAR_ENVELOPE}");
        let (cleaned, calls) = parse_leaked_tool_calls(&text).expect("later call healed");
        assert_eq!(cleaned, malformed);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "exec_command");
    }

    #[test]
    fn malformed_calls_fail_closed() {
        let cases = [
            "<｜｜DSML｜｜invoke_extra name=\"exec_command\"></｜｜DSML｜｜invoke>",
            "<｜｜DSML｜｜invoke name=\"exec_command\"><｜｜DSML｜｜metadata name=\"cmd\" string=\"echo bad\" /></｜｜DSML｜｜invoke>",
            "<｜｜DSML｜｜invoke name=\"exec_command\"><｜｜DSML｜｜invoke name=\"cmd\" /></｜｜DSML｜｜invoke>",
            "<｜｜DSML｜｜invoke name=\"exec_command\" name=\"other\"></｜｜DSML｜｜invoke>",
            "<｜｜DSML｜｜invoke name=\"exec_command\"><｜｜DSML｜｜invoke name=\"cmd\" string=\"one\" /><｜｜DSML｜｜invoke name=\"cmd\" string=\"two\" /></｜｜DSML｜｜invoke>",
            "<｜｜DSML｜｜invoke name=\"exec_command\"><｜｜DSML｜｜invoke name=\"cmd\" string=\"unterminated /></｜｜DSML｜｜invoke>",
        ];
        for text in cases {
            assert!(
                parse_leaked_tool_calls(text).is_none(),
                "malformed DSML must not execute: {text}"
            );
        }
    }

    #[test]
    fn malformed_call_is_preserved_when_later_call_heals() {
        let malformed =
            "<｜｜DSML｜｜invoke name=\"bad\"><｜｜DSML｜｜invoke name=\"cmd\" /></｜｜DSML｜｜invoke>";
        let text = format!("{malformed}\n{ENVELOPE}");
        let (cleaned, calls) = parse_leaked_tool_calls(&text).expect("later call healed");
        assert_eq!(cleaned, malformed);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn nested_call_inside_malformed_invoke_never_executes() {
        let malformed = "<｜｜DSML｜｜invoke name=\"bad\">\n<｜｜DSML｜｜invoke name=\"exec_command\">\n<｜｜DSML｜｜invoke name=\"cmd\" string=\"echo bad\" />\n</｜｜DSML｜｜invoke>\n</｜｜DSML｜｜invoke>";
        let text = format!("{malformed}\n{ENVELOPE}");
        let (cleaned, calls) = parse_leaked_tool_calls(&text).expect("later sibling healed");
        assert_eq!(cleaned, malformed);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
    }

    #[test]
    fn unclosed_parameter_cannot_borrow_close_from_nested_call() {
        let text = "<｜DSML｜invoke name=\"shell\">\n<｜DSML｜parameter name=\"command\" string=\"true\">echo bad\n<｜DSML｜invoke name=\"shell\">\n<｜DSML｜parameter name=\"command\" string=\"true\">echo good</｜DSML｜parameter>\n</｜DSML｜invoke>";
        assert!(parse_leaked_tool_calls(text).is_none());
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
    fn stream_filter_handles_mixed_dialects_one_character_at_a_time() {
        let text = format!("prefix\n{ENVELOPE}\n{DOUBLE_BAR_ENVELOPE}");
        let mut filter = DsmlStreamFilter::default();
        let mut emitted = String::new();
        for ch in text.chars() {
            emitted.push_str(&filter.push(&ch.to_string()));
        }
        let (leftover, calls) = filter.finish();
        assert_eq!(format!("{emitted}{leftover}"), "prefix\n");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[1].name, "exec_command");
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
