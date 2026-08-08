//! Healing for reasoning models that leak `<think>` markup into text content.
//!
//! Thinking models are supposed to expose chain of thought in a separate
//! `reasoning_content` (or `reasoning`) field. Several vLLM deployments —
//! including ones behind synthetic.new — run without a matching
//! `--reasoning-parser`, so the raw chat-template markup lands in the
//! assistant `content` instead:
//!
//! ```text
//! <think>The user said hi. I should greet them back.</think>Hello!
//! ```
//!
//! Codex then renders the chain of thought as assistant text, and the markup
//! is persisted into history and replayed upstream, where it teaches the model
//! to keep producing it. This module splits the markup back out so thinking is
//! emitted on the reasoning channel (`response.reasoning_summary_text.delta`)
//! and never reaches the visible message.
//!
//! Some chat templates pre-fill the opening `<think>` into the prompt, so a
//! blocking response may contain only the closing tag. Streaming repair only
//! recognizes explicit opening tags because interpreting a later bare closing
//! tag would otherwise depend on arbitrary SSE chunk boundaries.

use crate::types::ChatMessage;

/// Opening markers. `◁think▷` is Kimi's variant.
const OPEN_TAGS: &[&str] = &["<thinking>", "<think>", "◁think▷"];
/// Closing markers, matching [`OPEN_TAGS`].
const CLOSE_TAGS: &[&str] = &["</thinking>", "</think>", "◁/think▷"];

/// The two channels a content delta can be split across.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ThinkSplit {
    /// Text that belongs on the reasoning channel.
    pub reasoning: String,
    /// Text that belongs in the visible assistant message.
    pub text: String,
}

/// Incremental `<think>` splitter for streamed text content.
///
/// Feed content deltas through [`ThinkStreamFilter::push`]; it returns the
/// reasoning and visible text that are safe to emit now. Text that could be
/// (part of) a tag is withheld until the next delta disambiguates it, so a tag
/// split across SSE chunks (`"<thi"` then `"nk>"`) is still recognized.
#[derive(Debug)]
pub struct ThinkStreamFilter {
    pending: String,
    in_think: bool,
    emitted_text: bool,
    trim_after_close: bool,
    fired: bool,
    enabled: bool,
}

impl Default for ThinkStreamFilter {
    fn default() -> Self {
        Self::new(true)
    }
}

impl ThinkStreamFilter {
    /// A filter that splits when `enabled`, or passes all text through
    /// untouched as visible text when the `think_tags` quirk is disabled.
    pub fn new(enabled: bool) -> Self {
        Self {
            pending: String::new(),
            in_think: false,
            emitted_text: false,
            trim_after_close: false,
            fired: false,
            enabled,
        }
    }

    /// Whether any think markup was actually seen (quirk telemetry).
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Append a content delta; returns the portions safe to emit now.
    pub fn push(&mut self, delta: &str) -> ThinkSplit {
        let mut out = ThinkSplit::default();
        if !self.enabled {
            out.text.push_str(delta);
            return out;
        }
        self.pending.push_str(delta);
        loop {
            if self.in_think {
                if let Some((at, tag)) = first_tag(&self.pending, CLOSE_TAGS) {
                    out.reasoning.push_str(&self.pending[..at]);
                    self.pending.drain(..at + tag.len());
                    self.in_think = false;
                    self.trim_after_close = !self.emitted_text;
                    continue;
                }
                let keep =
                    self.pending.len() - longest_tag_prefix_suffix(&self.pending, CLOSE_TAGS);
                out.reasoning.push_str(&self.pending[..keep]);
                self.pending.drain(..keep);
                return out;
            }

            if self.trim_after_close {
                let first_non_whitespace = self
                    .pending
                    .char_indices()
                    .find_map(|(index, ch)| (!ch.is_whitespace()).then_some(index));
                match first_non_whitespace {
                    Some(index) => {
                        self.pending.drain(..index);
                        self.trim_after_close = false;
                    }
                    None => {
                        self.pending.clear();
                        return out;
                    }
                }
            }

            match first_tag(&self.pending, OPEN_TAGS) {
                Some((at, tag)) => {
                    self.emit_text(&mut out, at);
                    self.pending.drain(..at + tag.len());
                    self.in_think = true;
                    self.fired = true;
                }
                None => {
                    let keep =
                        self.pending.len() - longest_tag_prefix_suffix(&self.pending, OPEN_TAGS);
                    self.emit_text(&mut out, keep);
                    self.pending.drain(..keep);
                    return out;
                }
            }
        }
    }

    /// Consume the filter at end of stream, returning anything still withheld.
    /// An unterminated `<think>` block is treated as reasoning, not text.
    pub fn finish(mut self) -> ThinkSplit {
        let mut out = ThinkSplit::default();
        let rest = std::mem::take(&mut self.pending);
        if self.in_think {
            out.reasoning = rest;
        } else {
            out.text = if self.trim_after_close && !self.emitted_text {
                rest.trim_start().to_string()
            } else {
                rest
            };
        }
        out
    }

    /// Emit `self.pending[..upto]` as visible text without changing it.
    fn emit_text(&mut self, out: &mut ThinkSplit, upto: usize) {
        let chunk = &self.pending[..upto];
        if !chunk.is_empty() {
            self.emitted_text = true;
            out.text.push_str(chunk);
        }
    }
}

/// Heal a blocking Chat Completions assistant message in place: move leaked
/// think markup out of the visible content and into `reasoning_content`.
pub fn heal_chat_message(message: &mut ChatMessage) {
    let text = message.text_content().to_string();
    if !contains_think_markup(&text) {
        return;
    }
    let (split, fired) = if first_tag(&text, OPEN_TAGS).is_none() {
        match first_tag(&text, CLOSE_TAGS) {
            Some((at, tag)) => (
                ThinkSplit {
                    reasoning: text[..at].to_string(),
                    text: text[at + tag.len()..].trim_start().to_string(),
                },
                true,
            ),
            None => (ThinkSplit::default(), false),
        }
    } else {
        let mut filter = ThinkStreamFilter::new(true);
        let mut split = filter.push(&text);
        let fired = filter.fired();
        let tail = filter.finish();
        split.reasoning.push_str(&tail.reasoning);
        split.text.push_str(&tail.text);
        (split, fired)
    };
    if !fired {
        return;
    }
    tracing::warn!("quirk think_tags fired: healed leaked <think> markup from blocking response");
    message.content = if split.text.is_empty() {
        None
    } else {
        Some(serde_json::Value::String(split.text))
    };
    if !split.reasoning.is_empty() {
        match &mut message.reasoning_content {
            Some(existing) => existing.push_str(&split.reasoning),
            slot @ None => *slot = Some(split.reasoning),
        }
    }
}

/// Whether `text` contains any think marker at all (cheap pre-check).
pub fn contains_think_markup(text: &str) -> bool {
    OPEN_TAGS
        .iter()
        .chain(CLOSE_TAGS)
        .any(|tag| text.contains(tag))
}

/// Byte offset and matched tag of the earliest tag in `text`, preferring the
/// longest tag when several start at the same offset.
fn first_tag<'a>(text: &str, tags: &[&'a str]) -> Option<(usize, &'a str)> {
    tags.iter()
        .filter_map(|tag| text.find(tag).map(|at| (at, *tag)))
        .min_by_key(|(at, tag)| (*at, std::cmp::Reverse(tag.len())))
}

/// Length in bytes of the longest suffix of `text` that is a proper prefix of
/// any tag in `tags`.
fn longest_tag_prefix_suffix(text: &str, tags: &[&str]) -> usize {
    let mut best = 0;
    for tag in tags {
        for (i, _) in tag.char_indices().skip(1) {
            if i > best && text.ends_with(&tag[..i]) {
                best = i;
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive a filter over a sequence of deltas, returning the joined channels.
    fn run(deltas: &[&str]) -> ThinkSplit {
        let mut filter = ThinkStreamFilter::new(true);
        let mut all = ThinkSplit::default();
        for delta in deltas {
            let split = filter.push(delta);
            all.reasoning.push_str(&split.reasoning);
            all.text.push_str(&split.text);
        }
        let tail = filter.finish();
        all.reasoning.push_str(&tail.reasoning);
        all.text.push_str(&tail.text);
        all
    }

    #[test]
    fn splits_a_complete_think_block() {
        let out = run(&["<think>musing</think>\n\nHello!"]);
        assert_eq!(out.reasoning, "musing");
        assert_eq!(out.text, "Hello!");
    }

    #[test]
    fn recognizes_tags_split_across_deltas() {
        let out = run(&["<thi", "nk>mus", "ing</th", "ink>Hi"]);
        assert_eq!(out.reasoning, "musing");
        assert_eq!(out.text, "Hi");
    }

    #[test]
    fn streaming_bare_close_tag_is_chunk_invariant_visible_text() {
        let out = run(&["musing", "</think>", "Hi"]);
        assert_eq!(out.reasoning, "");
        assert_eq!(out.text, "musing</think>Hi");
        assert_eq!(out, run(&["musing</think>Hi"]));
    }

    #[test]
    fn passes_plain_text_through_untouched() {
        let out = run(&["Hello", " world"]);
        assert_eq!(out.reasoning, "");
        assert_eq!(out.text, "Hello world");
        assert!(!ThinkStreamFilter::new(true).fired());
    }

    #[test]
    fn preserves_plain_text_leading_whitespace() {
        assert_eq!(run(&["  indented"]).text, "  indented");
        assert_eq!(run(&["\n", "  indented"]).text, "\n  indented");
    }

    #[test]
    fn explicit_tags_are_chunk_invariant() {
        let expected = run(&["<think>musing</think>Hi"]);
        assert_eq!(expected, run(&["<thi", "nk>musing</th", "ink>Hi"]));
        assert_eq!(expected, run(&["<think>musing</think>", "Hi"]));
    }

    #[test]
    fn preserves_whitespace_around_mid_message_think_block() {
        let out = run(&["prefix ", "<think>x</think>", " suffix"]);
        assert_eq!(out.reasoning, "x");
        assert_eq!(out.text, "prefix  suffix");
    }

    #[test]
    fn close_tag_after_visible_text_is_not_reasoning() {
        // Once real text has been emitted a bare `</think>` is just text; only
        // a matched `<think>…</think>` pair splits.
        let out = run(&["see the </think> tag"]);
        assert_eq!(out.reasoning, "");
        assert_eq!(out.text, "see the </think> tag");
    }

    #[test]
    fn unterminated_think_block_stays_reasoning() {
        let out = run(&["<think>cut off mid-thought"]);
        assert_eq!(out.reasoning, "cut off mid-thought");
        assert_eq!(out.text, "");
    }

    #[test]
    fn kimi_markers_are_recognized() {
        let out = run(&["◁think▷musing◁/think▷Hi"]);
        assert_eq!(out.reasoning, "musing");
        assert_eq!(out.text, "Hi");
    }

    #[test]
    fn disabled_filter_passes_markup_through() {
        let mut filter = ThinkStreamFilter::new(false);
        let out = filter.push("<think>musing</think>Hi");
        assert_eq!(out.reasoning, "");
        assert_eq!(out.text, "<think>musing</think>Hi");
    }

    #[test]
    fn heals_blocking_message() {
        let mut message = ChatMessage {
            role: "assistant".into(),
            content: Some(serde_json::Value::String(
                "<think>musing</think>Hello!".into(),
            )),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        heal_chat_message(&mut message);
        assert_eq!(message.text_content(), "Hello!");
        assert_eq!(message.reasoning_content.as_deref(), Some("musing"));
    }

    #[test]
    fn heals_blocking_message_with_prefilled_open_tag() {
        let mut message = ChatMessage {
            role: "assistant".into(),
            content: Some("musing</think>Hi".into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        heal_chat_message(&mut message);
        assert_eq!(message.text_content(), "Hi");
        assert_eq!(message.reasoning_content.as_deref(), Some("musing"));
    }

    #[test]
    fn removes_empty_blocking_think_block() {
        let mut message = ChatMessage {
            role: "assistant".into(),
            content: Some("<think></think>".into()),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        };
        heal_chat_message(&mut message);
        assert!(message.content.is_none());
    }
}
