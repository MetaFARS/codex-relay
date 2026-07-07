//! Opt-in conversation corpus recorder.
//!
//! When enabled (via `--record-corpus <dir>`), the relay appends the
//! conversation flow of each completed turn to a daily-sharded JSONL file in
//! OpenAI messages format. Each line is an *incremental turn event*: only the
//! messages new to that turn are written, so a full conversation is
//! reconstructed downstream by concatenating the `messages` of every event
//! sharing a `conversation_id`.
//!
//! This subsystem is deliberately independent of [`crate::session::SessionStore`]:
//! the session store is an evictable *continuation cache*, whereas the corpus is
//! an append-only *archive* that is never evicted.
//!
//! Records may contain prompts, tool call arguments, and tool outputs. Treat
//! the output directory as sensitive. Recording is off unless explicitly
//! enabled.

use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use tracing::warn;

use crate::types::ChatMessage;

/// Upper bound on response_id chains tracked in memory for delta computation.
/// If a chain is evicted and a later child request references it, that child
/// simply starts a new conversation — acceptable fragmentation, never data loss.
const DEFAULT_MAX_TRACKED_CHAINS: usize = 4096;

#[derive(Clone)]
pub struct CorpusRecorder {
    inner: Arc<Mutex<CorpusState>>,
}

struct CorpusState {
    dir: PathBuf,
    max_tracked: usize,
    chains: HashMap<String, ChainPos>,
    order: VecDeque<String>,
}

#[derive(Clone)]
struct ChainPos {
    conversation_id: String,
    message_count: usize,
}

impl CorpusRecorder {
    /// Create a recorder that appends to `dir`, creating it if necessary.
    pub fn new(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(CorpusState {
                dir,
                max_tracked: DEFAULT_MAX_TRACKED_CHAINS,
                chains: HashMap::new(),
                order: VecDeque::new(),
            })),
        })
    }

    /// Record one completed turn.
    ///
    /// `messages` is the full conversation for the new `response_id` (replayed
    /// history + this turn's new input + the assistant reply). Only the
    /// incremental delta since the parent turn is persisted.
    pub fn record_turn(
        &self,
        parent_response_id: Option<&str>,
        response_id: &str,
        model: &str,
        messages: &[ChatMessage],
    ) {
        let mut state = self.inner.lock().expect("corpus mutex poisoned");
        state.record(parent_response_id, response_id, model, messages);
    }
}

impl CorpusState {
    fn record(
        &mut self,
        parent: Option<&str>,
        response_id: &str,
        model: &str,
        messages: &[ChatMessage],
    ) {
        let parent_pos = parent.and_then(|p| self.chains.get(p)).cloned();
        let conversation_id = parent_pos
            .as_ref()
            .map(|p| p.conversation_id.clone())
            .unwrap_or_else(|| response_id.to_string());
        let prev_count = parent_pos.as_ref().map(|p| p.message_count).unwrap_or(0);
        let start = prev_count.min(messages.len());
        let delta = &messages[start..];

        if !delta.is_empty() {
            let record = json!({
                "conversation_id": conversation_id,
                "response_id": response_id,
                "parent_response_id": parent,
                "timestamp_unix_ms": now_unix_ms(),
                "model": model,
                "messages": delta,
            });
            if let Err(e) = self.append(&record) {
                warn!("failed to append corpus turn for {response_id}: {e}");
            }
        }

        self.remember(response_id, conversation_id, messages.len());
    }

    fn remember(&mut self, response_id: &str, conversation_id: String, message_count: usize) {
        let existed = self
            .chains
            .insert(
                response_id.to_string(),
                ChainPos {
                    conversation_id,
                    message_count,
                },
            )
            .is_some();
        if existed {
            self.order.retain(|id| id != response_id);
        }
        self.order.push_back(response_id.to_string());

        while self.chains.len() > self.max_tracked {
            match self.order.pop_front() {
                Some(old) => {
                    self.chains.remove(&old);
                }
                None => break,
            }
        }
    }

    fn append(&self, record: &serde_json::Value) -> io::Result<()> {
        let path = self.dir.join(format!("corpus-{}.jsonl", today_utc_date()));
        let mut line = serde_json::to_string(record).map_err(io::Error::other)?;
        line.push('\n');
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        file.write_all(line.as_bytes())
    }
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

/// Current UTC date as `YYYY-MM-DD`, computed without pulling in a date crate.
fn today_utc_date() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let (y, m, d) = civil_from_days((secs / 86_400) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Convert a count of days since the Unix epoch into a `(year, month, day)`
/// Gregorian date. Howard Hinnant's `civil_from_days` algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use uuid::Uuid;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("codex-relay-corpus-{name}-{}", Uuid::new_v4().simple()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: Some(Value::String(content.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    fn read_lines(dir: &Path) -> Vec<Value> {
        let mut out = Vec::new();
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let text = fs::read_to_string(&path).unwrap();
            for line in text.lines().filter(|l| !l.trim().is_empty()) {
                out.push(serde_json::from_str(line).unwrap());
            }
        }
        out
    }

    #[test]
    fn records_full_conversation_on_first_turn() {
        let dir = temp_dir("first-turn");
        let recorder = CorpusRecorder::new(&dir).unwrap();

        let full = vec![
            msg("system", "you are helpful"),
            msg("user", "hi"),
            msg("assistant", "hello"),
        ];
        recorder.record_turn(None, "resp_a", "test-model", &full);

        let lines = read_lines(&dir);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0]["conversation_id"], "resp_a");
        assert_eq!(lines[0]["response_id"], "resp_a");
        assert!(lines[0]["parent_response_id"].is_null());
        assert_eq!(lines[0]["model"], "test-model");
        assert_eq!(lines[0]["messages"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn records_only_delta_on_continued_turn() {
        let dir = temp_dir("delta");
        let recorder = CorpusRecorder::new(&dir).unwrap();

        let turn1 = vec![
            msg("system", "sys"),
            msg("user", "q1"),
            msg("assistant", "a1"),
        ];
        recorder.record_turn(None, "resp_1", "m", &turn1);

        // Second turn replays turn1 as history and appends new input + reply.
        let turn2 = vec![
            msg("system", "sys"),
            msg("user", "q1"),
            msg("assistant", "a1"),
            msg("user", "q2"),
            msg("assistant", "a2"),
        ];
        recorder.record_turn(Some("resp_1"), "resp_2", "m", &turn2);

        let lines = read_lines(&dir);
        assert_eq!(lines.len(), 2);

        // Both events share the conversation id (the chain root).
        assert_eq!(lines[1]["conversation_id"], "resp_1");
        assert_eq!(lines[1]["response_id"], "resp_2");
        assert_eq!(lines[1]["parent_response_id"], "resp_1");

        let delta = lines[1]["messages"].as_array().unwrap();
        assert_eq!(delta.len(), 2, "only the new turn's messages are recorded");
        assert_eq!(delta[0]["content"], "q2");
        assert_eq!(delta[1]["content"], "a2");
    }

    #[test]
    fn preserves_tool_calls_and_reasoning() {
        let dir = temp_dir("tool-reasoning");
        let recorder = CorpusRecorder::new(&dir).unwrap();

        let mut assistant = msg("assistant", "");
        assistant.reasoning_content = Some("let me think".into());
        assistant.tool_calls = Some(vec![json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "exec", "arguments": "{\"cmd\":\"ls\"}"}
        })]);
        let mut tool = msg("tool", "file.txt");
        tool.tool_call_id = Some("call_1".into());

        let full = vec![msg("user", "list files"), assistant, tool];
        recorder.record_turn(None, "resp_t", "m", &full);

        let lines = read_lines(&dir);
        let messages = lines[0]["messages"].as_array().unwrap();
        assert_eq!(messages[1]["reasoning_content"], "let me think");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "exec");
        assert_eq!(messages[2]["tool_call_id"], "call_1");
    }

    #[test]
    fn isolated_child_with_unknown_parent_starts_new_conversation() {
        let dir = temp_dir("isolated");
        let recorder = CorpusRecorder::new(&dir).unwrap();

        // Parent id was never recorded (e.g. spawn-child isolation cleared history).
        let full = vec![msg("user", "child task"), msg("assistant", "done")];
        recorder.record_turn(Some("resp_unknown"), "resp_child", "m", &full);

        let lines = read_lines(&dir);
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0]["conversation_id"], "resp_child",
            "unknown parent => new conversation rooted at self"
        );
        assert_eq!(lines[0]["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(20_547), (2026, 4, 4));
    }
}
