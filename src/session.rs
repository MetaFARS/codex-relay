use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::warn;
use uuid::Uuid;

use crate::types::ChatMessage;

pub const DEFAULT_MAX_SESSIONS: usize = 256;
pub const DEFAULT_MAX_SESSION_BYTES: usize = 512 * 1024 * 1024;
pub const DEFAULT_SESSION_TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// Maps response_id → accumulated message history for that session.
/// Codex uses `previous_response_id` to continue a conversation; we maintain
/// the full messages[] here so each Chat Completions call is self-contained.
///
/// Also maintains call_id → reasoning_content so that thinking-capable models
/// (e.g. kimi-k2.6) can have their reasoning_content round-tripped back when
/// Codex replays tool-call history in subsequent requests.
///
/// For assistant messages without tool calls (pure text), reasoning_content
/// is indexed by a fingerprint of the prior messages + assistant content,
/// so it can be recovered when Codex replays the full conversation in `input`
/// without using `previous_response_id`.
#[derive(Clone)]
pub struct SessionStore {
    state: Arc<Mutex<SessionState>>,
}

struct SessionState {
    sessions: HashMap<String, SessionEntry>,
    session_order: VecDeque<String>,
    reasoning: HashMap<String, StoredString>,
    reasoning_order: VecDeque<String>,
    /// fingerprint(prior_messages, assistant_content) -> reasoning_content
    turn_reasoning: HashMap<u64, StoredString>,
    turn_reasoning_order: VecDeque<u64>,
    stored_bytes: usize,
    max_sessions: usize,
    max_stored_bytes: usize,
    ttl: Duration,
}

struct SessionEntry {
    messages: Vec<ChatMessage>,
    bytes: usize,
    last_used_at: Instant,
}

struct StoredString {
    value: String,
    bytes: usize,
    last_used_at: Instant,
}

impl SessionStore {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::with_limits_and_ttl(
            DEFAULT_MAX_SESSIONS,
            DEFAULT_MAX_SESSION_BYTES,
            DEFAULT_SESSION_TTL,
        )
    }

    #[allow(dead_code)]
    pub fn with_limits(max_sessions: usize, max_stored_bytes: usize) -> Self {
        Self::with_limits_and_ttl(max_sessions, max_stored_bytes, DEFAULT_SESSION_TTL)
    }

    pub fn with_limits_and_ttl(
        max_sessions: usize,
        max_stored_bytes: usize,
        ttl: Duration,
    ) -> Self {
        Self {
            state: Arc::new(Mutex::new(SessionState {
                sessions: HashMap::new(),
                session_order: VecDeque::new(),
                reasoning: HashMap::new(),
                reasoning_order: VecDeque::new(),
                turn_reasoning: HashMap::new(),
                turn_reasoning_order: VecDeque::new(),
                stored_bytes: 0,
                max_sessions: max_sessions.max(1),
                max_stored_bytes: max_stored_bytes.max(1),
                ttl: ttl.max(Duration::from_secs(1)),
            })),
        }
    }

    /// Store reasoning_content keyed by the tool call_id so it can be
    /// injected back when the same call_id appears in a subsequent request.
    pub fn store_reasoning(&self, call_id: String, reasoning: String) {
        if !reasoning.is_empty() {
            let mut state = self.state.lock().expect("session store mutex poisoned");
            state.insert_reasoning(call_id, reasoning);
            state.enforce_limits();
        }
    }

    /// Look up stored reasoning_content for a call_id.
    pub fn get_reasoning(&self, call_id: &str) -> Option<String> {
        let mut state = self.state.lock().expect("session store mutex poisoned");
        state.enforce_limits();
        let value = state.reasoning.get(call_id).map(|v| v.value.clone());
        if value.is_some() {
            state.touch_reasoning(call_id);
        }
        value
    }

    /// Store reasoning_content for an assistant turn, keyed by a fingerprint
    /// of the assistant message content and tool calls.
    pub fn store_turn_reasoning(
        &self,
        _prior: &[ChatMessage],
        assistant: &ChatMessage,
        reasoning: String,
    ) {
        if !reasoning.is_empty() {
            let mut state = self.state.lock().expect("session store mutex poisoned");

            // Store under content-only key so lookups work even when Codex
            // replays the assistant text and function_calls as separate items.
            let content = assistant.text_content();
            if !content.is_empty() {
                let key = Self::content_key(content);
                state.insert_turn_reasoning(key, reasoning.clone());
            }
            // Also store under each tool call_id (existing mechanism).
            if let Some(tcs) = &assistant.tool_calls {
                for tc in tcs {
                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                        if !id.is_empty() {
                            state.insert_reasoning(id.to_string(), reasoning.clone());
                        }
                    }
                }
            }
            state.enforce_limits();
        }
    }

    /// Look up reasoning_content for an assistant turn by its text content.
    pub fn get_turn_reasoning(
        &self,
        _prior: &[ChatMessage],
        assistant: &ChatMessage,
    ) -> Option<String> {
        let content = assistant.text_content();
        if content.is_empty() {
            return None;
        }
        let key = Self::content_key(content);
        let mut state = self.state.lock().expect("session store mutex poisoned");
        state.enforce_limits();
        let value = state.turn_reasoning.get(&key).map(|v| v.value.clone());
        if value.is_some() {
            state.touch_turn_reasoning(key);
        }
        value
    }

    /// Hash assistant message content for turn-level reasoning lookup.
    fn content_key(content: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        hasher.finish()
    }

    /// Retrieve history for a prior response_id, or empty vec if not found.
    pub fn get_history(&self, response_id: &str) -> Vec<ChatMessage> {
        let mut state = self.state.lock().expect("session store mutex poisoned");
        state.enforce_limits();
        let messages = state
            .sessions
            .get(response_id)
            .map(|entry| entry.messages.clone())
            .unwrap_or_default();
        if !messages.is_empty() {
            state.touch_session(response_id);
        }
        messages
    }

    /// Allocate a fresh response_id without storing anything yet.
    /// Use with save_with_id() for the streaming path.
    pub fn new_id(&self) -> String {
        format!("resp_{}", Uuid::new_v4().simple())
    }

    /// Store under a pre-allocated response_id (streaming path).
    pub fn save_with_id(&self, id: String, messages: Vec<ChatMessage>) {
        let mut state = self.state.lock().expect("session store mutex poisoned");
        state.insert_session(id, messages);
        state.enforce_limits();
    }

    /// Allocate an id and store atomically (non-streaming path).
    pub fn save(&self, messages: Vec<ChatMessage>) -> String {
        let id = self.new_id();
        self.save_with_id(id.clone(), messages);
        id
    }

    /// Drop expired or over-budget retained state.
    pub fn cleanup(&self) {
        let mut state = self.state.lock().expect("session store mutex poisoned");
        state.enforce_limits();
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    fn insert_session(&mut self, id: String, messages: Vec<ChatMessage>) {
        let bytes = messages_bytes(&messages);
        if bytes > self.max_stored_bytes {
            self.remove_session(&id);
            warn!(
                "session {id} is {} bytes, above {} byte retention limit; not caching history",
                bytes, self.max_stored_bytes
            );
            return;
        }

        self.remove_session(&id);
        let now = Instant::now();
        self.stored_bytes = self.stored_bytes.saturating_add(bytes);
        self.sessions.insert(
            id.clone(),
            SessionEntry {
                messages,
                bytes,
                last_used_at: now,
            },
        );
        self.session_order.push_back(id);
    }

    fn insert_reasoning(&mut self, call_id: String, reasoning: String) {
        if let Some(old) = self.reasoning.remove(&call_id) {
            self.stored_bytes = self.stored_bytes.saturating_sub(old.bytes);
        }
        self.reasoning_order.retain(|key| key != &call_id);

        let bytes = call_id.len().saturating_add(reasoning.len());
        self.stored_bytes = self.stored_bytes.saturating_add(bytes);
        self.reasoning.insert(
            call_id.clone(),
            StoredString {
                value: reasoning,
                bytes,
                last_used_at: Instant::now(),
            },
        );
        self.reasoning_order.push_back(call_id);
    }

    fn insert_turn_reasoning(&mut self, key: u64, reasoning: String) {
        if let Some(old) = self.turn_reasoning.remove(&key) {
            self.stored_bytes = self.stored_bytes.saturating_sub(old.bytes);
        }
        self.turn_reasoning_order
            .retain(|existing| *existing != key);

        let bytes = std::mem::size_of::<u64>().saturating_add(reasoning.len());
        self.stored_bytes = self.stored_bytes.saturating_add(bytes);
        self.turn_reasoning.insert(
            key,
            StoredString {
                value: reasoning,
                bytes,
                last_used_at: Instant::now(),
            },
        );
        self.turn_reasoning_order.push_back(key);
    }

    fn enforce_limits(&mut self) {
        self.remove_expired();

        while self.sessions.len() > self.max_sessions {
            self.remove_oldest_session();
        }

        while self.stored_bytes > self.max_stored_bytes && self.sessions.len() > 1 {
            self.remove_oldest_session();
        }

        while self.stored_bytes > self.max_stored_bytes && !self.reasoning_order.is_empty() {
            self.remove_oldest_reasoning();
        }

        while self.stored_bytes > self.max_stored_bytes && !self.turn_reasoning_order.is_empty() {
            self.remove_oldest_turn_reasoning();
        }
    }

    fn remove_expired(&mut self) {
        let cutoff = Instant::now() - self.ttl;

        while self
            .session_order
            .front()
            .and_then(|id| self.sessions.get(id))
            .is_some_and(|entry| entry.last_used_at <= cutoff)
        {
            self.remove_oldest_session();
        }

        while self
            .reasoning_order
            .front()
            .and_then(|id| self.reasoning.get(id))
            .is_some_and(|entry| entry.last_used_at <= cutoff)
        {
            self.remove_oldest_reasoning();
        }

        while self
            .turn_reasoning_order
            .front()
            .and_then(|key| self.turn_reasoning.get(key))
            .is_some_and(|entry| entry.last_used_at <= cutoff)
        {
            self.remove_oldest_turn_reasoning();
        }
    }

    fn remove_oldest_session(&mut self) {
        if let Some(id) = self.session_order.pop_front() {
            self.remove_session_entry(&id);
        }
    }

    fn remove_session(&mut self, id: &str) {
        self.session_order.retain(|existing| existing != id);
        self.remove_session_entry(id);
    }

    fn remove_session_entry(&mut self, id: &str) {
        if let Some(entry) = self.sessions.remove(id) {
            self.stored_bytes = self.stored_bytes.saturating_sub(entry.bytes);
        }
    }

    fn remove_oldest_reasoning(&mut self) {
        if let Some(key) = self.reasoning_order.pop_front() {
            if let Some(entry) = self.reasoning.remove(&key) {
                self.stored_bytes = self.stored_bytes.saturating_sub(entry.bytes);
            }
        }
    }

    fn remove_oldest_turn_reasoning(&mut self) {
        if let Some(key) = self.turn_reasoning_order.pop_front() {
            if let Some(entry) = self.turn_reasoning.remove(&key) {
                self.stored_bytes = self.stored_bytes.saturating_sub(entry.bytes);
            }
        }
    }

    fn touch_session(&mut self, id: &str) {
        if let Some(entry) = self.sessions.get_mut(id) {
            entry.last_used_at = Instant::now();
        }
        self.session_order.retain(|existing| existing != id);
        self.session_order.push_back(id.to_string());
    }

    fn touch_reasoning(&mut self, call_id: &str) {
        if let Some(entry) = self.reasoning.get_mut(call_id) {
            entry.last_used_at = Instant::now();
        }
        self.reasoning_order.retain(|existing| existing != call_id);
        self.reasoning_order.push_back(call_id.to_string());
    }

    fn touch_turn_reasoning(&mut self, key: u64) {
        if let Some(entry) = self.turn_reasoning.get_mut(&key) {
            entry.last_used_at = Instant::now();
        }
        self.turn_reasoning_order
            .retain(|existing| *existing != key);
        self.turn_reasoning_order.push_back(key);
    }
}

fn messages_bytes(messages: &[ChatMessage]) -> usize {
    messages.iter().map(message_bytes).sum()
}

fn message_bytes(message: &ChatMessage) -> usize {
    message
        .role
        .len()
        .saturating_add(
            message
                .content
                .as_ref()
                .map(value_bytes)
                .unwrap_or_default(),
        )
        .saturating_add(
            message
                .reasoning_content
                .as_ref()
                .map(String::len)
                .unwrap_or_default(),
        )
        .saturating_add(
            message
                .tool_calls
                .as_ref()
                .map(|calls| calls.iter().map(value_bytes).sum())
                .unwrap_or_default(),
        )
        .saturating_add(
            message
                .tool_call_id
                .as_ref()
                .map(String::len)
                .unwrap_or_default(),
        )
        .saturating_add(message.name.as_ref().map(String::len).unwrap_or_default())
}

fn value_bytes(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => 8,
        serde_json::Value::String(s) => s.len(),
        serde_json::Value::Array(values) => values.iter().map(value_bytes).sum(),
        serde_json::Value::Object(map) => map
            .iter()
            .map(|(key, value)| key.len().saturating_add(value_bytes(value)))
            .sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    fn msg(role: &str, content: Option<&str>) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: content.map(|s| serde_json::Value::String(s.into())),
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn test_store_and_get_reasoning() {
        let store = SessionStore::new();
        store.store_reasoning("call_1".into(), "think".into());
        assert_eq!(store.get_reasoning("call_1"), Some("think".into()));
    }

    #[test]
    fn test_get_reasoning_missing() {
        let store = SessionStore::new();
        assert_eq!(store.get_reasoning("nonexistent"), None);
    }

    #[test]
    fn test_empty_reasoning_not_stored() {
        let store = SessionStore::new();
        store.store_reasoning("call_e".into(), "".into());
        assert_eq!(store.get_reasoning("call_e"), None);
    }

    #[test]
    fn test_turn_reasoning_by_content() {
        let store = SessionStore::new();
        let assistant = msg("assistant", Some("hello world"));
        store.store_turn_reasoning(&[], &assistant, "deep thought".into());
        assert_eq!(
            store.get_turn_reasoning(&[], &assistant),
            Some("deep thought".into())
        );
    }

    #[test]
    fn test_turn_reasoning_empty_content() {
        let store = SessionStore::new();
        let assistant = msg("assistant", Some(""));
        store.store_turn_reasoning(&[], &assistant, "reason".into());
        assert_eq!(store.get_turn_reasoning(&[], &assistant), None);
    }

    #[test]
    fn test_turn_reasoning_also_stores_call_ids() {
        let store = SessionStore::new();
        let mut assistant = msg("assistant", Some("hi"));
        assistant.tool_calls = Some(vec![serde_json::json!({
            "id": "call_123",
            "type": "function",
            "function": {"name": "exec", "arguments": "{}"}
        })]);
        store.store_turn_reasoning(&[], &assistant, "reason_tc".into());
        assert_eq!(store.get_reasoning("call_123"), Some("reason_tc".into()));
    }

    #[test]
    fn test_history_save_and_get() {
        let store = SessionStore::new();
        let msgs = vec![msg("user", Some("hi")), msg("assistant", Some("hey"))];
        let id = store.save(msgs.clone());
        let got = store.get_history(&id);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].text_content(), "hi");

        // save_with_id
        let id2 = store.new_id();
        store.save_with_id(id2.clone(), vec![msg("user", Some("q"))]);
        assert_eq!(store.get_history(&id2).len(), 1);
    }

    #[test]
    fn test_content_key_deterministic() {
        let a = SessionStore::content_key("same text");
        let b = SessionStore::content_key("same text");
        assert_eq!(a, b);
        let c = SessionStore::content_key("different");
        assert_ne!(a, c);
    }

    #[test]
    fn test_evicts_oldest_session_by_count() {
        let store = SessionStore::with_limits(2, 1024);
        let id1 = store.save(vec![msg("user", Some("one"))]);
        let id2 = store.save(vec![msg("user", Some("two"))]);
        let id3 = store.save(vec![msg("user", Some("three"))]);

        assert!(store.get_history(&id1).is_empty());
        assert_eq!(store.get_history(&id2).len(), 1);
        assert_eq!(store.get_history(&id3).len(), 1);
    }

    #[test]
    fn test_evicts_oldest_session_by_bytes() {
        let store = SessionStore::with_limits(10, 64);
        let id1 = store.save(vec![msg("user", Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))]);
        let id2 = store.save(vec![msg("user", Some("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"))]);
        let id3 = store.save(vec![msg("user", Some("c"))]);

        assert!(store.get_history(&id1).is_empty());
        assert_eq!(store.get_history(&id2).len(), 1);
        assert_eq!(store.get_history(&id3).len(), 1);
    }

    #[test]
    fn test_oversized_session_not_cached() {
        let store = SessionStore::with_limits(10, 10);
        let id = store.save(vec![msg("user", Some("this message is too large"))]);
        assert!(store.get_history(&id).is_empty());
    }

    #[test]
    fn test_reasoning_entries_are_bounded_by_bytes() {
        let store = SessionStore::with_limits(10, 36);
        store.store_reasoning("call_1".into(), "aaaaaaaaaaaaaaaaaaaaaaaa".into());
        store.store_reasoning("call_2".into(), "bbbbbbbbbbbbbbbbbbbbbbbb".into());

        assert_eq!(store.get_reasoning("call_1"), None);
        assert_eq!(
            store.get_reasoning("call_2"),
            Some("bbbbbbbbbbbbbbbbbbbbbbbb".into())
        );
    }

    #[test]
    fn test_cleanup_removes_expired_session() {
        let store = SessionStore::with_limits_and_ttl(10, 1024, Duration::from_secs(60));
        let id = store.save(vec![msg("user", Some("old"))]);

        {
            let mut state = store.state.lock().unwrap();
            state.sessions.get_mut(&id).unwrap().last_used_at =
                Instant::now() - Duration::from_secs(61);
        }

        store.cleanup();
        assert!(store.get_history(&id).is_empty());
    }

    #[test]
    fn test_cleanup_removes_expired_reasoning() {
        let store = SessionStore::with_limits_and_ttl(10, 1024, Duration::from_secs(60));
        store.store_reasoning("call_old".into(), "old thought".into());

        {
            let mut state = store.state.lock().unwrap();
            state.reasoning.get_mut("call_old").unwrap().last_used_at =
                Instant::now() - Duration::from_secs(61);
        }

        store.cleanup();
        assert_eq!(store.get_reasoning("call_old"), None);
    }
}
