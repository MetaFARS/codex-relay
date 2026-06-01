use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
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
    disk: Option<DiskStore>,
}

struct SessionEntry {
    messages: Option<Vec<ChatMessage>>,
    bytes: usize,
    last_used_at: SystemTime,
}

struct StoredString {
    value: Option<String>,
    bytes: usize,
    last_used_at: SystemTime,
}

struct DiskStore {
    root: PathBuf,
}

#[derive(Serialize, Deserialize)]
struct DiskSessionRecord {
    schema_version: u32,
    response_id: String,
    created_at_unix_ms: u128,
    last_used_at_unix_ms: u128,
    bytes: usize,
    messages: Vec<ChatMessage>,
}

#[derive(Serialize, Deserialize)]
struct DiskReasoningRecord {
    schema_version: u32,
    key: String,
    created_at_unix_ms: u128,
    last_used_at_unix_ms: u128,
    bytes: usize,
    value: String,
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
        Self::with_optional_disk(max_sessions, max_stored_bytes, ttl, None)
    }

    pub fn with_disk_limits_and_ttl(
        root: impl AsRef<Path>,
        max_sessions: usize,
        max_stored_bytes: usize,
        ttl: Duration,
    ) -> io::Result<Self> {
        let disk = DiskStore::new(root.as_ref())?;
        Ok(Self::with_optional_disk(
            max_sessions,
            max_stored_bytes,
            ttl,
            Some(disk),
        ))
    }

    fn with_optional_disk(
        max_sessions: usize,
        max_stored_bytes: usize,
        ttl: Duration,
        disk: Option<DiskStore>,
    ) -> Self {
        let mut state = SessionState {
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
            disk,
        };
        state.load_disk_index();
        state.enforce_limits();

        Self {
            state: Arc::new(Mutex::new(state)),
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
        let value = state.load_reasoning_value(call_id);
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
        let value = state.load_turn_reasoning_value(key);
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
        let messages = state.load_session_messages(response_id);
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
    fn load_disk_index(&mut self) {
        let Some(disk) = &self.disk else {
            return;
        };

        let mut sessions = disk.load_sessions();
        sessions.sort_by_key(|record| record.last_used_at_unix_ms);
        for record in sessions {
            let last_used_at = system_time_from_millis(record.last_used_at_unix_ms);
            self.stored_bytes = self.stored_bytes.saturating_add(record.bytes);
            self.sessions.insert(
                record.response_id.clone(),
                SessionEntry {
                    messages: None,
                    bytes: record.bytes,
                    last_used_at,
                },
            );
            self.session_order.push_back(record.response_id);
        }

        let mut reasoning = disk.load_reasoning();
        reasoning.sort_by_key(|record| record.last_used_at_unix_ms);
        for record in reasoning {
            let last_used_at = system_time_from_millis(record.last_used_at_unix_ms);
            self.stored_bytes = self.stored_bytes.saturating_add(record.bytes);
            self.reasoning.insert(
                record.key.clone(),
                StoredString {
                    value: None,
                    bytes: record.bytes,
                    last_used_at,
                },
            );
            self.reasoning_order.push_back(record.key);
        }

        let mut turn_reasoning = disk.load_turn_reasoning();
        turn_reasoning.sort_by_key(|record| record.last_used_at_unix_ms);
        for record in turn_reasoning {
            let Ok(key) = record.key.parse::<u64>() else {
                warn!(
                    "ignoring disk turn reasoning record with invalid key {}",
                    record.key
                );
                continue;
            };
            let last_used_at = system_time_from_millis(record.last_used_at_unix_ms);
            self.stored_bytes = self.stored_bytes.saturating_add(record.bytes);
            self.turn_reasoning.insert(
                key,
                StoredString {
                    value: None,
                    bytes: record.bytes,
                    last_used_at,
                },
            );
            self.turn_reasoning_order.push_back(key);
        }
    }

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
        let now = SystemTime::now();
        let messages_to_store = if let Some(disk) = &self.disk {
            if let Err(e) = disk.write_session(&id, now, now, bytes, &messages) {
                warn!("failed to persist session {id}: {e}");
                Some(messages)
            } else {
                None
            }
        } else {
            Some(messages)
        };
        self.stored_bytes = self.stored_bytes.saturating_add(bytes);
        self.sessions.insert(
            id.clone(),
            SessionEntry {
                messages: messages_to_store,
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
        let now = SystemTime::now();
        let value_to_store = if let Some(disk) = &self.disk {
            if let Err(e) = disk.write_reasoning(&call_id, now, now, bytes, &reasoning) {
                warn!("failed to persist reasoning {call_id}: {e}");
                Some(reasoning)
            } else {
                None
            }
        } else {
            Some(reasoning)
        };
        self.stored_bytes = self.stored_bytes.saturating_add(bytes);
        self.reasoning.insert(
            call_id.clone(),
            StoredString {
                value: value_to_store,
                bytes,
                last_used_at: now,
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
        let key_string = key.to_string();
        let now = SystemTime::now();
        let value_to_store = if let Some(disk) = &self.disk {
            if let Err(e) = disk.write_turn_reasoning(&key_string, now, now, bytes, &reasoning) {
                warn!("failed to persist turn reasoning {key}: {e}");
                Some(reasoning)
            } else {
                None
            }
        } else {
            Some(reasoning)
        };
        self.stored_bytes = self.stored_bytes.saturating_add(bytes);
        self.turn_reasoning.insert(
            key,
            StoredString {
                value: value_to_store,
                bytes,
                last_used_at: now,
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
        let cutoff = SystemTime::now() - self.ttl;

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
        if let Some(disk) = &self.disk {
            disk.remove_session(id);
        }
    }

    fn remove_oldest_reasoning(&mut self) {
        if let Some(key) = self.reasoning_order.pop_front() {
            if let Some(entry) = self.reasoning.remove(&key) {
                self.stored_bytes = self.stored_bytes.saturating_sub(entry.bytes);
            }
            if let Some(disk) = &self.disk {
                disk.remove_reasoning(&key);
            }
        }
    }

    fn remove_oldest_turn_reasoning(&mut self) {
        if let Some(key) = self.turn_reasoning_order.pop_front() {
            if let Some(entry) = self.turn_reasoning.remove(&key) {
                self.stored_bytes = self.stored_bytes.saturating_sub(entry.bytes);
            }
            if let Some(disk) = &self.disk {
                disk.remove_turn_reasoning(key);
            }
        }
    }

    fn touch_session(&mut self, id: &str) {
        let now = SystemTime::now();
        if let Some(entry) = self.sessions.get_mut(id) {
            entry.last_used_at = now;
        }
        if let Some(disk) = &self.disk {
            if let Some(mut record) = disk.read_session(id) {
                record.last_used_at_unix_ms = system_time_millis(now);
                if let Err(e) = disk.write_session_record(&record) {
                    warn!("failed to touch disk session {id}: {e}");
                }
            }
        }
        self.session_order.retain(|existing| existing != id);
        self.session_order.push_back(id.to_string());
    }

    fn touch_reasoning(&mut self, call_id: &str) {
        let now = SystemTime::now();
        if let Some(entry) = self.reasoning.get_mut(call_id) {
            entry.last_used_at = now;
        }
        if let Some(disk) = &self.disk {
            if let Some(mut record) = disk.read_reasoning(call_id) {
                record.last_used_at_unix_ms = system_time_millis(now);
                if let Err(e) = disk.write_reasoning_record(&record) {
                    warn!("failed to touch disk reasoning {call_id}: {e}");
                }
            }
        }
        self.reasoning_order.retain(|existing| existing != call_id);
        self.reasoning_order.push_back(call_id.to_string());
    }

    fn touch_turn_reasoning(&mut self, key: u64) {
        let now = SystemTime::now();
        if let Some(entry) = self.turn_reasoning.get_mut(&key) {
            entry.last_used_at = now;
        }
        if let Some(disk) = &self.disk {
            if let Some(mut record) = disk.read_turn_reasoning(key) {
                record.last_used_at_unix_ms = system_time_millis(now);
                if let Err(e) = disk.write_turn_reasoning_record(&record) {
                    warn!("failed to touch disk turn reasoning {key}: {e}");
                }
            }
        }
        self.turn_reasoning_order
            .retain(|existing| *existing != key);
        self.turn_reasoning_order.push_back(key);
    }

    fn load_session_messages(&self, id: &str) -> Vec<ChatMessage> {
        let Some(entry) = self.sessions.get(id) else {
            return Vec::new();
        };
        if let Some(messages) = &entry.messages {
            return messages.clone();
        }
        self.disk
            .as_ref()
            .and_then(|disk| disk.read_session(id))
            .map(|record| record.messages)
            .unwrap_or_default()
    }

    fn load_reasoning_value(&self, call_id: &str) -> Option<String> {
        let entry = self.reasoning.get(call_id)?;
        if let Some(value) = &entry.value {
            return Some(value.clone());
        }
        self.disk
            .as_ref()
            .and_then(|disk| disk.read_reasoning(call_id))
            .map(|record| record.value)
    }

    fn load_turn_reasoning_value(&self, key: u64) -> Option<String> {
        let entry = self.turn_reasoning.get(&key)?;
        if let Some(value) = &entry.value {
            return Some(value.clone());
        }
        self.disk
            .as_ref()
            .and_then(|disk| disk.read_turn_reasoning(key))
            .map(|record| record.value)
    }
}

impl DiskStore {
    fn new(root: &Path) -> io::Result<Self> {
        let store = Self {
            root: root.to_path_buf(),
        };
        fs::create_dir_all(store.sessions_dir())?;
        fs::create_dir_all(store.reasoning_dir())?;
        fs::create_dir_all(store.turns_dir())?;
        Ok(store)
    }

    fn sessions_dir(&self) -> PathBuf {
        self.root.join("sessions")
    }

    fn reasoning_dir(&self) -> PathBuf {
        self.root.join("reasoning")
    }

    fn turns_dir(&self) -> PathBuf {
        self.root.join("turns")
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.sessions_dir().join(format!("{}.json", encode_key(id)))
    }

    fn reasoning_path(&self, key: &str) -> PathBuf {
        self.reasoning_dir()
            .join(format!("{}.json", encode_key(key)))
    }

    fn turn_path(&self, key: u64) -> PathBuf {
        self.turns_dir().join(format!("{key}.json"))
    }

    fn write_session(
        &self,
        id: &str,
        created_at: SystemTime,
        last_used_at: SystemTime,
        bytes: usize,
        messages: &[ChatMessage],
    ) -> io::Result<()> {
        self.write_session_record(&DiskSessionRecord {
            schema_version: 1,
            response_id: id.to_string(),
            created_at_unix_ms: system_time_millis(created_at),
            last_used_at_unix_ms: system_time_millis(last_used_at),
            bytes,
            messages: messages.to_vec(),
        })
    }

    fn write_session_record(&self, record: &DiskSessionRecord) -> io::Result<()> {
        write_json_atomic(&self.session_path(&record.response_id), record)
    }

    fn read_session(&self, id: &str) -> Option<DiskSessionRecord> {
        read_json(&self.session_path(id))
    }

    fn write_reasoning(
        &self,
        key: &str,
        created_at: SystemTime,
        last_used_at: SystemTime,
        bytes: usize,
        value: &str,
    ) -> io::Result<()> {
        self.write_reasoning_record(&DiskReasoningRecord {
            schema_version: 1,
            key: key.to_string(),
            created_at_unix_ms: system_time_millis(created_at),
            last_used_at_unix_ms: system_time_millis(last_used_at),
            bytes,
            value: value.to_string(),
        })
    }

    fn write_reasoning_record(&self, record: &DiskReasoningRecord) -> io::Result<()> {
        write_json_atomic(&self.reasoning_path(&record.key), record)
    }

    fn read_reasoning(&self, key: &str) -> Option<DiskReasoningRecord> {
        read_json(&self.reasoning_path(key))
    }

    fn write_turn_reasoning(
        &self,
        key: &str,
        created_at: SystemTime,
        last_used_at: SystemTime,
        bytes: usize,
        value: &str,
    ) -> io::Result<()> {
        self.write_turn_reasoning_record(&DiskReasoningRecord {
            schema_version: 1,
            key: key.to_string(),
            created_at_unix_ms: system_time_millis(created_at),
            last_used_at_unix_ms: system_time_millis(last_used_at),
            bytes,
            value: value.to_string(),
        })
    }

    fn write_turn_reasoning_record(&self, record: &DiskReasoningRecord) -> io::Result<()> {
        let key = record.key.parse::<u64>().map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid turn key: {e}"),
            )
        })?;
        write_json_atomic(&self.turn_path(key), record)
    }

    fn read_turn_reasoning(&self, key: u64) -> Option<DiskReasoningRecord> {
        read_json(&self.turn_path(key))
    }

    fn load_sessions(&self) -> Vec<DiskSessionRecord> {
        load_records(&self.sessions_dir())
    }

    fn load_reasoning(&self) -> Vec<DiskReasoningRecord> {
        load_records(&self.reasoning_dir())
    }

    fn load_turn_reasoning(&self) -> Vec<DiskReasoningRecord> {
        load_records(&self.turns_dir())
    }

    fn remove_session(&self, id: &str) {
        remove_file_if_exists(&self.session_path(id));
    }

    fn remove_reasoning(&self, key: &str) {
        remove_file_if_exists(&self.reasoning_path(key));
    }

    fn remove_turn_reasoning(&self, key: u64) {
        remove_file_if_exists(&self.turn_path(key));
    }
}

fn load_records<T: for<'de> Deserialize<'de>>(dir: &Path) -> Vec<T> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            read_json(&path).or_else(|| {
                warn!("ignoring corrupt disk history record {}", path.display());
                None
            })
        })
        .collect()
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    let tmp = path.with_extension(format!("json.tmp-{}", Uuid::new_v4().simple()));
    let bytes = serde_json::to_vec_pretty(value).map_err(io::Error::other)?;
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)?;
    Ok(())
}

fn remove_file_if_exists(path: &Path) {
    if let Err(e) = fs::remove_file(path) {
        if e.kind() != io::ErrorKind::NotFound {
            warn!(
                "failed to remove disk history record {}: {e}",
                path.display()
            );
        }
    }
}

fn encode_key(key: &str) -> String {
    let mut out = String::new();
    for byte in key.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'-' | b'.' => out.push(byte as char),
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn system_time_millis(time: SystemTime) -> u128 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

fn system_time_from_millis(millis: u128) -> SystemTime {
    UNIX_EPOCH + Duration::from_millis(millis.min(u64::MAX as u128) as u64)
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

    fn temp_history_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("codex-relay-{name}-{}", Uuid::new_v4().simple()));
        fs::create_dir_all(&dir).unwrap();
        dir
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
                SystemTime::now() - Duration::from_secs(61);
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
                SystemTime::now() - Duration::from_secs(61);
        }

        store.cleanup();
        assert_eq!(store.get_reasoning("call_old"), None);
    }

    #[test]
    fn test_disk_store_save_load_history_across_instances() {
        let dir = temp_history_dir("disk-history");
        let id = {
            let store =
                SessionStore::with_disk_limits_and_ttl(&dir, 10, 1024, Duration::from_secs(60))
                    .unwrap();
            store.save(vec![msg("user", Some("hi")), msg("assistant", Some("hey"))])
        };

        let store = SessionStore::with_disk_limits_and_ttl(&dir, 10, 1024, Duration::from_secs(60))
            .unwrap();
        let got = store.get_history(&id);
        assert_eq!(got.len(), 2);
        assert_eq!(got[0].text_content(), "hi");
        assert!(dir.join("sessions").join(format!("{id}.json")).exists());
    }

    #[test]
    fn test_disk_store_reasoning_across_instances() {
        let dir = temp_history_dir("disk-reasoning");
        {
            let store =
                SessionStore::with_disk_limits_and_ttl(&dir, 10, 1024, Duration::from_secs(60))
                    .unwrap();
            store.store_reasoning("call_1".into(), "think".into());
        }

        let store = SessionStore::with_disk_limits_and_ttl(&dir, 10, 1024, Duration::from_secs(60))
            .unwrap();
        assert_eq!(store.get_reasoning("call_1"), Some("think".into()));
    }

    #[test]
    fn test_disk_store_turn_reasoning_across_instances() {
        let dir = temp_history_dir("disk-turn-reasoning");
        let assistant = msg("assistant", Some("hello world"));
        {
            let store =
                SessionStore::with_disk_limits_and_ttl(&dir, 10, 1024, Duration::from_secs(60))
                    .unwrap();
            store.store_turn_reasoning(&[], &assistant, "deep thought".into());
        }

        let store = SessionStore::with_disk_limits_and_ttl(&dir, 10, 1024, Duration::from_secs(60))
            .unwrap();
        assert_eq!(
            store.get_turn_reasoning(&[], &assistant),
            Some("deep thought".into())
        );
    }

    #[test]
    fn test_disk_store_evicts_files_by_count() {
        let dir = temp_history_dir("disk-evict");
        let store =
            SessionStore::with_disk_limits_and_ttl(&dir, 1, 1024, Duration::from_secs(60)).unwrap();
        let id1 = store.save(vec![msg("user", Some("one"))]);
        let id2 = store.save(vec![msg("user", Some("two"))]);

        assert!(store.get_history(&id1).is_empty());
        assert_eq!(store.get_history(&id2).len(), 1);
        assert!(!dir.join("sessions").join(format!("{id1}.json")).exists());
        assert!(dir.join("sessions").join(format!("{id2}.json")).exists());
    }

    #[test]
    fn test_disk_store_ignores_corrupt_session_file() {
        let dir = temp_history_dir("disk-corrupt");
        let sessions = dir.join("sessions");
        fs::create_dir_all(&sessions).unwrap();
        fs::write(sessions.join("resp_bad.json"), b"{not json").unwrap();

        let store = SessionStore::with_disk_limits_and_ttl(&dir, 10, 1024, Duration::from_secs(60))
            .unwrap();
        assert!(store.get_history("resp_bad").is_empty());
    }
}
