//! Codex CLI 0.128 wire-shape compatibility tests (offline).
//!
//! Pinned versions covered by fixtures in `tests/fixtures/codex_0_128_0/`:
//!   - Codex CLI 0.128.0 (captured 2026-05-07)
//!   - codex-relay (current crate; see Cargo.toml `version`)
//!
//! These tests exist to lock in behavior added for codex 0.128's two new
//! wire-shape elements that the relay's translation handles:
//!
//!   1. `namespace`-typed tools (MCP plugin grouping). Codex 0.128 ships
//!      with `mcp__codex_apps__github` carrying ~90 child function tools.
//!      The relay must flatten these into individual function tools so
//!      non-OpenAI providers actually see them.
//!
//!   2. `type:reasoning` input items. Codex 0.128 may replay reasoning
//!      items in input history; the relay must drop them rather than
//!      let them fall through as empty user messages.
//!
//! Reasoning_content round-trip itself is covered by `compat_deepseek_v4_pro.rs`.
//! Live end-to-end coverage against real DeepSeek is in `compat_deepseek_live.rs`.

use codex_relay::session::SessionStore;
use codex_relay::translate::to_chat_request;
use codex_relay::types::ResponsesRequest;
use serde_json::Value;
use std::path::PathBuf;

fn fixture(name: &str) -> ResponsesRequest {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/codex_0_128_0");
    p.push(name);
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

#[test]
fn namespace_tools_are_flattened() {
    let req = fixture("with_namespace_tool.json");
    let chat = to_chat_request(&req, Vec::new(), &SessionStore::new());

    // Source tools: 1 function + 1 web_search + 1 image_generation + 1 namespace(2 sub-tools)
    // Expected upstream: 1 + 0 + 0 + 2 = 3 function tools (built-ins dropped, namespace flattened)
    assert_eq!(chat.tools.len(), 3, "tools: {:?}", chat.tools);

    let names: Vec<String> = chat
        .tools
        .iter()
        .map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        })
        .collect();
    assert_eq!(
        names,
        vec![
            "exec_command",
            "mcp__codex_apps__github-_add_comment_to_issue",
            "mcp__codex_apps__github-_close_issue"
        ]
    );

    // All emitted tools must be in nested Chat Completions shape.
    for t in &chat.tools {
        assert_eq!(t.get("type").and_then(Value::as_str), Some("function"));
        let f = t.get("function").expect("function field");
        assert!(f.get("name").is_some());
        assert!(f.get("parameters").is_some());
    }
}

#[test]
fn reasoning_input_items_are_dropped() {
    let req = fixture("with_reasoning_item.json");
    let chat = to_chat_request(&req, Vec::new(), &SessionStore::new());

    // Input: user, reasoning, user → expect: system (instructions) + user + user.
    // Reasoning item must NOT become an empty/garbled user message.
    let roles: Vec<&str> = chat.messages.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["system", "user", "user"]);

    let texts: Vec<&str> = chat.messages.iter().map(|m| m.text_content()).collect();
    assert_eq!(texts, ["system", "first turn", "second turn"]);
}

/// Codex 0.128 sends image attachments as `input_image` content parts with a
/// flat `image_url` string. The relay must translate these into the Chat
/// Completions multimodal shape (content as an array containing `image_url`
/// parts wrapped in `{url}`) so vision-capable upstreams (e.g. DeepSeek V4
/// Pro) can receive them. Locks in the wire-shape contract verified live in
/// `compat_deepseek_live.rs::live_deepseek_v4_pro_image_input_wire_shape`.
#[test]
fn input_image_becomes_chat_completions_multimodal() {
    let req = fixture("with_image_input.json");
    let chat = to_chat_request(&req, Vec::new(), &SessionStore::new());

    // messages[0] = system (from `instructions`), messages[1] = user with image.
    assert_eq!(chat.messages.len(), 2);
    assert_eq!(chat.messages[0].role, "system");
    assert_eq!(chat.messages[1].role, "user");

    let parts = chat.messages[1]
        .content
        .as_ref()
        .and_then(Value::as_array)
        .expect("user content must be a multimodal array");
    assert_eq!(parts.len(), 2, "parts: {parts:?}");

    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "What is in this image?");

    assert_eq!(parts[1]["type"], "image_url");
    let url = parts[1]["image_url"]["url"]
        .as_str()
        .expect("image_url.url must be a string (Chat Completions shape)");
    assert!(
        url.starts_with("data:image/png;base64,"),
        "url not a data URL: {url}"
    );
}

#[test]
fn unknown_top_level_fields_dont_break_parse() {
    // Codex 0.128.0 sends client_metadata, prompt_cache_key, tool_choice,
    // parallel_tool_calls, store, include, reasoning. None of these have
    // chat-completions equivalents; relay must silently ignore and not 422.
    // (Negative coverage — fixture::with_namespace_tool has a few of them.)
    let req = fixture("with_namespace_tool.json");
    assert_eq!(req.model, "deepseek-v4-pro");
}
