//! Vendor compatibility tests for reasoning_content round-trip.
//!
//! These tests simulate the exact request patterns observed from Codex CLI
//! when talking to DeepSeek V4 Pro (and similar thinking models) through the
//! relay. The key behavior: Codex CLI sends `previous_response_id: None` and
//! embeds all conversation history as `input` items. The relay must recover
//! `reasoning_content` and attach it to the corresponding assistant messages.

use codex_relay::session::SessionStore;
use codex_relay::translate::to_chat_request;
use codex_relay::types::*;
use serde_json::json;

fn base_req(input: ResponsesInput) -> ResponsesRequest {
    ResponsesRequest {
        model: "deepseek-v4-pro".into(),
        input,
        previous_response_id: None,
        tools: vec![],
        stream: false,
        temperature: None,
        max_output_tokens: None,
        system: None,
        instructions: None,
    }
}

fn assistant_msg(content: &str) -> ChatMessage {
    ChatMessage {
        role: "assistant".into(),
        content: Some(content.into()),
        reasoning_content: None,
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

fn assistant_msg_with_tool_calls(content: &str, tool_calls: Vec<serde_json::Value>) -> ChatMessage {
    ChatMessage {
        role: "assistant".into(),
        content: Some(content.into()),
        reasoning_content: None,
        tool_calls: Some(tool_calls),
        tool_call_id: None,
        name: None,
    }
}

/// DeepSeek V4 Pro: 2-turn text-only conversation where turn 1 produces
/// reasoning_content that must be recovered in turn 2.
#[test]
fn test_deepseek_v4_pro_reasoning_roundtrip_text_only() {
    let store = SessionStore::new();

    // Simulate turn 1: model returned text + reasoning
    let assistant = assistant_msg("Let me analyze this");
    store.store_turn_reasoning(&[], &assistant, "<think>analyzing the problem...</think>".into());

    // Turn 2: Codex replays full conversation history as input items
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Research task prompt"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me analyze this"}]}),
        json!({"type": "message", "role": "user", "content": "Continue"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user "Research task prompt"
    // messages[1] = assistant "Let me analyze this"  ← should have reasoning
    // messages[2] = user "Continue"
    assert_eq!(chat.messages.len(), 3);
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(chat.messages[1].content.as_deref(), Some("Let me analyze this"));
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>analyzing the problem...</think>"),
        "assistant text message should have reasoning_content recovered"
    );
}

/// DeepSeek V4 Pro: turn 1 returns text + reasoning + tool_calls. Codex
/// replays them as SEPARATE items (assistant message + function_call items +
/// function_call_output items). Both the text message and the grouped
/// tool-call message should get reasoning_content.
#[test]
fn test_deepseek_v4_pro_reasoning_roundtrip_with_tool_calls() {
    let store = SessionStore::new();

    // Simulate turn 1: model returned text + tool_calls + reasoning
    let assistant = assistant_msg_with_tool_calls(
        "Let me check",
        vec![json!({
            "id": "call_abc",
            "type": "function",
            "function": {"name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"}
        })],
    );
    store.store_turn_reasoning(
        &[],
        &assistant,
        "<think>need to read files</think>".into(),
    );

    // Turn 2: Codex replays conversation with separate items
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Prompt"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check"}]}),
        json!({"type": "function_call", "call_id": "call_abc", "name": "exec_command", "arguments": "{\"cmd\": \"ls\"}"}),
        json!({"type": "function_call_output", "call_id": "call_abc", "output": "file1.py\nfile2.py"}),
        json!({"type": "message", "role": "user", "content": "What next?"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user "Prompt"
    // messages[1] = assistant "Let me check"        ← reasoning via content key
    // messages[2] = assistant tool_calls [call_abc]  ← reasoning via call_id fallback
    // messages[3] = tool output
    // messages[4] = user "What next?"
    assert_eq!(chat.messages.len(), 5);

    // Assistant TEXT message should have reasoning_content
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(chat.messages[1].content.as_deref(), Some("Let me check"));
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>need to read files</think>"),
        "assistant text message should have reasoning_content"
    );

    // Assistant TOOL CALL message should also have reasoning_content (via call_id)
    assert_eq!(chat.messages[2].role, "assistant");
    assert!(chat.messages[2].tool_calls.is_some());
    assert_eq!(
        chat.messages[2].reasoning_content.as_deref(),
        Some("<think>need to read files</think>"),
        "assistant tool-call message should have reasoning_content via call_id fallback"
    );
}

/// DeepSeek V4 Pro: 3-turn conversation where each turn has its own reasoning
/// that must be independently recovered.
#[test]
fn test_deepseek_v4_pro_multi_turn_reasoning() {
    let store = SessionStore::new();

    // Store reasoning for turn 1
    let assistant1 = assistant_msg("Step 1 analysis");
    store.store_turn_reasoning(&[], &assistant1, "<think>first pass thinking</think>".into());

    // Store reasoning for turn 2
    let assistant2 = assistant_msg("Step 2 deeper look");
    store.store_turn_reasoning(&[], &assistant2, "<think>second pass thinking</think>".into());

    // Turn 3: Codex replays the full 2-turn history
    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Start research"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Step 1 analysis"}]}),
        json!({"type": "message", "role": "user", "content": "Go deeper"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Step 2 deeper look"}]}),
        json!({"type": "message", "role": "user", "content": "Finalize"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    assert_eq!(chat.messages.len(), 5);

    // Turn 1 assistant
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(chat.messages[1].content.as_deref(), Some("Step 1 analysis"));
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>first pass thinking</think>"),
        "turn 1 assistant should have its own reasoning_content"
    );

    // Turn 2 assistant
    assert_eq!(chat.messages[3].role, "assistant");
    assert_eq!(chat.messages[3].content.as_deref(), Some("Step 2 deeper look"));
    assert_eq!(
        chat.messages[3].reasoning_content.as_deref(),
        Some("<think>second pass thinking</think>"),
        "turn 2 assistant should have its own reasoning_content"
    );
}

/// Non-thinking model (e.g. deepseek-chat): when no reasoning was stored,
/// assistant messages should have reasoning_content=None.
#[test]
fn test_non_thinking_model_no_reasoning_content() {
    let store = SessionStore::new();

    // Don't store any reasoning — simulating a model that doesn't think

    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Hello"}),
        json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hi there!"}]}),
        json!({"type": "message", "role": "user", "content": "Thanks"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    assert_eq!(chat.messages.len(), 3);
    assert_eq!(chat.messages[1].role, "assistant");
    assert_eq!(chat.messages[1].content.as_deref(), Some("Hi there!"));
    assert!(
        chat.messages[1].reasoning_content.is_none(),
        "non-thinking model should have reasoning_content=None"
    );
}

/// Kimi K2.6: verify call_id-based reasoning recovery still works when Codex
/// DOES use previous_response_id and the function_call path is the main
/// recovery mechanism.
#[test]
fn test_kimi_k2_6_reasoning_via_call_id() {
    let store = SessionStore::new();

    // Store reasoning keyed by call_id (the existing mechanism)
    store.store_reasoning("call_xyz".into(), "<think>kimi is thinking</think>".into());

    let req = base_req(ResponsesInput::Messages(vec![
        json!({"type": "message", "role": "user", "content": "Do something"}),
        json!({"type": "function_call", "call_id": "call_xyz", "name": "run_cmd", "arguments": "{\"cmd\": \"pwd\"}"}),
        json!({"type": "function_call_output", "call_id": "call_xyz", "output": "/home/user"}),
        json!({"type": "message", "role": "user", "content": "Continue"}),
    ]));

    let chat = to_chat_request(&req, vec![], &store);

    // messages[0] = user
    // messages[1] = assistant (grouped function_call)  ← reasoning via call_id
    // messages[2] = tool output
    // messages[3] = user
    assert_eq!(chat.messages.len(), 4);
    assert_eq!(chat.messages[1].role, "assistant");
    assert!(chat.messages[1].tool_calls.is_some());
    assert_eq!(
        chat.messages[1].reasoning_content.as_deref(),
        Some("<think>kimi is thinking</think>"),
        "grouped assistant tool-call message should have reasoning_content via call_id"
    );
}
