//! Focused repro tests for recent GitHub issues.
//!
//! These tests use only local translation code or a local mock upstream; they
//! do not require a real LLM, Codex Desktop, or an MCP server.

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::Response,
    routing::{get, post},
    Router,
};
use codex_relay::session::SessionStore;
use codex_relay::translate::{
    from_chat_response_with_tool_map, namespace_tool_map, to_chat_request,
};
use codex_relay::types::{ChatChoice, ChatMessage, ChatResponse, ResponsesRequest};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RELAY_BIN: &str = env!("CARGO_BIN_EXE_codex-relay");

fn fixture(name: &str) -> ResponsesRequest {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures/codex_0_128_0");
    p.push(name);
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn pick_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
}

#[test]
fn issue_6_namespace_tools_keep_namespace_when_flattened() {
    let req = fixture("with_namespace_tool.json");
    let chat = to_chat_request(&req, Vec::new(), &SessionStore::new());

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

    assert!(
        names
            .iter()
            .any(|n| n == "mcp__codex_apps__github-_add_comment_to_issue"),
        "namespace child tool should be flattened with its namespace prefix: {names:?}"
    );
}

#[test]
fn issue_17_blocking_namespaced_tool_calls_emit_namespace_field() {
    let chat = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![json!({
                    "id": "call_js",
                    "type": "function",
                    "function": {
                        "name": "mcp__node_repl-js",
                        "arguments": "{}"
                    }
                })]),
                tool_call_id: None,
                name: None,
            },
        }],
        usage: None,
    };
    let tools = vec![json!({
        "type": "namespace",
        "name": "mcp__node_repl",
        "tools": [{"type": "function", "name": "js"}]
    })];
    let namespace_tools = namespace_tool_map(&tools);

    let (resp, _) =
        from_chat_response_with_tool_map("resp_17".into(), "mock-model", chat, &namespace_tools);
    assert_eq!(resp.output[0]["type"], "function_call");
    assert_eq!(resp.output[0]["namespace"], "mcp__node_repl");
    assert_eq!(resp.output[0]["name"], "js");
}

#[test]
fn issue_20_blocking_hyphen_flat_tool_name_is_not_namespaced() {
    let chat = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![json!({
                    "id": "call_flat",
                    "type": "function",
                    "function": {
                        "name": "foo-bar",
                        "arguments": "{}"
                    }
                })]),
                tool_call_id: None,
                name: None,
            },
        }],
        usage: None,
    };
    let tools = vec![json!({"type": "function", "name": "foo-bar"})];
    let namespace_tools = namespace_tool_map(&tools);

    let (resp, _) =
        from_chat_response_with_tool_map("resp_20".into(), "mock-model", chat, &namespace_tools);
    assert_eq!(resp.output[0]["type"], "function_call");
    assert!(resp.output[0].get("namespace").is_none());
    assert_eq!(resp.output[0]["name"], "foo-bar");
}

#[derive(Clone)]
struct MockState {
    bodies: Arc<Mutex<Vec<Value>>>,
    responses: Arc<Mutex<VecDeque<String>>>,
}

async fn models_handler() -> axum::Json<Value> {
    axum::Json(json!({"data": [{"id": "mock-model"}]}))
}

async fn chat_handler(State(state): State<MockState>, req: axum::extract::Request) -> Response {
    let bytes = match axum::body::to_bytes(req.into_body(), 1_000_000).await {
        Ok(bytes) => bytes,
        Err(_) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::from("bad body"))
                .unwrap();
        }
    };
    let body: Value = serde_json::from_slice(&bytes).expect("chat request json");
    state.bodies.lock().unwrap().push(body);

    let sse = state
        .responses
        .lock()
        .unwrap()
        .pop_front()
        .unwrap_or_else(default_ok_sse);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .body(Body::from(sse))
        .unwrap()
}

fn sse_from_chunks(chunks: Vec<Value>) -> String {
    let mut sse = String::new();
    for chunk in chunks {
        sse.push_str("data: ");
        sse.push_str(&chunk.to_string());
        sse.push_str("\n\n");
    }
    sse.push_str("data: [DONE]\n\n");
    sse
}

fn default_ok_sse() -> String {
    sse_from_chunks(vec![
        json!({"choices":[{"delta":{"role":"assistant","content":"OK"}}]}),
        json!({"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}),
    ])
}

async fn spawn_mock_upstream() -> (u16, Arc<Mutex<Vec<Value>>>) {
    spawn_mock_upstream_with_responses(Vec::new()).await
}

async fn spawn_mock_upstream_with_responses(
    responses: Vec<String>,
) -> (u16, Arc<Mutex<Vec<Value>>>) {
    let port = pick_port();
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let state = MockState {
        bodies: bodies.clone(),
        responses: Arc::new(Mutex::new(VecDeque::from(responses))),
    };
    let app = Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_handler))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind mock upstream");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("mock upstream serve");
    });
    (port, bodies)
}

async fn post_stream_completed(relay: &Relay, body: Value) -> Value {
    let events = post_stream_events(relay, body).await;
    events
        .into_iter()
        .find_map(|(event, data)| (event == "response.completed").then_some(data))
        .expect("response.completed event")
}

async fn post_stream_events(relay: &Relay, body: Value) -> Vec<(String, Value)> {
    let resp = reqwest::Client::new()
        .post(relay.url("/v1/responses"))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses");
    assert!(resp.status().is_success(), "status {}", resp.status());

    let mut events = resp.bytes_stream().eventsource();
    let mut out = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(8);
    while let Some(ev) = tokio::time::timeout(deadline - Instant::now(), events.next())
        .await
        .expect("stream timeout")
    {
        let ev = ev.expect("sse parse");
        let event = ev.event;
        let data: Value = serde_json::from_str(&ev.data).expect("event json");
        let terminal = event == "response.completed" || event == "response.failed";
        out.push((event, data));
        if terminal {
            return out;
        }
    }

    panic!("terminal response event");
}

struct Relay {
    child: Child,
    port: u16,
}

impl Drop for Relay {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Relay {
    fn spawn(upstream: &str) -> Self {
        let port = pick_port();
        let child = Command::new(RELAY_BIN)
            .env("CODEX_RELAY_PORT", port.to_string())
            .env("CODEX_RELAY_UPSTREAM", upstream)
            .env("CODEX_RELAY_API_KEY", "")
            .env("RUST_LOG", "codex_relay=warn")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn codex-relay");

        let mut handle = Relay { child, port };
        handle.wait_ready();
        handle
    }

    fn wait_ready(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(8);
        while Instant::now() < deadline {
            if std::net::TcpStream::connect(("127.0.0.1", self.port)).is_ok() {
                return;
            }
            std::thread::sleep(Duration::from_millis(80));
        }
        panic!("relay did not become ready on :{}", self.port);
    }

    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{}", self.port, path)
    }
}

#[tokio::test]
async fn issue_5_streaming_completed_event_includes_usage() {
    let (upstream_port, bodies) = spawn_mock_upstream().await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let body = json!({
        "model": "mock-model",
        "instructions": "Answer briefly.",
        "input": "Say OK.",
        "tools": [],
        "stream": true
    });

    let resp = reqwest::Client::new()
        .post(relay.url("/v1/responses"))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses");
    assert!(resp.status().is_success(), "status {}", resp.status());

    let mut events = resp.bytes_stream().eventsource();
    let mut completed: Option<Value> = None;
    let deadline = Instant::now() + Duration::from_secs(8);
    while let Some(ev) = tokio::time::timeout(deadline - Instant::now(), events.next())
        .await
        .expect("stream timeout")
    {
        let ev = ev.expect("sse parse");
        if ev.event == "response.completed" {
            completed = Some(serde_json::from_str(&ev.data).expect("completed json"));
            break;
        }
    }

    let completed = completed.expect("response.completed event");
    assert_eq!(
        completed["response"]["usage"],
        json!({"input_tokens": 7, "output_tokens": 2, "total_tokens": 9})
    );

    let request_bodies = bodies.lock().unwrap();
    let upstream_body = request_bodies.first().expect("upstream chat request");
    assert_eq!(
        upstream_body["stream_options"],
        json!({"include_usage": true}),
        "streaming Chat Completions requests must ask upstream to include usage"
    );
}

#[tokio::test]
async fn issue_17_streaming_namespaced_tool_calls_emit_namespace_field() {
    let tool_sse = sse_from_chunks(vec![
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_js",
                        "function": {
                            "name": "mcp__node_repl-js",
                            "arguments": "{}"
                        }
                    }]
                }
            }]
        }),
        json!({"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}),
    ]);
    let (upstream_port, bodies) = spawn_mock_upstream_with_responses(vec![tool_sse]).await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let events = post_stream_events(
        &relay,
        json!({
            "model": "mock-model",
            "input": "Use the JS REPL.",
            "tools": [{
                "type": "namespace",
                "name": "mcp__node_repl",
                "tools": [{
                    "type": "function",
                    "name": "js",
                    "parameters": {"type": "object"}
                }]
            }],
            "stream": true
        }),
    )
    .await;

    let added = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.added" && data["item"]["type"] == "function_call"
        })
        .map(|(_, data)| &data["item"])
        .expect("function_call added item");
    assert_eq!(added["namespace"], "mcp__node_repl");
    assert_eq!(added["name"], "js");

    let done = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.done" && data["item"]["type"] == "function_call"
        })
        .map(|(_, data)| &data["item"])
        .expect("function_call done item");
    assert_eq!(done["namespace"], "mcp__node_repl");
    assert_eq!(done["name"], "js");

    let completed = events
        .iter()
        .find_map(|(event, data)| (event == "response.completed").then_some(data))
        .expect("response.completed");
    let item = &completed["response"]["output"][0];
    assert_eq!(item["type"], "function_call");
    assert_eq!(item["namespace"], "mcp__node_repl");
    assert_eq!(item["name"], "js");

    let request_bodies = bodies.lock().unwrap();
    assert_eq!(
        request_bodies[0]["tools"][0]["function"]["name"], "mcp__node_repl-js",
        "namespace tools must be flattened with a reversible separator"
    );
}

#[tokio::test]
async fn issue_20_streaming_hyphen_flat_tool_name_is_not_namespaced() {
    let tool_sse = sse_from_chunks(vec![
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_flat",
                        "function": {
                            "name": "foo-bar",
                            "arguments": "{}"
                        }
                    }]
                }
            }]
        }),
        json!({"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}),
    ]);
    let (upstream_port, _bodies) = spawn_mock_upstream_with_responses(vec![tool_sse]).await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let events = post_stream_events(
        &relay,
        json!({
            "model": "mock-model",
            "input": "Use the flat tool.",
            "tools": [{
                "type": "function",
                "name": "foo-bar",
                "parameters": {"type": "object"}
            }],
            "stream": true
        }),
    )
    .await;

    let added = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.added" && data["item"]["type"] == "function_call"
        })
        .map(|(_, data)| &data["item"])
        .expect("function_call added item");
    assert!(added.get("namespace").is_none());
    assert_eq!(added["name"], "foo-bar");

    let done = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.done" && data["item"]["type"] == "function_call"
        })
        .map(|(_, data)| &data["item"])
        .expect("function_call done item");
    assert!(done.get("namespace").is_none());
    assert_eq!(done["name"], "foo-bar");

    let completed = events
        .iter()
        .find_map(|(event, data)| (event == "response.completed").then_some(data))
        .expect("response.completed");
    let item = &completed["response"]["output"][0];
    assert!(item.get("namespace").is_none());
    assert_eq!(item["name"], "foo-bar");
}

#[tokio::test]
async fn issue_12_spawn_agent_child_context_should_not_replay_parent_history() {
    let child_task = "Please compute 2+2 and return only the numeric result.";
    let parent_prompt = "Ask a subagent to solve 2+2.";
    let tool_args = json!({
        "task_name": "simple_math",
        "message": child_task,
    })
    .to_string();
    let spawn_agent_sse = sse_from_chunks(vec![
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_spawn_simple_math",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": tool_args
                        }
                    }]
                }
            }]
        }),
        json!({"choices":[],"usage":{"prompt_tokens":11,"completion_tokens":3,"total_tokens":14}}),
    ]);

    let (upstream_port, bodies) =
        spawn_mock_upstream_with_responses(vec![spawn_agent_sse, default_ok_sse()]).await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let parent_completed = post_stream_completed(
        &relay,
        json!({
            "model": "mock-model",
            "instructions": "You are the parent agent.",
            "input": parent_prompt,
            "tools": [{"type": "function", "name": "spawn_agent"}],
            "stream": true
        }),
    )
    .await;

    assert_eq!(
        parent_completed["response"]["output"][0]["name"], "spawn_agent",
        "mock upstream should first drive a spawn_agent call"
    );
    let parent_response_id = parent_completed["response"]["id"]
        .as_str()
        .expect("parent response id");

    // Simulate the child agent request that triggers #12: it asks the relay
    // for the spawned task while also reusing the parent's previous_response_id.
    // A correctly isolated child thread should send only the child task context
    // upstream, not the parent's prompt or assistant spawn_agent tool call.
    let _child_completed = post_stream_completed(
        &relay,
        json!({
            "model": "mock-model",
            "instructions": "You are the spawned child agent.",
            "previous_response_id": parent_response_id,
            "input": child_task,
            "tools": [
                {"type": "function", "name": "spawn_agent"},
                {"type": "function", "name": "wait_agent"}
            ],
            "stream": true
        }),
    )
    .await;

    let request_bodies = bodies.lock().unwrap();
    assert_eq!(request_bodies.len(), 2, "parent and child upstream calls");
    let child_messages = request_bodies[1]["messages"]
        .as_array()
        .expect("child upstream messages");

    assert!(
        !child_messages
            .iter()
            .any(|msg| msg["content"] == parent_prompt),
        "child upstream request leaked the parent prompt: {child_messages:#?}"
    );
    assert!(
        !child_messages.iter().any(|msg| {
            msg["tool_calls"].as_array().is_some_and(|calls| {
                calls
                    .iter()
                    .any(|call| call["function"]["name"] == "spawn_agent")
            })
        }),
        "child upstream request replayed the parent's spawn_agent tool call: {child_messages:#?}"
    );
    assert_eq!(
        child_messages
            .iter()
            .filter(|msg| msg["role"] == "user")
            .map(|msg| msg["content"].as_str().unwrap_or(""))
            .collect::<Vec<_>>(),
        vec![child_task],
        "child upstream request should contain exactly the spawned message as user input"
    );
}
