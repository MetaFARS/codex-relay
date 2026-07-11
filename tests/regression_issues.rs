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
    custom_tool_map, from_chat_response_with_tool_map, from_chat_response_with_tool_maps,
    namespace_tool_map, to_chat_request,
};
use codex_relay::types::{ChatChoice, ChatMessage, ChatResponse, ChatUsage, ResponsesRequest};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::VecDeque;
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

#[test]
fn issue_37_custom_apply_patch_round_trips_in_blocking_translation() {
    let tools = vec![json!({
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch"
    })];
    let req: ResponsesRequest = serde_json::from_value(json!({
        "model": "mock-model",
        "input": "Update the file.",
        "tools": tools,
        "stream": false
    }))
    .unwrap();
    let chat_req = to_chat_request(&req, Vec::new(), &SessionStore::new());
    assert_eq!(chat_req.tools[0]["type"], "function");
    assert_eq!(chat_req.tools[0]["function"]["name"], "apply_patch");
    assert_eq!(
        chat_req.tools[0]["function"]["parameters"]["required"],
        json!(["patch"])
    );

    let patch = "*** Begin Patch\n*** Update File: test.txt\n@@\n-old\n+new\n*** End Patch";
    let chat = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![json!({
                    "id": "call_patch",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": json!({"patch": patch}).to_string()
                    }
                })]),
                tool_call_id: None,
                name: None,
            },
        }],
        usage: None,
    };
    let custom_tools = custom_tool_map(&req.tools);
    let (response, _) = from_chat_response_with_tool_maps(
        "resp_37".into(),
        "mock-model",
        chat,
        &Default::default(),
        &custom_tools,
    );
    let item = &response.output[0];
    assert_eq!(item["type"], "custom_tool_call");
    assert_eq!(item["name"], "apply_patch");
    assert_eq!(item["call_id"], "call_patch");
    assert_eq!(item["input"], patch);
    assert!(item.get("arguments").is_none());
}

#[test]
fn blocking_response_usage_includes_cached_tokens() {
    let chat = ChatResponse {
        choices: vec![ChatChoice {
            message: ChatMessage {
                role: "assistant".into(),
                content: Some("OK".into()),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: None,
                name: None,
            },
        }],
        usage: Some(ChatUsage {
            prompt_tokens: 17,
            completion_tokens: 2,
            total_tokens: 19,
            prompt_cache_hit_tokens: Some(11),
            prompt_cache_miss_tokens: Some(6),
            prompt_tokens_details: None,
        }),
    };

    let (resp, _) = from_chat_response_with_tool_map(
        "resp_cached".into(),
        "mock-model",
        chat,
        &Default::default(),
    );

    assert_eq!(
        serde_json::to_value(resp.usage).expect("usage json"),
        json!({
            "input_tokens": 17,
            "output_tokens": 2,
            "total_tokens": 19,
            "input_tokens_details": {"cached_tokens": 11}
        })
    );
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

async fn delayed_chat_handler(
    State(state): State<MockState>,
    req: axum::extract::Request,
) -> Response {
    tokio::time::sleep(Duration::from_secs(7)).await;
    chat_handler(State(state), req).await
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

fn sse_from_chunks_without_done(chunks: Vec<Value>) -> String {
    let mut sse = String::new();
    for chunk in chunks {
        sse.push_str("data: ");
        sse.push_str(&chunk.to_string());
        sse.push_str("\n\n");
    }
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
    spawn_mock_upstream_with_chat_handler(responses, chat_handler).await
}

async fn spawn_delayed_mock_upstream() -> (u16, Arc<Mutex<Vec<Value>>>) {
    spawn_mock_upstream_with_chat_handler(Vec::new(), delayed_chat_handler).await
}

async fn spawn_mock_upstream_with_chat_handler<H, T>(
    responses: Vec<String>,
    handler: H,
) -> (u16, Arc<Mutex<Vec<Value>>>)
where
    H: axum::handler::Handler<T, MockState>,
    T: 'static,
{
    let bodies = Arc::new(Mutex::new(Vec::new()));
    let state = MockState {
        bodies: bodies.clone(),
        responses: Arc::new(Mutex::new(VecDeque::from(responses))),
    };
    let app = Router::new()
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(handler))
        .with_state(state);
    // Bind to port 0 and keep the listener so the OS-assigned port cannot be
    // grabbed by a concurrently running test (avoids a bind/drop/rebind race).
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .expect("bind mock upstream");
    let port = listener.local_addr().expect("mock upstream addr").port();
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
        Self::spawn_with_env(upstream, &[])
    }

    fn spawn_with_env(upstream: &str, extra_env: &[(&str, &str)]) -> Self {
        let mut command = Command::new(RELAY_BIN);
        command
            // Bind an ephemeral port; the real port is read from the child's
            // startup log. This avoids a bind/drop/rebind race where two
            // concurrent tests could pick the same port.
            .env("CODEX_RELAY_PORT", "0")
            .env("CODEX_RELAY_UPSTREAM", upstream)
            .env("CODEX_RELAY_API_KEY", "")
            .env("RUST_LOG", "codex_relay=info")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (key, value) in extra_env {
            command.env(key, value);
        }
        let mut child = command.spawn().expect("spawn codex-relay");

        let port = Self::read_listening_port(&mut child);
        let mut handle = Relay { child, port };
        handle.wait_ready();
        handle
    }

    /// Read the bound port from the relay's `listening on 127.0.0.1:PORT` log line.
    ///
    /// A background thread keeps draining stdout for the child's lifetime so the
    /// pipe never fills (which would block the relay) and stays open (closing it
    /// would kill the relay with SIGPIPE on its next log write).
    fn read_listening_port(child: &mut Child) -> u16 {
        use std::io::{BufRead, BufReader};
        use std::sync::mpsc;
        let stdout = child.stdout.take().expect("relay stdout");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            let mut tx = Some(tx);
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {}
                }
                if let Some(sender) = tx.as_ref() {
                    if let Some(rest) = line.split("listening on 127.0.0.1:").nth(1) {
                        if let Some(port) = rest
                            .split(|c: char| !c.is_ascii_digit())
                            .next()
                            .and_then(|s| s.parse::<u16>().ok())
                        {
                            let _ = sender.send(port);
                            tx = None;
                        }
                    }
                }
            }
        });
        rx.recv_timeout(Duration::from_secs(8))
            .expect("relay did not report a listening port")
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
async fn issue_40_streaming_headers_and_keepalive_do_not_wait_for_upstream_headers() {
    let (upstream_port, _bodies) = spawn_delayed_mock_upstream().await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let response = tokio::time::timeout(
        Duration::from_secs(1),
        reqwest::Client::new()
            .post(relay.url("/v1/responses"))
            .json(&json!({
                "model": "mock-model",
                "input": "Wait for the upstream.",
                "tools": [],
                "stream": true
            }))
            .send(),
    )
    .await
    .expect("relay should flush streaming response headers immediately")
    .expect("POST /v1/responses");
    assert!(response.status().is_success());
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );

    let mut chunks = response.bytes_stream();
    let first = tokio::time::timeout(Duration::from_secs(1), chunks.next())
        .await
        .expect("response.created should arrive immediately")
        .expect("response.created chunk")
        .expect("response.created bytes");
    assert!(String::from_utf8_lossy(&first).contains("response.created"));

    let keepalive = tokio::time::timeout(Duration::from_secs(6), chunks.next())
        .await
        .expect("keepalive should arrive before delayed upstream headers")
        .expect("keepalive chunk")
        .expect("keepalive bytes");
    assert_eq!(keepalive.as_ref(), b": keepalive\n\n");
}

#[tokio::test]
async fn issue_29_extra_and_drop_params_modify_streaming_upstream_request() {
    let (upstream_port, bodies) = spawn_mock_upstream().await;
    let relay = Relay::spawn_with_env(
        &format!("http://127.0.0.1:{upstream_port}/v1"),
        &[
            (
                "CODEX_RELAY_UPSTREAM_EXTRA_PARAMS",
                r#"{"thinking":{"type":"disabled"}}"#,
            ),
            ("CODEX_RELAY_DROP_PARAMS", r#"["stream_options"]"#),
        ],
    );

    let _ = post_stream_completed(
        &relay,
        json!({"model": "glm-5.2", "input": "hi", "tools": [], "stream": true}),
    )
    .await;

    let body = bodies
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("upstream body");
    assert_eq!(body["thinking"], json!({"type": "disabled"}));
    assert!(body.get("stream_options").is_none(), "body: {body}");
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
        json!({"input_tokens": 7, "output_tokens": 2, "total_tokens": 9, "input_tokens_details": {"cached_tokens": 0}})
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
async fn streaming_response_usage_includes_cached_tokens() {
    let cached_usage_sse = sse_from_chunks(vec![
        json!({"choices":[{"delta":{"role":"assistant","content":"OK"}}]}),
        json!({
            "choices": [],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 2,
                "total_tokens": 19,
                "prompt_cache_hit_tokens": 11,
                "prompt_cache_miss_tokens": 6
            }
        }),
    ]);
    let (upstream_port, _bodies) = spawn_mock_upstream_with_responses(vec![cached_usage_sse]).await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let completed = post_stream_completed(
        &relay,
        json!({
            "model": "mock-model",
            "input": "Say OK.",
            "tools": [],
            "stream": true
        }),
    )
    .await;

    assert_eq!(
        completed["response"]["usage"],
        json!({
            "input_tokens": 17,
            "output_tokens": 2,
            "total_tokens": 19,
            "input_tokens_details": {"cached_tokens": 11}
        })
    );
}

#[tokio::test]
async fn issue_26_glm_model_enables_thinking_on_upstream_request() {
    // GLM suppresses default auto-thinking under heavy agent prompts, so the
    // relay must send `thinking:{type:"enabled"}` for GLM-like models — otherwise
    // no reasoning_content is ever produced and there is nothing to translate.
    let (upstream_port, bodies) = spawn_mock_upstream().await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let _ = post_stream_completed(
        &relay,
        json!({"model": "glm-5.2", "input": "hi", "tools": [], "stream": true}),
    )
    .await;

    let body = bodies
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("upstream body");
    assert_eq!(
        body["thinking"],
        json!({"type": "enabled"}),
        "GLM request must enable thinking"
    );
}

#[tokio::test]
async fn issue_26_non_glm_model_does_not_send_thinking() {
    // DeepSeek/Kimi/etc. think by default and may reject unknown fields, so the
    // request shape for non-GLM models must be unchanged.
    let (upstream_port, bodies) = spawn_mock_upstream().await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let _ = post_stream_completed(
        &relay,
        json!({"model": "deepseek-reasoner", "input": "hi", "tools": [], "stream": true}),
    )
    .await;

    let body = bodies
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("upstream body");
    assert!(
        body.get("thinking").is_none(),
        "non-GLM request must not include thinking: {body}"
    );
}

#[tokio::test]
async fn issue_31_stream_without_done_still_completes_when_content_received() {
    // Some OpenAI-compatible providers (e.g. synthetic.new) close the SSE stream
    // cleanly without ever sending a terminating `[DONE]` line. A turn that
    // received content should still complete rather than be discarded.
    let no_done_sse = sse_from_chunks_without_done(vec![
        json!({"choices":[{"delta":{"role":"assistant","content":"Hello"}}]}),
        json!({"choices":[{"delta":{"content":" world"}}]}),
        json!({"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}),
    ]);
    let (upstream_port, _bodies) = spawn_mock_upstream_with_responses(vec![no_done_sse]).await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let events = post_stream_events(
        &relay,
        json!({"model": "mock-model", "input": "Say hi.", "tools": [], "stream": true}),
    )
    .await;

    let completed = events
        .iter()
        .find_map(|(event, data)| (event == "response.completed").then_some(data));
    let failed = events.iter().any(|(event, _)| event == "response.failed");
    assert!(!failed, "stream should not fail when content was received");
    let completed = completed.expect("response.completed");
    assert_eq!(
        completed["response"]["output"][0]["content"][0]["text"],
        "Hello world"
    );
}

#[tokio::test]
async fn issue_26_streaming_reasoning_alias_field_emits_reasoning_events() {
    // Some providers (OpenRouter/Together-style, newer GLM-5 deployments) stream
    // thinking under `delta.reasoning` rather than `delta.reasoning_content`.
    let reasoning_sse = sse_from_chunks(vec![
        json!({"choices":[{"delta":{"role":"assistant","reasoning":"alias "}}]}),
        json!({"choices":[{"delta":{"reasoning":"path"}}]}),
        json!({"choices":[{"delta":{"content":"OK"}}]}),
        json!({"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}),
    ]);
    let (upstream_port, _bodies) = spawn_mock_upstream_with_responses(vec![reasoning_sse]).await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let events = post_stream_events(
        &relay,
        json!({"model": "mock-model", "input": "Reason briefly.", "tools": [], "stream": true}),
    )
    .await;

    let deltas: Vec<&Value> = events
        .iter()
        .filter_map(|(event, data)| {
            (event == "response.reasoning_summary_text.delta").then_some(data)
        })
        .collect();
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0]["delta"], "alias ");
    assert_eq!(deltas[1]["delta"], "path");

    let completed = events
        .iter()
        .find_map(|(event, data)| (event == "response.completed").then_some(data))
        .expect("response.completed");
    assert_eq!(completed["response"]["output"][0]["type"], "reasoning");
    assert_eq!(
        completed["response"]["output"][0]["summary"],
        json!([{"type": "summary_text", "text": "alias path"}])
    );
}

#[tokio::test]
async fn issue_26_streaming_reasoning_content_emits_responses_reasoning_events() {
    let reasoning_sse = sse_from_chunks(vec![
        json!({"choices":[{"delta":{"role":"assistant","reasoning_content":"think "}}]}),
        json!({"choices":[{"delta":{"reasoning_content":"through it"}}]}),
        json!({"choices":[{"delta":{"content":"OK"}}]}),
        json!({"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}),
    ]);
    let (upstream_port, _bodies) = spawn_mock_upstream_with_responses(vec![reasoning_sse]).await;
    let relay = Relay::spawn(&format!("http://127.0.0.1:{upstream_port}/v1"));

    let events = post_stream_events(
        &relay,
        json!({
            "model": "mock-model",
            "input": "Reason briefly.",
            "tools": [],
            "stream": true
        }),
    )
    .await;

    let reasoning_added = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.added" && data["item"]["type"] == "reasoning"
        })
        .map(|(_, data)| data)
        .expect("reasoning output_item.added");
    assert_eq!(reasoning_added["output_index"], 0);
    let reasoning_item_id = reasoning_added["item"]["id"]
        .as_str()
        .expect("reasoning item id");

    let deltas: Vec<&Value> = events
        .iter()
        .filter_map(|(event, data)| {
            (event == "response.reasoning_summary_text.delta").then_some(data)
        })
        .collect();
    assert_eq!(deltas.len(), 2);
    assert_eq!(deltas[0]["delta"], "think ");
    assert_eq!(deltas[1]["delta"], "through it");
    for delta in deltas {
        assert_eq!(delta["item_id"], reasoning_item_id);
        assert_eq!(delta["output_index"], 0);
        assert_eq!(delta["summary_index"], 0);
    }

    let reasoning_done = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.done" && data["item"]["type"] == "reasoning"
        })
        .map(|(_, data)| data)
        .expect("reasoning output_item.done");
    assert_eq!(reasoning_done["output_index"], 0);
    assert_eq!(
        reasoning_done["item"]["summary"],
        json!([{"type": "summary_text", "text": "think through it"}])
    );

    let message_added = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.added" && data["item"]["type"] == "message"
        })
        .map(|(_, data)| data)
        .expect("message output_item.added");
    assert_eq!(message_added["output_index"], 1);

    let completed = events
        .iter()
        .find_map(|(event, data)| (event == "response.completed").then_some(data))
        .expect("response.completed");
    assert_eq!(completed["response"]["output"][0]["type"], "reasoning");
    assert_eq!(
        completed["response"]["output"][0]["summary"],
        json!([{"type": "summary_text", "text": "think through it"}])
    );
    assert_eq!(completed["response"]["output"][1]["type"], "message");
    assert_eq!(
        completed["response"]["output"][1]["content"],
        json!([{"type": "output_text", "text": "OK"}])
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
async fn issue_37_streaming_apply_patch_emits_custom_tool_events() {
    let patch = "*** Begin Patch\n*** Update File: test.txt\n@@\n-old\n+new\n*** End Patch";
    let arguments = json!({"patch": patch}).to_string();
    let tool_sse = sse_from_chunks(vec![
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_patch",
                        "function": {
                            "name": "apply_patch",
                            "arguments": arguments
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
            "input": "Apply the change.",
            "tools": [{
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch"
            }],
            "stream": true
        }),
    )
    .await;

    let added = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.added" && data["item"]["type"] == "custom_tool_call"
        })
        .map(|(_, data)| &data["item"])
        .expect("custom_tool_call added item");
    assert_eq!(added["name"], "apply_patch");
    assert_eq!(added["input"], "");
    assert!(added.get("arguments").is_none());

    let delta = events
        .iter()
        .find(|(event, _)| event == "response.custom_tool_call_input.delta")
        .map(|(_, data)| data)
        .expect("custom tool input delta");
    assert_eq!(delta["delta"], patch);

    let done = events
        .iter()
        .find(|(event, data)| {
            event == "response.output_item.done" && data["item"]["type"] == "custom_tool_call"
        })
        .map(|(_, data)| &data["item"])
        .expect("custom_tool_call done item");
    assert_eq!(done["input"], patch);

    let completed = events
        .iter()
        .find_map(|(event, data)| (event == "response.completed").then_some(data))
        .expect("response.completed");
    assert_eq!(
        completed["response"]["output"][0]["type"],
        "custom_tool_call"
    );
    assert_eq!(completed["response"]["output"][0]["input"], patch);

    let request_bodies = bodies.lock().unwrap();
    assert_eq!(
        request_bodies[0]["tools"][0]["function"]["name"],
        "apply_patch"
    );
    assert_eq!(
        request_bodies[0]["tools"][0]["function"]["parameters"]["required"],
        json!(["patch"])
    );
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

#[tokio::test]
async fn issue_24_v2_encrypted_spawn_child_context_is_isolated() {
    let child_task = "Inspect the repository and report the risky files.";
    let parent_prompt = "Ask a subagent to inspect the repository.";
    let tool_args = json!({
        "task_name": "repo_inspection",
        "fork_turns": "current_turn",
        "message": "encrypted:v2:opaque-child-task-ciphertext",
    })
    .to_string();
    let spawn_agent_sse = sse_from_chunks(vec![
        json!({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_spawn_repo_inspection",
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
            "tools": [
                {"type": "function", "name": "spawn_agent"},
                {"type": "function", "name": "wait_agent"}
            ],
            "stream": true
        }),
    )
    .await;
    let parent_response_id = parent_completed["response"]["id"]
        .as_str()
        .expect("parent response id");

    let _child_completed = post_stream_completed(
        &relay,
        json!({
            "model": "mock-model",
            "instructions": "You are the spawned child agent.",
            "previous_response_id": parent_response_id,
            "input": child_task,
            "tools": [
                {"type": "function", "name": "spawn_agent"},
                {"type": "function", "name": "wait_agent"},
                {"type": "function", "name": "list_agents"},
                {"type": "function", "name": "interrupt_agent"},
                {"type": "function", "name": "send_message"},
                {"type": "function", "name": "followup_task"}
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
        "V2 encrypted child request leaked the parent prompt: {child_messages:#?}"
    );
    assert!(
        !child_messages.iter().any(|msg| {
            msg["tool_calls"].as_array().is_some_and(|calls| {
                calls
                    .iter()
                    .any(|call| call["function"]["name"] == "spawn_agent")
            })
        }),
        "V2 encrypted child request replayed the parent's spawn_agent call: {child_messages:#?}"
    );
    assert_eq!(
        child_messages
            .iter()
            .filter(|msg| msg["role"] == "user")
            .map(|msg| msg["content"].as_str().unwrap_or(""))
            .collect::<Vec<_>>(),
        vec![child_task],
        "V2 encrypted child request should contain exactly the spawned message as user input"
    );
}
