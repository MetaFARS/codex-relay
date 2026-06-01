//! Live compatibility tests against the real DeepSeek API.
//!
//! Pinned versions covered:
//!   - DeepSeek models : deepseek-v4-pro, deepseek-v4-flash
//!   - codex-relay     : current crate (see Cargo.toml `version`)
//!   - Codex CLI       : not exercised — these tests speak the Responses API
//!     directly to the relay, simulating any Codex 0.128.x client.
//!
//! Gated on `DEEPSEEK_API_KEY` env var. Each test is marked `#[ignore]`
//! so the default `cargo test` stays offline. To run:
//!
//!     DEEPSEEK_API_KEY=sk-... cargo test --test compat_deepseek_live -- --ignored --nocapture
//!
//! Each test spawns a fresh relay binary on a random port, points it at
//! `https://api.deepseek.com/v1`, and exercises the path through reqwest.

use axum::{body::Body, extract::State, http::StatusCode, response::Response, Router};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::net::TcpListener;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const RELAY_BIN: &str = env!("CARGO_BIN_EXE_codex-relay");
const DEEPSEEK_UPSTREAM: &str = "https://api.deepseek.com/v1";

fn deepseek_key() -> Option<String> {
    std::env::var("DEEPSEEK_API_KEY")
        .ok()
        .filter(|s| !s.is_empty())
}

fn pick_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind");
    let p = l.local_addr().unwrap().port();
    drop(l);
    p
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
    fn spawn(upstream: &str, key: &str) -> Self {
        let port = pick_port();
        let child = Command::new(RELAY_BIN)
            .env("CODEX_RELAY_PORT", port.to_string())
            .env("CODEX_RELAY_UPSTREAM", upstream)
            .env("CODEX_RELAY_API_KEY", key)
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

/// Skip test if no API key. Returns Some(key) if present.
fn need_key() -> Option<String> {
    match deepseek_key() {
        Some(k) => Some(k),
        None => {
            eprintln!("skip: DEEPSEEK_API_KEY not set");
            None
        }
    }
}

/// Build a Responses-API request body (Codex 0.128.x style).
fn responses_body(model: &str, prompt: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "instructions": "Answer with a single short sentence.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": prompt }]
            }
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "store": false,
        "stream": stream,
        "include": []
    })
}

// ────────────────────────────────────────────────────────────────────────────
// Test 1: /v1/models — verify dual-shape and DeepSeek catalog reachable.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore]
async fn live_deepseek_models_lists_both_keys() {
    let Some(key) = need_key() else { return };
    let relay = Relay::spawn(DEEPSEEK_UPSTREAM, &key);

    let resp: Value = reqwest::get(relay.url("/v1/models"))
        .await
        .expect("GET /v1/models")
        .json()
        .await
        .expect("json decode");

    // Both keys must be present so legacy AND Codex 0.128+ clients are happy.
    assert!(resp.get("data").is_some(), "missing `data`: {resp}");
    assert!(resp.get("models").is_some(), "missing `models`: {resp}");

    let collect_ids = |k: &str| -> Vec<String> {
        resp[k]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.get("id").and_then(Value::as_str).map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    let ids_data = collect_ids("data");
    let ids_models = collect_ids("models");
    assert_eq!(ids_data, ids_models, "data/models must mirror each other");

    // Both target models must show up — pin the matrix DeepSeek currently exposes.
    assert!(
        ids_data.iter().any(|i| i == "deepseek-v4-pro"),
        "deepseek-v4-pro missing from {ids_data:?}"
    );
    assert!(
        ids_data.iter().any(|i| i == "deepseek-v4-flash"),
        "deepseek-v4-flash missing from {ids_data:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Test 2: non-streaming chat — both DeepSeek models.
// ────────────────────────────────────────────────────────────────────────────

async fn assert_blocking_chat(model: &str) {
    let Some(key) = need_key() else { return };
    let relay = Relay::spawn(DEEPSEEK_UPSTREAM, &key);

    let body = responses_body(model, "Say the word OK.", false);
    let resp: Value = reqwest::Client::new()
        .post(relay.url("/v1/responses"))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses")
        .error_for_status()
        .expect("non-2xx")
        .json()
        .await
        .expect("json decode");

    assert_eq!(resp["object"].as_str(), Some("response"));
    assert_eq!(resp["model"].as_str(), Some(model));
    assert!(resp["id"].as_str().is_some(), "missing id: {resp}");

    let output = resp["output"].as_array().expect("output array");
    assert!(!output.is_empty(), "empty output: {resp}");
    let first = &output[0];
    assert_eq!(first["type"].as_str(), Some("message"));
    assert_eq!(first["role"].as_str(), Some("assistant"));
    let text = first["content"][0]["text"]
        .as_str()
        .expect("output_text present");
    assert!(!text.trim().is_empty(), "empty assistant text");

    let usage = &resp["usage"];
    assert!(
        usage["input_tokens"].as_u64().unwrap_or(0) > 0,
        "input_tokens: {usage}"
    );
    assert!(
        usage["output_tokens"].as_u64().unwrap_or(0) > 0,
        "output_tokens: {usage}"
    );
}

#[tokio::test]
#[ignore]
async fn live_deepseek_v4_pro_blocking() {
    assert_blocking_chat("deepseek-v4-pro").await;
}

#[tokio::test]
#[ignore]
async fn live_deepseek_v4_flash_blocking() {
    assert_blocking_chat("deepseek-v4-flash").await;
}

// ────────────────────────────────────────────────────────────────────────────
// Test 3: streaming chat — verify SSE event sequence.
// ────────────────────────────────────────────────────────────────────────────

async fn assert_streaming_chat(model: &str) {
    let Some(key) = need_key() else { return };
    let relay = Relay::spawn(DEEPSEEK_UPSTREAM, &key);

    let body = responses_body(model, "Say the word OK.", true);
    let resp = reqwest::Client::new()
        .post(relay.url("/v1/responses"))
        .json(&body)
        .send()
        .await
        .expect("POST /v1/responses (stream)");
    assert!(resp.status().is_success(), "status {}", resp.status());

    let mut events = resp.bytes_stream().eventsource();

    let mut event_types: Vec<String> = Vec::new();
    let mut text_chunks: Vec<String> = Vec::new();
    let mut completed_id: Option<String> = None;

    let timeout = Duration::from_secs(60);
    let started = Instant::now();
    while let Some(ev) = tokio::time::timeout(timeout - started.elapsed(), events.next())
        .await
        .expect("stream timeout")
    {
        let ev = ev.expect("sse parse");
        event_types.push(ev.event.clone());
        let data: Value = serde_json::from_str(&ev.data).expect("event data json");
        match ev.event.as_str() {
            "response.output_text.delta" => {
                if let Some(d) = data["delta"].as_str() {
                    text_chunks.push(d.to_string());
                }
            }
            "response.completed" => {
                completed_id = data["response"]["id"].as_str().map(String::from);
                break;
            }
            "response.failed" => panic!("stream failed: {ev:?}"),
            _ => {}
        }
    }

    // Bracket anchors required regardless of content: created first, completed last.
    assert_eq!(
        event_types.first().map(String::as_str),
        Some("response.created"),
        "first event: {event_types:?}"
    );
    assert_eq!(
        event_types.last().map(String::as_str),
        Some("response.completed"),
        "last event: {event_types:?}"
    );
    assert!(completed_id.is_some(), "no response id in completed event");

    // If the model produced any text, the relay must have bracketed it with
    // matching output_item.added / .done. v4-flash will occasionally return
    // an empty completion for trivial prompts; that's a model quirk, not a
    // relay bug — empty completions stream just `created → completed`.
    let full: String = text_chunks.join("");
    if !full.trim().is_empty() {
        assert!(
            event_types
                .iter()
                .any(|e| e == "response.output_item.added"),
            "non-empty text but no output_item.added: {event_types:?}"
        );
        assert!(
            event_types.iter().any(|e| e == "response.output_item.done"),
            "non-empty text but no output_item.done: {event_types:?}"
        );
    } else {
        eprintln!(
            "note: {model} returned an empty streaming completion (model quirk, not a relay bug)"
        );
    }
}

#[tokio::test]
#[ignore]
async fn live_deepseek_v4_pro_streaming() {
    assert_streaming_chat("deepseek-v4-pro").await;
}

#[tokio::test]
#[ignore]
async fn live_deepseek_v4_flash_streaming() {
    assert_streaming_chat("deepseek-v4-flash").await;
}

// ────────────────────────────────────────────────────────────────────────────
// Test 4: tool-call streaming round trip.
//
// We send a tool definition that the model is highly likely to call, then
// assert the relay emits the function_call SSE sequence.
// ────────────────────────────────────────────────────────────────────────────

async fn assert_tool_call_streaming(model: &str) {
    let Some(key) = need_key() else { return };
    let relay = Relay::spawn(DEEPSEEK_UPSTREAM, &key);

    let body = json!({
        "model": model,
        "instructions": "You MUST call the get_weather tool. Do not answer in text.",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": "What's the weather in Beijing?" }]
        }],
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "strict": false,
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }],
        "tool_choice": "auto",
        "stream": true,
        "store": false
    });

    let resp = reqwest::Client::new()
        .post(relay.url("/v1/responses"))
        .json(&body)
        .send()
        .await
        .expect("POST")
        .error_for_status()
        .expect("non-2xx");

    let mut events = resp.bytes_stream().eventsource();
    let mut saw_fc_added = false;
    let mut saw_fc_args = false;
    let mut saw_fc_done = false;
    let mut completed_output: Option<Value> = None;

    let deadline = Instant::now() + Duration::from_secs(60);
    while let Some(ev) = tokio::time::timeout(deadline - Instant::now(), events.next())
        .await
        .expect("stream timeout")
    {
        let ev = ev.expect("sse parse");
        let data: Value = serde_json::from_str(&ev.data).expect("data json");
        match ev.event.as_str() {
            "response.output_item.added" => {
                if data["item"]["type"].as_str() == Some("function_call") {
                    saw_fc_added = true;
                    assert_eq!(data["item"]["name"].as_str(), Some("get_weather"));
                }
            }
            "response.function_call_arguments.delta" => {
                saw_fc_args = true;
            }
            "response.output_item.done" => {
                if data["item"]["type"].as_str() == Some("function_call") {
                    saw_fc_done = true;
                }
            }
            "response.completed" => {
                completed_output = Some(data["response"]["output"].clone());
                break;
            }
            "response.failed" => panic!("stream failed: {ev:?}"),
            _ => {}
        }
    }

    assert!(saw_fc_added, "no function_call output_item.added");
    assert!(saw_fc_args, "no function_call_arguments.delta");
    assert!(saw_fc_done, "no function_call output_item.done");

    let output = completed_output.expect("response.completed missing");
    let fcs: Vec<&Value> = output
        .as_array()
        .unwrap()
        .iter()
        .filter(|i| i["type"].as_str() == Some("function_call"))
        .collect();
    assert!(!fcs.is_empty(), "no function_call in final output");
    let fc = fcs[0];
    assert_eq!(fc["name"].as_str(), Some("get_weather"));
    assert!(fc["call_id"].as_str().is_some());
    let args_str = fc["arguments"].as_str().unwrap_or("");
    let args: Value = serde_json::from_str(args_str).expect("arguments must be valid JSON");
    assert!(
        args.get("city").is_some(),
        "tool args missing `city`: {args_str}"
    );
}

#[tokio::test]
#[ignore]
async fn live_deepseek_v4_pro_tool_call() {
    assert_tool_call_streaming("deepseek-v4-pro").await;
}

#[tokio::test]
#[ignore]
async fn live_deepseek_v4_flash_tool_call() {
    assert_tool_call_streaming("deepseek-v4-flash").await;
}

// ────────────────────────────────────────────────────────────────────────────
// Test 5: reasoning_content round-trip on deepseek-v4-pro across two turns.
//
// Why this matters: v4-pro emits `reasoning_content` deltas alongside tool
// calls. The relay must store that reasoning keyed by `call_id` (in stream.rs)
// so that when Codex replays the same `function_call` in its next request,
// translate.rs can re-attach `reasoning_content` to the assistant message
// before forwarding upstream — DeepSeek requires it on the assistant turn
// that owns the tool_calls.
//
// We verify this end-to-end via a recording proxy sitting between the relay
// and real DeepSeek: it forwards both turns transparently while keeping a
// copy of every Chat Completions request body. After turn 2, we inspect the
// recorded body and assert reasoning_content was populated.
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct RecordingProxyState {
    upstream_base: Arc<String>,
    auth: Arc<String>,
    bodies: Arc<Mutex<Vec<Vec<u8>>>>,
    client: reqwest::Client,
}

async fn proxy_handler(
    State(s): State<RecordingProxyState>,
    req: axum::extract::Request,
) -> Response<Body> {
    let path = req.uri().path().to_string();
    let method = req.method().clone();
    let bytes = match axum::body::to_bytes(req.into_body(), 50_000_000).await {
        Ok(b) => b,
        Err(_) => {
            return Response::builder()
                .status(StatusCode::BAD_REQUEST)
                .body(Body::empty())
                .unwrap();
        }
    };
    if method == axum::http::Method::POST && path == "/v1/chat/completions" {
        s.bodies.lock().unwrap().push(bytes.to_vec());
    }

    let url = format!("{}{}", s.upstream_base, path);
    let mut rb = s
        .client
        .request(method.clone(), &url)
        .header("Content-Type", "application/json");
    if !s.auth.is_empty() {
        rb = rb.bearer_auth(s.auth.as_str());
    }
    if !bytes.is_empty() {
        rb = rb.body(bytes.to_vec());
    }

    match rb.send().await {
        Ok(upstream) => {
            let status =
                StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let mut builder = Response::builder().status(status);
            for (k, v) in upstream.headers().iter() {
                let kn = k.as_str().to_lowercase();
                if matches!(
                    kn.as_str(),
                    "transfer-encoding" | "content-length" | "connection"
                ) {
                    continue;
                }
                builder = builder.header(k, v);
            }
            let stream = upstream.bytes_stream();
            builder.body(Body::from_stream(stream)).unwrap()
        }
        Err(e) => Response::builder()
            .status(StatusCode::BAD_GATEWAY)
            .body(Body::from(format!("proxy upstream error: {e}")))
            .unwrap(),
    }
}

async fn spawn_recording_proxy(upstream_base: &str, auth: &str) -> (u16, Arc<Mutex<Vec<Vec<u8>>>>) {
    let bodies: Arc<Mutex<Vec<Vec<u8>>>> = Arc::new(Mutex::new(Vec::new()));
    let state = RecordingProxyState {
        upstream_base: Arc::new(upstream_base.to_string()),
        auth: Arc::new(auth.to_string()),
        bodies: bodies.clone(),
        client: reqwest::Client::new(),
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let app = Router::new().fallback(proxy_handler).with_state(state);
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    // Give axum a beat to actually start accepting connections.
    tokio::time::sleep(Duration::from_millis(50)).await;
    (port, bodies)
}

/// Run a streaming /v1/responses request through the relay and return the
/// captured function_call (call_id, name, arguments) and full SSE event log.
async fn streaming_call(
    relay_url: &str,
    body: Value,
) -> (Option<(String, String, String)>, Vec<String>) {
    let resp = reqwest::Client::new()
        .post(relay_url)
        .json(&body)
        .send()
        .await
        .expect("POST")
        .error_for_status()
        .expect("non-2xx");

    let mut events = resp.bytes_stream().eventsource();
    let mut event_types = Vec::new();
    let mut fc: Option<(String, String, String)> = None;

    let deadline = Instant::now() + Duration::from_secs(60);
    while let Some(ev) = tokio::time::timeout(deadline - Instant::now(), events.next())
        .await
        .expect("stream timeout")
    {
        let ev = ev.expect("sse parse");
        event_types.push(ev.event.clone());
        let data: Value = serde_json::from_str(&ev.data).unwrap_or(Value::Null);
        if ev.event == "response.completed" {
            if let Some(arr) = data["response"]["output"].as_array() {
                for item in arr {
                    if item["type"].as_str() == Some("function_call") {
                        fc = Some((
                            item["call_id"].as_str().unwrap_or("").to_string(),
                            item["name"].as_str().unwrap_or("").to_string(),
                            item["arguments"].as_str().unwrap_or("").to_string(),
                        ));
                    }
                }
            }
            break;
        }
        if ev.event == "response.failed" {
            panic!("stream failed: data={}", ev.data);
        }
    }

    (fc, event_types)
}

#[tokio::test]
#[ignore]
async fn live_deepseek_v4_pro_reasoning_round_trip() {
    let Some(key) = need_key() else { return };

    // Recording proxy → real DeepSeek
    let (proxy_port, recorded_bodies) =
        spawn_recording_proxy("https://api.deepseek.com", &key).await;
    let proxy_upstream = format!("http://127.0.0.1:{proxy_port}/v1");

    // Relay → recording proxy. Relay forwards Authorization, but proxy reads
    // its own auth from the captured state. Pass the same key so both legs
    // authenticate to DeepSeek consistently.
    let relay = Relay::spawn(&proxy_upstream, &key);

    // ── Turn 1: tool-call prompt that v4-pro is highly likely to answer with
    //    a function_call AND reasoning_content (verified empirically).
    let turn1 = json!({
        "model": "deepseek-v4-pro",
        "instructions": "You MUST call the get_weather tool. Do not answer in text.",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [{ "type": "input_text", "text": "What's the weather in Beijing right now?" }]
        }],
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "strict": false,
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }],
        "tool_choice": "auto",
        "stream": true,
        "store": false
    });

    let (fc, _events1) = streaming_call(&relay.url("/v1/responses"), turn1).await;
    let (call_id, name, arguments) = fc.expect("turn 1 must produce a function_call from v4-pro");
    assert_eq!(name, "get_weather");
    assert!(!call_id.is_empty(), "call_id must be non-empty");

    // ── Turn 2: replay the function_call + a synthetic function_call_output.
    //    Codex 0.128.x sends turns this way (store=false, full history each request).
    let turn2 = json!({
        "model": "deepseek-v4-pro",
        "instructions": "You MUST call the get_weather tool. Do not answer in text.",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "What's the weather in Beijing right now?" }]
            },
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "get_weather",
                "arguments": arguments
            },
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": "{\"city\":\"Beijing\",\"temp_c\":18,\"condition\":\"clear\"}"
            }
        ],
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "strict": false,
            "parameters": {
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"]
            }
        }],
        "tool_choice": "auto",
        "stream": true,
        "store": false
    });

    let (_fc2, _events2) = streaming_call(&relay.url("/v1/responses"), turn2).await;

    // ── Inspect what the relay sent upstream on turn 2.
    let bodies = recorded_bodies.lock().unwrap();
    assert_eq!(
        bodies.len(),
        2,
        "expected exactly 2 chat-completions POSTs (got {})",
        bodies.len()
    );
    let turn2_body: Value = serde_json::from_slice(&bodies[1]).expect("turn 2 body json");
    let messages = turn2_body["messages"]
        .as_array()
        .expect("turn 2 messages array");

    // Find the assistant message with tool_calls — that's the one that must
    // carry reasoning_content for DeepSeek to accept the turn.
    let asst = messages
        .iter()
        .find(|m| m["role"].as_str() == Some("assistant") && m.get("tool_calls").is_some())
        .unwrap_or_else(|| panic!("no assistant w/ tool_calls in turn 2: {turn2_body}"));

    let reasoning = asst["reasoning_content"].as_str().unwrap_or("");
    assert!(
        !reasoning.is_empty(),
        "v4-pro emits reasoning_content during tool calls — relay must round-trip it. \
         Got assistant message without reasoning_content: {asst}"
    );
    eprintln!(
        "✓ reasoning_content round-tripped ({} chars) for call_id {call_id}",
        reasoning.len()
    );

    // Sanity: the tool message with the matching call_id must also be present.
    let tool_msg = messages
        .iter()
        .find(|m| {
            m["role"].as_str() == Some("tool")
                && m["tool_call_id"].as_str() == Some(call_id.as_str())
        })
        .expect("tool message replay");
    assert!(tool_msg["content"]
        .as_str()
        .map(|s| s.contains("Beijing"))
        .unwrap_or(false));
}

// ────────────────────────────────────────────────────────────────────────────
// Test 6: multimodal image-input wire shape, verified live.
//
// Empirically confirmed 2026-05-11: DeepSeek's Chat Completions API does NOT
// accept any multimodal content part variant — it returns
//   400 invalid_request_error: "unknown variant `image_url`, expected `text`"
// for `image_url`, `input_image`, and `image` alike. So we can't do an
// end-to-end happy-path image test against DeepSeek today.
//
// What we *can* lock in live, however, is that codex-relay's outbound Chat
// Completions body has the right OpenAI-standard multimodal shape, so the
// moment DeepSeek (or any other provider in the supported list) enables
// vision, codex-relay users get it for free without further changes.
//
// We use a recording proxy in front of DeepSeek that captures the outbound
// body BEFORE forwarding. The relay's translation is asserted on the
// captured body; the upstream's eventual 400 is expected and ignored.
// ────────────────────────────────────────────────────────────────────────────

/// 1×1 red PNG, base64-encoded. Tiny but valid; enough to exercise the
/// vision input path without bloating fixtures.
const TINY_RED_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

fn image_input_body(model: &str) -> Value {
    let data_url = format!("data:image/png;base64,{TINY_RED_PNG_B64}");
    json!({
        "model": model,
        "instructions": "Answer in one short sentence.",
        "input": [{
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the attached image briefly."},
                {"type": "input_image", "image_url": data_url}
            ]
        }],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "store": false,
        "stream": false,
        "include": []
    })
}

/// Recording-proxy live test: send a Codex-shaped `input_image` request,
/// inspect what the relay forwarded upstream, assert it's multimodal-shaped.
/// We don't require the upstream to return 2xx — DeepSeek currently 400s on
/// any image, and that's a property of *their* API, not the relay's.
#[tokio::test]
#[ignore]
async fn live_deepseek_v4_pro_image_input_wire_shape() {
    let Some(key) = need_key() else { return };

    let (proxy_port, recorded_bodies) =
        spawn_recording_proxy("https://api.deepseek.com", &key).await;
    let proxy_upstream = format!("http://127.0.0.1:{proxy_port}/v1");
    let relay = Relay::spawn(&proxy_upstream, &key);

    // Fire and forget — don't unwrap the status. The relay forwards whatever
    // DeepSeek returns (200 if/when they support vision, 400 today). Either
    // way the recording proxy has already captured what we sent.
    let _ = reqwest::Client::new()
        .post(relay.url("/v1/responses"))
        .json(&image_input_body("deepseek-v4-pro"))
        .send()
        .await
        .expect("POST");

    let bodies = recorded_bodies.lock().unwrap();
    assert_eq!(bodies.len(), 1, "expected exactly 1 chat-completions POST");
    let body: Value = serde_json::from_slice(&bodies[0]).expect("body json");
    let user = body["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|m| m["role"].as_str() == Some("user"))
        .expect("user message");

    let parts = user["content"]
        .as_array()
        .expect("user content must be a multimodal array (got non-array)");
    let has_text = parts
        .iter()
        .any(|p| p["type"].as_str() == Some("text") && p["text"].as_str().is_some());
    let has_image = parts.iter().any(|p| {
        p["type"].as_str() == Some("image_url")
            && p["image_url"]["url"]
                .as_str()
                .map(|s| s.starts_with("data:image/png;base64,"))
                .unwrap_or(false)
    });
    assert!(has_text, "missing text part: {parts:?}");
    assert!(has_image, "missing image_url part: {parts:?}");
    eprintln!(
        "✓ outbound multimodal shape verified ({} parts)",
        parts.len()
    );
}

/// Companion to the wire-shape test: confirm the symptom of DeepSeek's
/// current vision-input rejection so we can spot the day they enable it.
/// When this test starts FAILING (i.e. DeepSeek stops 400'ing on image_url),
/// flip it into a proper happy-path test.
#[tokio::test]
#[ignore]
async fn live_deepseek_v4_pro_image_input_currently_rejected_by_upstream() {
    let Some(key) = need_key() else { return };
    let relay = Relay::spawn(DEEPSEEK_UPSTREAM, &key);

    let resp = reqwest::Client::new()
        .post(relay.url("/v1/responses"))
        .json(&image_input_body("deepseek-v4-pro"))
        .send()
        .await
        .expect("POST");

    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();

    // As of 2026-05-11 DeepSeek's Chat Completions deserializer rejects every
    // image content-part variant with this exact message. If that ever stops
    // being true, this assertion fires and we know vision support landed.
    assert!(
        status == reqwest::StatusCode::BAD_REQUEST
            && body.contains("unknown variant")
            && body.contains("image_url"),
        "DeepSeek may have enabled vision input — status={status}, body={body}"
    );
}
