mod session;
mod stream;
mod translate;
mod types;
mod upstream_request;

use anyhow::{bail, Result};
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use reqwest::{Client, Url};
use session::{SessionStore, DEFAULT_MAX_SESSIONS, DEFAULT_MAX_SESSION_BYTES, DEFAULT_SESSION_TTL};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tracing::{debug, error, info, warn};
use types::*;
use upstream_request::UpstreamRequestConfig;

const DEBUG_NAME_LIMIT: usize = 80;

#[derive(Parser, Debug)]
#[command(
    name = "codex-relay",
    about = "Responses API ↔ Chat Completions bridge",
    version = env!("CARGO_PKG_VERSION")
)]
struct Args {
    #[arg(long, env = "CODEX_RELAY_PORT", default_value = "4444")]
    port: u16,

    /// IP address to bind the listener to (e.g. 0.0.0.0 to accept remote connections).
    #[arg(long, env = "CODEX_RELAY_BIND", default_value = "127.0.0.1")]
    bind: std::net::IpAddr,

    #[arg(
        long,
        env = "CODEX_RELAY_UPSTREAM",
        default_value = "https://openrouter.ai/api/v1"
    )]
    upstream: String,

    #[arg(long, env = "CODEX_RELAY_API_KEY", default_value = "")]
    api_key: String,

    /// JSON object merged into every upstream Chat Completions request body.
    #[arg(long, env = "CODEX_RELAY_UPSTREAM_EXTRA_PARAMS")]
    upstream_extra_params: Option<String>,

    /// JSON array of top-level upstream request parameter names to remove.
    #[arg(long, env = "CODEX_RELAY_DROP_PARAMS")]
    drop_upstream_params: Option<String>,

    /// Print a ready-to-use Codex config.toml snippet (including model_properties)
    /// for all models exposed by the upstream provider.
    #[arg(long)]
    print_config: bool,

    /// Maximum completed response histories retained for previous_response_id.
    #[arg(
        long,
        env = "CODEX_RELAY_MAX_SESSIONS",
        default_value_t = DEFAULT_MAX_SESSIONS
    )]
    max_sessions: usize,

    /// Approximate memory budget for retained session/reasoning state, in MiB.
    #[arg(
        long,
        env = "CODEX_RELAY_MAX_SESSION_MEMORY_MB",
        default_value_t = DEFAULT_MAX_SESSION_BYTES / 1024 / 1024
    )]
    max_session_memory_mb: usize,

    /// Retain idle session/reasoning state for this many hours.
    #[arg(
        long,
        env = "CODEX_RELAY_SESSION_TTL_HOURS",
        default_value_t = DEFAULT_SESSION_TTL.as_secs() / 60 / 60
    )]
    session_ttl_hours: u64,

    /// History retention backend: memory or disk.
    #[arg(long, env = "CODEX_RELAY_HISTORY_STORE", default_value = "memory")]
    history_store: String,

    /// Directory used when CODEX_RELAY_HISTORY_STORE=disk.
    #[arg(
        long,
        env = "CODEX_RELAY_HISTORY_DIR",
        default_value = ".codex-relay-history"
    )]
    history_dir: PathBuf,
}

#[derive(Clone)]
struct AppState {
    sessions: SessionStore,
    client: Client,
    upstream: Arc<Url>,
    api_key: Arc<String>,
    upstream_request: Arc<UpstreamRequestConfig>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "codex_relay=info".into()),
        )
        .init();

    let args = Args::parse();

    let upstream = validate_upstream(&args.upstream)?;
    let upstream_request = Arc::new(UpstreamRequestConfig::from_raw(
        args.upstream_extra_params.as_deref(),
        args.drop_upstream_params.as_deref(),
    )?);

    let client = Client::new();
    let api_key = Arc::new(args.api_key);

    // --print-config: fetch models and print Codex config snippet, then exit.
    if args.print_config {
        let provider_name = upstream
            .host_str()
            .map(|h| {
                let h = h.trim_start_matches("api.").trim_start_matches("www.");
                h.trim_end_matches(".com")
                    .trim_end_matches(".cn")
                    .trim_end_matches(".ai")
                    .trim_end_matches(".org")
                    .trim_end_matches(".io")
            })
            .unwrap_or("custom");
        print_codex_config(&client, &upstream, &api_key, provider_name).await;
        return Ok(());
    }

    let max_session_bytes = args
        .max_session_memory_mb
        .saturating_mul(1024)
        .saturating_mul(1024);
    let session_ttl = Duration::from_secs(args.session_ttl_hours.saturating_mul(60 * 60));
    let sessions = match args.history_store.as_str() {
        "memory" => {
            SessionStore::with_limits_and_ttl(args.max_sessions, max_session_bytes, session_ttl)
        }
        "disk" => SessionStore::with_disk_limits_and_ttl(
            &args.history_dir,
            args.max_sessions,
            max_session_bytes,
            session_ttl,
        )?,
        other => bail!("history store must be 'memory' or 'disk', got: {other}"),
    };
    let state = AppState {
        sessions,
        client: client.clone(),
        upstream: Arc::new(upstream.clone()),
        api_key: api_key.clone(),
        upstream_request: upstream_request.clone(),
    };
    info!(
        "session retention: store={} dir={} ttl={}h max_sessions={} max_session_memory={} MiB",
        args.history_store,
        args.history_dir.display(),
        args.session_ttl_hours,
        args.max_sessions,
        args.max_session_memory_mb
    );
    if !upstream_request.is_empty() {
        info!(
            "upstream request params: extra={} drop={}",
            upstream_request.extra_param_count(),
            upstream_request.drop_param_count()
        );
    }

    // Fetch upstream model list asynchronously for user visibility
    tokio::spawn(log_upstream_models(client, Arc::new(upstream), api_key));

    tokio::spawn(cleanup_sessions(state.sessions.clone()));

    // Disable axum's default 2 MiB body cap: Codex CLI may send base64-encoded
    // image attachments that easily exceed it, and a framework-level 413 looks
    // like a transport-layer death to Codex and crashes the session (#2).
    // The relay binds 127.0.0.1 by default; --bind can expose it more widely,
    // so be mindful of DoS when binding to a non-loopback address.
    let app = Router::new()
        .route("/v1/responses", post(handle_responses))
        .route("/v1/models", get(handle_models))
        .fallback(handle_fallback)
        .layer(DefaultBodyLimit::disable())
        .with_state(state.clone());

    let addr = std::net::SocketAddr::new(args.bind, args.port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    // Log the actual bound address so that `--port 0` (ephemeral port) reports
    // the real port instead of `:0`.
    info!(
        "codex-relay listening on {} → {}",
        listener.local_addr()?,
        state.upstream.as_ref()
    );
    axum::serve(listener, app).await?;

    Ok(())
}

/// Validate that `--upstream` is an acceptable HTTP(S) URL.
fn validate_upstream(raw: &str) -> Result<Url> {
    let url = Url::parse(raw.trim_end_matches('/'))?;
    match url.scheme() {
        "http" | "https" => {}
        s => bail!("upstream URL scheme must be http or https, got: {s}"),
    }
    if url.host_str().is_none() {
        bail!("upstream URL must have a host");
    }
    Ok(url)
}

/// Fetch upstream models and log them at startup so users know what's available.
async fn log_upstream_models(client: Client, upstream: Arc<Url>, api_key: Arc<String>) {
    let url = format!("{}models", join_base(&upstream));
    let mut builder = client.get(&url);
    if !api_key.is_empty() {
        builder = builder.bearer_auth(api_key.as_str());
    }

    let result = tokio::time::timeout(Duration::from_secs(5), builder.send()).await;

    match result {
        Ok(Ok(r)) if r.status().is_success() => {
            if let Ok(body) = r.json::<serde_json::Value>().await {
                let models: Vec<_> = body
                    .get("data")
                    .or_else(|| body.get("models"))
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|m| m.get("id").and_then(|id| id.as_str()))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                if !models.is_empty() {
                    info!("upstream models: {}", models.join(", "));
                    info!(
                        "⚠️  To configure Codex with model metadata, run:  codex-relay --print-config --upstream {} {}",
                        upstream.as_str(),
                        if api_key.is_empty() { "" } else { "--api-key ..." }
                    );
                }
            }
        }
        Ok(Ok(r)) => warn!(
            "upstream models: status {} (check credentials?)",
            r.status()
        ),
        Ok(Err(e)) => warn!("upstream models: request error: {e}"),
        Err(_elapsed) => warn!("upstream models: request timed out (upstream unreachable?)"),
    }
}

async fn cleanup_sessions(sessions: SessionStore) {
    let mut interval = tokio::time::interval(Duration::from_secs(60 * 60));
    loop {
        interval.tick().await;
        sessions.cleanup();
    }
}

/// Print a Codex config.toml snippet that includes model_properties for all
/// upstream models, so users can avoid "model metadata not found" warnings.
async fn print_codex_config(client: &Client, upstream: &Url, api_key: &str, provider_name: &str) {
    let url = format!("{}models", join_base(upstream));
    let mut builder = client.get(&url);
    if !api_key.is_empty() {
        builder = builder.bearer_auth(api_key);
    }

    let models: Vec<String> = match builder.send().await {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
            Ok(body) => body
                .get("data")
                .or_else(|| body.get("models"))
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|m| m.get("id").and_then(|id| id.as_str()).map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            Err(e) => {
                eprintln!("// Failed to parse upstream models: {e}");
                eprintln!("// Falling back to a generic snippet. Replace <YOUR_MODEL> below.");
                vec!["<YOUR_MODEL>".into()]
            }
        },
        status => {
            eprintln!("// Failed to fetch upstream models (status: {status:?})");
            eprintln!("// Falling back to a generic snippet. Replace <YOUR_MODEL> below.");
            vec!["<YOUR_MODEL>".into()]
        }
    };

    println!(
        "# ── Codex config snippet for {} ──",
        upstream.host_str().unwrap_or("custom")
    );
    println!("# Copy the lines below into ~/.codex/config.toml");
    println!();
    println!("model_provider = \"{provider_name}\"");

    if !models.is_empty() && !models[0].starts_with('<') {
        println!("model = \"{}\"", models[0]);
    } else {
        println!("model = \"<CHOOSE_A_MODEL>\"");
    }
    println!();
    println!("[model_providers.{provider_name}]");
    println!("name = \"{}\"", provider_name);
    println!("base_url = \"{}\"", upstream.as_str().trim_end_matches('/'));
    println!("wire_api = \"responses\"");
    println!(
        "env_key = \"{}_API_KEY\"",
        provider_name.to_uppercase().replace(['-', '.'], "_")
    );
    println!();

    for model in &models {
        let props = estimate_model_properties(model);
        println!("[model_properties.\"{}\"]", model);
        println!("context_window = {}", props.context_window);
        println!("max_context_window = {}", props.max_context_window);
        println!(
            "supports_parallel_tool_calls = {}",
            props.supports_parallel_tool_calls
        );
        println!(
            "supports_reasoning_summaries = {}",
            props.supports_reasoning_summaries
        );
        println!("input_modalities = [\"text\"]");
        println!();
    }
}

struct ModelProps {
    context_window: u32,
    max_context_window: u32,
    supports_parallel_tool_calls: bool,
    supports_reasoning_summaries: bool,
}

/// Heuristic-based model property estimation.
/// Providers don't expose context window sizes in their /v1/models endpoint,
/// so we use conservative defaults based on model family name.
fn estimate_model_properties(model_id: &str) -> ModelProps {
    let lower = model_id.to_lowercase();

    // Reasoning models (DeepSeek-R1, kimi-k2.6, etc.)
    let has_reasoning = lower.contains("reasoner")
        || lower.contains("r1")
        || lower.contains("k2")
        || lower.contains("o1")
        || lower.contains("thinking")
        || lower.contains("deepseek-v4");

    // Context window estimation by family
    let (ctx, max_ctx) = if lower.contains("gpt-5") {
        (272_000, 1_000_000)
    } else if lower.contains("gpt-4.5") || lower.contains("gpt-4o") {
        (128_000, 128_000)
    } else if lower.contains("claude") {
        (200_000, 200_000)
    } else if lower.contains("gemini") {
        (1_000_000, 2_000_000)
    } else if lower.contains("deepseek") {
        (262_144, 1_048_576)
    } else if lower.contains("qwen") {
        (131_072, 131_072)
    } else if lower.contains("kimi")
        || lower.contains("moonshot")
        || lower.contains("mistral")
        || lower.contains("llama")
        || lower.contains("codestral")
    {
        (128_000, 128_000)
    } else {
        // Conservative default for unknown models
        (128_000, 128_000)
    };

    ModelProps {
        context_window: ctx,
        max_context_window: max_ctx,
        supports_parallel_tool_calls: true,
        supports_reasoning_summaries: has_reasoning,
    }
}

fn join_base(url: &Url) -> String {
    let s = url.as_str();
    if s.ends_with('/') {
        s.to_string()
    } else {
        format!("{s}/")
    }
}

/// GET /v1/models — proxy to upstream and normalize so both legacy
/// (`{data:[...]}`) and Codex 0.128+ (`{models:[...]}`) consumers are happy.
async fn handle_models(State(state): State<AppState>) -> Response {
    info!("GET /v1/models");
    let url = format!("{}models", join_base(&state.upstream));
    let mut builder = state.client.get(&url);
    if !state.api_key.is_empty() {
        builder = builder.bearer_auth(state.api_key.as_str());
    }

    let upstream_body: Option<serde_json::Value> = match builder.send().await {
        Ok(r) if r.status().is_success() => match r.json::<serde_json::Value>().await {
            Ok(b) => Some(b),
            Err(e) => {
                warn!("upstream models: parse error: {e}");
                None
            }
        },
        Ok(r) => {
            warn!("upstream models: status {}", r.status());
            None
        }
        Err(e) => {
            warn!("upstream models: request error: {e}");
            None
        }
    };

    let list = upstream_body
        .as_ref()
        .and_then(|b| b.get("data").or_else(|| b.get("models")))
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    Json(serde_json::json!({
        "object": "list",
        "data": list.clone(),
        "models": list,
    }))
    .into_response()
}

/// Catch-all: log unknown requests so we can see what Codex is sending.
async fn handle_fallback(req: Request) -> Response {
    warn!("unhandled {} {}", req.method(), req.uri().path());
    (StatusCode::NOT_FOUND, "not found").into_response()
}

fn summarize_debug_names(names: Vec<String>) -> String {
    if names.is_empty() {
        return "(none)".to_string();
    }

    let total = names.len();
    let mut shown = names
        .into_iter()
        .take(DEBUG_NAME_LIMIT)
        .collect::<Vec<_>>()
        .join(", ");
    if total > DEBUG_NAME_LIMIT {
        shown.push_str(&format!(", ... (+{} more)", total - DEBUG_NAME_LIMIT));
    }
    shown
}

fn response_tool_debug_names(tools: &[serde_json::Value]) -> Vec<String> {
    let mut names = Vec::new();
    for tool in tools {
        match tool.get("type").and_then(serde_json::Value::as_str) {
            Some("function") => {
                if let Some(name) = tool
                    .get("name")
                    .or_else(|| tool.get("function").and_then(|f| f.get("name")))
                    .and_then(serde_json::Value::as_str)
                {
                    names.push(name.to_string());
                }
            }
            Some("namespace") => {
                let namespace = tool
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                if let Some(subs) = tool.get("tools").and_then(serde_json::Value::as_array) {
                    for sub in subs {
                        if sub.get("type").and_then(serde_json::Value::as_str) == Some("function") {
                            if let Some(name) = sub.get("name").and_then(serde_json::Value::as_str)
                            {
                                names.push(
                                    crate::translate::chat_function_name_for_namespace_tool(
                                        namespace, name,
                                    ),
                                );
                            }
                        }
                    }
                }
            }
            Some(kind) => names.push(format!("<{kind}>")),
            None => {}
        }
    }
    names
}

fn chat_tool_debug_names(tools: &[serde_json::Value]) -> Vec<String> {
    tools
        .iter()
        .filter_map(|tool| {
            tool.get("function")
                .and_then(|f| f.get("name"))
                .and_then(serde_json::Value::as_str)
                .or_else(|| tool.get("name").and_then(serde_json::Value::as_str))
                .map(String::from)
        })
        .collect()
}

fn chat_response_tool_call_debug_names(chat_resp: &ChatResponse) -> Vec<String> {
    chat_resp
        .choices
        .iter()
        .flat_map(|choice| choice.message.tool_calls.iter())
        .flatten()
        .filter_map(|tool_call| {
            tool_call
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(serde_json::Value::as_str)
                .map(String::from)
        })
        .collect()
}

async fn handle_responses(State(state): State<AppState>, body: axum::body::Bytes) -> Response {
    let req: ResponsesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            error!("JSON parse error: {e}");
            error!(
                "body prefix: {}",
                String::from_utf8_lossy(&body[..body.len().min(200)])
            );
            return (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response();
        }
    };
    debug!(
        "→ model={} stream={} input_items={} tools={} prev_resp={:?}",
        req.model,
        req.stream,
        match &req.input {
            crate::types::ResponsesInput::Messages(v) => v.len(),
            _ => 1,
        },
        req.tools.len(),
        req.previous_response_id
    );
    debug!(
        "→ response tools={}",
        summarize_debug_names(response_tool_debug_names(&req.tools))
    );

    handle_responses_inner(state, req).await
}

async fn handle_responses_inner(state: AppState, req: ResponsesRequest) -> Response {
    let mut history = req
        .previous_response_id
        .as_deref()
        .map(|id| state.sessions.get_history(id))
        .unwrap_or_default();
    if should_isolate_spawn_child_request(&req, &history) {
        debug!("isolating spawned child request from parent response history");
        history.clear();
    }

    let model = req.model.clone();
    let namespace_tools = translate::namespace_tool_map(&req.tools);
    let mut chat_req = translate::to_chat_request(&req, history, &state.sessions);
    debug!(
        "→ upstream tools={}",
        summarize_debug_names(chat_tool_debug_names(&chat_req.tools))
    );
    let url = format!("{}chat/completions", join_base(&state.upstream));

    if req.stream {
        let response_id = state.sessions.new_id();
        chat_req.stream = true;
        let request_messages = chat_req.messages.clone();
        stream::translate_stream(stream::StreamArgs {
            client: state.client,
            url,
            api_key: state.api_key,
            chat_req,
            upstream_request: state.upstream_request,
            response_id,
            sessions: state.sessions,
            request_messages,
            namespace_tools,
            model,
        })
        .into_response()
    } else {
        chat_req.stream = false;
        handle_blocking(state, chat_req, url, model, namespace_tools).await
    }
}

fn should_isolate_spawn_child_request(req: &ResponsesRequest, history: &[ChatMessage]) -> bool {
    let Some(input_text) = isolated_user_text(&req.input) else {
        return false;
    };
    let completed_tool_calls: std::collections::HashSet<&str> = history
        .iter()
        .filter_map(|msg| msg.tool_call_id.as_deref())
        .collect();
    let pending_spawns = history
        .iter()
        .flat_map(|msg| msg.tool_calls.as_deref().unwrap_or(&[]))
        .filter(|call| {
            let call_id = call.get("id").and_then(serde_json::Value::as_str);
            call_id.is_none_or(|id| !completed_tool_calls.contains(id))
        })
        .filter_map(parse_spawn_agent_call)
        .collect::<Vec<_>>();

    if pending_spawns
        .iter()
        .any(|spawn| spawn.message.as_deref() == Some(input_text))
    {
        return true;
    }

    pending_spawns.len() == 1 && pending_spawns[0].is_v2_encrypted_candidate()
}

fn isolated_user_text(input: &ResponsesInput) -> Option<&str> {
    match input {
        ResponsesInput::Text(text) => Some(text.as_str()),
        ResponsesInput::Messages(items) => {
            if items.len() != 1 {
                return None;
            }
            let item = &items[0];
            if item.get("type").and_then(serde_json::Value::as_str) != Some("message")
                || item.get("role").and_then(serde_json::Value::as_str) != Some("user")
            {
                return None;
            }
            match item.get("content") {
                Some(serde_json::Value::String(text)) => Some(text.as_str()),
                Some(serde_json::Value::Array(parts)) if parts.len() == 1 => {
                    parts[0].get("text").and_then(serde_json::Value::as_str)
                }
                _ => None,
            }
        }
    }
}

struct SpawnAgentCall {
    message: Option<String>,
    fork_turns: Option<String>,
}

impl SpawnAgentCall {
    fn is_v2_encrypted_candidate(&self) -> bool {
        self.fork_turns.is_some()
            && self
                .message
                .as_deref()
                .is_some_and(|message| !message.is_empty())
    }
}

fn parse_spawn_agent_call(call: &serde_json::Value) -> Option<SpawnAgentCall> {
    if call
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(serde_json::Value::as_str)
        != Some("spawn_agent")
    {
        return None;
    }
    let arguments = call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .and_then(serde_json::Value::as_str)?;
    let arguments: serde_json::Value = serde_json::from_str(arguments).ok()?;
    Some(SpawnAgentCall {
        message: arguments
            .get("message")
            .and_then(serde_json::Value::as_str)
            .map(String::from),
        fork_turns: arguments
            .get("fork_turns")
            .and_then(serde_json::Value::as_str)
            .map(String::from),
    })
}

async fn handle_blocking(
    state: AppState,
    chat_req: types::ChatRequest,
    url: String,
    model: String,
    namespace_tools: translate::NamespaceToolMap,
) -> Response {
    let mut builder = state
        .client
        .post(&url)
        .header("Content-Type", "application/json");

    if !state.api_key.is_empty() {
        builder = builder.bearer_auth(state.api_key.as_str());
    }

    let upstream_body = match state.upstream_request.request_body(&chat_req) {
        Ok(body) => body,
        Err(e) => {
            error!("upstream request body error: {e}");
            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    };

    match builder.json(&upstream_body).send().await {
        Err(e) => {
            error!("upstream error: {e}");
            (StatusCode::BAD_GATEWAY, e.to_string()).into_response()
        }
        Ok(r) if !r.status().is_success() => {
            let status = r.status();
            let body = r.text().await.unwrap_or_default();
            error!("upstream {status}: {body}");
            (
                StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                body,
            )
                .into_response()
        }
        Ok(r) => match r.json::<ChatResponse>().await {
            Err(e) => {
                error!("parse error: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
            }
            Ok(chat_resp) => {
                debug!(
                    "← upstream function_calls={}",
                    summarize_debug_names(chat_response_tool_call_debug_names(&chat_resp))
                );
                let assistant_msg = chat_resp
                    .choices
                    .first()
                    .map(|c| c.message.clone())
                    .unwrap_or_else(|| ChatMessage {
                        role: "assistant".into(),
                        content: Some(serde_json::Value::String(String::new())),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });

                let mut full_history = chat_req.messages.clone();
                full_history.push(assistant_msg);
                let response_id = state.sessions.save(full_history);

                let (resp, _) = if namespace_tools.is_empty() {
                    translate::from_chat_response(response_id, &model, chat_resp)
                } else {
                    translate::from_chat_response_with_tool_map(
                        response_id,
                        &model,
                        chat_resp,
                        &namespace_tools,
                    )
                };
                Json(resp).into_response()
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_upstream_https() {
        let url = validate_upstream("https://openrouter.ai/api/v1").unwrap();
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str(), Some("openrouter.ai"));
    }

    #[test]
    fn test_validate_upstream_http_localhost() {
        let url = validate_upstream("http://localhost:8080/v1").unwrap();
        assert_eq!(url.scheme(), "http");
        assert_eq!(url.host_str(), Some("localhost"));
    }

    #[test]
    fn test_validate_upstream_rejects_ftp() {
        assert!(validate_upstream("ftp://evil.com").is_err());
    }

    #[test]
    fn test_validate_upstream_rejects_file() {
        assert!(validate_upstream("file:///etc/passwd").is_err());
    }

    #[test]
    fn test_validate_upstream_rejects_garbage() {
        assert!(validate_upstream("not-a-url").is_err());
    }

    #[test]
    fn test_validate_upstream_trailing_slash_stripped() {
        let url = validate_upstream("https://api.example.com/v1/").unwrap();
        assert!(!url.as_str().ends_with("/v1//"));
    }

    #[test]
    fn test_join_base_adds_trailing_slash() {
        let url = Url::parse("https://api.example.com/v1").unwrap();
        assert_eq!(join_base(&url), "https://api.example.com/v1/");
    }

    #[test]
    fn test_join_base_preserves_trailing_slash() {
        let url = Url::parse("https://api.example.com/v1/").unwrap();
        assert_eq!(join_base(&url), "https://api.example.com/v1/");
    }

    #[test]
    fn test_response_tool_debug_names_include_flat_and_namespace_tools() {
        let tools = vec![
            json!({"type": "function", "name": "spawn_agent"}),
            json!({
                "type": "namespace",
                "name": "mcp__codex_apps__github",
                "tools": [
                    {"type": "function", "name": "_fetch_issue"},
                    {"type": "web_search"}
                ]
            }),
            json!({"type": "web_search"}),
        ];

        assert_eq!(
            response_tool_debug_names(&tools),
            vec![
                "spawn_agent".to_string(),
                "mcp__codex_apps__github-_fetch_issue".to_string(),
                "<web_search>".to_string(),
            ]
        );
    }

    #[test]
    fn test_chat_response_tool_call_debug_names_do_not_include_arguments() {
        let chat_resp = ChatResponse {
            choices: vec![ChatChoice {
                message: ChatMessage {
                    role: "assistant".into(),
                    content: None,
                    reasoning_content: None,
                    tool_calls: Some(vec![json!({
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"task\":\"secret\"}"
                        }
                    })]),
                    tool_call_id: None,
                    name: None,
                },
            }],
            usage: None,
        };

        assert_eq!(
            chat_response_tool_call_debug_names(&chat_resp),
            vec!["spawn_agent".to_string()]
        );
    }

    #[test]
    fn test_spawn_child_request_isolated_when_input_matches_spawn_message() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("child task".into()),
            previous_response_id: Some("resp_parent".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
        };
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![json!({
                "id": "call_spawn",
                "type": "function",
                "function": {
                    "name": "spawn_agent",
                    "arguments": "{\"task_name\":\"child\",\"message\":\"child task\"}"
                }
            })]),
            tool_call_id: None,
            name: None,
        }];

        assert!(should_isolate_spawn_child_request(&req, &history));
    }

    #[test]
    fn test_spawn_child_request_isolated_for_single_v2_encrypted_spawn() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("child task decrypted by codex".into()),
            previous_response_id: Some("resp_parent".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
        };
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![json!({
                "id": "call_spawn",
                "type": "function",
                "function": {
                    "name": "spawn_agent",
                    "arguments": "{\"task_name\":\"child\",\"fork_turns\":\"current_turn\",\"message\":\"encrypted:v2:ciphertext\"}"
                }
            })]),
            tool_call_id: None,
            name: None,
        }];

        assert!(should_isolate_spawn_child_request(&req, &history));
    }

    #[test]
    fn test_spawn_child_v2_encrypted_fallback_requires_unambiguous_spawn() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("child task decrypted by codex".into()),
            previous_response_id: Some("resp_parent".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
        };
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![
                json!({
                    "id": "call_spawn_a",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"task_name\":\"a\",\"fork_turns\":\"current_turn\",\"message\":\"encrypted:v2:a\"}"
                    }
                }),
                json!({
                    "id": "call_spawn_b",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"task_name\":\"b\",\"fork_turns\":\"current_turn\",\"message\":\"encrypted:v2:b\"}"
                    }
                }),
            ]),
            tool_call_id: None,
            name: None,
        }];

        assert!(!should_isolate_spawn_child_request(&req, &history));
    }

    #[test]
    fn test_spawn_child_isolation_does_not_match_tool_outputs() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Messages(vec![json!({
                "type": "function_call_output",
                "call_id": "call_spawn",
                "output": "child result"
            })]),
            previous_response_id: Some("resp_parent".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
        };
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![json!({
                "id": "call_spawn",
                "type": "function",
                "function": {
                    "name": "spawn_agent",
                    "arguments": "{\"task_name\":\"child\",\"message\":\"child task\"}"
                }
            })]),
            tool_call_id: None,
            name: None,
        }];

        assert!(!should_isolate_spawn_child_request(&req, &history));
    }

    #[test]
    fn test_spawn_child_isolation_ignores_completed_spawn_calls() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Text("child task".into()),
            previous_response_id: Some("resp_parent".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
        };
        let history = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![json!({
                    "id": "call_spawn",
                    "type": "function",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"task_name\":\"child\",\"message\":\"child task\"}"
                    }
                })]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(serde_json::Value::String("4".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: Some("call_spawn".into()),
                name: None,
            },
        ];

        assert!(!should_isolate_spawn_child_request(&req, &history));
    }

    #[test]
    fn test_estimate_model_properties_deepseek() {
        let props = estimate_model_properties("deepseek-v4-pro");
        assert_eq!(props.context_window, 262_144);
        assert_eq!(props.max_context_window, 1_048_576);
        assert!(props.supports_reasoning_summaries);
        assert!(props.supports_parallel_tool_calls);
    }

    #[test]
    fn test_estimate_model_properties_deepseek_r1() {
        let props = estimate_model_properties("deepseek-r1");
        assert!(props.supports_reasoning_summaries);
    }

    #[test]
    fn test_estimate_model_properties_unknown() {
        let props = estimate_model_properties("some-unknown-model");
        assert_eq!(props.context_window, 128_000);
        assert_eq!(props.max_context_window, 128_000);
        assert!(!props.supports_reasoning_summaries);
        assert!(props.supports_parallel_tool_calls);
    }
}
