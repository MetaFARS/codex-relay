mod corpus;
mod dsml;
mod quirks;
mod session;
mod stream;
mod think;
mod translate;
mod types;
mod upstream_request;

use anyhow::{bail, Context, Result};
use axum::{
    extract::{DefaultBodyLimit, Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use corpus::CorpusRecorder;
use reqwest::{Client, Url};
use session::{SessionStore, DEFAULT_MAX_SESSIONS, DEFAULT_MAX_SESSION_BYTES, DEFAULT_SESSION_TTL};
use std::{fs, path::PathBuf, process::Command, sync::Arc, time::Duration};
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

    /// Print a ready-to-use Codex config.toml snippet.
    #[arg(long)]
    print_config: bool,

    /// Write a version-matched Codex model catalog and reference it from --print-config.
    #[arg(long, requires = "print_config", value_name = "PATH")]
    model_catalog: Option<PathBuf>,

    /// Bundled Codex model whose tool protocol and instructions custom models inherit.
    #[arg(long, requires = "model_catalog", value_name = "MODEL")]
    model_template: Option<String>,

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

    /// Append the conversation flow of every completed turn to daily-sharded
    /// JSONL files (OpenAI messages format) in this directory. Off by default.
    /// Records contain prompts, tool call arguments, and tool outputs — treat
    /// the directory as sensitive.
    #[arg(long, env = "CODEX_RELAY_RECORD_CORPUS")]
    record_corpus: Option<PathBuf>,
}

#[derive(Clone)]
struct AppState {
    sessions: SessionStore,
    client: Client,
    upstream: Arc<Url>,
    api_key: Arc<String>,
    upstream_request: Arc<UpstreamRequestConfig>,
    corpus: Option<CorpusRecorder>,
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
        print_codex_config(
            &client,
            &upstream,
            &api_key,
            provider_name,
            args.model_catalog.as_deref(),
            args.model_template.as_deref(),
        )
        .await?;
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
    let corpus = match &args.record_corpus {
        Some(dir) => {
            let recorder = CorpusRecorder::new(dir)?;
            warn!(
                "corpus recording ENABLED → {} (records prompts, tool arguments, and tool outputs; treat as sensitive)",
                dir.display()
            );
            Some(recorder)
        }
        None => None,
    };
    let state = AppState {
        sessions,
        client: client.clone(),
        upstream: Arc::new(upstream.clone()),
        api_key: api_key.clone(),
        upstream_request: upstream_request.clone(),
        corpus,
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
                        "⚠️  To configure Codex with model metadata, run:  codex-relay --print-config --model-catalog ~/.codex/codex-relay-models.json --upstream {} {}",
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

/// Print a Codex config.toml snippet and optionally generate the full model
/// catalog required by recent Codex versions.
async fn print_codex_config(
    client: &Client,
    upstream: &Url,
    api_key: &str,
    provider_name: &str,
    model_catalog_path: Option<&std::path::Path>,
    model_template: Option<&str>,
) -> Result<()> {
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

    if let Some(path) = model_catalog_path {
        let bundled = load_bundled_codex_catalog()?;
        let catalog = build_model_catalog(bundled, &models, model_template, provider_name)?;
        let bytes = serde_json::to_vec_pretty(&catalog)?;
        fs::write(path, bytes)
            .with_context(|| format!("failed to write model catalog to {}", path.display()))?;
    }

    println!(
        "# ── Codex config snippet for {} ──",
        upstream.host_str().unwrap_or("custom")
    );
    println!("# Copy the lines below into ~/.codex/config.toml");
    println!();
    println!("model_provider = {}", toml_string(provider_name)?);

    if !models.is_empty() && !models[0].starts_with('<') {
        println!("model = {}", toml_string(&models[0])?);
    } else {
        println!("model = \"<CHOOSE_A_MODEL>\"");
    }
    if let Some(path) = model_catalog_path {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()?.join(path)
        };
        let absolute = absolute
            .to_str()
            .context("model catalog path is not valid UTF-8")?;
        println!("model_catalog_json = {}", toml_string(absolute)?);
    } else {
        println!("# To register upstream models in Codex's model picker, rerun with:");
        println!("#   --model-catalog ~/.codex/codex-relay-models.json");
    }
    println!();
    println!("[model_providers.{}]", toml_string(provider_name)?);
    println!("name = {}", toml_string(provider_name)?);
    println!(
        "base_url = {}",
        toml_string(upstream.as_str().trim_end_matches('/'))?
    );
    println!("wire_api = \"responses\"");
    let env_key = format!(
        "{}_API_KEY",
        provider_name.to_uppercase().replace(['-', '.'], "_")
    );
    println!("env_key = {}", toml_string(&env_key)?);
    println!();
    Ok(())
}

fn toml_string(value: &str) -> Result<String> {
    serde_json::to_string(value).context("failed to quote TOML string")
}

fn load_bundled_codex_catalog() -> Result<serde_json::Value> {
    let output = Command::new("codex")
        .args(["debug", "models", "--bundled"])
        .output()
        .context("failed to run `codex debug models --bundled`; install Codex CLI or omit --model-catalog")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("`codex debug models --bundled` failed: {}", stderr.trim());
    }
    serde_json::from_slice(&output.stdout)
        .context("failed to parse the bundled model catalog from the installed Codex CLI")
}

fn build_model_catalog(
    mut catalog: serde_json::Value,
    upstream_models: &[String],
    template_slug: Option<&str>,
    provider_name: &str,
) -> Result<serde_json::Value> {
    let models = catalog
        .get_mut("models")
        .and_then(serde_json::Value::as_array_mut)
        .context("the installed Codex CLI returned a catalog without a models array")?;
    let template = if let Some(slug) = template_slug {
        models
            .iter()
            .find(|model| model.get("slug").and_then(serde_json::Value::as_str) == Some(slug))
            .cloned()
            .with_context(|| {
                format!("model template `{slug}` was not found in the bundled catalog")
            })?
    } else {
        models
            .iter()
            .find(|model| {
                model.get("visibility").and_then(serde_json::Value::as_str) == Some("list")
            })
            .or_else(|| models.first())
            .cloned()
            .context("the installed Codex CLI returned an empty bundled catalog")?
    };

    for (index, slug) in upstream_models.iter().enumerate() {
        if slug.starts_with('<') {
            continue;
        }
        let existing_index = models
            .iter()
            .position(|model| model.get("slug").and_then(serde_json::Value::as_str) == Some(slug));
        let props = estimate_model_properties(slug);
        let mut model = template.clone();
        let object = model
            .as_object_mut()
            .context("the bundled model template was not a JSON object")?;
        object.insert("slug".into(), serde_json::Value::String(slug.clone()));
        object.insert(
            "display_name".into(),
            serde_json::Value::String(slug.clone()),
        );
        object.insert(
            "description".into(),
            serde_json::Value::String(format!("{slug} via {provider_name}")),
        );
        object.insert(
            "visibility".into(),
            serde_json::Value::String("list".into()),
        );
        object.insert("supported_in_api".into(), serde_json::Value::Bool(true));
        object.insert("priority".into(), serde_json::json!(10_000 + index));
        object.insert(
            "context_window".into(),
            serde_json::json!(props.context_window),
        );
        object.insert(
            "max_context_window".into(),
            serde_json::json!(props.max_context_window),
        );
        object.insert(
            "supports_parallel_tool_calls".into(),
            serde_json::Value::Bool(props.supports_parallel_tool_calls),
        );
        set_existing(
            object,
            "supports_reasoning_summaries",
            serde_json::Value::Bool(props.supports_reasoning_summaries),
        );
        set_existing(
            object,
            "supports_reasoning_summary_parameter",
            serde_json::Value::Bool(props.supports_reasoning_summaries),
        );
        object.insert("input_modalities".into(), serde_json::json!(["text"]));

        // Preserve version-sensitive instructions and tool encodings from the
        // template, while disabling capabilities the relay does not promise.
        set_existing(object, "prefer_websockets", serde_json::Value::Bool(false));
        object.insert("support_verbosity".into(), serde_json::Value::Bool(false));
        object.insert("default_verbosity".into(), serde_json::Value::Null);
        object.insert("default_reasoning_level".into(), serde_json::Value::Null);
        object.insert("supported_reasoning_levels".into(), serde_json::json!([]));
        set_existing(
            object,
            "supports_image_detail_original",
            serde_json::Value::Bool(false),
        );
        object.insert(
            "supports_search_tool".into(),
            serde_json::Value::Bool(false),
        );
        object.insert("use_responses_lite".into(), serde_json::Value::Bool(false));
        object.insert("tool_mode".into(), serde_json::Value::Null);
        object.insert("multi_agent_version".into(), serde_json::Value::Null);
        object.insert("experimental_supported_tools".into(), serde_json::json!([]));
        object.insert("additional_speed_tiers".into(), serde_json::json!([]));
        object.insert("service_tiers".into(), serde_json::json!([]));
        object.insert("default_service_tier".into(), serde_json::Value::Null);
        object.insert("availability_nux".into(), serde_json::Value::Null);
        object.insert("upgrade".into(), serde_json::Value::Null);
        object.insert("auto_review_model_override".into(), serde_json::Value::Null);
        object.insert("auto_compact_token_limit".into(), serde_json::Value::Null);
        object.insert("comp_hash".into(), serde_json::Value::Null);
        object.remove("minimal_client_version");
        object.remove("available_in_plans");
        if let Some(index) = existing_index {
            models[index] = model;
        } else {
            models.push(model);
        }
    }

    Ok(catalog)
}

fn set_existing(
    object: &mut serde_json::Map<String, serde_json::Value>,
    key: &str,
    value: serde_json::Value,
) {
    if object.contains_key(key) {
        object.insert(key.into(), value);
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
            error!(
                error_category = ?e.classify(),
                line = e.line(),
                column = e.column(),
                body_bytes = body.len(),
                "JSON parse error"
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

async fn handle_responses_inner(state: AppState, mut req: ResponsesRequest) -> Response {
    if let Err(message) = translate::validate_unique_chat_tool_names(&req.tools) {
        return (StatusCode::UNPROCESSABLE_ENTITY, message).into_response();
    }
    let mut history = req
        .previous_response_id
        .as_deref()
        .map(|id| state.sessions.get_history(id))
        .unwrap_or_default();
    if let Err(message) = normalize_agent_message_content(&mut req.input, &history) {
        return (StatusCode::UNPROCESSABLE_ENTITY, message).into_response();
    }
    if should_isolate_spawn_child_request(&req, &history) {
        debug!("isolating spawned child request from parent response history");
        history.clear();
    }

    let model = req.model.clone();
    let namespace_tools = translate::namespace_tool_map(&req.tools);
    let custom_tools = translate::custom_tool_map(&req.tools);
    let mut chat_req = translate::to_chat_request(&req, history, &state.sessions);
    debug!(
        "→ upstream tools={}",
        summarize_debug_names(chat_tool_debug_names(&chat_req.tools))
    );
    let url = format!("{}chat/completions", join_base(&state.upstream));

    let previous_response_id = req.previous_response_id.clone();
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
            custom_tools,
            model,
            corpus: state.corpus,
            previous_response_id,
        })
        .into_response()
    } else {
        chat_req.stream = false;
        handle_blocking(
            state,
            chat_req,
            url,
            model,
            namespace_tools,
            custom_tools,
            previous_response_id,
        )
        .await
    }
}

fn should_isolate_spawn_child_request(req: &ResponsesRequest, history: &[ChatMessage]) -> bool {
    let Some(input) = isolated_child_input(&req.input) else {
        return false;
    };
    let pending_spawns = pending_spawn_agent_calls(history);

    if let Some(recipient) = input.recipient.as_deref() {
        return pending_spawns
            .iter()
            .any(|spawn| spawn.matches_recipient(recipient));
    }

    if pending_spawns
        .iter()
        .any(|spawn| spawn.message.as_deref() == input.text.as_deref())
    {
        return true;
    }

    pending_spawns.len() == 1 && pending_spawns[0].is_v2_encrypted_candidate()
}

fn pending_spawn_agent_calls(history: &[ChatMessage]) -> Vec<SpawnAgentCall> {
    let mut call_id_counts = std::collections::HashMap::new();
    for call_id in history
        .iter()
        .flat_map(|msg| msg.tool_calls.as_deref().unwrap_or(&[]))
        .filter_map(|call| call.get("id").and_then(serde_json::Value::as_str))
    {
        *call_id_counts.entry(call_id).or_insert(0usize) += 1;
    }
    let completed_tool_calls: std::collections::HashSet<&str> = history
        .iter()
        .filter_map(|msg| msg.tool_call_id.as_deref())
        .collect();
    history
        .iter()
        .flat_map(|msg| msg.tool_calls.as_deref().unwrap_or(&[]))
        .filter(|call| {
            let call_id = call.get("id").and_then(serde_json::Value::as_str);
            call_id.is_none_or(|id| {
                id.is_empty()
                    || call_id_counts.get(id) != Some(&1)
                    || !completed_tool_calls.contains(id)
            })
        })
        .filter_map(parse_spawn_agent_call)
        .collect()
}

struct IsolatedChildInput {
    text: Option<String>,
    recipient: Option<String>,
}

fn isolated_child_input(input: &ResponsesInput) -> Option<IsolatedChildInput> {
    match input {
        ResponsesInput::Text(text) => Some(IsolatedChildInput {
            text: Some(text.clone()),
            recipient: None,
        }),
        ResponsesInput::Messages(items) => {
            if items.len() != 1 {
                return None;
            }
            let item = &items[0];
            match item.get("type").and_then(serde_json::Value::as_str) {
                Some("message")
                    if item.get("role").and_then(serde_json::Value::as_str) == Some("user") =>
                {
                    let text = match item.get("content") {
                        Some(serde_json::Value::String(text)) => Some(text.clone()),
                        Some(serde_json::Value::Array(parts)) if parts.len() == 1 => parts[0]
                            .get("text")
                            .and_then(serde_json::Value::as_str)
                            .map(String::from),
                        _ => None,
                    }?;
                    Some(IsolatedChildInput {
                        text: Some(text),
                        recipient: None,
                    })
                }
                Some("agent_message") => {
                    let recipient = item
                        .get("recipient")
                        .and_then(serde_json::Value::as_str)?
                        .to_string();
                    let parts = item.get("content").and_then(serde_json::Value::as_array)?;
                    if parts.iter().any(|part| {
                        !matches!(
                            part.get("type").and_then(serde_json::Value::as_str),
                            Some("input_text" | "text")
                        ) || part
                            .get("text")
                            .and_then(serde_json::Value::as_str)
                            .is_none()
                    }) {
                        return None;
                    }
                    let text = parts
                        .iter()
                        .filter_map(|part| part.get("text").and_then(serde_json::Value::as_str))
                        .collect::<Vec<_>>()
                        .join("");
                    if text.is_empty() {
                        return None;
                    }
                    Some(IsolatedChildInput {
                        text: Some(text),
                        recipient: Some(recipient),
                    })
                }
                _ => None,
            }
        }
    }
}

fn normalize_agent_message_content(
    input: &mut ResponsesInput,
    history: &[ChatMessage],
) -> Result<(), &'static str> {
    let ResponsesInput::Messages(items) = input else {
        return Ok(());
    };
    let pending_spawns = pending_spawn_agent_calls(history);
    for item in items {
        if item.get("type").and_then(serde_json::Value::as_str) != Some("agent_message") {
            continue;
        }
        if contains_encrypted_content(item.get("content").unwrap_or(&serde_json::Value::Null)) {
            normalize_legacy_encrypted_agent_message(item, &pending_spawns)?;
        }
        let content = item.get("content").unwrap_or(&serde_json::Value::Null);
        let Some(parts) = content.as_array() else {
            return Err("agent_message content must be an array of plaintext text parts");
        };
        if parts.is_empty()
            || parts.iter().any(|part| {
                !matches!(
                    part.get("type").and_then(serde_json::Value::as_str),
                    Some("input_text" | "text")
                ) || part
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .is_none()
            })
        {
            return Err("agent_message content must contain only plaintext text parts");
        }
    }
    Ok(())
}

/// Recover the legacy non-OpenAI V2 shape only when session history proves the
/// wrapper contains the exact plaintext task. Never infer plaintext from the
/// encrypted value's format: genuine provider ciphertext must remain rejected.
fn normalize_legacy_encrypted_agent_message(
    item: &mut serde_json::Value,
    pending_spawns: &[SpawnAgentCall],
) -> Result<(), &'static str> {
    let recipient = item
        .get("recipient")
        .and_then(serde_json::Value::as_str)
        .ok_or(
            "encrypted agent_message content cannot be forwarded to a Chat Completions upstream",
        )?;
    let parts = item
        .get("content")
        .and_then(serde_json::Value::as_array)
        .ok_or(
            "encrypted agent_message content cannot be forwarded to a Chat Completions upstream",
        )?;
    let encrypted_parts = parts
        .iter()
        .filter(|part| {
            part.get("type").and_then(serde_json::Value::as_str) == Some("encrypted_content")
        })
        .collect::<Vec<_>>();
    if encrypted_parts.len() != 1
        || parts.iter().any(|part| {
            !matches!(
                part.get("type").and_then(serde_json::Value::as_str),
                Some("input_text" | "text" | "encrypted_content")
            )
        })
    {
        return Err(
            "encrypted agent_message content cannot be forwarded to a Chat Completions upstream",
        );
    }
    let wrapped_text = encrypted_parts[0]
        .get("encrypted_content")
        .and_then(serde_json::Value::as_str)
        .ok_or(
            "encrypted agent_message content cannot be forwarded to a Chat Completions upstream",
        )?
        .to_string();
    let matching_spawns = pending_spawns
        .iter()
        .filter(|spawn| {
            spawn.matches_recipient(recipient)
                && spawn.message.as_deref() == Some(wrapped_text.as_str())
        })
        .count();
    if matching_spawns != 1 {
        return Err(
            "encrypted agent_message content cannot be forwarded to a Chat Completions upstream",
        );
    }

    let parts = item
        .get_mut("content")
        .and_then(serde_json::Value::as_array_mut)
        .expect("agent_message content checked as array");
    for part in parts {
        if part.get("type").and_then(serde_json::Value::as_str) == Some("encrypted_content") {
            *part = serde_json::json!({"type": "input_text", "text": &wrapped_text});
        }
    }
    Ok(())
}

fn contains_encrypted_content(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Array(values) => values.iter().any(contains_encrypted_content),
        serde_json::Value::Object(object) => {
            object.get("type").and_then(serde_json::Value::as_str) == Some("encrypted_content")
                || object.values().any(contains_encrypted_content)
        }
        _ => false,
    }
}

struct SpawnAgentCall {
    task_name: Option<String>,
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

    fn matches_recipient(&self, recipient: &str) -> bool {
        self.task_name.as_deref().is_some_and(|task_name| {
            !task_name.is_empty()
                && (recipient == task_name
                    || recipient
                        .strip_suffix(task_name)
                        .is_some_and(|prefix| prefix.ends_with('/')))
        })
    }
}

fn parse_spawn_agent_call(call: &serde_json::Value) -> Option<SpawnAgentCall> {
    if call
        .get("function")
        .and_then(|function| function.get("name"))
        .and_then(serde_json::Value::as_str)
        .is_none_or(|name| !matches!(name, "spawn_agent" | "collaboration-spawn_agent"))
    {
        return None;
    }
    let arguments = call
        .get("function")
        .and_then(|function| function.get("arguments"))
        .and_then(serde_json::Value::as_str)?;
    let arguments: serde_json::Value = serde_json::from_str(arguments).ok()?;
    Some(SpawnAgentCall {
        task_name: arguments
            .get("task_name")
            .and_then(serde_json::Value::as_str)
            .map(String::from),
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
    custom_tools: translate::CustomToolMap,
    previous_response_id: Option<String>,
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
    let allowed_tool_names =
        translate::allowed_upstream_tool_names(&chat_req.tools, &upstream_body);

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
        Ok(r) => {
            let value = match r.json::<serde_json::Value>().await {
                Ok(value) => value,
                Err(e) => {
                    error!("parse error: {e}");
                    return (StatusCode::BAD_GATEWAY, e.to_string()).into_response();
                }
            };
            if let Some(upstream_error) = value.get("error") {
                let message = upstream_error
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .or_else(|| upstream_error.as_str())
                    .unwrap_or("upstream returned an error")
                    .to_string();
                error!("upstream response error: {message}");
                return (StatusCode::BAD_GATEWAY, message).into_response();
            }
            match serde_json::from_value::<ChatResponse>(value) {
                Err(e) => {
                    error!("parse error: {e}");
                    (StatusCode::BAD_GATEWAY, e.to_string()).into_response()
                }
                Ok(chat_resp) => {
                    debug!(
                        "← upstream function_calls={}",
                        summarize_debug_names(chat_response_tool_call_debug_names(&chat_resp))
                    );
                    let response_id = state.sessions.new_id();
                    let (resp, assistant_messages) =
                        if namespace_tools.is_empty() && custom_tools.is_empty() {
                            translate::from_chat_response(response_id.clone(), &model, chat_resp)
                        } else {
                            translate::from_chat_response_with_tool_maps(
                                response_id.clone(),
                                &model,
                                chat_resp,
                                &namespace_tools,
                                &custom_tools,
                            )
                        };
                    let tool_call_entries = assistant_messages
                        .iter()
                        .flat_map(|message| message.tool_calls.as_deref().unwrap_or(&[]))
                        .map(|call| {
                            let call_id = call
                                .get("id")
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("");
                            let name = call
                                .get("function")
                                .and_then(|function| function.get("name"))
                                .and_then(serde_json::Value::as_str)
                                .unwrap_or("");
                            (call_id, name)
                        });
                    if let Err(message) = translate::validate_tool_call_entries(
                        tool_call_entries,
                        &allowed_tool_names,
                    ) {
                        warn!("rejecting invalid upstream tool calls: {message}");
                        return (StatusCode::BAD_GATEWAY, message).into_response();
                    }

                    for assistant in &assistant_messages {
                        if let Some(reasoning) = assistant
                            .reasoning_content
                            .as_ref()
                            .filter(|reasoning| !reasoning.is_empty())
                        {
                            state.sessions.store_turn_reasoning(
                                &chat_req.messages,
                                assistant,
                                reasoning.clone(),
                            );
                        }
                    }
                    let mut full_history = chat_req.messages;
                    full_history.extend(assistant_messages);
                    if let Some(corpus) = &state.corpus {
                        corpus.record_turn(
                            previous_response_id.as_deref(),
                            &response_id,
                            &model,
                            &full_history,
                        );
                    }
                    state.sessions.save_with_id(response_id, full_history);
                    Json(resp).into_response()
                }
            }
        }
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
            reasoning: None,
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
    fn test_plaintext_agent_message_isolated_by_recipient() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Messages(vec![json!({
                "type": "agent_message",
                "author": "/root",
                "recipient": "/root/worker-b",
                "content": [{
                    "type": "input_text",
                    "text": "Message Type: NEW_TASK\nPayload:\ndo B"
                }]
            })]),
            previous_response_id: Some("resp_parent".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
        };
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![
                json!({
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "collaboration-spawn_agent",
                        "arguments": "{\"task_name\":\"worker-a\",\"message\":\"do A\"}"
                    }
                }),
                json!({
                    "id": "call_b",
                    "type": "function",
                    "function": {
                        "name": "collaboration-spawn_agent",
                        "arguments": "{\"task_name\":\"worker-b\",\"message\":\"do B\"}"
                    }
                }),
            ]),
            tool_call_id: None,
            name: None,
        }];

        assert!(should_isolate_spawn_child_request(&req, &history));
    }

    #[test]
    fn test_plaintext_agent_message_does_not_isolate_unmatched_recipient() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Messages(vec![json!({
                "type": "agent_message",
                "recipient": "/root/existing-child",
                "content": [{"type": "input_text", "text": "follow up"}]
            })]),
            previous_response_id: Some("resp_child".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
        };
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![json!({
                "id": "call_new",
                "type": "function",
                "function": {
                    "name": "collaboration-spawn_agent",
                    "arguments": "{\"task_name\":\"new-child\",\"message\":\"new task\"}"
                }
            })]),
            tool_call_id: None,
            name: None,
        }];

        assert!(!should_isolate_spawn_child_request(&req, &history));
    }

    #[test]
    fn test_encrypted_agent_message_is_rejected_before_translation() {
        for content in [
            json!([
                {"type": "input_text", "text": "routing"},
                {"type": "encrypted_content", "encrypted_content": "opaque-secret"}
            ]),
            json!({"type": "encrypted_content", "encrypted_content": "opaque-secret"}),
            json!([{
                "type": "wrapper",
                "payload": {"type": "encrypted_content", "encrypted_content": "opaque-secret"}
            }]),
        ] {
            let mut input = ResponsesInput::Messages(vec![json!({
                "type": "agent_message",
                "recipient": "/root/worker",
                "content": content
            })]);
            assert!(normalize_agent_message_content(&mut input, &[]).is_err());
            assert!(isolated_child_input(&input).is_none());
        }
    }

    #[test]
    fn test_opaque_agent_message_is_not_treated_as_plaintext_child() {
        let mut input = ResponsesInput::Messages(vec![json!({
            "type": "agent_message",
            "recipient": "/root/worker",
            "content": [{"type": "unknown", "payload": "opaque"}]
        })]);

        assert!(normalize_agent_message_content(&mut input, &[]).is_err());
        assert!(isolated_child_input(&input).is_none());
    }

    #[test]
    fn test_legacy_encrypted_agent_message_is_normalized_on_exact_pending_spawn_match() {
        let task = "请列出当前目录下的文件。";
        let mut input = ResponsesInput::Messages(vec![json!({
            "type": "agent_message",
            "recipient": "/root/list_files",
            "content": [
                {"type": "input_text", "text": "Message Type: NEW_TASK\nPayload:\n"},
                {"type": "encrypted_content", "encrypted_content": task}
            ]
        })]);
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![json!({
                "id": "call_spawn",
                "type": "function",
                "function": {
                    "name": "collaboration-spawn_agent",
                    "arguments": json!({
                        "fork_turns": "none",
                        "message": task,
                        "task_name": "list_files"
                    }).to_string()
                }
            })]),
            tool_call_id: None,
            name: None,
        }];

        assert!(normalize_agent_message_content(&mut input, &history).is_ok());
        let ResponsesInput::Messages(items) = &input else {
            panic!("message input");
        };
        assert_eq!(
            items[0]["content"][1],
            json!({"type": "input_text", "text": task})
        );
        assert!(isolated_child_input(&input).is_some());
    }

    #[test]
    fn test_legacy_encrypted_agent_message_rejects_ambiguous_pending_spawn_match() {
        let task = "same task";
        let mut input = ResponsesInput::Messages(vec![json!({
            "type": "agent_message",
            "recipient": "/root/worker",
            "content": [{"type": "encrypted_content", "encrypted_content": task}]
        })]);
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(
                ["call_a", "call_b"]
                    .into_iter()
                    .map(|call_id| {
                        json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "collaboration-spawn_agent",
                                "arguments": json!({
                                    "message": task,
                                    "task_name": "worker"
                                }).to_string()
                            }
                        })
                    })
                    .collect(),
            ),
            tool_call_id: None,
            name: None,
        }];

        assert!(normalize_agent_message_content(&mut input, &history).is_err());
        assert!(contains_encrypted_content(match &input {
            ResponsesInput::Messages(items) => &items[0]["content"],
            ResponsesInput::Text(_) => unreachable!(),
        }));
    }

    #[test]
    fn test_legacy_encrypted_agent_message_requires_both_recipient_and_message_match() {
        let history = vec![ChatMessage {
            role: "assistant".into(),
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![json!({
                "id": "call_spawn",
                "type": "function",
                "function": {
                    "name": "collaboration-spawn_agent",
                    "arguments": "{\"task_name\":\"worker\",\"message\":\"expected task\"}"
                }
            })]),
            tool_call_id: None,
            name: None,
        }];

        for (recipient, wrapped_text) in [
            ("/root/other-worker", "expected task"),
            ("/root/worker", "gAAAAA-opaque-ciphertext"),
        ] {
            let mut input = ResponsesInput::Messages(vec![json!({
                "type": "agent_message",
                "recipient": recipient,
                "content": [{
                    "type": "encrypted_content",
                    "encrypted_content": wrapped_text
                }]
            })]);

            assert!(normalize_agent_message_content(&mut input, &history).is_err());
        }
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
            reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
            reasoning: None,
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
    fn test_duplicate_spawn_call_ids_cannot_disable_child_isolation() {
        let req = ResponsesRequest {
            model: "test".into(),
            input: ResponsesInput::Messages(vec![json!({
                "type": "agent_message",
                "recipient": "/root/worker-b",
                "content": [{"type": "input_text", "text": "task B"}]
            })]),
            previous_response_id: Some("resp_parent".into()),
            tools: vec![],
            stream: false,
            temperature: None,
            max_output_tokens: None,
            system: None,
            instructions: None,
            reasoning: None,
        };
        let history = vec![
            ChatMessage {
                role: "assistant".into(),
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![
                    json!({
                        "id": "duplicate",
                        "type": "function",
                        "function": {
                            "name": "collaboration-spawn_agent",
                            "arguments": "{\"task_name\":\"worker-a\",\"message\":\"task A\"}"
                        }
                    }),
                    json!({
                        "id": "duplicate",
                        "type": "function",
                        "function": {
                            "name": "collaboration-spawn_agent",
                            "arguments": "{\"task_name\":\"worker-b\",\"message\":\"task B\"}"
                        }
                    }),
                ]),
                tool_call_id: None,
                name: None,
            },
            ChatMessage {
                role: "tool".into(),
                content: Some(serde_json::Value::String("spawned worker-a".into())),
                reasoning_content: None,
                tool_calls: None,
                tool_call_id: Some("duplicate".into()),
                name: None,
            },
        ];

        assert!(should_isolate_spawn_child_request(&req, &history));
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

    #[test]
    fn test_toml_string_quotes_untrusted_model_ids() {
        assert_eq!(
            toml_string("model\"\nnotify = [\"bad\"]").unwrap(),
            "\"model\\\"\\nnotify = [\\\"bad\\\"]\""
        );
    }

    #[test]
    fn test_build_model_catalog_preserves_bundled_models_and_template_protocol() {
        let bundled = serde_json::json!({
            "models": [{
                "slug": "codex-template",
                "display_name": "Codex Template",
                "description": "bundled",
                "visibility": "list",
                "supported_in_api": true,
                "priority": 1,
                "base_instructions": "version-matched instructions",
                "model_messages": {"instructions_template": "version-matched template"},
                "shell_type": "shell_command",
                "apply_patch_tool_type": "freeform",
                "prefer_websockets": true,
                "support_verbosity": true,
                "default_verbosity": "medium",
                "default_reasoning_level": "high",
                "supported_reasoning_levels": [{"effort": "high", "description": "deep"}],
                "supports_reasoning_summaries": true,
                "supports_reasoning_summary_parameter": true,
                "supports_search_tool": true,
                "supports_image_detail_original": true,
                "use_responses_lite": true,
                "tool_mode": "code_mode_only",
                "multi_agent_version": "v2",
                "experimental_supported_tools": ["unknown"],
                "additional_speed_tiers": ["fast"],
                "service_tiers": [{"id": "fast"}],
                "availability_nux": {"message": "new"},
                "upgrade": {"id": "next"},
                "comp_hash": "template-only"
            }]
        });

        let catalog = build_model_catalog(
            bundled,
            &["deepseek-r1".into()],
            Some("codex-template"),
            "provider",
        )
        .unwrap();
        let models = catalog["models"].as_array().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0]["slug"], "codex-template");
        let generated = &models[1];
        assert_eq!(generated["slug"], "deepseek-r1");
        assert_eq!(
            generated["base_instructions"],
            "version-matched instructions"
        );
        assert_eq!(
            generated["model_messages"]["instructions_template"],
            "version-matched template"
        );
        assert_eq!(generated["shell_type"], "shell_command");
        assert_eq!(generated["apply_patch_tool_type"], "freeform");
        assert_eq!(generated["prefer_websockets"], false);
        assert_eq!(
            generated["supported_reasoning_levels"],
            serde_json::json!([])
        );
        assert_eq!(generated["supports_reasoning_summaries"], true);
        assert_eq!(generated["supports_reasoning_summary_parameter"], true);
        assert_eq!(generated["supports_image_detail_original"], false);
        assert_eq!(generated["context_window"], 262_144);
        assert_eq!(generated["tool_mode"], serde_json::Value::Null);
    }

    #[test]
    fn test_build_model_catalog_sanitizes_bundled_slug_collision() {
        let bundled = serde_json::json!({"models": [{
            "slug": "same-model",
            "display_name": "Bundled",
            "visibility": "list",
            "prefer_websockets": true,
            "supports_reasoning_summary_parameter": true,
            "supports_search_tool": true,
            "supports_image_detail_original": true,
            "use_responses_lite": true,
            "tool_mode": "code_mode_only",
            "multi_agent_version": "v2"
        }]});

        let catalog = build_model_catalog(
            bundled,
            &["same-model".into()],
            Some("same-model"),
            "provider",
        )
        .unwrap();
        let models = catalog["models"].as_array().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0]["display_name"], "same-model");
        assert_eq!(models[0]["prefer_websockets"], false);
        assert_eq!(models[0]["supports_reasoning_summary_parameter"], false);
        assert_eq!(models[0]["supports_search_tool"], false);
        assert_eq!(models[0]["supports_image_detail_original"], false);
        assert_eq!(models[0]["use_responses_lite"], false);
        assert_eq!(models[0]["tool_mode"], serde_json::Value::Null);
    }

    #[test]
    fn test_build_model_catalog_rejects_unknown_template() {
        let bundled = serde_json::json!({"models": [{
            "slug": "known",
            "visibility": "list"
        }]});
        let error = build_model_catalog(bundled, &["upstream".into()], Some("missing"), "provider")
            .unwrap_err();
        assert!(error.to_string().contains("was not found"));
    }
}
