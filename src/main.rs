mod session;
mod stream;
mod translate;
mod types;

use anyhow::{bail, Result};
use axum::{
    extract::{Request, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use reqwest::{Client, Url};
use session::SessionStore;
use std::sync::Arc;
use tracing::{debug, error, info, warn};
use types::*;

#[derive(Parser, Debug)]
#[command(name = "codex-relay", about = "Responses API ↔ Chat Completions bridge")]
struct Args {
    #[arg(long, env = "CODEX_RELAY_PORT", default_value = "4444")]
    port: u16,

    #[arg(
        long,
        env = "CODEX_RELAY_UPSTREAM",
        default_value = "https://openrouter.ai/api/v1"
    )]
    upstream: String,

    #[arg(long, env = "CODEX_RELAY_API_KEY", default_value = "")]
    api_key: String,
}

#[derive(Clone)]
struct AppState {
    sessions: SessionStore,
    client: Client,
    upstream: Arc<Url>,
    api_key: Arc<String>,
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

    let state = AppState {
        sessions: SessionStore::new(),
        client: Client::new(),
        upstream: Arc::new(upstream),
        api_key: Arc::new(args.api_key),
    };

    let app = Router::new()
        .route("/v1/responses", post(handle_responses))
        .route("/v1/models", get(handle_models))
        .fallback(handle_fallback)
        .with_state(state.clone());

    let addr = format!("127.0.0.1:{}", args.port);
    info!("codex-relay listening on {addr} → {}", state.upstream.as_ref());

    let listener = tokio::net::TcpListener::bind(&addr).await?;
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

fn join_base(url: &Url) -> String {
    let s = url.as_str();
    if s.ends_with('/') { s.to_string() } else { format!("{s}/") }
}

/// GET /v1/models — proxy to upstream so Codex gets real model metadata.
async fn handle_models(State(state): State<AppState>) -> Response {
    info!("GET /v1/models");
    let url = format!("{}models", join_base(&state.upstream));
    let mut builder = state.client.get(&url);
    if !state.api_key.is_empty() {
        builder = builder.bearer_auth(state.api_key.as_str());
    }
    match builder.send().await {
        Ok(r) if r.status().is_success() => {
            match r.json::<serde_json::Value>().await {
                Ok(body) => Json(body).into_response(),
                Err(e) => {
                    warn!("upstream models: parse error: {e}");
                    Json(serde_json::json!({ "object": "list", "data": [] })).into_response()
                }
            }
        }
        Ok(r) => {
            warn!("upstream models: status {}", r.status());
            Json(serde_json::json!({ "object": "list", "data": [] })).into_response()
        }
        Err(e) => {
            warn!("upstream models: request error: {e}");
            Json(serde_json::json!({ "object": "list", "data": [] })).into_response()
        }
    }
}

/// Catch-all: log unknown requests so we can see what Codex is sending.
async fn handle_fallback(req: Request) -> Response {
    warn!("unhandled {} {}", req.method(), req.uri().path());
    (StatusCode::NOT_FOUND, "not found").into_response()
}

async fn handle_responses(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Response {
    let req: ResponsesRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            error!("JSON parse error: {e}");
            error!("body prefix: {}", String::from_utf8_lossy(&body[..body.len().min(200)]));
            return (StatusCode::UNPROCESSABLE_ENTITY, e.to_string()).into_response();
        }
    };
    debug!("→ model={} stream={} input_items={} tools={} prev_resp={:?}",
        req.model, req.stream,
        match &req.input { crate::types::ResponsesInput::Messages(v) => v.len(), _ => 1 },
        req.tools.len(), req.previous_response_id);
    handle_responses_inner(state, req).await
}

async fn handle_responses_inner(
    state: AppState,
    req: ResponsesRequest,
) -> Response {
    let history = req
        .previous_response_id
        .as_deref()
        .map(|id| state.sessions.get_history(id))
        .unwrap_or_default();

    let model = req.model.clone();
    let mut chat_req = translate::to_chat_request(&req, history.clone(), &state.sessions);
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
            response_id,
            sessions: state.sessions,
            prior_messages: history,
            request_messages,
            model,
        })
        .into_response()
    } else {
        chat_req.stream = false;
        handle_blocking(state, chat_req, url, model).await
    }
}

async fn handle_blocking(
    state: AppState,
    chat_req: types::ChatRequest,
    url: String,
    model: String,
) -> Response {
    let mut builder = state
        .client
        .post(&url)
        .header("Content-Type", "application/json");

    if !state.api_key.is_empty() {
        builder = builder.bearer_auth(state.api_key.as_str());
    }

    match builder.json(&chat_req).send().await {
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
                let assistant_msg = chat_resp
                    .choices
                    .first()
                    .map(|c| c.message.clone())
                    .unwrap_or_else(|| ChatMessage {
                        role: "assistant".into(),
                        content: Some(String::new()),
                        reasoning_content: None,
                        tool_calls: None,
                        tool_call_id: None,
                        name: None,
                    });

                let mut full_history = chat_req.messages.clone();
                full_history.push(assistant_msg);
                let response_id = state.sessions.save(full_history);

                let (resp, _) = translate::from_chat_response(response_id, &model, chat_resp);
                Json(resp).into_response()
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
