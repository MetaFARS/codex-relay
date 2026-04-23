mod session;
mod stream;
mod translate;
mod types;

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
    Json, Router,
};
use clap::Parser;
use reqwest::Client;
use session::SessionStore;
use std::sync::Arc;
use tracing::{error, info};
use types::*;

#[derive(Parser, Debug)]
#[command(name = "codex-bridge", about = "Responses API ↔ Chat Completions bridge")]
struct Args {
    #[arg(long, env = "CODEX_BRIDGE_PORT", default_value = "4444")]
    port: u16,

    #[arg(
        long,
        env = "CODEX_BRIDGE_UPSTREAM",
        default_value = "https://openrouter.ai/api/v1"
    )]
    upstream: String,

    #[arg(long, env = "CODEX_BRIDGE_API_KEY", default_value = "")]
    api_key: String,
}

#[derive(Clone)]
struct AppState {
    sessions: SessionStore,
    client: Client,
    upstream: Arc<String>,
    api_key: Arc<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "codex_bridge=info".into()),
        )
        .init();

    let args = Args::parse();

    let state = AppState {
        sessions: SessionStore::new(),
        client: Client::new(),
        upstream: Arc::new(args.upstream.clone()),
        api_key: Arc::new(args.api_key.clone()),
    };

    let app = Router::new()
        .route("/v1/responses", post(handle_responses))
        .with_state(state);

    let addr = format!("127.0.0.1:{}", args.port);
    info!("codex-bridge listening on {addr} → {}", args.upstream);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn handle_responses(
    State(state): State<AppState>,
    Json(req): Json<ResponsesRequest>,
) -> Response {
    let history = req
        .previous_response_id
        .as_deref()
        .map(|id| state.sessions.get_history(id))
        .unwrap_or_default();

    let model = req.model.clone();
    let mut chat_req = translate::to_chat_request(&req, history.clone());
    let url = format!("{}/chat/completions", state.upstream);

    if req.stream {
        let response_id = state.sessions.new_id();
        chat_req.stream = true;
        stream::translate_stream(
            state.client,
            url,
            state.api_key,
            chat_req,
            response_id,
            state.sessions,
            history,
            model,
        )
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
