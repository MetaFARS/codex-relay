use async_stream::stream;
use axum::response::{
    sse::{Event, KeepAlive},
    Sse,
};
use eventsource_stream::Eventsource as EventsourceExt;
use futures_util::StreamExt;
use serde_json::json;
use std::sync::Arc;
use tracing::{error, warn};

use crate::{
    session::SessionStore,
    types::{ChatMessage, ChatRequest, ChatStreamChunk},
};

/// Translate an upstream Chat Completions SSE stream into a Responses API SSE stream.
///
/// Events emitted:
///   response.created           — sent immediately with the pre-allocated response_id
///   response.output_text.delta — one per non-empty text chunk from upstream
///   response.completed         — final event with the full accumulated response
pub fn translate_stream(
    client: reqwest::Client,
    url: String,
    api_key: Arc<String>,
    chat_req: ChatRequest,
    response_id: String,
    sessions: SessionStore,
    prior_messages: Vec<ChatMessage>,
    model: String,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let event_stream = stream! {
        // Send response.created immediately so Codex CLI knows the response_id.
        yield Ok(Event::default()
            .event("response.created")
            .data(json!({
                "type": "response.created",
                "response": { "id": &response_id, "status": "in_progress", "model": &model }
            }).to_string()));

        let mut builder = client.post(&url).header("Content-Type", "application/json");
        if !api_key.is_empty() {
            builder = builder.bearer_auth(api_key.as_str());
        }

        let upstream = match builder.json(&chat_req).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                error!("upstream {status}: {body}");
                yield Ok(Event::default().event("error").data(
                    json!({"type": "error", "code": status.as_u16(), "message": body}).to_string()
                ));
                return;
            }
            Err(e) => {
                error!("upstream request failed: {e}");
                yield Ok(Event::default().event("error").data(
                    json!({"type": "error", "message": e.to_string()}).to_string()
                ));
                return;
            }
        };

        let mut accumulated = String::new();
        let mut source = upstream.bytes_stream().eventsource();

        while let Some(ev) = source.next().await {
            match ev {
                Err(e) => {
                    warn!("SSE parse error: {e}");
                    break;
                }
                Ok(ev) if ev.data.trim() == "[DONE]" => break,
                Ok(ev) if ev.data.is_empty() => continue,
                Ok(ev) => {
                    match serde_json::from_str::<ChatStreamChunk>(&ev.data) {
                        Err(e) => warn!("chunk parse error: {e} — data: {}", ev.data),
                        Ok(chunk) => {
                            for choice in &chunk.choices {
                                let content = choice.delta.content.as_deref().unwrap_or("");
                                if !content.is_empty() {
                                    accumulated.push_str(content);
                                    yield Ok(Event::default()
                                        .event("response.output_text.delta")
                                        .data(json!({
                                            "type": "response.output_text.delta",
                                            "delta": content
                                        }).to_string()));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Persist the complete turn to the session store.
        let mut messages = prior_messages;
        messages.push(ChatMessage {
            role: "assistant".into(),
            content: Some(accumulated.clone()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        });
        sessions.save_with_id(response_id.clone(), messages);

        yield Ok(Event::default()
            .event("response.completed")
            .data(json!({
                "type": "response.completed",
                "response": {
                    "id": &response_id,
                    "status": "completed",
                    "model": &model,
                    "output": [{
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": &accumulated}]
                    }]
                }
            }).to_string()));
    };

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}
