use async_stream::stream;
use axum::response::{
    sse::{Event, KeepAlive},
    Sse,
};
use eventsource_stream::Eventsource as EventsourceExt;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use tracing::{debug, error, warn};

use crate::{
    corpus::CorpusRecorder,
    session::SessionStore,
    translate::{
        custom_tool_input, response_function_name_for_responses, CustomToolMap, NamespaceToolMap,
    },
    types::{ChatMessage, ChatRequest, ChatStreamChunk, ChatUsage},
    upstream_request::UpstreamRequestConfig,
};

pub struct StreamArgs {
    pub client: reqwest::Client,
    pub url: String,
    pub api_key: Arc<String>,
    pub chat_req: ChatRequest,
    pub upstream_request: Arc<UpstreamRequestConfig>,
    pub response_id: String,
    pub sessions: SessionStore,
    /// The fully translated request messages (including replayed history).
    /// Used to save correct session history so turn-level reasoning can be
    /// recovered when Codex replays the conversation without previous_response_id.
    pub request_messages: Vec<ChatMessage>,
    pub namespace_tools: NamespaceToolMap,
    pub custom_tools: CustomToolMap,
    pub model: String,
    pub corpus: Option<CorpusRecorder>,
    pub previous_response_id: Option<String>,
}

struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

fn summarize_stream_tool_call_names(tool_calls: &BTreeMap<usize, ToolCallAccum>) -> String {
    if tool_calls.is_empty() {
        return "(none)".to_string();
    }

    tool_calls
        .values()
        .map(|tc| tc.name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

/// Translate an upstream Chat Completions SSE stream into a Responses API SSE stream.
///
/// Text response event sequence:
///   response.created → response.output_item.added (message) → response.output_text.delta*
///   → response.output_item.done → response.completed
///
/// Tool call response event sequence:
///   response.created → [accumulate deltas] → response.output_item.added (function_call)
///   → response.function_call_arguments.delta → response.output_item.done → response.completed
pub fn translate_stream(
    args: StreamArgs,
) -> Sse<impl futures_util::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let StreamArgs {
        client,
        url,
        api_key,
        chat_req,
        upstream_request,
        response_id,
        sessions,
        request_messages,
        namespace_tools,
        custom_tools,
        model,
        corpus,
        previous_response_id,
    } = args;
    let msg_item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let reasoning_item_id = format!("rs_{}", uuid::Uuid::new_v4().simple());

    let event_stream = stream! {
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

        let upstream_body = match upstream_request.request_body(&chat_req) {
            Ok(body) => body,
            Err(e) => {
                error!("upstream request body error: {e}");
                yield Ok(Event::default().event("response.failed").data(
                    json!({"type": "response.failed", "response": {"id": &response_id, "status": "failed", "error": {"code": "request_body_error", "message": e.to_string()}}}).to_string()
                ));
                return;
            }
        };

        let upstream = match builder.json(&upstream_body).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                let status = r.status();
                let body = r.text().await.unwrap_or_default();
                error!("upstream {status}: {body}");
                yield Ok(Event::default().event("response.failed").data(
                    json!({"type": "response.failed", "response": {"id": &response_id, "status": "failed", "error": {"code": status.as_u16().to_string(), "message": body}}}).to_string()
                ));
                return;
            }
            Err(e) => {
                error!("upstream request failed: {e}");
                yield Ok(Event::default().event("response.failed").data(
                    json!({"type": "response.failed", "response": {"id": &response_id, "status": "failed", "error": {"code": "connection_error", "message": e.to_string()}}}).to_string()
                ));
                return;
            }
        };

        let mut accumulated_text = String::new();
        let mut accumulated_reasoning = String::new();
        let mut reasoning_chunks: usize = 0;
        let mut tool_calls: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();
        let mut reasoning_output_index: Option<usize> = None;
        let mut message_output_index: Option<usize> = None;
        let mut next_output_index: usize = 0;
        let mut stream_done = false;
        let mut stream_err = false;
        let mut stream_usage: Option<ChatUsage> = None;
        let mut source = upstream.bytes_stream().eventsource();

        while let Some(ev) = source.next().await {
            match ev {
                Err(e) => {
                    warn!("SSE parse error: {e}");
                    stream_err = true;
                    break;
                }
                Ok(ev) if ev.data.trim() == "[DONE]" => {
                    stream_done = true;
                    break;
                }
                Ok(ev) if ev.data.is_empty() => continue,
                Ok(ev) => {
                    match serde_json::from_str::<ChatStreamChunk>(&ev.data) {
                        Err(e) => warn!("chunk parse error: {e} — data: {}", ev.data),
                        Ok(chunk) => {
                            let ChatStreamChunk { choices, usage } = chunk;
                            if usage.is_some() {
                                stream_usage = usage;
                            }
                            for choice in &choices {
                                // Reasoning/thinking content (kimi-k2.6, GLM, etc.).
                                // Field name varies by provider (reasoning_content
                                // vs reasoning) — normalized via reasoning_text().
                                if let Some(rc) = choice.delta.reasoning_text() {
                                    if !rc.is_empty() {
                                        reasoning_chunks += 1;
                                        let output_index = match reasoning_output_index {
                                            Some(idx) => idx,
                                            None => {
                                                let idx = next_output_index;
                                                next_output_index += 1;
                                                reasoning_output_index = Some(idx);
                                                yield Ok(Event::default()
                                                    .event("response.output_item.added")
                                                    .data(json!({
                                                        "type": "response.output_item.added",
                                                        "output_index": idx,
                                                        "item": {
                                                            "type": "reasoning",
                                                            "id": &reasoning_item_id,
                                                            "summary": [{"type": "summary_text", "text": ""}]
                                                        }
                                                    }).to_string()));
                                                idx
                                            }
                                        };
                                        accumulated_reasoning.push_str(rc);
                                        yield Ok(Event::default()
                                            .event("response.reasoning_summary_text.delta")
                                            .data(json!({
                                                "type": "response.reasoning_summary_text.delta",
                                                "item_id": &reasoning_item_id,
                                                "output_index": output_index,
                                                "summary_index": 0,
                                                "delta": rc
                                            }).to_string()));
                                    }
                                }

                                // Text content
                                let content = choice.delta.content.as_deref().unwrap_or("");
                                if !content.is_empty() {
                                    let output_index = match message_output_index {
                                        Some(idx) => idx,
                                        None => {
                                            let idx = next_output_index;
                                            next_output_index += 1;
                                            message_output_index = Some(idx);
                                            yield Ok(Event::default()
                                                .event("response.output_item.added")
                                                .data(json!({
                                                    "type": "response.output_item.added",
                                                    "output_index": idx,
                                                    "item": {
                                                        "type": "message",
                                                        "id": &msg_item_id,
                                                        "role": "assistant",
                                                        "status": "in_progress",
                                                        "content": []
                                                    }
                                                }).to_string()));
                                            idx
                                        }
                                    };
                                    accumulated_text.push_str(content);
                                    yield Ok(Event::default()
                                        .event("response.output_text.delta")
                                        .data(json!({
                                            "type": "response.output_text.delta",
                                            "item_id": &msg_item_id,
                                            "output_index": output_index,
                                            "delta": content
                                        }).to_string()));
                                }

                                // Tool call deltas
                                if let Some(tcs) = &choice.delta.tool_calls {
                                    for tc in tcs {
                                        let entry = tool_calls.entry(tc.index).or_insert_with(|| ToolCallAccum {
                                            id: String::new(),
                                            name: String::new(),
                                            arguments: String::new(),
                                        });
                                        if let Some(id) = &tc.id {
                                            if !id.is_empty() {
                                                entry.id = id.clone();
                                            }
                                        }
                                        if let Some(f) = &tc.function {
                                            if let Some(n) = &f.name {
                                                if !n.is_empty() {
                                                    entry.name.push_str(n);
                                                }
                                            }
                                            if let Some(a) = &f.arguments {
                                                entry.arguments.push_str(a);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if let Some(output_index) = reasoning_output_index {
            yield Ok(Event::default()
                .event("response.output_item.done")
                .data(json!({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": {
                        "type": "reasoning",
                        "id": &reasoning_item_id,
                        "summary": [{"type": "summary_text", "text": &accumulated_reasoning}]
                    }
                }).to_string()));
        }

        if let Some(output_index) = message_output_index {
            yield Ok(Event::default()
                .event("response.output_item.done")
                .data(json!({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": {
                        "type": "message",
                        "id": &msg_item_id,
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": &accumulated_text}]
                    }
                }).to_string()));
        }

        // Emit function_call items for each accumulated tool call
        let base_index = next_output_index;
        let mut fc_items: Vec<(usize, Value)> = Vec::new();
        debug!(
            "← upstream stream function_calls={}",
            summarize_stream_tool_call_names(&tool_calls)
        );
        // Counts only (never the reasoning text) so issue #26 can be diagnosed:
        // distinguishes "upstream sent no reasoning" from "received but not translated".
        debug!(
            "← upstream stream reasoning chunks={} bytes={}",
            reasoning_chunks,
            accumulated_reasoning.len()
        );

        for (rel_idx, (_, tc)) in tool_calls.iter().enumerate() {
            let output_index = base_index + rel_idx;
            let (namespace, name) = response_function_name_for_responses(&tc.name, &namespace_tools);
            let custom = custom_tools.get(&tc.name);
            let fc_item_id = if custom.is_some() {
                format!("ctc_{}", uuid::Uuid::new_v4().simple())
            } else {
                format!("fc_{}", uuid::Uuid::new_v4().simple())
            };
            let custom_input = custom
                .map(|tool| custom_tool_input(&tc.arguments, &tool.argument_field));
            let (added_item, done_item) = if let (Some(custom), Some(input)) = (custom, custom_input.as_deref()) {
                (
                    json!({
                        "type": "custom_tool_call",
                        "id": &fc_item_id,
                        "call_id": &tc.id,
                        "name": &custom.name,
                        "input": "",
                        "status": "in_progress"
                    }),
                    json!({
                        "type": "custom_tool_call",
                        "id": &fc_item_id,
                        "call_id": &tc.id,
                        "name": &custom.name,
                        "input": input,
                        "status": "completed"
                    }),
                )
            } else {
                let mut added = json!({
                    "type": "function_call",
                    "id": &fc_item_id,
                    "call_id": &tc.id,
                    "name": &name,
                    "arguments": "",
                    "status": "in_progress"
                });
                let mut done = json!({
                    "type": "function_call",
                    "id": &fc_item_id,
                    "call_id": &tc.id,
                    "name": &name,
                    "arguments": &tc.arguments,
                    "status": "completed"
                });
                if let Some(namespace) = namespace {
                    added["namespace"] = Value::String(namespace.clone());
                    done["namespace"] = Value::String(namespace);
                }
                (added, done)
            };

            yield Ok(Event::default()
                .event("response.output_item.added")
                .data(json!({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": added_item
                }).to_string()));

            if let Some(input) = custom_input.as_deref() {
                if !input.is_empty() {
                    yield Ok(Event::default()
                        .event("response.custom_tool_call_input.delta")
                        .data(json!({
                            "type": "response.custom_tool_call_input.delta",
                            "item_id": &fc_item_id,
                            "output_index": output_index,
                            "delta": input
                        }).to_string()));
                }
            } else if !tc.arguments.is_empty() {
                yield Ok(Event::default()
                    .event("response.function_call_arguments.delta")
                    .data(json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": &fc_item_id,
                        "output_index": output_index,
                        "delta": &tc.arguments
                    }).to_string()));
            }

            yield Ok(Event::default()
                .event("response.output_item.done")
                .data(json!({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": done_item
                }).to_string()));

            fc_items.push((output_index, done_item));
        }

        // Some OpenAI-compatible providers (e.g. synthetic.new) close the SSE
        // stream cleanly without emitting a terminating `[DONE]` line. If the
        // stream ended without an error and we already received a full turn
        // (text and/or tool calls), treat it as complete so the response is
        // persisted and `response.completed` is emitted. A mid-stream error
        // (stream_err) still discards the partial turn.
        if !stream_done && !stream_err && (!accumulated_text.is_empty() || !tool_calls.is_empty()) {
            warn!("stream ended without [DONE] but content was received — treating as complete");
            stream_done = true;
        }

        if stream_done {
            // Persist turn to session store
            // Store reasoning_content per call_id so translate.rs can inject it
            // back when Codex replays function_call items in the next request.
            for tc in tool_calls.values() {
                if !tc.id.is_empty() {
                    sessions.store_reasoning(tc.id.clone(), accumulated_reasoning.clone());
                }
            }

            let assistant_tool_calls: Option<Vec<Value>> = if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.values().map(|tc| json!({
                    "id": &tc.id,
                    "type": "function",
                    "function": { "name": &tc.name, "arguments": &tc.arguments }
                })).collect())
            };
            let assistant_msg = ChatMessage {
                role: "assistant".into(),
                content: if accumulated_text.is_empty() { None } else { Some(serde_json::Value::String(accumulated_text.clone())) },
                reasoning_content: if accumulated_reasoning.is_empty() { None } else { Some(accumulated_reasoning.clone()) },
                tool_calls: assistant_tool_calls,
                tool_call_id: None,
                name: None,
            };

            // Index reasoning by turn fingerprint so it can be recovered when
            // Codex replays the full conversation in input[] without previous_response_id.
            if !accumulated_reasoning.is_empty() {
                sessions.store_turn_reasoning(&request_messages, &assistant_msg, accumulated_reasoning.clone());
            }

            // Save the full request conversation (including current input items)
            // so that history is complete for the next turn.
            let mut messages = request_messages;
            messages.push(assistant_msg);
            if let Some(corpus) = &corpus {
                corpus.record_turn(
                    previous_response_id.as_deref(),
                    &response_id,
                    &model,
                    &messages,
                );
            }
            sessions.save_with_id(response_id.clone(), messages);

            // Build output array for response.completed
            let mut indexed_output_items: Vec<(usize, Value)> = Vec::new();
            if let Some(output_index) = reasoning_output_index {
                indexed_output_items.push((output_index, json!({
                    "type": "reasoning",
                    "id": &reasoning_item_id,
                    "summary": [{"type": "summary_text", "text": &accumulated_reasoning}]
                })));
            }
            if let Some(output_index) = message_output_index {
                indexed_output_items.push((output_index, json!({
                    "type": "message",
                    "id": &msg_item_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": &accumulated_text}]
                })));
            }
            indexed_output_items.extend(fc_items);
            indexed_output_items.sort_by_key(|(idx, _)| *idx);
            let output_items: Vec<Value> = indexed_output_items
                .into_iter()
                .map(|(_, item)| item)
                .collect();
            let usage = stream_usage.unwrap_or_default();
            debug!("cache(stream): {}", usage.cache_summary());

            yield Ok(Event::default()
                .event("response.completed")
                .data(json!({
                    "type": "response.completed",
                    "response": {
                        "id": &response_id,
                        "status": "completed",
                        "model": &model,
                        "output": output_items,
                        "usage": {
                            "input_tokens": usage.prompt_tokens,
                            "output_tokens": usage.completion_tokens,
                            "total_tokens": usage.total_tokens,
                            "input_tokens_details": {
                                "cached_tokens": usage.cache_hit()
                            }
                        }
                    }
                }).to_string()));
        } else {
            // Stream did not complete cleanly: do NOT save session state
            // to avoid creating an assistant-with-tool_calls gap in history
            // that causes upstream "insufficient tool messages" errors.
            warn!("stream disconnected before [DONE] — discarding partial turn");
            yield Ok(Event::default()
                .event("response.failed")
                .data(json!({
                    "type": "response.failed",
                    "response": {
                        "id": &response_id,
                        "status": "failed",
                        "error": {
                            "code": "stream_incomplete",
                            "message": "stream disconnected before completion"
                        }
                    }
                }).to_string()));
        }
    };

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}
