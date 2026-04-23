use crate::types::*;

/// Convert a Responses API request + prior history into a Chat Completions request.
pub fn to_chat_request(req: &ResponsesRequest, history: Vec<ChatMessage>) -> ChatRequest {
    let mut messages = history;

    // Inject system prompt if present and not already in history
    if let Some(system) = &req.system {
        if messages.is_empty() || messages[0].role != "system" {
            messages.insert(
                0,
                ChatMessage {
                    role: "system".into(),
                    content: Some(system.clone()),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
            );
        }
    }

    // Append new input
    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(ChatMessage {
                role: "user".into(),
                content: Some(text.clone()),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
        ResponsesInput::Messages(items) => {
            for item in items {
                messages.push(ChatMessage {
                    role: item.role.clone(),
                    content: Some(content_to_string(&item.content)),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                });
            }
        }
    }

    ChatRequest {
        model: req.model.clone(),
        messages,
        tools: req.tools.clone(),
        temperature: req.temperature,
        max_tokens: req.max_output_tokens,
        stream: req.stream,
    }
}

/// Convert a Chat Completions response into a Responses API response.
pub fn from_chat_response(
    id: String,
    model: &str,
    chat: ChatResponse,
) -> (ResponsesResponse, Vec<ChatMessage>) {
    let choice = chat.choices.into_iter().next().unwrap_or_else(|| ChatChoice {
        message: ChatMessage {
            role: "assistant".into(),
            content: Some(String::new()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        },
    });

    let text = choice.message.content.clone().unwrap_or_default();
    let usage = chat.usage.unwrap_or(ChatUsage {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
    });

    let response = ResponsesResponse {
        id,
        object: "response",
        model: model.to_string(),
        output: vec![ResponsesOutputItem {
            kind: "message".into(),
            role: "assistant".into(),
            content: vec![ContentPart {
                kind: "output_text".into(),
                text: Some(text),
            }],
        }],
        usage: ResponsesUsage {
            input_tokens: usage.prompt_tokens,
            output_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
        },
    };

    (response, vec![choice.message])
}

fn content_to_string(content: &ResponsesContent) -> String {
    match content {
        ResponsesContent::Text(s) => s.clone(),
        ResponsesContent::Parts(parts) => parts
            .iter()
            .filter_map(|p| p.text.as_deref())
            .collect::<Vec<_>>()
            .join(""),
    }
}
