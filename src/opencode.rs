//! Opt-in native Responses bridge. Tool identities are request-scoped: names
//! sent to the provider must never become authoritative client tool names.

use anyhow::{bail, Context, Result};
use axum::response::{
    sse::{Event, KeepAlive},
    IntoResponse, Response, Sse,
};
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use crate::translate::{
    apply_patch_chat_description, custom_argument_field, declared_function_name,
    function_tool_string_field, tool_denylist_from_env, uses_plaintext_collaboration_args,
};

#[derive(Clone, Debug, PartialEq, Eq)]
struct Tool {
    name: String,
    namespace: Option<String>,
    custom_field: Option<String>,
}

impl Tool {
    fn upstream_name(&self) -> String {
        let name = self
            .namespace
            .as_ref()
            .map_or_else(|| self.name.clone(), |ns| format!("{ns}-{}", self.name));
        if name.len() <= 64 {
            return name;
        }
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        name.hash(&mut hasher);
        let suffix = format!("-{:016x}", hasher.finish());
        let mut end = 64 - suffix.len();
        while !name.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}{suffix}", &name[..end])
    }
}

#[derive(Clone, Debug, Default)]
pub struct ToolMap(HashMap<String, Tool>, HashSet<String>);

impl ToolMap {
    fn insert(&mut self, tool: Tool) -> Result<String> {
        let name = tool.upstream_name();
        if self.0.contains_key(&name) {
            bail!("ambiguous upstream tool name");
        }
        // Codex routes custom tools by bare name, including namespace children.
        if tool.custom_field.is_some()
            && self
                .0
                .values()
                .any(|other| other.custom_field.is_some() && other.name == tool.name)
        {
            bail!("ambiguous custom tool name");
        }
        self.0.insert(name.clone(), tool);
        Ok(name)
    }

    fn resolve(&self, item: &Value) -> Result<Tool> {
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .context("tool item needs a name")?;
        let namespace = item
            .get("namespace")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        if let Some(tool) = self
            .0
            .values()
            .find(|tool| tool.name == name && tool.namespace.as_deref() == namespace)
        {
            return Ok(tool.clone());
        }
        // Historical custom calls use bare names even for namespace children.
        if namespace.is_none()
            && item.get("type").and_then(Value::as_str) == Some("custom_tool_call")
        {
            if let Some(tool) = self
                .0
                .values()
                .find(|tool| tool.name == name && tool.custom_field.is_some())
            {
                return Ok(tool.clone());
            }
        }
        // History may contain tools no longer declared in this request. Encode
        // their identity for replay, but do not authorize them in responses.
        Ok(Tool {
            name: name.into(),
            namespace: namespace.map(str::to_owned),
            custom_field: (item["type"] == "custom_tool_call")
                .then(|| custom_argument_field(name).into()),
        })
    }

    fn restore_item(&self, item: &mut Value, complete: bool) -> Result<()> {
        if item["type"] == "custom_tool_call" {
            bail!("upstream returned a custom call without a translated function identity");
        }
        if item["type"] != "function_call" {
            if let Some(kind) = item["type"].as_str() {
                if kind.ends_with("_call") && !self.1.contains(kind) {
                    bail!("upstream returned an undeclared native tool call");
                }
            }
            return Ok(());
        }
        let tool = self
            .0
            .get(item["name"].as_str().unwrap_or(""))
            .context("upstream returned an undeclared tool")?;
        let object = item.as_object_mut().context("invalid tool call")?;
        object.insert("name".into(), json!(tool.name));
        object.remove("namespace");
        if let Some(field) = &tool.custom_field {
            let input = if complete {
                custom_input(
                    object
                        .get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                    field,
                )?
            } else {
                String::new()
            };
            object.insert("type".into(), json!("custom_tool_call"));
            object.remove("arguments");
            object.remove("encrypted_function_args");
            object.insert("input".into(), json!(input));
        } else if let Some(namespace) = &tool.namespace {
            object.insert("namespace".into(), json!(namespace));
            if uses_plaintext_collaboration_args(Some(namespace), &tool.name) {
                object.insert("encrypted_function_args".into(), json!([]));
            }
        }
        Ok(())
    }

    fn restore_response(&self, response: &mut Value, model: &str) -> Result<()> {
        if !response.is_object() {
            bail!("upstream returned an invalid Responses object");
        }
        if let Some(output) = response.get_mut("output").and_then(Value::as_array_mut) {
            let mut calls = HashSet::new();
            for item in output {
                if item["type"] == "function_call" {
                    let id = item["call_id"]
                        .as_str()
                        .filter(|id| !id.is_empty())
                        .context("upstream returned an empty tool call ID")?;
                    if !calls.insert(id.to_owned()) {
                        bail!("upstream returned duplicate tool call IDs");
                    }
                }
                self.restore_item(item, true)?;
            }
        }
        response["model"] = json!(model);
        Ok(())
    }
}

fn custom_input(arguments: &str, field: &str) -> Result<String> {
    if arguments.trim().is_empty() {
        return Ok(String::new());
    }
    let value: Value =
        serde_json::from_str(arguments).context("invalid custom tool arguments JSON")?;
    value[field]
        .as_str()
        .map(str::to_owned)
        .context("custom tool arguments need their declared string field")
}

#[derive(Debug, Default)]
pub struct Changes {
    pub orphan_outputs: usize,
    pub schema_refs: usize,
    pub tool_fields: usize,
}

pub struct PreparedRequest {
    pub payload: Value,
    pub tools: ToolMap,
    pub changes: Changes,
}

impl PreparedRequest {
    /// Overrides are applied before normalization. Only tools requested by the
    /// client AND actually forwarded can authorize a returned function call.
    pub fn new(mut payload: Value, client_tools: &[Value]) -> Result<Self> {
        let denied = tool_denylist_from_env();
        let mut changes = Changes::default();
        let (_, requested) = normalize_tools(client_tools, &denied, &mut Changes::default())?;
        let raw_tools = payload
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let (tools, mut registry) = normalize_tools(&raw_tools, &denied, &mut changes)?;
        if payload.get("tools").is_some() {
            payload["tools"] = json!(tools);
        }
        if let Some(items) = payload.get_mut("input").and_then(Value::as_array_mut) {
            for item in items {
                normalize_item(item, &registry, &mut changes)?;
            }
        }
        if let Some(choice) = payload.get_mut("tool_choice") {
            normalize_choice(choice, &registry)?;
        }
        if let Some(allowed) = payload
            .get_mut("allowed_tools")
            .and_then(Value::as_array_mut)
        {
            for choice in allowed {
                normalize_choice(choice, &registry)?;
            }
        }
        if let Some(reasoning) = payload.get_mut("reasoning").and_then(Value::as_object_mut) {
            if reasoning.get("effort").and_then(Value::as_str) == Some("none") {
                reasoning.remove("effort");
                changes.tool_fields += 1;
            }
        }
        // An override cannot grant the provider access to an undeclared tool.
        registry
            .0
            .retain(|name, tool| requested.0.get(name) == Some(tool));
        registry.1.retain(|kind| requested.1.contains(kind));
        Ok(Self {
            payload,
            tools: registry,
            changes,
        })
    }
}

fn normalize_tools(
    tools: &[Value],
    denied: &HashSet<String>,
    changes: &mut Changes,
) -> Result<(Vec<Value>, ToolMap)> {
    let mut out = Vec::new();
    let mut registry = ToolMap::default();
    for tool in tools {
        if tool["type"] == "namespace" {
            let namespace = tool["name"].as_str().context("namespace needs a name")?;
            for child in tool["tools"].as_array().context("namespace needs tools")? {
                normalize_tool(
                    child,
                    Some(namespace),
                    denied,
                    &mut out,
                    &mut registry,
                    changes,
                )?;
            }
        } else {
            normalize_tool(tool, None, denied, &mut out, &mut registry, changes)?;
        }
    }
    Ok((out, registry))
}

fn normalize_tool(
    tool: &Value,
    namespace: Option<&str>,
    denied: &HashSet<String>,
    out: &mut Vec<Value>,
    registry: &mut ToolMap,
    changes: &mut Changes,
) -> Result<()> {
    match tool["type"].as_str().unwrap_or("") {
        "function" | "custom" => {
            let name = declared_function_name(tool).context("tool needs a name")?;
            let function = tool.get("function").unwrap_or(tool);
            let custom = tool["type"] == "custom" || name == "apply_patch";
            let identity = Tool {
                name: name.into(),
                namespace: namespace.map(str::to_owned),
                custom_field: custom.then(|| {
                    function_tool_string_field(function)
                        .unwrap_or_else(|| custom_argument_field(name).into())
                }),
            };
            let full_name = namespace.map_or_else(|| name.to_owned(), |ns| format!("{ns}-{name}"));
            if denied.contains(name)
                || denied.contains(&full_name)
                || denied.contains(&identity.upstream_name())
            {
                return Ok(());
            }
            let upstream_name = registry.insert(identity.clone())?;
            let mut normalized = function.clone();
            let object = normalized
                .as_object_mut()
                .context("invalid function tool")?;
            object.insert("type".into(), json!("function"));
            object.insert("name".into(), json!(upstream_name));
            for field in ["format", "input_format", "search_content_types"] {
                changes.tool_fields += usize::from(object.remove(field).is_some());
            }
            if tool["type"] == "custom" {
                let field = identity.custom_field.as_ref().unwrap();
                object.insert("parameters".into(), json!({"type":"object", "properties":{field:{"type":"string"}}, "required":[field], "additionalProperties":false}));
            }
            if let Some(field) = &identity.custom_field {
                // The upstream interface is JSON even if Codex advertises a
                // freeform tool. Never retain contradictory FREEFORM guidance.
                object.insert("description".into(), json!(if name == "apply_patch" {
                    apply_patch_chat_description(field, function.get("description").and_then(Value::as_str))
                } else { format!("Call the {name} tool with a JSON object whose {field:?} string contains the complete tool input. The following instructions describe the contents of that string, not the function-call wire format:\n{}", function.get("description").and_then(Value::as_str).unwrap_or("Provide the tool input.")) }));
            }
            if let Some(schema) = object.get_mut("parameters") {
                changes.schema_refs += strip_schema_refs(schema);
            }
            out.push(normalized);
        }
        "web_search_preview" => {
            registry.1.insert("web_search_call".into());
            out.push(tool.clone());
        }
        // Keep the original compatibility policy for unsupported hosted tools.
        _ => changes.tool_fields += 1,
    }
    Ok(())
}

fn normalize_item(item: &mut Value, tools: &ToolMap, changes: &mut Changes) -> Result<()> {
    let kind = item["type"].as_str().unwrap_or("").to_owned();
    if kind == "custom_tool_call_output" {
        item["type"] = json!("function_call_output");
    }
    if item["type"] == "function_call_output" && item["call_id"].as_str().is_none_or(str::is_empty)
    {
        // Tool results are untrusted data, never developer/system instructions.
        let output = item.get("output").cloned().unwrap_or(Value::Null);
        *item = json!({"type":"message", "role":"user", "content": format!("Untrusted tool output (no call_id):\n{}", json!({"name":item.get("name"), "namespace":item.get("namespace"), "output":output}))});
        changes.orphan_outputs += 1;
    } else if kind == "function_call" || kind == "custom_tool_call" {
        let identity = tools.resolve(item)?;
        let name = identity.upstream_name();
        let object = item.as_object_mut().context("invalid history tool call")?;
        if kind == "custom_tool_call" {
            let field = identity
                .custom_field
                .unwrap_or_else(|| custom_argument_field(&identity.name).into());
            let input = object.remove("input").unwrap_or(json!(""));
            if !input.is_string() {
                bail!("custom tool input must be a string");
            }
            object.insert("arguments".into(), json!(json!({field:input}).to_string()));
            object.insert("type".into(), json!("function_call"));
        }
        object.remove("namespace");
        object.remove("encrypted_function_args");
        object.insert("name".into(), json!(name));
    }
    Ok(())
}

fn normalize_choice(choice: &mut Value, tools: &ToolMap) -> Result<()> {
    if !choice.is_object() {
        return Ok(());
    }
    if choice["type"] == "allowed_tools" {
        for child in choice["tools"]
            .as_array_mut()
            .context("allowed_tools needs a tools array")?
        {
            normalize_choice(child, tools)?;
        }
        return Ok(());
    }
    if choice["type"] == "function" || choice["type"] == "custom" {
        // Resolve against declarations only; a forced choice cannot target a
        // dropped or unknown tool, including tools removed by the denylist.
        let mut lookup = choice.clone();
        if lookup["type"] == "custom" {
            lookup["type"] = json!("custom_tool_call");
        }
        let tool = tools.resolve(&lookup)?;
        let name = tool.upstream_name();
        if !tools.0.contains_key(&name) {
            bail!("tool_choice targets an unavailable tool");
        }
        *choice = json!({"type":"function", "name":name});
    } else if choice["type"] != "web_search_preview" {
        bail!("tool_choice targets an unsupported tool type");
    }
    Ok(())
}

/// Walk schema keywords rather than arbitrary JSON: `$ref` is also a valid
/// property name, and examples/defaults must not be rewritten as schemas.
fn strip_schema_refs(schema: &mut Value) -> usize {
    let Some(object) = schema.as_object_mut() else {
        return 0;
    };
    let mut count = usize::from(object.remove("$ref").is_some());
    for (key, value) in object {
        match key.as_str() {
            "properties" | "patternProperties" | "$defs" | "definitions" | "dependentSchemas" => {
                if let Some(map) = value.as_object_mut() {
                    count += map.values_mut().map(strip_schema_refs).sum::<usize>();
                }
            }
            "allOf" | "anyOf" | "oneOf" | "prefixItems" => {
                if let Some(items) = value.as_array_mut() {
                    count += items.iter_mut().map(strip_schema_refs).sum::<usize>();
                }
            }
            "items" if value.is_array() => {
                count += value
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .map(strip_schema_refs)
                    .sum::<usize>();
            }
            "items"
            | "additionalProperties"
            | "unevaluatedProperties"
            | "unevaluatedItems"
            | "contains"
            | "propertyNames"
            | "not"
            | "if"
            | "then"
            | "else" => count += strip_schema_refs(value),
            _ => {}
        }
    }
    count
}

#[derive(Clone)]
pub struct SessionRegistry {
    state: Arc<Mutex<VecDeque<(String, String, Instant)>>>,
    capacity: usize,
    ttl: Duration,
}

impl SessionRegistry {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(VecDeque::new())),
            capacity: capacity.max(1),
            ttl,
        }
    }

    pub fn session_for_request(&self, previous: Option<&str>) -> String {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.retain(|(_, _, created)| created.elapsed() < self.ttl);
        if let Some((_, session, _)) = state
            .iter()
            .find(|(id, _, _)| Some(id.as_str()) == previous)
        {
            return session.clone();
        }
        // Identical prompts can belong to unrelated conversations. Only an
        // explicit header or known previous_response_id establishes identity.
        format!("codex-relay-{}", uuid::Uuid::new_v4().simple())
    }

    pub fn bind(&self, response: &Value, session: &str) {
        let Some(id) = response["id"]
            .as_str()
            .filter(|id| !id.is_empty() && id.len() <= 512)
        else {
            return;
        };
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.retain(|(existing, _, created)| existing != id && created.elapsed() < self.ttl);
        state.push_back((id.to_owned(), session.to_owned(), Instant::now()));
        while state.len() > self.capacity {
            state.pop_front();
        }
    }
}

struct PendingCustom {
    field: String,
    arguments: String,
    emitted: bool,
}

struct StreamTranslator {
    tools: ToolMap,
    custom: HashMap<String, PendingCustom>,
    calls: HashMap<String, (String, String)>,
    completed_items: HashSet<String>,
    response_id: Option<String>,
    model: String,
    sequence: u64,
    terminal: bool,
}

impl StreamTranslator {
    fn new(tools: ToolMap, model: String) -> Self {
        Self {
            tools,
            custom: HashMap::new(),
            calls: HashMap::new(),
            completed_items: HashSet::new(),
            response_id: Some(format!("resp_relay_{}", uuid::Uuid::new_v4().simple())),
            model,
            sequence: 0,
            terminal: false,
        }
    }

    fn custom_events(
        &mut self,
        item_id: &str,
        arguments: Option<&str>,
        base: &Value,
    ) -> Result<Vec<Value>> {
        let pending = self
            .custom
            .get_mut(item_id)
            .context("custom arguments without an added item")?;
        if pending.emitted {
            if let Some(arguments) = arguments {
                if custom_input(arguments, &pending.field)?
                    != custom_input(&pending.arguments, &pending.field)?
                {
                    bail!("upstream changed completed custom tool input");
                }
            }
            return Ok(Vec::new());
        }
        if let Some(arguments) = arguments {
            pending.arguments = arguments.into();
        }
        let input = custom_input(&pending.arguments, &pending.field)?;
        pending.emitted = true;
        let mut delta = base.clone();
        let object = delta.as_object_mut().context("invalid tool event")?;
        object.remove("arguments");
        object.remove("item");
        object.insert("item_id".into(), json!(item_id));
        object.insert(
            "type".into(),
            json!("response.custom_tool_call_input.delta"),
        );
        object.insert("delta".into(), json!(input));
        let mut done = delta.clone();
        done.as_object_mut().unwrap().remove("delta");
        done["type"] = json!("response.custom_tool_call_input.done");
        done["input"] = json!(input);
        Ok(vec![delta, done])
    }

    fn transform(&mut self, mut event: Value) -> Result<Vec<Value>> {
        let kind = event["type"]
            .as_str()
            .context("upstream SSE event needs a type")?
            .to_owned();
        let mut prefix = Vec::new();
        if kind == "response.completed" {
            let response = event
                .get("response")
                .context("completed event needs a response")?;
            if !response["id"].is_string() || !response["output"].is_array() {
                bail!("upstream sent an invalid completed Responses object");
            }
            for item in response["output"].as_array().unwrap() {
                if item["type"] != "function_call" {
                    continue;
                }
                let item_id = item["id"].as_str().unwrap_or("");
                if let Some((call_id, name)) = self.calls.get(item_id) {
                    if item["call_id"].as_str() != Some(call_id.as_str())
                        || item["name"].as_str() != Some(name.as_str())
                    {
                        bail!("upstream changed a completed tool identity");
                    }
                }
                if let Some(pending) = self.custom.get(item_id).filter(|pending| pending.emitted) {
                    if custom_input(item["arguments"].as_str().unwrap_or(""), &pending.field)?
                        != custom_input(&pending.arguments, &pending.field)?
                    {
                        bail!("upstream changed completed custom tool input");
                    }
                }
            }
        }
        if let Some(response) = event.get_mut("response") {
            if let Some(id) = response["id"].as_str() {
                self.response_id = Some(id.into());
            }
            self.tools.restore_response(response, &self.model)?;
        }
        match kind.as_str() {
            "response.output_item.added" | "response.output_item.done"
                if event["item"]["type"] == "function_call" =>
            {
                let item_id = event["item"]["id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .context("tool item needs an ID")?
                    .to_owned();
                let call_id = event["item"]["call_id"]
                    .as_str()
                    .filter(|s| !s.is_empty())
                    .context("tool item needs a call_id")?
                    .to_owned();
                if self
                    .calls
                    .iter()
                    .any(|(id, (call, _))| id != &item_id && call == &call_id)
                {
                    bail!("upstream returned duplicate tool call IDs");
                }
                let name = event["item"]["name"]
                    .as_str()
                    .context("tool item needs a name")?
                    .to_owned();
                let identity = (call_id, name.clone());
                if self
                    .calls
                    .get(&item_id)
                    .is_some_and(|existing| existing != &identity)
                {
                    bail!("upstream changed a tool call identity");
                }
                if self.completed_items.contains(&item_id)
                    || (kind.ends_with(".added") && self.calls.contains_key(&item_id))
                {
                    bail!("upstream repeated a tool item");
                }
                self.calls.insert(item_id.clone(), identity);
                if kind.ends_with(".done") {
                    self.completed_items.insert(item_id.clone());
                }
                let tool = self
                    .tools
                    .0
                    .get(&name)
                    .context("upstream returned an undeclared tool")?;
                if let Some(field) = &tool.custom_field {
                    self.custom
                        .entry(item_id.clone())
                        .or_insert_with(|| PendingCustom {
                            field: field.clone(),
                            arguments: event["item"]["arguments"].as_str().unwrap_or("").into(),
                            emitted: false,
                        });
                    if kind == "response.output_item.done" {
                        if event["item"].get("arguments").is_none() {
                            event["item"]["arguments"] = json!(self.custom[&item_id].arguments);
                        }
                        prefix = self.custom_events(
                            &item_id,
                            event["item"]["arguments"].as_str(),
                            &event,
                        )?;
                    }
                }
                self.tools
                    .restore_item(&mut event["item"], kind == "response.output_item.done")?;
            }
            "response.function_call_arguments.delta" | "response.function_call_arguments.done" => {
                let item_id = event["item_id"]
                    .as_str()
                    .context("tool arguments need an item_id")?
                    .to_owned();
                if !self.calls.contains_key(&item_id) || self.completed_items.contains(&item_id) {
                    bail!("upstream sent arguments without an active tool item");
                }
                if let Some(pending) = self.custom.get_mut(&item_id) {
                    if kind.ends_with(".delta") {
                        if pending.emitted {
                            bail!("upstream sent arguments after custom input completion");
                        }
                        pending
                            .arguments
                            .push_str(event["delta"].as_str().context("tool delta must be text")?);
                        return Ok(Vec::new());
                    }
                    return self.custom_events(&item_id, event["arguments"].as_str(), &event);
                }
            }
            "response.output_item.added" | "response.output_item.done"
                if event["item"]["type"] == "custom_tool_call" =>
            {
                bail!("upstream returned an untranslated custom tool call");
            }
            "response.output_item.added" | "response.output_item.done" => {
                self.tools.restore_item(
                    event
                        .get_mut("item")
                        .context("output event needs an item")?,
                    kind == "response.output_item.done",
                )?;
            }
            "response.completed" | "response.failed" | "response.incomplete" | "error" => {
                self.terminal = true
            }
            _ => {}
        }
        prefix.push(event);
        Ok(prefix)
    }

    fn event(&mut self, mut value: Value) -> Event {
        // A custom arguments-done event can expand to two client events. Give
        // every emitted event a monotonically increasing sequence number.
        value["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        Event::default()
            .event(value["type"].as_str().unwrap_or("error"))
            .data(value.to_string())
    }

    fn failure(&mut self, message: &str) -> Event {
        self.event(
            json!({"type":"response.failed", "response": {"id":self.response_id,
            "status":"failed", "error":{"code":"upstream_response_error", "message":message}}}),
        )
    }
}

/// Start the upstream request inside the stream so headers and keepalives reach
/// Codex even while the provider is waiting to send its response headers.
pub fn forward_stream(
    builder: reqwest::RequestBuilder,
    tools: ToolMap,
    model: String,
    sessions: SessionRegistry,
    session: String,
) -> Response {
    let events = async_stream::stream! {
        let mut translator = StreamTranslator::new(tools, model);
        let upstream = match builder.send().await {
            Ok(upstream) if upstream.status().is_success() => upstream,
            Ok(upstream) => {
                yield Ok::<Event, std::convert::Infallible>(translator.failure(&format!("upstream HTTP {}", upstream.status())));
                return;
            }
            Err(_) => { yield Ok(translator.failure("upstream connection failed")); return; }
        };
        let mut source = upstream.bytes_stream()
            .chain(futures_util::stream::once(std::future::ready(Ok::<_, reqwest::Error>(bytes::Bytes::from_static(b"\n\n")))))
            .eventsource();
        while let Some(event) = source.next().await {
            let event = match event {
                Ok(event) => event,
                Err(_) => { yield Ok(translator.failure("upstream SSE read failed")); return; }
            };
            if event.data.trim() == "[DONE]" { break; }
            let data: Value = match serde_json::from_str(&event.data) {
                Ok(data) => data,
                Err(_) => { yield Ok(translator.failure("upstream sent invalid SSE JSON")); return; }
            };
            let transformed = match translator.transform(data) {
                Ok(events) => events,
                Err(error) => { yield Ok(translator.failure(&error.to_string())); return; }
            };
            for value in transformed {
                if value["type"] == "response.created" || value["type"] == "response.completed" {
                    sessions.bind(&value["response"], &session);
                }
                yield Ok(translator.event(value));
            }
            if translator.terminal { return; }
        }
        yield Ok(translator.failure("upstream stream ended before a terminal Responses event"));
    };
    Sse::new(events)
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(5))
                .text("keepalive"),
        )
        .into_response()
}

pub async fn forward_json(
    builder: reqwest::RequestBuilder,
    tools: ToolMap,
    model: &str,
    sessions: &SessionRegistry,
    session: &str,
) -> Response {
    use axum::http::StatusCode;
    let upstream = match builder.send().await {
        Ok(response) => response,
        Err(_) => return (StatusCode::BAD_GATEWAY, "upstream connection failed").into_response(),
    };
    let status = upstream.status();
    let headers = upstream.headers().clone();
    let body = match upstream.bytes().await {
        Ok(body) => body,
        Err(_) => return (StatusCode::BAD_GATEWAY, "upstream body read failed").into_response(),
    };
    if !status.is_success() {
        let mut response = (status, body).into_response();
        for name in ["content-type", "retry-after", "x-request-id"] {
            if let Some(value) = headers.get(name) {
                response.headers_mut().insert(name, value.clone());
            }
        }
        return response;
    }
    let result = (|| -> Result<Value> {
        let mut response: Value =
            serde_json::from_slice(&body).context("upstream sent invalid Responses JSON")?;
        if response.get("error").is_some_and(|error| !error.is_null()) {
            bail!("upstream returned an error response");
        }
        if !response["id"].is_string() || !response["output"].is_array() {
            bail!("upstream sent an invalid Responses object");
        }
        tools.restore_response(&mut response, model)?;
        Ok(response)
    })();
    match result {
        Ok(response) => {
            sessions.bind(&response, session);
            let mut result = (status, axum::Json(response)).into_response();
            if let Some(id) = headers.get("x-request-id") {
                result.headers_mut().insert("x-request-id", id.clone());
            }
            result
        }
        Err(error) => (StatusCode::BAD_GATEWAY, error.to_string()).into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_references_do_not_destroy_siblings_property_names_or_example_data() {
        let mut schema = json!({
            "$ref":"#/$defs/Base", "type":"object", "required":["child"],
            "properties":{
                "$ref":{"type":"string"},
                "child":{"$ref":"#/$defs/Node","description":"keep this"},
                "list":{"items":{"anyOf":[{"$ref":"#/$defs/Node"},{"type":"null"}]}}
            },
            "$defs":{"Node":{"properties":{"child":{"$ref":"#/$defs/Node"}}}},
            "default":{"$ref":"literal data"}, "examples":[{"$ref":"literal example"}]
        });
        assert_eq!(strip_schema_refs(&mut schema), 4);
        assert_eq!(schema["type"], "object");
        assert_eq!(schema["required"], json!(["child"]));
        assert_eq!(schema["properties"]["$ref"]["type"], "string");
        assert_eq!(
            schema["properties"]["child"],
            json!({"description":"keep this"})
        );
        assert_eq!(schema["default"]["$ref"], "literal data");
        assert_eq!(schema["examples"][0]["$ref"], "literal example");
    }

    #[test]
    fn names_are_bounded_in_bytes_and_flattening_collisions_are_rejected() {
        let tool = Tool {
            name: "界".repeat(40),
            namespace: Some("plugin".into()),
            custom_field: None,
        };
        assert!(tool.upstream_name().len() <= 64);
        assert_eq!(tool.upstream_name(), tool.upstream_name());
        let mut registry = ToolMap::default();
        registry
            .insert(Tool {
                name: "b".into(),
                namespace: Some("a".into()),
                custom_field: None,
            })
            .unwrap();
        assert!(registry
            .insert(Tool {
                name: "a-b".into(),
                namespace: None,
                custom_field: None
            })
            .is_err());
    }

    #[test]
    fn session_registry_expires_and_evicts_response_bindings() {
        let sessions = SessionRegistry::new(1, Duration::from_secs(60));
        sessions.bind(&json!({"id":"first"}), "session-1");
        assert_eq!(sessions.session_for_request(Some("first")), "session-1");
        sessions.bind(&json!({"id":"second"}), "session-2");
        assert_ne!(sessions.session_for_request(Some("first")), "session-1");
        assert_eq!(sessions.session_for_request(Some("second")), "session-2");
        let expired = SessionRegistry::new(1, Duration::ZERO);
        expired.bind(&json!({"id":"expired"}), "old-session");
        assert_ne!(expired.session_for_request(Some("expired")), "old-session");
    }

    #[test]
    fn stream_rejects_changed_identities_and_duplicate_call_ids() {
        let tools = vec![
            json!({"type":"function","name":"first"}),
            json!({"type":"function","name":"second"}),
        ];
        let (_, map) = normalize_tools(&tools, &HashSet::new(), &mut Changes::default()).unwrap();
        let mut translator = StreamTranslator::new(map, "model".into());
        let event = json!({"type":"response.output_item.added","item":{"type":"function_call","name":"first","id":"fc_1","call_id":"call_1","arguments":""}});
        translator.transform(event.clone()).unwrap();
        let mut changed = event.clone();
        changed["type"] = json!("response.output_item.done");
        changed["item"]["name"] = json!("second");
        assert!(translator.transform(changed).is_err());
        let mut duplicate = event;
        duplicate["item"]["id"] = json!("fc_2");
        assert!(translator.transform(duplicate).is_err());
    }
}
