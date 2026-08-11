# codex-relay

A lightweight Rust proxy that translates the OpenAI **Responses API** (used by [Codex CLI](https://github.com/openai/codex)) into the **Chat Completions API**, letting Codex work with any OpenAI-compatible provider — DeepSeek, Kimi, Qwen, Mistral, Groq, xAI, OpenRouter, and more.

## Why

Codex CLI speaks the OpenAI Responses API, which is an OpenAI-proprietary stateful protocol. Every other provider exposes the standard Chat Completions API. `codex-relay` sits between Codex and your chosen provider, translating on the fly — no code changes to Codex required.

## Install

```bash
# From PyPI — prebuilt binary for your platform
pip install codex-relay

# From crates.io
cargo install codex-relay
```

## Quick start

**1. Start the relay**

```bash
CODEX_RELAY_UPSTREAM=https://api.deepseek.com/v1 \
CODEX_RELAY_API_KEY=$DEEPSEEK_API_KEY \
CODEX_RELAY_PORT=4446 \
codex-relay
```

On startup, the relay logs the available upstream models and prints a hint:

```
ℹ upstream models: deepseek-chat, deepseek-reasoner
⚠  To configure Codex with model metadata, run:  codex-relay --print-config --upstream ...
```

**2. Generate your Codex config**

```bash
codex-relay --print-config \
  --upstream https://api.deepseek.com/v1 \
  --api-key $DEEPSEEK_API_KEY \
  --model-catalog ~/.codex/codex-relay-models.json
```

This prints a ready-to-use `~/.codex/config.toml` snippet and writes a model
catalog containing both Codex's built-in models and every upstream model, so
you won't see the **"Model metadata … not found"** warning. The generated
entries inherit version-sensitive instructions and tool formats from the
installed Codex CLI. Use `--model-template <MODEL>` to select a different
bundled model as that template.

If you prefer to write the config by hand, here is the minimal form:

```toml
model = "deepseek-chat"
model_provider = "deepseek-relay"
model_catalog_json = "/home/user/.codex/codex-relay-models.json"

[model_providers.deepseek-relay]
name = "DeepSeek"
base_url = "http://127.0.0.1:4446/v1"
wire_api = "responses"
env_key = "DEEPSEEK_API_KEY"
```

> Recent Codex versions use `model_catalog_json`; the former
> `[model_properties]` config syntax is no longer supported. A custom catalog
> replaces the built-in catalog, so `codex-relay` starts with the catalog from
> your installed Codex version and appends upstream models instead of creating
> an incomplete replacement.

**3. Use Codex normally** — it routes through the relay transparently.

## CLI reference

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--port` | `CODEX_RELAY_PORT` | `4444` | Listen port |
| `--bind` | `CODEX_RELAY_BIND` | `127.0.0.1` | IP address to bind the listener to (e.g. `0.0.0.0` to accept remote connections) |
| `--upstream` | `CODEX_RELAY_UPSTREAM` | `https://openrouter.ai/api/v1` | Upstream Chat Completions base URL |
| `--api-key` | `CODEX_RELAY_API_KEY` | _(empty)_ | API key forwarded to upstream |
| `--upstream-extra-params` | `CODEX_RELAY_UPSTREAM_EXTRA_PARAMS` | _(empty)_ | JSON object merged into each upstream Chat Completions request |
| `--drop-upstream-params` | `CODEX_RELAY_DROP_PARAMS` | _(empty)_ | JSON array of top-level upstream request parameters to remove |
| `--model-map` | `CODEX_RELAY_MODEL_MAP` | _(empty)_ | Comma-separated `source:target` model name translations |
| `--print-config` | _(none)_ | — | Print a Codex config snippet and exit |
| `--model-catalog` | _(none)_ | — | With `--print-config`, write a version-matched full model catalog to this path |
| `--model-template` | _(none)_ | first visible bundled model | Bundled Codex model whose instructions and tool formats generated entries inherit |
| `--record-corpus` | `CODEX_RELAY_RECORD_CORPUS` | _(off)_ | Append the conversation flow of every completed turn to daily JSONL files (OpenAI messages format) in this directory |
| `--session-ttl-hours` | `CODEX_RELAY_SESSION_TTL_HOURS` | `168` | Retain idle `previous_response_id` history and reasoning state for this many hours |
| `--max-sessions` | `CODEX_RELAY_MAX_SESSIONS` | `256` | Maximum completed response histories retained for continuation |
| `--max-session-memory-mb` | `CODEX_RELAY_MAX_SESSION_MEMORY_MB` | `512` | Approximate memory budget for retained session/reasoning state |

## Supported providers

| Provider | Base URL | Suggested port |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | 4446 |
| Kimi (Moonshot) | `https://api.moonshot.cn/v1` | 4447 |
| GLM (Zhipu) | `https://open.bigmodel.cn/api/coding/paas/v4` | 4453 |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 4448 |
| Mistral | `https://api.mistral.ai/v1` | 4449 |
| Groq | `https://api.groq.com/openai/v1` | 4450 |
| xAI | `https://api.x.ai/v1` | 4451 |
| OpenRouter | `https://openrouter.ai/api/v1` | 4452 |

Any OpenAI-compatible endpoint works.

### Upstream request parameters

Some providers expose non-standard Chat Completions parameters. You can merge
top-level JSON fields into every upstream request, and optionally drop generated
top-level fields before the merge. For example, to disable DeepSeek V4
thinking/reasoning mode:

```bash
CODEX_RELAY_UPSTREAM_EXTRA_PARAMS='{"thinking":{"type":"disabled"}}' \
CODEX_RELAY_DROP_PARAMS='["reasoning_effort"]' \
codex-relay --upstream https://api.deepseek.com/v1 --api-key "$DEEPSEEK_API_KEY"
```

## Features

- **Streaming** — full SSE streaming with correct event sequencing
- **Tool calls** — accumulates streaming deltas and emits structured function_call items
- **Parallel tool calls** — consecutive function_call input items merged into one assistant message
- **Reasoning models** — streams `reasoning_content` (or the `reasoning` alias) as Responses reasoning summaries and preserves it across turns (Kimi k2.6, DeepSeek-R1, GLM). For GLM/Zhipu models the relay automatically sends `thinking: {"type": "enabled"}`, since GLM otherwise suppresses reasoning under Codex's system prompt
- **Model catalog** — proxies `/v1/models` from the upstream provider
- **Auto-config** — `--print-config` generates a complete Codex config with model metadata

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CODEX_RELAY_PORT` | `4444` | Port to listen on |
| `CODEX_RELAY_BIND` | `127.0.0.1` | IP address to bind the listener to (e.g. `0.0.0.0` to accept remote connections) |
| `CODEX_RELAY_UPSTREAM` | `https://openrouter.ai/api/v1` | Upstream Chat Completions base URL |
| `CODEX_RELAY_API_KEY` | _(empty)_ | API key forwarded to upstream |
| `CODEX_RELAY_UPSTREAM_EXTRA_PARAMS` | _(empty)_ | JSON object merged into each upstream Chat Completions request body |
| `CODEX_RELAY_DROP_PARAMS` | _(empty)_ | JSON array of top-level upstream request parameter names to remove before forwarding |
| `CODEX_RELAY_MODEL_MAP` | _(empty)_ | Comma-separated `source:target` model name translations (e.g., `gpt-5.4:deepseek-v4-pro`) |
| `CODEX_RELAY_TOOL_DENYLIST` | _(empty)_ | Comma-separated tool names to remove before forwarding tools to the upstream model |
| `CODEX_RELAY_DISABLE_QUIRKS` | _(empty)_ | Comma-separated [platform quirk](#platform-quirks) names to disable (e.g. `dsml_heal,glm_thinking`) |
| `CODEX_RELAY_SESSION_TTL_HOURS` | `168` | Retain idle session/reasoning state for this many hours |
| `CODEX_RELAY_MAX_SESSIONS` | `256` | Maximum completed response histories retained for `previous_response_id` |
| `CODEX_RELAY_MAX_SESSION_MEMORY_MB` | `512` | Approximate memory budget for retained session/reasoning state |
| `CODEX_RELAY_HISTORY_STORE` | `memory` | Retained history backend: `memory` or `disk` |
| `CODEX_RELAY_HISTORY_DIR` | `.codex-relay-history` | Directory for disk-backed history records |
| `CODEX_RELAY_RECORD_CORPUS` | _(off)_ | Directory to append per-turn conversation records (OpenAI messages JSONL); off unless set |
| `RUST_LOG` | `codex_relay=info` | Log verbosity |

## Platform quirks

Some providers need workarounds that are not part of the Responses ⇄ Chat Completions translation itself. These are registered as named quirks (see `src/quirks.rs` for the full registry, triggers, and removal criteria):

| Quirk | Kind | What it does |
|---|---|---|
| `glm_thinking` | request-shaping | Sends `thinking: enabled` for GLM/Zhipu models so they emit `reasoning_content` (issue #26) |
| `dsml_heal` | response-healing | Parses DeepSeek V4's intermittently leaked DSML tool-call markup in text content back into structured tool calls |
| `missing_done` | response-healing | Treats a cleanly closed SSE stream without `[DONE]` as complete when a full turn was received (issue #31) |

Response-healing quirks activate only when the anomaly is detected and log a `quirk <name> fired` warning each time, so you can tell from the logs whether a workaround is still needed. Once the platform fixes the underlying bug, disable a quirk immediately with:

```bash
CODEX_RELAY_DISABLE_QUIRKS=dsml_heal codex-relay
```

## Python API

```python
from codex_relay import start

proc = start(port=4446, upstream="https://api.deepseek.com/v1", api_key="sk-...")
# ... use Codex ...
proc.terminate()
```

## Testing

Two layers — offline tests pin behavior against captured Codex wire-shape;
live tests pin behavior against real provider APIs.

## Debugging tool round-trips

For tool-routing issues, enable debug logs:

```bash
RUST_LOG=codex_relay=debug codex-relay
```

The relay logs tool names only, never tool arguments or message content:

- `response tools=...` — tools received from Codex's Responses API request
- `upstream tools=...` — tools forwarded to the Chat Completions upstream
- `upstream function_calls=...` — function calls returned by a blocking upstream response
- `upstream stream function_calls=...` — function calls returned by a streaming upstream response

These lines are useful for checking whether a tool such as `spawn_agent`
was preserved by the relay, and whether the failure happened before or after
the model selected that tool.

### Disk-backed history

By default, `codex-relay` keeps retained `previous_response_id` histories and
reasoning lookups in memory. For longer-running processes or deeper debugging,
you can opt into an inspectable on-disk store:

```bash
CODEX_RELAY_HISTORY_STORE=disk \
CODEX_RELAY_HISTORY_DIR=.codex-relay-history \
codex-relay
```

The disk backend writes JSON records under:

```text
.codex-relay-history/
  sessions/
  reasoning/
  turns/
```

Session records contain the translated Chat Completions `messages` retained for
a response id. Reasoning records keep call-id and turn-fingerprint lookups used
to round-trip provider reasoning content. The relay keeps only an in-memory
index for disk-backed entries and loads payloads on demand.

Treat this directory as sensitive: records may contain prompts, tool outputs,
and other conversation data. The same TTL/count/byte retention knobs apply to
disk-backed records, and evicted entries are removed from disk.

### Corpus recording

For building datasets, `--record-corpus <dir>` continuously appends the
conversation flow to daily-sharded JSONL files in **OpenAI messages format**:

```bash
codex-relay --record-corpus ./corpus \
  --upstream https://api.deepseek.com/v1 --api-key "$DEEPSEEK_API_KEY"
```

This is **off by default** and is a separate subsystem from the retention cache
above: the corpus is an **append-only archive** that is never evicted, whereas
the session store is an evictable continuation cache.

Each line is an *incremental turn event* — only the messages new to that turn
are written, so the same conversation is reconstructed by concatenating the
`messages` of every event that shares a `conversation_id`:

```text
corpus/
  corpus-2026-04-04.jsonl
```

```json
{
  "conversation_id": "resp_abc…",
  "response_id": "resp_def…",
  "parent_response_id": "resp_abc…",
  "timestamp_unix_ms": 1783447750503,
  "model": "deepseek-chat",
  "messages": [ { "role": "user", "content": "…" }, { "role": "assistant", "content": "…" } ]
}
```

The `messages` payload uses the standard OpenAI schema and preserves
`tool_calls`, `role: "tool"` outputs (`tool_call_id`), and assistant
`reasoning_content` (a widely-used non-standard field that training frameworks
ignore if unknown). The first event of a conversation includes the system
prompt; subsequent events omit it. Isolated `spawn_agent` child requests start
their own `conversation_id`.

To fold the turn events back into whole-conversation OpenAI records:

```bash
jq -s 'group_by(.conversation_id)[]
       | {messages: (map(.messages) | add)}' corpus/*.jsonl
```

> ⚠️ Records contain prompts, tool call **arguments**, and tool outputs — more
> than the debug logs ever emit. Treat the directory as sensitive, especially
> when combined with `--bind` on a non-loopback address.

### Subagent tool routing

Codex subagent tools such as `spawn_agent`, `wait_agent`, and `close_agent`
are runtime tools. The relay can preserve them in the tool schema and round-trip
the model's selected function call, but it cannot reliably detect whether the
local Codex app-server daemon is new enough to execute those calls.

If Codex shows `unsupported call: spawn_agent`, first verify that the Codex CLI
and app-server daemon versions match. A stale daemon can expose a newer tool
schema to the model while lacking the handler that executes the returned call.
Also check your Codex config: `[features] subagents = true` is not recognized;
use `[features] multi_agent = true` only if you need to override the default.

As an escape hatch for affected runtimes, remove unsupported tools before they
reach the upstream model:

```bash
CODEX_RELAY_TOOL_DENYLIST=spawn_agent,wait_agent,close_agent codex-relay
```

The denylist matches the tool name forwarded to Chat Completions. Namespaced
MCP tools use their flattened name, for example
`mcp__codex_apps__github-_fetch_issue`.

**Offline (always green, default `cargo test`)**

Replays Codex CLI fixtures through the translation layer and asserts
role/tool/reasoning behavior. Each fixture pins a Codex CLI version under
`tests/fixtures/codex_<major>_<minor>_<patch>/`.

```bash
cargo test
```

**Live (gated on provider API key, `#[ignore]` by default)**

Spawns the relay binary on a random port, points it at the real provider, and
exercises `/v1/models`, blocking + streaming, tool calls, and (for thinking
models) the `reasoning_content` round-trip via an in-process recording proxy.

```bash
DEEPSEEK_API_KEY=sk-... cargo test --test compat_deepseek_live -- --ignored --test-threads=1
```

**Regenerating fixtures after a Codex upgrade**

1. Add a debug dump to the relay (write `body` bytes from `handle_responses`
   to a file before parsing).
2. Run a real `codex exec` against it; copy `inbound_*.json` to a new
   `tests/fixtures/codex_<major>_<minor>_<patch>/` folder.
3. Trim each payload down to the smallest one that exercises the feature you
   want to lock in.
4. Add a row to `tests/fixtures/VERSIONS.md` and a test pointing at the new
   directory.

The old fixture directory stays as a regression net so the relay keeps
working with the previous Codex CLI release.

## Disclaimer

This project is **not affiliated with, endorsed by, or sponsored by OpenAI**. "Codex" refers to [OpenAI Codex CLI](https://github.com/openai/codex), an open-source project licensed under Apache-2.0. codex-relay is an independent, community-built translation proxy.

## Contributors

- [myk5010](https://github.com/myk5010) — system/developer message ordering fix and model name mapping ([\#4](https://github.com/MetaFARS/codex-relay/pull/4))
- [qcnhy](https://github.com/qcnhy) — streaming usage, MCP namespace bug reports, namespace tool-routing analysis, and independent verification ([\#5](https://github.com/MetaFARS/codex-relay/issues/5), [\#6](https://github.com/MetaFARS/codex-relay/issues/6), [\#17](https://github.com/MetaFARS/codex-relay/issues/17))
- [JasonC93](https://github.com/JasonC93) — subagent tool-routing and spawned-agent context isolation reports ([\#10](https://github.com/MetaFARS/codex-relay/issues/10), [\#12](https://github.com/MetaFARS/codex-relay/issues/12))
- [ma-buting](https://github.com/ma-buting) — namespace tool-name separator fix ([\#19](https://github.com/MetaFARS/codex-relay/pull/19))
- [SaladDay](https://github.com/SaladDay) — prompt-cache accounting debug logs ([\#22](https://github.com/MetaFARS/codex-relay/pull/22))
- [Cherno76](https://github.com/Cherno76) — prompt-cache hit tokens in Responses API usage ([\#23](https://github.com/MetaFARS/codex-relay/pull/23))

## License

MIT
