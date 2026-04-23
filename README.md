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
CODEX_BRIDGE_UPSTREAM=https://api.deepseek.com/v1 \
CODEX_BRIDGE_API_KEY=$DEEPSEEK_API_KEY \
CODEX_BRIDGE_PORT=4446 \
codex-relay
```

**2. Configure Codex** (`~/.codex/config.toml`)

```toml
model = "deepseek-chat"
model_provider = "deepseek-relay"

[model_providers.deepseek-relay]
name = "DeepSeek"
api_base_url = "http://127.0.0.1:4446/v1"
env_key = "DEEPSEEK_API_KEY"
```

**3. Use Codex normally** — it routes through the relay transparently.

## Supported providers

| Provider | Base URL | Suggested port |
|---|---|---|
| DeepSeek | `https://api.deepseek.com/v1` | 4446 |
| Kimi (Moonshot) | `https://api.moonshot.cn/v1` | 4447 |
| Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | 4448 |
| Mistral | `https://api.mistral.ai/v1` | 4449 |
| Groq | `https://api.groq.com/openai/v1` | 4450 |
| xAI | `https://api.x.ai/v1` | 4451 |
| OpenRouter | `https://openrouter.ai/api/v1` | 4452 |

Any OpenAI-compatible endpoint works.

## Features

- **Streaming** — full SSE streaming with correct event sequencing
- **Tool calls** — accumulates streaming deltas and emits structured function_call items
- **Parallel tool calls** — consecutive function_call input items merged into one assistant message
- **Reasoning models** — preserves `reasoning_content` across turns (Kimi k2.6, DeepSeek-R1)
- **Model catalog** — proxies `/v1/models` from the upstream provider

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CODEX_BRIDGE_PORT` | `4444` | Port to listen on |
| `CODEX_BRIDGE_UPSTREAM` | `https://openrouter.ai/api/v1` | Upstream Chat Completions base URL |
| `CODEX_BRIDGE_API_KEY` | _(empty)_ | API key forwarded to upstream |
| `RUST_LOG` | `codex_relay=info` | Log verbosity |

## Python API

```python
from codex_relay import start

proc = start(port=4446, upstream="https://api.deepseek.com/v1", api_key="sk-...")
# ... use Codex ...
proc.terminate()
```

## License

MIT
