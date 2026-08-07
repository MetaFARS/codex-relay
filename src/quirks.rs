//! Registry of platform-specific quirks: workarounds for provider bugs or
//! provider-specific behavior that is not part of the Responses ⇄ Chat
//! Completions contract itself.
//!
//! # Quirk classes
//!
//! **Request-shaping (class A)** quirks proactively change what the relay
//! sends upstream. They affect every request, so they must be gated by
//! provider/model identity (e.g. model-name matching).
//!
//! **Response-healing (class B)** quirks repair malformed upstream responses.
//! They are gated by detecting the anomaly itself and are no-ops on healthy
//! responses — model-name or version gating would be both unreliable (OpenAI-
//! compatible APIs rarely expose versions; the same broken model is served
//! under many names by NIM/Ollama/vLLM hosts) and unnecessary (the trigger
//! condition is precise). They self-deactivate once the platform fixes the
//! bug.
//!
//! # Lifecycle and removal
//!
//! Every class-B quirk logs a `warn!` each time it fires. That telemetry is
//! the removal signal: once the upstream fix ships and logs show no firings
//! over a sustained period, the quirk is a deletion candidate. Until then,
//! users can disable any quirk immediately — without waiting for a relay
//! release — via:
//!
//! ```text
//! CODEX_RELAY_DISABLE_QUIRKS=dsml_heal,glm_thinking
//! ```
//!
//! # Registered quirks
//!
//! | name           | class | trigger                                   | remove when |
//! |----------------|-------|-------------------------------------------|-------------|
//! | `glm_thinking` | A     | model name looks like GLM/Zhipu           | GLM emits reasoning_content without the explicit `thinking` switch (issue #26) |
//! | `dsml_heal`    | B     | leaked `<｜DSML｜` markup in text content  | DeepSeek V4 stops leaking DSML (vllm-project/vllm#40801) and telemetry shows no firings for ~a month |
//! | `missing_done` | B     | SSE stream closes cleanly without `[DONE]`| providers terminate streams spec-compliantly (issue #31) |
//! | `think_tags`   | B     | `<think>` markup leaked into text content  | providers deploy reasoning models with a matching vLLM `--reasoning-parser` so thinking arrives in `reasoning_content` |

use std::collections::HashSet;

/// Names of all registered quirks, for validation and documentation.
pub const QUIRK_NAMES: &[&str] = &["glm_thinking", "dsml_heal", "missing_done", "think_tags"];

/// Whether a quirk is enabled, honoring the `CODEX_RELAY_DISABLE_QUIRKS`
/// kill switch (comma-separated quirk names, case-insensitive).
pub fn quirk_enabled(name: &str) -> bool {
    !disabled_quirks_from_env().contains(name)
}

fn disabled_quirks_from_env() -> HashSet<String> {
    parse_disabled_quirks(&std::env::var("CODEX_RELAY_DISABLE_QUIRKS").unwrap_or_default())
}

fn parse_disabled_quirks(raw: &str) -> HashSet<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_ascii_lowercase)
        .inspect(|name| {
            if !QUIRK_NAMES.contains(&name.as_str()) {
                tracing::warn!("CODEX_RELAY_DISABLE_QUIRKS: unknown quirk name `{name}`");
            }
        })
        .collect()
}

/// Whether a model name looks like a GLM/Zhipu reasoning model that needs the
/// explicit `thinking` switch to emit reasoning_content (quirk `glm_thinking`).
pub fn is_glm_like_model(model: &str) -> bool {
    let m = model.to_ascii_lowercase();
    m.contains("glm") || m.contains("zhipu") || m.contains("bigmodel")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_disable_list() {
        let disabled = parse_disabled_quirks("dsml_heal, GLM_THINKING");
        assert!(disabled.contains("dsml_heal"));
        assert!(disabled.contains("glm_thinking"));
        assert!(!disabled.contains("missing_done"));
        assert!(parse_disabled_quirks("").is_empty());
    }

    #[test]
    fn glm_like_model_matching() {
        assert!(is_glm_like_model("glm-5.2"));
        assert!(is_glm_like_model("ZhipuAI/foo"));
        assert!(is_glm_like_model("bigmodel-x"));
        assert!(!is_glm_like_model("deepseek-v4-pro"));
    }
}
