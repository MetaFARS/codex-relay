//! Opt-in request-lifecycle tracing.
//!
//! Enabled only when `CODEX_RELAY_TRACE_TIMINGS=1` (or `true`), read once per
//! process. Disabled by default: marks then cost one boolean check and
//! nothing is logged.
//!
//! Recorded data is limited to monotonic phase timestamps, byte counts, token
//! counts, and the relay's own response id as correlation id. Prompts, tool
//! arguments, API keys, authorization headers, and upstream response content
//! are never recorded.
//!
//! Phase semantics (all offsets measured from upstream request dispatch where
//! applicable, so HTTP TTFB, first SSE event, first reasoning token, and
//! first visible-text token stay clearly distinguishable):
//!
//! | phase                | meaning                                        |
//! |----------------------|------------------------------------------------|
//! | request_received     | handler entry (t0)                             |
//! | translated           | Responses→Chat translation finished            |
//! | upstream_sent        | upstream HTTP request dispatched               |
//! | upstream_headers     | upstream response headers received (HTTP TTFB) |
//! | first_upstream_event | first upstream SSE event parsed                |
//! | first_reasoning      | first reasoning token forwarded downstream     |
//! | first_text           | first visible-text token forwarded downstream  |
//! | upstream_completed   | upstream stream finished ([DONE]/close/error)  |

use std::sync::OnceLock;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    UpstreamSent,
    UpstreamHeaders,
    FirstUpstreamEvent,
    FirstReasoning,
    FirstText,
    UpstreamCompleted,
}

pub fn tracing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("CODEX_RELAY_TRACE_TIMINGS")
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str(),
            "1" | "true" | "yes"
        )
    })
}

#[derive(Debug)]
pub struct RequestTrace {
    enabled: bool,
    start: Instant,
    translated: Option<Instant>,
    marks: [Option<Instant>; 6],
    request_bytes: usize,
    tools_bytes: usize,
    tool_count: usize,
    usage: Option<(u32, u32)>,
}

impl RequestTrace {
    pub fn start() -> Self {
        let enabled = tracing_enabled();
        Self {
            enabled,
            start: Instant::now(),
            translated: None,
            marks: [None; 6],
            request_bytes: 0,
            tools_bytes: 0,
            tool_count: 0,
            usage: None,
        }
    }

    /// Byte counts for the inbound request (body and serialized tools).
    /// Only measured when tracing is enabled.
    pub fn record_request_size(&mut self, body_bytes: usize, tools: &[serde_json::Value]) {
        if !self.enabled {
            return;
        }
        self.request_bytes = body_bytes;
        self.tool_count = tools.len();
        self.tools_bytes = tools
            .iter()
            .filter_map(|tool| serde_json::to_vec(tool).ok())
            .map(|bytes| bytes.len())
            .sum();
    }

    pub fn mark_translated(&mut self) {
        if self.enabled {
            self.translated = Some(Instant::now());
        }
    }

    pub fn mark(&mut self, phase: Phase) {
        if !self.enabled {
            return;
        }
        let slot = &mut self.marks[phase as usize];
        if slot.is_none() {
            *slot = Some(Instant::now());
        }
    }

    pub fn record_usage(&mut self, input_tokens: u32, output_tokens: u32) {
        if self.enabled {
            self.usage = Some((input_tokens, output_tokens));
        }
    }

    /// Emit the aggregated single-line report. `ok` distinguishes a normally
    /// completed response from one that failed mid-lifecycle.
    pub fn finish(&self, response_id: &str, ok: bool) {
        if !self.enabled {
            return;
        }
        let since_start = |t: Option<Instant>| t.map(|t| t.duration_since(self.start).as_millis());
        let sent = self.marks[Phase::UpstreamSent as usize];
        let since_sent = |mark: Option<Instant>| match (sent, mark) {
            (Some(sent), Some(mark)) => Some(mark.duration_since(sent).as_millis()),
            _ => None,
        };
        tracing::info!(
            trace_id = %response_id,
            ok,
            total_ms = self.start.elapsed().as_millis(),
            translate_ms = since_start(self.translated),
            connect_ms = since_sent(self.marks[Phase::UpstreamHeaders as usize]),
            first_sse_ms = since_sent(self.marks[Phase::FirstUpstreamEvent as usize]),
            first_reasoning_ms = since_sent(self.marks[Phase::FirstReasoning as usize]),
            first_text_ms = since_sent(self.marks[Phase::FirstText as usize]),
            upstream_done_ms = since_sent(self.marks[Phase::UpstreamCompleted as usize]),
            request_bytes = self.request_bytes,
            tools_bytes = self.tools_bytes,
            tool_count = self.tool_count,
            input_tokens = self.usage.map(|(input, _)| input),
            output_tokens = self.usage.map(|(_, output)| output),
            "request lifecycle trace"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_trace_is_inert() {
        // The default (env unset) must be off; the struct records nothing.
        let trace = RequestTrace::start();
        assert!(!trace.enabled, "tests must run with tracing disabled");
        let mut trace = trace;
        trace.mark(Phase::UpstreamSent);
        trace.record_usage(1, 2);
        trace.finish("trace_test", true);
        assert!(trace.marks[Phase::UpstreamSent as usize].is_none());
        assert!(trace.usage.is_none());
    }
}
