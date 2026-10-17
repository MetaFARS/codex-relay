//! Real process signals against local, deliberately stalled upstreams.
#![cfg(unix)]

use axum::{
    body::Body,
    extract::State,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Child, Command, ExitStatus, Stdio},
    sync::{mpsc, Arc, Mutex},
    time::Duration,
};
use tokio::{
    sync::Notify,
    task::JoinHandle,
    time::{sleep, timeout},
};

const ANSWER: &str = "beforeafter";

#[derive(Clone)]
struct MockState {
    native: bool,
    stream: bool,
    started: Arc<Notify>,
    release: Arc<Notify>,
}

struct Mock {
    port: u16,
    state: MockState,
    task: JoinHandle<()>,
}

impl Mock {
    async fn start(native: bool, stream: bool) -> Self {
        let state = MockState {
            native,
            stream,
            started: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        };
        let app = Router::new()
            .route("/v1/models", get(|| async { Json(json!({"data":[]})) }))
            .route("/v1/responses", post(upstream))
            .route("/v1/chat/completions", post(upstream))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let task = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self { port, state, task }
    }
}

impl Drop for Mock {
    fn drop(&mut self) {
        self.state.release.notify_one();
        self.task.abort();
    }
}

fn event(value: Value) -> String {
    format!("data: {value}\n\n")
}

async fn upstream(State(state): State<MockState>, Json(body): Json<Value>) -> Response {
    assert_eq!(body["stream"], state.stream);
    let message = json!({"type":"message", "role":"assistant", "content":[{"type":"output_text", "text":ANSWER}]});
    if !state.stream {
        state.started.notify_one();
        state.release.notified().await;
        return Json(if state.native {
            json!({"id":"resp_mock", "output":[message]})
        } else {
            json!({"choices":[{"message":{"role":"assistant", "content":ANSWER}}]})
        })
        .into_response();
    }
    let prefix = if state.native {
        event(json!({"type":"response.created", "response":{"id":"resp_mock", "output":[]}}))
            + &event(
                json!({"type":"response.output_text.delta", "item_id":"msg_mock", "output_index":0, "content_index":0, "delta":"before"}),
            )
    } else {
        event(json!({"choices":[{"index":0, "delta":{"role":"assistant", "content":"before"}}]}))
    };
    let suffix = if state.native {
        event(
            json!({"type":"response.output_text.delta", "item_id":"msg_mock", "output_index":0, "content_index":0, "delta":"after"}),
        ) + &event(json!({"type":"response.output_item.done", "output_index":0, "item":message}))
            + &event(
                json!({"type":"response.completed", "response":{"id":"resp_mock", "status":"completed", "output":[message]}}),
            )
    } else {
        event(json!({"choices":[{"index":0, "delta":{"content":"after"}, "finish_reason":"stop"}]}))
            + "data: [DONE]\n\n"
    };
    let body = async_stream::stream! {
        yield Ok::<_, std::convert::Infallible>(bytes::Bytes::from(prefix));
        state.started.notify_one();
        state.release.notified().await;
        yield Ok(bytes::Bytes::from(suffix));
    };
    Response::builder()
        .header("content-type", "text/event-stream")
        .body(Body::from_stream(body))
        .unwrap()
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let path =
            std::env::temp_dir().join(format!("codex-relay-shutdown-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct Relay {
    child: Child,
    port: u16,
    logs: Arc<Mutex<String>>,
    directory: TestDirectory,
}

impl Relay {
    fn start(mock: &Mock) -> Self {
        let directory = TestDirectory::new();
        let mut command = Command::new(env!("CARGO_BIN_EXE_codex-relay"));
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("CODEX_RELAY_") {
                command.env_remove(key);
            }
        }
        command
            .args([
                "--port",
                "0",
                "--upstream",
                &format!("http://127.0.0.1:{}/v1", mock.port),
                "--api-key",
                "",
            ])
            .env("RUST_LOG", "codex_relay=info")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        if mock.state.native {
            command.arg("--opencode-go-compat");
        } else {
            command
                .env("CODEX_RELAY_HISTORY_STORE", "disk")
                .env("CODEX_RELAY_HISTORY_DIR", directory.0.join("history"))
                .arg("--record-corpus")
                .arg(directory.0.join("corpus"));
        }
        let mut child = command.spawn().unwrap();
        let stdout = child.stdout.take().unwrap();
        let logs = Arc::new(Mutex::new(String::new()));
        let captured = logs.clone();
        let (ready, port) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(port) = line
                    .split("listening on 127.0.0.1:")
                    .nth(1)
                    .and_then(|address| address.split(|c: char| !c.is_ascii_digit()).next())
                    .and_then(|port| port.parse::<u16>().ok())
                {
                    let _ = ready.send(port);
                }
                captured.lock().unwrap().push_str(&line);
            }
        });
        let mut relay = Self {
            child,
            port: 0,
            logs,
            directory,
        };
        relay.port = port
            .recv_timeout(Duration::from_secs(5))
            .expect("relay readiness");
        relay
    }

    fn signal(&self, signal: &str) {
        assert!(Command::new("kill")
            .args(["-s", signal, &self.child.id().to_string()])
            .status()
            .unwrap()
            .success());
    }

    async fn wait_for_drain(&mut self) {
        timeout(Duration::from_secs(5), async {
            loop {
                if self
                    .logs
                    .lock()
                    .unwrap()
                    .contains("shutdown signal received")
                {
                    break;
                }
                assert!(
                    self.child.try_wait().unwrap().is_none(),
                    "relay exited before draining"
                );
                sleep(Duration::from_millis(5)).await;
            }
            while tokio::net::TcpStream::connect(("127.0.0.1", self.port))
                .await
                .is_ok()
            {
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("relay should stop accepting new connections");
        assert!(
            self.child.try_wait().unwrap().is_none(),
            "active request must be allowed to finish"
        );
    }

    async fn wait_exit(&mut self) -> ExitStatus {
        timeout(Duration::from_secs(5), async {
            loop {
                if let Some(status) = self.child.try_wait().unwrap() {
                    return status;
                }
                sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .expect("relay should exit")
    }

    fn request(
        &self,
        stream: bool,
    ) -> JoinHandle<Result<(reqwest::StatusCode, String), reqwest::Error>> {
        let port = self.port;
        tokio::spawn(async move {
            let response = reqwest::Client::new().post(format!("http://127.0.0.1:{port}/v1/responses"))
                .json(&json!({"model":"mock", "instructions":"base rules", "input":"test", "stream":stream}))
                .send().await?;
            Ok((response.status(), response.text().await?))
        })
    }

    fn assert_persisted(&self) {
        for directory in [
            self.directory.0.join("history/sessions"),
            self.directory.0.join("corpus"),
        ] {
            assert!(
                std::fs::read_dir(directory).unwrap().any(|entry| {
                    std::fs::read_to_string(entry.unwrap().path())
                        .unwrap()
                        .contains(ANSWER)
                }),
                "completed request must be persisted before exiting"
            );
        }
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[tokio::test]
async fn first_signal_drains_both_protocols_and_response_modes() {
    for native in [false, true] {
        for stream in [false, true] {
            for signal in ["INT", "TERM"] {
                let mock = Mock::start(native, stream).await;
                let mut relay = Relay::start(&mock);
                let response = relay.request(stream);
                timeout(Duration::from_secs(5), mock.state.started.notified())
                    .await
                    .unwrap();
                relay.signal(signal);
                relay.wait_for_drain().await;
                assert!(!response.is_finished());
                mock.state.release.notify_one();
                let (status, body) = timeout(Duration::from_secs(5), response)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
                assert!(status.is_success());
                assert!(body.contains(ANSWER), "{body}");
                if stream {
                    assert!(body.contains("response.completed"));
                }
                assert!(relay.wait_exit().await.success());
                if !native {
                    relay.assert_persisted();
                }
            }
        }
    }
}

#[tokio::test]
async fn second_signal_interrupts_stalled_requests_for_every_signal_pair() {
    for native in [false, true] {
        for stream in [false, true] {
            for first in ["INT", "TERM"] {
                for second in ["INT", "TERM"] {
                    let mock = Mock::start(native, stream).await;
                    let mut relay = Relay::start(&mock);
                    let response = relay.request(stream);
                    timeout(Duration::from_secs(5), mock.state.started.notified())
                        .await
                        .unwrap();
                    relay.signal(first);
                    relay.wait_for_drain().await;
                    assert!(!response.is_finished());
                    relay.signal(second);
                    assert_eq!(
                        relay.wait_exit().await.code(),
                        Some(1),
                        "{first} then {second}, native={native}, stream={stream}"
                    );
                    let _ = timeout(Duration::from_secs(5), response)
                        .await
                        .unwrap()
                        .unwrap();
                }
            }
        }
    }
}

#[tokio::test]
async fn idle_server_exits_cleanly_on_either_signal() {
    let mock = Mock::start(false, false).await;
    for signal in ["INT", "TERM"] {
        let mut relay = Relay::start(&mock);
        relay.signal(signal);
        assert!(relay.wait_exit().await.success());
    }
}

// Interpose sigaction only in the child. Other platforms still run the real
// Unix signal tests above; this fault injection needs dynamic Linux/glibc.
#[cfg(all(
    target_os = "linux",
    target_env = "gnu",
    not(target_feature = "crt-static")
))]
#[test]
fn registration_errors_fail_before_readiness_with_nonzero_status() {
    let directory = TestDirectory::new();
    let library = directory.0.join("fail_signal_registration.so");
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fail_signal_registration.c");
    let compilation = Command::new("cc")
        .args(["-shared", "-fPIC"])
        .arg(fixture)
        .args(["-ldl", "-o"])
        .arg(&library)
        .output()
        .unwrap();
    assert!(
        compilation.status.success(),
        "{}",
        String::from_utf8_lossy(&compilation.stderr)
    );
    for signal in ["INT", "TERM"] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_codex-relay"));
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("CODEX_RELAY_") {
                command.env_remove(key);
            }
        }
        let mut child = command
            .args([
                "--port",
                "0",
                "--upstream",
                "http://127.0.0.1:1/v1",
                "--api-key",
                "",
            ])
            .env("LD_PRELOAD", &library)
            .env("RELAY_SIGNAL_FAULT", signal)
            .env("RUST_LOG", "codex_relay=info")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let started = std::time::Instant::now();
        while child.try_wait().unwrap().is_none() {
            if started.elapsed() > Duration::from_secs(5) {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("registration failure should fail startup");
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let output = child.wait_with_output().unwrap();
        let logs = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!output.status.success(), "{logs}");
        assert!(
            logs.contains(&format!("failed to install SIG{signal} handler")),
            "{logs}"
        );
        assert!(!logs.contains("listening on"), "{logs}");
        assert!(!logs.contains("shutdown signal received"), "{logs}");
    }
}
