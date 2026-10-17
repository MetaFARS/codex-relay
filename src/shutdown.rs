//! Register signals before serving, and keep listening throughout draining.
use anyhow::{bail, Context, Result};
use axum::Router;
use std::future::IntoFuture;
use tokio::{net::TcpListener, sync::oneshot};
use tracing::{info, warn};

pub struct Signals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: tokio::signal::unix::Signal,
    #[cfg(windows)]
    interrupt: tokio::signal::windows::CtrlC,
}

impl Signals {
    pub fn new() -> Result<Self> {
        #[cfg(unix)]
        let interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .context("failed to install SIGINT handler")?;
        #[cfg(unix)]
        let terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .context("failed to install SIGTERM handler")?;
        #[cfg(windows)]
        let interrupt =
            tokio::signal::windows::ctrl_c().context("failed to install Ctrl-C handler")?;
        Ok(Self {
            interrupt,
            #[cfg(unix)]
            terminate,
        })
    }

    async fn recv(&mut self) -> Result<&'static str> {
        #[cfg(unix)]
        {
            tokio::select! {
                signal = self.interrupt.recv() => signal.context("SIGINT listener closed").map(|()| "SIGINT"),
                signal = self.terminate.recv() => signal.context("SIGTERM listener closed").map(|()| "SIGTERM"),
            }
        }
        #[cfg(windows)]
        {
            self.interrupt
                .recv()
                .await
                .context("Ctrl-C listener closed")?;
            Ok("Ctrl-C")
        }
    }
}

pub async fn serve(listener: TcpListener, app: Router, mut signals: Signals) -> Result<()> {
    let (stop, stopped) = oneshot::channel();
    let serving = axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = stopped.await;
        })
        .into_future();
    tokio::pin!(serving);

    let signal = tokio::select! {
        biased;
        result = &mut serving => return result.context("HTTP server failed"),
        signal = signals.recv() => signal?,
    };
    info!(
        signal,
        "shutdown signal received; waiting for active requests"
    );
    let _ = stop.send(());

    // Tokio keeps the OS handlers installed after a receiver is dropped. Keep
    // both receivers alive so a second signal can interrupt a stalled drain.
    tokio::select! {
        biased;
        result = &mut serving => result.context("HTTP server failed while draining"),
        signal = signals.recv() => {
            let signal = signal?;
            warn!(signal, "second shutdown signal received; forcing exit");
            bail!("forced shutdown after second {signal}")
        }
    }
}
