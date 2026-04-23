"""
codex-relay Python shim.

Provides a minimal interface to start/stop the relay process.
The actual binary is installed to PATH by the wheel.
"""

import os
import shutil
import subprocess
from pathlib import Path


def _find_binary() -> Path:
    path = shutil.which("codex-relay")
    if path:
        return Path(path)
    # Fallback: look next to this file (editable / dev install)
    local = Path(__file__).parent / "_bin" / "codex-relay"
    if local.exists():
        return local
    raise FileNotFoundError(
        "codex-relay binary not found. "
        "Install with: pip install codex-relay  or  cargo install codex-relay"
    )


def start(
    port: int = 4444,
    upstream: str = "https://openrouter.ai/api/v1",
    api_key: str = "",
) -> subprocess.Popen:
    """Start codex-relay as a background process and return the Popen handle."""
    env = os.environ.copy()
    if api_key:
        env["CODEX_BRIDGE_API_KEY"] = api_key

    return subprocess.Popen(
        [str(_find_binary()), "--port", str(port), "--upstream", upstream],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


__all__ = ["start"]
