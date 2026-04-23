#!/usr/bin/env python3
"""Guardrail runner for codex-bridge (Rust project): cargo check, clippy, test.

Exit 0 → all pass (prints JSON systemMessage).
Exit 0 → failures in informational mode (prints JSON systemMessage warning).
With --block: prints JSON {"continue": false} to block the calling tool.
"""

import json
import os
import subprocess
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

block = "--block" in sys.argv

checks = [
    ("check", ["cargo", "check"]),
    ("clippy", ["cargo", "clippy", "--", "-D", "warnings"]),
    ("test", ["cargo", "test", "--quiet"]),
]

fails: list[str] = []
details: list[str] = []
for name, cmd in checks:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        fails.append(name)
        out = "\n".join((r.stdout + r.stderr).strip().split("\n")[-10:])
        details.append(f"[{name}]\n{out}")

if not fails:
    print(json.dumps({"systemMessage": "✓ guardrails passed: check · clippy · test"}))
elif block:
    msg = (
        "Commit blocked — fix guardrails first: " + ", ".join(fails) + "\n\n" + "\n\n".join(details)
    )
    print(json.dumps({"continue": False, "stopReason": msg, "systemMessage": msg}))
else:
    print(json.dumps({"systemMessage": "⚠ guardrails: " + ", ".join(f + " FAIL" for f in fails)}))
