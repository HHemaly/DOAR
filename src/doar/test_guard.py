"""
test_guard.py — one shared final-test access guard (B3).

Every command that reads, exports, fuses or evaluates the test split must call
require_test_access(). Test access requires ALL of:
    --unlock-test  AND  --confirm-final-evaluation  AND  --initiated-by <name>
Missing any of these (including an empty --initiated-by) is refused. Each granted
access writes an append-only audit event to final_test_unlock_log.jsonl.
"""

from __future__ import annotations
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GUARD_VERSION = "doar_test_guard_v1"


class TestAccessDenied(RuntimeError):
    pass


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _dirty():
    try:
        return bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return None


def require_test_access(split: str, *, unlock_test: bool, confirm_final_evaluation: bool,
                        initiated_by: str | None, command: str,
                        audit_dir, timestamp: str | None = None,
                        **audit_fields) -> dict | None:
    """Enforce the guard for a split. Returns the audit event when test access is
    granted, or None for non-test splits. Raises TestAccessDenied otherwise."""
    if split != "test":
        return None
    missing = []
    if not unlock_test:
        missing.append("--unlock-test")
    if not confirm_final_evaluation:
        missing.append("--confirm-final-evaluation")
    if not (initiated_by and str(initiated_by).strip()):
        missing.append("--initiated-by <name>")
    if missing:
        raise TestAccessDenied(
            f"Test-split access for '{command}' is locked. Provide: {', '.join(missing)}. "
            f"The test split must never be used for selection, calibration, weighting or "
            f"stacking.")
    event = {
        "guard_version": GUARD_VERSION,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "command": command,
        "initiated_by": str(initiated_by).strip(),
        "split": split,
        "git_commit": _git("rev-parse", "HEAD"),
        "repo_dirty": _dirty(),
        "confirmation_flags": ["--unlock-test", "--confirm-final-evaluation", "--initiated-by"],
        **audit_fields,
    }
    out = Path(audit_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "final_test_unlock_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return event
