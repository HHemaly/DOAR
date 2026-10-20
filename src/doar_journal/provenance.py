"""Provenance helpers: file hashes, environment capture, resumable markers.

Every experiment writes a `run_metadata.json` next to its outputs and a
`DONE.json` marker only after all of its outputs exist. Re-running a stage
skips work whose marker is present and whose recorded input hashes still
match, so a Colab disconnect never forces completed work to be repeated and
never silently mixes outputs computed from different inputs.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

DEFAULT_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "scikit-image", "opencv-python-headless",
    "opencv-python", "Pillow", "torch", "torchvision", "open-clip-torch", "timm",
)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def package_versions(packages=DEFAULT_PACKAGES) -> dict[str, str | None]:
    out = {}
    for p in packages:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def git_commit(repo: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def git_dirty(repo: str | Path | None = None) -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True)
        return bool(out.strip())
    except Exception:
        return None


def run_metadata(stage: str, config: dict, inputs: dict[str, str | Path] | None = None,
                 repo: str | Path | None = None) -> dict:
    """Describe a run: stage, config, input-file hashes, code version, environment."""
    input_hashes = {}
    for name, p in (inputs or {}).items():
        p = Path(p)
        input_hashes[name] = {"path": str(p), "sha256": sha256_file(p) if p.is_file() else None}
    return {
        "stage": stage,
        "utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "config_sha256": sha256_json(config),
        "inputs": input_hashes,
        "git_commit": git_commit(repo),
        "git_dirty": git_dirty(repo),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def write_json(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(path)  # atomic on POSIX: a disconnect never leaves half a file
    return path


def is_done(out_dir: str | Path, expected_meta: dict | None = None) -> bool:
    """True when DONE.json exists and (if given) its config/input hashes match."""
    marker = Path(out_dir) / "DONE.json"
    if not marker.exists():
        return False
    if expected_meta is None:
        return True
    done = json.loads(marker.read_text(encoding="utf-8"))
    same_config = done.get("config_sha256") == expected_meta.get("config_sha256")
    same_inputs = ({k: v.get("sha256") for k, v in done.get("inputs", {}).items()}
                   == {k: v.get("sha256") for k, v in expected_meta.get("inputs", {}).items()})
    return same_config and same_inputs


def mark_done(out_dir: str | Path, meta: dict, outputs: list[str | Path]) -> Path:
    out_dir = Path(out_dir)
    missing = [str(p) for p in outputs if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"refusing to mark {out_dir} done; missing outputs: {missing}")
    meta = dict(meta)
    meta["outputs"] = {str(Path(p).relative_to(out_dir) if Path(p).is_relative_to(out_dir) else p):
                       sha256_file(p) for p in outputs if Path(p).is_file()}
    return write_json(out_dir / "DONE.json", meta)
