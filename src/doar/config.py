from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path


def load_config(path: str | Path | None, allowed: dict[str, set[str]]) -> dict:
    if not path:
        return {}
    source = Path(path)
    data = tomllib.loads(source.read_text(encoding="utf-8"))
    unknown_sections = set(data) - set(allowed)
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown_sections)}")
    for section, values in data.items():
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section {section!r} must be a table")
        unknown = set(values) - allowed[section]
        if unknown:
            raise ValueError(f"Unknown fields in [{section}]: {sorted(unknown)}")
    return data


def resolve(cli: dict, config: dict, mapping: dict[str, tuple[str, str]], supplied: set[str]) -> dict:
    result = {}
    for destination, (section, field) in mapping.items():
        value = cli.get(destination)
        if destination not in supplied and section in config and field in config[section]:
            value = config[section][field]
        result[destination] = value
    return result


def assert_all_config_consumed(config: dict, mapping: dict[str, tuple[str, str]]) -> None:
    """Fail if any accepted TOML field is never mapped to a destination (Item 2).
    A field accepted by load_config but not wired to code is a silent no-op and
    must be an error, not ignored."""
    mapped = {(section, field) for (section, field) in mapping.values()}
    unused = []
    for section, values in config.items():
        for field in values:
            if (section, field) not in mapped:
                unused.append(f"[{section}] {field}")
    if unused:
        raise ValueError(
            "Configuration fields were accepted but never used (would be silently "
            f"ignored): {sorted(unused)}. Wire them or remove them from the config.")


def environment_metadata() -> dict:
    """Dependency, CUDA, GPU and git versions for reproducibility (Item 2)."""
    import platform
    import subprocess
    import sys

    def _ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return "not_installed"

    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _ver("numpy"), "torch": _ver("torch"),
        "torchvision": _ver("torchvision"), "sklearn": _ver("sklearn"),
        "cuda_available": False, "gpu_name": None, "cuda_version": None,
    }
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info["git_commit"] = "unknown"
    return info


def save_run_metadata(output: str | Path, command: str, requested: dict,
                      resolved: dict) -> str:
    """Save requested + resolved config, config hash, and environment (Item 2).
    Returns the config hash. `save_resolved` remains for backward compatibility."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    digest = save_resolved(directory, command, resolved)
    (directory / "requested_config.json").write_text(
        json.dumps({"command": command, "requested": requested}, indent=2, default=str),
        encoding="utf-8")
    (directory / "environment.json").write_text(
        json.dumps(environment_metadata(), indent=2), encoding="utf-8")
    return digest


def save_resolved(output: str | Path, command: str, values: dict) -> str:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"command": command, "resolved": values}
    canonical = json.dumps(payload, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    payload["configuration_sha256"] = digest
    (directory / "resolved_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return digest
