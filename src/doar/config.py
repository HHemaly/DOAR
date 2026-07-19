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


def save_resolved(output: str | Path, command: str, values: dict) -> str:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"command": command, "resolved": values}
    canonical = json.dumps(payload, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    payload["configuration_sha256"] = digest
    (directory / "resolved_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return digest
