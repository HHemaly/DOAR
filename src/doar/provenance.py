"""
provenance.py — artifact provenance + cross-artifact verification (B5).

Feature and embedding artifacts record where they came from (manifest hash,
leakage report hash + status, ordered sample-ID hash, class order, etc.). Before
feature-model / embedding-comparison / fusion training, verify_artifacts()
confirms the artifacts came from the SAME manifest with matching sample-ID sets
and acceptable leakage status — rejecting mismatches instead of silently
intersecting. An override is permitted only through a written, audit-logged reason.
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ACCEPTABLE_LEAKAGE = {"PASS", "OVERRIDDEN"}


def sha256_file(path):
    p = Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


def sample_id_hash(sample_ids) -> str:
    """Deterministic ordered hash of a sample-ID sequence."""
    joined = "\n".join(str(s) for s in sample_ids).encode()
    return hashlib.sha256(joined).hexdigest()


def _leakage_report_near(manifest_path):
    """Locate the leakage report a manifest was gated with, if present."""
    p = Path(manifest_path)
    for cand in (p.parent / "leakage_gate" / "leakage_report.json",
                 p.with_suffix(".summary.json")):
        if cand.exists():
            return cand
    return None


def build_feature_provenance(manifest_path, sample_ids, class_order,
                             extraction_config_hash, feature_schema_version) -> dict:
    report = _leakage_report_near(manifest_path)
    status = "UNKNOWN"
    if report:
        try:
            data = json.loads(Path(report).read_text(encoding="utf-8"))
            status = data.get("leakage_status") or ("PASS" if data.get("leakage_ok") else "FAIL")
        except Exception:
            status = "UNKNOWN"
    return {
        "artifact": "objective_features",
        "manifest_sha256": sha256_file(manifest_path),
        "leakage_report_sha256": sha256_file(report) if report else None,
        "leakage_status": status,
        "extraction_config_hash": extraction_config_hash,
        "feature_schema_version": feature_schema_version,
        "sample_id_hash": sample_id_hash(sample_ids),
        "sample_count": len(sample_ids),
        "class_order": list(class_order),
    }


def build_embedding_provenance(manifest_path, sample_ids, class_order,
                               extraction_config_hash, backbone, revision,
                               checkpoint_hash, preprocessing_hash,
                               embedding_dimension) -> dict:
    report = _leakage_report_near(manifest_path)
    status = "UNKNOWN"
    if report:
        try:
            data = json.loads(Path(report).read_text(encoding="utf-8"))
            status = data.get("leakage_status") or ("PASS" if data.get("leakage_ok") else "FAIL")
        except Exception:
            status = "UNKNOWN"
    return {
        "artifact": "embeddings",
        "manifest_sha256": sha256_file(manifest_path),
        "leakage_report_sha256": sha256_file(report) if report else None,
        "leakage_status": status,
        "extraction_config_hash": extraction_config_hash,
        "sample_id_hash": sample_id_hash(sample_ids),
        "sample_count": len(sample_ids),
        "backbone": backbone, "revision": revision,
        "checkpoint_hash": checkpoint_hash,
        "preprocessing_hash": preprocessing_hash,
        "embedding_dimension": embedding_dimension,
        "class_order": list(class_order),
    }


class ProvenanceError(RuntimeError):
    pass


def verify_artifacts(feature_prov: dict, embedding_prov: dict, *,
                     allow_override: bool = False, override_justification: str | None = None,
                     audit_dir=None, timestamp: str | None = None) -> dict:
    """Verify two artifacts are compatible for training (B5). Raises unless clean
    or overridden with a written, audit-logged justification."""
    issues = []
    if not feature_prov or not embedding_prov:
        issues.append("missing provenance on one or both artifacts")
    else:
        if feature_prov.get("manifest_sha256") != embedding_prov.get("manifest_sha256"):
            issues.append("features and embeddings came from DIFFERENT manifests")
        if feature_prov.get("sample_id_hash") != embedding_prov.get("sample_id_hash"):
            issues.append("sample-ID sets differ between features and embeddings")
        if feature_prov.get("class_order") != embedding_prov.get("class_order"):
            issues.append("class order differs between artifacts")
        for prov in (feature_prov, embedding_prov):
            if prov.get("leakage_status") not in ACCEPTABLE_LEAKAGE:
                issues.append(f"{prov.get('artifact')} leakage_status "
                              f"{prov.get('leakage_status')!r} is not acceptable")

    if not issues:
        return {"ok": True, "issues": []}

    if not allow_override:
        raise ProvenanceError("Artifact provenance verification failed: " + "; ".join(issues))
    if not (override_justification and override_justification.strip()):
        raise ProvenanceError("Provenance override requires a written justification.")
    event = {
        "event": "provenance_override",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "justification": override_justification.strip(),
        "issues": issues,
    }
    if audit_dir:
        out = Path(audit_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "provenance_override_audit.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    return {"ok": True, "overridden": True, "issues": issues}
