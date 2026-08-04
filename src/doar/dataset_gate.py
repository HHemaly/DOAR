"""
dataset_gate.py -- the code-enforced "clean duplicate-controlled split"
gate for Stage 0 model experiments.

Mirrors test_guard.py's philosophy (a real, importable check every caller
must pass, not just a documentation promise): `check_clean_split_gate()`
inspects the actual files on disk and returns a structured pass/fail
report with an exact remediation list. `require_clean_split_for_full_run()`
raises `CleanSplitGateFailed` when the gate is not satisfied, so nothing
in this codebase can accidentally run "full" (non-smoke) Stage 0 training
against an unapproved split.

The eight conditions checked correspond 1:1 to the gate specified for this
task's Stage 0 experiment programme:
  1. Human Phase 7B duplicate review has recorded decisions.
  2. Human pair/group decisions have been exported.
  3. An explicit, human-authored duplicate-detection policy approval
     record exists (this module never creates or infers one).
  4. A frozen, duplicate-controlled partition manifest exists (an explicit
     freeze marker, not just any partition_manifest.csv on disk --
     outputs/phase7b/final_partition/ existing is NOT sufficient, since it
     is explicitly provisional -- see PHASE7B_DUPLICATE_POLICY.md).
  5. No accepted duplicate group crosses train/valid/test (read from the
     frozen partition's own leakage_verification_report.json -- this
     module reads that real, already-computed verification, it does not
     recompute it).
  6. Previously-exposed images are excluded from valid/test (same source).
  7. Class counts and image paths verified (same source).
  8. The final test set has not been unlocked for this manifest (checked
     via the shared test_guard.py audit log).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GATE_VERSION = "doar_dataset_gate_v1"

# The canonical location a human-approved policy record must exist at.
# This module never writes this file -- only a human (or a session
# explicitly instructed to record the user's approval) may create it.
APPROVAL_RECORD_PATH = "outputs/phase7b/APPROVED_POLICY.json"
REQUIRED_APPROVAL_FIELDS = (
    "approved_by", "approved_at", "near_dup_threshold", "hash_field",
    "clustering_method", "notes",
)

# The canonical location the frozen manifest's own freeze marker must
# exist at, alongside outputs/phase7b/final_partition/partition_manifest.csv.
FREEZE_MARKER_PATH = "outputs/phase7b/final_partition/FROZEN.json"
REQUIRED_FREEZE_FIELDS = ("frozen_by", "frozen_at", "approval_record", "manifest_sha256")


class CleanSplitGateFailed(RuntimeError):
    pass


def _check(passed: bool, detail: str) -> dict:
    return {"pass": bool(passed), "detail": detail}


def check_clean_split_gate(repo_root: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_root)
    checks: dict[str, dict] = {}

    decisions_path = root / "outputs/phase7b/human_review/app_data/decisions.json"
    registry_path = root / "outputs/phase7b/human_review/app_data/items_registry.json"
    if decisions_path.exists():
        try:
            decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
        except Exception:
            decisions = {}
        n_decided = len(decisions)
        n_total = None
        if registry_path.exists():
            try:
                registry = json.loads(registry_path.read_text(encoding="utf-8"))
                n_total = sum(len(v) for v in registry.get("categories_order", {}).values())
            except Exception:
                n_total = None
        complete = n_total is not None and n_decided >= n_total
        checks["human_review_decisions_recorded"] = _check(
            complete,
            f"{n_decided} decisions recorded" + (f" of {n_total} items" if n_total else "")
            + ("" if complete else " -- review is not yet complete"),
        )
    else:
        checks["human_review_decisions_recorded"] = _check(
            False, f"{decisions_path} does not exist -- no human review has been recorded yet")

    exports_dir = root / "outputs/phase7b/human_review/exports"
    required_exports = ("human_pair_reviews.csv", "human_group_reviews.csv",
                        "reviewer_agreement_report.json", "threshold_precision_summary.json",
                        "unresolved_items.csv")
    missing_exports = [f for f in required_exports if not (exports_dir / f).exists()]
    checks["human_review_exported"] = _check(
        not missing_exports,
        "all export files present" if not missing_exports
        else f"missing: {', '.join(missing_exports)} under {exports_dir}",
    )

    approval_path = root / APPROVAL_RECORD_PATH
    if approval_path.exists():
        try:
            approval = json.loads(approval_path.read_text(encoding="utf-8"))
        except Exception:
            approval = {}
        missing_fields = [f for f in REQUIRED_APPROVAL_FIELDS if not approval.get(f)]
        checks["duplicate_policy_approved"] = _check(
            not missing_fields,
            "approval record present and complete" if not missing_fields
            else f"{approval_path} exists but missing fields: {', '.join(missing_fields)}",
        )
    else:
        checks["duplicate_policy_approved"] = _check(
            False, f"{approval_path} does not exist -- no human has approved a "
                   f"near-dup threshold/clustering policy")

    freeze_path = root / FREEZE_MARKER_PATH
    manifest_path = root / "outputs/phase7b/final_partition/partition_manifest.csv"
    if freeze_path.exists() and manifest_path.exists():
        try:
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        except Exception:
            freeze = {}
        missing_fields = [f for f in REQUIRED_FREEZE_FIELDS if not freeze.get(f)]
        checks["manifest_frozen"] = _check(
            not missing_fields,
            "freeze marker present and complete" if not missing_fields
            else f"{freeze_path} exists but missing fields: {', '.join(missing_fields)}",
        )
    else:
        checks["manifest_frozen"] = _check(
            False, f"{freeze_path} does not exist -- "
                   f"outputs/phase7b/final_partition/ is explicitly provisional "
                   f"(PHASE7B_DUPLICATE_POLICY.md) and must not be treated as frozen")

    verification_path = root / "outputs/phase7b/final_partition/leakage_verification_report.json"
    if verification_path.exists():
        try:
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
        except Exception:
            verification = {}
        group_ok = verification.get("no_group_crosses_partitions", {}).get("ok")
        exposed_ok = verification.get("no_exposed_group_in_valid_or_test", {}).get("ok")
        counts_ok = verification.get("counts_match_manifest", {}).get("ok")
        checks["no_duplicate_group_crosses_splits"] = _check(
            bool(group_ok), "ok=true in leakage_verification_report.json" if group_ok
            else "not verified ok=true in leakage_verification_report.json")
        checks["exposed_images_excluded_from_valid_test"] = _check(
            bool(exposed_ok), "ok=true in leakage_verification_report.json" if exposed_ok
            else "not verified ok=true in leakage_verification_report.json")
        checks["class_counts_and_paths_verified"] = _check(
            bool(counts_ok), "ok=true in leakage_verification_report.json" if counts_ok
            else "not verified ok=true in leakage_verification_report.json")
    else:
        msg = f"{verification_path} does not exist -- cannot verify group/exposure/count integrity"
        checks["no_duplicate_group_crosses_splits"] = _check(False, msg)
        checks["exposed_images_excluded_from_valid_test"] = _check(False, msg)
        checks["class_counts_and_paths_verified"] = _check(False, msg)

    unlock_log = root / "outputs/phase7b/final_partition/final_test_unlock_log.jsonl"
    checks["final_test_set_untouched"] = _check(
        not unlock_log.exists() or unlock_log.stat().st_size == 0,
        "no test-unlock events recorded for this manifest" if not unlock_log.exists()
        or unlock_log.stat().st_size == 0
        else f"{unlock_log} contains unlock events -- the test set has been accessed",
    )

    gate_passed = all(c["pass"] for c in checks.values())
    remediation = [] if gate_passed else [
        "1. Launch the review app: .\\.venv\\Scripts\\Activate.ps1 ; "
        "python -m streamlit run phase7b_review_app.py",
        "2. Complete the human review of all 225 pairs across the 4 tabs "
        "(outputs/phase7b/human_review/app_data/decisions.json will be populated).",
        "3. Click Export in the app sidebar to write "
        "outputs/phase7b/human_review/exports/.",
        f"4. A human reviews the exported evidence and records an explicit decision "
        f"in {APPROVAL_RECORD_PATH} with fields {REQUIRED_APPROVAL_FIELDS}.",
        "5. Re-run `python main.py build-partition` with the approved threshold/"
        "hash-field/clustering method to (re)generate outputs/phase7b/final_partition/.",
        f"6. A human (or an explicitly-instructed session) writes "
        f"{FREEZE_MARKER_PATH} with fields {REQUIRED_FREEZE_FIELDS} referencing the "
        f"approval record and the manifest's own SHA-256, only after re-confirming "
        f"leakage_verification_report.json shows ok=true for every check.",
        "7. Re-run check_clean_split_gate() -- Stage 0 full training may only "
        "proceed once every check above reports pass=true.",
    ]
    return {
        "gate_version": GATE_VERSION, "gate_passed": gate_passed,
        "checks": checks, "remediation": remediation,
    }


def require_clean_split_for_full_run(repo_root: str | Path = ".") -> dict:
    """Raise CleanSplitGateFailed unless every gate condition passes. Callers
    that only need a SMOKE run (synthetic/tiny data, no real training
    decision) should not call this -- smoke mode is always permitted."""
    report = check_clean_split_gate(repo_root)
    if not report["gate_passed"]:
        failed = [name for name, c in report["checks"].items() if not c["pass"]]
        raise CleanSplitGateFailed(
            "Clean-split gate failed; full Stage 0 training is blocked. "
            f"Failed checks: {', '.join(failed)}. "
            "See check_clean_split_gate()['remediation'] for exact steps, or run in --smoke mode."
        )
    return report
