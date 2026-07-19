"""
leakage.py — the single enforcement gate every training path must pass (Item 1).

Detects and BLOCKS:
  * exact cross-split duplicates (same sha256 in >1 split),
  * near-duplicate cross-split leakage (perceptual hash within threshold),
  * conflicting labels among duplicate images (same image, different class),
  * subject/child-level cross-split leakage (same subject_id in >1 split),
    when a `subject_id` column is available.

On any violation, `enforce_leakage_gate` raises LeakageError UNLESS an explicit
override with a written justification is supplied — the override is appended to
an audit log. A clean (or quarantined) manifest is always written.

The same gate is called by feature extraction, embedding extraction, statistical
training, and deep image training, so no path can bypass it.
"""

from __future__ import annotations
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .dataset import _hamming, NEAR_DUP_THRESHOLD, build_manifest


class LeakageError(RuntimeError):
    """Raised when cross-split leakage is detected and not overridden."""


def _load_manifest_rows(manifest_csv: str | Path) -> list[dict]:
    with open(manifest_csv, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assess_leakage(rows: list[dict], subject_key: str = "subject_id",
                   near_dup_threshold: int = NEAR_DUP_THRESHOLD) -> dict:
    """Pure assessment over manifest rows. Returns a structured report."""
    by_sha = defaultdict(list)
    for r in rows:
        if r.get("sha256"):
            by_sha[r["sha256"]].append(r)

    # Exact cross-split duplicates + conflicting labels among duplicates.
    exact_cross_split, conflicting_labels = [], []
    for sha, members in by_sha.items():
        if len(members) < 2:
            continue
        splits = sorted({m["split"] for m in members})
        classes = sorted({m["class"] for m in members})
        if len(splits) > 1:
            exact_cross_split.append({
                "sha256": sha, "image_ids": [m["image_id"] for m in members],
                "splits": splits,
            })
        if len(classes) > 1:
            conflicting_labels.append({
                "sha256": sha, "image_ids": [m["image_id"] for m in members],
                "classes": classes,
            })

    # Near-duplicate cross-split leakage (perceptual hash).
    hashed = [r for r in rows if r.get("phash") and r["phash"] not in
              ("0000000000000000", "ffffffffffffffff")]
    near_cross_split = []
    for i in range(len(hashed)):
        for j in range(i + 1, len(hashed)):
            a, b = hashed[i], hashed[j]
            if a["split"] == b["split"]:
                continue
            dist = _hamming(a["phash"], b["phash"])
            if dist <= near_dup_threshold:
                near_cross_split.append({
                    "image_ids": [a["image_id"], b["image_id"]],
                    "splits": sorted({a["split"], b["split"]}),
                    "hamming": dist,
                })

    # Subject/child-level cross-split leakage (optional subject_id column).
    subject_cross_split = []
    has_subject = any(r.get(subject_key) for r in rows)
    if has_subject:
        by_subject = defaultdict(set)
        subject_ids = defaultdict(list)
        for r in rows:
            sid = r.get(subject_key)
            if sid:
                by_subject[sid].add(r["split"])
                subject_ids[sid].append(r["image_id"])
        for sid, splits in by_subject.items():
            if len(splits) > 1:
                subject_cross_split.append({
                    "subject_id": sid, "splits": sorted(splits),
                    "image_ids": subject_ids[sid],
                })

    leaked_ids = set()
    for group in exact_cross_split + near_cross_split:
        leaked_ids.update(group["image_ids"])
    for group in subject_cross_split:
        leaked_ids.update(group["image_ids"])

    clean = (not exact_cross_split and not near_cross_split
             and not conflicting_labels and not subject_cross_split)
    return {
        "leakage_ok": clean,
        "status": "PASS" if clean else "FAIL_LEAKAGE_DETECTED",
        "exact_cross_split_leakage": exact_cross_split,
        "near_cross_split_leakage": near_cross_split,
        "conflicting_labels": conflicting_labels,
        "subject_cross_split_leakage": subject_cross_split,
        "subject_grouping_available": has_subject,
        "near_dup_threshold": near_dup_threshold,
        "leaked_image_ids": sorted(leaked_ids),
    }


def write_quarantined_manifest(rows: list[dict], leaked_ids: set,
                               output_dir: str | Path) -> dict:
    """Write clean_manifest.csv (leaked rows removed) + quarantine.csv."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["image_id", "path", "split", "class"]
    clean_rows = [r for r in rows if r["image_id"] not in leaked_ids]
    quarantined = [r for r in rows if r["image_id"] in leaked_ids]
    clean_path = out / "clean_manifest.csv"
    quar_path = out / "quarantine.csv"
    for path, subset in ((clean_path, clean_rows), (quar_path, quarantined)):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(subset)
    return {"clean_manifest": str(clean_path), "quarantine_manifest": str(quar_path),
            "clean_rows": len(clean_rows), "quarantined_rows": len(quarantined)}


def _append_audit(audit_log: str | Path, entry: dict) -> None:
    path = Path(audit_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def enforce_leakage_gate(
    source: str | Path, output: str | Path, *,
    subject_key: str = "subject_id",
    allow_override: bool = False, override_justification: str | None = None,
    initiated_by: str = "unspecified",
    timestamp: str | None = None,
) -> dict:
    """The gate. `source` may be a manifest CSV or a dataset root (a manifest is
    built for a root). Raises LeakageError on violation unless overridden with a
    non-empty justification (which is logged). Always writes a clean/quarantined
    manifest and a leakage report.

    `timestamp` is passed in (never generated here) so calls are reproducible.
    """
    source = Path(source)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    if source.is_dir():
        summary = build_manifest(source, out / "manifest.csv")  # noqa: F841
        rows = _load_manifest_rows(out / "manifest.csv")
    else:
        rows = _load_manifest_rows(source)

    report = assess_leakage(rows, subject_key=subject_key)
    quarantine = write_quarantined_manifest(rows, set(report["leaked_image_ids"]), out)
    report["quarantine"] = quarantine
    (out / "leakage_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if report["leakage_ok"]:
        report["gate"] = "passed"
        return report

    # Leakage present.
    if not allow_override:
        raise LeakageError(
            f"Cross-split leakage detected ({report['status']}): "
            f"{len(report['exact_cross_split_leakage'])} exact, "
            f"{len(report['near_cross_split_leakage'])} near, "
            f"{len(report['conflicting_labels'])} conflicting-label, "
            f"{len(report['subject_cross_split_leakage'])} subject-level. "
            f"Training blocked. Use the clean_manifest, or override with an "
            f"explicit written justification."
        )
    if not (override_justification and override_justification.strip()):
        raise LeakageError(
            "Leakage override requires a non-empty --override-justification.")

    entry = {
        "event": "leakage_override",
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "initiated_by": initiated_by,
        "justification": override_justification.strip(),
        "status": report["status"],
        "counts": {
            "exact": len(report["exact_cross_split_leakage"]),
            "near": len(report["near_cross_split_leakage"]),
            "conflicting_labels": len(report["conflicting_labels"]),
            "subject_level": len(report["subject_cross_split_leakage"]),
        },
    }
    _append_audit(out / "leakage_override_audit.jsonl", entry)
    report["gate"] = "overridden"
    report["override"] = entry
    return report
