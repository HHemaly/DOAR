"""Real trigger-distribution audit (DOAR-TRACE Phase 2A, Section 10) for
every rule in `rules_registry_v2.json`, on a real, non-test,
class-balanced sample. Trigger counts here are **not** interpreted as
psychological prevalence -- they measure how often this specific
implementation's threshold fires on this specific dataset sample under
this specific segmentation/feature pipeline, nothing more. Only the 10
currently-executable rules (`allowed_output_level == individual_heuristic_only`)
are actually run through the pipeline; every other (disabled) rule is
listed with `executable=False` and no counts, since it has no evaluator
to trigger at all.
"""

from __future__ import annotations

import csv
import random
import tempfile
from pathlib import Path
from typing import Any

from .analysis import analyze_image
from .registry_v2_build import build_registry_v2

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "outputs" / "phase5" / "manifest.csv"
CSV_PATH = ROOT / "artifacts" / "phase2a" / "rule_trigger_distribution.csv"

CSV_FIELDS = [
    "rule_id", "observable", "executable", "trigger_count", "assessable_count",
    "not_assessable_count", "trigger_rate_among_assessable_pct", "source_evidence_grade", "limitations",
]


def _sample_balanced(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = {}
    for row in rows:
        if row["split"] == "test":
            continue
        by_class.setdefault(row["class"], []).append(row)
    sample = []
    for cls, items in sorted(by_class.items()):
        sample.extend(items if len(items) <= per_class else rng.sample(items, per_class))
    return sample


def run_trigger_distribution(
    manifest_path: Path = DEFAULT_MANIFEST, per_class: int = 50, seed: int = 7, output_path: Path = CSV_PATH,
) -> dict[str, Any]:
    """`output_path` defaults to the canonical committed artifact --
    override it (e.g. in tests using a small `per_class`) so a reduced
    sample never clobbers the real, full-sample trigger distribution."""
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    sample = _sample_balanced(rows, per_class=per_class, seed=seed)

    registry_v2 = build_registry_v2()
    rules_by_id = {r["rule_id"]: r for r in registry_v2["rules"]}
    executable_ids = {rid for rid, r in rules_by_id.items() if r["allowed_output_level"] == "individual_heuristic_only"}

    counts = {rid: {"trigger": 0, "assessable": 0, "not_assessable": 0, "errors": 0} for rid in executable_ids}
    n_sampled_ok = 0
    for row in sample:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = analyze_image(row["path"], tmp)
            except Exception:  # a real per-image failure, honestly recorded, not silently skipped
                for rid in executable_ids:
                    counts[rid]["errors"] += 1
                continue
        n_sampled_ok += 1
        by_id = {r["rule_id"]: r for r in result.to_dict()["rule_evaluations"]}
        for rid in executable_ids:
            r = by_id.get(rid)
            if r is None:
                continue
            if r["status"] == "not_assessable":
                counts[rid]["not_assessable"] += 1
            else:
                counts[rid]["assessable"] += 1
                if r["status"] == "weak_support":
                    counts[rid]["trigger"] += 1

    out_rows = []
    for rule in registry_v2["rules"]:
        rid = rule["rule_id"]
        limitations = "; ".join(rule["limitations"]) or "none recorded"
        grade = ", ".join(rule["evidence_level_as_written"] or []) if rule["evidence_level_as_written"] else "not graded by source"
        if rid in executable_ids:
            c = counts[rid]
            rate = round(100 * c["trigger"] / max(1, c["assessable"]), 2)
            out_rows.append({
                "rule_id": rid, "observable": rule["observable"], "executable": True,
                "trigger_count": c["trigger"], "assessable_count": c["assessable"],
                "not_assessable_count": c["not_assessable"], "trigger_rate_among_assessable_pct": rate,
                "source_evidence_grade": grade, "limitations": limitations,
            })
        else:
            out_rows.append({
                "rule_id": rid, "observable": rule["observable"], "executable": False,
                "trigger_count": None, "assessable_count": None, "not_assessable_count": None,
                "trigger_rate_among_assessable_pct": None,
                "source_evidence_grade": grade,
                "limitations": limitations + " (allowed_output_level=disabled -- no evaluator wired, never run)",
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for out_row in out_rows:
            writer.writerow(out_row)

    return {
        "n_sampled": len(sample), "n_sampled_ok": n_sampled_ok, "n_rules": len(out_rows),
        "n_executable_rules": len(executable_ids), "output_path": str(output_path),
    }


if __name__ == "__main__":
    print(run_trigger_distribution())
