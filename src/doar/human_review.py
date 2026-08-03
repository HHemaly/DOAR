"""Core logic for the Phase 7B duplicate-pair human-review interface.

Deliberately kept free of any UI framework import so it can be unit tested
directly; `phase7b_review_app.py` (Streamlit) is a thin presentation layer
over these functions.

Blinding contract: `blind_item_for_display` is the single choke point that
decides what the reviewer's screen is allowed to see. It must never return
`class_a`/`class_b`/`split_a`/`split_b`/`same_label`/`hamming_ahash`/
`hamming_dhash`/`ai_preliminary_judgment`/`ai_notes`/`source_category`/
`group_id`/`group_size`/`edge_type` -- those exist only in the full
registry record, which the UI layer must not read directly.
"""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HUMAN_JUDGMENT_CHOICES = (
    "definite_duplicate",
    "same_drawing_transformed",
    "different_drawings",
    "uncertain",
)

HUMAN_JUDGMENT_LABELS = {
    "definite_duplicate": "Definite duplicate",
    "same_drawing_transformed": "Same underlying drawing with transformation",
    "different_drawings": "Different drawings",
    "uncertain": "Uncertain",
}

DUPLICATE_LIKE_CHOICES = frozenset({"definite_duplicate", "same_drawing_transformed"})

# Claude's 5-way preliminary vocabulary (used only in blind_audit / boundary_audit
# judgments.json) collapsed onto the human interface's 4-way vocabulary, purely
# for AI-vs-human agreement comparison. "not_duplicate" and "similar_but_different"
# both map to "different_drawings" since the human UI intentionally offers one
# fewer bucket than the AI's original judging vocabulary.
AI_TO_HUMAN_BUCKET = {
    "definite_duplicate": "definite_duplicate",
    "same_drawing_transformed": "same_drawing_transformed",
    "similar_but_different": "different_drawings",
    "not_duplicate": "different_drawings",
    "uncertain": "uncertain",
}

_HIDDEN_KEYS = (
    "class_a", "class_b", "split_a", "split_b", "same_label",
    "hamming_ahash", "hamming_dhash", "ai_preliminary_judgment", "ai_notes",
    "source_category", "group_id", "group_size", "edge_type", "source_pair_id",
    "image_a", "image_b",
)


def wilson_score_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Returns (lo, hi).
    n == 0 returns the maximally uncertain (0.0, 1.0) rather than dividing by zero."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def _atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def load_registry(registry_path: str | Path) -> dict:
    return json.loads(Path(registry_path).read_text(encoding="utf-8"))


def load_decisions(decisions_path: str | Path) -> dict:
    p = Path(decisions_path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_decision(decisions_path: str | Path, item_id: str, decision: str | None,
                   notes: str = "") -> dict:
    """Write one reviewer decision immediately (atomic replace) so progress
    survives a closed browser tab or killed process. decision=None clears
    an existing entry (lets the reviewer un-decide an item)."""
    decisions = load_decisions(decisions_path)
    if decision is None:
        decisions.pop(item_id, None)
    else:
        if decision not in HUMAN_JUDGMENT_CHOICES:
            raise ValueError(f"Unknown decision {decision!r}; must be one of {HUMAN_JUDGMENT_CHOICES}")
        decisions[item_id] = {
            "decision": decision,
            "notes": notes or "",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
    _atomic_write_json(Path(decisions_path), decisions)
    return decisions


def blind_item_for_display(item: dict) -> dict:
    """The only fields a reviewer's screen may receive for an undecided item."""
    return {
        "item_id": item["item_id"],
        "category": item["category"],
        "image_rel_path": item["image_rel_path"],
    }


def progress_by_category(categories_order: dict, decisions: dict) -> dict:
    out = {}
    for cat, ids in categories_order.items():
        reviewed = sum(1 for i in ids if i in decisions)
        out[cat] = {"total": len(ids), "reviewed": reviewed, "remaining": len(ids) - reviewed}
    return out


def compute_reviewer_agreement(registry: dict, decisions: dict) -> dict:
    """AI-vs-human agreement, restricted to items where Claude recorded a
    preliminary judgment (ambiguous + threshold_boundary categories only --
    component_17 and policy_change items were never separately pre-judged
    by the AI) AND the human has since recorded a decision."""
    items = registry["items"]
    confusion: dict[str, dict[str, int]] = {}
    per_category: dict[str, dict[str, int]] = {}
    n_compared = 0
    n_agree = 0
    for item_id, item in items.items():
        ai_judgment = item.get("ai_preliminary_judgment")
        if not ai_judgment or item_id not in decisions:
            continue
        ai_bucket = AI_TO_HUMAN_BUCKET.get(ai_judgment, ai_judgment)
        human_choice = decisions[item_id]["decision"]
        confusion.setdefault(ai_bucket, {}).setdefault(human_choice, 0)
        confusion[ai_bucket][human_choice] += 1
        cat = item["category"]
        pc = per_category.setdefault(cat, {"n": 0, "agree": 0})
        pc["n"] += 1
        n_compared += 1
        if ai_bucket == human_choice:
            pc["agree"] += 1
            n_agree += 1
    overall_rate = (n_agree / n_compared) if n_compared else None
    return {
        "methodology": (
            "AI preliminary judgments (5-way: definite_duplicate, "
            "same_drawing_transformed, similar_but_different, not_duplicate, "
            "uncertain) are collapsed onto the human interface's 4-way "
            "vocabulary before comparison (similar_but_different and "
            "not_duplicate both map to different_drawings). Only items with "
            "both an AI judgment and a recorded human decision are counted. "
            "This is not an inter-rater reliability statistic in the formal "
            "sense (one of the two 'raters' designed the study); it is a "
            "descriptive agreement check only."
        ),
        "n_items_compared": n_compared,
        "overall_agreement_rate": overall_rate,
        "agreement_by_category": {
            cat: {"n": v["n"], "agreement_rate": v["agree"] / v["n"] if v["n"] else None}
            for cat, v in per_category.items()
        },
        "confusion_matrix_ai_bucket_by_human_choice": confusion,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def compute_precision_by_distance(registry: dict, decisions: dict, distance_field: str,
                                   judgment_source: str = "human", min_n_for_confidence: int = 5) -> dict:
    """Precision-by-exact-distance table with Wilson 95% CIs.

    precision at distance d := (# judged duplicate_like) / (# judged at all,
    at that exact distance) -- items judged "uncertain" count in the
    denominator but not the numerator, matching the convention used in the
    prior blinded audits (PHASE7B_DUPLICATE_POLICY.md Section 14).

    judgment_source: "human" uses recorded human decisions; "ai" uses
    Claude's preliminary judgments (collapsed via AI_TO_HUMAN_BUCKET) --
    "ai" is only meant for a provisional, clearly-labeled early look before
    the human review is complete.
    """
    if judgment_source not in ("human", "ai"):
        raise ValueError("judgment_source must be 'human' or 'ai'")
    items = registry["items"]
    buckets: dict[int, list[str]] = {}
    for item_id, item in items.items():
        d = item.get(distance_field)
        if d is None:
            continue
        if judgment_source == "human":
            entry = decisions.get(item_id)
            if entry is None:
                continue
            choice = entry["decision"]
        else:
            ai_judgment = item.get("ai_preliminary_judgment")
            if not ai_judgment:
                continue
            choice = AI_TO_HUMAN_BUCKET.get(ai_judgment, ai_judgment)
        buckets.setdefault(int(d), []).append(choice)

    table = {}
    for d in sorted(buckets):
        choices = buckets[d]
        n = len(choices)
        n_dup = sum(1 for c in choices if c in DUPLICATE_LIKE_CHOICES)
        n_uncertain = sum(1 for c in choices if c == "uncertain")
        lo, hi = wilson_score_interval(n_dup, n)
        table[str(d)] = {
            "n_judged": n,
            "n_duplicate_like": n_dup,
            "n_uncertain": n_uncertain,
            "precision": n_dup / n if n else None,
            "wilson_95ci_lower": lo,
            "wilson_95ci_upper": hi,
            "insufficient_sample": n < min_n_for_confidence,
        }
    return table


def build_threshold_precision_summary(registry: dict, decisions: dict) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "precision = fraction of judged pairs at that exact hash distance "
            "rated definite_duplicate or same_drawing_transformed; small "
            "per-bucket samples mean wide Wilson intervals -- do not read a "
            "single point estimate as settled evidence."
        ),
        "human_precision_by_dhash_distance": compute_precision_by_distance(
            registry, decisions, "hamming_dhash", judgment_source="human"),
        "human_precision_by_ahash_distance": compute_precision_by_distance(
            registry, decisions, "hamming_ahash", judgment_source="human"),
        "provisional_ai_precision_by_dhash_distance": compute_precision_by_distance(
            registry, decisions, "hamming_dhash", judgment_source="ai"),
        "provisional_ai_precision_by_ahash_distance": compute_precision_by_distance(
            registry, decisions, "hamming_ahash", judgment_source="ai"),
    }


def find_unresolved(registry: dict, decisions: dict) -> list[dict]:
    out = []
    for item_id, item in registry["items"].items():
        entry = decisions.get(item_id)
        if entry is None:
            out.append({"item_id": item_id, "category": item["category"],
                        "image_rel_path": item["image_rel_path"], "status": "not_yet_reviewed"})
        elif entry["decision"] == "uncertain":
            out.append({"item_id": item_id, "category": item["category"],
                        "image_rel_path": item["image_rel_path"], "status": "marked_uncertain"})
    return out


def export_pair_reviews_csv(registry: dict, decisions: dict, out_path: str | Path) -> int:
    fields = ["item_id", "category", "source_category", "image_a", "image_b", "class_a",
              "class_b", "same_label", "hamming_ahash", "hamming_dhash",
              "ai_preliminary_judgment", "ai_notes", "human_judgment", "human_notes",
              "reviewed_at"]
    rows = []
    for item_id, item in registry["items"].items():
        entry = decisions.get(item_id, {})
        rows.append({
            "item_id": item_id, "category": item["category"],
            "source_category": item.get("source_category"),
            "image_a": item["image_a"], "image_b": item["image_b"],
            "class_a": item["class_a"], "class_b": item["class_b"],
            "same_label": item["same_label"],
            "hamming_ahash": item.get("hamming_ahash"), "hamming_dhash": item.get("hamming_dhash"),
            "ai_preliminary_judgment": item.get("ai_preliminary_judgment"),
            "ai_notes": item.get("ai_notes"),
            "human_judgment": entry.get("decision", ""), "human_notes": entry.get("notes", ""),
            "reviewed_at": entry.get("reviewed_at", ""),
        })
    rows.sort(key=lambda r: r["item_id"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def export_group_reviews_csv(registry: dict, decisions: dict, out_path: str | Path) -> int:
    """Aggregates pairwise human decisions into two group-level views:
    (1) per-member summary of the 17-image component's internal edges, and
    (2) per-threshold-transition summary of the sampled dataset-wide
    bridging edges -- both derived only from decisions actually recorded,
    never inferred or assumed."""
    items = registry["items"]

    member_edges: dict[str, list[tuple[str, dict | None]]] = {}
    for item_id, item in items.items():
        if item["category"] != "component_17":
            continue
        entry = decisions.get(item_id)
        for member in (item["image_a"], item["image_b"]):
            member_edges.setdefault(member, []).append((item_id, entry))

    rows = []
    for member in sorted(member_edges):
        edges = member_edges[member]
        n_reviewed = sum(1 for _, e in edges if e is not None)
        n_dup = sum(1 for _, e in edges if e is not None and e["decision"] in DUPLICATE_LIKE_CHOICES)
        n_diff = sum(1 for _, e in edges if e is not None and e["decision"] == "different_drawings")
        n_unc = sum(1 for _, e in edges if e is not None and e["decision"] == "uncertain")
        if n_reviewed == 0:
            verdict = "insufficient_data"
        elif n_dup > n_diff and n_dup > n_unc:
            verdict = "duplicate_like_majority"
        elif n_diff >= n_dup and n_diff >= n_unc:
            verdict = "different_majority"
        else:
            verdict = "uncertain_majority"
        rows.append({
            "review_scope": "component_17_member", "group_or_transition_id": member,
            "n_internal_edges": len(edges), "n_reviewed": n_reviewed,
            "n_judged_duplicate_like": n_dup, "n_judged_different": n_diff,
            "n_judged_uncertain": n_unc, "majority_verdict": verdict,
        })

    transitions: dict[str, list[str]] = {}
    for item_id, item in items.items():
        if item["category"] != "policy_change":
            continue
        transitions.setdefault(item["source_category"], []).append(item_id)

    for transition in sorted(transitions):
        ids = transitions[transition]
        decided = [decisions[i]["decision"] for i in ids if i in decisions]
        n_reviewed = len(decided)
        n_dup = sum(1 for d in decided if d in DUPLICATE_LIKE_CHOICES)
        n_diff = sum(1 for d in decided if d == "different_drawings")
        n_unc = sum(1 for d in decided if d == "uncertain")
        lo, hi = wilson_score_interval(n_dup, n_reviewed) if n_reviewed else (0.0, 1.0)
        rows.append({
            "review_scope": "threshold_transition", "group_or_transition_id": transition,
            "n_internal_edges": len(ids), "n_reviewed": n_reviewed,
            "n_judged_duplicate_like": n_dup, "n_judged_different": n_diff,
            "n_judged_uncertain": n_unc,
            "majority_verdict": (
                f"pct_duplicate_like={n_dup / n_reviewed:.1%} "
                f"(95% CI {lo:.1%}-{hi:.1%})" if n_reviewed else "insufficient_data"
            ),
        })

    fields = ["review_scope", "group_or_transition_id", "n_internal_edges", "n_reviewed",
              "n_judged_duplicate_like", "n_judged_different", "n_judged_uncertain", "majority_verdict"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def export_unresolved_csv(registry: dict, decisions: dict, out_path: str | Path) -> int:
    unresolved = find_unresolved(registry, decisions)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["item_id", "category", "image_rel_path", "status"])
        w.writeheader()
        w.writerows(sorted(unresolved, key=lambda r: r["item_id"]))
    return len(unresolved)


def export_all(registry_path: str | Path, decisions_path: str | Path, out_dir: str | Path) -> dict:
    registry = load_registry(registry_path)
    decisions = load_decisions(decisions_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_pairs = export_pair_reviews_csv(registry, decisions, out_dir / "human_pair_reviews.csv")
    n_groups = export_group_reviews_csv(registry, decisions, out_dir / "human_group_reviews.csv")
    n_unresolved = export_unresolved_csv(registry, decisions, out_dir / "unresolved_items.csv")
    agreement = compute_reviewer_agreement(registry, decisions)
    _atomic_write_json(out_dir / "reviewer_agreement_report.json", agreement)
    precision_summary = build_threshold_precision_summary(registry, decisions)
    _atomic_write_json(out_dir / "threshold_precision_summary.json", precision_summary)

    progress = progress_by_category(registry["categories_order"], decisions)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_pair_rows": n_pairs, "n_group_rows": n_groups, "n_unresolved": n_unresolved,
        "progress_by_category": progress,
        "total_items": sum(v["total"] for v in progress.values()),
        "total_reviewed": sum(v["reviewed"] for v in progress.values()),
        "files_written": [
            str(out_dir / "human_pair_reviews.csv"), str(out_dir / "human_group_reviews.csv"),
            str(out_dir / "reviewer_agreement_report.json"),
            str(out_dir / "threshold_precision_summary.json"),
            str(out_dir / "unresolved_items.csv"),
        ],
    }
    return summary
