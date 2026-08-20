#!/usr/bin/env python
"""E6X_F10 -- before/after eligibility-fix impact figure. Two panels:
(1) per-case matched-rule count before vs after the fix, sorted by
disabled-rules-removed descending; (2) headline summary counts. Source
data: raw/eligibility_before_after_per_case.csv (no live computation, no
randomness -- deterministic sort order only, no seed needed)."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = EXP_DIR / "raw" / "eligibility_before_after_per_case.csv"
SUMMARY_PATH = EXP_DIR / "eligibility_fix" / "IMPACT_SUMMARY.csv"
OUT_STEM = EXP_DIR / "figures" / "E6X_F10_before_after_disabled_rule_impact"


def main():
    with open(RAW_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(SUMMARY_PATH, encoding="utf-8") as f:
        summary = next(csv.DictReader(f))

    rows_sorted = sorted(rows, key=lambda r: (-int(r["n_disabled_removed"]), r["case_id"]))
    case_ids = [r["case_id"] for r in rows_sorted]
    before = [int(r["n_matched_before"]) for r in rows_sorted]
    after = [int(r["n_matched_after"]) for r in rows_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [2.2, 1]})

    x = range(len(case_ids))
    width = 0.4
    ax1.bar([i - width / 2 for i in x], before, width, label="Before fix (disabled rules included)", color="#d9534f")
    ax1.bar([i + width / 2 for i in x], after, width, label="After fix (eligible only)", color="#5cb85c")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(case_ids, rotation=90, fontsize=6)
    ax1.set_ylabel("Matched rules (semantic + deterministic)")
    ax1.set_title("Per-case matched-rule count, before vs after the eligibility fix\n"
                   "(sorted by disabled rules removed, descending)")
    ax1.legend()
    ax1.set_ylim(0, max(before) + 1)

    labels = ["Cases\naffected", "Disabled matches\nremoved (total)", "Synthesis level\nchanged",
              "Candidate hypotheses\nchanged"]
    values = [int(summary["n_cases_affected_by_fix"]), int(summary["n_disabled_matches_removed_total"]),
              int(summary["n_cases_overall_synthesis_level_changed"]), int(summary["n_cases_candidate_hypotheses_changed"])]
    bars = ax2.bar(labels, values, color=["#f0ad4e", "#d9534f", "#5bc0de", "#292b2c"])
    ax2.set_title(f"Headline impact (N={summary['n_cases_total']} cases)")
    ax2.set_ylabel("Count")
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.3, str(v), ha="center", fontsize=10)
    ax2.set_ylim(0, max(values) + 3)

    fig.tight_layout()
    fig.savefig(OUT_STEM.with_suffix(".png"), dpi=200)
    fig.savefig(OUT_STEM.with_suffix(".pdf"))
    plt.close(fig)

    with open(OUT_STEM.with_suffix(".csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "n_matched_before", "n_matched_after", "n_disabled_removed"])
        writer.writeheader()
        for r in rows_sorted:
            writer.writerow({"case_id": r["case_id"], "n_matched_before": r["n_matched_before"],
                              "n_matched_after": r["n_matched_after"], "n_disabled_removed": r["n_disabled_removed"]})

    print(f"Wrote {OUT_STEM}.png/.pdf/.csv")


if __name__ == "__main__":
    main()
