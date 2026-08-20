#!/usr/bin/env python
"""E6-EXPANDED -- qualitative/error analysis CSV exports (Section 8/D).
Selection criteria are fixed BEFORE inspecting which strategy "wins" --
this script only filters/re-labels rows already computed by run_e6_expanded.py
and compute_stats.py, it never re-derives a trigger decision itself."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation_expanded"
PRIMARY_STRATEGIES = ["R1", "R2", "R3", "R4", "R5"]


def main() -> None:
    per_case = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_case.csv")
    per_domain = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_domain.csv")
    out_dir = EXP_DIR / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # disagreement_cases.csv: any primary strategy disagrees with R4's trigger decision
    wide = per_case[per_case["strategy"].isin(PRIMARY_STRATEGIES)].pivot(
        index="case_id", columns="strategy", values="any_domain_triggered")[PRIMARY_STRATEGIES]
    disagree = wide[wide[["R1", "R2", "R3", "R5"]].any(axis=1)].copy()
    disagree["disagrees_with_R4_on"] = disagree.apply(
        lambda r: ";".join(s for s in ("R1", "R2", "R3", "R5") if r[s] and not r["R4"]), axis=1)
    disagree.to_csv(out_dir / "disagreement_cases.csv")
    print(f"Wrote disagreement_cases.csv ({len(disagree)} rows)")

    # failure_cases.csv: every structurally-violated (case, domain, strategy) row, primary strategies
    fail = per_domain[(per_domain["structural_violation"]) & (per_domain["strategy"].isin(PRIMARY_STRATEGIES))][
        ["case_id", "concern_domain", "strategy", "matched_rule_count", "unique_family_count",
         "duplicate_inflation", "cross_domain_convergence", "ineligible_evidence_used",
         "inappropriate_unverified_use", "ignored_true_contradiction", "disabled_rules_present_but_excluded"]]
    fail.to_csv(out_dir / "failure_cases.csv", index=False)
    print(f"Wrote failure_cases.csv ({len(fail)} rows)")

    # r4_vs_r5_disagreement.csv -- specifically requested comparison
    r45 = wide[wide["R4"] != wide["R5"]]
    r45.to_csv(out_dir / "r4_vs_r5_disagreement.csv")
    print(f"Wrote r4_vs_r5_disagreement.csv ({len(r45)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
