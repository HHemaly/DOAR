#!/usr/bin/env python
"""E6-EXPANDED -- required figures. Fixed explicit integer seeds only
(Section B.6 fix) -- NEVER `hash()` on a string, which is subject to
per-process PYTHONHASHSEED randomization and is not reproducible across
runs/processes (the exact bug E6-PILOT had in its own F5).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation_expanded"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_STRATEGIES = ["R1", "R2", "R3", "R4", "R5"]
ALL_STRATEGIES = ["R1", "R1_broad", "R2", "R2_broad", "R3", "R4", "R5"]
COLORS = {"R1": "#9e9e9e", "R1_broad": "#616161", "R2": "#e57373", "R2_broad": "#c62828",
          "R3": "#ffb74d", "R4": "#1976d2", "R5": "#66bb6a"}
# Fixed, explicit per-strategy jitter seeds (Section B.6 fix) -- NOT hash().
JITTER_SEEDS = {"R1": 1, "R1_broad": 2, "R2": 3, "R2_broad": 4, "R3": 5, "R4": 6, "R5": 7}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 10,
                      "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / (name + '.png')} and .pdf")


def main() -> None:
    t1 = pd.read_csv(EXP_DIR / "tables" / "E6X_T1_strategy_comparison.csv").set_index("strategy").loc[ALL_STRATEGIES]
    per_domain = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_domain.csv")

    # --- F1: coverage vs structural-risk (primary strategies only) ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    p1 = t1.loc[PRIMARY_STRATEGIES]
    p1[["supported_coverage_pct", "structurally_unsupported_pct"]].to_csv(FIG_DIR / "E6X_F1_coverage_vs_structural_risk.csv")
    for strat in PRIMARY_STRATEGIES:
        ax.scatter(p1.loc[strat, "supported_coverage_pct"], p1.loc[strat, "structurally_unsupported_pct"],
                   s=180, color=COLORS[strat], edgecolor="black", zorder=3)
        ax.annotate(strat, (p1.loc[strat, "supported_coverage_pct"], p1.loc[strat, "structurally_unsupported_pct"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=11, fontweight="bold")
    ax.set_xlabel("Supported coverage (%) -- triggered AND free of objective structural errors")
    ax.set_ylabel("Structurally unsupported convergence (%)")
    ax.set_title("E6X F1 -- Coverage vs. structural risk (corrected, non-circular)")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3, zorder=0)
    save(fig, "E6X_F1_coverage_vs_structural_risk")

    # --- F2: trigger / abstention (all strategies incl. _broad) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ALL_STRATEGIES))
    triggered = t1["triggered_pct"].values
    abstained = t1["abstained_pct"].values
    ax.bar(x, triggered, color=[COLORS[s] for s in ALL_STRATEGIES], label="Triggered")
    ax.bar(x, abstained, bottom=triggered, color="#e0e0e0", label="Abstained")
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_STRATEGIES, rotation=20)
    ax.set_ylabel("% of usable cases")
    ax.set_title("E6X F2 -- Trigger vs. abstention rate by strategy")
    ax.legend(loc="upper right", frameon=False)
    t1[["triggered_pct", "abstained_pct"]].to_csv(FIG_DIR / "E6X_F2_trigger_abstention.csv")
    save(fig, "E6X_F2_trigger_abstention")

    # --- F3: duplicate inflation ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, t1["duplicate_inflation_pct"].values, color=[COLORS[s] for s in ALL_STRATEGIES])
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_STRATEGIES, rotation=20)
    ax.set_ylabel("% of cases with duplicate-family-inflated convergence")
    ax.set_title("E6X F3 -- Duplicate evidence-family inflation by strategy")
    t1[["duplicate_inflation_pct"]].to_csv(FIG_DIR / "E6X_F3_duplicate_inflation.csv")
    save(fig, "E6X_F3_duplicate_inflation")

    # --- F4: cross-domain convergence ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, t1["cross_domain_convergence_pct"].values, color=[COLORS[s] for s in ALL_STRATEGIES])
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_STRATEGIES, rotation=20)
    ax.set_ylabel("% of cases with cross-domain convergence")
    ax.set_title("E6X F4 -- Cross-domain (unsupported) convergence by strategy")
    t1[["cross_domain_convergence_pct"]].to_csv(FIG_DIR / "E6X_F4_cross_domain_convergence.csv")
    save(fig, "E6X_F4_cross_domain_convergence")

    # --- F5: policy disagreement vs R4 (NEW -- separates policy from error) ---
    fig, ax = plt.subplots(figsize=(6, 4))
    p1["policy_disagreement_vs_r4_pct"].plot(kind="bar", ax=ax, color=[COLORS[s] for s in PRIMARY_STRATEGIES])
    ax.set_xticklabels(PRIMARY_STRATEGIES, rotation=0)
    ax.set_ylabel("% of cases where trigger decision differs from R4's\n(policy choice, NOT an error)")
    ax.set_title("E6X F5 -- Policy disagreement vs. R4 (non-circular)")
    p1[["policy_disagreement_vs_r4_pct"]].to_csv(FIG_DIR / "E6X_F5_policy_disagreement.csv")
    save(fig, "E6X_F5_policy_disagreement")

    # --- F6: rules vs independent families (triggered domain-rows, primary strategies) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    trig = per_domain[(per_domain["triggered"]) & (per_domain["strategy"].isin(PRIMARY_STRATEGIES))]
    trig[["case_id", "concern_domain", "strategy", "matched_rule_count", "unique_family_count"]].to_csv(
        FIG_DIR / "E6X_F6_rules_vs_families.csv", index=False)
    for strat in PRIMARY_STRATEGIES:
        sub = trig[trig["strategy"] == strat]
        if len(sub) == 0:
            continue
        rng = np.random.default_rng(JITTER_SEEDS[strat])
        jitter = rng.uniform(-0.08, 0.08, size=len(sub))
        ax.scatter(sub["matched_rule_count"] + jitter, sub["unique_family_count"], color=COLORS[strat],
                   label=f"{strat} (n={len(sub)})", s=70, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Matched (eligible) rule count")
    ax.set_ylabel("Independent evidence-family count")
    ax.set_title("E6X F6 -- Rule count vs. independent family count\n(triggered domain-instances, primary strategies)")
    max_val = max(trig["matched_rule_count"].max(), trig["unique_family_count"].max()) + 1 if len(trig) else 5
    ax.plot([0, max_val], [0, max_val], "--", color="gray", alpha=0.5, label="rules == families (no duplication)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    save(fig, "E6X_F6_rules_vs_families")

    # --- F7: case x strategy heatmap ---
    wide = pd.read_csv(EXP_DIR / "tables" / "E6X_T3_case_comparison.csv", index_col="case_id")[ALL_STRATEGIES].astype(int)
    fig, ax = plt.subplots(figsize=(7, max(4, 0.18 * len(wide))))
    im = ax.imshow(wide.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ALL_STRATEGIES)))
    ax.set_xticklabels(ALL_STRATEGIES, rotation=20)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(wide.index, fontsize=6)
    ax.set_title("E6X F7 -- Per-case trigger heatmap (dark = triggered)")
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="abstained (0) / triggered (1)", shrink=0.6)
    save(fig, "E6X_F7_case_strategy_heatmap")

    # --- F8: domain x strategy heatmap (trigger rate per domain) ---
    dom = per_domain[per_domain["strategy"].isin(PRIMARY_STRATEGIES)]
    pivot = dom.pivot_table(index="concern_domain", columns="strategy", values="triggered", aggfunc="mean")[PRIMARY_STRATEGIES]
    pivot.to_csv(FIG_DIR / "E6X_F8_domain_strategy_heatmap.csv")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.4 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(PRIMARY_STRATEGIES)))
    ax.set_xticklabels(PRIMARY_STRATEGIES)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("E6X F8 -- Trigger rate by concern domain x strategy")
    fig.colorbar(im, ax=ax, label="fraction of that domain's cases triggered", shrink=0.7)
    save(fig, "E6X_F8_domain_strategy_heatmap")

    # --- F9: same-domain convergence examples (conditional -- only if any exist) ---
    same_domain_examples = per_domain[(per_domain["strategy"] == "R4") & (per_domain["triggered"])]
    if len(same_domain_examples) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(same_domain_examples)), same_domain_examples["unique_family_count"],
               color=COLORS["R4"])
        ax.set_xticks(range(len(same_domain_examples)))
        ax.set_xticklabels(same_domain_examples["case_id"] + "\n" + same_domain_examples["concern_domain"],
                            rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Independent evidence families (same domain)")
        ax.set_title("E6X F9 -- Same-domain multi-family convergence examples (R4 triggers)")
        same_domain_examples[["case_id", "concern_domain", "unique_family_count", "unique_evidence_families"]].to_csv(
            FIG_DIR / "E6X_F9_same_domain_convergence_examples.csv", index=False)
        save(fig, "E6X_F9_same_domain_convergence_examples")
    else:
        print("F9 SKIPPED: zero R4-triggered cases exist in this cohort -- no same-domain convergence example "
              "to plot. This is reported explicitly, not silently omitted -- see KNOWN_LIMITATIONS.md.")
        (FIG_DIR / "E6X_F9_SKIPPED_no_r4_triggers.txt").write_text(
            "F9 was not generated: 0 cases in this cohort reach R4's own same-domain, >=2-independent-family "
            "trigger condition. See tables/E6X_T2_cohort_diagnostics.csv "
            "('cases_with_ge2_independent_families_SAME_domain') and KNOWN_LIMITATIONS.md.\n",
            encoding="utf-8")

    print("\nAll figures written.")


if __name__ == "__main__":
    main()
