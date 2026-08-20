#!/usr/bin/env python
"""E6 -- required figures F1-F6. Each figure's exact source data is
either one of the already-saved tables/ CSVs or a small derived CSV
saved alongside the figure under figures/ (never inline-only numbers).
No decorative elements -- every figure answers one specific part of the
E6 research question, stated in its own title and in
FIGURE_TABLE_REGISTER.md.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["R1", "R2", "R3", "R4", "R5"]
COLORS = {"R1": "#9e9e9e", "R2": "#e57373", "R3": "#ffb74d", "R4": "#1976d2", "R5": "#66bb6a"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name: str):
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG_DIR / (name + '.png')} and .pdf")


def main() -> None:
    t1 = pd.read_csv(EXP_DIR / "tables" / "E6_T1_strategy_comparison.csv").set_index("strategy").loc[STRATEGIES]
    per_domain = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_domain.csv")

    # --- F1: coverage vs overclaiming ------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    src = t1[["triggered_pct", "supported_coverage_pct", "structurally_unsupported_pct"]].copy()
    src.to_csv(FIG_DIR / "E6_F1_coverage_vs_overclaiming.csv")
    for strat in STRATEGIES:
        ax.scatter(t1.loc[strat, "supported_coverage_pct"], t1.loc[strat, "structurally_unsupported_pct"],
                   s=180, color=COLORS[strat], edgecolor="black", zorder=3)
        ax.annotate(strat, (t1.loc[strat, "supported_coverage_pct"], t1.loc[strat, "structurally_unsupported_pct"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=11, fontweight="bold")
    ax.set_xlabel("Supported coverage (%) -- triggered AND structurally clean")
    ax.set_ylabel("Structurally unsupported convergence (%)")
    ax.set_title("E6 F1 -- Coverage vs. overclaiming trade-off (N=15 cases)")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3, zorder=0)
    save(fig, "E6_F1_coverage_vs_overclaiming")

    # --- F2: trigger / abstention ------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(STRATEGIES))
    triggered = t1["triggered_pct"].values
    abstained = t1["abstained_pct"].values
    ax.bar(x, triggered, color=[COLORS[s] for s in STRATEGIES], label="Triggered")
    ax.bar(x, abstained, bottom=triggered, color="#e0e0e0", label="Abstained")
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("% of 15 cases")
    ax.set_title("E6 F2 -- Trigger vs. abstention rate by strategy")
    ax.legend(loc="upper right", frameon=False)
    t1[["triggered_pct", "abstained_pct"]].to_csv(FIG_DIR / "E6_F2_trigger_abstention.csv")
    save(fig, "E6_F2_trigger_abstention")

    # --- F3: duplicate inflation --------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, t1["duplicate_inflation_pct"].values, color=[COLORS[s] for s in STRATEGIES])
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("% of 15 cases with duplicate-family-inflated convergence")
    ax.set_title("E6 F3 -- Duplicate evidence-family inflation by strategy")
    t1[["duplicate_inflation_pct"]].to_csv(FIG_DIR / "E6_F3_duplicate_inflation.csv")
    save(fig, "E6_F3_duplicate_inflation")

    # --- F4: cross-domain convergence ---------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, t1["cross_domain_convergence_pct"].values, color=[COLORS[s] for s in STRATEGIES])
    ax.set_xticks(x)
    ax.set_xticklabels(STRATEGIES)
    ax.set_ylabel("% of 15 cases with cross-domain convergence")
    ax.set_title("E6 F4 -- Cross-domain (unsupported) convergence by strategy")
    t1[["cross_domain_convergence_pct"]].to_csv(FIG_DIR / "E6_F4_cross_domain_convergence.csv")
    save(fig, "E6_F4_cross_domain_convergence")

    # --- F5: rules vs families (triggered domain-rows only) -----------------
    fig, ax = plt.subplots(figsize=(6, 5))
    trig = per_domain[per_domain["triggered"]]
    trig[["case_id", "concern_domain", "strategy", "matched_rule_count", "unique_family_count"]].to_csv(
        FIG_DIR / "E6_F5_rules_vs_families.csv", index=False)
    for strat in STRATEGIES:
        sub = trig[trig["strategy"] == strat]
        if len(sub) == 0:
            continue
        jitter = np.random.default_rng(hash(strat) % (2**31)).uniform(-0.08, 0.08, size=len(sub))
        ax.scatter(sub["matched_rule_count"] + jitter, sub["unique_family_count"], color=COLORS[strat],
                   label=f"{strat} (n={len(sub)})", s=70, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Matched rule count (raw)")
    ax.set_ylabel("Independent evidence-family count")
    ax.set_title("E6 F5 -- Raw rule count vs. independent family count\n(triggered domain-instances only)")
    max_val = max(trig["matched_rule_count"].max(), trig["unique_family_count"].max()) + 1 if len(trig) else 5
    ax.plot([0, max_val], [0, max_val], "--", color="gray", alpha=0.5, label="rules == families (no duplication)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)
    save(fig, "E6_F5_rules_vs_families")

    # --- F6: case x strategy heatmap ----------------------------------------
    wide = pd.read_csv(EXP_DIR / "tables" / "E6_T2_case_comparison.csv", index_col="case_id")[STRATEGIES].astype(int)
    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(wide.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES)
    ax.set_yticks(range(len(wide.index)))
    ax.set_yticklabels(wide.index, fontsize=8)
    ax.set_title("E6 F6 -- Per-case trigger heatmap (dark = triggered)")
    # Deliberately no per-cell text overlay -- color alone encodes the binary
    # value; the exact values are in E6_T2_case_comparison.csv (this figure's
    # own source data) for anyone who needs the precise per-cell reading.
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="abstained (0) / triggered (1)", shrink=0.6)
    save(fig, "E6_F6_case_strategy_heatmap")

    print("\nAll 6 figures written.")


if __name__ == "__main__":
    main()
