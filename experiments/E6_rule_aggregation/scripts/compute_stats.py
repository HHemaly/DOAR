#!/usr/bin/env python
"""E6 -- metrics + paired statistical comparisons across R1-R5.

Reads experiments/E6_rule_aggregation/raw/aggregation_per_case.csv and
aggregation_per_domain.csv (written by run_e6.py) and writes:
  statistics/confidence_intervals.csv
  statistics/statistical_tests.csv
  statistics/effect_sizes.csv
  tables/E6_T1_strategy_comparison.csv (+ .tex)
  tables/E6_T2_case_comparison.csv
  tables/E6_T3_domain_comparison.csv
  tables/E6_T4_failure_cases.csv
  tables/E6_T5_pairwise_effects.csv

All comparisons are PAIRED (same 15 cases scored by every strategy).
N=15 is small -- every p-value here is reported alongside its effect
size and CI, never alone; see KNOWN_LIMITATIONS.md.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation"

RNG_SEED = 20260819
N_BOOTSTRAP = 10000
STRATEGIES = ["R1", "R2", "R3", "R4", "R5"]


def wilson_ci(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return (float("nan"), float("nan"))
    result = sstats.binomtest(k, n).proportion_ci(confidence_level=1 - alpha, method="wilson")
    return (result.low, result.high)


def bootstrap_ci_proportion(values: np.ndarray, rng: np.random.Generator, n_boot=N_BOOTSTRAP):
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = values[idx].mean(axis=1)
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def bootstrap_ci_paired_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot=N_BOOTSTRAP):
    """CI on mean(a) - mean(b) via paired case-resampling (same case
    indices drawn for both arms each bootstrap iteration -- correct for
    paired/same-cases-scored-by-both-strategies data)."""
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean()))


def cochrans_q(x: np.ndarray) -> tuple[float, float, int]:
    """Cochran's Q for k paired binary treatments over N subjects (x is
    N x k, entries in {0,1}). Hand-implemented (no statsmodels dependency
    in this environment) from the standard closed-form:
        Q = (k-1) * [k * sum(Cj^2) - (sum(Cj))^2] / [k * sum(Ri) - sum(Ri^2)]
    Q ~ chi-square(k-1) under H0 (all treatments have equal trigger
    probability). Returns (Q, p_value, df)."""
    n, k = x.shape
    col_sums = x.sum(axis=0)
    row_sums = x.sum(axis=1)
    numerator = (k - 1) * (k * np.sum(col_sums ** 2) - np.sum(col_sums) ** 2)
    denominator = k * np.sum(row_sums) - np.sum(row_sums ** 2)
    if denominator == 0:
        return (float("nan"), float("nan"), k - 1)
    q_stat = numerator / denominator
    p_value = 1 - sstats.chi2.cdf(q_stat, df=k - 1)
    return (float(q_stat), float(p_value), k - 1)


def holm_correct(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    per_case = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_case.csv")
    per_domain = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_domain.csv")
    n_cases = per_case["case_id"].nunique()

    # --- Per-strategy summary metrics (Section 6 of the task, Table E6_T1) ---
    summary_rows = []
    trigger_vectors = {}
    coverage_vectors = {}
    dup_vectors = {}
    cross_vectors = {}
    struct_vectors = {}
    contra_vectors = {}

    case_ids_sorted = sorted(per_case["case_id"].unique())
    for strategy in STRATEGIES:
        sdf = per_case[per_case["strategy"] == strategy].set_index("case_id").loc[case_ids_sorted]
        triggered = sdf["any_domain_triggered"].astype(bool).values
        struct_viol = sdf["any_structural_violation"].astype(bool).values
        dup = sdf["any_duplicate_inflation"].astype(bool).values
        cross = sdf["any_cross_domain_convergence"].astype(bool).values
        contra = sdf["any_ignored_contradiction"].astype(bool).values
        # "Supported coverage": triggered AND structurally clean (no violation of any kind).
        supported = triggered & ~struct_viol

        trigger_vectors[strategy] = triggered.astype(int)
        coverage_vectors[strategy] = supported.astype(int)
        dup_vectors[strategy] = dup.astype(int)
        cross_vectors[strategy] = cross.astype(int)
        struct_vectors[strategy] = struct_viol.astype(int)
        contra_vectors[strategy] = contra.astype(int)

        ddf = per_domain[(per_domain["strategy"] == strategy) & (per_domain["triggered"])]
        rules_per = ddf["matched_rule_count"]
        fams_per = ddf["unique_family_count"]

        n_trig, n_cov = int(triggered.sum()), int(supported.sum())
        n_dup, n_cross, n_struct, n_contra = int(dup.sum()), int(cross.sum()), int(struct_viol.sum()), int(contra.sum())

        summary_rows.append({
            "strategy": strategy, "N": n_cases,
            "triggered_n": n_trig, "triggered_pct": 100 * n_trig / n_cases,
            "abstained_n": n_cases - n_trig, "abstained_pct": 100 * (n_cases - n_trig) / n_cases,
            "supported_coverage_n": n_cov, "supported_coverage_pct": 100 * n_cov / n_cases,
            "mean_rules_per_triggered_concern": float(rules_per.mean()) if len(rules_per) else float("nan"),
            "median_rules_per_triggered_concern": float(rules_per.median()) if len(rules_per) else float("nan"),
            "mean_independent_families": float(fams_per.mean()) if len(fams_per) else float("nan"),
            "median_independent_families": float(fams_per.median()) if len(fams_per) else float("nan"),
            "duplicate_inflation_n": n_dup, "duplicate_inflation_pct": 100 * n_dup / n_cases,
            "cross_domain_convergence_n": n_cross, "cross_domain_convergence_pct": 100 * n_cross / n_cases,
            "structurally_unsupported_n": n_struct, "structurally_unsupported_pct": 100 * n_struct / n_cases,
            "ignored_contradiction_n": n_contra, "ignored_contradiction_pct": 100 * n_contra / n_cases,
        })

    summary = pd.DataFrame(summary_rows).set_index("strategy")
    r4_row = summary.loc["R4"]
    for col in ("triggered_pct", "supported_coverage_pct", "duplicate_inflation_pct",
                "cross_domain_convergence_pct", "structurally_unsupported_pct", "ignored_contradiction_pct"):
        summary[f"abs_diff_vs_R4_{col}"] = summary[col] - r4_row[col]
        rel = []
        for strat, val in summary[col].items():
            base = r4_row[col]
            if base == 0:
                rel.append(float("nan"))  # undefined -- R4's baseline is 0; reported as n/a, never divided
            else:
                rel.append(100 * (val - base) / base)
        summary[f"rel_pct_diff_vs_R4_{col}"] = rel

    summary = summary.reset_index()
    tables_dir = EXP_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables_dir / "E6_T1_strategy_comparison.csv", index=False)

    # LaTeX version -- compact, thesis-ready subset of columns.
    tex_cols = ["strategy", "triggered_pct", "supported_coverage_pct", "duplicate_inflation_pct",
                "cross_domain_convergence_pct", "structurally_unsupported_pct"]
    tex_df = summary[tex_cols].round(1)
    tex_df.columns = ["Strategy", "Triggered \\%", "Supported cov. \\%", "Dup. inflation \\%",
                       "Cross-domain \\%", "Structural viol. \\%"]
    with open(tables_dir / "E6_T1_strategy_comparison.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/E6_rule_aggregation/scripts/compute_stats.py\n")
        f.write("% Do not hand-edit -- regenerate from source data instead.\n")
        f.write(tex_df.to_latex(index=False, escape=False,
                                 caption="E6: rule/evidence aggregation strategy comparison (N=15 development-set cases).",
                                 label="tab:e6-strategy-comparison"))
    print(f"Wrote {tables_dir / 'E6_T1_strategy_comparison.csv'} and .tex")

    # --- Confidence intervals (Wilson for single proportions, bootstrap for paired diffs) ---
    ci_rows = []
    for strategy in STRATEGIES:
        n_trig = int(trigger_vectors[strategy].sum())
        n_cov = int(coverage_vectors[strategy].sum())
        lo, hi = wilson_ci(n_trig, n_cases)
        clo, chi = wilson_ci(n_cov, n_cases)
        boot_lo, boot_hi = bootstrap_ci_proportion(trigger_vectors[strategy].astype(float), rng)
        ci_rows.append({
            "strategy": strategy, "metric": "triggered_rate", "point_estimate": n_trig / n_cases,
            "wilson_ci_low": lo, "wilson_ci_high": hi, "bootstrap_ci_low": boot_lo, "bootstrap_ci_high": boot_hi,
            "n": n_cases, "k": n_trig,
        })
        ci_rows.append({
            "strategy": strategy, "metric": "supported_coverage_rate", "point_estimate": n_cov / n_cases,
            "wilson_ci_low": clo, "wilson_ci_high": chi,
            "bootstrap_ci_low": bootstrap_ci_proportion(coverage_vectors[strategy].astype(float), rng)[0],
            "bootstrap_ci_high": bootstrap_ci_proportion(coverage_vectors[strategy].astype(float), rng)[1],
            "n": n_cases, "k": n_cov,
        })

    for strategy in STRATEGIES:
        if strategy == "R4":
            continue
        lo, hi, mean_diff = bootstrap_ci_paired_diff(
            trigger_vectors[strategy].astype(float), trigger_vectors["R4"].astype(float), rng)
        ci_rows.append({
            "strategy": f"{strategy}_vs_R4", "metric": "triggered_rate_diff", "point_estimate": mean_diff,
            "wilson_ci_low": float("nan"), "wilson_ci_high": float("nan"),
            "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "n": n_cases, "k": float("nan"),
        })

    stats_dir = EXP_DIR / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ci_rows).to_csv(stats_dir / "confidence_intervals.csv", index=False)
    print(f"Wrote {stats_dir / 'confidence_intervals.csv'}")

    # --- Pairwise exact McNemar tests (Holm-corrected) on 'triggered' ---
    pairs = list(combinations(STRATEGIES, 2))
    test_rows = []
    raw_pvals = []
    for s1, s2 in pairs:
        a, b = trigger_vectors[s1], trigger_vectors[s2]
        b_disc = int(((a == 1) & (b == 0)).sum())  # s1 triggers, s2 doesn't
        c_disc = int(((a == 0) & (b == 1)).sum())  # s2 triggers, s1 doesn't
        n_disc = b_disc + c_disc
        if n_disc == 0:
            pval = 1.0
            note = "no discordant pairs -- strategies agree on every case"
        else:
            res = sstats.binomtest(min(b_disc, c_disc), n_disc, 0.5)
            pval = res.pvalue
            note = ("degenerate: one side has 0 discordant cases" if 0 in (b_disc, c_disc)
                     else "")
        raw_pvals.append(pval)
        test_rows.append({
            "comparison": f"{s1}_vs_{s2}", "test": "exact_mcnemar_sign_test",
            "n_discordant": n_disc, f"{s1}_only_triggers": b_disc, f"{s2}_only_triggers": c_disc,
            "p_value_raw": pval, "note": note,
        })
    holm = holm_correct(raw_pvals)
    for row, p_adj in zip(test_rows, holm):
        row["p_value_holm_adjusted"] = p_adj

    # Cochran's Q across R1,R2,R3,R5 (R4 excluded: constant all-zero vector makes Q's
    # variance term degenerate/undefined -- documented, not silently dropped).
    q_matrix = np.vstack([trigger_vectors[s] for s in ("R1", "R2", "R3", "R5")]).T
    q_stat, q_p, q_df = cochrans_q(q_matrix)
    test_rows.append({
        "comparison": "cochrans_q_R1_R2_R3_R5", "test": "cochrans_q", "n_discordant": None,
        "p_value_raw": q_p, "p_value_holm_adjusted": q_p,  # single omnibus test, no multiplicity correction needed
        "note": f"Q={q_stat:.3f}, df={q_df}; R4 excluded (constant zero vector, degenerate for Cochran's Q); "
                f"omnibus test of whether R1/R2/R3/R5 trigger rates differ across the same 15 cases",
    })

    pd.DataFrame(test_rows).to_csv(stats_dir / "statistical_tests.csv", index=False)
    print(f"Wrote {stats_dir / 'statistical_tests.csv'}")

    # --- Effect sizes: risk difference (proportion diff) + odds ratio where defined ---
    effect_rows = []
    for s1, s2 in pairs:
        a, b = trigger_vectors[s1], trigger_vectors[s2]
        risk_diff = a.mean() - b.mean()
        lo, hi, _ = bootstrap_ci_paired_diff(a.astype(float), b.astype(float), rng)
        b_disc = int(((a == 1) & (b == 0)).sum())
        c_disc = int(((a == 0) & (b == 1)).sum())
        odds_ratio = (b_disc / c_disc) if c_disc > 0 else float("inf") if b_disc > 0 else float("nan")
        effect_rows.append({
            "comparison": f"{s1}_vs_{s2}", "risk_difference": risk_diff,
            "risk_difference_bootstrap_ci_low": lo, "risk_difference_bootstrap_ci_high": hi,
            "discordant_odds_ratio": odds_ratio,
            "note": "odds ratio undefined/infinite when one discordant cell is 0 -- reported honestly, not clipped",
        })
    pd.DataFrame(effect_rows).to_csv(stats_dir / "effect_sizes.csv", index=False)
    print(f"Wrote {stats_dir / 'effect_sizes.csv'}")

    # --- Table E6_T5: pairwise effects (thesis-facing subset) ---
    pd.DataFrame(effect_rows).round(4).to_csv(tables_dir / "E6_T5_pairwise_effects.csv", index=False)

    # --- Table E6_T2: per-case, per-strategy comparison (wide) ---
    wide = per_case.pivot(index="case_id", columns="strategy", values="any_domain_triggered")
    wide = wide[STRATEGIES].astype(bool)
    wide.to_csv(tables_dir / "E6_T2_case_comparison.csv")
    print(f"Wrote {tables_dir / 'E6_T2_case_comparison.csv'}")

    # --- Table E6_T3: per-domain comparison (wide, triggered flag) ---
    dom_wide = per_domain.pivot_table(index=["case_id", "concern_domain"], columns="strategy",
                                       values="triggered", aggfunc="first")
    dom_wide = dom_wide[STRATEGIES]
    dom_wide.to_csv(tables_dir / "E6_T3_domain_comparison.csv")
    print(f"Wrote {tables_dir / 'E6_T3_domain_comparison.csv'}")

    # --- Table E6_T4: failure/violation cases (any strategy, any structural violation) ---
    failure_rows = per_domain[per_domain["structural_violation"]].copy()
    failure_rows = failure_rows[["case_id", "concern_domain", "strategy", "matched_rule_count",
                                  "unique_family_count", "duplicate_inflation", "cross_domain_convergence",
                                  "insufficient_independent_evidence", "inappropriate_unverified_use",
                                  "ignored_contradiction", "reason_for_trigger"]]
    failure_rows.to_csv(tables_dir / "E6_T4_failure_cases.csv", index=False)
    print(f"Wrote {tables_dir / 'E6_T4_failure_cases.csv'} ({len(failure_rows)} rows)")

    print("\nDone with statistics. Run make_figures.py next.")


if __name__ == "__main__":
    main()
