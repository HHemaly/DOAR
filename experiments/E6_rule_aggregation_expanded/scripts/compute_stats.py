#!/usr/bin/env python
"""E6-EXPANDED -- cohort diagnostics, strategy metrics, and paired
statistics, using the CORRECTED (non-circular) outcome definitions.

Structural violation = OBJECTIVE error only (duplicate inflation,
cross-domain pooling, disabled/ineligible-rule use, unverified-evidence
misuse, ignored TRUE contradiction). "Fewer than 2 independent families"
is NEVER counted as a violation -- it is a policy choice, reported
separately as policy_disagreement_vs_r4.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments" / "E6_rule_aggregation_expanded"

RNG_SEED = 20260820  # fixed -- the date this experiment was (re)run; never changed between reruns
N_BOOTSTRAP = 10000
PRIMARY_STRATEGIES = ["R1", "R2", "R3", "R4", "R5"]
ALL_STRATEGIES = ["R1", "R1_broad", "R2", "R2_broad", "R3", "R4", "R5"]


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
    n = len(a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(diffs.mean()))


def cochrans_q(x: np.ndarray) -> tuple[float, float, int]:
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


def build_cohort_diagnostics(per_domain: pd.DataFrame, atomic: pd.DataFrame, cohort_manifest: pd.DataFrame) -> pd.DataFrame:
    """Section D -- for the EXPANDED, usable cohort (whatever N that is at
    freeze time), counts/percentages of cases with each structural
    property. Computed from the ELIGIBLE (E6-filtered) evidence only,
    since that is what the strategy comparison itself uses."""
    usable = cohort_manifest[(cohort_manifest["cohort_role"] != "locked_benchmark_reserved_do_not_process")
                              & (cohort_manifest["has_cached_evidence"] == True)]  # noqa: E712
    case_ids = sorted(usable["image_id"].unique())
    n = len(case_ids)

    elig = atomic[atomic["eligible_for_e6"]]
    by_case = elig.groupby("case_id")

    def pct(k):
        return 100 * k / n if n else float("nan")

    n_zero_rules = sum(1 for cid in case_ids if cid not in elig["case_id"].values)
    n_ge1_rule = n - n_zero_rules
    n_ge2_rules = sum(1 for cid in case_ids if (elig["case_id"] == cid).sum() >= 2)

    fam_by_case = {cid: set(g["evidence_family"]) for cid, g in by_case}
    n_ge2_families_overall = sum(1 for cid in case_ids if len(fam_by_case.get(cid, set())) >= 2)

    # >=2 independent families WITHIN the same concern domain (R4's own trigger condition)
    n_ge2_same_domain = 0
    for cid in case_ids:
        sub = elig[elig["case_id"] == cid]
        if sub.empty:
            continue
        per_domain_fams = sub.groupby("concern_domain")["evidence_family"].nunique()
        if (per_domain_fams >= 2).any():
            n_ge2_same_domain += 1

    n_dup_family_pairs = 0
    for cid in case_ids:
        sub = elig[elig["case_id"] == cid]
        counts = sub.groupby(["concern_domain", "evidence_family"]).size()
        if (counts >= 2).any():
            n_dup_family_pairs += 1

    n_cross_domain_evidence = sum(1 for cid in case_ids
                                   if elig[elig["case_id"] == cid]["concern_domain"].nunique() >= 2)

    n_true_contradiction = sum(1 for cid in case_ids
                                if elig[(elig["case_id"] == cid)]["is_true_contradiction"].any())
    n_contextual_limitation = sum(1 for cid in case_ids
                                   if elig[(elig["case_id"] == cid)]["is_contextual_limitation"].any())
    n_unverified = sum(1 for cid in case_ids
                        if (elig[elig["case_id"] == cid]["verification_status"] != "verified").any()
                        and not elig[elig["case_id"] == cid].empty
                        and (elig[elig["case_id"] == cid]["verification_status"].str.contains(
                            "uncertain|unreviewed|rejected", na=False)).any())

    case_summary = pd.read_csv(EXP_DIR / "raw" / "case_level_summary.csv").set_index("case_id")
    page_by_case = case_summary["page_assessable"].reindex(case_ids)
    n_page_assessable = int((page_by_case == True).sum()) if len(case_ids) else 0  # noqa: E712

    diag = [
        ("N_usable_cases", n, None),
        ("cases_with_0_eligible_rules", n_zero_rules, pct(n_zero_rules)),
        ("cases_with_ge1_eligible_rule", n_ge1_rule, pct(n_ge1_rule)),
        ("cases_with_ge2_eligible_rules", n_ge2_rules, pct(n_ge2_rules)),
        ("cases_with_ge2_independent_families_overall_any_domain", n_ge2_families_overall, pct(n_ge2_families_overall)),
        ("cases_with_ge2_independent_families_SAME_domain", n_ge2_same_domain, pct(n_ge2_same_domain)),
        ("cases_with_duplicate_family_rule_pairs", n_dup_family_pairs, pct(n_dup_family_pairs)),
        ("cases_with_evidence_spanning_ge2_concern_domains", n_cross_domain_evidence, pct(n_cross_domain_evidence)),
        ("cases_with_true_literature_contradiction", n_true_contradiction, pct(n_true_contradiction)),
        ("cases_with_contextual_cultural_limitation", n_contextual_limitation, pct(n_contextual_limitation)),
        ("cases_using_unverified_uncertain_evidence", n_unverified, pct(n_unverified)),
        ("cases_with_page_assessable", n_page_assessable, pct(n_page_assessable)),
        ("cases_with_page_NOT_assessable", n - n_page_assessable, pct(n - n_page_assessable)),
    ]
    return pd.DataFrame(diag, columns=["metric", "count", "pct_of_N"])


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    per_case = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_case.csv")
    per_domain = pd.read_csv(EXP_DIR / "raw" / "aggregation_per_domain.csv")
    atomic = pd.read_csv(EXP_DIR / "raw" / "input_atomic_evidence.csv")
    cohort_manifest = pd.read_csv(EXP_DIR / "raw" / "cohort_manifest.csv")
    n_cases = per_case["case_id"].nunique()
    case_ids_sorted = sorted(per_case["case_id"].unique())

    tables_dir = EXP_DIR / "tables"
    stats_dir = EXP_DIR / "statistics"
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # --- Cohort diagnostics table (Section D) ---
    diag = build_cohort_diagnostics(per_domain, atomic, cohort_manifest)
    diag.to_csv(tables_dir / "E6X_T2_cohort_diagnostics.csv", index=False)
    print(f"Wrote E6X_T2_cohort_diagnostics.csv ({n_cases} usable cases)")

    # --- Per-strategy summary metrics ---
    summary_rows = []
    trigger_vectors, struct_vectors, dup_vectors, cross_vectors = {}, {}, {}, {}
    ineligible_vectors, contra_vectors, policy_vectors, coverage_vectors = {}, {}, {}, {}

    for strategy in ALL_STRATEGIES:
        sdf = per_case[per_case["strategy"] == strategy].set_index("case_id").loc[case_ids_sorted]
        triggered = sdf["any_domain_triggered"].astype(bool).values
        struct_viol = sdf["any_structural_violation"].astype(bool).values
        dup = sdf["any_duplicate_inflation"].astype(bool).values
        cross = sdf["any_cross_domain_convergence"].astype(bool).values
        ineligible = sdf["any_ineligible_evidence_used"].astype(bool).values
        contra = sdf["any_ignored_true_contradiction"].astype(bool).values
        policy = sdf["any_policy_disagreement_vs_r4"].astype(bool).values
        supported = triggered & ~struct_viol  # "supported coverage" now uses the NARROW, correct violation set

        trigger_vectors[strategy] = triggered.astype(int)
        struct_vectors[strategy] = struct_viol.astype(int)
        dup_vectors[strategy] = dup.astype(int)
        cross_vectors[strategy] = cross.astype(int)
        ineligible_vectors[strategy] = ineligible.astype(int)
        contra_vectors[strategy] = contra.astype(int)
        policy_vectors[strategy] = policy.astype(int)
        coverage_vectors[strategy] = supported.astype(int)

        ddf = per_domain[(per_domain["strategy"] == strategy) & (per_domain["triggered"])]
        rules_per = ddf["matched_rule_count"]
        fams_per = ddf["unique_family_count"]

        n_trig, n_cov = int(triggered.sum()), int(supported.sum())
        summary_rows.append({
            "strategy": strategy, "N": n_cases,
            "triggered_n": n_trig, "triggered_pct": 100 * n_trig / n_cases,
            "abstained_n": n_cases - n_trig, "abstained_pct": 100 * (n_cases - n_trig) / n_cases,
            "supported_coverage_n": n_cov, "supported_coverage_pct": 100 * n_cov / n_cases,
            "mean_rules_per_triggered_concern": float(rules_per.mean()) if len(rules_per) else float("nan"),
            "median_rules_per_triggered_concern": float(rules_per.median()) if len(rules_per) else float("nan"),
            "mean_independent_families": float(fams_per.mean()) if len(fams_per) else float("nan"),
            "median_independent_families": float(fams_per.median()) if len(fams_per) else float("nan"),
            "duplicate_inflation_n": int(dup.sum()), "duplicate_inflation_pct": 100 * dup.sum() / n_cases,
            "cross_domain_convergence_n": int(cross.sum()), "cross_domain_convergence_pct": 100 * cross.sum() / n_cases,
            "ineligible_evidence_used_n": int(ineligible.sum()), "ineligible_evidence_used_pct": 100 * ineligible.sum() / n_cases,
            "structurally_unsupported_n": int(struct_viol.sum()), "structurally_unsupported_pct": 100 * struct_viol.sum() / n_cases,
            "ignored_true_contradiction_n": int(contra.sum()), "ignored_true_contradiction_pct": 100 * contra.sum() / n_cases,
            "policy_disagreement_vs_r4_n": int(policy.sum()), "policy_disagreement_vs_r4_pct": 100 * policy.sum() / n_cases,
        })

    summary = pd.DataFrame(summary_rows).set_index("strategy")
    r4_row = summary.loc["R4"]
    for col in ("triggered_pct", "supported_coverage_pct", "duplicate_inflation_pct",
                "cross_domain_convergence_pct", "structurally_unsupported_pct"):
        summary[f"abs_diff_vs_R4_{col}"] = summary[col] - r4_row[col]
        rel = [float("nan") if r4_row[col] == 0 else 100 * (v - r4_row[col]) / r4_row[col]
               for v in summary[col]]
        summary[f"rel_pct_diff_vs_R4_{col}"] = rel
    summary = summary.reset_index()
    summary.to_csv(tables_dir / "E6X_T1_strategy_comparison.csv", index=False)
    tex_cols = ["strategy", "triggered_pct", "supported_coverage_pct", "duplicate_inflation_pct",
                "cross_domain_convergence_pct", "structurally_unsupported_pct", "policy_disagreement_vs_r4_pct"]
    tex_df = summary[tex_cols].round(1)
    tex_df.columns = ["Strategy", "Triggered \\%", "Supported cov. \\%", "Dup. inflation \\%",
                       "Cross-domain \\%", "Structural viol. \\%", "Policy vs R4 \\%"]
    with open(tables_dir / "E6X_T1_strategy_comparison.tex", "w", encoding="utf-8") as f:
        f.write("% Auto-generated by experiments/E6_rule_aggregation_expanded/scripts/compute_stats.py\n")
        f.write(tex_df.to_latex(index=False, escape=False,
                                 caption="E6-EXPANDED: corrected rule/evidence aggregation strategy comparison.",
                                 label="tab:e6x-strategy-comparison"))
    print("Wrote E6X_T1_strategy_comparison.csv (+.tex)")

    # --- Confidence intervals ---
    ci_rows = []
    for strategy in ALL_STRATEGIES:
        n_trig, n_cov = int(trigger_vectors[strategy].sum()), int(coverage_vectors[strategy].sum())
        lo, hi = wilson_ci(n_trig, n_cases)
        clo, chi = wilson_ci(n_cov, n_cases)
        ci_rows.append({"strategy": strategy, "metric": "triggered_rate", "point_estimate": n_trig / n_cases,
                         "wilson_ci_low": lo, "wilson_ci_high": hi, "n": n_cases, "k": n_trig})
        ci_rows.append({"strategy": strategy, "metric": "supported_coverage_rate", "point_estimate": n_cov / n_cases,
                         "wilson_ci_low": clo, "wilson_ci_high": chi, "n": n_cases, "k": n_cov})
    for strategy in PRIMARY_STRATEGIES:
        if strategy == "R4":
            continue
        lo, hi, mean_diff = bootstrap_ci_paired_diff(
            trigger_vectors[strategy].astype(float), trigger_vectors["R4"].astype(float), rng)
        ci_rows.append({"strategy": f"{strategy}_vs_R4", "metric": "triggered_rate_diff", "point_estimate": mean_diff,
                         "wilson_ci_low": float("nan"), "wilson_ci_high": float("nan"),
                         "bootstrap_ci_low": lo, "bootstrap_ci_high": hi, "n": n_cases, "k": float("nan")})
    pd.DataFrame(ci_rows).to_csv(stats_dir / "confidence_intervals.csv", index=False)
    print("Wrote confidence_intervals.csv")

    # --- Pairwise exact McNemar (primary strategies only), Holm-corrected ---
    pairs = list(combinations(PRIMARY_STRATEGIES, 2))
    test_rows, raw_pvals = [], []
    for s1, s2 in pairs:
        a, b = trigger_vectors[s1], trigger_vectors[s2]
        b_disc = int(((a == 1) & (b == 0)).sum())
        c_disc = int(((a == 0) & (b == 1)).sum())
        n_disc = b_disc + c_disc
        if n_disc == 0:
            pval, note = 1.0, "no discordant pairs"
        else:
            pval = sstats.binomtest(min(b_disc, c_disc), n_disc, 0.5).pvalue
            note = "degenerate: one side has 0 discordant cases" if 0 in (b_disc, c_disc) else ""
        raw_pvals.append(pval)
        test_rows.append({"comparison": f"{s1}_vs_{s2}", "test": "exact_mcnemar_sign_test", "n_discordant": n_disc,
                           f"{s1}_only_triggers": b_disc, f"{s2}_only_triggers": c_disc,
                           "p_value_raw": pval, "note": note})
    holm = holm_correct(raw_pvals)
    for row, p_adj in zip(test_rows, holm):
        row["p_value_holm_adjusted"] = p_adj

    q_matrix = np.vstack([trigger_vectors[s] for s in ("R1", "R2", "R3", "R5")]).T
    q_stat, q_p, q_df = cochrans_q(q_matrix)
    test_rows.append({"comparison": "cochrans_q_R1_R2_R3_R5", "test": "cochrans_q", "n_discordant": None,
                       "p_value_raw": q_p, "p_value_holm_adjusted": q_p,
                       "note": f"Q={q_stat:.3f}, df={q_df}; R4 excluded"})
    pd.DataFrame(test_rows).to_csv(stats_dir / "statistical_tests.csv", index=False)
    print("Wrote statistical_tests.csv")

    # --- Effect sizes ---
    effect_rows = []
    for s1, s2 in pairs:
        a, b = trigger_vectors[s1], trigger_vectors[s2]
        risk_diff = a.mean() - b.mean()
        lo, hi, _ = bootstrap_ci_paired_diff(a.astype(float), b.astype(float), rng)
        b_disc = int(((a == 1) & (b == 0)).sum())
        c_disc = int(((a == 0) & (b == 1)).sum())
        odds_ratio = (b_disc / c_disc) if c_disc > 0 else (float("inf") if b_disc > 0 else float("nan"))
        effect_rows.append({"comparison": f"{s1}_vs_{s2}", "risk_difference": risk_diff,
                             "risk_difference_bootstrap_ci_low": lo, "risk_difference_bootstrap_ci_high": hi,
                             "discordant_odds_ratio": odds_ratio})
    pd.DataFrame(effect_rows).to_csv(stats_dir / "effect_sizes.csv", index=False)
    pd.DataFrame(effect_rows).round(4).to_csv(tables_dir / "E6X_T7_pairwise_effects.csv", index=False)
    print("Wrote effect_sizes.csv, E6X_T7_pairwise_effects.csv")

    # --- Table: per-case comparison (wide) ---
    wide = per_case.pivot(index="case_id", columns="strategy", values="any_domain_triggered")[ALL_STRATEGIES].astype(bool)
    wide.to_csv(tables_dir / "E6X_T3_case_comparison.csv")

    # --- Table: per-domain comparison (wide) ---
    dom_wide = per_domain.pivot_table(index=["case_id", "concern_domain"], columns="strategy",
                                       values="triggered", aggfunc="first")[ALL_STRATEGIES]
    dom_wide.to_csv(tables_dir / "E6X_T4_domain_comparison.csv")

    # --- Table: policy disagreement (Section E) ---
    policy_rows = per_domain[(per_domain["strategy"].isin(PRIMARY_STRATEGIES)) & (per_domain["triggered"])
                              & (per_domain["policy_disagreement_vs_r4"])]
    policy_rows[["case_id", "concern_domain", "strategy", "matched_rule_count", "unique_family_count",
                 "assessable_family_count", "support_ratio", "meets_r4_family_floor"]].to_csv(
        tables_dir / "E6X_T5_policy_disagreement.csv", index=False)
    print(f"Wrote E6X_T5_policy_disagreement.csv ({len(policy_rows)} rows)")

    # --- Table: structural failure (objective errors only) ---
    failure_rows = per_domain[per_domain["structural_violation"]]
    failure_rows[["case_id", "concern_domain", "strategy", "duplicate_inflation", "cross_domain_convergence",
                  "ineligible_evidence_used", "inappropriate_unverified_use", "ignored_true_contradiction",
                  "disabled_rules_present_but_excluded"]].to_csv(
        tables_dir / "E6X_T6_structural_failures.csv", index=False)
    print(f"Wrote E6X_T6_structural_failures.csv ({len(failure_rows)} rows)")

    print(f"\nDone with statistics. N={n_cases} usable cases.")


if __name__ == "__main__":
    main()
