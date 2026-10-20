"""C4 — statistics and figures for the measurement-validity results (CPU).

  python experiments/journal/scripts/c4_measurement_analysis.py \
      --c3 <Drive>/DOAR_JOURNAL_2026/C3_reference_evaluation \
      --out <Drive>/DOAR_JOURNAL_2026/C4_measurement_analysis

Pre-registered choices (PREREGISTRATION.md): primary mask metric = pixel F1;
"bad measurement" = F1 < 0.5 on an assessable image, or any image a human
marked NOT_ASSESSABLE (a reliable gate should withhold those); paired
method comparisons against frozen V2 use a paired bootstrap of the mean
per-image F1 difference (10,000 resamples) and an exact Wilcoxon / sign
test, Holm-corrected across baselines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from scipy import stats as st  # noqa: E402

from doar_journal import provenance  # noqa: E402
from doar_journal.measurement import feature_agreement  # noqa: E402
from doar_journal.risk_coverage import (A0_QC_SCORE, CONFIDENCE_SCORE, POLICY_SCORE,  # noqa: E402
                                        auroc_bad, risk_at_coverage, risk_coverage)
from doar_journal.stats import holm  # noqa: E402

BAD_F1 = 0.5
N_BOOT = 10_000
RNG_SEED = 20261020


def boot_mean(x: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(RNG_SEED)
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    c3, out = Path(a.c3), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pm = pd.read_csv(c3 / "per_image_method.csv")
    ass = pm[pm["assessability"] == "ASSESSABLE"].copy()

    # 1. Mask accuracy by method (overall and by V2 support regime)
    rows = []
    for (method, regime), g in [((m, "ALL"), g) for m, g in ass.groupby("method")] + \
            [((m, r), g) for (m, r), g in ass.groupby(["method", "v2_regime"])]:
        rec = {"method": method, "regime": regime, "n": len(g)}
        for metric in ("f1", "iou", "boundary_f1", "precision", "recall"):
            est, lo, hi = boot_mean(g[metric].to_numpy())
            rec.update({f"{metric}_mean": est, f"{metric}_lo": lo, f"{metric}_hi": hi})
        rows.append(rec)
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "T_mask_accuracy.csv", index=False)

    # 2. Paired comparisons: V2 minus each other method (per-image F1)
    wide = ass.pivot(index="task_id", columns="method", values="f1")
    comp, pvals = [], {}
    rng = np.random.default_rng(RNG_SEED)
    for m in [c for c in wide.columns if c != "v2"]:
        d = (wide["v2"] - wide[m]).dropna().to_numpy()
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        bm = d[idx].mean(axis=1)
        w = st.wilcoxon(d) if np.any(d != 0) else None
        p = float(w.pvalue) if w is not None else 1.0
        pvals[m] = p
        comp.append({"comparison": f"v2 - {m}", "n": len(d), "mean_diff_f1": float(d.mean()),
                     "ci_lo": float(np.percentile(bm, 2.5)), "ci_hi": float(np.percentile(bm, 97.5)),
                     "v2_better": int((d > 0).sum()), "v2_worse": int((d < 0).sum()), "p_wilcoxon": p})
    comp = pd.DataFrame(comp)
    adj = holm(pvals)
    comp["p_holm"] = comp["comparison"].str.replace("v2 - ", "").map(lambda m: adj[m]["p_holm"])
    comp.to_csv(out / "T_paired_v2_vs_methods.csv", index=False)

    # 3. Feature propagation (reference-mask features vs each method's)
    fl = pd.read_csv(c3 / "features_long.csv")
    fw = fl.pivot_table(index=["task_id", "feature"], columns="source", values="value").reset_index()
    prop = []
    for m in [c for c in fw.columns if c not in ("task_id", "feature", "reference")]:
        ref_list, est_list = [], []
        for t, g in fw.groupby("task_id"):
            ref_list.append(dict(zip(g["feature"], g["reference"])))
            est_list.append(dict(zip(g["feature"], g[m])))
        for feat, r in feature_agreement(ref_list, est_list).items():
            prop.append({"method": m, "feature": feat, **r})
    pd.DataFrame(prop).to_csv(out / "T_feature_agreement.csv", index=False)

    # 4. Gate validity: do states predict bad measurements?
    gi = pd.read_csv(c3 / "gate_inputs.csv")
    err = np.where(gi["assessability"] != "ASSESSABLE", 1.0, 1.0 - gi["v2_f1"].astype(float))
    a0_err = np.where(gi["assessability"] != "ASSESSABLE", 1.0, 1.0 - gi["a0_f1"].astype(float))
    gates = {
        "v2_measurement_policy": (gi["v2_policy"].map(POLICY_SCORE), err),
        "v2_support_confidence": (gi["v2_confidence"].map(CONFIDENCE_SCORE), err),
        "a0_qc_state(on A0 masks)": (gi["a0_state"].map({"OK": A0_QC_SCORE["OK"], "FLAGGED": 0}), a0_err),
    }
    gate_rows, curves = [], {}
    for name, (score, e) in gates.items():
        rc = risk_coverage(score.to_numpy(float), e)
        curves[name] = rc
        gate_rows.append({"gate": name, "n": rc["n"], "auroc_bad": auroc_bad(score, e, 1 - BAD_F1),
                          "aurc": rc["aurc"], "aurc_random": rc["aurc_random"], "aurc_oracle": rc["aurc_oracle"],
                          "risk_at_80pct": risk_at_coverage(rc, 0.8), "risk_at_100pct": rc["risk"][-1]})
    pd.DataFrame(gate_rows).to_csv(out / "T_gate_validity.csv", index=False)
    provenance.write_json(out / "risk_coverage_curves.json", curves)

    # 5. Figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    overall = acc[acc["regime"] == "ALL"].sort_values("f1_mean")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.errorbar(overall["f1_mean"], overall["method"],
                xerr=[overall["f1_mean"] - overall["f1_lo"], overall["f1_hi"] - overall["f1_mean"]],
                fmt="o", color="#1f4e79", ecolor="#7f7f7f", capsize=3)
    ax.set_xlabel("Pixel F1 vs human reference (mean, 95% bootstrap CI)")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "F_mask_f1_by_method.pdf")
    fig.savefig(out / "F_mask_f1_by_method.png", dpi=200)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for name, rc in curves.items():
        ax.plot(rc["coverage"], rc["risk"], label=f"{name} (AURC {rc['aurc']:.3f})")
    ax.axhline(curves["v2_measurement_policy"]["aurc_random"], ls="--", color="grey", lw=1, label="no gate")
    ax.set_xlabel("Coverage (fraction of images measured)")
    ax.set_ylabel("Risk (mean 1 - F1)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "F_risk_coverage.pdf")
    fig.savefig(out / "F_risk_coverage.png", dpi=200)

    meta = provenance.run_metadata("C4_measurement_analysis", {"bad_f1": BAD_F1, "n_boot": N_BOOT},
                                   inputs={"per_image_method": c3 / "per_image_method.csv"})
    provenance.mark_done(out, meta, [out / "T_mask_accuracy.csv", out / "T_paired_v2_vs_methods.csv",
                                     out / "T_feature_agreement.csv", out / "T_gate_validity.csv"])
    print(overall[["method", "n", "f1_mean", "f1_lo", "f1_hi"]].to_string(index=False))
    print(pd.DataFrame(gate_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
