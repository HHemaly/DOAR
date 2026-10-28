"""Stage B analysis: tables, rankings, paired comparisons and the frozen selection.

Pre-registered rules (PREREGISTRATION.md, C4):
* shortlist  = the 3 encoders with the highest 10-seed mean frozen-probe macro-F1
* finalists  = the 2 (encoder, mode) configurations with the highest 3-seed
               mean adaptation macro-F1; each gets 2 further seeds (11, 37)
* selection  = highest 5-seed mean validation macro-F1; tie -> lower SD
* model uncertainty = SD over seeds; evaluation uncertainty = CLUSTER
  bootstrap (unit = duplicate/source group from the manifest) of the
  seed-ensemble (mean probability over seeds) predictions;
  finalist comparison = group-paired bootstrap (inference) + McNemar counts
  (descriptive only, because images within a group are not independent)
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .stats import CLASSES, bootstrap_metric, kendall_tau, macro_f1, mcnemar_exact, paired_bootstrap_diff

RUN_RE = re.compile(r"(?P<model>[a-z0-9_]+?)__(?P<mode>probe|frozen|partial|full)__seed_(?P<seed>\d+)$")


def collect_runs(root: str | Path) -> dict[tuple[str, str], dict[int, Path]]:
    """{(model, mode): {seed: predictions.csv}} for run dirs named <model>__<mode>__seed_<s>."""
    out: dict[tuple[str, str], dict[int, Path]] = defaultdict(dict)
    for p in Path(root).rglob("best_valid_predictions.csv"):
        m = RUN_RE.match(p.parent.name)
        if m:
            out[(m["model"], m["mode"])][int(m["seed"])] = p
            continue
        # archived E1-A2R layout: runs/<model>/seed_<s>/best_valid_predictions.csv
        s = re.fullmatch(r"seed_(\d+)", p.parent.name)
        if s:
            out[(p.parent.parent.name, "probe")][int(s.group(1))] = p
    return dict(out)


def read_preds(p: Path) -> dict[str, dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return {r["image_id"]: r for r in csv.DictReader(f)}


def ensemble(paths: list[Path]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Seed ensemble = mean of class probabilities; returns ids, truth, predicted labels."""
    preds = [read_preds(p) for p in paths]
    ids = sorted(preds[0])
    for q in preds[1:]:
        if sorted(q) != ids:
            raise ValueError("seed runs disagree on the validation image set")
    probs = np.mean([[[float(q[i][f"prob_{c}"]) for c in CLASSES] for i in ids] for q in preds], axis=0)
    truth = np.array([preds[0][i]["true_class"] for i in ids])
    return ids, truth, np.array(CLASSES)[probs.argmax(axis=1)]


def config_table(runs: dict, n_boot: int = 10_000, groups: dict[str, str] | None = None) -> list[dict]:
    rows = []
    for (model, mode), seeds in sorted(runs.items()):
        f1s = []
        for s, p in sorted(seeds.items()):
            r = read_preds(p)
            f1s.append(macro_f1([v["true_class"] for v in r.values()], [v["predicted_class"] for v in r.values()]))
        ids, y, yhat = ensemble([seeds[s] for s in sorted(seeds)])
        g = None if groups is None else [groups.get(i, i) for i in ids]
        ci = bootstrap_metric(y, yhat, n_boot=n_boot, groups=g)
        rows.append({"model": model, "mode": mode, "n_seeds": len(f1s), "seeds": ";".join(map(str, sorted(seeds))),
                     "mean_macro_f1": float(np.mean(f1s)), "sd_macro_f1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
                     "min_macro_f1": float(np.min(f1s)), "max_macro_f1": float(np.max(f1s)),
                     "ensemble_macro_f1": ci.estimate, "ensemble_ci_low": ci.low, "ensemble_ci_high": ci.high,
                     "n_valid": len(ids), "n_groups": len(set(g)) if g else len(ids),
                     "resampling_unit": "group" if g else "image"})
    return rows


def shortlist(probe_rows: list[dict], k: int = 3) -> list[str]:
    probe = [r for r in probe_rows if r["mode"] == "probe"]
    return [r["model"] for r in sorted(probe, key=lambda r: (-r["mean_macro_f1"], r["sd_macro_f1"]))[:k]]


def finalists(runs: dict, k: int = 2, seeds=(19, 42, 73)) -> list[tuple[str, str]]:
    """Rank adaptation configs on the 3 shared E1-B seeds ONLY, so seeds added
    later for the finalists can never influence which configs became finalists."""
    scored = []
    for (model, mode), by_seed in runs.items():
        if mode == "probe":
            continue
        if not set(seeds) <= set(by_seed):
            raise ValueError(f"{model}/{mode} is missing E1-B seeds {sorted(set(seeds) - set(by_seed))}")
        f1s = []
        for s in seeds:
            r = read_preds(by_seed[s])
            f1s.append(macro_f1([v["true_class"] for v in r.values()], [v["predicted_class"] for v in r.values()]))
        scored.append(((model, mode), float(np.mean(f1s)), float(np.std(f1s, ddof=1))))
    return [c for c, _, _ in sorted(scored, key=lambda t: (-t[1], t[2]))[:k]]


def select(final_rows: list[dict]) -> tuple[str, str]:
    best = sorted(final_rows, key=lambda r: (-r["mean_macro_f1"], r["sd_macro_f1"]))[0]
    return best["model"], best["mode"]


def compare_finalists(runs: dict, a: tuple[str, str], b: tuple[str, str], n_boot: int = 10_000,
                      groups: dict[str, str] | None = None) -> dict:
    ids_a, y_a, p_a = ensemble([runs[a][s] for s in sorted(runs[a])])
    ids_b, y_b, p_b = ensemble([runs[b][s] for s in sorted(runs[b])])
    if ids_a != ids_b or not np.array_equal(y_a, y_b):
        raise ValueError("finalists were evaluated on different validation images")
    g = None if groups is None else [groups.get(i, i) for i in ids_a]
    return {"a": "/".join(a), "b": "/".join(b),
            "paired_bootstrap_macro_f1": paired_bootstrap_diff(y_a, p_a, p_b, n_boot=n_boot, groups=g),
            "mcnemar_counts_descriptive": mcnemar_exact(p_a == y_a, p_b == y_b)}


def ranking_agreement(rows_t0: list[dict], rows_t1: list[dict], mode: str = "probe") -> dict:
    o0 = [r["model"] for r in sorted([r for r in rows_t0 if r["mode"] == mode], key=lambda r: -r["mean_macro_f1"])]
    o1 = [r["model"] for r in sorted([r for r in rows_t1 if r["mode"] == mode], key=lambda r: -r["mean_macro_f1"])]
    common = [m for m in o0 if m in o1]
    return {"mode": mode, "t0_order": o0, "t1_order": o1,
            **kendall_tau(common, [m for m in o1 if m in common])}
