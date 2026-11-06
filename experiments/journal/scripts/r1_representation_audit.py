"""R1 — source-shortcut and representation audit from SAVED outputs (CPU, read-only; no training).

  python experiments/journal/scripts/r1_representation_audit.py --root <Drive>/DOAR_JOURNAL_2026 \
      --t0-runs <Drive>/DOAR_GPU_RESULTS/E1 [--t1 T1_v2]

Post hoc and exploratory (designed after B3). It selects nothing and changes no frozen result.
It reads the T1 manifest, the cached frozen-encoder embeddings of the T1 train+valid images
(written by the B1 probe stage), the A2 tables of the saved T0 runs, and the already-saved
predictions of the single test run. It never computes a new test-set prediction.

Hypotheses
  H1 (shortcut capacity)  the contributing source (filename family) predicts the emotion label
                          better than chance on T1 train+valid.
  H2 (copy sensitivity)   encoders differ in how strongly they map a transformed copy of a drawing
                          to the same point: nearest neighbour of an image is its copy (copy recall@1).
                          Copies are defined by Roboflow filename stems only, so the measure is
                          independent of the CLIP-based duplicate evidence used to build T1.
  H3 (copy shortcut)      kNN emotion macro-F1 drops when neighbours from the same duplicate group
                          are excluded; the drop measures how much an encoder can profit from copies.
  H4 (mechanism)          encoders with higher copy sensitivity gained more from leaked copies on T0
                          (A2: macro-F1 on shared-source minus independent validation images), and
                          their ranking moved more between T0 and T1. Nine encoders: descriptive only.
  R1b (descriptive)       per-source macro-F1 of the selected/comparator/probe systems on the
                          locked test set, from the saved predictions (no model is run).
Also: the A2 gap re-computed within class x source-family strata, so it is not explained by a
different class mix or source mix of shared and independent images.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import stats  # noqa: E402
from doar_journal.leakage import evaluation_groups, path_of, source_key  # noqa: E402
from doar_journal.provenance import run_metadata, sha256_file, write_json  # noqa: E402
from doar_journal.rescoring import find_runs, read_predictions, source_family  # noqa: E402

CLASSES = ["Angry", "Fear", "Happy", "Sad"]
MODELS = ["mobilenet_v3_small", "resnet18", "efficientnet_b0", "densenet121", "convnext_tiny",
          "vit_b_16", "clip_vit_b32", "dinov2_vits14", "siglip2_vit_b16"]


def read(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(p: Path, rows: list[dict]):
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cramers_v(table: np.ndarray) -> float:
    from scipy.stats import chi2_contingency
    table = table[table.sum(1) > 0][:, table.sum(0) > 0]
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def group_folds(groups: list[str], k: int = 5, seed: int = 20261106) -> np.ndarray:
    uniq = sorted(set(groups))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {g: int(order[i] % k) for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


# ---------------------------------------------------------------- R1a source vs label
def source_vs_label(rows: list[dict]) -> dict:
    fams = sorted({r["family"] for r in rows})
    tab = np.array([[sum(r["family"] == f and r["class"] == c for r in rows) for c in CLASSES] for f in fams])
    folds = group_folds([r["group_id"] for r in rows])
    y = [r["class"] for r in rows]
    pred = [None] * len(rows)
    for k in range(5):
        train = [i for i in range(len(rows)) if folds[i] != k]
        maj = {}
        for f in fams:
            c = Counter(y[i] for i in train if rows[i]["family"] == f)
            maj[f] = c.most_common(1)[0][0] if c else Counter(y[i] for i in train).most_common(1)[0][0]
        for i in range(len(rows)):
            if folds[i] == k:
                pred[i] = maj[rows[i]["family"]]
    groups = [r["group_id"] for r in rows]
    prior = Counter(y).most_common(1)[0][0]
    return {"families": fams, "table_family_x_class": {f: dict(zip(CLASSES, map(int, tab[i]))) for i, f in enumerate(fams)},
            "cramers_v_family_class": cramers_v(tab),
            "source_only_macro_f1_groupcv": stats.bootstrap_metric(y, pred, groups=groups, n_boot=2000).as_dict(),
            "majority_class_macro_f1": stats.macro_f1(y, [prior] * len(y)),
            "chance_macro_f1_uniform_guess": 0.25,
            "n": len(rows)}


# ---------------------------------------------------------------- R1c embedding diagnostics
def load_embeddings(e1: Path, model: str) -> tuple[np.ndarray, list[str]] | None:
    d = e1 / "probe" / "embeddings" / model
    parts = [d / f"{s}.npz" for s in ("train", "valid")]
    if not all(p.exists() for p in parts):
        return None
    X, ids = [], []
    for p in parts:
        z = np.load(p, allow_pickle=False)
        X.append(z["embeddings"].astype(np.float32))
        ids += [str(i) for i in z["image_ids"]]
    X = np.concatenate(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X, ids


def knn_scores(X, y, groups, k=10, chunk=512) -> tuple[list, list, np.ndarray]:
    """Leave-one-out kNN votes; (a) any neighbour, (b) neighbours from other duplicate groups only."""
    n = len(y)
    ga = np.array(groups)
    ya = np.array(y)
    pa, pb, nn1 = [], [], np.empty(n, dtype=int)
    for s in range(0, n, chunk):
        sim = X[s:s + chunk] @ X.T
        for r in range(sim.shape[0]):
            i = s + r
            sim[r, i] = -np.inf
            order = np.argsort(-sim[r])
            nn1[i] = order[0]
            pa.append(Counter(ya[order[:k]]).most_common(1)[0][0])
            other = order[ga[order] != ga[i]][:k]
            pb.append(Counter(ya[other]).most_common(1)[0][0])
    return pa, pb, nn1


def source_probe(X, fam, groups) -> float:
    from sklearn.linear_model import LogisticRegression
    folds = group_folds(groups)
    pred = np.empty(len(fam), dtype=object)
    fam = np.array(fam)
    for k in range(5):
        tr, te = folds != k, folds == k
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], fam[tr])
        pred[te] = clf.predict(X[te])
    return stats.macro_f1(list(fam), list(pred), labels=sorted(set(fam)))


def embedding_diagnostics(e1: Path, meta: dict[str, dict], groups_t1: dict[str, str]) -> list[dict]:
    out = []
    for m in MODELS:
        loaded = load_embeddings(e1, m)
        if loaded is None:
            print(f"  {m}: no cached embeddings - skipped")
            continue
        X, ids = loaded
        keep = [i for i, x in enumerate(ids) if x in meta]
        X, ids = X[keep], [ids[i] for i in keep]
        y = [meta[i]["class"] for i in ids]
        g = [groups_t1[i] for i in ids]
        stems = [meta[i]["rf_stem"] for i in ids]
        stem_n = Counter(s for s in stems if s)
        pa, pb, nn1 = knn_scores(X, y, g)
        has_copy = [i for i, s in enumerate(stems) if s and stem_n[s] >= 2]
        recall1 = float(np.mean([stems[nn1[i]] == stems[i] for i in has_copy])) if has_copy else float("nan")
        f1_a, f1_b = stats.macro_f1(y, pa), stats.macro_f1(y, pb)
        out.append({"model": m, "n_images": len(ids), "n_with_rf_copy": len(has_copy),
                    "copy_recall_at_1": round(recall1, 4),
                    "knn_f1_with_copies": round(f1_a, 4), "knn_f1_without_copies": round(f1_b, 4),
                    "knn_copy_gain": round(f1_a - f1_b, 4),
                    "source_family_probe_f1": round(source_probe(X, [meta[i]["family"] for i in ids], g), 4)})
        print(f"  {m}: copy recall@1 {recall1:.3f}, kNN F1 {f1_a:.3f} -> {f1_b:.3f} without copies, "
              f"source probe F1 {out[-1]['source_family_probe_f1']:.3f}")
    return out


# ---------------------------------------------------------------- R1d leakage link (T0)
def model_mode_of(run: str) -> tuple[str | None, str | None]:
    """Model key and mode from a saved run path, whatever the folder layout."""
    low = run.replace("\\", "/").lower()
    m = next((k for k in sorted(MODELS, key=len, reverse=True) if k in low), None)
    mode = next((x for x in ("frozen", "partial", "full") if f"__{x}__" in low or f"/{x}/" in low or f"_{x}_" in low), None)
    if mode is None and "probe" in low:
        mode = "probe"
    return m, mode


def stratified_gap(preds: list[dict], expo: dict[str, str], fam: dict[str, str]) -> float | None:
    """Accuracy(shared) - accuracy(independent) within class x family strata, weighted by shared count."""
    cells = defaultdict(lambda: {"shared": [], "independent": []})
    for r in preds:
        e = expo.get(r["image_id"])
        if e:
            cells[(r["true_class"], fam.get(r["image_id"], ""))][e].append(r["true_class"] == r["predicted_class"])
    num = den = 0.0
    for c in cells.values():
        if c["shared"] and c["independent"]:
            w = len(c["shared"])
            num += w * (np.mean(c["shared"]) - np.mean(c["independent"]))
            den += w
    return num / den if den else None


def leakage_link(root: Path, t0_runs: Path | None, diag: list[dict], t1: str) -> dict:
    a2 = root / "A2_rescoring_groupboot"
    table = read(a2 / "rescoring_by_run.csv")
    expo_rows = read(a2 / "valid_exposure.csv")
    expo = {r["image_id"]: r["exposure"] for r in expo_rows}
    fam = {r["image_id"]: r["source_family"] for r in expo_rows}
    per_model = defaultdict(lambda: {"gap": [], "strat_gap": []})
    strat_runs = []
    paths = {}
    if t0_runs and t0_runs.exists():
        paths = {str(p.parent.relative_to(t0_runs)): p for p in find_runs(t0_runs)}
    unmapped = []
    for r in table:
        m, mode = model_mode_of(r["run"])
        if m is None or mode is None:
            unmapped.append(r["run"])
            continue
        per_model[(m, mode)]["gap"].append(float(r["f1_shared"]) - float(r["f1_independent"]))
        if r["run"] in paths:
            sg = stratified_gap(read_predictions(paths[r["run"]]), expo, fam)
            if sg is not None:
                per_model[(m, mode)]["strat_gap"].append(sg)
                strat_runs.append({"run": r["run"], "model": m, "mode": mode, "stratified_acc_gap": round(sg, 4)})
    b3 = root / f"B3_{t1}"
    rank = {}
    if (b3 / "T_probe_T0.csv").exists() and (b3 / "T_probe_T1.csv").exists():
        for tag in ("T0", "T1"):
            rows = [x for x in read(b3 / f"T_probe_{tag}.csv") if x["mode"] == "probe"]
            for i, x in enumerate(sorted(rows, key=lambda x: -float(x["mean_macro_f1"]))):
                rank.setdefault(x["model"], {})[tag] = i + 1
    link = []
    d = {x["model"]: x for x in diag}
    for m in MODELS:
        g = per_model.get((m, "probe"), {"gap": [], "strat_gap": []})
        link.append({"model": m, "t0_probe_runs": len(g["gap"]),
                     "t0_gap_macro_f1_median": round(float(np.median(g["gap"])), 4) if g["gap"] else None,
                     "t0_gap_stratified_acc_median": round(float(np.median(g["strat_gap"])), 4) if g["strat_gap"] else None,
                     "rank_T0": rank.get(m, {}).get("T0"), "rank_T1": rank.get(m, {}).get("T1"),
                     "copy_recall_at_1": d.get(m, {}).get("copy_recall_at_1"),
                     "knn_copy_gain": d.get(m, {}).get("knn_copy_gain")})
    from scipy.stats import spearmanr

    def rho(a, b):
        pairs = [(x[a], x[b]) for x in link if x[a] is not None and x[b] is not None]
        if len(pairs) < 4:
            return None
        r_, p_ = spearmanr(*zip(*pairs))
        return {"rho": round(float(r_), 3), "p": round(float(p_), 3), "n_models": len(pairs)}

    for x in link:
        if x["rank_T0"] and x["rank_T1"]:
            x["rank_gain_on_T0"] = x["rank_T1"] - x["rank_T0"]     # >0: looked better on T0 than on T1
        else:
            x["rank_gain_on_T0"] = None
    all_gaps = [s["stratified_acc_gap"] for s in strat_runs]
    if unmapped:
        print(f"  {len(unmapped)} A2 runs not mapped to a model/mode, e.g. {unmapped[:3]}")
    return {"per_model": link, "stratified_runs": strat_runs, "unmapped_runs": unmapped,
            "stratified_gap_all_runs": {"n_runs": len(all_gaps),
                                        "median": round(float(np.median(all_gaps)), 4) if all_gaps else None,
                                        "n_positive": int(sum(g > 0 for g in all_gaps))},
            "spearman": {"copy_recall_vs_t0_gap": rho("copy_recall_at_1", "t0_gap_macro_f1_median"),
                         "knn_copy_gain_vs_t0_gap": rho("knn_copy_gain", "t0_gap_macro_f1_median"),
                         "copy_recall_vs_rank_gain_on_T0": rho("copy_recall_at_1", "rank_gain_on_T0")}}


# ---------------------------------------------------------------- R1b per-source test results
def per_source_test(root: Path, t1: str, meta_all: dict[str, dict], groups_t1: dict[str, str]) -> list[dict]:
    b4 = root / f"B4_TEST_{t1}"
    lock = b4 / "TEST_RUN_LOCK.json"
    if not lock.exists() or json.loads(lock.read_text()).get("state") != "COMPLETE":
        print("R1b skipped: single test run not COMPLETE")
        return []
    rows = []
    for name in ("selected_ensemble", "comparator_ensemble", "probe_baseline"):
        p = b4 / f"test_predictions_{name}.csv"
        if not p.exists():
            continue
        preds = read(p)
        by = defaultdict(list)
        for r in preds:
            by[meta_all[r["image_id"]]["family"]].append(r)
        for f, rs in sorted(by.items()):
            y = [r["true_class"] for r in rs]
            yp = [r["predicted_class"] for r in rs]
            ci = stats.bootstrap_metric(y, yp, n_boot=2000, groups=[groups_t1[r["image_id"]] for r in rs])
            rows.append({"system": name, "source_family": f, "n": len(rs),
                         "accuracy": round(float(np.mean(np.array(y) == np.array(yp))), 4),
                         "macro_f1": round(ci.estimate, 4), "ci_low": round(ci.low, 4), "ci_high": round(ci.high, 4)})
    return rows


def figure(out: Path, link: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [x for x in link if x["copy_recall_at_1"] is not None and x["t0_gap_macro_f1_median"] is not None]
    if len(pts) < 3:
        return
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.scatter([x["copy_recall_at_1"] for x in pts], [x["t0_gap_macro_f1_median"] for x in pts], s=24,
               color="#2a78d6", edgecolor="white", lw=0.6, zorder=3)
    for x in pts:
        ax.annotate(x["model"], (x["copy_recall_at_1"], x["t0_gap_macro_f1_median"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points", color="#0b0b0b")
    ax.axhline(0, color="#52514e", lw=0.8, ls="--")
    ax.set_xlabel("copy recall@1 (frozen embedding)")
    ax.set_ylabel("T0 gain on shared-source images\n(median macro-F1 difference, probe runs)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color="#e4e3df", lw=0.6)
    fig.savefig(out / "F7_copy_sensitivity_vs_leakage_gain.pdf", bbox_inches="tight")
    fig.savefig(out / "F7_copy_sensitivity_vs_leakage_gain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--t0-runs", default=None, help="DOAR_GPU_RESULTS/E1 (for the stratified A2 gap)")
    ap.add_argument("--t1", default="T1_v2")
    a = ap.parse_args(argv)
    root = Path(a.root)
    t1_manifest = root / a.t1 / "t1_partition_manifest.csv"
    man = read(t1_manifest)
    meta_all = {}
    for r in man:
        k = source_key(path_of(r))
        meta_all[r["image_id"]] = {"class": r["class"], "split": r["new_split"], "group_id": r["group_id"],
                                   "family": source_family(path_of(r)), "rf_stem": k.rf_stem or ""}
    groups_t1 = evaluation_groups(man)
    dev = {i: m for i, m in meta_all.items() if m["split"] in ("train", "valid")}
    out = root / "R1_representation_audit"
    out.mkdir(parents=True, exist_ok=True)

    print("R1a source vs label (T1 train+valid) ...")
    r1a = source_vs_label([{**m, "group_id": groups_t1[i]} for i, m in dev.items()])
    print(f"  Cramer's V(family, class) = {r1a['cramers_v_family_class']:.3f}; source-only macro-F1 "
          f"{r1a['source_only_macro_f1_groupcv']['estimate']:.3f} (majority-class {r1a['majority_class_macro_f1']:.3f})")
    print("R1c embedding diagnostics (T1 train+valid, cached frozen embeddings) ...")
    diag = embedding_diagnostics(root / f"E1_{a.t1}", dev, groups_t1)
    if diag:
        write_csv(out / "T_embedding_diagnostics.csv", diag)
    print("R1d link to T0 leakage gain ...")
    link = leakage_link(root, Path(a.t0_runs) if a.t0_runs else None, diag, a.t1)
    write_csv(out / "T_leakage_link_per_model.csv", link["per_model"])
    if link["stratified_runs"]:
        write_csv(out / "T_a2_stratified_gap_per_run.csv", link["stratified_runs"])
    print(f"  stratified A2 gap: {link['stratified_gap_all_runs']}")
    print(f"  Spearman (descriptive, 9 encoders): {link['spearman']}")
    figure(out, link["per_model"])
    print("R1b per-source results on the locked test set (saved predictions) ...")
    r1b = per_source_test(root, a.t1, meta_all, groups_t1)
    if r1b:
        write_csv(out / "T_test_by_source_family.csv", r1b)
        for x in r1b:
            print(f"  {x['system']:20s} {x['source_family']:15s} n={x['n']:4d} macro-F1 {x['macro_f1']:.3f} "
                  f"[{x['ci_low']:.3f}, {x['ci_high']:.3f}]")
    write_json(out / "R1_summary.json", {
        "status": "post hoc / exploratory; no selection, no new test prediction",
        "R1a_source_vs_label": r1a, "R1c_embedding_diagnostics": diag,
        "R1d_leakage_link": {k: v for k, v in link.items() if k != "stratified_runs"}, "R1b_test_by_source": r1b,
        "inputs": {"t1_manifest_sha256": sha256_file(t1_manifest)},
        "metadata": run_metadata("R1_representation_audit", vars(a))})
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
