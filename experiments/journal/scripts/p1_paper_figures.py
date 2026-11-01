"""P1 — paper figures from results that already exist (CPU, read-only).

  python experiments/journal/scripts/p1_paper_figures.py --root <Drive>/DOAR_JOURNAL_2026 \
      --dataset-root <Combined_Drawing> [--e2-holdout-review <...>/E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv]

Draws only what has been computed; a figure whose inputs do not exist yet is
skipped with a message (no placeholder is ever drawn). Every figure is written
as PDF (for the manuscript) and PNG (to look at), with a JSON record of the
input files and their SHA-256.

  F1_duplicate_pairs        confirmed Roboflow-copy pairs (train/valid only) and
                            same-filename pairs judged DIFFERENT drawings (A5)
  F2_shared_vs_independent  per saved T0 run: macro-F1 on validation images that
                            share a source with train vs independent images (A2)
  F3_probe_T0_vs_T1         linear-probe macro-F1 per encoder on T0 vs T1 (B3)
  F4_adaptation_T1          adaptation macro-F1 per encoder x mode, seed mean +- SD (B3)
  F5_confusion_test         test confusion matrices: selected, comparator, probe (B4)
  F6_v2_mask_gallery        original / A0 / frozen V2 masks by C1 verdict (C1)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal.provenance import sha256_file, write_json  # noqa: E402

CLASSES = ["Angry", "Fear", "Happy", "Sad"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3 (validated reference palette)


def read(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "pdf.fonttype": 42,
                         "savefig.bbox": "tight", "figure.dpi": 100})
    return plt


def save(fig, out: Path, name: str, inputs: list[Path], note: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=220)
    write_json(out / f"{name}.json", {"figure": name, "note": note,
                                      "inputs": {str(p): sha256_file(p) for p in inputs if p.exists()}})
    print("wrote", out / f"{name}.pdf")


def load_img(root: Path, rel: str, side: int = 360) -> np.ndarray:
    from PIL import Image
    rel = rel.replace("\\", "/")
    p = root / rel
    if not p.exists():
        p = root.joinpath(*rel.split("/")[-3:])
    with Image.open(p) as im:
        im = im.convert("RGB")
        im.thumbnail((side, side))
        return np.asarray(im)


# ---------------------------------------------------------------------------------------------
def f1_duplicate_pairs(plt, root: Path, ds: Path, out: Path, n_same: int = 4, n_diff: int = 2):
    rev = root / "A5_pair_review"
    ps, rs = rev / "pairs_rf_stem.csv", rev / "review_rf_stem.csv"
    pn, rn = rev / "pairs_name_key.csv", rev / "review_name_key.csv"
    if not (ps.exists() and rs.exists()):
        return print("F1 skipped: A5 rf_stem review not found")
    verdict = {r["pair_id"]: r for r in read(rs)}
    same = [p for p in read(ps) if p["pair_type_hidden"].startswith("same")
            and verdict.get(p["pair_id"], {}).get("verdict") == "SAME_DRAWING"]
    same = sorted(same, key=lambda p: (verdict[p["pair_id"]]["transform_seen"] == "none", p["pair_id"]))[:n_same]
    diff = []
    if pn.exists() and rn.exists():
        vn = {r["pair_id"]: r for r in read(rn)}
        diff = [p for p in read(pn) if p["pair_type_hidden"].startswith("same")
                and vn.get(p["pair_id"], {}).get("verdict") == "DIFFERENT_DRAWING"][:n_diff]
    rows = [(p, "same drawing (" + verdict[p["pair_id"]]["transform_seen"].replace("_", " ") + ")") for p in same] + \
           [(p, "same file name, different drawing") for p in diff]
    if not rows:
        return print("F1 skipped: no judged pairs")
    fig, axes = plt.subplots(len(rows), 2, figsize=(3.6, 2.05 * len(rows)),
                             gridspec_kw={"wspace": 0.06, "hspace": 0.45})
    axes = np.atleast_2d(axes)
    for i, (p, label) in enumerate(rows):
        for j, k in enumerate(("a", "b")):
            ax = axes[i, j]
            ax.imshow(load_img(ds, p[f"path_{k}"]))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_xlabel(p[f"split_{k}"], fontsize=6.5, color=INK2, labelpad=2)
        axes[i, 0].set_title(label, fontsize=7.5, color=INK, loc="left")
    save(fig, out, "F1_duplicate_pairs", [ps, rs, pn, rn], "pairs chosen by fixed rule: first by pair id, "
         "rotation/flip copies first; same-name pairs judged different shown as counter-examples")
    plt.close(fig)


def f2_shared_vs_independent(plt, root: Path, out: Path):
    p = root / "A2_rescoring_groupboot" / "rescoring_by_run.csv"
    if not p.exists():
        return print("F2 skipped: A2 not found")
    rows = read(p)
    x = np.array([float(r["f1_independent"]) for r in rows])
    y = np.array([float(r["f1_shared"]) for r in rows])
    sig = np.array([float(r["diff_ci_low"]) > 0 for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    lo, hi = min(x.min(), y.min()) - 0.03, max(x.max(), y.max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--", zorder=1)
    ax.scatter(x[~sig], y[~sig], s=14, color=S1, alpha=0.55, edgecolor="white", lw=0.5,
               label=f"CI includes 0 (n={int((~sig).sum())})", zorder=2)
    ax.scatter(x[sig], y[sig], s=14, color=S2, edgecolor="white", lw=0.5,
               label=f"group-bootstrap CI above 0 (n={int(sig.sum())})", zorder=3)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi), ax.set_aspect("equal")
    ax.set_xlabel("macro-F1, independent validation images")
    ax.set_ylabel("macro-F1, images sharing a source with train")
    above = int((y > x).sum())
    ax.set_title(f"Saved T0 runs: {above} of {len(rows)} above the diagonal", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, out, "F2_shared_vs_independent", [p], "exploratory association; subsets also differ in class mix "
         "and source family")
    plt.close(fig)


def f3_probe(plt, root: Path, t1: str, out: Path):
    b3 = root / f"B3_{t1}"
    p0, p1 = b3 / "T_probe_T0.csv", b3 / "T_probe_T1.csv"
    if not (p0.exists() and p1.exists()):
        return print("F3 skipped: B3 probe tables not found")
    r0 = {r["model"]: float(r["mean_macro_f1"]) for r in read(p0) if r["mode"] == "probe"}
    r1 = {r["model"]: float(r["mean_macro_f1"]) for r in read(p1) if r["mode"] == "probe"}
    models = sorted(set(r0) & set(r1), key=lambda m: -r1[m])
    rank0 = {m: i + 1 for i, m in enumerate(sorted(models, key=lambda m: -r0[m]))}
    rank1 = {m: i + 1 for i, m in enumerate(models)}
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    for m in models:
        moved = abs(rank0[m] - rank1[m]) >= 3
        c = S2 if moved else S1
        ax.plot([0, 1], [r0[m], r1[m]], color=c, lw=2 if moved else 1.2, marker="o", ms=4, zorder=3 if moved else 2)
        ax.text(1.04, r1[m], f"{m}  ({rank0[m]}→{rank1[m]})", va="center", fontsize=6.5, color=INK)
    ax.set_xticks([0, 1], ["T0 split\n(copies across splits)", "T1 split\n(source-grouped)"])
    ax.set_xlim(-0.15, 1.9)
    ax.set_ylabel("linear-probe validation macro-F1 (10-seed mean)")
    ax.set_title("Encoder ranking before and after source grouping", fontsize=8, color=INK)
    ax.grid(axis="x", visible=False)
    save(fig, out, "F3_probe_T0_vs_T1", [p0, p1], "orange = rank changed by 3 or more places; T0 and T1 "
         "validation sets differ, so absolute values are not paired")
    plt.close(fig)


def f4_adapt(plt, root: Path, t1: str, out: Path):
    p = root / f"B3_{t1}" / "T_adapt_T1.csv"
    if not p.exists():
        return print("F4 skipped: B3 adaptation table not found")
    rows = [r for r in read(p) if r["mode"] in ("frozen", "partial", "full")]
    models = sorted({r["model"] for r in rows})
    modes = ["frozen", "partial", "full"]
    col = dict(zip(modes, (S1, S2, S3)))
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for i, m in enumerate(models):
        for j, md in enumerate(modes):
            r = next((r for r in rows if r["model"] == m and r["mode"] == md), None)
            if r is None:
                continue
            x = i + (j - 1) * 0.22
            mu, sd = float(r["mean_macro_f1"]), float(r["sd_macro_f1"])
            ax.errorbar(x, mu, yerr=sd, fmt="o", color=col[md], ms=5, lw=1.2, capsize=0,
                        label=md if i == 0 else None)
            ax.text(x, mu + sd + 0.006, f"n={r['n_seeds']}", ha="center", fontsize=5.5, color=INK2)
    ax.set_xticks(range(len(models)), models)
    ax.set_ylabel("validation macro-F1 (seed mean ± SD)")
    ax.set_title("Adaptation on T1 (shortlisted encoders)", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="x", visible=False)
    save(fig, out, "F4_adaptation_T1", [p], "n = seeds; finalists have 5 seeds, others 3")
    plt.close(fig)


def f5_confusion(plt, root: Path, t1: str, out: Path):
    b4 = root / f"B4_TEST_{t1}"
    lock = b4 / "TEST_RUN_LOCK.json"
    names = [("selected_ensemble", "selected"), ("comparator_ensemble", "comparator"), ("probe_baseline", "linear probe")]
    files = [b4 / f"test_predictions_{n}.csv" for n, _ in names]
    if not lock.exists() or json.loads(lock.read_text()).get("state") != "COMPLETE" or not all(f.exists() for f in files):
        return print("F5 skipped: the single test run (B4) is not COMPLETE yet")
    sel = json.loads((root / f"B3_{t1}" / "SELECTION_FROZEN.json").read_text())
    titles = {"selected": f"{sel['selected']['model']} {sel['selected']['mode']}",
              "comparator": f"{sel['comparator']['model']} {sel['comparator']['mode']}",
              "linear probe": f"{sel['probe_baseline']['model']} probe"}
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    fig.subplots_adjust(wspace=0.12)
    for ax, f, (_, role) in zip(axes, files, names):
        rows = read(f)
        cm = np.zeros((4, 4), int)
        for r in rows:
            cm[CLASSES.index(r["true_class"]), CLASSES.index(r["predicted_class"])] += 1
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for a in range(4):
            for b in range(4):
                ax.text(b, a, f"{cm[a, b]}", ha="center", va="center", fontsize=7,
                        color="white" if norm[a, b] > 0.55 else INK)
        ax.set_xticks(range(4), CLASSES, fontsize=6.5, rotation=30)
        ax.set_yticks(range(4), CLASSES if ax is axes[0] else [], fontsize=6.5)
        ax.grid(False)
        ax.set_xlabel("predicted")
        ax.set_title(f"{role}: {titles[role]}", fontsize=7.5, color=INK)
    axes[0].set_ylabel("benchmark label")
    fig.suptitle(f"Locked test set (n = {len(read(files[0]))}); counts, shading = row share", fontsize=8, color=INK)
    save(fig, out, "F5_confusion_test", files + [lock], "counts per cell; shading normalised per true class")
    plt.close(fig)


def t_test_summary(root: Path, t1: str, out: Path):
    """Table of the single test result (macro-F1 with group-bootstrap CI, accuracy, per-class F1, paired diffs)."""
    p = root / f"B4_TEST_{t1}" / "test_results.json"
    if not p.exists():
        return print("T_test skipped: test_results.json not found")
    r = json.loads(p.read_text())
    rows = []
    for name in ("selected_ensemble", "comparator_ensemble", "probe_baseline"):
        v = r[name]
        m = v["macro_f1"]
        rows.append({"system": name, "macro_f1": round(m["estimate"], 4), "ci_low": round(m["ci_low"], 4),
                     "ci_high": round(m["ci_high"], 4), "accuracy": round(v["accuracy"], 4), "n": v["n"],
                     "n_groups": v["n_groups"], **{f"f1_{c}": round(v["per_class_f1"][c], 4) for c in CLASSES}})
    seeds = {role: [r[k]["macro_f1"]["estimate"] for k in r if k.startswith(f"{role}_seed_")]
             for role in ("selected", "comparator")}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "T_test_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["system                 macro-F1 [95% CI, group bootstrap]   accuracy"]
    lines += [f"{x['system']:22s} {x['macro_f1']:.3f} [{x['ci_low']:.3f}, {x['ci_high']:.3f}]           {x['accuracy']:.3f}"
              for x in rows]
    for role, v in seeds.items():
        lines.append(f"{role} single seeds: mean {np.mean(v):.3f}, SD {np.std(v, ddof=1):.3f}, range {min(v):.3f}-{max(v):.3f}")
    for k in ("selected_vs_comparator_ensemble", "selected_vs_probe_baseline"):
        d = r[k]["paired_group_bootstrap"]
        mc = r[k]["mcnemar_counts_descriptive"]
        lines.append(f"{k}: diff {d['estimate']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
                     f"only-selected-right {mc['a_right_b_wrong']}, only-other-right {mc['a_wrong_b_right']}")
    (out / "T_test_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def f6_gallery(plt, review_csv: Path | None, ds: Path, out: Path, per: int = 2):
    if review_csv is None or not review_csv.exists():
        return print("F6 skipped: holdout review CSV not given/found")
    rows = [r for r in read(review_csv) if r.get("review_complete", "").lower() == "true"]
    if not rows:
        return print("F6 skipped: holdout review not completed")
    from doar_journal import foreground as fg
    order = ["GOOD", "OVERSEGMENTED", "FAILED", "PARTIAL", "UNDERSEGMENTED", "APPROPRIATELY_AMBIGUOUS"]
    counts = Counter(r["visual_mask_quality"] for r in rows)
    chosen = []
    for v in order:
        chosen += sorted([r for r in rows if r["visual_mask_quality"] == v], key=lambda r: r.get("image_id", ""))[:per]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(5.0, 1.75 * len(chosen)))
    axes = np.atleast_2d(axes)
    for i, r in enumerate(chosen):
        img = fg.load_rgb(fg_path(ds, r["relative_path"]))
        panels = (img, fg.a0(img).mask, fg.v2(img).mask)
        for j, (x, t) in enumerate(zip(panels, ("original", "A0 mask", "frozen V2 mask"))):
            ax = axes[i, j]
            ax.imshow(x if j == 0 else x.astype(np.uint8) * 255, cmap=None if j == 0 else "gray", vmin=0, vmax=255)
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(t, fontsize=7.5, color=INK)
        v = r["visual_mask_quality"]
        axes[i, 0].set_ylabel(f"{v.lower().replace('_', ' ')}\n({counts[v]} of {len(rows)})", fontsize=6.5,
                              rotation=0, ha="right", va="center", labelpad=6, color=INK)
    fig.tight_layout()
    save(fig, out, "F6_v2_mask_gallery", [review_csv], f"first {per} images per verdict by image id; "
         "single-reviewer visual audit")
    plt.close(fig)


def fg_path(ds: Path, rel: str) -> Path:
    rel = rel.replace("\\", "/")
    p = ds / rel
    return p if p.exists() else ds.joinpath(*rel.split("/")[-3:])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--t1", default="T1_v2")
    ap.add_argument("--e2-holdout-review", default=None)
    a = ap.parse_args(argv)
    root, ds = Path(a.root), Path(a.dataset_root)
    out = root / "PAPER_FIGURES"
    plt = style()
    f1_duplicate_pairs(plt, root, ds, out)
    f2_shared_vs_independent(plt, root, out)
    f3_probe(plt, root, a.t1, out)
    f4_adapt(plt, root, a.t1, out)
    f5_confusion(plt, root, a.t1, out)
    t_test_summary(root, a.t1, out)
    f6_gallery(plt, Path(a.e2_holdout_review) if a.e2_holdout_review else None, ds, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
