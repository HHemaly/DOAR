"""
thesis.py — automate thesis-ready figures + tables from EXISTING outputs (Item 18).

Every figure is generated ONLY when its source experiment output exists, and each
figure is written together with its matching source data (JSON/CSV) — never from
invented results. Missing sources are reported as "not_generated", not fabricated.

A thesis_manifest.json records, for each figure, the source file it was built
from. Psychologist-agreement figures appear only when real reviews exist.
"""

from __future__ import annotations
import json
import shutil
from pathlib import Path

# The three data-provenance labels a cross-family comparison row may carry.
# EXPERIMENT_PROTOCOL.md Phase 8: "Never place these in the same table
# without explicit labels" -- build_cross_family_model_comparison() below
# REQUIRES the caller to pick exactly one per scan, never infers it, and
# stamps every row so a reader can never lose track of which is which.
DATA_PROVENANCE_LABELS = (
    "preliminary_contaminated_split",  # outputs/phase5/manifest.csv-family runs; leakage-gate overridden
    "clean_development_split",         # a gate-approved, frozen, duplicate-controlled train/valid split
    "locked_test",                     # a --unlock-test/--confirm-final-evaluation evaluation
    "smoke_test_not_a_result",         # synthetic/tiny pipeline-verification data; never a finding
)

# family_name -> (leaderboard filename to search for, JSON key holding the
# list of per-configuration result rows, key names for the metric/label
# fields within each row).
_FAMILY_LEADERBOARDS = {
    "A_objective_features": ("validation_leaderboard.json", "leaderboard", "model", "mean_macro_f1"),
    "B_handcrafted_groups": ("handcrafted_group_leaderboard.json", "leaderboard", "configuration", "mean_macro_f1"),
    "C_frozen_embeddings": ("embedding_classifier_leaderboard.json", "leaderboard", "model", "mean_macro_f1"),
    "D_E_deep_image_models": ("deep_comparison.json", "leaderboard", "model", "mean_valid_macro_f1"),
    "F_fusion": ("fusion_leaderboard.json", "leaderboard", "method", "mean_macro_f1"),
}


def build_cross_family_model_comparison(
    output_root: str | Path, *, data_provenance: str, output_path: str | Path | None = None,
) -> dict:
    """Scan `output_root` (recursively) for each family's own leaderboard
    JSON and pull out its single best (top-ranked) row into one unified,
    explicitly-labeled comparison table. Never mixes provenance labels
    within one call -- run this once per split/provenance and compare the
    written tables side by side, never merge them by hand into one file."""
    if data_provenance not in DATA_PROVENANCE_LABELS:
        raise ValueError(f"data_provenance must be one of {DATA_PROVENANCE_LABELS}, got {data_provenance!r}")
    root = Path(output_root)
    rows, missing = [], []
    for family, (filename, list_key, label_key, metric_key) in _FAMILY_LEADERBOARDS.items():
        matches = sorted(root.rglob(filename))
        if not matches:
            missing.append(f"{family}: {filename} not found under {root}")
            continue
        try:
            data = json.loads(matches[0].read_text(encoding="utf-8"))
            leaderboard = data.get(list_key) or []
            if not leaderboard:
                missing.append(f"{family}: {matches[0]} has no {list_key!r} entries")
                continue
            best = leaderboard[0]  # every family's own leaderboard is already sorted best-first
            rows.append({
                "family": family, "best_configuration": best.get(label_key),
                "mean_macro_f1": best.get(metric_key),
                "std_macro_f1": best.get("std_macro_f1") or best.get("std_valid_macro_f1"),
                "data_provenance": data_provenance,
                "source_file": str(matches[0]),
                "test_used": best.get("test_used", False),
            })
        except Exception as exc:
            missing.append(f"{family}: failed to read {matches[0]} ({exc})")

    result = {
        "data_provenance": data_provenance,
        "warning": (
            "This table is NOT a clean-split scientific result." if data_provenance != "clean_development_split"
            else "Clean-split development result -- still not the locked test set."
        ),
        "rows": rows, "missing_families": missing,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _emit(plt, fig, out_dir: Path, name: str, source_path: Path,
          source_data, manifest: list):
    """Save a figure (png+svg) AND its matching source data. Registers in manifest."""
    figures = out_dir / "figures"
    data = out_dir / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    data_path = data / f"{name}.json"
    data_path.write_text(json.dumps(source_data, indent=2), encoding="utf-8")
    manifest.append({"figure": f"figures/{name}.png",
                     "source_experiment": str(source_path),
                     "source_data": f"data/{name}.json"})


def _bar_from_leaderboard(plt, rows, value_key, label_key, std_key=None,
                          title="", ylabel=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r[label_key] for r in rows]
    vals = [r[value_key] for r in rows]
    errs = [r.get(std_key, 0) for r in rows] if std_key else None
    ax.bar(names, vals, yerr=errs, capsize=4, color="#3498db", edgecolor="white")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(vals + [0.01]) * 1.2)
    import matplotlib.pyplot as _p
    _p.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return fig


def generate_thesis_outputs(output_root) -> dict:
    """Scan output_root for known experiment artifacts and build matching figures.
    Returns a manifest of generated figures + a list of missing sources."""
    root = Path(output_root)
    thesis = root / "thesis"
    thesis.mkdir(parents=True, exist_ok=True)
    plt = _mpl()
    manifest: list = []
    missing: list = []

    def _load(rel):
        matches = [root / rel]
        matches.extend(sorted(root.rglob(Path(rel).name)))
        p = next((candidate for candidate in matches if candidate.exists()), None)
        if p is not None:
            try:
                return p, json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                return p, None
        missing.append(f"{rel} (searched recursively)")
        return None, None

    # Deep-model comparison (mean ± std validation macro-F1).
    p, data = _load("deep_comparison.json")
    if plt and data and data.get("leaderboard"):
        fig = _bar_from_leaderboard(
            plt, data["leaderboard"], "mean_valid_macro_f1", "model",
            "std_valid_macro_f1", "Deep model comparison (validation)", "macro-F1")
        _emit(plt, fig, thesis, "deep_model_comparison", p, data["leaderboard"], manifest)

    # Ablation chart.
    p, data = _load("ablation.json")
    if plt and data and data.get("results"):
        fig = _bar_from_leaderboard(
            plt, data["results"], "mean_macro_f1", "configuration", "std_macro_f1",
            "Feature ablation (validation)", "macro-F1")
        _emit(plt, fig, thesis, "ablation", p, data["results"], manifest)

    # Experiment A: objective-feature classical-classifier comparison.
    p, data = _load("validation_leaderboard.json")
    if plt and data and data.get("leaderboard"):
        fig = _bar_from_leaderboard(
            plt, data["leaderboard"], "mean_macro_f1", "model", "std_macro_f1",
            "Objective-feature classifier comparison (validation)", "macro-F1")
        _emit(plt, fig, thesis, "objective_feature_classifiers", p, data["leaderboard"], manifest)

    # Experiment B: HOG / colour / geometry handcrafted-feature ablation.
    p, data = _load("handcrafted_group_leaderboard.json")
    if plt and data and data.get("leaderboard"):
        fig = _bar_from_leaderboard(
            plt, data["leaderboard"], "mean_macro_f1", "configuration", "std_macro_f1",
            "Handcrafted feature group comparison: HOG/colour/geometry (validation)", "macro-F1")
        _emit(plt, fig, thesis, "handcrafted_group_comparison", p, data["leaderboard"], manifest)

    # Experiment C: frozen-embedding classifier comparison.
    p, data = _load("embedding_classifier_leaderboard.json")
    if plt and data and data.get("leaderboard"):
        fig = _bar_from_leaderboard(
            plt, data["leaderboard"], "mean_macro_f1", "model", "std_macro_f1",
            "Frozen-embedding classifier comparison (validation)", "macro-F1")
        _emit(plt, fig, thesis, "embedding_classifier_comparison", p, data["leaderboard"], manifest)

    # Fusion comparison.
    p, data = _load("fusion_leaderboard.json")
    if plt and data and data.get("leaderboard"):
        fig = _bar_from_leaderboard(
            plt, data["leaderboard"], "mean_macro_f1", "method", "std_macro_f1",
            "Fusion method comparison (validation)", "macro-F1")
        _emit(plt, fig, thesis, "fusion_comparison", p, data["leaderboard"], manifest)

    # Calibration ECE/Brier/NLL before vs after.
    p, data = _load("calibration.json")
    if plt and data and "validation_ece_before" in data:
        fig, ax = plt.subplots(figsize=(7, 5))
        metrics = ["ece", "brier", "nll"]
        before = [data.get(f"validation_{m}_before") for m in metrics]
        after = [data.get(f"validation_{m}_after") for m in metrics]
        x = range(len(metrics))
        ax.bar([i - 0.2 for i in x], before, width=0.4, label="before", color="#e74c3c")
        ax.bar([i + 0.2 for i in x], after, width=0.4, label="after", color="#27ae60")
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_title("Calibration: before vs after (validation)")
        ax.legend()
        _emit(plt, fig, thesis, "calibration_before_after", p,
              {k: data.get(k) for k in data if k.startswith("validation_")}, manifest)

    # Confusion matrix (from an evaluation metrics.json).
    p, data = _load("evaluation/metrics.json")
    if plt and data and data.get("confusion_matrix"):
        import numpy as np
        cm = np.array(data["confusion_matrix"], dtype=float)
        classes = data.get("class_order", [str(i) for i in range(len(cm))])
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_title("Confusion matrix (test)")
        fig.colorbar(im, fraction=.046)
        _emit(plt, fig, thesis, "confusion_matrix", p,
              {"confusion_matrix": data["confusion_matrix"], "class_order": classes}, manifest)

    # Psychologist agreement (ONLY when real reviews exist).
    p, data = _load("agreement.json")
    if plt and data and data.get("has_reviews"):
        per = data.get("per_item_agreement_pct", {})
        if per:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(list(per.keys()), list(per.values()), color="#9b59b6", edgecolor="white")
            ax.set_title("Psychologist agreement per item (%)")
            ax.set_ylim(0, 100)
            import matplotlib.pyplot as _p
            _p.setp(ax.get_xticklabels(), rotation=20, ha="right")
            _emit(plt, fig, thesis, "psychologist_agreement", p, data, manifest)
    elif data is not None and not data.get("has_reviews"):
        missing.append("agreement.json (no real reviews yet)")

    # Master manifest — every figure paired with its source data.
    (thesis / "thesis_manifest.json").write_text(json.dumps({
        "generated_figures": manifest,
        "not_generated_missing_sources": missing,
        "note": ("Figures are generated only from existing experiment outputs; "
                 "each has matching source data. Missing sources are not fabricated."),
    }, indent=2), encoding="utf-8")

    # A small tables collation: copy any *.csv leaderboards present.
    tables = thesis / "tables"
    tables.mkdir(exist_ok=True)
    for rel in ("ablation.csv", "fusion_leaderboard.csv", "deep_comparison.json"):
        src = next(iter(sorted(root.rglob(rel))), None)
        if src is not None:
            shutil.copy2(src, tables / src.name)

    return {"generated": len(manifest), "figures": manifest, "missing": missing,
            "thesis_dir": str(thesis)}
