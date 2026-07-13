"""
thesis_collate.py — gather thesis-ready figures and tables that already exist
in the outputs tree into a single outputs/thesis/ folder, and write a summary.

It copies (does not regenerate) artefacts produced by inspect / train / evaluate
so the thesis chapter has one place to look. It never invents metrics — if an
artefact is absent, it is listed as "not yet generated".
"""

from __future__ import annotations
import os
import json
import shutil
from pathlib import Path


def collate_thesis(out_dir: str) -> dict:
    out = Path(out_dir)
    thesis = out / "thesis"
    (thesis / "figures").mkdir(parents=True, exist_ok=True)
    (thesis / "tables").mkdir(parents=True, exist_ok=True)

    collected = {"figures": [], "tables": [], "missing": []}

    # Known artefact locations
    fig_sources = [
        out / "dataset_analysis" / "figures",
        out / "training" / "figures",
        out / "evaluation" / "figures",
    ]
    table_sources = [
        out / "dataset_analysis" / "dataset_summary.csv",
        out / "dataset_analysis" / "class_distribution.csv",
        out / "evaluation" / "classification_report.csv",
        out / "evaluation" / "per_class_metrics.csv",
        out / "evaluation" / "predictions_test.csv",
    ]

    for src in fig_sources:
        if src.exists():
            for f in src.glob("*.*"):
                if f.suffix.lower() in (".png", ".svg"):
                    shutil.copy2(f, thesis / "figures" / f.name)
                    collected["figures"].append(f.name)
        else:
            collected["missing"].append(str(src))

    for src in table_sources:
        if src.exists():
            shutil.copy2(src, thesis / "tables" / src.name)
            collected["tables"].append(src.name)
        else:
            collected["missing"].append(str(src))

    # Pull headline metrics if evaluation ran
    metrics_path = out / "evaluation" / "metrics.json"
    headline = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        headline = {k: m.get(k) for k in
                    ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
                     "n_test", "roc_auc_macro_ovr", "roc_auc")}

    # Psychologist agreement (only if real reviews exist)
    agreement = {}
    master_csv = out / "psychologist_review" / "review_master.csv"
    if master_csv.exists():
        try:
            from src.reports.psych_review import compute_agreement
            agreement = compute_agreement(str(master_csv))
            with open(thesis / "tables" / "psychologist_agreement.json",
                      "w", encoding="utf-8") as f:
                json.dump(agreement, f, indent=2, ensure_ascii=False)
        except Exception as e:
            agreement = {"error": str(e)}

    # Count generated example folders
    examples_dir = out / "examples"
    n_examples = len([d for d in examples_dir.iterdir() if d.is_dir()]) if examples_dir.exists() else 0

    summary_md = thesis / "thesis_results_summary.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# DOAR — Thesis Results Summary\n\n")
        if headline:
            f.write("## Headline test metrics\n\n")
            for k, v in headline.items():
                if v is not None:
                    f.write(f"- **{k}**: {v}\n")
            f.write("\n")
        else:
            f.write("_No evaluation metrics found yet. Run `python main.py evaluate`._\n\n")
        f.write(f"## Collected figures ({len(collected['figures'])})\n\n")
        for fig in sorted(collected["figures"]):
            f.write(f"- figures/{fig}\n")
        f.write(f"\n## Collected tables ({len(collected['tables'])})\n\n")
        for t in sorted(collected["tables"]):
            f.write(f"- tables/{t}\n")
        f.write(f"\n## Example case folders: {n_examples}\n\n")
        if agreement:
            f.write("## Psychologist agreement\n\n")
            if agreement.get("has_reviews"):
                f.write(f"- reviewers: {agreement.get('n_reviewers')}\n")
                f.write(f"- items reviewed: {agreement.get('n_items_reviewed')}\n")
                f.write(f"- raw agreement: {agreement.get('raw_agreement_pct')}%\n")
                if "cohen_kappa" in agreement:
                    f.write(f"- Cohen's kappa: {agreement['cohen_kappa']}\n")
                if "fleiss_kappa" in agreement:
                    f.write(f"- Fleiss' kappa: {agreement['fleiss_kappa']}\n")
            else:
                f.write("_No psychologist reviews submitted yet — agreement not "
                        "computed (nothing fabricated)._\n")
            f.write("\n")
        if collected["missing"]:
            f.write("## Not yet generated\n\n")
            for miss in collected["missing"]:
                f.write(f"- {miss}\n")

    print(f"[thesis] figures={len(collected['figures'])} "
          f"tables={len(collected['tables'])} -> {thesis}")
    return collected
