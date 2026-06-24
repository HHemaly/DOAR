"""
output_manager.py — Save per-image analysis results and generate report files.

Creates an organised output folder structure:
  outputs/
    YYYY-MM-DD_HH-MM-SS/
      per_image/
        <stem>/
          analysis_en.json    — full English analysis (all fields)
          analysis_ar.json    — Arabic translation of parent-facing output
          report_card.png     — visual report card for psychologist review
          feature_chart.png   — color / composition / stroke breakdown
      summary_report.json     — dataset-level summary
      thesis_figures/
        overview_grid.png
        emotion_distribution.png
        rule_frequency.png
        validation_summary.png
"""

from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for Windows
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    _MPL = False

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False


# ---------------------------------------------------------------------------
# Output session directory
# ---------------------------------------------------------------------------

def make_session_dir(base_output: str = "outputs") -> str:
    """Create a timestamped session folder and return its path."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session = os.path.join(base_output, ts)
    os.makedirs(os.path.join(session, "per_image"), exist_ok=True)
    os.makedirs(os.path.join(session, "thesis_figures"), exist_ok=True)
    return session


def make_image_dir(session_dir: str, image_stem: str) -> str:
    d = os.path.join(session_dir, "per_image", image_stem)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Save per-image JSON results
# ---------------------------------------------------------------------------

def save_image_results(
    image_dir: str,
    doc: dict,
    parent_output: dict,
    judgment: dict,
    parent_output_ar: dict | None = None,
) -> dict:
    """Save English and Arabic JSONs for one image. Return file paths."""
    paths = {}

    # English
    full_result = {
        "image":           doc.get("source_image", ""),
        "dataset_label":   doc.get("metadata", {}).get("label_from_dataset", ""),
        "image_quality":   doc.get("image_quality", {}),
        "color_features":  doc.get("color_features", {}),
        "composition":     doc.get("composition_features", {}),
        "stroke_features": doc.get("stroke_features", {}),
        "ocr_results":     doc.get("ocr_results", []),
        "detected_objects": doc.get("detected_objects", []),
        "emotional_tendency": doc.get("feature_based_emotional_tendency", {}),
        "rule_activations_v1": doc.get("psychological_rule_activations", []),
        "rule_activations_v2": doc.get("psychological_rule_activations_v2", []),
        "theme_scores":    doc.get("theme_scores_v2", {}),
        "validation_summary": doc.get("validation_summary", {}),
        "analysis_en": {
            "parent_answer":    parent_output.get("parent_answer", ""),
            "gentle_questions": parent_output.get("gentle_questions", []),
            "safety_note":      parent_output.get("safety_note", ""),
            "disclaimer":       parent_output.get("disclaimer", ""),
        },
        "final_judgment": {
            "status":       judgment.get("final_answer_status", ""),
            "safe_to_show": judgment.get("safe_to_show", False),
            "checks_passed": judgment.get("checks_passed", 0),
            "checks_total":  judgment.get("checks_total", 10),
            "issues":        judgment.get("issues", []),
        },
    }

    if parent_output_ar:
        full_result["analysis_ar"] = parent_output_ar

    en_path = os.path.join(image_dir, "analysis_en.json")
    with open(en_path, "w", encoding="utf-8") as f:
        json.dump(full_result, f, indent=2, ensure_ascii=False)
    paths["analysis_en"] = en_path

    if parent_output_ar:
        ar_path = os.path.join(image_dir, "analysis_ar.json")
        with open(ar_path, "w", encoding="utf-8") as f:
            json.dump(parent_output_ar, f, indent=2, ensure_ascii=False)
        paths["analysis_ar"] = ar_path

    return paths


# ---------------------------------------------------------------------------
# Per-image report card (psychologist view)
# ---------------------------------------------------------------------------

def save_report_card(
    image_dir: str,
    image_path: str,
    doc: dict,
    parent_output: dict,
    judgment: dict,
    validated_claims: list[dict] | None = None,
) -> str | None:
    """Generate and save a report card PNG for psychologist review."""
    if not _MPL:
        return None

    cf   = doc.get("color_features", {})
    comp = doc.get("composition_features", {})
    sf   = doc.get("stroke_features", {})
    iq   = doc.get("image_quality", {})
    ht   = doc.get("feature_based_emotional_tendency", {})
    label = doc.get("metadata", {}).get("label_from_dataset", "unknown")

    # Collect active rules
    v1 = [r for r in doc.get("psychological_rule_activations", []) if r.get("activated")]
    v2 = [r for r in doc.get("psychological_rule_activations_v2", []) if r.get("activated")]
    all_rules = v1 + v2

    fig = plt.figure(figsize=(18, 22), facecolor="#f8f9fa")
    fig.suptitle(
        f"DOAR Analysis Report Card\n{Path(image_path).name}  |  Dataset label: {label}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: the drawing itself
    ax_img = fig.add_subplot(gs[0, 0])
    if _CV2:
        import cv2 as _cv2
        img = _cv2.imread(image_path)
        if img is not None:
            img_rgb = _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
            ax_img.imshow(img_rgb)
    ax_img.set_title("Drawing", fontsize=10, fontweight="bold")
    ax_img.axis("off")

    # Panel 2: color breakdown
    ax_col = fig.add_subplot(gs[0, 1])
    ratios = cf.get("color_ratios", {})
    col_data = {k: v for k, v in ratios.items() if v > 0.005 and k not in ("white",)}
    if col_data:
        COLOR_HEX = {
            "red": "#e74c3c", "blue": "#3498db", "green": "#27ae60",
            "yellow": "#f1c40f", "black": "#2c3e50", "orange": "#e67e22",
            "purple": "#9b59b6", "brown": "#795548", "red2": "#c0392b",
        }
        colors_hex = [COLOR_HEX.get(k, "#95a5a6") for k in col_data]
        ax_col.bar(list(col_data.keys()), list(col_data.values()),
                   color=colors_hex, edgecolor="white")
        ax_col.set_title("Color Distribution", fontsize=10, fontweight="bold")
        ax_col.set_ylabel("Ratio")
        ax_col.tick_params(axis="x", labelsize=7)
    else:
        ax_col.text(0.5, 0.5, "No color data", ha="center", va="center")
        ax_col.set_title("Color Distribution", fontsize=10)
    ax_col.axis("on")

    # Panel 3: metrics summary
    ax_met = fig.add_subplot(gs[0, 2])
    ax_met.axis("off")
    emotion = ht.get("estimated_emotion", "neutral_or_unclear")
    metrics = [
        ("Quality score",    f"{iq.get('quality_score', 0):.2f}"),
        ("Empty space",      f"{comp.get('empty_space_ratio', 0)*100:.0f}%"),
        ("Dark tones",       f"{cf.get('dark_dominance', 0)*100:.0f}%"),
        ("Warm tones",       f"{cf.get('warm_dominance', 0)*100:.0f}%"),
        ("Color diversity",  str(cf.get("color_diversity_count", 0))),
        ("Fragmentation",    f"{sf.get('fragmentation_ratio', 0)*100:.0f}%"),
        ("Heuristic emotion",emotion),
        ("Rules activated",  str(len(all_rules))),
        ("Judge status",     judgment.get("final_answer_status", "?")),
    ]
    tbl = ax_met.table(
        cellText=[[k, v] for k, v in metrics],
        colLabels=["Metric", "Value"],
        cellLoc="left", loc="center",
        colWidths=[0.62, 0.38],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.3)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if r % 2 == 0 else "white")
    ax_met.set_title("Key Metrics", fontsize=10, fontweight="bold", pad=15)

    # Panel 4: parent-facing answer
    ax_ans = fig.add_subplot(gs[1, :])
    ax_ans.axis("off")
    answer = parent_output.get("parent_answer", "No answer generated.")
    ax_ans.text(
        0.01, 0.98, "PARENT-FACING ANSWER (English)",
        transform=ax_ans.transAxes, fontsize=9, fontweight="bold",
        va="top", color="#2c3e50",
    )
    ax_ans.text(
        0.01, 0.85, answer[:1200],
        transform=ax_ans.transAxes, fontsize=8, va="top", wrap=True,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eaf4fb", alpha=0.9),
    )
    qs = parent_output.get("gentle_questions", [])
    if qs:
        q_text = "Gentle questions:\n" + "\n".join(f"  • {q}" for q in qs[:4])
        ax_ans.text(
            0.01, 0.22, q_text,
            transform=ax_ans.transAxes, fontsize=8, va="top", color="#1a5276",
        )

    # Panel 5: activated rules
    ax_rules = fig.add_subplot(gs[2, :])
    ax_rules.axis("off")
    ax_rules.text(
        0.01, 0.99, "ACTIVATED PSYCHOLOGICAL RULES",
        transform=ax_rules.transAxes, fontsize=9, fontweight="bold",
        va="top", color="#2c3e50",
    )
    if all_rules:
        rule_lines = []
        for r in all_rules[:8]:
            rid   = r.get("rule_id", "?")
            tier  = r.get("tier", "?")
            interp = r.get("interpretation", "")[:130]
            sources = ", ".join(r.get("sources", [])[:2])
            conf  = r.get("rule_confidence", 0)
            rule_lines.append(
                f"[{rid}] Tier {tier} | conf={conf:.2f}\n"
                f"  {interp}\n"
                f"  Sources: {sources}"
            )
        rule_text = "\n\n".join(rule_lines)
    else:
        rule_text = "No psychological rules activated for this drawing."
    ax_rules.text(
        0.01, 0.88, rule_text,
        transform=ax_rules.transAxes, fontsize=7.5, va="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdfefe", alpha=0.9),
    )

    # Panel 6: disclaimer
    ax_disc = fig.add_subplot(gs[3, :])
    ax_disc.axis("off")
    disc_text = parent_output.get("disclaimer", "")
    safety    = parent_output.get("safety_note", "")
    ax_disc.text(
        0.01, 0.95,
        f"DISCLAIMER: {disc_text}\n\nSAFETY NOTE: {safety}",
        transform=ax_disc.transAxes, fontsize=8, va="top", style="italic",
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9e7", alpha=0.9),
    )

    out_path = os.path.join(image_dir, "report_card.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Thesis-ready figures from a batch of results
# ---------------------------------------------------------------------------

def save_thesis_figures(session_dir: str, all_results: list[dict]) -> dict:
    """Generate dataset-level thesis figures from a list of per-image result dicts."""
    if not _MPL or not all_results:
        return {}

    thesis_dir = os.path.join(session_dir, "thesis_figures")
    paths = {}

    # ── 1. Emotion distribution ──────────────────────────────────
    emotions = [r.get("emotion", "neutral_or_unclear") for r in all_results]
    em_counts: dict[str, int] = {}
    for e in emotions:
        em_counts[e] = em_counts.get(e, 0) + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    EM_COLORS = {
        "happy": "#2ecc71", "sad": "#3498db", "angry": "#e74c3c",
        "fear": "#9b59b6", "neutral_or_unclear": "#95a5a6",
    }
    bars = ax.bar(
        list(em_counts.keys()), list(em_counts.values()),
        color=[EM_COLORS.get(e, "#bdc3c7") for e in em_counts],
        edgecolor="white", linewidth=1.5,
    )
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Heuristic Emotional Tendency Distribution\n(feature-based, not a trained model)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of drawings")
    ax.set_xlabel("Estimated emotional tendency")
    ax.set_ylim(0, max(em_counts.values(), default=1) * 1.25)
    _style_ax(ax)
    p = os.path.join(thesis_dir, "emotion_distribution.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["emotion_distribution"] = p

    # ── 2. Rule frequency ────────────────────────────────────────
    rule_counts: dict[str, int] = {}
    for r in all_results:
        for rule in r.get("rules_activated", []):
            rid = rule.get("rule_id", "?")
            rule_counts[rid] = rule_counts.get(rid, 0) + 1
    if rule_counts:
        sorted_rules = sorted(rule_counts.items(), key=lambda x: -x[1])[:15]
        labels = [x[0] for x in sorted_rules]
        counts = [x[1] for x in sorted_rules]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(labels[::-1], counts[::-1], color="#3498db", edgecolor="white")
        ax.bar_label(bars, padding=3, fontsize=9)
        ax.set_title("Most Frequently Activated Rules (across dataset)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of drawings")
        _style_ax(ax)
        p = os.path.join(thesis_dir, "rule_frequency.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["rule_frequency"] = p

    # ── 3. Validation summary ────────────────────────────────────
    total = len(all_results)
    judge_counts = {"PASS": 0, "REWRITE_REQUIRED": 0, "BLOCK": 0}
    for r in all_results:
        s = r.get("judge_status", "PASS")
        judge_counts[s] = judge_counts.get(s, 0) + 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Judge pie
    ax_pie = axes[0]
    pie_labels = [k for k, v in judge_counts.items() if v > 0]
    pie_vals   = [v for v in judge_counts.values() if v > 0]
    PIE_COLORS = {"PASS": "#2ecc71", "REWRITE_REQUIRED": "#f39c12", "BLOCK": "#e74c3c"}
    ax_pie.pie(
        pie_vals, labels=pie_labels,
        colors=[PIE_COLORS.get(l, "#bdc3c7") for l in pie_labels],
        autopct="%1.0f%%", startangle=140,
    )
    ax_pie.set_title("Final Judge Verdict Distribution", fontsize=11, fontweight="bold")

    # Avg claims verified
    ax_bar = axes[1]
    avg_verified = sum(r.get("verified_claims", 0) for r in all_results) / max(total, 1)
    avg_uncertain = sum(r.get("uncertain_claims", 0) for r in all_results) / max(total, 1)
    ax_bar.bar(["Avg Verified", "Avg Uncertain"], [avg_verified, avg_uncertain],
               color=["#2ecc71", "#f39c12"], edgecolor="white")
    ax_bar.set_title("Average Claims per Drawing", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Claim count")
    _style_ax(ax_bar)

    p = os.path.join(thesis_dir, "validation_summary.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["validation_summary"] = p

    # ── 4. Feature overview grid ─────────────────────────────────
    empties  = [r.get("empty_space_pct", 50) for r in all_results]
    darks    = [r.get("dark_ratio_pct", 20)  for r in all_results]
    colors_n = [r.get("color_count", 3)       for r in all_results]
    labels   = [r.get("label", "?") for r in all_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, title, color in [
        (axes[0], empties,  "Empty Space (%)",   "#3498db"),
        (axes[1], darks,    "Dark Tone Ratio (%)", "#2c3e50"),
        (axes[2], colors_n, "Color Diversity",    "#9b59b6"),
    ]:
        ax.scatter(range(len(vals)), vals, c=color, alpha=0.7, s=60, edgecolors="white")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Drawing index")
        _style_ax(ax)
    fig.suptitle("Visual Feature Overview — All Drawings", fontsize=13, fontweight="bold")
    p = os.path.join(thesis_dir, "feature_overview.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["feature_overview"] = p

    return paths


def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------------
# Dataset-level summary JSON
# ---------------------------------------------------------------------------

def save_summary_report(session_dir: str, all_results: list[dict]) -> str:
    total = len(all_results)
    judge_counts = {}
    for r in all_results:
        s = r.get("judge_status", "PASS")
        judge_counts[s] = judge_counts.get(s, 0) + 1

    emotion_counts: dict[str, int] = {}
    for r in all_results:
        e = r.get("emotion", "neutral_or_unclear")
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    report = {
        "total_images_analysed": total,
        "judge_verdicts":        judge_counts,
        "emotion_distribution":  emotion_counts,
        "avg_verified_claims":   sum(r.get("verified_claims", 0) for r in all_results) / max(total, 1),
        "avg_rules_activated":   sum(r.get("rules_count", 0) for r in all_results) / max(total, 1),
        "images": [
            {
                "image":        r.get("image", ""),
                "label":        r.get("label", ""),
                "emotion":      r.get("emotion", ""),
                "judge_status": r.get("judge_status", ""),
                "rules_count":  r.get("rules_count", 0),
            }
            for r in all_results
        ],
    }

    path = os.path.join(session_dir, "summary_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
