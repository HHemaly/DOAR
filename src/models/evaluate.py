"""
evaluate.py — evaluate a trained checkpoint on the untouched TEST split.

Produces every metric/artefact the thesis requires. All numbers are computed
from real predictions on real files — nothing is fabricated. If sklearn is not
installed, core metrics (accuracy, per-class, confusion matrix) are still
computed manually; ROC/PR-AUC are skipped with a logged note.

Saves under <out>/evaluation/:
    metrics.json, classification_report.csv, per_class_metrics.csv,
    predictions_test.csv
    figures/confusion_matrix.(png|svg), normalized_confusion_matrix.png,
            per_class_f1.png, confidence_distribution.png,
            correct_vs_incorrect_confidence.png
"""

from __future__ import annotations
import os
import csv
import json
import time


def evaluate_model(split_csv: str, checkpoint: str, out_dir: str,
                   timestamp: str, batch_size: int = 32,
                   img_size: int = 224, num_workers: int = 0) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from src.models.dataset import load_split, make_torch_dataset
    from src.models.classifier import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eval_dir = os.path.join(out_dir, "evaluation")
    fig_dir = os.path.join(eval_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    ckpt = torch.load(checkpoint, map_location=device)
    classes = ckpt["classes"]
    model_name = ckpt.get("model_name", "transfer")
    img_size = ckpt.get("img_size", img_size)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    rows, _, _ = load_split(split_csv)
    test_ds = make_torch_dataset(rows, "test", class_to_idx, img_size, train=False)
    if len(test_ds) == 0:
        raise ValueError("Empty test split — check split.csv")
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers)

    model = build_model(model_name, len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred, y_conf, all_probs, paths = [], [], [], [], []
    total_time, n_imgs = 0.0, 0

    import torch as _t
    with _t.no_grad():
        for imgs, labels, batch_paths in test_dl:
            imgs = imgs.to(device)
            t0 = time.perf_counter()
            logits = model(imgs)
            probs = _t.softmax(logits, dim=1)
            total_time += time.perf_counter() - t0
            n_imgs += imgs.size(0)
            conf, pred = probs.max(1)
            y_true.extend(labels.tolist())
            y_pred.extend(pred.cpu().tolist())
            y_conf.extend(conf.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())
            paths.extend(batch_paths)

    n = len(y_true)
    n_cls = len(classes)
    correct = sum(int(t == p) for t, p in zip(y_true, y_pred))
    accuracy = correct / n

    # Confusion matrix (manual, no sklearn needed)
    cm = [[0] * n_cls for _ in range(n_cls)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    # Per-class precision/recall/F1
    per_class = []
    for i, c in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n_cls)) - tp
        fn = sum(cm[i][r] for r in range(n_cls)) - tp
        support = sum(cm[i])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append({"class": c, "precision": round(prec, 4),
                          "recall": round(rec, 4), "f1": round(f1, 4),
                          "support": support})

    macro_f1 = sum(pc["f1"] for pc in per_class) / n_cls
    weighted_f1 = sum(pc["f1"] * pc["support"] for pc in per_class) / max(n, 1)
    balanced_acc = sum(pc["recall"] for pc in per_class) / n_cls

    metrics = {
        "n_test": n,
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "inference_time_ms_per_image": round(total_time / max(n_imgs, 1) * 1000, 3),
        "classes": classes,
        "confusion_matrix": cm,
        "checkpoint": checkpoint,
        "device": device,
    }

    # Optional sklearn extras (ROC-AUC, PR-AUC)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        import numpy as np
        y_true_oh = np.eye(n_cls)[y_true]
        probs_arr = np.array(all_probs)
        if n_cls == 2:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, probs_arr[:, 1])), 4)
            metrics["pr_auc"] = round(float(average_precision_score(y_true, probs_arr[:, 1])), 4)
        else:
            metrics["roc_auc_macro_ovr"] = round(float(
                roc_auc_score(y_true_oh, probs_arr, average="macro", multi_class="ovr")), 4)
            metrics["pr_auc_macro"] = round(float(
                average_precision_score(y_true_oh, probs_arr, average="macro")), 4)
    except Exception as e:
        metrics["auc_note"] = f"ROC/PR-AUC skipped: {e}"

    # ── Write artefacts ──────────────────────────────────────────
    with open(os.path.join(eval_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    _write_csv(os.path.join(eval_dir, "per_class_metrics.csv"), per_class,
               ["class", "precision", "recall", "f1", "support"])

    # classification_report.csv (per-class + macro/weighted rows)
    report_rows = list(per_class) + [
        {"class": "macro avg", "precision": "", "recall": "",
         "f1": round(macro_f1, 4), "support": n},
        {"class": "weighted avg", "precision": "", "recall": "",
         "f1": round(weighted_f1, 4), "support": n},
        {"class": "accuracy", "precision": "", "recall": "",
         "f1": round(accuracy, 4), "support": n},
    ]
    _write_csv(os.path.join(eval_dir, "classification_report.csv"), report_rows,
               ["class", "precision", "recall", "f1", "support"])

    # predictions_test.csv with top-k
    pred_rows = []
    for i in range(n):
        probs = all_probs[i]
        topk = sorted(range(n_cls), key=lambda j: -probs[j])[:min(3, n_cls)]
        pred_rows.append({
            "path": paths[i],
            "true_class": classes[y_true[i]],
            "pred_class": classes[y_pred[i]],
            "confidence": round(y_conf[i], 4),
            "correct": y_true[i] == y_pred[i],
            "top_k": "; ".join(f"{classes[j]}:{probs[j]:.3f}" for j in topk),
        })
    _write_csv(os.path.join(eval_dir, "predictions_test.csv"), pred_rows,
               ["path", "true_class", "pred_class", "confidence", "correct", "top_k"])

    # ── Figures ──────────────────────────────────────────────────
    _plot_confusion(fig_dir, cm, classes, normalize=False)
    _plot_confusion(fig_dir, cm, classes, normalize=True)
    _plot_per_class_f1(fig_dir, per_class)
    _plot_confidence(fig_dir, pred_rows)

    print(f"[eval] accuracy={accuracy:.4f} macro_f1={macro_f1:.4f} "
          f"balanced_acc={balanced_acc:.4f}  (n_test={n})")
    return metrics


# ---------------------------------------------------------------------------
def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _plot_confusion(fig_dir, cm, classes, normalize=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    arr = np.array(cm, dtype=float)
    if normalize:
        row_sums = arr.sum(axis=1, keepdims=True)
        arr = np.divide(arr, row_sums, out=np.zeros_like(arr), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(1.4 * len(classes) + 3, 1.4 * len(classes) + 2))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    title = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
    ax.set_title(title, fontweight="bold")
    thresh = arr.max() / 2 if arr.max() else 0.5
    for i in range(len(classes)):
        for j in range(len(classes)):
            txt = f"{arr[i, j]:.2f}" if normalize else f"{int(arr[i, j])}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if arr[i, j] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    name = "normalized_confusion_matrix" if normalize else "confusion_matrix"
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_per_class_f1(fig_dir, per_class):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    classes = [pc["class"] for pc in per_class]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(classes))
    ax.bar([i - 0.2 for i in x], [pc["precision"] for pc in per_class],
           width=0.2, label="precision", color="#3498db")
    ax.bar([i for i in x], [pc["recall"] for pc in per_class],
           width=0.2, label="recall", color="#2ecc71")
    ax.bar([i + 0.2 for i in x], [pc["f1"] for pc in per_class],
           width=0.2, label="F1", color="#e67e22")
    ax.set_xticks(list(x)); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Per-class Precision / Recall / F1", fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"per_class_f1.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_confidence(fig_dir, pred_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    all_conf = [r["confidence"] for r in pred_rows]
    corr = [r["confidence"] for r in pred_rows if r["correct"]]
    inc  = [r["confidence"] for r in pred_rows if not r["correct"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_conf, bins=20, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_title("Prediction Confidence Distribution", fontweight="bold")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"confidence_distribution.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    if corr:
        ax.hist(corr, bins=20, alpha=0.6, label="correct", color="#2ecc71", edgecolor="white")
    if inc:
        ax.hist(inc, bins=20, alpha=0.6, label="incorrect", color="#e74c3c", edgecolor="white")
    ax.set_title("Confidence: Correct vs Incorrect", fontweight="bold")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count"); ax.legend()
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"correct_vs_incorrect_confidence.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
