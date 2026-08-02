"""
train.py — supervised training with early stopping, checkpointing, LR schedule.

Runs on CPU, NVIDIA GPU, or Colab GPU (device auto-detected). Honest: if the
model does not learn, the saved history/metrics reflect that. Nothing is faked.

Saves under <out>/training/:
    best_model.pt, last_model.pt
    training_history.csv, training_config.json, training_log.txt, classes.json
    figures/training_curves.png (+ .svg)
"""

from __future__ import annotations
import os
import csv
import json


def train_model(split_csv: str, out_dir: str, timestamp: str,
                model_name: str = "transfer",
                epochs: int = 25, batch_size: int = 32,
                lr: float = 1e-3, img_size: int = 224,
                seed: int = 42, patience: int = 6,
                use_class_weights: bool = True,
                num_workers: int = 0) -> dict:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from src.utils.reproducibility import set_global_seed, build_manifest, save_manifest
    from src.models.dataset import load_split, class_weights, make_torch_dataset
    from src.models.classifier import build_model

    set_global_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = os.path.join(out_dir, "training")
    fig_dir = os.path.join(train_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    rows, classes, class_to_idx = load_split(split_csv)
    n_classes = len(classes)
    print(f"[train] classes={classes} device={device} model={model_name}")

    train_ds = make_torch_dataset(rows, "train", class_to_idx, img_size, train=True)
    val_ds   = make_torch_dataset(rows, "val",   class_to_idx, img_size, train=False)
    if len(train_ds) == 0:
        raise ValueError("Empty training split — check split.csv")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers) if len(val_ds) else None

    model = build_model(model_name, n_classes, pretrained=True).to(device)

    weights = class_weights(rows, class_to_idx, "train").to(device) if use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    config = {
        "model_name": model_name, "epochs": epochs, "batch_size": batch_size,
        "lr": lr, "img_size": img_size, "seed": seed, "patience": patience,
        "use_class_weights": use_class_weights, "classes": classes,
        "n_train": len(train_ds), "n_val": len(val_ds), "device": device,
    }
    with open(os.path.join(train_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(train_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "class_to_idx": class_to_idx}, f, indent=2)
    save_manifest(build_manifest(seed, timestamp, {"stage": "training"}), train_dir)

    history = []
    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    log_lines = []

    def _run_epoch(dl, training):
        model.train() if training else model.eval()
        total_loss, correct, n = 0.0, 0, 0
        import torch as _t
        ctx = _t.enable_grad() if training else _t.no_grad()
        with ctx:
            for imgs, labels, _ in dl:
                imgs, labels = imgs.to(device), labels.to(device)
                if training:
                    optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                if training:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                correct += (out.argmax(1) == labels).sum().item()
                n += imgs.size(0)
        return total_loss / max(n, 1), correct / max(n, 1)

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(train_dl, True)
        if val_dl:
            va_loss, va_acc = _run_epoch(val_dl, False)
        else:
            va_loss, va_acc = tr_loss, tr_acc
        scheduler.step(va_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc, "lr": cur_lr})
        line = (f"epoch {epoch:03d}  train_loss={tr_loss:.4f} acc={tr_acc:.4f}  "
                f"val_loss={va_loss:.4f} acc={va_acc:.4f}  lr={cur_lr:.2e}")
        print(f"[train] {line}")
        log_lines.append(line)

        # Checkpoint best
        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "model_name": model_name, "img_size": img_size,
                        "epoch": epoch, "val_loss": va_loss},
                       os.path.join(train_dir, "best_model.pt"))
        else:
            epochs_no_improve += 1

        torch.save({"model_state": model.state_dict(), "classes": classes,
                    "model_name": model_name, "img_size": img_size,
                    "epoch": epoch},
                   os.path.join(train_dir, "last_model.pt"))

        if epochs_no_improve >= patience:
            log_lines.append(f"Early stopping at epoch {epoch} "
                             f"(best epoch {best_epoch}, val_loss {best_val:.4f})")
            print(f"[train] {log_lines[-1]}")
            break

    # Save history + log
    with open(os.path.join(train_dir, "training_history.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc",
                                          "val_loss", "val_acc", "lr"])
        w.writeheader(); w.writerows(history)
    with open(os.path.join(train_dir, "training_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    _plot_curves(fig_dir, history)

    print(f"[train] Done. best epoch={best_epoch} val_loss={best_val:.4f}")
    return {"best_epoch": best_epoch, "best_val_loss": best_val,
            "history": history, "classes": classes,
            "best_model": os.path.join(train_dir, "best_model.pt")}


def _plot_curves(fig_dir, history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ep, [h["train_loss"] for h in history], label="train", marker="o")
    axes[0].plot(ep, [h["val_loss"] for h in history], label="val", marker="s")
    axes[0].set_title("Loss curves", fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-entropy loss"); axes[0].legend()
    axes[1].plot(ep, [h["train_acc"] for h in history], label="train", marker="o")
    axes[1].plot(ep, [h["val_acc"] for h in history], label="val", marker="s")
    axes[1].set_title("Accuracy curves", fontweight="bold")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].legend()
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(linestyle="--", alpha=0.4)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(fig_dir, f"training_curves.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
