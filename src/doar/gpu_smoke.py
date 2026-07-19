"""
gpu_smoke.py — a self-contained GPU/CPU smoke test (Item 19).

Runs one forward pass, one backward pass, a tiny 1-step training update, and one
inference on a small model, reporting the resolved device and (when CUDA is used)
peak GPU memory. Honest by construction: it reports `cuda_used=False` on CPU, so
the GPU path is only ever claimed verified when CUDA actually ran.

Uses device="auto": CUDA when available, clean CPU fallback otherwise. Never
hardcodes .cuda(). Safe for a 6 GB GPU (tiny batch, mixed precision on CUDA).
"""

from __future__ import annotations
import json
from pathlib import Path

from .dataset import CLASSES


def run_gpu_smoke(output: str | Path | None = None, device: str = "auto",
                  batch_size: int = 4, image_size: int = 64) -> dict:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install PyTorch support with: pip install -e ".[deep]"') from exc
    from .deep.registry import build_model

    selected = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device)
    cuda_used = selected.startswith("cuda")

    if cuda_used:
        torch.cuda.reset_peak_memory_stats()

    model = build_model("small_cnn", len(CLASSES), pretrained=False).to(selected)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=cuda_used)

    images = torch.randn(batch_size, 3, image_size, image_size, device=selected)
    labels = torch.randint(0, len(CLASSES), (batch_size,), device=selected)

    # Forward + backward + one optimizer step (mixed precision on CUDA).
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda" if cuda_used else "cpu", enabled=cuda_used):
        loss = criterion(model(images), labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Inference.
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(images[:1]), dim=1)[0].cpu().tolist()

    peak_mb = None
    if cuda_used:
        peak_mb = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2)

    result = {
        "status": "ok",
        "requested_device": device,
        "resolved_device": selected,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_used": cuda_used,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_used else None,
        "peak_gpu_memory_mb": peak_mb,
        "mixed_precision": cuda_used,
        "batch_size": batch_size,
        "train_step_loss": float(loss.detach().cpu()),
        "inference_probabilities": {c: probs[i] for i, c in enumerate(CLASSES)},
        "note": ("GPU path exercised" if cuda_used
                 else "Ran on CPU; GPU path NOT verified (no CUDA device)."),
    }
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "gpu_smoke.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
