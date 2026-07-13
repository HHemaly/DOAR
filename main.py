"""
main.py — unified DOAR command-line interface.

Subcommands:
    inspect          inspect the dataset -> CSVs, JSON, figures
    split            build a leak-safe train/val/test split from the inspection
    train            train a classifier (baseline | transfer | mobilenet | efficientnet)
    evaluate         evaluate a checkpoint on the untouched test split
    analyze-image    run the interpretation pipeline on one drawing
    analyze-dataset  run the interpretation pipeline on a folder of drawings
    thesis           collate thesis figures/tables from existing outputs

Examples (Windows / VS Code, from the DOAR folder):
    python main.py inspect  --data "C:\\Users\\Ahmed\\Downloads\\Combined_Drawing\\Combined_Drawing"
    python main.py split    --out outputs
    python main.py train    --out outputs --model transfer --epochs 25
    python main.py evaluate --out outputs --checkpoint outputs\\training\\best_model.pt
    python main.py analyze-image   --input "path\\to\\drawing.jpg"
    python main.py analyze-dataset --data "C:\\...\\Combined_Drawing" --max 5
    python main.py thesis   --out outputs

All paths use pathlib; nothing is hardcoded to Colab.
"""

from __future__ import annotations
import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Make 'src' importable as a package (src.models.train etc.)
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

# Default dataset path — edit here or pass --data on the CLI.
DEFAULT_DATASET = os.environ.get(
    "DOAR_DATASET",
    r"C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing",
)
DEFAULT_OUT = os.environ.get("DOAR_OUTPUT", str(ROOT / "outputs"))


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_inspect(args):
    from src.data.inspect_dataset import inspect_dataset
    inspect_dataset(args.data, args.out, near_dup_threshold=args.near_dup)


def cmd_split(args):
    from src.data.split import make_split
    summary = Path(args.out) / "dataset_analysis" / "dataset_summary.csv"
    if not summary.exists():
        print(f"ERROR: {summary} not found. Run 'inspect' first.")
        sys.exit(1)
    make_split(str(summary), args.out, seed=args.seed, near_dup_threshold=args.near_dup)


def cmd_train(args):
    from src.models.train import train_model
    split_csv = Path(args.out) / "splits" / "split.csv"
    if not split_csv.exists():
        print(f"ERROR: {split_csv} not found. Run 'split' first.")
        sys.exit(1)
    train_model(str(split_csv), args.out, _ts(), model_name=args.model,
                epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                seed=args.seed)


def cmd_evaluate(args):
    from src.models.evaluate import evaluate_model
    split_csv = Path(args.out) / "splits" / "split.csv"
    ckpt = args.checkpoint or str(Path(args.out) / "training" / "best_model.pt")
    if not Path(ckpt).exists():
        print(f"ERROR: checkpoint {ckpt} not found. Run 'train' first.")
        sys.exit(1)
    evaluate_model(str(split_csv), ckpt, args.out, _ts(), batch_size=args.batch_size)


def cmd_analyze_image(args):
    from pipeline import run_full_pipeline_v2
    from src.data.inspect_dataset import SUPPORTED_EXT  # noqa
    out = Path(args.out) / _ts()
    out.mkdir(parents=True, exist_ok=True)
    result = run_full_pipeline_v2(args.input, parent_question=args.question,
                                  run_ocr=not args.no_ocr,
                                  run_arabic=not args.no_arabic,
                                  session_dir=str(out))
    print("\n── Parent answer ──")
    print(result["parent_facing_output"].get("parent_answer", ""))
    print(f"\n── Judge: {result['final_judgment'].get('final_answer_status')} ──")
    if result.get("saved_paths"):
        print(f"Saved: {list(result['saved_paths'].values())}")


def cmd_analyze_dataset(args):
    from pipeline import run_dataset
    run_dataset(dataset_root=args.data, output_dir=args.out,
                max_per_class=(args.max if args.max > 0 else None),
                run_arabic=not args.no_arabic, run_ocr=not args.no_ocr)


def cmd_thesis(args):
    """Collate whatever thesis-ready artefacts already exist into thesis/."""
    from src.reports.thesis_collate import collate_thesis
    collate_thesis(args.out)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(prog="doar", description="DOAR unified CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--out", default=DEFAULT_OUT, help="Output root folder")

    sp = sub.add_parser("inspect", help="Inspect the dataset")
    add_common(sp)
    sp.add_argument("--data", default=DEFAULT_DATASET)
    sp.add_argument("--near-dup", type=int, default=5, dest="near_dup")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("split", help="Build leak-safe split")
    add_common(sp)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--near-dup", type=int, default=5, dest="near_dup")
    sp.set_defaults(func=cmd_split)

    sp = sub.add_parser("train", help="Train classifier")
    add_common(sp)
    sp.add_argument("--model", default="transfer",
                    choices=["baseline", "transfer", "resnet18", "mobilenet", "efficientnet"])
    sp.add_argument("--epochs", type=int, default=25)
    sp.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    sp.add_argument("--lr", type=float, default=1e-3)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("evaluate", help="Evaluate checkpoint on test split")
    add_common(sp)
    sp.add_argument("--checkpoint", default=None)
    sp.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("analyze-image", help="Interpretation pipeline on one image")
    add_common(sp)
    sp.add_argument("--input", required=True)
    sp.add_argument("--question", default="")
    sp.add_argument("--no-ocr", action="store_true")
    sp.add_argument("--no-arabic", action="store_true")
    sp.set_defaults(func=cmd_analyze_image)

    sp = sub.add_parser("analyze-dataset", help="Interpretation pipeline on a folder")
    add_common(sp)
    sp.add_argument("--data", default=DEFAULT_DATASET)
    sp.add_argument("--max", type=int, default=5, help="Max per class; 0 = all")
    sp.add_argument("--no-ocr", action="store_true")
    sp.add_argument("--no-arabic", action="store_true")
    sp.set_defaults(func=cmd_analyze_dataset)

    sp = sub.add_parser("thesis", help="Collate thesis figures/tables")
    add_common(sp)
    sp.set_defaults(func=cmd_thesis)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
