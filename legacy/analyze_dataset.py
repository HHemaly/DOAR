"""
analyze_dataset.py — Batch-process the Combined_Drawing dataset locally.

Usage (from the DOAR-main folder in VS Code terminal):
    python analyze_dataset.py
    python analyze_dataset.py --root "C:\\path\\to\\Combined_Drawing" --max 10
    python analyze_dataset.py --max 0          # process ALL images
    python analyze_dataset.py --no-arabic      # skip Arabic translation
    python analyze_dataset.py --no-ocr         # skip OCR (faster)

All outputs are saved in outputs/<timestamp>/
"""

import argparse
import sys
import os

# Make sure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pipeline import run_dataset, DATASET_ROOT, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="DOAR — Batch dataset analysis"
    )
    parser.add_argument(
        "--root", default=None,
        help=f"Path to Combined_Drawing folder (default: {DATASET_ROOT})"
    )
    parser.add_argument(
        "--output", default=OUTPUT_DIR,
        help="Output folder (default: ./outputs)"
    )
    parser.add_argument(
        "--max", type=int, default=5,
        help="Max images per class; 0 = all (default: 5)"
    )
    parser.add_argument("--no-arabic", action="store_true", help="Skip Arabic translation")
    parser.add_argument("--no-ocr",    action="store_true", help="Skip OCR")
    args = parser.parse_args()

    max_per = args.max if args.max > 0 else None

    print("=" * 65)
    print("DOAR — Batch Dataset Analysis")
    print(f"  Dataset : {args.root or DATASET_ROOT}")
    print(f"  Output  : {args.output}")
    print(f"  Max/cls : {max_per or 'ALL'}")
    print(f"  Arabic  : {'off' if args.no_arabic else 'on'}")
    print(f"  OCR     : {'off' if args.no_ocr else 'on'}")
    print("=" * 65)

    run_dataset(
        dataset_root=args.root,
        output_dir=args.output,
        max_per_class=max_per,
        run_arabic=not args.no_arabic,
        run_ocr=not args.no_ocr,
    )


if __name__ == "__main__":
    main()
