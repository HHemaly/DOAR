"""A2 / X1 — re-score saved validation predictions by source exposure (CPU).

  python experiments/journal/scripts/a2_rescore_saved_predictions.py \
      --manifest <T0 partition_manifest.csv or e1a_train_valid_manifest.csv> \
      --runs-root /content/drive/MyDrive/Masters/DOAR_GPU_RESULTS/E1 \
      --out /content/drive/MyDrive/Masters/DOAR_JOURNAL_2026/A2_rescoring

Writes per-run results (rescoring_by_run.csv), a JSON with full intervals, and
a check that each run's recomputed overall macro-F1 equals the value recorded
historically (sanity check that the right files and ids were joined).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import load_manifest, path_of, split_of  # noqa: E402
from doar_journal.rescoring import (find_runs, macro_f1_from_rows, read_predictions,  # noqa: E402
                                    rescore_run, shared_source_valid_ids, source_family)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--evidence", choices=("rf_stem", "any"), default="rf_stem")
    args = ap.parse_args(argv)

    rows = load_manifest(args.manifest)
    shared, indep = shared_source_valid_ids(rows, evidence=args.evidence)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fam = {r["image_id"]: source_family(path_of(r)) for r in rows if split_of(r) == "valid"}
    with open(out / "valid_exposure.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "exposure", "source_family"])
        for i in sorted(shared | indep):
            w.writerow([i, "shared" if i in shared else "independent", fam.get(i, "")])

    if provenance.is_done(out):
        print(f"A2 already complete in {out} (delete DONE.json to recompute)")
        return 0
    print(f"searching {args.runs_root} for saved validation predictions ...", flush=True)
    paths = find_runs(args.runs_root)
    print(f"found {len(paths)} prediction files", flush=True)
    cache_dir = out / "per_run"
    cache_dir.mkdir(exist_ok=True)
    results, table = {}, []
    for k, p in enumerate(paths, start=1):
        run = str(p.parent.relative_to(args.runs_root))
        cache = cache_dir / (run.replace("/", "__") + f".boot{args.n_boot}.json")
        if cache.exists():  # finished before a disconnect
            r = json.loads(cache.read_text(encoding="utf-8"))
        else:
            preds = read_predictions(p)
            r = rescore_run(preds, shared, indep, n_boot=args.n_boot)
            r["recomputed_macro_f1_all"] = macro_f1_from_rows(preds)
            provenance.write_json(cache, r)
        print(f"[{k}/{len(paths)}] ", end="")
        results[run] = r
        d = r["diff_shared_minus_independent"]
        table.append({"run": run, "n_shared": r["n_shared"], "n_independent": r["n_independent"],
                      "f1_all": round(r["macro_f1_all"]["estimate"], 4),
                      "f1_shared": round(r["macro_f1_shared"]["estimate"], 4),
                      "f1_independent": round(r["macro_f1_independent"]["estimate"], 4),
                      "diff": round(d["estimate"], 4), "diff_ci_low": round(d["ci_low"], 4),
                      "diff_ci_high": round(d["ci_high"], 4), "p_bootstrap": round(d["p_bootstrap"], 4)})
        print(f"{run:60s} all={r['macro_f1_all']['estimate']:.4f} shared={r['macro_f1_shared']['estimate']:.4f} "
              f"indep={r['macro_f1_independent']['estimate']:.4f} diff={d['estimate']:+.4f} "
              f"[{d['ci_low']:+.4f},{d['ci_high']:+.4f}]", flush=True)
    if not table:
        raise SystemExit(f"no best_valid_predictions.csv found under {args.runs_root}")
    with open(out / "rescoring_by_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    meta = provenance.run_metadata("A2_rescore", vars(args), inputs={"manifest": args.manifest})
    provenance.write_json(out / "rescoring_full.json", {"results": results, "metadata": meta,
                                                         "n_shared": len(shared), "n_independent": len(indep)})
    provenance.mark_done(out, meta, [out / "rescoring_by_run.csv", out / "rescoring_full.json",
                                     out / "valid_exposure.csv"])
    print(json.dumps({"runs": len(table), "n_shared": len(shared), "n_independent": len(indep)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
