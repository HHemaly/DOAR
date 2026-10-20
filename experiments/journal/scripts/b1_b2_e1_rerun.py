"""B1-B2 — rerun E1 on the FROZEN T1 split (Colab GPU; resumable).

  python experiments/journal/scripts/b1_b2_e1_rerun.py \
      --t1-dir <Drive>/DOAR_JOURNAL_2026/T1 \
      --dataset-root /content/drive/MyDrive/Masters/Datasets/Combined_Drawing \
      --out <Drive>/DOAR_JOURNAL_2026/E1_T1 --stage all

Stages (each skips work whose outputs already exist):
  probe      9 encoders x 10 seeds frozen linear probe (E1-A2 + E1-A2R protocol)
  adapt      shortlist (top-3 probe means) x frozen/partial/full x seeds 19,42,73 (E1-B)
  confirm    the 2 finalists (ranked on seeds 19,42,73 only) x seeds 11,37 (E1-C1)
  all        the three in order
The locked T1 test split is never loaded here (rows are counted only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import e1_analysis as ea  # noqa: E402
from doar_journal import e1_runner as er  # noqa: E402
from doar_journal import provenance  # noqa: E402
from doar_journal.t1_partition import require_frozen  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-dir", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("probe", "adapt", "confirm", "all"), default="all")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    manifest = require_frozen(a.t1_dir)
    rows = er.load_rows(manifest, a.dataset_root)
    missing = [r["path"] for r in rows["train"] + rows["valid"] if not Path(r["path"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} image paths do not resolve, e.g. {missing[:2]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = provenance.git_commit(Path(__file__).resolve().parents[3])
    provenance.write_json(out / f"run_metadata_{a.stage}.json",
                          provenance.run_metadata(f"E1_T1_{a.stage}", vars(a), inputs={"t1_manifest": manifest}))
    print(f"T1 manifest {rows['manifest_sha256'][:12]}  train={len(rows['train'])} valid={len(rows['valid'])} "
          f"test={rows['n_test']} (locked, not loaded)")

    if a.stage in ("probe", "all"):
        er.run_probe(rows, out, a.device)
    if a.stage in ("adapt", "all"):
        probe_rows = ea.config_table(ea.collect_runs(out / "probe" / "runs"), n_boot=1000)
        sl = ea.shortlist(probe_rows, 3)
        provenance.write_json(out / "SHORTLIST.json", {"shortlist": sl, "rule": "top-3 10-seed mean probe macro-F1"})
        print("shortlist:", sl)
        er.run_adaptation(rows, out, a.dataset_root, commit or "unknown", a.device, models=sl)
    if a.stage in ("confirm", "all"):
        fin = ea.finalists(ea.collect_runs(out / "adapt" / "runs"), k=2)
        provenance.write_json(out / "FINALISTS.json", {"finalists": ["/".join(f) for f in fin],
                                                         "rule": "top-2 mean macro-F1 on seeds 19,42,73"})
        print("finalists:", fin)
        for model, mode in fin:
            er.run_adaptation(rows, out, a.dataset_root, commit or "unknown", a.device,
                              models=[model], modes=[mode], seeds=er.CONFIRM_SEEDS)
    print(json.dumps({"done": a.stage}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
