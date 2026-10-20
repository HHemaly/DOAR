"""B3 — E1 tables, T0-vs-T1 comparison and the FROZEN model selection (CPU).

  python experiments/journal/scripts/b3_e1_analysis.py \
      --t1-out <Drive>/DOAR_JOURNAL_2026/E1_T1 \
      --t0-root /content/drive/MyDrive/Masters/DOAR_GPU_RESULTS/E1 \
      --out <Drive>/DOAR_JOURNAL_2026/B3_E1_analysis [--freeze-selection]

Writes T_probe_{T0,T1}.csv, T_adapt_{T0,T1}.csv, ranking agreement, the
finalist comparison, and -- only with --freeze-selection -- SELECTION_FROZEN.json,
after which B4 may run the single test evaluation. Freezing is refused if any
finalist lacks one of its 5 seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import e1_analysis as ea  # noqa: E402
from doar_journal import provenance  # noqa: E402


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-out", required=True)
    ap.add_argument("--t0-root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze-selection", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t1 = Path(a.t1_out)
    report = {}

    probe1 = ea.config_table(ea.collect_runs(t1 / "probe" / "runs"), a.n_boot)
    runs_adapt1 = ea.collect_runs(t1 / "adapt" / "runs")
    adapt1 = ea.config_table(runs_adapt1, a.n_boot)
    write(out / "T_probe_T1.csv", probe1)
    write(out / "T_adapt_T1.csv", adapt1)
    report["shortlist_T1"] = ea.shortlist(probe1)
    fin = ea.finalists(runs_adapt1, k=2)
    report["finalists_T1"] = ["/".join(f) for f in fin]
    final_rows = [r for r in adapt1 if (r["model"], r["mode"]) in fin]
    report["finalist_5seed"] = final_rows
    if len(fin) == 2 and all(len(runs_adapt1[f]) == 5 for f in fin):
        report["finalist_comparison"] = ea.compare_finalists(runs_adapt1, fin[0], fin[1], a.n_boot)

    if a.t0_root:
        t0 = ea.collect_runs(a.t0_root)
        probe0 = ea.config_table({k: v for k, v in t0.items() if k[1] == "probe"}, a.n_boot)
        adapt0 = ea.config_table({k: v for k, v in t0.items() if k[1] != "probe"}, a.n_boot)
        write(out / "T_probe_T0.csv", probe0)
        write(out / "T_adapt_T0.csv", adapt0)
        report["probe_ranking_T0_vs_T1"] = ea.ranking_agreement(probe0, probe1, "probe")

    if a.freeze_selection:
        if len(fin) != 2 or not all(len(runs_adapt1[f]) == 5 for f in fin):
            raise SystemExit("cannot freeze: both finalists need 5 seeds (run b1_b2 --stage confirm)")
        sel = ea.select(final_rows)
        runner_up = [f for f in fin if f != sel][0]
        record = {"selected": {"model": sel[0], "mode": sel[1], "seeds": sorted(runs_adapt1[sel])},
                  "comparator": {"model": runner_up[0], "mode": runner_up[1], "seeds": sorted(runs_adapt1[runner_up])},
                  "probe_baseline": {"model": sel[0], "mode": "probe", "seeds": [42]},
                  "rule": "highest 5-seed mean validation macro-F1; tie -> lower SD",
                  "checkpoints": {f"{m}/{md}/{s}": str(p.parent / "best.pt")
                                  for (m, md) in (sel, runner_up) for s, p in runs_adapt1[(m, md)].items()},
                  "frozen_utc": provenance.run_metadata("freeze", {})["utc"]}
        provenance.write_json(out / "SELECTION_FROZEN.json", record)
        report["selection"] = record
    provenance.write_json(out / "e1_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "finalist_5seed"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
