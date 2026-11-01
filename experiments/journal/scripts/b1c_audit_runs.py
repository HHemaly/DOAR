"""B1c — read-only audit of a Stage B results folder before resuming (CPU, seconds).

  python experiments/journal/scripts/b1c_audit_runs.py --t1-dir <root>/T1_v2 --out <root>/E1_T1_v2

Checks, without loading any model or any test image:
  * the folder is bound to the frozen T1 manifest (BOUND_MANIFEST.json);
  * SHORTLIST.json (if present) equals the pre-registered rule applied to the probe results;
  * every probe run (9 models x 10 seeds): result.json complete, recorded manifest hash = T1,
    best_valid_predictions.csv covers exactly the T1 validation images and no T1 test image;
  * every adaptation run (shortlist x frozen/partial/full x seeds 19,42,73): result.json complete,
    best.pt present, epoch_history.csv hash column = T1, predictions = T1 validation images only;
  * which runs a resumed B1-B2 will SKIP and which it will RUN.
Writes <out>/AUDIT_<timestamp>.json; never changes any run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import e1_analysis as ea  # noqa: E402
from doar_journal import e1_runner as er  # noqa: E402
from doar_journal.provenance import sha256_file, write_json  # noqa: E402
from doar_journal.t1_partition import require_frozen  # noqa: E402


def ids_of(pred_csv: Path) -> list[str]:
    with open(pred_csv, newline="", encoding="utf-8") as f:
        return [r["image_id"] for r in csv.DictReader(f)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    manifest = require_frozen(a.t1_dir)
    sha = sha256_file(manifest)
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    valid = {r["image_id"] for r in rows if r["new_split"] == "valid"}
    test = {r["image_id"] for r in rows if r["new_split"] == "test"}
    out = Path(a.out)
    problems, report = [], {"t1_dir": a.t1_dir, "t1_manifest_sha256": sha}

    bound = out / "BOUND_MANIFEST.json"
    b = json.loads(bound.read_text(encoding="utf-8"))["manifest_sha256"] if bound.exists() else None
    report["bound_manifest_ok"] = b == sha
    if b != sha:
        problems.append(f"results folder bound to {b and b[:12]}, T1 is {sha[:12]}")

    def check_preds(run_dir: Path, label: str):
        p = run_dir / "best_valid_predictions.csv"
        if not p.exists():
            problems.append(f"{label}: best_valid_predictions.csv missing")
            return
        ids = ids_of(p)
        if set(ids) & test:
            problems.append(f"{label}: predictions contain {len(set(ids) & test)} T1 TEST images")
        if set(ids) != valid or len(ids) != len(valid):
            problems.append(f"{label}: predictions cover {len(set(ids))} ids, T1 validation has {len(valid)}")

    probe_done, probe_todo = [], []
    for m in er.PROBE_MODELS:
        for s in er.PROBE_SEEDS:
            d = out / "probe" / "runs" / f"{m}__probe__seed_{s}"
            r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
            if r.get("status") == "complete":
                probe_done.append(d.name)
                if r.get("manifest_sha256") != sha:
                    problems.append(f"{d.name}: recorded manifest {str(r.get('manifest_sha256'))[:12]} != T1")
                check_preds(d, d.name)
            else:
                probe_todo.append(d.name)
    report["probe"] = {"complete": len(probe_done), "expected": len(er.PROBE_MODELS) * len(er.PROBE_SEEDS),
                       "will_run": probe_todo}

    shortlist = None
    if len(probe_done) == len(er.PROBE_MODELS) * len(er.PROBE_SEEDS):
        try:
            shortlist = ea.shortlist(ea.config_table(ea.collect_runs(out / "probe" / "runs"), n_boot=200), 3)
        except ValueError as exc:
            problems.append(f"probe results cannot be combined: {exc}")
        saved = out / "SHORTLIST.json"
        if shortlist and saved.exists():
            s_saved = json.loads(saved.read_text())["shortlist"]
            report["shortlist_saved"] = s_saved
            if list(s_saved) != list(shortlist):
                problems.append(f"SHORTLIST.json {s_saved} != rule applied to probe results {shortlist}")
    report["shortlist_by_rule"] = shortlist

    adapt_done, adapt_todo, incomplete = [], [], []
    for m in (shortlist or []):
        for mode in er.ADAPT_MODES:
            for s in er.ADAPT_SEEDS:
                d = out / "adapt" / "runs" / f"{m}__{mode}__seed_{s}"
                r = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
                if r.get("status") == "complete":
                    adapt_done.append(d.name)
                    if not (d / "best.pt").exists():
                        problems.append(f"{d.name}: best.pt missing")
                    eh = d / "epoch_history.csv"
                    if eh.exists():
                        with open(eh, newline="", encoding="utf-8") as f:
                            hs = {row.get("t0_sha") for row in csv.DictReader(f)}
                        if hs != {sha}:
                            problems.append(f"{d.name}: epoch history manifest {hs} != T1")
                    else:
                        problems.append(f"{d.name}: epoch_history.csv missing")
                    check_preds(d, d.name)
                else:
                    adapt_todo.append(d.name)
                    if d.exists():
                        incomplete.append({"run": d.name, "files": sorted(p.name for p in d.iterdir())})
    report["adapt"] = {"complete": len(adapt_done), "expected": 3 * len(er.ADAPT_MODES) * len(er.ADAPT_SEEDS),
                       "complete_runs": adapt_done, "will_run": adapt_todo,
                       "started_but_incomplete": incomplete}
    confirm = sorted((out / "adapt" / "runs").glob("*__seed_11")) + sorted((out / "adapt" / "runs").glob("*__seed_37"))
    report["confirm"] = {"existing": [p.name for p in confirm],
                         "note": "finalists are chosen only after all 27 adaptation runs exist (2 finalists x seeds 11,37 = 4 runs)"}
    report["problems"] = problems
    report["verdict"] = "SAFE TO RESUME" if not problems else "DO NOT RESUME - see problems"
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(out / f"AUDIT_{stamp}.json", report)
    print(f"T1 {Path(a.t1_dir).name} manifest {sha[:12]}; results folder bound: {report['bound_manifest_ok']}")
    print(f"probe: {len(probe_done)}/{report['probe']['expected']} complete; will run: {probe_todo or 'none'}")
    print(f"shortlist (rule): {shortlist}; saved: {report.get('shortlist_saved')}")
    print(f"adaptation: {len(adapt_done)}/{report['adapt']['expected']} complete")
    print("  will run:", adapt_todo or "none")
    if incomplete:
        print("  started but incomplete (will restart; any last.pt is resumed):", incomplete)
    print("confirmation runs existing:", report["confirm"]["existing"] or "none (4 to run after adaptation)")
    print("problems:", problems or "none")
    print("VERDICT:", report["verdict"])
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
