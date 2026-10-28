"""A5b — turn the blinded pair review into the edge files T1 may use (CPU).

  python experiments/journal/scripts/a5b_accepted_edges.py \
      --manifest <T0 partition_manifest.csv> --root <Drive>/DOAR_JOURNAL_2026

Reads A5_pair_review/review_scores.json and applies the PRE-REGISTERED rules
(PREREGISTRATION.md C1/C1b) to each evidence type. Nothing is decided by eye.

  rf_stem    always used by T1 (filename evidence); the review only reports how
             reliable it is. Stems are "confirmed" if the lower Wilson bound of
             judged-same >= 0.80 and controls judged same <= 10%.
  name_key   identical plain file names across splits. Accepted if >= 90% of
             reviewed pairs were judged the same drawing -> name_key_edges.csv.
  dihedral   rotation/flip dHash <= 6. Accepted if >= 90% of the reviewed pairs
             that the image evidence ADDS (pairs not already linked by filename
             or the T0 group) were judged the same drawing. Otherwise the
             pre-registered tighter threshold (<= 4) is written as
             A5_pair_review/dihedral_tight_edges_candidates.csv, which needs its
             own blinded review (stratum dihedral_tight) before it may be used.
  embedding  CLIP cosine >= 0.95; same logic, tighter threshold 0.97.
UNSURE counts as NOT the same drawing (conservative: no merge without evidence).

Output: A5_pair_review/ACCEPTED_EDGES.json listing the files to give A3, the
decision for every evidence type and the reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.leakage import load_manifest, path_of, source_key, split_of  # noqa: E402

ACCEPT_RATE = 0.90
STEM_LOWER_BOUND, CONTROL_MAX = 0.80, 0.10
TIGHTER = {"dihedral": ("hamming", 4), "embedding": ("cos", 0.97)}


def rate(block: dict) -> tuple[float, float, int]:
    r = block["rate_same"]
    return r["estimate"], r["ci_low"], block["n"]


def name_key_edges(rows: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        k = source_key(path_of(r))
        if not k.is_roboflow:
            by[k.name_key].append(r)
    edges = []
    for key, members in sorted(by.items()):
        if len({split_of(m) for m in members}) < 2:
            continue  # only names that recur ACROSS splits are leakage candidates
        ids = sorted(m["image_id"] for m in members)
        edges += [{"a": ids[0], "b": o, "evidence": "manual", "detail": f"name_key={key}"} for o in ids[1:]]
    return edges


def filter_edges(src: Path, kind: str) -> list[dict]:
    field, thr = TIGHTER[kind]
    keep = []
    with open(src, newline="", encoding="utf-8") as f:
        for e in csv.DictReader(f):
            vals = dict(x.split("=") for x in e["detail"].split(";") if "=" in x)
            v = float(vals[field])
            if (field == "hamming" and v <= thr) or (field == "cos" and v >= thr):
                keep.append({k: e[k] for k in ("a", "b", "evidence", "detail")})
    return keep


def write(path: Path, edges: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["a", "b", "evidence", "detail"])
        w.writeheader()
        w.writerows(edges)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args(argv)
    root = Path(a.root)
    rev = root / "A5_pair_review"
    scores = json.loads((rev / "review_scores.json").read_text(encoding="utf-8"))
    rows = load_manifest(a.manifest)
    decisions, accepted = {}, []

    if "rf_stem" in scores and "same_rf_stem" in scores["rf_stem"]:
        est, lo, n = rate(scores["rf_stem"]["same_rf_stem"])
        ctl = scores["rf_stem"]["control"]["rate_same"]["estimate"]
        ok = lo >= STEM_LOWER_BOUND and ctl <= CONTROL_MAX
        decisions["rf_stem"] = {"used_by_T1": True, "confirmed": ok,
                                "reason": f"judged same {est:.0%} (lower bound {lo:.0%}, n={n}); controls {ctl:.0%}"
                                          + ("" if ok else " -> NOT confirmed: report as a limitation and discuss before approving T1")}
    else:
        decisions["rf_stem"] = {"used_by_T1": True, "confirmed": None, "reason": "review not scored yet"}

    if "name_key" in scores and "same_name_key" not in scores["name_key"]:
        decisions["name_key"] = {"accepted": False, "reason": "no plain-filename pairs across train/valid to review; nothing to add"}
    elif "name_key" in scores:
        est, lo, n = rate(scores["name_key"]["same_name_key"])
        ctl = scores["name_key"].get("control", {}).get("rate_same", {}).get("estimate", 0.0)
        ok = est >= ACCEPT_RATE and ctl <= CONTROL_MAX
        if ok:
            p = rev / "name_key_edges.csv"
            write(p, name_key_edges(rows))
            accepted.append(str(p))
        decisions["name_key"] = {"accepted": ok, "reason": f"judged same {est:.0%} of {n}; controls {ctl:.0%} "
                                                           "(rule: >= 90% and controls <= 10%)"}
    else:
        decisions["name_key"] = {"accepted": False, "reason": "review not scored yet"}

    for kind in ("dihedral", "embedding"):
        src = root / "A4_image_evidence" / f"{kind}_edges_candidates.csv"
        tight_src = rev / f"{kind}_tight_edges_candidates.csv"
        key = f"candidate_{kind}"
        if kind not in scores or key not in scores[kind]:
            decisions[kind] = {"accepted": False, "reason": "review not scored yet (or A4 not finished)"}
            continue
        est, lo, n = rate(scores[kind][key])
        if est >= ACCEPT_RATE:
            accepted.append(str(src))
            decisions[kind] = {"accepted": True, "threshold": "original",
                               "reason": f"judged same {est:.0%} of {n} reviewed NEW pairs (rule >= 90%)"}
            continue
        tight_scores = scores.get(f"{kind}_tight", {}).get(key)
        if tight_scores is not None:
            t_est, _, t_n = rate(tight_scores)
            ok = t_est >= ACCEPT_RATE
            if ok:
                accepted.append(str(tight_src))
            decisions[kind] = {"accepted": ok, "threshold": "tighter",
                               "reason": f"original threshold {est:.0%} of {n} (< 90%); tighter threshold "
                                         f"{t_est:.0%} of {t_n} -> " + ("accepted" if ok else
                                         "also < 90%: this evidence type is NOT used; report it as a limitation")}
            continue
        if not tight_src.exists():
            write(tight_src, filter_edges(src, kind))
        decisions[kind] = {"accepted": False, "threshold": "original",
                           "reason": f"judged same only {est:.0%} of {n} (< 90%). Wrote the pre-registered tighter "
                                     f"threshold to {tight_src.name}. NEXT: run the A5 make cell again, review "
                                     f"STRATUM='{kind}_tight', run A5 score and this cell again."}
    out = {"accepted_edge_files": accepted, "decisions": decisions,
           "metadata": provenance.run_metadata("A5b_accepted_edges", vars(a), inputs={"scores": rev / "review_scores.json"})}
    provenance.write_json(rev / "ACCEPTED_EDGES.json", out)
    print(json.dumps({"accepted_edge_files": accepted, "decisions": decisions}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
