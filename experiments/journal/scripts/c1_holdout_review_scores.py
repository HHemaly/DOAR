"""C1 — score the completed visual review of the frozen-V2 80-image holdout (CPU).

  python experiments/journal/scripts/c1_holdout_review_scores.py \
      --review <Drive>/.../E2A0_R1_V2_UNSEEN_HOLDOUT/tables/E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv \
      --results <Drive>/.../E2A0_R1_V2_UNSEEN_HOLDOUT/tables/E2A0_R1_V2_UNSEEN_HOLDOUT_RESULTS.csv \
      [--exclude-overlap <A6>/holdout_source_overlap.csv] --out <Drive>/DOAR_JOURNAL_2026/C1_holdout_review

Uses the frozen rubric (GOOD, PARTIAL, FAILED, OVERSEGMENTED, UNDERSEGMENTED,
APPROPRIATELY_AMBIGUOUS, UNCERTAIN). Reports, with Wilson 95% CIs:
* the share of masks judged GOOD among those V2 labelled MASK_USABLE
  (the claim the gate makes), by holdout group
* the share of A0-"OK" control masks that are GOOD on the A0 side is NOT
  covered by this rubric (the review judges V2 masks); see C3 for A0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from doar_journal import provenance  # noqa: E402
from doar_journal.stats import wilson  # noqa: E402

RUBRIC = ("GOOD", "PARTIAL", "FAILED", "OVERSEGMENTED", "UNDERSEGMENTED", "APPROPRIATELY_AMBIGUOUS", "UNCERTAIN")


def read(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--exclude-overlap", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    review = {r["image_id"]: r for r in read(a.review)}
    results = {r["image_id"]: r for r in read(a.results)}
    bad = [i for i, r in review.items() if str(r.get("review_complete", "")).strip().lower() != "true"
           or r.get("visual_mask_quality") not in RUBRIC]
    if bad:
        raise SystemExit(f"review incomplete for {len(bad)} rows, e.g. {bad[:3]}")
    excluded = set()
    if a.exclude_overlap:
        excluded = {r["image_id"] for r in read(a.exclude_overlap) if r["overlap"] in ("rf_stem", "name_key")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for subset_name, keep in (("all", lambda i: True), ("excluding_dev_source_overlap", lambda i: i not in excluded)):
        ids = [i for i in review if keep(i)]
        block = {"n": len(ids), "verdicts": dict(Counter(review[i]["visual_mask_quality"] for i in ids))}
        for group in sorted({results[i]["holdout_group"] for i in ids}):
            g = [i for i in ids if results[i]["holdout_group"] == group]
            usable = [i for i in g if results[i]["v2_measurement_policy"] == "MASK_USABLE"]
            good = [i for i in usable if review[i]["visual_mask_quality"] == "GOOD"]
            withheld = [i for i in g if results[i]["v2_measurement_policy"] != "MASK_USABLE"]
            withheld_bad = [i for i in withheld if review[i]["visual_mask_quality"]
                            in ("FAILED", "OVERSEGMENTED", "UNDERSEGMENTED", "PARTIAL")]
            block[group] = {
                "n": len(g), "usable": len(usable),
                "good_given_usable": wilson(len(good), len(usable)).as_dict(),
                "withheld": len(withheld),
                "defective_given_withheld": wilson(len(withheld_bad), len(withheld)).as_dict(),
                "verdicts": dict(Counter(review[i]["visual_mask_quality"] for i in g)),
            }
        summary[subset_name] = block
    meta = provenance.run_metadata("C1_holdout_review", vars(a), inputs={"review": a.review, "results": a.results})
    provenance.write_json(out / "holdout_review_scores.json", {"summary": summary, "metadata": meta})
    provenance.mark_done(out, meta, [out / "holdout_review_scores.json"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
