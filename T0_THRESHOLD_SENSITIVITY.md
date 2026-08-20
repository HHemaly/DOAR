# T0 Automated Conservative — Threshold Sensitivity (≤2 vs ≤3)

**Not a model experiment.** Purpose only: demonstrate the selected duplicate
policy (dHash ≤2) was not arbitrarily chosen, using duplicate-quality/
leakage-control evidence alone — never classifier performance or labels.

## Existing precision evidence (already computed, `PHASE7B_DUPLICATE_POLICY.md` §14)

Purpose-built boundary audit, blinded, 6 pairs sampled per exact dHash distance:

| dHash distance | n | True-duplicate-like | Precision | 95% Wilson CI |
|---|---|---|---|---|
| 2 | 6 | 6 | 100.0% | 61.0%–100.0% |
| 3 | 6 | 6 | 100.0% | 61.0%–100.0% |
| 4 | 6 | 3 | 50.0% | 18.8%–81.2% |
| 5 | 6 | 0 | 0.0% | 0.0%–39.0% |

Distances 2 and 3 are statistically indistinguishable (identical point
estimate, overlapping CIs); precision breaks down sharply at 4.

## This session's own full-dataset structural recomputation

Computed by re-running `main.py build-partition` against
`outputs/t0_automated/source_manifest.csv` (the real, already-hashed
3,688-image manifest reused from the Phase 7B/2C1 reconstruction — no new
image access, no re-hashing) at both thresholds, seed=42, dhash field, same
4 exposed-image exclusions:

| Metric | dHash ≤2 | dHash ≤3 |
|---|---|---|
| Total duplicate groups | 2,367 | 2,344 |
| Multi-member groups | 756 | 760 |
| Largest group | 9 | 9 |
| Exact-match edges | 688 | 688 |
| Near-dup edges | 2,103 | 2,184 |
| Images in multi-member groups | 2,077 | 2,104 |
| Conflict (cross-label) groups | 93 | 95 |
| Excluded (conflict) images | 293 | 301 |
| % of dataset excluded | 7.94% | 8.16% |
| Included (train+valid+test) | 3,395 | 3,387 |
| Train | 2,599 | 2,592 |
| Valid | 284 | 284 |
| Test | 512 | 511 |
| Any duplicate group crosses splits | 0 | 0 |
| Exposed images in valid/test | 0 | 0 |

Both thresholds independently reproduce `PHASE7B_DUPLICATE_POLICY.md` §25's
own previously-published structural comparison exactly (near-dup edges
2,103/2,184, conflict images 293/301) — confirming this session's
recomputation from the reconstructed manifest is consistent with the
original full-dataset scan, not a new or different result.

## Decision

**dHash ≤2 selected as primary.** Rationale: no leakage-control quality
difference exists between ≤2 and ≤3 (identical largest-group size, no
chaining at either, statistically indistinguishable precision) — under a
tie on quality, the strictly more conservative threshold (≤2, admitting a
smaller near-dup radius) is preferred, at a cost of only 8 fewer included
images than ≤3 (3,395 vs 3,387 — a 0.24 percentage-point difference).
Threshold 4 was not considered as a candidate primary (precision already
drops to 50% there per existing evidence) — it appears in the earlier
project documentation only as a contrast point, never a candidate here.

**No label or downstream model-performance signal of any kind was used to
make this selection.** Full methodology and final result:
`T0_AUTOMATED_CONSERVATIVE_PARTITION.md`.
