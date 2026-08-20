# T0 — Automated Conservative Duplicate-Controlled Partition (T0_AUTOMATED_CONSERVATIVE_V1)

**Status: dataset partition only. No model was trained or evaluated.**
This methodology exists as a SEPARATE alternative to the incomplete Phase 7B
human-review workflow (`PHASE7B_DUPLICATE_POLICY.md`, `outputs/phase7b/
human_review/`, 225 blind pairs never completed) — it does not fabricate,
assume, or simulate any human review decision. If the Phase 7B human review
is completed in the future, its own gate (`dataset_gate.check_clean_split_
gate`) remains fully intact and unmodified, and may supersede this one.

---

## 1. Why an automated methodology, not the Phase 7B human-review gate

The Phase 7B interactive review app (`phase7b_review_app.py`) was built and
launched, but 0 of its 225 items had been decided
(`outputs/phase7b/human_review/app_data/decisions.json` does not exist).
Per explicit instruction, this phase does **not** ask for that review to be
completed, and does **not** fabricate any decision to force
`check_clean_split_gate()` to pass. Instead, it builds and verifies a
second, independent, fully-automated methodology from evidence Phase 7B
**already produced and published** — no new human judgment, no new AI
judgment, no model performance of any kind.

## 2. Threshold selection (evidence-only, no labels, no model performance)

`PHASE7B_DUPLICATE_POLICY.md` had already found dHash ≤6 (its prior,
never-approved choice) too permissive (§14: precision falls to 50% by
distance 4, 0% by distance 5) and left the correct threshold an open human
decision (§17/§25). This phase evaluated the two most conservative
candidates already discussed there — dHash ≤2 and ≤3 — using only
already-published precision and structural evidence (§14/§25) plus a fresh
full-dataset structural recomputation at both thresholds this session (no
raw image re-access needed — the reconstructed manifest already carries a
real, previously-computed `dhash` column for all 3,688 images; see §7 for
why this is a legitimate reuse, not a new hashing run).

**Full comparison, existing precision evidence, and the decision rationale:
`T0_THRESHOLD_SENSITIVITY.md`.**

**Selected: dHash ≤2, seed 42.** Structurally indistinguishable from ≤3
(largest duplicate component = 9 for both, no chaining at either) and
statistically indistinguishable in precision (100% at both, per the
existing blinded boundary audit) — under a tie on leakage-control quality,
the strictly more conservative option is preferred, at a cost of only 8
fewer included images than ≤3 (3,395 vs 3,387 of 3,688).

## 3. Build method

`main.py build-partition --manifest outputs/t0_automated/source_manifest.csv
--output outputs/t0_automated/final_partition --seed 42
--near-dup-threshold 2 --hash-field dhash --exposed-image-ids
e892f4f710e144bf,b06dabe271cb4451,5cc50fc211ecae36,07a118a7f414a9d9`
(`src/doar/partition.py::run_partition_design`, unmodified — the exact
same group-disjoint, largest-remainder stratified-partition algorithm
Phase 7A/7B already used, no new methodology invented).

`outputs/t0_automated/source_manifest.csv` is a column-renamed view of the
already-real, already-hashed 3,688-row manifest reconstructed during the
Phase 2C1 annotation-provenance audit (`sha256`/`phash`/`dhash` computed
directly from the real dataset files in that earlier session) — `original_
split` renamed back to `split` and every partition-derived column (`group_
id`, `new_split`, etc.) dropped, so `run_partition_design` re-derives
duplicate groups, conflicts, and the split assignment fresh from real,
unmodified per-image hash/class/path data, not from any prior run's
conclusions.

The 4 `--exposed-image-ids` are the same Phase 6-disclosed smoke-test
images every prior phase has excluded from valid/test
(`PHASE6_RESULTS.md` §2), looked up by their real `image_id` in the
reconstructed manifest, not re-typed from memory.

**Determinism verified**: re-ran the identical command in a separate
process; output `partition_manifest.csv` was byte-identical (same
SHA-256).

## 4. Result

| Metric | Value |
|---|---|
| Total dataset N | 3,688 |
| Included (train+valid+test) | 3,395 |
| Excluded (conflict) | 293 (7.94%) |
| Train | 2,599 |
| Valid | 284 |
| Test | 512 |
| Exact-match edges | 688 |
| Near-dup edges (dHash ≤2) | 2,103 |
| Total duplicate groups | 2,367 (756 multi-member, largest = 9) |
| Groups containing ≥1 exact-match edge | 645 |
| Groups containing ≥1 near-dup edge | 755 |
| Cross-label conflict groups excluded | 93 (293 images) |
| Duplicate group crossing a split | **0** (verified two independent ways — see §5) |
| Exposed image in valid/test | **0** |

**Class distribution by split:**

| Split | Angry | Fear | Happy | Sad |
|---|---|---|---|---|
| Train | 572 | 461 | 887 | 679 |
| Valid | 63 | 50 | 97 | 74 |
| Test | 112 | 91 | 175 | 134 |
| Excluded (conflict) | 94 | 75 | 40 | 84 |

**Manifest**: `outputs/t0_automated/final_partition/partition_manifest.csv`
(gitignored, local-only, per the established `outputs/` convention — this
document + its recorded SHA-256 is the durable, committed record).

**Manifest SHA-256**: `4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d`

Freeze/status record (all required fields, `not_human_reviewed: true`):
`outputs/t0_automated/T0_AUTOMATED_CONSERVATIVE_V1.json`.

## 5. Verification

Two independent checks confirm no duplicate group crosses a split
boundary: (a) `partition.run_partition_design`'s own union-find-based
`verify_no_group_crosses_partitions` (part of `leakage_verification_
report.json`, `ok: true`), and (b) a separate, edge-level check added this
phase (`dataset_gate.check_automated_conservative_gate`'s `no_exact_
duplicate_group_crosses_splits` / `no_near_duplicate_group_crosses_splits`)
that directly re-reads every exact and near-dup edge and confirms both
endpoints share a split — 0 violations of either kind, computed
independently of the partition algorithm's own internal bookkeeping.

**`dataset_gate.check_automated_conservative_gate()` — PASS.** All 10
checks pass: manifest exists, SHA-256 recorded and matches, methodology
explicitly self-labeled `not_human_reviewed: true` (never claims Phase 7B
review completion), no exact-duplicate cross-split edge, no near-duplicate
cross-split edge, no conflict group improperly included, no exposed image
in valid/test, class/count totals internally consistent
(`counts_match_manifest.ok: true`), train/valid/test disjoint
(`no_image_in_multiple_partitions.ok: true`), and every manifest row's
path fields are present and well-formed.

**Stated limitation, not hidden**: `all_paths_well_formed` verifies path
FIELDS are non-empty and correctly prefixed (`train/`, `valid/`, `test/`,
`excluded_conflict/`) — it does **not** verify the raw image files exist
on disk, because the raw Combined_Drawing dataset is not accessible in
this session's environment (path root references a different machine/user
than this checkout). Physical file existence must be re-verified on
whichever machine actually runs E1/E2 training before that training
starts — this is a real, disclosed gap in what this gate can check from
this environment, not a silently-skipped requirement.

`dataset_gate.check_clean_split_gate()` (the original Phase 7B human-review
gate) is completely unmodified by this addition — it still correctly
reports `gate_passed: false` against the real repository (0/225 human
decisions recorded), exactly as before. The two gates are independent
functions in the same module; neither reads or writes the other's files.

## 6. Important limitation — stated explicitly, per instruction

**Near-duplicate groups in this manifest were algorithmically defined
(dHash Hamming distance ≤2, transitively closed via union-find), NOT
human-adjudicated.** No person has visually confirmed any of the 2,103
near-dup edges or 645+111=756 resulting multi-member groups. The evidence
supporting this threshold (§2, `T0_THRESHOLD_SENSITIVITY.md`) is itself
AI-preliminary judgment from an earlier session's blinded audit, not
independent human ground truth — the same limitation `PHASE7B_DUPLICATE_
POLICY.md` §3/§17 already disclosed for that underlying evidence. This
methodology accepts that limitation explicitly rather than deferring
indefinitely on an incomplete manual review, and chose the most
conservative well-evidenced threshold available specifically to bound the
resulting risk.

## 7. What was NOT done, per explicit instruction

No raw dataset image was accessed or re-hashed this phase (the `dhash`
column was already real and present in the reconstructed manifest from an
earlier session). No Phase 7B human-review item was decided, simulated, or
skipped-and-assumed. No model was trained, tuned, or evaluated. E6 was not
touched. `annotations/A1`/`A2` were not touched.
