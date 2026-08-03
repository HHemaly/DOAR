# Phase 7B — Duplicate-Policy Validation and Final Partition Refinement

**Status: dataset analysis and split construction only. No model was
trained, tuned, calibrated, or evaluated in this phase.** The original
dataset was never modified (re-verified, see §9). Every Phase 3–7A artifact
and commit is unchanged. The Phase 7A threshold-5 partition is **provisional**
— this phase supersedes it with an evidence-based final policy, without
deleting or altering the Phase 7A artifacts themselves.

---

## 1. Separating reproduction, provisional partition, and final policy

Three distinct things, deliberately not conflated:

| | What it is | Status |
|---|---|---|
| **Original leakage-analysis method** | `leakage.assess_leakage()`, SHA-256 exact + aHash near-dup (threshold 5), cross-split-only comparison | Unchanged, still the project's original detector, still reproducible exactly (re-confirmed this phase, §2) |
| **Phase 7A provisional partition** | Full-dataset union-find over the *same* aHash/threshold-5 method, group-disjoint stratified split | Preserved as-is at `outputs/phase7a/` (untouched, re-verified §9) — was always labeled provisional, now formally superseded for development use |
| **Phase 7B final policy** | Full-dataset union-find over a **different, evidence-selected** hash method (dHash) and threshold (6), same transitive-closure clustering approach, plus exposed-image handling | New, at `outputs/phase7b/final_partition/` — this phase's actual deliverable |

Nothing about the original aHash-based leakage analysis was changed to produce
this result — Phase 7B ran a **separate, additional** analysis and only
adopted a different method after a blinded audit produced clear evidence for
it (§4).

---

## 2. Threshold audit (aHash, thresholds 0–5)

Full detail: `outputs/phase7b/threshold_audit_detailed.json`. Reproduced and
extended Phase 7A's own scan with per-class and cross-label detail:

| Threshold | Near edges | Groups | Multi-image groups | Conflict groups | Conflict images | Largest group | Cross-label near edges | Cross-label edge ratio |
|---|---|---|---|---|---|---|---|---|
| 0 | 1,796 | 2,481 | 761 | 85 | 251 | 6 | 173 | 9.6% |
| 1 | 2,071 | 2,368 | 768 | 98 | 308 | 8 | 220 | 10.6% |
| 2 | 2,219 | 2,324 | 768 | 106 | 347 | 9 | 261 | 11.8% |
| 3 | 2,372 | 2,270 | 759 | 110 | 426 | 22 | 339 | 14.3% |
| 4 | 2,593 | 2,206 | 754 | 122 | 521 | 63 | 481 | 18.5% |
| **5 (original default)** | **3,069** | **2,106** | **730** | **124** | **674** | **137** | **791** | **25.8%** |

**The cross-label edge ratio climbs monotonically and substantially, from
9.6% at threshold 0 to 25.8% at threshold 5** — since two genuinely
duplicate images are very unlikely to have been independently mislabeled
into different emotion classes, this ratio is itself an implicit proxy for
the false-positive rate, computable without any manual review. It already
signals, before any visual audit, that threshold 5 is admitting a
substantially higher proportion of non-duplicate pairs than lower
thresholds.

### Why threshold 5 creates the 137-image component (directly inspected)

Composition check (`outputs/phase7a/large_group_composition.json`, restated
here): **13 of the 15 largest aHash-5 groups span 2–4 different emotion
classes** — the single largest (137 images) spans all 4. The blinded audit
(§3) directly sampled both **direct graph edges** and **non-adjacent
("shortcut") member pairs** from this exact group:

| Sample from the 137-image group | n | Judged true duplicate |
|---|---|---|
| Direct edges (pairs the algorithm actually linked) | 3 | **3/3 (100%)** |
| Non-adjacent members (both in the group, never directly linked) | 3 | **0/3 (0%)** |

**This is the mechanism, directly observed, not inferred:** the individual
edges that built this component are mostly real near-duplicates in
isolation. But **single-linkage transitive closure chains locally-real
matches into one giant, globally-heterogeneous component** — if A~B and B~C
are each genuine close matches, A and C end up in the same group even
though they may be completely unrelated drawings. The audit additionally
found that several very simple, low-visual-entropy templates recur many
times in this dataset and act as chaining hubs: a specific "sad face in the
rain" line sketch appeared as one side of 4 different sampled pairs (all
against unrelated drawings), a simple "circle with two dots and a curved
mouth" template appeared in 7 different sampled pairs, and a "branching
twig-person" stick-figure template appeared in 7 different sampled pairs.
Because aHash reduces an image to a coarse 8×8 brightness pattern, these
visually simple drawings collapse to nearly indistinguishable hashes
regardless of their actual (different) content, and single-linkage closure
then chains them all together transitively.

---

## 3. Blinded manual pair audit

### Method

87 pairs were sampled (fixed seed) across every required category: exact
duplicates (8), near-dup pairs at each exact aHash Hamming distance 0–5
(7–13 per distance, oversampling cross-label pairs within each), hard
negatives at distance 6 — just outside the current threshold (8), small
groups of size 2–3 (8), and both direct edges and non-adjacent shortcut
pairs from the 3 largest chained groups (sizes 137/33/30 — 6 each). Each
pair's two images were composited side-by-side into an anonymized file
(`pair_0001.jpg` … `pair_0087.jpg`, final order re-shuffled so category and
Hamming distance are not inferable from the filename), with all
ground-truth metadata (which images, which split, which class, which
category) held in a **separate file never opened during judgment**
(`outputs/phase7b/blind_audit/pair_metadata.json`). Each pair was judged
independently by viewing only the composite image, recording one of:
**definite duplicate**, **same underlying drawing with transformation**,
**visually similar but different drawing**, or **uncertain** —
`outputs/phase7b/blind_audit/judgments.json`. Judgments were joined with the
ground-truth metadata only after all 87 were recorded.

**Limitation stated plainly**: this audit was conducted by the same AI
system performing this analysis, not an independent human rater. Genuine
blinding was maintained procedurally (metadata never consulted during
judgment, composite images carry no split/label/category information), but
this is not equivalent to an independent second rater, and no inter-rater
reliability statistic can be computed. This is disclosed as a real
methodological limitation, not hidden.

### Results: precision by exact aHash Hamming distance

| Hamming distance | n | True-duplicate-like (definite + transformed) | Precision estimate | 95% Wilson CI |
|---|---|---|---|---|
| 0 | 7 | 7 | 100.0% | [64.6%, 100.0%] |
| 1 | 7 | 7 | 100.0% | [64.6%, 100.0%] |
| 2 | 7 | 2 | 28.6% | [8.2%, 64.1%] |
| 3 | 7 | 1 | 14.3% | [2.6%, 51.3%] |
| 4 | 7 | 2 | 28.6% | [8.2%, 64.1%] |
| **5 (original threshold boundary)** | **10** | **0** | **0.0%** | **[0.0%, 27.8%]** |
| 6 (hard negative, just outside threshold 5) | 8 | 0 | 0.0% | [0.0%, 32.4%] |

**Recall cannot be established** — there is no duplicate ground truth for
this dataset (no independent list of which images are "true" duplicates
exists), so it is impossible to know how many real duplicates the near-dup
detector *misses* at any threshold. Only precision (of the pairs flagged as
near-duplicate, what fraction are real) is estimable, and only for the
specific Hamming-distance strata sampled.

**Interpretation, stated honestly given small per-bucket samples (n=7–10):**
the confidence intervals are wide and adjacent buckets overlap somewhat.
But four independent lines of evidence all point the same direction and
corroborate each other: (1) the audit's point-estimate precision drops from
100% to ≤29% between Hamming 1 and 2, and reaches 0% at Hamming 5; (2) the
population-wide cross-label edge ratio climbs monotonically with threshold
(§2); (3) the largest-component chaining analysis (§2) shows exactly this
mechanism operating on the real data; (4) small groups (size 2–3, sampled
independently of Hamming distance) were **8/8 (100%) true duplicates**,
confirming that reliability is a property of *tight* matches, not of group
size per se. Taken together, this is a coherent, multi-angle case that
threshold 5 is admitting a large fraction of false positives, even though
no single number here should be over-interpreted in isolation.

### Additional finding: same-label vs. cross-label near-dup pairs

Cross-label near-dup pairs (n=13 sampled): 4/13 (31%) true-duplicate-like.
Same-label near-dup pairs (n=32 sampled): 15/32 (47%) true-duplicate-like.
Same-label pairs are somewhat more likely to be real duplicates, as
expected, but a meaningful fraction of even same-label "near-dup" pairs at
higher Hamming distances were still judged unrelated — confirming that
label agreement alone is not a reliable proxy for true duplication, and
`leakage.assess_leakage`'s exact-only conflict check (which never inspects
same-label near-dup pairs at all) cannot catch or rule out either error
mode.

---

## 4. Comparing aHash with dHash

**Not adopted merely for complexity** — dHash (difference hash: compare
each pixel to its horizontal neighbor in a resized 9×8 grayscale thumbnail,
64 bits) is computed with the same dependency-light approach as aHash
(resize + threshold, no new dependency), same 64-bit output, same Hamming
comparison. It is not a more complex method, only a different one.

**Evidence for switching**, computed on the *same* 45 near-dup audit pairs
where both hash distances could be measured:

| | aHash Hamming distance | dHash Hamming distance |
|---|---|---|
| Mean, true-duplicate-like pairs | 1.16 | 0.89 |
| Mean, not-duplicate pairs | 3.77 | 27.35 |

**Every true-duplicate-like pair in the sample had a dHash distance ≤5;
every not-duplicate pair had a dHash distance ≥16** — a clean separation
with no overlap in this sample, versus aHash's overlapping 0–5 range for
both classes. This is a substantially stronger discriminator on this
dataset.

**Full-dataset validation** (not just the audit sample — a dHash-based
threshold scan across the whole 3,688-image dataset,
`outputs/phase7b/dhash_threshold_sensitivity.txt`):

| dHash threshold | Groups | Near edges | Largest group | Conflict groups | Conflict images | Cross-label near edges |
|---|---|---|---|---|---|---|
| 2 | 2,367 | 2,103 | 9 | 93 | 293 | 209 |
| 4 | 2,327 | 2,227 | 9 | 98 | 311 | 222 |
| **6 (chosen)** | **2,303** | **2,290** | **17** | **103** | **338** | **243** |
| 8 | 2,257 | 2,425 | 48 | 111 | 412 | 313 |
| 10 | 2,157 | 2,811 | 170 | 115 | 554 | 549 |
| 12 | 1,858 | 4,257 | 553 | 114 | 974 | 1,493 |

dHash chains too at high enough thresholds (it is not immune to the
single-linkage mechanism, §2) — but at **threshold 6**, the largest
component is only 17 images (vs. aHash-5's 137), and composition-checking
the 8 largest dHash-6 groups shows **only 1 of 8 is multi-class** (the
17-image group itself — correctly excluded as a conflict below), compared
to 13 of 15 for aHash-5. This is a dramatically cleaner clustering
structure.

**Chosen threshold: 6.** This sits one Hamming unit above the highest
distance observed among true-duplicate-like audit pairs (5, a
"same-drawing-transformed" pair with a photo-angle difference), giving a
small margin, while the full-dataset scan confirms it keeps chaining
contained. This specific numeric choice is acknowledged as based on a
modest audit sample (45 pairs with both hash distances measured) and should
be revisited if a larger audit becomes available in the future — stated
honestly as a provisional-but-evidenced choice, not a definitive one.

**Implementation**: `dataset._difference_hash()` (new), `dataset.DHASH_THRESHOLD = 6` (new constant, kept separate from the pre-existing `NEAR_DUP_THRESHOLD = 5` so the original aHash reproduction stays exactly reproducible), `partition.compute_dhash_column()` (new), `partition.compute_duplicate_groups(..., hash_field=...)` (extended, defaults to `"phash"` for backward compatibility).

---

## 5. Clustering policy decision

**Single-linkage transitive closure is retained** — the audit evidence
(§3–4) shows the *threshold and hash method* were the actual problem, not
the linkage rule itself. At dHash threshold 6, single-linkage produces a
clean structure (largest component 17, mostly single-class groups). Adding
a more elaborate clustering method (complete-linkage, a maximum-diameter
constraint, or per-edge manual verification of every one of the ~2,290
near-dup edges) was considered and **not adopted**, because:

- It would be solving a problem (chaining) that is already resolved by the
  threshold/method change, at a real cost (complexity, more code, more
  failure modes) for no demonstrated additional benefit on this data.
- Per-edge manual verification of ~2,290 edges is not feasible at the scope
  of this phase (the 87-pair audit is already a substantial, real sampling
  effort; ~26x more images would be required for full coverage).
- The one remaining large-ish component (17 images, dHash-6) is correctly
  and automatically excluded as an unresolved label conflict regardless of
  linkage method, since it spans multiple classes (§7) — the existing
  conflict-exclusion mechanism already neutralizes the residual chaining
  risk without any new clustering machinery.

**This decision was made from the audit evidence in §2–4, not from which
policy retains the most images** — the chosen policy (dHash, threshold 6)
in fact retains *fewer* raw conflict-excluded images than a naively looser
policy would, and the decision was locked in before computing what the
resulting split sizes would be.

---

## 6. Handling previously-exposed images

The 4 Phase 6 smoke-test images (`PHASE6_RESULTS.md` §2) and every member of
their duplicate groups (under the final dHash-6 policy) are excluded from
valid/test:

| Image | Group size (dHash-6) | New split |
|---|---|---|
| `2-1_jpg.rf...` (Angry) | 4 | train |
| `Fear_1_10_jpg.rf...` (Fear) | 1 | train |
| `Happy_1_19_jpg.rf...` (Happy) | 1 | train |
| `3-4_jpg.rf...` (Sad) | 2 | train |

All 4 groups were clean (no label conflict), so all were force-assigned to
`train` (not a separate excluded-exposed bucket, since training does not
require blindness — only evaluation does, and forcing them to train keeps
the data usable rather than discarding it). Had any been an unresolved
conflict group, conflict status would take precedence and it would remain
`excluded_conflict`, never `train`. Verified automatically:
`verify_no_exposed_group_in_valid_or_test()` — **PASS**, 0 violations
(`outputs/phase7b/final_partition/leakage_verification_report.json`).

Notably, the Angry image's group shrank from 137 members (under the
Phase 7A aHash-5 provisional policy, where it was excluded as a conflict)
to just 4 members under the Phase 7B dHash-6 final policy — direct,
concrete evidence of the policy change's effect on this exact,
already-scrutinized image.

---

## 7. Duplicate weighting for future training (proposal only, not implemented)

Because both the Phase 7A and Phase 7B partitions place all multi-image
groups into `train` (a consequence of the largest-remainder allocation
algorithm always favoring `train`'s much larger target count — see
`PHASE7A_DATASET_READINESS.md` §5.3), a future training run's `train` split
contains non-independent, duplicated examples (in the final Phase 7B
partition: 2,564 images across only 1,414 independent groups — see §8).
Options considered:

| Option | Description | Assessment |
|---|---|---|
| One representative image per group | Drop all but one image per group before training | Simplest, but discards real (if partially redundant) data, including genuine minor variations (different photo angle/lighting of the same physical drawing) that could have modest naturalistic-augmentation value |
| All images, inverse-group-size sample weighting | Each image's loss contribution scaled by `1/group_size`, so each independent group contributes a total weight of 1 regardless of how many duplicate images it contains | Statistically well-motivated (standard correction for clustered/non-independent samples, analogous to survey design weights); simple to implement (a per-sample weight in the loss); does not discard any data |
| Group-balanced sampling | A custom sampler that draws groups (not images) uniformly per epoch, then one image per drawn group | Similar effect to inverse weighting in expectation, but requires new sampler code and is harder to reason about/debug than a per-sample weight |
| Controlled comparison of the above | Run all three under the existing multi-seed protocol | Full 3-way comparison is a legitimate future experiment (would fit naturally into Phase 7's Family E, fine-tuning ablations) but is explicitly **not run** here per this phase's scope |

**Recommended default: inverse-group-size sample weighting.** It is the
simplest option that does not discard data, is statistically the most
directly justified (equal-weight-per-independent-unit is the standard fix
for clustered sampling), and requires only a per-sample weight
multiplication already compatible with the existing `class_weighting`
mechanism in `train_image_model()` (the two weights would multiply:
class-balance weight × inverse-group-size weight).

**Recommended single limited ablation** (future phase, not run here): on
the Phase 7 Family A shortlist winner only, compare **no group weighting
(status quo)** vs. **inverse-group-size weighting**, 3 seeds each, holding
every other protocol element fixed — a single, well-scoped 2-condition
ablation, not a 3-way comparison, consistent with this programme's
"equal/normalized tuning budget" and "no speculative complexity expansion"
governance rules (`PHASE7_RESULTS.md` §4).

---

## 8. Final partition

### Configuration

`outputs/phase7b/final_partition/partition_config.json`: `seed=42`,
`hash_field="dhash"`, `near_dup_threshold=6`,
`split_ratios=[0.7651, 0.0841, 0.1511]` (unchanged target ratios — this
dataset's own original proportions), `algorithm=
"largest_remainder_greedy_bin_packing_per_class"`, 4 exposed image IDs
supplied.

### Raw image counts and effective independent-group counts, every split × class

| Split | Class | Images | Independent groups |
|---|---|---|---|
| train | Angry | 556 | 321 |
| train | Fear | 452 | 234 |
| train | Happy | 883 | 459 |
| train | Sad | 673 | 400 |
| **train total** | | **2,564** | **1,414** |
| valid | Angry | 61 | 61 |
| valid | Fear | 50 | 50 |
| valid | Happy | 97 | 97 |
| valid | Sad | 74 | 74 |
| **valid total** | | **282** | **282** (all singleton) |
| test | Angry | 109 | 109 |
| test | Fear | 89 | 89 |
| test | Happy | 174 | 174 |
| test | Sad | 132 | 132 |
| **test total (newly locked)** | | **504** | **504** (all singleton) |
| excluded_conflict | Angry | 115 | 62 |
| excluded_conflict | Fear | 86 | 55 |
| excluded_conflict | Happy | 45 | 43 |
| excluded_conflict | Sad | 92 | 50 |
| **excluded_conflict total** | | **338** | **103** |

As in Phase 7A, valid and test contain **only singleton groups** — a
structural consequence of the allocation algorithm (train's much larger
target count absorbs essentially every multi-member group), which remains
a genuine evaluation-validity positive (zero within-split duplication in
valid/test) at the cost of train's raw image count somewhat overstating its
independent information content (2,564 images / 1,414 groups ≈ 1.81 images
per independent unit).

### Comparison across all three partitions considered this session

| | Original (contaminated) | Phase 7A (aHash-5, provisional) | **Phase 7B (dHash-6, final)** |
|---|---|---|---|
| Train images | 2,821 | 2,305 | **2,564** |
| Train independent groups | *not measured* | 1,273 | **1,414** |
| Valid images | 310 | 254 | **282** |
| Test images | 557 | 455 | **504** |
| Excluded (conflict) | *not measured* | 674 (18.3%) | **338 (9.2%)** |
| Largest duplicate component | *not computed* | 137 | **17** |

Phase 7B's final policy recovers substantially more usable data than
Phase 7A's provisional one (259 more train images, 28 more valid, 49 more
test, 336 fewer images needlessly excluded) while being **more**, not less,
conservative about what counts as a true duplicate — this is the direct
payoff of correcting the detector rather than just tuning a threshold on
the same flawed method.

### Verification (all required checks, all passing)

| Check | Result |
|---|---|
| No duplicate group crosses a partition | **PASS** |
| No image assigned to more than one partition | **PASS** |
| No unresolved label conflict enters a supervised split | **PASS** |
| No previously-exposed group enters valid or test | **PASS** |
| Split/class count tally internally consistent | **PASS** |
| Manifest paths resolve to real, on-disk files | **PASS** |
| Deterministic reproduction (separate process, same seed) | **PASS** — re-ran `main.py build-partition` and confirmed byte-identical output hashes |

Full report: `outputs/phase7b/final_partition/leakage_verification_report.json`.

**The final test set (504 images) has not been visually inspected, has not
had any prediction generated against it, and no performance metric of any
kind has been computed from it in this phase.**

---

## 9. Original data and prior-phase preservation — verified, not assumed

- Re-hashed the same already-disclosed smoke-test image
  (`2-1_jpg.rf.f967ea55654e6fa4badbe2906373b7f7.jpg`) directly from disk
  this phase: SHA-256 prefix `56238adc1155eafb`, matching every prior
  phase's recorded value exactly.
- Re-verified all 12 Phase 7A artifact hashes
  (`outputs/phase7a/ARTIFACT_HASHES.json`) against the files currently on
  disk: **ALL MATCH** — Phase 7A's provisional partition and analysis are
  untouched by this phase.
- `git status` at the end of this phase shows only the files listed in §11
  changed or added — no file under `outputs/`, no prior-phase `.md`, and no
  previously-existing test file was modified (`test_partition.py` from
  Phase 7A is untouched; Phase 7B's additions are a new file,
  `test_partition_phase7b.py`, plus new, additive functions in
  `partition.py`/`dataset.py`).

---

## 10. Phase 7 experiment-feasibility reassessment (revision 2)

Given the improved effective training size (1,414 independent groups vs.
Phase 7A's 1,273) and cleaner partition:

- **Classical baselines (C1–C5)**: unchanged, still Recommended — unaffected by this data-quality improvement either way.
- **DenseNet121 (A1)**: still Recommended; the params-to-independent-group ratio improves marginally (≈4,950 vs. Phase 7A's ≈5,500 params/group) but not enough to change the assessment materially.
- **ConvNeXt-Tiny (A2)**: **stays downgraded to Optional, probe-first** (per `PHASE7_EXPERIMENT_MATRIX_REVISION.md`) — the params/group ratio (≈19,660) remains unfavorable even with the improved partition; this is not reversed just because more data survived.
- **DINOv2/OpenCLIP frozen representations (B1–B4)**: still relatively favorable versus full fine-tuning; unchanged.
- **Transformer probes (A3–A5)**: unchanged, still capped/conditional.
- **Fusion experiments (D1–D4)**: unchanged in design; must now use `outputs/phase7b/final_partition/` specifically (superseding Phase 7A's split as the development-time reference).
- **Three-seed screening vs. stronger shortlist confirmation**: the underlying data-quality concern that motivated caution is now measurably reduced (a verified, audit-backed near-dup definition instead of an untested inherited default) — this supports, as an optional strengthening rather than a requirement, allocating more seeds (e.g. 5 instead of 3) specifically for the **final shortlist confirmation** stage once one is reached, since the partition itself is now on firmer evidential footing. Initial screening can remain at 3 seeds.

**No experiment is added or removed in this revision** beyond what
`PHASE7_EXPERIMENT_MATRIX_REVISION.md` already proposed — this section only
updates the *reasoning* behind those proposals with the final partition's
actual numbers. `PHASE7_EXPERIMENT_MATRIX.csv` remains unmodified.

---

## 11. Verification (tests, Ruff, compileall)

- **New/extended code**: `src/doar/dataset.py` (`_difference_hash`,
  `DHASH_THRESHOLD` — additive, `_average_hash`/`NEAR_DUP_THRESHOLD`
  unchanged), `src/doar/partition.py` (`hash_field` parameter on
  `compute_duplicate_groups`, `compute_dhash_column`,
  `identify_exposed_groups`, `exposed_group_ids` support in
  `build_group_disjoint_partition`/`build_partition_manifest_rows`,
  `verify_no_exposed_group_in_valid_or_test`, `run_partition_design`
  extended with `hash_field`/`exposed_image_ids`), `main.py`
  (`build-partition` gains `--hash-field` and `--exposed-image-ids`).
- **New tests**: `tests/test_partition_phase7b.py` — 9 tests covering dHash
  determinism, `hash_field` selection, exposed-group identification and
  precedence-over-conflict, the new verification check, and an end-to-end
  `run_partition_design` determinism test under the full Phase 7B policy
  (dhash + threshold 6 + exposed images).
- **Test suite**: `pytest tests/` — **255 passed** (246 pre-existing + 9
  new), 9 pre-existing warnings, 0 failures.
- **compileall**: exit 0.
- **Ruff**: `ruff check .` — **792 findings, exit 1**, identical to the
  established Phase 5–7A baseline. No new finding introduced by this
  phase's code changes (confirmed via `git status` — only the files listed
  above changed).

---

## 12. Deliverables and artifact locations

All under `outputs/phase7b/` (gitignored; `ARTIFACT_MANIFEST.tsv` — 104
files, ≈7.4 MB — is the durable hash record):

1. This document, `PHASE7B_DUPLICATE_POLICY.md` (committed).
2. `outputs/phase7b/threshold_audit_detailed.json` — per-threshold audit (§2).
3. `outputs/phase7b/blind_audit/` — `pair_0001.jpg`…`pair_0087.jpg`, `pair_metadata.json`, `judgments.json` (§3).
4. `outputs/phase7a/large_group_composition.json` (Phase 7A, reused, not recomputed) + this phase's direct-edge/shortcut audit samples (§2–3) — chaining analysis.
5. `outputs/phase7b/final_partition/partition_config.json` — final duplicate-policy configuration.
6. `outputs/phase7b/final_partition/partition_manifest.csv` — revised train/valid/test/excluded_conflict manifest.
7. `outputs/phase7b/final_partition/label_conflict_review.csv` — exposed-image handling is recorded in `partition_manifest.csv`'s `exposed_status` column; no separate exposed-only manifest file was needed since only 4 already-small groups were affected (§6).
8. `outputs/phase7b/final_partition/leakage_verification_report.json` — leakage-verification report.
9. `PHASE7_EXPERIMENT_MATRIX_REVISION.md` (Phase 7A, unchanged) + §10 above — revised Phase 7 experiment recommendation.
10. `SESSION_HANDOFF.md` (updated separately, see the commit).

---

## 13. Explicitly out of scope for this phase (per user instruction)

The following were **not** done and should not be inferred from this
document: training, tuning, calibrating, or evaluating any model; using the
new test split for model or hyperparameter selection; inspecting individual
test-split predictions; activating any psychological interpretation;
claiming subject-level independence; silently modifying
`PHASE7_EXPERIMENT_MATRIX.csv`; starting Stage 0 or any other Phase 7
experiment.

**This phase stops here**, per explicit instruction, pending review of the
duplicate policy, the regenerated partition, and the revised experiment
recommendation.
