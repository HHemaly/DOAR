# Phase 7B — Duplicate-Policy Validation and Final Partition Refinement

**Status: dataset analysis and split construction only. No model was
trained, tuned, calibrated, or evaluated in this phase.** The original
dataset was never modified (re-verified, see §9). Every Phase 3–7A artifact
and commit is unchanged.

**IMPORTANT CORRECTION (same session, continuation): the dHash-threshold-6
policy and the unrestricted-single-linkage clustering described in §§4–8
below are NOT finalized.** A closer, purpose-built audit (§14) found that
threshold 6 was chosen from an insufficiently targeted evidence base and is
very likely too permissive; a genuinely heterogeneous, multi-template
17-image component was found at threshold 6 (§15); and unrestricted
single-linkage was found to under-perform a complete-linkage constraint on
that same component (§16). **`outputs/phase7b/final_partition/` — built at
threshold 6 in the prior part of this session — remains on disk for
reference but must NOT be treated as final or locked.** No partition has
been regenerated in this continuation; §17 explains why, and what is needed
before one can be. Everything in §§2–13 that reports the *original* aHash
threshold-5 reproduction, the existence of chaining, and the qualitative
case for switching hash methods remains accurate and is not retracted —
only the specific numeric conclusion "threshold 6 is the right choice" is
withdrawn, replaced by the more careful analysis in §§14–17.

**Second continuation, same phase: an interactive human-review application
now exists (§24), and the human-review package's static CSVs (§17) are
superseded by it as the recommended way to record decisions.** A
provisional, evidence-grounded comparison of conservative policies (exact
duplicates + dHash ≤2, ≤3, and a selectively-reviewed distance-4 tier) is
in §25 — explicitly a comparison, not a selection; no threshold or
clustering policy is approved and no partition has been regenerated.

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

**§14 CORRECTION: this "should be revisited" caveat turned out to matter.**
The 45-pair sample above was drawn from pairs that aHash *already* flagged
as near-duplicates (at various aHash distances), with dHash distance
recorded only incidentally for those same pairs — it is not a sample of
what dHash itself considers close across its own full distance range. A
second, purpose-built audit sampling directly at dHash distances 2–8 (§14)
found precision already falls to 50% at distance 4 and further at 5–8 —
materially earlier than this section implied. **Threshold 6 is likely too
permissive; see §14 for the corrected evidence and §17 for why no new
threshold has been locked in yet.**

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

**§16 CORRECTION: single-linkage was retained too quickly here.** Direct
inspection of the 17-image dHash-6 component (§15) found it is not one
coherent cluster but a chain of at least 4 visually distinct sub-groups
plus several clearly-unrelated singletons swept in transitively. §16
implements and tests complete-linkage on this exact component: it correctly
isolates the clearest outlier and tightens the largest sub-cluster, but
does **not** fully resolve two remaining pairs where the underlying hash
distance itself (not the linkage rule) is misleadingly small for two
visually unrelated images. Complete-linkage is a real improvement, not a
complete fix — see §16 for the concrete evidence and §17 for the combined,
still-not-finalized recommendation.

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

**§17 CORRECTION: despite the section title, this partition is NOT final
and is NOT locked.** It was built at dHash threshold 6 earlier in this
session; §14–16's corrected evidence indicates threshold 6 is likely too
permissive and that single-linkage alone leaves residual heterogeneous
groups. The artifacts below remain on disk, byte-verified and untouched,
for reference and comparison — but no training, evaluation, or downstream
use should treat `outputs/phase7b/final_partition/` as the approved split
until the human-review package (§17) is reviewed and a policy is
explicitly approved.

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

## 14. Targeted dHash boundary audit and corrected threshold evidence

**Why a second audit was needed.** §3–4's comparison of aHash against dHash
was computed on 45 pairs that were originally selected because aHash
flagged them as near-duplicates at various aHash distances — dHash
distance was only recorded *incidentally* for those same pairs. That is a
biased sample for choosing a dHash-specific threshold: it says how dHash
scores pairs aHash already likes, not how precision behaves across dHash's
own distance range. A second, purpose-built audit was run: **42 new pairs,
sampled directly from dHash's own near-dup edges at each exact distance 2
through 8** (6 pairs per distance, oversampling cross-label pairs),
composited and judged with the same blinding procedure as §3
(`outputs/phase7b/boundary_audit/pair_metadata.json` / `judgments.json`).

### Results

| dHash distance | n | True-duplicate-like | Precision | 95% Wilson CI |
|---|---|---|---|---|
| 2 | 6 | 6 | 100.0% | [61.0%, 100%] |
| 3 | 6 | 6 | 100.0% | [61.0%, 100%] |
| **4** | 6 | 3 | **50.0%** | [18.8%, 81.2%] |
| **5** | 6 | 0 | **0.0%** | [0.0%, 39.0%] |
| 6 | 6 | 2 | 33.3% | [9.7%, 70.0%] |
| 7 | 6 | 2 | 33.3% | [9.7%, 70.0%] |
| 8 | 6 | 1 | 16.7% | [3.0%, 56.4%] |

**Precision falls sharply between distance 3 and distance 4 — not between
distance 5 and 16 as §4's aHash-conditioned sample implied.** Distances 6–7
show a partial rebound to 33% in this small sample, most plausibly sampling
noise (n=6 per bucket, wide overlapping CIs) rather than a genuine
non-monotonic effect — the honest reading is "precision is already
compromised from distance 4 onward," not "there is a safe plateau at 6–7."

**This directly contradicts §4's chosen threshold of 6.** §4's claim that
"every true-duplicate-like pair had dHash distance ≤5; every false-positive
pair had dHash distance ≥16" was accurate for its own (confounded) sample,
but did not generalize — this targeted sample shows real false positives
already appearing at distance 4–5, well before 16.

**Full-dataset cross-check**: `outputs/phase7b/dhash_threshold_sensitivity.txt`
(§4) shows the largest single-linkage component stays at 9 images through
threshold 4, then jumps to 17 at threshold 6 — consistent with this
section's finding that something changes for the worse in the 4-to-6
range, not smoothly.

**Revised reading of the evidence**: thresholds 2–3 are well-supported (100%
precision in both this and considering the population-level cross-label
edge ratio, §2, which is lowest at low thresholds). Threshold 4 is
borderline (50%, wide CI). Threshold 6 — this phase's prior choice — sits
in a zone with materially lower demonstrated precision than assumed.
**No single threshold is locked in by this document** (§17) — the human
reviewer should weigh recall (a stricter threshold catches fewer true
near-duplicates) against this precision evidence.

---

## 15. The 17-image cross-label component, inspected directly

The single dHash-threshold-6 component that spans multiple classes (flagged
as a conflict group in §7/§8) was extracted and rendered as a labeled
contact sheet: `outputs/phase7b/contact_sheets/component_17_dhash6.jpg`.

**Direct visual inspection finds it is not one coherent duplicate cluster.**
It contains at least four distinguishable sub-groups:

1. **A tight, genuinely-duplicate-looking cluster** (≈5 images) — the same
   "angry twig-figure with triangle-toothed face and scattered triangles"
   drawing, appearing in near-identical framing across multiple files.
2. **A second, related but distinct cluster** (≈2–3 images) — the same
   twig-figure template *without* the face/triangle elements.
3. **A third, distinct simple-face template** (≈2 images) — a plain oval
   with two dot eyes and a curved mouth, unrelated in actual content to
   clusters 1–2 despite superficial low-entropy similarity.
4. **Several clearly-unrelated singletons swept in transitively**: a noisy,
   low-legibility scribble on a glittery/speckled surface; a notebook-paper
   portrait sketch; a photo of a child crouching/hugging their knees. None
   of these three resemble any other member of the component.

This confirms, on real data, the general chaining mechanism already
described in §2/§14: locally-real similarity (cluster 1's members really do
look alike) gets transitively bridged through weaker intermediate links
into a component that is globally heterogeneous. **The threshold-growth
trace makes this concrete**
(`outputs/phase7b/contact_sheets/threshold_growth_t{2,3,4,6}.jpg`,
tracing the same 17 images' largest shared group across thresholds): 2
images at threshold 2 → 5 at threshold 3 (all visually identical to each
other) → 9 at threshold 4 (visually coherent except one questionable
inclusion) → all 17 at threshold 6 (visually heterogeneous, as above).

---

## 16. Complete-linkage vs. unrestricted single-linkage — a direct comparison

`partition.refine_with_complete_linkage()` (new) re-clusters every
single-linkage component of 3+ members using complete-linkage agglomeration
(two sub-clusters may only merge if *every* cross-pair between their
members stays within the threshold — i.e. the merged cluster's diameter,
not just one edge, must satisfy the threshold). Exact SHA-256 duplicates
are always kept together regardless of measured hash distance, since they
are definitionally identical content, not a near-dup judgment call.

**Full-dataset result** (dHash, threshold 6): of 2,303 single-linkage
components, only **5 actually change** under complete-linkage — most
components are already internally coherent, so this constraint costs
almost nothing where it isn't needed. Of the 5 that split:

| Original component | Original size | Sub-clusters after complete-linkage |
|---|---|---|
| The 17-image component (§15) | 17 | 6 sub-clusters: sizes 5, 4, 3, 2, 2, 1 |
| 4 other components | 3–7 each | 2 sub-clusters each |

**Contact sheet** (color-coded by resulting sub-cluster):
`outputs/phase7b/contact_sheets/component_17_complete_linkage_subclusters.jpg`.

**Complete-linkage measurably improves the 17-image component**: the
clearest outlier (the noisy glitter-surface scribble) is correctly isolated
as its own singleton sub-cluster, and the tightest, most visually coherent
5-image cluster (§15's cluster 1) is preserved intact and separated from
the rest.

**But it does not fully solve the problem.** Two of the six resulting
sub-clusters still pair visually unrelated images:
- One 2-member sub-cluster pairs a twig-figure drawing with a photo of a
  child crouching/hugging their knees — visually unrelated, yet their
  *measured* hash distance is within the threshold, so complete-linkage's
  "all pairs must be close" rule cannot separate them (with only 2
  members, that rule reduces to checking the single pairwise distance,
  identical to single-linkage's own criterion for a pair).
- A 4-member sub-cluster mixes a simple face template, a twig-figure
  variant, and a notebook-paper portrait — again, all pairwise distances
  among these 4 are within the threshold per the hash itself.

**Conclusion**: complete-linkage fixes *chaining* (transitive bridging
through weak intermediate links) but cannot fix *individual hash
mismeasurement* (two directly-compared images whose hash distance is
misleadingly small). These are two different failure modes needing two
different remedies — a stricter threshold (§14) addresses the second;
complete-linkage addresses the first; **neither alone is sufficient, and
the two residual mismatched pairs identified above are exactly the kind of
case flagged for human review (§17), not resolved by any automated method
tried here.**

---

## 17. Human-review package and why no partition has been (re)locked

**Per explicit instruction, no partition was regenerated or locked in this
continuation.** `outputs/phase7b/final_partition/` (threshold 6,
single-linkage, built earlier in this session) remains on disk unmodified,
but is explicitly **not** endorsed as final given §14–16's findings.
Building a new "final" partition now, before a human has weighed in on the
specific ambiguous pairs and groups this session surfaced, would repeat the
same mistake this correction is fixing — picking a number and moving on
without adequate evidence.

**Everything AI-judged in this phase — both the original 87-pair audit
(§3) and this continuation's 42-pair boundary audit (§14) — is a
preliminary annotation, not independent ground truth.** All judgments were
made by the same AI system that designed the sampling and thresholds, not
an independent human or expert rater; no inter-rater reliability statistic
exists. This was already stated in §3 and is restated here because it
governs how the human-review package below should be used: as a
**starting point to correct or confirm**, not a result to accept as-is.

**Human-review package**: `outputs/phase7b/human_review/`
(see `README.md` there for full usage instructions):

- **`pair_review.csv`** (129 rows) — every pair from both audits, with the
  AI's preliminary judgment, a link to the anonymized composite image, and
  empty `human_judgment`/`human_notes` columns to fill in.
- **`group_review.csv`** (50 rows) — the 17-image component's 6
  complete-linkage sub-clusters, plus its threshold-growth trace (2/3/4/6),
  with empty `human_decision`/`human_notes` columns.
- Supporting contact sheets in `outputs/phase7b/contact_sheets/`:
  `component_17_dhash6.jpg` (all 17, unsorted),
  `component_17_complete_linkage_subclusters.jpg` (color-coded by
  sub-cluster), `threshold_growth_t{2,3,4,6}.jpg` (the core cluster's
  growth), and `ambiguous_boundary_pairs_d4_5_6.jpg` (the 18
  boundary-audit pairs at the specific distances — 4, 5, 6 — where
  precision was most uncertain).

**What is needed before a partition can be locked**: a human reviewer
completing (in whole or in part) `pair_review.csv`/`group_review.csv`, from
which a final threshold and clustering policy can be chosen with actual
confirmed/corrected ground truth rather than AI-preliminary annotations
alone. Only after that should `main.py build-partition` be re-run to
produce a genuinely final partition.

**Superseded by §24 (second continuation):** editing these CSVs by hand is
no longer the recommended workflow — an interactive, point-and-click
review application now exists for exactly this purpose. The CSVs above
remain as a read-only snapshot of Claude's original preliminary judgments.

---

## 18. Dataset root and full verification (this continuation)

**Dataset root**, determined two independent ways from
`outputs/phase5/manifest.csv`'s `path` column and cross-checked against its
`relative_path` column: **`C:\Users\Ahmed\Downloads\Combined_Drawing\Combined_Drawing`**.
Both `os.path.commonpath()` over all 3,688 `path` values and subtracting
each row's own `relative_path` suffix from its `path` agree exactly on this
root.

**Path resolution**: every one of the 3,688 manifest rows' `path` values
was checked directly against the filesystem this continuation — **0
missing, 3,688/3,688 resolve** to a real, readable file.

**Full test suite**: `pytest tests/` — **260 passed** (255 prior + 5 new
`CompleteLinkageRefinementTests`), 9 pre-existing warnings, 0 failures.

**compileall**: exit 0.

**Ruff**: `ruff check .` — **792 findings, exit 1**, identical to the
established baseline (confirmed via `git status`: only
`src/doar/partition.py` [extended with `refine_with_complete_linkage`] and
`tests/test_partition_phase7b.py` [5 new tests] changed this continuation
— no new Ruff finding was introduced).

---

## 19. Original data and prior-phase preservation — verified, not assumed

- Re-hashed the same already-disclosed smoke-test image
  (`2-1_jpg.rf.f967ea55654e6fa4badbe2906373b7f7.jpg`) directly from disk:
  SHA-256 prefix `56238adc1155eafb`, matching every prior phase's recorded
  value exactly (checked again in this continuation, §18).
- Re-verified all 12 Phase 7A artifact hashes
  (`outputs/phase7a/ARTIFACT_HASHES.json`) against the files currently on
  disk: **ALL MATCH** — Phase 7A's provisional partition and analysis are
  untouched.
- `git status` at the end of this continuation shows only
  `src/doar/partition.py` (extended) and `tests/test_partition_phase7b.py`
  (5 new tests) changed — no file under `outputs/`, no prior-phase `.md`,
  and no previously-existing test file besides the ones just named was
  modified.

---

## 20. Phase 7 experiment-feasibility reassessment (revision 2)

Given the improved effective training size (1,414 independent groups vs.
Phase 7A's 1,273) and cleaner partition **as it stood before §14–17's
correction**:

- **Classical baselines (C1–C5)**: unchanged, still Recommended — unaffected by this data-quality improvement either way.
- **DenseNet121 (A1)**: still Recommended; the params-to-independent-group ratio improves marginally (≈4,950 vs. Phase 7A's ≈5,500 params/group) but not enough to change the assessment materially.
- **ConvNeXt-Tiny (A2)**: **stays downgraded to Optional, probe-first** (per `PHASE7_EXPERIMENT_MATRIX_REVISION.md`) — the params/group ratio (≈19,660) remains unfavorable even with the improved partition; this is not reversed just because more data survived.
- **DINOv2/OpenCLIP frozen representations (B1–B4)**: still relatively favorable versus full fine-tuning; unchanged.
- **Transformer probes (A3–A5)**: unchanged, still capped/conditional.
- **Fusion experiments (D1–D4)**: unchanged in design; must use whichever partition is eventually approved (§17) as the development-time reference, not necessarily `outputs/phase7b/final_partition/` as-is.
- **Three-seed screening vs. stronger shortlist confirmation**: this recommendation is **now itself provisional** pending §17's review — it assumed the threshold-6 partition's data quality, which §14–16 found to be less certain than stated. Revisit once a reviewed policy exists.

**No experiment is added or removed in this revision** beyond what
`PHASE7_EXPERIMENT_MATRIX_REVISION.md` already proposed — this section only
updates the *reasoning* behind those proposals; `PHASE7_EXPERIMENT_MATRIX.csv`
remains unmodified. **This entire section's specific numbers should be
re-checked once §17's review concludes and any new partition exists.**

---

## 21. Verification (tests, Ruff, compileall)

- **New/extended code (this continuation)**: `src/doar/partition.py` gains
  `refine_with_complete_linkage()` (§16). No other function was modified;
  `compute_duplicate_groups`, `build_group_disjoint_partition`, and every
  other Phase 7A/7B function are unchanged from the prior commit.
- **New tests**: `tests/test_partition_phase7b.py` gains a
  `CompleteLinkageRefinementTests` class (5 tests): singleton/pair
  components pass through unchanged, a synthetic heterogeneous chain
  correctly splits, exact duplicates always merge regardless of measured
  hash distance, the split report records correct sub-cluster sizes, and
  every image still appears exactly once after refinement.
- **Test suite**: `pytest tests/` — **260 passed** (255 prior + 5 new), 9
  pre-existing warnings, 0 failures. See §18 for the full report.
- **compileall**: exit 0.
- **Ruff**: `ruff check .` — **792 findings, exit 1**, identical to the
  established baseline. No new finding introduced (confirmed via `git
  status`).

**Second continuation (the interactive review app, §24-25)**: added
`src/doar/human_review.py` (new module) and `tests/test_human_review.py`
(new, 29 tests covering Wilson-interval math, the blinding guarantee,
incremental decision save/reload, agreement-rate computation, and all five
export files' correctness against synthetic registries). `pytest tests/`
— **289 passed** (260 prior + 29 new), 9 pre-existing warnings, 0
failures. `compileall src main.py streamlit_app.py phase7b_review_app.py`
— exit 0. `ruff check src tests main.py streamlit_app.py
phase7b_review_app.py` — 0 findings in the touched files; `ruff check .`
repo-wide — 792 findings, exit 1, identical to the established baseline
(no new finding introduced). The Streamlit app was also smoke-tested
headlessly (`streamlit run ... --server.headless true`), confirmed to
serve HTTP 200 and pass its own health check, and confirmed to create no
`decisions.json` or other file until a decision is actually saved — no
partition, model, or dataset file was touched by this verification.

---

## 22. Deliverables and artifact locations

All under `outputs/phase7b/` (gitignored; `ARTIFACT_MANIFEST.tsv` — 255
files, ≈15.1 MB — is the durable hash record, rebuilt this continuation):

1. This document, `PHASE7B_DUPLICATE_POLICY.md` (committed).
2. `outputs/phase7b/threshold_audit_detailed.json` — per-threshold audit (§2).
3. `outputs/phase7b/blind_audit/` — round-1 audit, 87 pairs (§3).
4. `outputs/phase7b/boundary_audit/` — round-2 targeted dHash audit, 42 pairs (§14, new this continuation).
5. `outputs/phase7b/contact_sheets/` — `component_17_dhash6.jpg`, `component_17_complete_linkage_subclusters.jpg`, `threshold_growth_t{2,3,4,6}.jpg`, `ambiguous_boundary_pairs_d4_5_6.jpg` (§15–17, new this continuation).
6. `outputs/phase7b/human_review/` — `pair_review.csv` (129 rows), `group_review.csv` (50 rows), `README.md` (§17, superseded as the primary workflow by §24), plus `app_data/items_registry.json` (225 items) and `app_data/decisions.json` (created on first save) and `exports/` (created on first export) — new this continuation.
7. `outputs/phase7b/human_review/review_pairs/` — 96 new anonymized composite images for the 17-image component (47), the policy-change sample (20), and the distance-4 worklist (29), new this continuation.
8. `src/doar/human_review.py` + `phase7b_review_app.py` (Streamlit) — the interactive review application (§24), new this continuation.
9. `tests/test_human_review.py` — 29 tests for the review application's core logic, new this continuation.
10. `outputs/phase7b/final_partition/` — the threshold-6 partition from earlier in this session, preserved but **explicitly not final** (§8, §17).
11. `PHASE7_EXPERIMENT_MATRIX_REVISION.md` (Phase 7A, unchanged) + §20 above — provisionally revised Phase 7 experiment recommendation.
12. `SESSION_HANDOFF.md` (updated separately, see the commit).

---

## 23. Explicitly out of scope for this phase (per user instruction)

The following were **not** done and should not be inferred from this
document: training, tuning, calibrating, or evaluating any model; using the
test split for model or hyperparameter selection; inspecting individual
test-split predictions; activating any psychological interpretation;
claiming subject-level independence; silently modifying
`PHASE7_EXPERIMENT_MATRIX.csv`; regenerating or locking a final partition
(§17); starting Stage 0 or any other Phase 7 experiment.

---

## 24. Interactive human-review application (second continuation)

**Why:** the CSV-editing workflow in §17 works but is tedious and offers
no structural guarantee against accidentally revealing label/split/hash/AI
information while reviewing. A local, point-and-click alternative removes
both problems: it enforces blinding in code (`blind_item_for_display` in
`src/doar/human_review.py` is the single function allowed to decide what
the reviewer's screen receives) and saves every decision immediately.

**What it covers — 225 pairs across the same 4 navigation sections the
instruction specified**, built from real, computed data (never guessed):

| Tab | Pairs | Source |
|---|---|---|
| Ambiguous pairs | 87 | Round 1 blind audit (§3), unchanged |
| Threshold-boundary pairs | 71 | Round 2's 42 pairs (§14) + all 29 remaining dataset-wide dHash-distance-4 edges not already covered elsewhere (of 43 total; see §25) |
| The 17-image component | 47 | Every real near-dup edge (dHash ≤6) connecting two of the 17 members — the complete internal edge set, not a sample |
| Groups that change between candidate thresholds 2/4/6 | 20 | A seeded, cross-label-prioritized sample of dataset-wide bridging edges whose distance falls strictly between two candidate thresholds (2<d≤4 or 4<d≤6), excluding the 17-image component (already covered exhaustively) |

**Blinding**: each pair is shown as an anonymized side-by-side composite
image with no emotion label, split, hash distance, or Claude's preliminary
judgment. The reviewer picks one of exactly 4 decisions (`Definite
duplicate` / `Same underlying drawing with transformation` / `Different
drawings` / `Uncertain`) plus optional notes. Nothing is preselected; a
previously-recorded decision is shown again only when the reviewer
navigates back to an item *they themselves already decided* (so they can
revise it), never derived from or defaulted to Claude's judgment. An
optional "reveal hidden details" panel is gated so it only unlocks **after**
a decision for that item is already saved — it cannot influence an answer
that no longer needs to be given.

**Persistence**: every click writes immediately (atomic file replace) to
`outputs/phase7b/human_review/app_data/decisions.json`. Closing the
browser tab or terminal loses nothing; re-launching resumes exactly where
you left off, including per-tab progress and a "jump to first unreviewed
item" shortcut.

**Launch it** (Windows PowerShell, from the repository root):
```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run phase7b_review_app.py
```
Open the local URL Streamlit prints — normally **http://localhost:8501**.
Full instructions: `outputs/phase7b/human_review/README.md` and
`RUN_GUIDE_WINDOWS.md` §11.

**Export** (sidebar button, safe to run at any point, repeatedly, including
mid-review) writes to `outputs/phase7b/human_review/exports/`:
`human_pair_reviews.csv` (one row per pair, AI-preliminary and human
columns side by side), `human_group_reviews.csv` (per-member verdict for
the 17-image component + per-transition verdict for the policy-change
sample), `reviewer_agreement_report.json` (AI-vs-human agreement, only
where the AI actually pre-judged), `threshold_precision_summary.json`
(precision-by-exact-distance with Wilson 95% CIs, computed separately from
real human decisions and from Claude's preliminary judgments so the two
are never conflated), and `unresolved_items.csv` (never-reviewed +
marked-`uncertain` items).

**Tests**: `tests/test_human_review.py` (29 tests) cover the blinding
guarantee, incremental save/reload, decision validation, agreement-rate
math, Wilson-interval math, and all five export files' row-level
correctness, using synthetic registries — no dependency on this session's
specific 225 real pairs, so the logic stays covered even as items change.

**No policy is selected or locked by this tool.** It produces evidence;
choosing a threshold and clustering method from that evidence remains a
separate, explicit human decision (§17), after which `main.py
build-partition` would need to be re-run.

---

## 25. Conservative-policy comparison — provisional, pending human review

**This is a comparison of options, not a recommendation to adopt any one
of them, and not a locked policy.** Real human review (§24) has not
happened yet as of this writing — the numbers below combine (a) dataset-
wide structural counts, which are exact and reproducible, and (b) the two
prior audits' AI-preliminary judgments as a provisional precision estimate,
explicitly not equivalent to human-confirmed ground truth. Once the review
app's decisions exist, re-run `compute_precision_by_distance`/
`build_threshold_precision_summary` (`src/doar/human_review.py`) with
`judgment_source="human"` for the real answer.

**AI-preliminary precision by exact dHash distance** (Round 2 boundary
audit, n=6 per bucket, Wilson 95% CI — small samples, wide intervals):

| Distance | Precision | 95% CI |
|---|---|---|
| 2 | 100.0% | 61.0%–100.0% |
| 3 | 100.0% | 61.0%–100.0% |
| 4 | 50.0% | 18.8%–81.2% |
| 5 | 0.0% | 0.0%–39.0% |
| 6 | 33.3% | 9.7%–70.0% |
| 7 | 33.3% | 9.7%–70.0% |
| 8 | 16.7% | 3.0%–56.4% |

**Structural comparison of candidate thresholds** (computed on the real
manifest, exact sha256 duplicates always merge regardless of threshold
since identical files always have dHash distance 0):

| Policy | Largest component | Near-dup edges included | Cross-label edge ratio | Images in a conflicted group |
|---|---|---|---|---|
| exact + dHash ≤2 | 9 | 2,103 | 9.9% | 293 |
| exact + dHash ≤3 | 9 | 2,184 | 9.9% | 301 |
| exact + dHash ≤4 (blanket, shown only for contrast) | 9 | 2,227 | 10.0% | 311 |
| exact + dHash ≤6 (§8's provisional, not-approved partition) | 17 | 2,290 | — | 338 |

Distances 2 and 3 both show 100% AI-preliminary precision (small n) and
near-identical structural cost (largest component stays at 9; cross-label
ratio and conflict counts barely move between them) — **either is a
plausible conservative floor**, with ≤3 recovering slightly more usable
data for a similar apparent risk. Distance 4 is where precision visibly
splits (50% in the small sample) without yet showing the sharp structural
blow-up seen at 6, which is exactly why it is treated as a selective,
edge-by-edge tier rather than blanket-included.

**The "selectively reviewed distance-4" policy, concretely**: include
distance ≤3 automatically; for distance-4 edges, include a specific edge
in a duplicate group *only if* a human has reviewed and confirmed it.
There are **43 such edges dataset-wide**, all now included in the review
app's "Threshold-boundary pairs" tab (6 from the original Round 2 sample,
plus all 29 remaining ones added this continuation) — a fully tractable,
bounded review task, not a sample extrapolated from 6 pairs.

**What this does NOT do**: select a threshold, choose single- vs.
complete-linkage, or regenerate/lock `outputs/phase7b/final_partition/`.
Those remain explicit next steps, gated on the human review this section's
evidence is meant to inform.

---

**This phase stops here**, per explicit instruction, after producing the
(intentionally non-final) threshold recommendation, the comparison
artifacts, the human-review package and its interactive application, the
conservative-policy comparison, the completed tests, and this updated
report. No duplicate-detection policy has been approved; no partition has
been locked; no model was trained or evaluated.
