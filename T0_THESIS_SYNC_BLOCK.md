# T0 Thesis Sync Block — Automated Conservative Dataset Partition

**Milestone:** T0 (dataset-integrity foundation for E1/E2), not itself a
numbered experiment.
**Status:** Complete. Automated integrity gate PASSES.
**Methodology ID:** `T0_AUTOMATED_CONSERVATIVE_V1`.

**One-line summary:** A duplicate-controlled, group-disjoint Train/
Validation/Test partition of the 3,688-image Combined_Drawing corpus was
built using a fully automated, evidence-selected conservative near-duplicate
policy (dHash Hamming distance ≤2, seed 42) — chosen from Phase 7B's own
already-published blinded precision audit, not from any new human review,
AI judgment, or model performance — after the Phase 7B interactive
human-review workflow (225 blind pairs) was found incomplete (0 decisions
recorded) and explicitly not to be relied upon or fabricated.

**Why this matters for the thesis:** every existing model checkpoint in
this repository was trained on the original, duplicate-contaminated,
leakage-gate-overridden split (no exceptions) — a fact this programme's
own tooling (`dataset_gate.py`) now enforces cannot recur silently. T0
produces the first split in this project's history that (a) is
group-disjoint under a real, evidence-audited near-duplicate definition,
(b) excludes every cross-label conflict group from supervised use, (c)
excludes the 4 already-disclosed non-blind smoke-test images from
evaluation, and (d) passes a fully automated, code-enforced integrity gate
— all without waiting on an unfinished manual review.

**Method (1 sentence):** `main.py build-partition --hash-field dhash
--near-dup-threshold 2 --seed 42`, applied to the real, already-hashed
3,688-image manifest, using `src/doar/partition.py`'s existing (Phase 7A/7B)
group-disjoint largest-remainder stratified-partition algorithm, unmodified.

**Result (numbers):** Included 3,395/3,688 (92.06%); Train 2,599 / Valid
284 / Test 512; 293 images (7.94%) excluded as unresolved cross-label
duplicate conflicts; 0 duplicate groups cross a split; 0 previously-exposed
images in valid/test; largest duplicate component = 9 images (no
chaining). Full table: `T0_AUTOMATED_CONSERVATIVE_PARTITION.md` §4.

**Threshold justification (1 sentence):** dHash ≤2 and ≤3 are
statistically indistinguishable in precision (both 100%, n=6, Wilson 95%
CI 61.0–100.0%) and structurally near-identical (largest component 9 for
both); ≤2 was chosen as the more conservative of two equally well-evidenced
options. Full sensitivity comparison: `T0_THRESHOLD_SENSITIVITY.md`.

**Limitation to disclose, always:** near-duplicate groups were
algorithmically defined (dHash ≤2, transitive closure), not
human-adjudicated — no person has visually confirmed any of the 2,103
near-dup edges. The precision evidence justifying the threshold is itself
AI-preliminary judgment from an earlier blinded audit, not independent
human ground truth. Path/file-existence on the machine that will actually
run E1/E2 training has not been re-verified in this session (the raw
dataset is not accessible in this environment).

**Reproducibility statement:** Deterministic — re-running the exact same
`build-partition` command in a separate process produced a byte-identical
manifest (same SHA-256) this session. Manifest SHA-256:
`4631ce8bddde64755ba44758827310703f1b332bcb28f72b55f22b3b19b92c9d`.
Freeze/status record: `outputs/t0_automated/T0_AUTOMATED_CONSERVATIVE_V1.json`
(`not_human_reviewed: true`, explicit, never silently omitted).

**Relationship to Phase 7B:** independent, parallel methodology — does not
supersede, complete, or fabricate any part of the incomplete human-review
workflow. If that review is completed later, `dataset_gate.check_clean_
split_gate()` remains fully intact and may be used instead or in addition.

**Next step:** E1 representation screening, reusing this frozen manifest
unchanged.
