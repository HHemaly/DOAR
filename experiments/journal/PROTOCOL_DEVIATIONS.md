# Protocol deviations (reported in the paper)

## D1 — T1 frozen before two review groups were scored (2026-10-29)

What happened (from the executed notebook, code version e024319):
1. A5 score and A5b ran when `rf_stem` and `name_key` were scored and `dihedral` (40 pairs) and
   `embedding` (23 pairs) were still unanswered. A5b therefore recorded those two as "not scored".
2. A3 built the proposal with no extra edge files (manifest SHA-256 6868aed0…); A3 printed the
   "review still open … provisional" warning.
3. The `dihedral` and `embedding` reviews were then completed (63/63 answered).
4. T1 was frozen at 2026-10-29T15:55:44Z **without re-running A5 score, A5b and A3**.
5. B0 smoke test passed; B1 probe (9 encoders × 10 seeds) completed and B2 adaptation started
   on that T1 (interrupted during siglip2_vit_b16 partial seed 73).

Why the evidence decision is still blind to model results: all review verdicts were recorded
before the freeze and before any Stage B run. Scoring them and applying the pre-registered rule
(A5b) is mechanical and uses no model output.

Resolution rule (fixed before the two groups are scored):
- If A5b accepts neither `dihedral` nor `embedding` (nor asks for a tighter-threshold review),
  the frozen T1 is exactly the split the protocol would have produced; Stage B resumes and this
  deviation is reported as a sequencing error with no effect on the split.
- Otherwise the frozen T1 is superseded: a new proposal is built with the accepted edges, checked,
  approved and frozen as a new version, and **all** Stage B runs are repeated on it. Results from
  the superseded T1 are not reported as main results.

Safeguard added: the freeze, smoke-test and B1–B2 cells now refuse to run while any review group
is unanswered or unscored, while A5b is older than the latest scoring, or (B1–B2) while the frozen
T1 used different extra evidence than A5b now accepts.

### D1 outcome (2026-10-31, A5 score + A5b on the verdicts recorded before the freeze)
- `embedding`: 23 of 23 reviewed new pairs judged the same drawing → **accepted** (rule ≥ 90%).
- `dihedral` (dHash ≤ 6): 15 of 40 judged same (38%) → not accepted; the pre-registered tighter
  threshold (dHash ≤ 4) was written and must pass its own blinded review (`dihedral_tight`).
- `name_key`: 0 of 16 → not accepted.
- Consequence under the rule above: the T1 frozen on 2026-10-29 (manifest 6868aed0…) is
  **superseded**. The new proposal with embedding edges (manifest 5c2fc755…, all four balance
  checks pass) can be frozen only after the `dihedral_tight` review is scored. All Stage B runs
  on the superseded T1 (`E1_T1/`) are kept for this report and are not used as results;
  Stage B is repeated on the new version (`E1_T1_v2/`).
- 2026-10-31 final A5b: `dihedral_tight` (dHash ≤ 4) 23 of 40 judged same (57%, 95% CI 42–71%)
  → also < 90%: dihedral evidence is **not used** and is reported as a limitation. At the tighter
  threshold A5c counted 334 candidate links not explained by the accepted evidence, 87 of them
  across the proposed splits (47 touching test); given the 57% review rate, some of these are
  probably real duplicates that remain undetected (residual leakage, to be stated in the paper).
- Stage A complete. Proposal 5c2fc755… passes all four balance checks; to be frozen as T1_v2,
  superseding T1 (6868aed0…).

## D2 — DINOv2 adaptation added to the archived E1-B code (2026-11-01)

On T1_v2 the pre-registered shortlist rule (top-3 10-seed probe means) selected
`siglip2_vit_b16`, `clip_vit_b32`, `dinov2_vits14`. The archived E1-B code supports only the T0
shortlist (SigLIP2, CLIP, EfficientNet-B0) and stopped with `ValueError: dinov2_vits14` at
`dinov2_vits14__frozen__seed_19`, after 90/90 probe runs and 18/27 adaptation runs had completed.

Fix: `src/doar_journal/e1b_dinov2.py` adds DINOv2 without editing the archived file. It reuses
the probe's backbone call (`torch.hub facebookresearch/dinov2 dinov2_vits14`), CLS-token features
and eval preprocessing, the archived augmentation-insertion rule, and the archived
OpenCLIP-tower trainability rule (partial = last blocks until ≥ 25% of backbone parameters +
final norm). Optimiser, learning rates, epochs, patience, batch size, loss, early stopping and
outputs are the archived code path. Nothing about the shortlist, seeds, T1_v2 or the test
safeguards changed. Completed runs are reused only after the read-only audit (`b1c_audit_runs.py`)
confirms they belong to T1_v2. To report: the DINOv2 adaptation implementation was written after
the shortlist was known (it could not have been written for a model not shortlisted on T0), and
its partial mode trains ~32% of the backbone (CLIP 32.7%, SigLIP2 30.5%).
