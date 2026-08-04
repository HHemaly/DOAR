# Model Experiment Run Guide (Windows PowerShell)

**Read `EXPERIMENT_PROTOCOL.md` §0 first.** The dataset gate is currently
**closed** — every "full" command below will refuse to run
(`CleanSplitGateFailed`) until it passes. `--smoke` commands always work
and are safe to run right now; they use synthetic or tiny non-test-split
data and never produce a reportable result.

All commands assume `cd` to the repository root and the venv activated:
```powershell
.\.venv\Scripts\Activate.ps1
```

---

## 1. Check the gate

```powershell
python main.py run-stage0 --manifest outputs\phase5\manifest.csv --output outputs\_gate_check --gate-check-only
```
Prints the full `check_clean_split_gate()` report, including the exact
remediation steps, without running or creating anything else. (Deletes
nothing — `outputs\_gate_check` is created empty and can be removed.)

## 2. Smoke test (safe right now, always permitted)

```powershell
python main.py run-stage0 --manifest outputs\phase5\manifest.csv --output outputs\stage0_smoke --smoke
```
Runs all 6 stages against synthetic/tiny data (Stage A/B use whatever
manifest you pass, filtered to `train`/`valid` rows only, capped
internally; Stage C uses a fully synthetic embeddings cache — no network
call; Stages D/E build their own tiny synthetic `ImageFolder` and train 1
CPU epoch each; Stage F will correctly *fail* with a provenance error in
smoke mode — this is expected, see `EXPERIMENT_MATRIX.md` Experiment F).
Takes well under a minute. Inspect
`outputs\stage0_smoke\stage0_run_manifest.json` afterward.

To smoke-test just one stage:
```powershell
python main.py run-stage0 --manifest outputs\phase5\manifest.csv --output outputs\stage0_smoke --smoke --only A_objective_features
```
Valid `--only` values: `A_objective_features`, `B_handcrafted_groups`,
`C_dinov2_embeddings`, `D_densenet121`, `E_convnext_tiny_probe`, `F_fusion`
(comma-separated for more than one).

## 3. Run one experiment directly (also gated where applicable)

```powershell
# Experiment A — objective-feature classical baseline (needs the frozen manifest)
python main.py extract-features --manifest <frozen_manifest.csv> --output outputs\features\v3_1
python main.py train-feature-model --features outputs\features\v3_1\features.csv --output outputs\experiments\objective_features --models logistic_regression,linear_svm,rbf_svm,random_forest,extra_trees,hist_gradient_boosting --seeds 42,123,2026

# Experiment B — HOG/colour/geometry (needs Experiment A's features.csv first)
python main.py extract-hog-features --manifest <frozen_manifest.csv> --output outputs\hog_features\v1
python main.py train-handcrafted-groups --objective-features outputs\features\v3_1\features.csv --hog-features outputs\hog_features\v1\hog_features.csv --output outputs\experiments\handcrafted_groups --seeds 42,123,2026

# Experiment C — DINOv2 frozen embeddings (needs network access the FIRST time; caches after)
python main.py extract-embeddings --manifest <frozen_manifest.csv> --output outputs\embeddings\dinov2_vits14 --backbone dinov2_vits14
python main.py train-embedding-classifier --embeddings outputs\embeddings\dinov2_vits14\embeddings.npz --output outputs\experiments\dinov2_classifier --seeds 42,123,2026

# Experiment D — DenseNet121 (config-driven; edit configs\training\densenet121.toml's [data].dataset path first)
python main.py train-image-model --config configs\training\densenet121.toml

# Experiment E — ConvNeXt-Tiny FEASIBILITY PROBE ONLY (single seed, restricted epochs)
python main.py train-image-model --config configs\training\convnext_tiny_probe.toml

# Experiment F — fusion (needs A's features.csv and C's embeddings.npz to share the same manifest)
python main.py train-fusion-model --config configs\training\primary_fusion_dinov2.toml
```

`extract-features`/`extract-hog-features` themselves do not check the
gate (they are read-only feature extraction, not training or a
reportable result), but every classical/deep/fusion trainer command
above should only be run against the frozen manifest once the gate
passes — running them against `outputs\phase5\manifest.csv` produces a
`preliminary_contaminated_split`-labeled result only (see
`EXPERIMENT_PROTOCOL.md` §4), never a Stage 0 finding.

## 4. Run the full Stage 0 matrix (BLOCKED until the gate passes)

```powershell
python main.py run-stage0 --manifest <frozen_manifest.csv> --output outputs\stage0
```
Raises `CleanSplitGateFailed` and prints the remediation list if the gate
has not passed. Once it has, this runs all 6 stages in dependency order
(F waits for A and C) with the real Stage 0 protocol (1 seed, restricted
budget — see `EXPERIMENT_PROTOCOL.md` §6). Add `--dinov2-backbone <name>`
to pin a different DINOv2 variant.

## 5. Resume an interrupted run

```powershell
python main.py run-stage0 --manifest <frozen_manifest.csv> --output outputs\stage0
```
Re-running the exact same command is the resume mechanism — completed
stages (a `STAGE_COMPLETE.json` marker exists) are skipped automatically.
To force a full re-run of every stage instead:
```powershell
python main.py run-stage0 --manifest <frozen_manifest.csv> --output outputs\stage0 --no-resume
```
For a single deep-training run interrupted mid-training (not via
`run-stage0`), `train_image_model`'s own `resume=<path to last.pt>`
parameter is real and tested
(`tests/test_trainer_regression.py`) — pass `--resume
outputs\experiments\densenet121_seed42\last.pt` (not yet exposed as a
`train-image-model` CLI flag; add `--resume` to `main.py`'s
`train-image-model` parser if you need this from the command line before
it's wired — currently only reachable by calling
`doar.deep.trainers.train_image_model(..., resume=...)` directly).

## 6. Produce comparison reports (only after real Stage 0 results exist)

```powershell
python main.py generate-thesis-outputs --output outputs\stage0
```
Auto-discovers every leaderboard JSON under `outputs\stage0` (including
this session's 3 new types: `validation_leaderboard.json`,
`handcrafted_group_leaderboard.json`, `embedding_classifier_leaderboard.json`)
and writes matching figures + a `thesis_manifest.json` under
`outputs\stage0\thesis\` — never fabricates a figure for a missing source.

For the unified cross-family table (**always pass an explicit
`data_provenance` label — the function raises otherwise**):
```powershell
python -c "import sys; sys.path.insert(0,'src'); from doar.thesis import build_cross_family_model_comparison; import json; print(json.dumps(build_cross_family_model_comparison('outputs/stage0', data_provenance='clean_development_split', output_path='outputs/stage0/cross_family_comparison.json'), indent=2))"
```
Valid `data_provenance` values: `preliminary_contaminated_split`,
`clean_development_split`, `locked_test`, `smoke_test_not_a_result` — pick
the one that actually describes the data you ran, never mix two labels'
worth of results in one table (`EXPERIMENT_PROTOCOL.md` §4).

## 7. Device selection

Every command above auto-detects CUDA (`device="auto"` in every deep
config) and reports the selected device in
`outputs\...\environment.json`. This machine currently has CUDA available
(Quadro P3200, confirmed live this session). To force CPU regardless
(e.g. for a fast sanity check), edit the relevant `.toml`'s
`[training] device = "cpu"`, or use `--smoke` (§2, always CPU-only).

## 8. Tests

```powershell
python -m pytest tests\ -q                          # full suite (410 tests as of this session)
python -m pytest tests\test_dataset_gate.py tests\test_hog_features.py tests\test_handcrafted_comparison.py tests\test_embedding_classifier.py tests\test_stage0_runner.py tests\test_densenet121.py tests\test_thesis_cross_family.py tests\test_new_experiment_cli.py -q   # this session's new tests only
python -m compileall src main.py streamlit_app.py phase7b_review_app.py doar_prototype_app.py
python -m ruff check src tests main.py streamlit_app.py phase7b_review_app.py doar_prototype_app.py
```
