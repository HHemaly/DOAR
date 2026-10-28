"""Build colab/journal/DOAR_Journal_Validation_2026.ipynb from this file.

The notebook holds no experiment logic: every cell configures paths and calls
a version-controlled script under experiments/journal/scripts/. Edit THIS file
and re-run it; never hand-edit the generated notebook.

  python experiments/journal/tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "colab/journal/DOAR_Journal_Validation_2026.ipynb"
OUT_STANDALONE = ROOT / "colab/journal/DOAR_Journal_Validation_2026_STANDALONE.ipynb"

# Files embedded in the standalone notebook (everything the journal scripts import).
BUNDLE_GLOBS = ("src/doar/__init__.py", "src/doar/partition.py", "src/doar/dataset.py", "src/doar/deep/*.py",
                "src/doar_journal/*.py", "experiments/journal/scripts/*.py",
                "experiments/E1_visual_representation/scripts/e1a2_common.py")


def make_bundle() -> tuple[str, list[str], str]:
    import base64
    import hashlib
    import io
    import subprocess
    import zipfile
    files = sorted({str(p.relative_to(ROOT)) for g in BUNDLE_GLOBS for p in ROOT.glob(g)})
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in files:
            info = zipfile.ZipInfo(f, date_time=(2026, 1, 1, 0, 0, 0))  # fixed time: reproducible bytes
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, (ROOT / f).read_bytes())
    data = buf.getvalue()
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True).strip()
    return base64.b64encode(data).decode(), files, f"{commit}/{hashlib.sha256(data).hexdigest()[:12]}"

CELLS: list[tuple[str, str]] = []


def md(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


md("""
# DOAR — Journal validation (Machine Vision and Applications), 2026

This notebook **runs** the journal experiments; all logic lives in the GitHub
branch `journal/mva-2026` (`src/doar_journal/`, `experiments/journal/scripts/`).
The original notebook `DOAR_21_8_26_V1.ipynb` is left untouched.

| Stage | Runtime | Needs you? |
|---|---|---|
| 0 Setup | CPU | — |
| A Dataset integrity (A1, A2, A6, A4, A5) | CPU (A4 embeddings: GPU optional) | **Yes**: blinded pair review (A5) |
| A3 Propose / freeze T1 split | CPU | **Yes**: approve the split before freezing |
| B Representation rerun on T1 | **GPU (A100 recommended)** | — |
| B3 Freeze selection | CPU | — |
| B4 Single test evaluation | GPU | **Yes**: explicit confirmation, runs once |
| C Foreground validation | CPU | **Yes**: holdout review + reference masks |

**Runtime:** use a CPU (or T4) runtime for Stages A and C — they do not need an A100, which
uses compute units much faster. Switch to an A100 only for Stage B and B4.

Typical durations (CPU runtime, Drive-mounted dataset): A1 < 1 min · A2 2–5 min ·
A6 < 1 min · A4 20–40 min (first read of ~3,700 images from Drive; GPU makes the
embedding part faster) · A5 make 2–5 min · C2 20–40 min. Every script prints progress
lines; if a cell shows nothing for more than ~5 minutes, check `DOAR_JOURNAL_2026/logs/`.

Every stage writes to `DOAR_JOURNAL_2026/` on your Drive, writes `DONE.json`
when finished, and is skipped on re-run. After a disconnect just run the
setup cells again and continue from the stage you were at.
""")

code("""
# 0.1 Mount Drive and set paths (edit only this cell if your folders differ)
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
MASTERS       = Path('/content/drive/MyDrive/Masters')
DATASET_ROOT  = MASTERS / 'Datasets/Combined_Drawing'
T0_MANIFEST   = MASTERS / 'Datasets/DOAR_FROZEN_TO/final_partition/partition_manifest.csv'
E1_HISTORY    = MASTERS / 'DOAR_GPU_RESULTS/E1'             # archived T0 runs (read-only)
E2_HISTORY    = MASTERS / 'DOAR_GPU_RESULTS/E2/E2A_INTERPRETABLE_FEATURES'
OUT           = MASTERS / 'DOAR_JOURNAL_2026'               # everything new goes here
REPO          = Path('/content/DOAR')
BRANCH        = 'journal/mva-2026'
OUT.mkdir(parents=True, exist_ok=True)
for p in (DATASET_ROOT, T0_MANIFEST, E1_HISTORY, E2_HISTORY):
    print(('OK      ' if p.exists() else 'MISSING ') + str(p))
""")

SETUP_CELL_INDEX = len(CELLS)
code("__SETUP__")

md("## Stage A — dataset integrity (CPU)")

code("""
# A1 Filename-level source-stem audit on the FULL T0 manifest (incl. excluded conflicts)
run('a1_stem_audit.py', '--manifest', T0_MANIFEST, '--out', OUT / 'A1_stem_audit_T0_full')
""")

code("""
# A2 (X1) Re-score SAVED validation predictions: shared-source vs independent images. No training.
run('a2_rescore_saved_predictions.py', '--manifest', T0_MANIFEST, '--runs-root', E1_HISTORY,
    '--out', OUT / 'A2_rescoring')
""")

code("""
# A6 Do the 80 'unseen' foreground-holdout images share a source with the 84 development images?
run('a6_dev_holdout_overlap.py', '--manifest', T0_MANIFEST,
    '--dev', E2_HISTORY / 'E2A0_R1_PILOT_V1/tables/E2A0_R1_PILOT_MASK_RESULTS.csv',
    '--holdout', E2_HISTORY / 'E2A0_R1_V2_UNSEEN_HOLDOUT/tables/E2A0_R1_V2_UNSEEN_HOLDOUT_MANIFEST.csv',
    '--out', OUT / 'A6_dev_holdout_overlap')
""")

code("""
# A4 Image-level near-duplicate evidence (rotation/flip-invariant dHash; CLIP embeddings).
# Label-blind. Uses the GPU for embeddings if present; add '--skip-embedding' on a CPU runtime.
args = ['--t0-manifest', T0_MANIFEST, '--dataset-root', DATASET_ROOT, '--out', OUT / 'A4_image_evidence']
if not torch.cuda.is_available():
    args += ['--device', 'cpu']
run('a4_image_evidence.py', *args)
""")

code("""
# A5 Blinded pair-review materials (train-valid pairs only; the locked test split is never shown)
edges = [OUT / 'A4_image_evidence' / f for f in ('dihedral_edges_candidates.csv', 'embedding_edges_candidates.csv')]
run('a5_pair_review_materials.py', 'make', '--manifest', T0_MANIFEST, '--dataset-root', DATASET_ROOT,
    '--out', OUT / 'A5_pair_review', *sum([['--edges', e] for e in edges if e.exists()], []))
""")

md("""
### ✋ Your task: blinded pair review (click buttons below — no files to edit)
Put your initials in `REVIEWER`, choose a group in `STRATUM`, run the cell. For each pair click
**Same Drawing**, **Different Drawing** or **Unsure** (if same, first pick what changed, e.g. rotation).
Every click is saved to Drive at once; you can stop any time and re-run the cell to continue.
Do the groups one by one: `rf_stem`, `name_key`, then `dihedral` and `embedding` (these two exist
only after A4 finished and the A5 make cell was run again). Then run the score cell.
""")

code("""
REVIEWER = ''          # your initials, e.g. 'HE'
STRATUM  = 'rf_stem'   # then 'name_key', 'dihedral', 'embedding'
from doar_journal.review_widgets import pair_review
pair_review(OUT / 'A5_pair_review', DATASET_ROOT, STRATUM, REVIEWER)
""")

code("""
run('a5_pair_review_materials.py', 'score', '--out', OUT / 'A5_pair_review')
""")

code("""
# A3 Propose T1 (both strategies). Image-evidence edges are added only for evidence types
# that passed review (PREREGISTRATION.md, C1b): list the accepted files here.
ACCEPTED_EDGE_FILES = []   # e.g. [OUT/'A4_image_evidence'/'dihedral_edges_candidates.csv']
for strategy in ('minimal_change', 'regroup_full'):
    run('a3_build_t1.py', 'propose', '--t0-manifest', T0_MANIFEST, '--strategy', strategy,
        '--out', OUT / f'T1_proposal_{strategy}', *sum([['--extra-edges', e] for e in ACCEPTED_EDGE_FILES], []))
""")

md("""
### ✋ Your decision: approve one T1 proposal
Compare `T1_proposal_*/T1_STATUS.json` (split sizes, class counts, T0→T1 moves).
`minimal_change` keeps most images where they were (best for a paired T0-vs-T1 comparison;
validation/test shrink). `regroup_full` re-draws all splits at T0's proportions.
Copy the approved folder to `DOAR_JOURNAL_2026/T1`, put your name below and run the cell.
""")

code("""
APPROVED_BY = ''          # your full name; freezing is refused if empty
APPROVED_FOLDER = OUT / 'T1'
assert APPROVED_BY, 'type your name to approve the split'
run('a3_build_t1.py', 'freeze', '--out', APPROVED_FOLDER, '--approved-by', APPROVED_BY)
""")

md("## Stage B — representation experiments on T1 (GPU)")

code("""
# B1-B2 probe (9 encoders x 10 seeds) -> adaptation (shortlist x 3 modes x 3 seeds) -> confirm (finalists x 2 seeds)
# Resumable: re-run this cell after a disconnect; finished runs are skipped.
assert torch.cuda.is_available(), 'switch the runtime to GPU'
run('b1_b2_e1_rerun.py', '--t1-dir', OUT / 'T1', '--dataset-root', DATASET_ROOT, '--out', OUT / 'E1_T1', '--stage', 'all')
""")

code("""
# B3 Tables, T0-vs-T1 comparison and the frozen selection (CPU is fine)
run('b3_e1_analysis.py', '--t1-out', OUT / 'E1_T1', '--t0-root', E1_HISTORY, '--out', OUT / 'B3_E1_analysis',
    '--freeze-selection')
""")

md("""
### ✋ Single test evaluation
Run the next cell **only** when every analysis decision is final. It can run once; a lock file
prevents a second run. Set `CONFIRM = True` to proceed.
""")

code("""
CONFIRM = False
assert CONFIRM, 'set CONFIRM = True only when everything else is frozen'
run('b4_single_test_evaluation.py', '--t1-dir', OUT / 'T1',
    '--selection', OUT / 'B3_E1_analysis' / 'SELECTION_FROZEN.json',
    '--probe-root', OUT / 'E1_T1' / 'probe', '--dataset-root', DATASET_ROOT,
    '--out', OUT / 'B4_TEST', '--confirm-single-test-run')
""")

md("""
## Stage C — foreground and measurement validation (CPU)

### ✋ Your task 1: finish the existing holdout review (click buttons below)
For each of the 80 holdout images you see the original, the A0 mask and the frozen V2 mask
(white = drawing marks). Judge the **V2** mask with one button; optionally say whether V2's
support regime is right. Every click is saved at once; re-run to continue.
""")

code("""
from doar_journal.review_widgets import holdout_review
holdout_review(E2_HISTORY / 'E2A0_R1_V2_UNSEEN_HOLDOUT/tables/E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv', DATASET_ROOT)
""")

code("""
holdout = E2_HISTORY / 'E2A0_R1_V2_UNSEEN_HOLDOUT/tables'
run('c1_holdout_review_scores.py', '--review', holdout / 'E2A0_R1_V2_HOLDOUT_VISUAL_REVIEW.csv',
    '--results', holdout / 'E2A0_R1_V2_UNSEEN_HOLDOUT_RESULTS.csv',
    '--exclude-overlap', OUT / 'A6_dev_holdout_overlap' / 'holdout_source_overlap.csv',
    '--out', OUT / 'C1_holdout_review')
""")

code("""
# C2 Choose and package the blind reference-mask batch (~120 images, 40 double-annotated)
run('c2_reference_batch.py', '--manifest', T0_MANIFEST, '--dataset-root', DATASET_ROOT,
    '--exclude', E2_HISTORY / 'E2A0_R1_PILOT_V1/tables/E2A0_R1_PILOT_MASK_RESULTS.csv',
    '--exclude', E2_HISTORY / 'E2A0_R1_V2_UNSEEN_HOLDOUT/tables/E2A0_R1_V2_UNSEEN_HOLDOUT_MANIFEST.csv',
    '--out', OUT / 'C2_reference_batch', '--n', 120, '--n-double', 40)
""")

md("""
### ✋ Your task 2: reference masks
Download `DOAR_JOURNAL_2026/C2_reference_batch/` **without** `key_PRIVATE.csv`, annotate with
`experiments/journal/annotation/reference_mask_app.py` following `ANNOTATION_PROTOCOL.md`,
then upload `masks/` and `records/` back into the same Drive folder.
Set the annotator initials below.
""")

code("""
PRIMARY, SECONDARY = 'HE', ''   # initials used in the app
args = ['--batch', OUT / 'C2_reference_batch', '--primary', PRIMARY, '--out', OUT / 'C3_reference_evaluation']
if SECONDARY:
    args += ['--secondary', SECONDARY]
run('c3_evaluate_references.py', *args)
run('c4_measurement_analysis.py', '--c3', OUT / 'C3_reference_evaluation', '--out', OUT / 'C4_measurement_analysis')
""")

md("## Export results for the manuscript")

code("""
# Zip every table, JSON and figure (no images, no checkpoints) for return / the manuscript.
import zipfile, datetime
stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M')
zpath = OUT / f'journal_results_{stamp}.zip'
skip = ('.pt', '.npz', '.png' , '.jpg', '.jpeg')
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file() and 'C2_reference_batch/images' not in str(p) and not p.name.endswith(skip) and p != zpath:
            z.write(p, p.relative_to(OUT))
    for p in OUT.rglob('F_*.png'):
        z.write(p, p.relative_to(OUT))
print('wrote', zpath, round(zpath.stat().st_size / 1e6, 2), 'MB')
""")


SETUP_GIT = '# 0.2 Get the code at the journal branch and install what Colab lacks\nimport subprocess, sys\nif not REPO.exists():\n    subprocess.run([\'git\', \'clone\', \'--branch\', BRANCH, \'https://github.com/HHemaly/DOAR.git\', str(REPO)], check=True)\nelse:\n    subprocess.run([\'git\', \'-C\', str(REPO), \'fetch\', \'origin\', BRANCH], check=True)\n    subprocess.run([\'git\', \'-C\', str(REPO), \'checkout\', BRANCH], check=True)\n    subprocess.run([\'git\', \'-C\', str(REPO), \'pull\', \'--ff-only\'], check=True)\nCOMMIT = subprocess.check_output([\'git\', \'-C\', str(REPO), \'rev-parse\', \'HEAD\'], text=True).strip()\nprint(\'code commit:\', COMMIT)\nsubprocess.run([sys.executable, \'-m\', \'pip\', \'install\', \'-q\', \'open_clip_torch\', \'timm\',\n                \'scikit-image\', \'opencv-python-headless\'], check=True)\nsys.path.insert(0, str(REPO / \'src\'))\n\nimport datetime, time\nLOGS = OUT / \'logs\'\nLOGS.mkdir(exist_ok=True)\n\ndef run(script, *args):\n    """Run a journal script and show its output LIVE in this cell (Colab does not\n    display a child process\'s output unless it is piped through Python), also\n    saving it to DOAR_JOURNAL_2026/logs/. Stops the notebook on failure."""\n    cmd = [sys.executable, \'-u\', str(REPO / \'experiments/journal/scripts\' / script), *map(str, args)]\n    log = LOGS / f"{script.removesuffix(\'.py\')}_{datetime.datetime.now():%Y%m%dT%H%M%S}.log"\n    print(\'$\', \' \'.join(cmd), \'\\nlog:\', log, flush=True)\n    t0 = time.time()\n    with open(log, \'w\', encoding=\'utf-8\') as fh:\n        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n        for line in p.stdout:\n            print(line, end=\'\', flush=True)\n            fh.write(line)\n        p.wait()\n    print(f\'-- {script} finished in {(time.time() - t0) / 60:.1f} min, exit code {p.returncode}\')\n    if p.returncode != 0:\n        raise RuntimeError(f\'{script} failed with exit code {p.returncode}; see {log}\')\n\nimport torch\nprint(\'GPU:\', torch.cuda.get_device_name(0) if torch.cuda.is_available() else \'none (CPU runtime)\')\n'


def _standalone_setup(bundle_id: str) -> str:
    head = SETUP_GIT.split("if not REPO.exists():")[0]
    tail = SETUP_GIT.split("print('code commit:', COMMIT)")[1]
    body = (
        "# All code is embedded in this notebook (cell 0.2a); nothing is downloaded from GitHub.\n"
        "import base64, io, shutil, zipfile\n"
        "if REPO.exists():\n"
        "    shutil.rmtree(REPO)\n"
        "zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE_BUNDLE))).extractall(REPO)\n"
        f"COMMIT = '{bundle_id}'\n"
        "print('embedded code version:', COMMIT)\n")
    return head.replace("# 0.2 Get the code at the journal branch", "# 0.2 Unpack the embedded code") + body + tail


def _write(path: Path, cells: list[tuple[str, str]]) -> None:
    nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "name": path.name},
                                    "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    for kind, text in cells:
        cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
            compile(text, "<cell>", "exec")
        nb["cells"].append(cell)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path} ({len(cells)} cells, {path.stat().st_size / 1e3:.0f} kB)")


def main() -> int:
    git_cells = list(CELLS)
    git_cells[SETUP_CELL_INDEX] = ("code", SETUP_GIT)
    _write(OUT, git_cells)

    b64, files, bundle_id = make_bundle()
    sa = list(CELLS)
    sa[SETUP_CELL_INDEX] = ("code", _standalone_setup(bundle_id))
    sa.insert(SETUP_CELL_INDEX, ("code",
              f"# 0.2a Embedded code ({len(files)} files, version {bundle_id}). Do not edit; just run.\n"
              f"CODE_BUNDLE = '{b64}'"))
    sa[0] = ("markdown", sa[0][1].replace(
        "This notebook **runs** the journal experiments; all logic lives in the GitHub",
        "**Standalone version:** all code is embedded in cell 0.2a, so this notebook needs only your "
        "Google Drive (dataset and results), not GitHub. It is a snapshot of the GitHub"))
    _write(OUT_STANDALONE, sa)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
