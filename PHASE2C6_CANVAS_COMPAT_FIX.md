# Phase 2C.6 — Canvas Compatibility Fix

**Status: fix, verified in the real environment.** Branch `feature/doar-phase2c-annotation-expansion`,
starting HEAD `d8c91cf` (verified clean). Fixes the reported runtime failure in the Phase 2C.6
graphical annotation UI; does not touch the Phase 2C.4/2C.4A frozen benchmark, does not rerun any
detector, does not process the remaining 250 expansion images, does not fine-tune anything.

## 1. Root cause

`streamlit-drawable-canvas==0.9.3` (PyPI's latest release; the upstream GitHub repository is
archived) calls, at `st_canvas()`'s Python-side execution:

```python
import streamlit.elements.image as st_image
...
background_image_url = st_image.image_to_url(background_image, width, True, "RGB", "PNG", image_id)
```

Verified this session by reading both packages' actually-installed source (not assumed from prior
knowledge): in Streamlit 1.61.1, `streamlit.elements.image` **no longer defines `image_to_url` at
all**. The function moved to `streamlit.elements.lib.image_utils.image_to_url`, **and its signature
changed**: the second positional parameter is no longer a raw `width: int` but a
`streamlit.elements.lib.layout_utils.LayoutConfig` object. This is not cosmetic — reading
`image_to_url`'s implementation confirms `layout_config.width` (used only when it is an `int`) drives
the same image-resize behavior the old raw `width` argument did, via `_ensure_image_size_and_format`.
So the break is a genuine relocation-plus-reshape, not a simple rename.

## 2. Exact fix

`src/doar/phase2c6/canvas_compat.py` (new) — a small, documented monkeypatch:

```python
def _shim_image_to_url(image, width, clamp, channels, output_format, image_id):
    from streamlit.elements.lib.image_utils import image_to_url as _real_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig
    return _real_image_to_url(image, LayoutConfig(width=width), clamp, channels, output_format, image_id)

def apply():
    import streamlit.elements.image as st_image_mod
    if not hasattr(st_image_mod, "image_to_url"):
        st_image_mod.image_to_url = _shim_image_to_url
```

`phase2c5_part_annotation_app.py` calls `canvas_compat.apply()` once, at module import time, **before**
`from streamlit_drawable_canvas import st_canvas` — the assignment is a module-attribute mutation on
an already-imported module object in memory, so it lives entirely in this repository and needs no edit
under `.venv/Lib/site-packages/`; it is fully reproducible after a fresh `pip install -e ".[phase2c6]"`.
`apply()` is a no-op if `image_to_url` is ever genuinely present again (a future Streamlit release, or
a maintained replacement component), so it cannot stomp on a real fix later.

### Options considered, in the requested order

1. **Minimal compatibility shim (chosen).** Six lines of adapter code, entirely in-repo, targets
   exactly the one relocated function. Verified working end-to-end (§6).
2. **Replace `streamlit-drawable-canvas` with a maintained component.** Not pursued: no clearly
   better-maintained, comparably lightweight alternative was identified without a deeper survey than
   this fix warranted, and the shim above fully resolves the reported failure with a much smaller,
   more auditable change. Left as a future option if the shim ever breaks again on a subsequent
   Streamlit release.
3. **Downgrade Streamlit.** Not done. Would need pinning below wherever `image_to_url` last lived at
   its old location and signature — likely Streamlit <1.4x, many releases behind this project's
   current `streamlit>=1.36` floor. That would risk breaking `phase2c_annotation_app.py` and
   `phase2c5_part_annotation_app.py`'s own other `st.*` calls, any other Streamlit-dependent tooling
   in this repo, and reintroduce whatever bugs/CVEs 1.36–1.61 fixed — a strictly larger and riskier
   change than a six-line adapter for one function. Streamlit itself was **not** downgraded.

## 3. `streamlit-drawable-canvas`: kept, not replaced

Same package, same version (`0.9.3`), same `pyproject.toml` `phase2c6` extra
(`streamlit-drawable-canvas>=0.9,<1`, unchanged). Only its documentation comment was updated to
describe the compatibility shim and point at this report.

## 4. Dependency changes

None beyond documentation. No new package added, no version bumped, no package removed.

## 5. Streamlit downgraded?

No. Still `streamlit==1.61.1`, matching the project's existing `ui` extra (`streamlit>=1.36,<2`).

## 6. Verification in the real environment

**Not stopped at an import test**, per instruction:

1. **Direct shim call**, reproducing the exact reported call signature:
   `st_image_mod.image_to_url(img, 64, True, "RGB", "PNG", "test-id")` after `canvas_compat.apply()`
   — no exception (confirmed both standalone and as a permanent test, §8).
2. **Real `streamlit run` server launch** (`python -m streamlit run phase2c5_part_annotation_app.py
   --server.headless true --server.port 8765`) — started cleanly, no startup exception in server logs.
   (Multiple pre-existing `streamlit run phase2c5...` processes were already running on this machine
   from your own earlier attempts — none of those were touched; only the server this fix launched for
   verification was stopped afterward.)
3. **Decisive check**: Streamlit's own `streamlit.testing.v1.AppTest` harness (the same tool
   `tests/test_phase2c1_app.py` already uses for `phase2c_annotation_app.py`) — executes the REAL
   script, with REAL widget interaction, in a simulated session, and captures REAL exceptions,
   without a browser:
   - Ran the app, filled in the annotator-ID text input, and forced `Status` to `"present"` — the
     ONLY code path that reaches `st_canvas(...)`, the exact previous failure site.
   - **Result: zero exceptions**, and the post-canvas caption
     (`"N box(es) shown on canvas -- N will be saved"`) rendered correctly, confirming execution
     continued past `st_canvas()` through the reconciliation logic, not merely that some earlier
     line didn't crash.

```
initial run exceptions: 0
post-annotator-id run exceptions: 0
PASS: app reached and passed the st_canvas() call with no exception
```

(`scripts/phase2c6_apptest_verify.py` — a standalone, rerunnable version of this same check against
the real private 15-image pilot data, for your own manual confirmation.)

### Exact browser steps for you to confirm visually

1. `python -m streamlit run phase2c5_part_annotation_app.py` (with the same `DOAR_PHASE2C1_IMAGES_DIR`
   / `DOAR_PHASE2C5_PROPOSALS_PATH` environment variables as before, pointed at whichever image set
   you want to review).
2. Enter an annotator ID in the sidebar.
3. Pick a target/image where `Status` is (or can be set to) **Present** — the canvas should render
   the drawing with any proposal boxes (dashed) overlaid.
4. Click-drag on empty canvas area with mode **"Draw new box"** selected → a new solid box should
   appear.
5. Switch to **"Move / resize / select / delete"** mode → click an existing box, drag it, drag a
   corner handle to resize it, and use the canvas's own toolbar (top-right of the canvas widget) to
   delete a selected box or undo the last change.
6. Confirm the caption below the canvas updates its box count as you edit.
7. Click Save → confirm no error banner appears and the success message shows.

## 7. Graphical annotation capabilities preserved

All of them, unchanged from the original Phase 2C.6 design: image background, proposal boxes shown,
draw/select/move/resize/delete, multiple boxes, normalized coordinate conversion
(`src/doar/phase2c6/canvas_helpers.py`, untouched by this fix), proposal provenance, explicit
accept/edit/reject, continuous save, resume, previous/next image, previous/next target.
**MODEL PROPOSAL != HUMAN GROUND TRUTH is unweakened** — `canvas_compat.py` only affects how the
background image gets a URL; it has no interaction with `bbox_source`, `savable_instances`, or any
provenance logic at all (verified: `canvas_compat.py` does not import or reference
`schema.PartInstance` or any provenance-related symbol).

## 8. Regression tests added

`tests/test_phase2c6_canvas_compat.py` — 8 tests, all passing:

- `CanvasCompatShimTests` (4): the real installed Streamlit genuinely lacks `image_to_url` at the old
  location before the shim runs (reproduces the bug's actual precondition, not an assumption); the
  shim accepts the exact reported 6-positional-argument call signature without raising; `apply()` is
  idempotent; `apply()` never overrides a genuinely-present `image_to_url` (future-proofing).
- `CanvasRegressionTests` (4, `AppTest`-based, isolated temp-dir images, mirrors
  `tests/test_phase2c1_app.py`'s established pattern): renders without exception before/after setting
  an annotator ID; forces `Status="present"` to reach the exact `st_canvas()` call site and asserts
  zero exceptions AND that execution continued past it (post-canvas caption present); explicitly pins
  that no exception is ever an `AttributeError` mentioning `image_to_url` again.

## 9. Full test results

- `tests/test_phase2c5_*.py` + `tests/test_phase2c6_*.py`: **159 passed** (151 prior + 8 new).
- Ruff: all checks passed.
- `compileall`: clean.
- Full repository suite: **1254 passed, 7 skipped, 0 failures** (1246 prior + 8 new, exact match), 350.0s.

## 10. Commit

See final report to user for the exact commit hash (created after full-suite verification completed).

## 11. Pushed?

No.

## 12. What to run now

```
$env:DOAR_PHASE2C1_IMAGES_DIR = "outputs/phase2c6/expansion_images_private"
$env:DOAR_PHASE2C5_PROPOSALS_PATH = "outputs/phase2c6/expansion_raw_proposals_private.csv"
python -m streamlit run phase2c5_part_annotation_app.py
```

(Unchanged from the original Phase 2C.6 report's launch instructions — the fix requires no new
environment variables or flags.)
