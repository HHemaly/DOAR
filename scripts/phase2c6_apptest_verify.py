#!/usr/bin/env python
"""Phase 2C.6 canvas-compat fix verification: runs the REAL
phase2c5_part_annotation_app.py script inside Streamlit's own
`streamlit.testing.v1.AppTest` harness -- executes actual Python code in
a simulated Streamlit session (real widget tree, real exception capture)
without a browser. This is the decisive check for the
`image_to_url` AttributeError: if the shim doesn't work, `at.exception`
below is non-empty and this script exits non-zero.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["DOAR_PHASE2C1_IMAGES_DIR"] = str(ROOT / "outputs/phase2c6/expansion_images_private")
os.environ["DOAR_PHASE2C5_PROPOSALS_PATH"] = str(ROOT / "outputs/phase2c6/expansion_raw_proposals_private.csv")

from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(ROOT / "phase2c5_part_annotation_app.py"), default_timeout=60)
at.run()

print(f"initial run exceptions: {len(at.exception)}")
for exc in at.exception:
    print(f"  {exc.value!r}")
if at.exception:
    sys.exit(1)

# The app blocks on an empty annotator_id text_input until one is provided
# -- fill it in and rerun, which is the path that actually reaches the
# st_canvas(...) call (the exact failure point in the bug report).
text_inputs = at.text_input
annotator_box = next((w for w in text_inputs if w.label == "Your annotator ID"), None)
if annotator_box is None:
    print("FAIL: could not find the annotator-ID text_input widget")
    sys.exit(1)
annotator_box.set_value("apptest_verify_agent").run()

print(f"post-annotator-id run exceptions: {len(at.exception)}")
for exc in at.exception:
    print(f"  {type(exc.value).__name__}: {exc.value}")
if at.exception:
    sys.exit(1)

print("PASS: app reached and passed the st_canvas() call with no exception")
