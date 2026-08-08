"""Phase 2C.6 compatibility shim: `streamlit-drawable-canvas` 0.9.3
(PyPI's latest release, last published ~2022; upstream GitHub repo is
archived) against Streamlit 1.61.1.

Root cause (verified by reading both packages' installed source this
session, not assumed): `streamlit_drawable_canvas/__init__.py` calls

    st_image.image_to_url(background_image, width, True, "RGB", "PNG", image_id)

where `st_image` is `streamlit.elements.image`, looked up as a module
ATTRIBUTE at call time (not imported by name), and `width` is a raw
`int`. In Streamlit 1.61.1, `streamlit.elements.image` no longer defines
`image_to_url` at all -- the function moved to
`streamlit.elements.lib.image_utils.image_to_url`, and its second
positional parameter changed from a raw `width: int` to a
`streamlit.elements.lib.layout_utils.LayoutConfig` object. This is not a
cosmetic rename: `image_to_url`'s internal `_ensure_image_size_and_format`
helper reads `layout_config.width` (only using it when it is an `int`) to
resize the image before handing it to the MediaFileManager -- the exact
same role the old raw `width` argument played. Wrapping the same integer
in `LayoutConfig(width=width)` before calling the relocated function
reproduces that behavior exactly; this is the same code path modern
`st.image()` itself uses internally, not a reverse-engineered guess.

`apply()` monkeypatches `streamlit.elements.image.image_to_url` to a thin
adapter closing over the relocated function -- a module-attribute
assignment on an already-imported module object in memory, never a file
under `.venv/Lib/site-packages/`. It is therefore fully reproducible after
a fresh `pip install -e ".[phase2c6]"`: this file ships in the repo, and
the app calls `apply()` once at import time, before `st_canvas(...)` is
ever invoked (the actual call site streamlit-drawable-canvas resolves the
attribute against). Only patches when the attribute is genuinely missing
-- a future Streamlit release that restores `image_to_url` at its old
location, or a maintained canvas-component replacement, makes this a
no-op rather than a stale double-patch.

Why a shim over the alternatives (see PHASE2C6_CANVAS_COMPAT_FIX.md
report for the full evaluation): the archived package's OWN publicly
documented replacements (a maintained fork, or a different annotation
component) were not adopted without your review per instruction; pinning
Streamlit down to satisfy a single archived component would revert
`server.baseUrlPath`, dataframe, and other API surfaces the rest of this
project's tooling (and this same file's own `st.image` calls elsewhere)
already depend on at 1.61 -- a strictly larger, less targeted change than
a six-line adapter for one relocated internal function.
"""
from __future__ import annotations

from typing import Any


def _shim_image_to_url(image: Any, width: int, clamp: bool, channels: str,
                        output_format: str, image_id: str) -> str:
    from streamlit.elements.lib.image_utils import image_to_url as _real_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig

    return _real_image_to_url(image, LayoutConfig(width=width), clamp, channels, output_format, image_id)


def apply() -> None:
    """Idempotent -- safe to call on every Streamlit script rerun (the
    whole module re-executes on every interaction)."""
    import streamlit.elements.image as st_image_mod

    if not hasattr(st_image_mod, "image_to_url"):
        st_image_mod.image_to_url = _shim_image_to_url
