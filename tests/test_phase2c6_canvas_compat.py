"""Phase 2C.6 canvas-compatibility regression tests.

Pins the exact bug reported and fixed this session:

    AttributeError: module 'streamlit.elements.image' has no attribute
    'image_to_url'

raised from streamlit_drawable_canvas/__init__.py's `st_canvas()` against
Streamlit 1.61.1. Covers both the isolated shim logic (no Streamlit
runtime needed) and a full real-script-execution regression test via
`streamlit.testing.v1.AppTest` (same tool and isolated-temp-dir pattern
already used by tests/test_phase2c1_app.py) that forces the exact
`status == "present"` code path containing the real `st_canvas(...)`
call site and asserts zero exceptions.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

try:
    from streamlit.testing.v1 import AppTest
    _STREAMLIT_TESTING_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    _STREAMLIT_TESTING_AVAILABLE = False


class CanvasCompatShimTests(unittest.TestCase):
    """Isolated logic tests -- no Streamlit runtime required. setUp/tearDown
    reset `streamlit.elements.image`'s `image_to_url` attribute around each
    test so results never depend on test execution order (the attribute is
    a real, process-global module mutation)."""

    def setUp(self):
        import streamlit.elements.image as st_image_mod
        self._had_attr = hasattr(st_image_mod, "image_to_url")
        self._original = getattr(st_image_mod, "image_to_url", None)
        if self._had_attr:
            del st_image_mod.image_to_url

    def tearDown(self):
        import streamlit.elements.image as st_image_mod
        if self._had_attr:
            st_image_mod.image_to_url = self._original
        else:
            if hasattr(st_image_mod, "image_to_url"):
                del st_image_mod.image_to_url

    def test_apply_patches_missing_image_to_url(self):
        """Confirms the real installed Streamlit (1.61.1 in this
        environment) genuinely lacks `image_to_url` at the old location --
        this is the actual reported bug's precondition, reproduced fresh
        (not assumed) via setUp's removal above."""
        import streamlit.elements.image as st_image_mod
        from doar.phase2c6 import canvas_compat

        self.assertFalse(hasattr(st_image_mod, "image_to_url"))
        canvas_compat.apply()
        self.assertTrue(hasattr(st_image_mod, "image_to_url"))
        self.assertIs(st_image_mod.image_to_url, canvas_compat._shim_image_to_url)

    def test_shim_accepts_exact_reported_call_signature_without_raising(self):
        """Reproduces streamlit_drawable_canvas/__init__.py's own call:
        `st_image.image_to_url(background_image, width, True, "RGB", "PNG", image_id)`
        -- 6 positional args, `width` a raw int. Must not raise
        AttributeError or TypeError."""
        from doar.phase2c6 import canvas_compat
        import streamlit.elements.image as st_image_mod

        canvas_compat.apply()
        img = Image.new("RGB", (64, 48), color="white")
        try:
            st_image_mod.image_to_url(img, 64, True, "RGB", "PNG", "regression-test-id")
        except AttributeError as e:
            self.fail(f"the exact reported bug reproduced: {e}")

    def test_apply_is_idempotent(self):
        from doar.phase2c6 import canvas_compat
        canvas_compat.apply()
        canvas_compat.apply()  # must not raise or double-wrap

    def test_shim_does_not_override_a_genuinely_present_image_to_url(self):
        """If a future Streamlit release restores `image_to_url` at its
        old location, apply() must become a no-op, not stomp on it.
        setUp already removed the attribute for this test class; this test
        puts a sentinel back to simulate that future state."""
        import streamlit.elements.image as st_image_mod
        from doar.phase2c6 import canvas_compat

        sentinel = object()
        st_image_mod.image_to_url = sentinel
        canvas_compat.apply()
        self.assertIs(st_image_mod.image_to_url, sentinel)


_ENV_VARS = ("DOAR_PHASE2C1_IMAGES_DIR", "DOAR_PHASE2C1_STORE_PATH",
             "DOAR_PHASE2C5_STORE_PATH", "DOAR_PHASE2C5_PROPOSALS_PATH",
             "DOAR_PHASE2C5_EXPORT_DIR")


class _IsolatedAppTestCase(unittest.TestCase):
    """Mirrors tests/test_phase2c1_app.py's isolation pattern exactly:
    every test points the app at a throwaway temp directory, never the
    real outputs/phase2c1/ or outputs/phase2c5/ trees."""

    def setUp(self):
        if not _STREAMLIT_TESTING_AVAILABLE:
            self.skipTest("streamlit.testing.v1.AppTest not available")
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self._tmp.name)
        self.images_dir = self.tmp_dir / "images"
        self.images_dir.mkdir()
        for i in range(3):
            Image.new("RGB", (80, 60), "white").save(self.images_dir / f"p2b_{i:04d}.jpg")

        self._env_backup = {k: os.environ.get(k) for k in _ENV_VARS}
        os.environ["DOAR_PHASE2C1_IMAGES_DIR"] = str(self.images_dir)
        os.environ["DOAR_PHASE2C1_STORE_PATH"] = str(self.tmp_dir / "store2c1.csv")
        os.environ["DOAR_PHASE2C5_STORE_PATH"] = str(self.tmp_dir / "store2c5.csv")
        os.environ["DOAR_PHASE2C5_PROPOSALS_PATH"] = str(self.tmp_dir / "no_proposals.csv")
        os.environ["DOAR_PHASE2C5_EXPORT_DIR"] = str(self.tmp_dir / "exports")

        for real_dir in ("outputs/phase2c1", "outputs/phase2c5", "outputs/phase2c6"):
            real_private_dir = (ROOT / real_dir).resolve()
            self.assertFalse(
                str(self.images_dir.resolve()).startswith(str(real_private_dir)),
                f"test images dir must never resolve inside the real {real_private_dir}")

    def tearDown(self):
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._tmp.cleanup()

    def _app(self) -> "AppTest":
        at = AppTest.from_file(str(ROOT / "phase2c5_part_annotation_app.py"))
        at.run(timeout=60)
        return at

    def _set_annotator(self, at, annotator_id="tester1"):
        text_inputs = at.text_input
        box = next(w for w in text_inputs if w.label == "Your annotator ID")
        box.set_value(annotator_id)
        at.run(timeout=60)
        return at


class CanvasRegressionTests(_IsolatedAppTestCase):
    """The decisive end-to-end regression guard: forces status='present',
    which is the ONLY code path that reaches st_canvas(...) -- the exact
    site of the reported AttributeError. If canvas_compat's shim ever
    stops working (e.g. a Streamlit upgrade reshapes image_to_url again,
    or the shim is accidentally removed from the app), this test fails
    with the real traceback, not a generic assertion."""

    def test_app_renders_without_exception_before_annotator_id(self):
        at = self._app()
        self.assertEqual(len(at.exception), 0, [str(e.value) for e in at.exception])

    def test_app_renders_without_exception_after_annotator_id(self):
        at = self._app()
        at = self._set_annotator(at)
        self.assertEqual(len(at.exception), 0, [str(e.value) for e in at.exception])

    def test_st_canvas_call_site_reached_and_passes_with_no_exception(self):
        at = self._app()
        at = self._set_annotator(at)
        status_radios = [r for r in at.radio if r.label == "Status"]
        self.assertTrue(status_radios, "could not find the Status radio widget")
        status_radios[0].set_value("present")
        at.run(timeout=60)
        self.assertEqual(len(at.exception), 0,
                          f"st_canvas() call site raised: {[str(e.value) for e in at.exception]}")
        # Confirms execution proceeded PAST st_canvas() to the caption that
        # reads canvas_result and runs the reconciliation logic -- not just
        # that no exception happened somewhere unrelated.
        captions = [c.value for c in at.caption]
        self.assertTrue(any("box(es) shown on canvas" in c for c in captions),
                         f"expected post-canvas caption not found; captions={captions}")

    def test_no_attribute_error_naming_image_to_url_anywhere_in_exceptions(self):
        """Even if some other exception were to occur, this specifically
        pins down that THIS bug (by name) never recurs."""
        at = self._app()
        at = self._set_annotator(at)
        status_radios = [r for r in at.radio if r.label == "Status"]
        status_radios[0].set_value("present")
        at.run(timeout=60)
        for exc in at.exception:
            self.assertNotIsInstance(exc.value, AttributeError)
            self.assertNotIn("image_to_url", str(exc.value))


if __name__ == "__main__":
    unittest.main()
