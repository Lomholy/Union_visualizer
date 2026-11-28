"""Tests for tapered Union_box support (xwidth2 / yheight2).

McStas's Union_box takes optional xwidth2/yheight2 giving a different width
and height at the +z face, turning the box into a rectangular frustum whose
cross-section interpolates linearly along local z. See Union_box.comp: the
parameters default to -1, its INITIALIZE treats any negative value as "same
as xwidth/yheight", and it rejects a value that is <= 0 but not exactly -1.

This directory is intentionally NOT under tests/test_instruments/, since
.github/workflows/run-tests.yml feeds every file in tests/test_instruments/* to
unviz --export as a pipeline smoke test; a plain assertion script
there would break that loop.
"""

import io
import sys
import unittest
import contextlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from unviz import preprocess  # noqa: E402
from unviz import bounding_box  # noqa: E402


class FakeBox:
    """Stand-in for a mcstasscript Union_box component.

    Real components are built by mcstasscript's component reader from the
    installed Union_box.comp, which requires a McStas installation. The code
    under test only ever reads .name/.xwidth/.yheight/.zdepth and the two
    optional taper parameters, so a plain object keeps these tests runnable
    (and fast) wherever CI happens to run. The mcstasscript-backed path is
    covered end to end by tests/test_instruments/box_taper_test.instr instead."""

    component_name = "Union_box"

    def __init__(self, xwidth, yheight, zdepth, xwidth2=None, yheight2=None,
                 name="box"):
        self.name = name
        self.xwidth = xwidth
        self.yheight = yheight
        self.zdepth = zdepth
        if xwidth2 is not None:
            self.xwidth2 = xwidth2
        if yheight2 is not None:
            self.yheight2 = yheight2


def frustum_volume(x1, y1, x2, y2, zdepth):
    return zdepth / 6 * (x1 * y1 + (x1 + x2) * (y1 + y2) + x2 * y2)


class TestBoxDimensions(unittest.TestCase):
    """The single place that decides what 'unset' means."""

    def test_missing_attributes_are_untapered(self):
        # mcstasscript omits a setting parameter the instrument never wrote.
        comp = FakeBox(0.6, 0.4, 0.2)
        self.assertEqual(preprocess.box_dimensions(comp), (0.6, 0.4, 0.6, 0.4))

    def test_none_is_untapered(self):
        # mcstasscript reports an unset setting parameter as None rather than
        # as Union_box's own -1 default - the most likely source of a bug.
        comp = FakeBox(0.6, 0.4, 0.2)
        comp.xwidth2 = None
        comp.yheight2 = None
        self.assertEqual(preprocess.box_dimensions(comp), (0.6, 0.4, 0.6, 0.4))

    def test_minus_one_sentinel_is_untapered(self):
        comp = FakeBox(0.6, 0.4, 0.2, xwidth2=-1, yheight2=-1)
        self.assertEqual(preprocess.box_dimensions(comp), (0.6, 0.4, 0.6, 0.4))

    def test_real_values_pass_through(self):
        comp = FakeBox(0.6, 0.4, 0.2, xwidth2=0.2, yheight2=0.1)
        self.assertEqual(preprocess.box_dimensions(comp), (0.6, 0.4, 0.2, 0.1))

    def test_one_sided_taper(self):
        # Tapering only the width is legal and common.
        comp = FakeBox(0.6, 0.4, 0.2, xwidth2=0.2)
        self.assertEqual(preprocess.box_dimensions(comp), (0.6, 0.4, 0.2, 0.4))

    def test_numeric_strings_are_accepted(self):
        # The .instr parser can hand back an already-numeric value as a
        # string; parse_param passes those through untouched.
        comp = FakeBox("0.6", "0.4", "0.2", xwidth2="0.2")
        self.assertEqual(preprocess.box_dimensions(comp), (0.6, 0.4, 0.2, 0.4))

    def test_zero_warns_and_falls_back(self):
        # McStas itself errors on this; we warn and stay usable, matching how
        # the rest of preprocess.py degrades rather than aborting a run.
        comp = FakeBox(0.6, 0.4, 0.2, xwidth2=0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dims = preprocess.box_dimensions(comp)
        self.assertEqual(dims, (0.6, 0.4, 0.6, 0.4))
        self.assertIn("xwidth2", buf.getvalue())

    def test_negative_other_than_sentinel_warns_and_falls_back(self):
        comp = FakeBox(0.6, 0.4, 0.2, yheight2=-2.5)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dims = preprocess.box_dimensions(comp)
        self.assertEqual(dims, (0.6, 0.4, 0.6, 0.4))
        self.assertIn("yheight2", buf.getvalue())

    def test_unresolved_expression_warns_and_falls_back(self):
        # parse_param leaves an expression it could not evaluate as the
        # original string rather than a number.
        comp = FakeBox(0.6, 0.4, 0.2, xwidth2="some_undefined_var")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dims = preprocess.box_dimensions(comp)
        self.assertEqual(dims, (0.6, 0.4, 0.6, 0.4))
        self.assertIn("xwidth2", buf.getvalue())

    def test_box_is_tapered(self):
        self.assertFalse(preprocess.box_is_tapered(FakeBox(0.6, 0.4, 0.2)))
        self.assertFalse(
            preprocess.box_is_tapered(
                FakeBox(0.6, 0.4, 0.2, xwidth2=0.6, yheight2=0.4)
            )
        )
        self.assertTrue(
            preprocess.box_is_tapered(FakeBox(0.6, 0.4, 0.2, xwidth2=0.2))
        )


class TestBoundingBox(unittest.TestCase):
    def test_uses_the_larger_of_the_two_faces(self):
        comp = FakeBox(0.6, 0.2, 0.4, xwidth2=0.2, yheight2=0.8)
        bmin, bmax = bounding_box.compute_local_bbox(comp)
        # max width 0.6 (the -z face), max height 0.8 (the +z face).
        np.testing.assert_allclose(bmax, [0.3, 0.4, 0.2])
        np.testing.assert_allclose(bmin, [-0.3, -0.4, -0.2])

    def test_untapered_unchanged(self):
        comp = FakeBox(0.6, 0.2, 0.4)
        bmin, bmax = bounding_box.compute_local_bbox(comp)
        np.testing.assert_allclose(bmax, [0.3, 0.1, 0.2])
        np.testing.assert_allclose(bmin, [-0.3, -0.1, -0.2])

    def test_bbox_contains_the_frustum(self):
        """Every corner of the frustum must lie inside the reported box."""
        comp = FakeBox(0.6, 0.6, 0.6, xwidth2=0.2, yheight2=0.4)
        bmin, bmax = bounding_box.compute_local_bbox(comp)
        # the frustum's corners: 0.6 x 0.6 at -z, 0.2 x 0.4 at +z
        pts = np.array([[sx * w / 2, sy * h / 2, sz * 0.3]
                        for sz, w, h in ((-1, 0.6, 0.6), (1, 0.2, 0.4)) for sx in (-1, 1) for sy in (-1, 1)])
        self.assertTrue(np.all(pts >= bmin - 1e-12))
        self.assertTrue(np.all(pts <= bmax + 1e-12))


if __name__ == "__main__":
    unittest.main()
