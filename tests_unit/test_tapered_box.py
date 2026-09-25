"""Tests for tapered Union_box support (xwidth2 / yheight2).

McStas's Union_box takes optional xwidth2/yheight2 giving a different width
and height at the +z face, turning the box into a rectangular frustum whose
cross-section interpolates linearly along local z. See Union_box.comp: the
parameters default to -1, its INITIALIZE treats any negative value as "same
as xwidth/yheight", and it rejects a value that is <= 0 but not exactly -1.

The expected frustum volume used below is the standard prismatoid formula
V = h/6 * (A_bottom + 4*A_middle + A_top), which for rectangular faces is

    zdepth/6 * (x1*y1 + (x1+x2)*(y1+y2) + x2*y2)

since 4*A_middle = 4 * ((x1+x2)/2) * ((y1+y2)/2) = (x1+x2)*(y1+y2). It is
also exactly the integral of the interpolated cross-section along z, which is
the property both the SDF and the BRep loft are built to have.

This directory is intentionally NOT under tests/, since
.github/workflows/run-tests.yml feeds every file in tests/* to
unviz --export as a pipeline smoke test; a plain assertion script
there would break that loop.
"""

import io
import sys
import unittest
import contextlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import preprocess  # noqa: E402
import bounding_box  # noqa: E402
import signed_distance_functions as sdfs  # noqa: E402
import plot_union_cloud  # noqa: E402


class FakeBox:
    """Stand-in for a mcstasscript Union_box component.

    Real components are built by mcstasscript's component reader from the
    installed Union_box.comp, which requires a McStas installation. The code
    under test only ever reads .name/.xwidth/.yheight/.zdepth and the two
    optional taper parameters, so a plain object keeps these tests runnable
    (and fast) wherever CI happens to run. The mcstasscript-backed path is
    covered end to end by tests/box_taper_test.instr instead."""

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


class TestSdfBox(unittest.TestCase):
    def test_untapered_is_unchanged(self):
        """The plain-cuboid branch must stay the exact Euclidean box SDF."""
        comp = FakeBox(0.6, 0.4, 0.2)
        b = np.array([0.6, 0.4, 0.2]) / 2

        rng = np.random.default_rng(0)
        p = rng.uniform(-0.5, 0.5, size=(200, 3))

        d = np.abs(p) - b
        expected = (
            np.linalg.norm(np.maximum(d, 0), axis=-1)
            + np.minimum(np.maximum.reduce(d, axis=-1), 0)
        )
        np.testing.assert_allclose(sdfs.sdf_box(comp, p), expected)

    def test_explicit_taper_equal_to_base_matches_cuboid(self):
        """xwidth2 == xwidth must take the cuboid branch, not the frustum
        one, so writing the taper out explicitly changes nothing."""
        plain = FakeBox(0.6, 0.4, 0.2)
        explicit = FakeBox(0.6, 0.4, 0.2, xwidth2=0.6, yheight2=0.4)

        rng = np.random.default_rng(1)
        p = rng.uniform(-0.5, 0.5, size=(200, 3))
        np.testing.assert_allclose(
            sdfs.sdf_box(plain, p), sdfs.sdf_box(explicit, p)
        )

    def test_frustum_sign(self):
        """Inside is negative, outside positive, at points chosen from the
        interpolated half-width rather than from either end face alone."""
        # Half-width runs 0.3 -> 0.1 and half-height 0.3 -> 0.2 over
        # z in [-0.3, +0.3].
        comp = FakeBox(0.6, 0.6, 0.6, xwidth2=0.2, yheight2=0.4)

        # At z = 0 the half-extents are 0.2 (x) and 0.25 (y).
        inside = np.array([
            [0.0, 0.0, 0.0],
            [0.19, 0.24, 0.0],
            [0.29, 0.29, -0.29],   # inside near the wide -z face
            [0.09, 0.19, 0.29],    # inside near the narrow +z face
        ])
        outside = np.array([
            [0.21, 0.0, 0.0],      # past the interpolated half-width
            [0.0, 0.26, 0.0],
            [0.29, 0.29, 0.29],    # -z face extents, but at the +z face
            [0.0, 0.0, 0.31],      # beyond zdepth
        ])

        self.assertTrue(np.all(sdfs.sdf_box(comp, inside) < 0))
        self.assertTrue(np.all(sdfs.sdf_box(comp, outside) > 0))

    def test_frustum_volume_by_sampling(self):
        """Monte-Carlo the fraction of a bounding volume that is inside, as
        an independent check on the interpolation itself."""
        x1, y1, x2, y2, zdepth = 0.6, 0.6, 0.6, 0.2, 0.4
        comp = FakeBox(x1, y1, zdepth, xwidth2=x2, yheight2=y2)

        rng = np.random.default_rng(2)
        n = 400_000
        p = np.column_stack([
            rng.uniform(-x1 / 2, x1 / 2, n),
            rng.uniform(-y1 / 2, y1 / 2, n),
            rng.uniform(-zdepth / 2, zdepth / 2, n),
        ])
        box_volume = x1 * y1 * zdepth
        measured = box_volume * np.mean(sdfs.sdf_box(comp, p) < 0)

        self.assertAlmostEqual(
            measured, frustum_volume(x1, y1, x2, y2, zdepth), delta=1e-3
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
        """Every sampled surface point must lie inside the reported box."""
        comp = FakeBox(0.6, 0.6, 0.6, xwidth2=0.2, yheight2=0.4)
        bmin, bmax = bounding_box.compute_local_bbox(comp)
        pts = plot_union_cloud.sample_union_box(comp, 600)
        self.assertTrue(np.all(pts >= bmin - 1e-12))
        self.assertTrue(np.all(pts <= bmax + 1e-12))


class TestSampleUnionBox(unittest.TestCase):
    def test_samples_lie_on_the_surface(self):
        """The sampler and the SDF must agree about where the surface is -
        prioritise_points projects these points with the SDF, so a sampler
        that disagrees would silently produce an empty cloud."""
        comp = FakeBox(0.6, 0.6, 0.6, xwidth2=0.2, yheight2=0.4)
        pts = plot_union_cloud.sample_union_box(comp, 600)
        np.testing.assert_allclose(
            sdfs.sdf_box(comp, pts), np.zeros(len(pts)), atol=1e-12
        )

    def test_untapered_samples_lie_on_the_surface(self):
        comp = FakeBox(0.6, 0.4, 0.2)
        pts = plot_union_cloud.sample_union_box(comp, 600)
        np.testing.assert_allclose(
            sdfs.sdf_box(comp, pts), np.zeros(len(pts)), atol=1e-12
        )

    def test_covers_both_end_faces(self):
        comp = FakeBox(0.6, 0.6, 0.6, xwidth2=0.2, yheight2=0.4)
        pts = plot_union_cloud.sample_union_box(comp, 600)
        # The widest |x| must appear at the -z face and the narrowest at +z.
        at_minus_z = pts[np.isclose(pts[:, 2], -0.3)]
        at_plus_z = pts[np.isclose(pts[:, 2], 0.3)]
        self.assertAlmostEqual(np.abs(at_minus_z[:, 0]).max(), 0.3)
        self.assertAlmostEqual(np.abs(at_plus_z[:, 0]).max(), 0.1)
        self.assertAlmostEqual(np.abs(at_plus_z[:, 1]).max(), 0.2)


class TestBrepBox(unittest.TestCase):
    """The BRep path needs pythonocc-core, which the unviz environment
    provides; skip rather than fail if it is missing."""

    @classmethod
    def setUpClass(cls):
        try:
            import brep  # noqa: F401
            from OCC.Core.BRepGProp import brepgprop  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"pythonocc-core unavailable: {exc}")

    def _volume(self, comp):
        import brep
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        world = {comp.name: np.eye(4)}
        shape = brep.build_comp_brep(comp, world)

        props = GProp_GProps()
        brepgprop.VolumeProperties(shape, props)
        return props.Mass()

    def test_tapered_volume_matches_analytic_frustum(self):
        x1, y1, x2, y2, zdepth = 0.6, 0.6, 0.2, 0.4, 0.6
        comp = FakeBox(x1, y1, zdepth, xwidth2=x2, yheight2=y2)
        self.assertAlmostEqual(
            self._volume(comp), frustum_volume(x1, y1, x2, y2, zdepth), places=9
        )

    def test_one_sided_taper_volume(self):
        x1, y1, x2, y2, zdepth = 0.4, 0.5, 0.1, 0.5, 0.3
        comp = FakeBox(x1, y1, zdepth, xwidth2=x2)
        self.assertAlmostEqual(
            self._volume(comp), frustum_volume(x1, y1, x2, y2, zdepth), places=9
        )

    def test_untapered_volume_is_the_cuboid(self):
        comp = FakeBox(0.6, 0.4, 0.2)
        self.assertAlmostEqual(self._volume(comp), 0.6 * 0.4 * 0.2, places=9)

    def test_explicit_taper_equal_to_base_matches_cuboid(self):
        comp = FakeBox(0.6, 0.4, 0.2, xwidth2=0.6, yheight2=0.4)
        self.assertAlmostEqual(self._volume(comp), 0.6 * 0.4 * 0.2, places=9)

    def test_tapered_solid_is_usable_in_a_boolean(self):
        """A lofted shape has to behave like any other solid under the
        BRepAlgoAPI_Cut that priority subtraction performs."""
        import brep
        from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCC.Core.BRepGProp import brepgprop
        from OCC.Core.GProp import GProp_GProps

        tapered = FakeBox(0.6, 0.6, 0.6, xwidth2=0.2, yheight2=0.4,
                          name="tapered")
        cutter = FakeBox(2.0, 2.0, 0.3, name="cutter")

        world = {"tapered": np.eye(4), "cutter": np.eye(4)}
        # Cut away everything below z = -0.15, which removes the widest slab.
        world["cutter"][2, 3] = -0.3

        shape = brep.build_comp_brep(tapered, world)
        cut = BRepAlgoAPI_Cut(shape, brep.build_comp_brep(cutter, world))
        cut.Build()
        self.assertTrue(cut.IsDone())

        props = GProp_GProps()
        brepgprop.VolumeProperties(cut.Shape(), props)
        remaining = props.Mass()

        # What survives is the frustum from z = -0.15 to z = +0.3, i.e. the
        # same interpolation restricted to the upper three quarters.
        t = 0.25  # fraction along the box where the cut lands
        x_at_cut = 0.6 + t * (0.2 - 0.6)
        y_at_cut = 0.6 + t * (0.4 - 0.6)
        expected = frustum_volume(x_at_cut, y_at_cut, 0.2, 0.4, 0.45)
        self.assertAlmostEqual(remaining, expected, places=9)

    def test_degenerate_taper_is_reported_clearly(self):
        """A zero-size face would make the loft fail inside OpenCascade;
        box_dimensions intercepts it first and falls back to untapered."""
        comp = FakeBox(0.6, 0.4, 0.2, xwidth2=0.0)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            volume = self._volume(comp)
        self.assertAlmostEqual(volume, 0.6 * 0.4 * 0.2, places=9)
        self.assertIn("xwidth2", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
