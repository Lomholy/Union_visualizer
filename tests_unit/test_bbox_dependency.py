"""Tests for the world-bbox overlap helpers in bounding_box.py: the
prefilter that skips a higher-priority component's boolean/SDF term when
its box can't possibly touch the target, and the dependency signature that
uses the same overlap relation to decide whether a component needs
remeshing between reloads.

This directory is intentionally NOT under tests/, since
.github/workflows/run-tests.yml feeds every file in tests/* to
unviz --export as a pipeline smoke test; a plain assertion script
there would break that loop.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import bounding_box  # noqa: E402


class FakeBox:
    """Stand-in for a mcstasscript Union_box component: only the fields
    compute_world_bbox/component_dependency_signature actually read."""

    component_name = "Union_box"

    def __init__(self, name, priority, xwidth=1.0, yheight=1.0, zdepth=1.0,
                 mask_string=None, mask_setting=None):
        self.name = name
        self.priority = priority
        self.xwidth = xwidth
        self.yheight = yheight
        self.zdepth = zdepth
        self.mask_string = mask_string
        self.mask_setting = mask_setting


def translation(x, y, z):
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


class TestBoxesOverlap(unittest.TestCase):
    def test_separated_boxes_do_not_overlap(self):
        self.assertFalse(
            bounding_box.boxes_overlap(
                np.array([0, 0, 0]), np.array([1, 1, 1]),
                np.array([5, 5, 5]), np.array([6, 6, 6]),
            )
        )

    def test_nested_boxes_overlap(self):
        self.assertTrue(
            bounding_box.boxes_overlap(
                np.array([0, 0, 0]), np.array([10, 10, 10]),
                np.array([2, 2, 2]), np.array([3, 3, 3]),
            )
        )

    def test_partially_overlapping_boxes_overlap(self):
        self.assertTrue(
            bounding_box.boxes_overlap(
                np.array([0, 0, 0]), np.array([2, 2, 2]),
                np.array([1, 1, 1]), np.array([3, 3, 3]),
            )
        )

    def test_touching_boxes_overlap_within_tolerance(self):
        # A common, deliberate arrangement in Union instruments - must not
        # be treated as separated just because they share a face exactly.
        self.assertTrue(
            bounding_box.boxes_overlap(
                np.array([0, 0, 0]), np.array([1, 1, 1]),
                np.array([1, 0, 0]), np.array([2, 1, 1]),
            )
        )

    def test_conservative_on_a_rotated_shape(self):
        """Two boxes whose *world axis-aligned* extents overlap even though
        the true (rotated) shapes might not - boxes_overlap only ever
        over-estimates, so this is the expected, safe answer."""
        self.assertTrue(
            bounding_box.boxes_overlap(
                np.array([-1, -1, -1]), np.array([1, 1, 1]),
                np.array([-1, -1, -1]), np.array([1, 1, 1]),
            )
        )


class TestOverlapping(unittest.TestCase):
    def test_returns_only_overlapping_candidates(self):
        world_bboxes = {
            "a": (np.array([0, 0, 0]), np.array([1, 1, 1])),
            "b": (np.array([0.5, 0.5, 0.5]), np.array([1.5, 1.5, 1.5])),
            "c": (np.array([10, 10, 10]), np.array([11, 11, 11])),
        }
        self.assertEqual(
            bounding_box.overlapping("a", ["b", "c"], world_bboxes), ["b"]
        )

    def test_excludes_self(self):
        world_bboxes = {"a": (np.array([0, 0, 0]), np.array([1, 1, 1]))}
        self.assertEqual(bounding_box.overlapping("a", ["a"], world_bboxes), [])


class TestComputeAllWorldBboxes(unittest.TestCase):
    def test_matches_compute_world_bbox(self):
        comps = {"a": FakeBox("a", priority=1, xwidth=2, yheight=2, zdepth=2)}
        world_matrices = {"a": translation(3, 0, 0)}
        all_bboxes = bounding_box.compute_all_world_bboxes(comps, world_matrices)
        expected = bounding_box.compute_world_bbox(
            comps["a"], world_matrices, margin=0.0
        )
        np.testing.assert_allclose(all_bboxes["a"][0], expected[0])
        np.testing.assert_allclose(all_bboxes["a"][1], expected[1])


class TestComponentDependencySignature(unittest.TestCase):
    def _bboxes(self, comps, world_matrices):
        return bounding_box.compute_all_world_bboxes(comps, world_matrices)

    def test_identical_state_gives_identical_signature(self):
        comps = {
            "low": FakeBox("low", priority=1),
            "high": FakeBox("high", priority=2),
        }
        world_matrices = {"low": translation(0, 0, 0), "high": translation(0, 0, 0)}
        bboxes = self._bboxes(comps, world_matrices)
        sig1 = bounding_box.component_dependency_signature("low", comps, bboxes)
        sig2 = bounding_box.component_dependency_signature("low", comps, bboxes)
        self.assertEqual(sig1, sig2)

    def test_moving_an_overlapping_higher_priority_cutter_changes_signature(self):
        comps = {
            "low": FakeBox("low", priority=1),
            "high": FakeBox("high", priority=2),
        }
        world_matrices = {"low": translation(0, 0, 0), "high": translation(0, 0, 0)}
        bboxes_before = self._bboxes(comps, world_matrices)
        sig_before = bounding_box.component_dependency_signature(
            "low", comps, bboxes_before
        )

        world_matrices["high"] = translation(0.3, 0, 0)  # still overlapping "low"
        bboxes_after = self._bboxes(comps, world_matrices)
        sig_after = bounding_box.component_dependency_signature(
            "low", comps, bboxes_after
        )

        self.assertNotEqual(sig_before, sig_after)

    def test_moving_a_non_overlapping_higher_priority_component_does_not_change_signature(self):
        """The core of the prefilter: a cutter that never overlaps the
        target, and still doesn't after moving, must not appear in the
        signature at all - so its own movement is invisible to "low"."""
        comps = {
            "low": FakeBox("low", priority=1),
            "far": FakeBox("far", priority=2),
        }
        world_matrices = {"low": translation(0, 0, 0), "far": translation(100, 0, 0)}
        bboxes_before = self._bboxes(comps, world_matrices)
        sig_before = bounding_box.component_dependency_signature(
            "low", comps, bboxes_before
        )

        world_matrices["far"] = translation(200, 0, 0)  # moved, still far away
        bboxes_after = self._bboxes(comps, world_matrices)
        sig_after = bounding_box.component_dependency_signature(
            "low", comps, bboxes_after
        )

        self.assertEqual(sig_before, sig_after)

    def test_lower_priority_overlapping_component_is_not_a_dependency(self):
        comps = {
            "low": FakeBox("low", priority=1),
            "lower": FakeBox("lower", priority=0),
        }
        world_matrices = {"low": translation(0, 0, 0), "lower": translation(0, 0, 0)}
        bboxes = self._bboxes(comps, world_matrices)
        sig = bounding_box.component_dependency_signature("low", comps, bboxes)
        # No cutters at all: "lower" has lower priority than "low".
        own_min, own_max, cutters, masks, mask_string, mask_setting = sig
        self.assertEqual(cutters, ())

    def test_priority_reordering_that_changes_the_candidate_set_is_detected(self):
        """"mid"'s own numeric priority moving past "low" changes who
        counts as higher-priority for "low", even with no geometry moving
        at all."""
        comps = {
            "low": FakeBox("low", priority=5),
            "mid": FakeBox("mid", priority=6),
        }
        world_matrices = {"low": translation(0, 0, 0), "mid": translation(0, 0, 0)}
        bboxes = self._bboxes(comps, world_matrices)
        sig_before = bounding_box.component_dependency_signature(
            "low", comps, bboxes
        )

        comps["mid"].priority = 4  # now lower priority than "low"
        sig_after = bounding_box.component_dependency_signature(
            "low", comps, bboxes
        )

        self.assertNotEqual(sig_before, sig_after)

    def test_mask_setting_change_is_detected_without_any_bbox_move(self):
        comps = {
            "masked": FakeBox("masked", priority=1, mask_string="mask",
                              mask_setting="All"),
            "mask": FakeBox("mask", priority=1),
        }
        world_matrices = {"masked": translation(0, 0, 0), "mask": translation(0, 0, 0)}
        bboxes = self._bboxes(comps, world_matrices)
        sig_before = bounding_box.component_dependency_signature(
            "masked", comps, bboxes
        )

        comps["masked"].mask_setting = "Any"
        sig_after = bounding_box.component_dependency_signature(
            "masked", comps, bboxes
        )

        self.assertNotEqual(sig_before, sig_after)

    def test_moving_a_mask_changes_the_masked_component_even_far_away(self):
        """Masks are matched by name, not bbox overlap - a mask sitting far
        from what it masks is still a real dependency (its intersection
        with a distant mask just yields an empty result), so moving it
        must still register as a change."""
        comps = {
            "masked": FakeBox("masked", priority=1, mask_string=None),
        }
        comps["masker"] = FakeBox("masker", priority=1, mask_string="masked")
        world_matrices = {
            "masked": translation(0, 0, 0),
            "masker": translation(100, 0, 0),
        }
        bboxes_before = self._bboxes(comps, world_matrices)
        sig_before = bounding_box.component_dependency_signature(
            "masked", comps, bboxes_before
        )

        world_matrices["masker"] = translation(200, 0, 0)
        bboxes_after = self._bboxes(comps, world_matrices)
        sig_after = bounding_box.component_dependency_signature(
            "masked", comps, bboxes_after
        )

        self.assertNotEqual(sig_before, sig_after)

    def test_material_string_is_not_part_of_the_signature(self):
        """A colour-only change must not trigger a remesh - the GUI's own
        colour system handles it without touching geometry."""
        comp = FakeBox("a", priority=1)
        comp.material_string = "Al"
        comps = {"a": comp}
        world_matrices = {"a": translation(0, 0, 0)}
        bboxes = self._bboxes(comps, world_matrices)
        sig_before = bounding_box.component_dependency_signature("a", comps, bboxes)

        comp.material_string = "Vacuum"
        sig_after = bounding_box.component_dependency_signature("a", comps, bboxes)

        self.assertEqual(sig_before, sig_after)


class TestUnionMeshBboxScale(unittest.TestCase):
    """compute_local_bbox previously loaded a Union_mesh's file without
    applying coordinate_scale, making its box ~1000x too large for a
    millimetre-unit STL and defeating any overlap check involving it."""

    def test_coordinate_scale_is_applied(self):
        class FakeMeshComp:
            component_name = "Union_mesh"

            def __init__(self, filename, coordinate_scale):
                self.filename = filename
                self.coordinate_scale = coordinate_scale

        stl_path = str(
            Path(__file__).resolve().parent.parent / "tests" / "crack_cryst.stl"
        )
        unscaled = bounding_box.compute_local_bbox(
            FakeMeshComp(stl_path, coordinate_scale=1.0)
        )
        scaled = bounding_box.compute_local_bbox(
            FakeMeshComp(stl_path, coordinate_scale=1e-3)
        )
        np.testing.assert_allclose(
            np.array(scaled[0]), np.array(unscaled[0]) * 1e-3
        )
        np.testing.assert_allclose(
            np.array(scaled[1]), np.array(unscaled[1]) * 1e-3
        )

    def test_none_coordinate_scale_defaults_to_1e_minus_3(self):
        class FakeMeshComp:
            component_name = "Union_mesh"

            def __init__(self, filename):
                self.filename = filename
                self.coordinate_scale = None

        stl_path = str(
            Path(__file__).resolve().parent.parent / "tests" / "crack_cryst.stl"
        )
        default = bounding_box.compute_local_bbox(FakeMeshComp(stl_path))

        class FakeMeshCompScaled(FakeMeshComp):
            def __init__(self, filename):
                self.filename = filename
                self.coordinate_scale = 1e-3

        explicit = bounding_box.compute_local_bbox(FakeMeshCompScaled(stl_path))
        np.testing.assert_allclose(np.array(default[0]), np.array(explicit[0]))
        np.testing.assert_allclose(np.array(default[1]), np.array(explicit[1]))


if __name__ == "__main__":
    unittest.main()
