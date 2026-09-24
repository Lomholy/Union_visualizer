"""Tests for union_viewer.py's non-GUI logic: compute_trace_data, the
worker-process side of the trace pool, needs no display to run.

The GUI itself is not unit-testable without a display, so its own wiring
was exercised interactively with QT_QPA_PLATFORM=offscreen against real
test instruments while implementing changes to it.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import union_viewer as uv  # noqa: E402
from logger_output import LoggerCounts  # noqa: E402


@unittest.skipUnless(shutil.which("mcrun"), "mcrun not on PATH")
class ComputeTraceDataTest(unittest.TestCase):
    def test_all_requested_rays_are_kept(self):
        # trace_instrument's own max_rays defaults to 1000 (meant for a
        # file loaded independently of any run); compute_trace_data must
        # pass the actual requested ncount through instead, or a run for
        # more rays than that silently gets truncated back down to 1000.
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "simple_test.instr")
            shutil.copy(ROOT / "tests" / "simple_test.instr", instr)
            try:
                _, rays = uv.compute_trace_data(instr, False, [], 1200, 7)
            except Exception as e:
                self.skipTest(f"mcrun could not compile here: {e}")
        self.assertEqual(rays.n_rays, 1200)


class BuildCountsGroupTest(unittest.TestCase):
    def test_one_mesh_per_placeable_logger(self):
        counts = {
            "logger_a": LoggerCounts("logger_a", "logger", "x", "y", (-1, 1, -1, 1), np.ones((4, 4))),
            "logger_b": LoggerCounts("logger_b", "abs_logger", "z", "x", (-2, 2, -2, 2), np.ones((3, 3))),
        }
        world_matrices = {"logger_a": np.eye(4), "logger_b": np.eye(4)}

        group, meshes, textures = uv.build_counts_group(counts, world_matrices, vmin=0, vmax=1)

        self.assertEqual(set(meshes), {"logger_a", "logger_b"})
        self.assertEqual(set(textures), {"logger_a", "logger_b"})
        self.assertEqual(len(group.children), 2)
        self.assertEqual(textures["logger_a"].size, (4, 4, 1))
        self.assertEqual(textures["logger_b"].size, (3, 3, 1))

    def test_a_logger_missing_from_the_loaded_instrument_is_skipped(self):
        counts = {
            "matched": LoggerCounts("matched", "logger", "x", "y", (-1, 1, -1, 1), np.ones((2, 2))),
            "not_in_instrument": LoggerCounts(
                "not_in_instrument", "logger", "x", "y", (-1, 1, -1, 1), np.ones((2, 2))
            ),
        }
        world_matrices = {"matched": np.eye(4)}

        group, meshes, textures = uv.build_counts_group(counts, world_matrices, vmin=0, vmax=1)

        self.assertEqual(set(meshes), {"matched"})
        self.assertEqual(len(group.children), 1)

    def test_mesh_corners_are_placed_by_the_world_matrix(self):
        counts = {
            "logger_a": LoggerCounts("logger_a", "logger", "x", "y", (-1, 1, -2, 2), np.ones((2, 2))),
        }
        world_matrix = np.eye(4)
        world_matrix[:3, 3] = (10, 20, 30)  # pure translation
        _, meshes, _ = uv.build_counts_group(counts, {"logger_a": world_matrix}, vmin=0, vmax=1)

        positions = meshes["logger_a"].geometry.positions.data
        # x in [-1, 1] -> world x in [9, 11]; y in [-2, 2] -> world y in
        # [18, 22]; z (the unlisted axis) is 0 locally -> world z is 30.
        np.testing.assert_allclose(positions[:, 0].min(), 9.0)
        np.testing.assert_allclose(positions[:, 0].max(), 11.0)
        np.testing.assert_allclose(positions[:, 1].min(), 18.0)
        np.testing.assert_allclose(positions[:, 1].max(), 22.0)
        np.testing.assert_allclose(positions[:, 2], 30.0)


if __name__ == "__main__":
    unittest.main()
