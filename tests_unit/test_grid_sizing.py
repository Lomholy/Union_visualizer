"""Tests for union_viewer.grid_size_for_bbox: the floor grid's footprint
should grow to cover instruments larger than the default grid, and stay at
the default for anything smaller.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from union_viewer import DEFAULT_GRID_SIZE, grid_size_for_bbox  # noqa: E402


class TestGridSizeForBbox(unittest.TestCase):
    def test_no_bbox_uses_default(self):
        self.assertEqual(grid_size_for_bbox(None), DEFAULT_GRID_SIZE)

    def test_small_geometry_uses_default(self):
        bbox = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
        self.assertEqual(grid_size_for_bbox(bbox), DEFAULT_GRID_SIZE)

    def test_large_geometry_grows_the_grid(self):
        # A 200 m long instrument along z, well past the default 100 m grid.
        bbox = (np.array([-1.0, -1.0, -100.0]), np.array([1.0, 1.0, 100.0]))
        self.assertGreater(grid_size_for_bbox(bbox), DEFAULT_GRID_SIZE)

    def test_uses_the_larger_of_x_and_z_extent(self):
        bbox = (np.array([-150.0, -1.0, -10.0]), np.array([150.0, 1.0, 10.0]))
        self.assertEqual(grid_size_for_bbox(bbox, margin=1.0), 300.0)

    def test_tall_geometry_does_not_grow_the_grid(self):
        # Height (y) shouldn't affect the floor grid's footprint.
        bbox = (np.array([-1.0, -500.0, -1.0]), np.array([1.0, 500.0, 1.0]))
        self.assertEqual(grid_size_for_bbox(bbox), DEFAULT_GRID_SIZE)


if __name__ == "__main__":
    unittest.main()
