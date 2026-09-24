"""Tests for logger_output.py: turning mcstasscript's McStasData objects for
any spatially-placeable detector output (Union loggers/abs_loggers, or
ordinary McStas monitors like PSD_monitor/Monitor_nD) into world-placeable
counts - a 2D heatmap plane, or a 1D line.

Fakes stand in for mcstasscript's McStasDataBinned/McStasDataEvent (mirrors
tests_unit/test_gui_helpers.py's FakeComponent), covering only the
attributes logger_output.py actually reads, so these tests need neither a
McStas install nor real .dat/mccode.sim files on disk.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import logger_output  # noqa: E402


class FakeMetadata:
    def __init__(self, info, limits=(), component_name=""):
        self.info = info
        self.limits = list(limits)
        self.component_name = component_name


class FakeBinnedDataset:
    """Stand-in for mcstasscript's McStasDataBinned (1D or 2D):
    logger_output.py only reads .data_type, .metadata.info/.limits/
    .component_name, and .Intensity."""

    def __init__(self, xvar, yvar, limits, intensity, component_name=""):
        self.data_type = f"Binned {np.ndim(intensity)}D"
        self.metadata = FakeMetadata({"xvar": xvar, "yvar": yvar}, limits, component_name)
        self.Intensity = intensity


class FakeEventDataset:
    """Stand-in for mcstasscript's McStasDataEvent: logger_output.py only
    reads .data_type, .variables, .get_data_column(axis), and
    .metadata.component_name."""

    def __init__(self, columns, component_name=""):
        self.data_type = "Events"
        self.variables = list(columns)
        self._columns = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
        self.metadata = FakeMetadata({}, component_name=component_name)

    def get_data_column(self, axis):
        return self._columns[axis]


class SpatialGridBinnedTest(unittest.TestCase):
    def test_space_logger_uses_its_own_xvar_yvar(self):
        intensity = np.array([[1.0, 2.0], [3.0, 4.0]])
        dataset = FakeBinnedDataset("z", "x", (-0.1, 0.2, -0.3, 0.4), intensity)
        axis1, axis2, limits, grid = logger_output.spatial_grid(dataset)
        self.assertEqual((axis1, axis2), ("z", "x"))
        self.assertEqual(limits, (-0.1, 0.2, -0.3, 0.4))
        np.testing.assert_array_equal(grid, intensity)

    def test_time_logger_has_no_spatial_axes(self):
        dataset = FakeBinnedDataset("t", "(I,I_err)", (0.035, 0.037), np.zeros(1000))
        self.assertIsNone(logger_output.spatial_grid(dataset))

    def test_reciprocal_space_logger_has_no_spatial_axes(self):
        dataset = FakeBinnedDataset("Q1", "Q2", (-1, 1, -1, 1), np.zeros((10, 10)))
        self.assertIsNone(logger_output.spatial_grid(dataset))


class SpatialGrid1DTest(unittest.TestCase):
    def test_1d_spatial_monitor_is_a_line(self):
        intensity = np.array([1.0, 2.0, 3.0, 4.0])
        dataset = FakeBinnedDataset("x", "", (-0.1, 0.1), intensity)
        axis1, axis2, limits, grid = logger_output.spatial_grid(dataset)
        self.assertEqual(axis1, "x")
        self.assertIsNone(axis2)
        self.assertEqual(limits, (-0.1, 0.1))
        np.testing.assert_array_equal(grid, intensity)

    def test_1d_wavelength_monitor_has_no_spatial_axis(self):
        dataset = FakeBinnedDataset("L", "", (0.5, 5.0), np.zeros(100))
        self.assertIsNone(logger_output.spatial_grid(dataset))


class SpatialGridEventTest(unittest.TestCase):
    def test_xyz_columns_are_histogrammed_into_the_xy_plane(self):
        dataset = FakeEventDataset({
            "p": [1.0, 1.0, 2.0],
            "x": [-0.5, 0.5, 0.5],
            "y": [-0.5, -0.5, 0.5],
            "z": [0.0, 0.0, 0.0],
            "t": [0.0, 0.0, 0.0],
        })
        axis1, axis2, limits, grid = logger_output.spatial_grid(dataset, event_bins=2)
        self.assertEqual((axis1, axis2), ("x", "y"))
        self.assertAlmostEqual(grid.sum(), 4.0)  # total weight preserved

    def test_prefers_xz_when_y_is_absent(self):
        dataset = FakeEventDataset({"p": [1.0], "x": [0.0], "z": [0.0]})
        axis1, axis2, _, _ = logger_output.spatial_grid(dataset, event_bins=2)
        self.assertEqual((axis1, axis2), ("x", "z"))

    def test_fewer_than_two_spatial_columns_is_not_histogrammable(self):
        dataset = FakeEventDataset({"p": [1.0], "t": [0.0]})
        self.assertIsNone(logger_output.spatial_grid(dataset))


class LoadCountsTest(unittest.TestCase):
    def test_includes_any_spatial_detector_not_just_union_loggers(self):
        # Every detector present, spatial or not, Union or not - load_counts
        # should keep exactly the spatially-placeable ones regardless of
        # component type, and label each with its own McStas component type.
        space_logger = FakeBinnedDataset(
            "z", "x", (-1, 1, -1, 1), np.ones((2, 2)), component_name="space_logger")
        time_logger = FakeBinnedDataset(
            "t", "(I,I_err)", (0, 1), np.zeros(4), component_name="time_logger")
        abs_logger = FakeBinnedDataset(
            "x", "y", (-1, 1, -1, 1), np.ones((3, 3)), component_name="abs_signal")
        psd_monitor = FakeBinnedDataset(
            "x", "y", (-1, 1, -1, 1), np.ones((2, 2)), component_name="psd")
        line_monitor = FakeBinnedDataset(
            "y", "(I,I_err)", (-0.05, 0.05), np.ones(50), component_name="strip")
        wavelength_monitor = FakeBinnedDataset(
            "L", "(I,I_err)", (0.5, 5.0), np.zeros(100), component_name="lmon")

        component_types = {
            "space_logger": "Union_logger_2D_space",
            "time_logger": "Union_logger_1D",
            "abs_signal": "Union_abs_logger_2D_space",
            "psd": "PSD_monitor",
            "strip": "Monitor_nD",
            "lmon": "L_monitor",
        }

        with mock.patch.object(
            logger_output.ms, "load_data",
            return_value=[
                space_logger, time_logger, abs_logger, psd_monitor, line_monitor, wavelength_monitor,
            ],
        ):
            counts = logger_output.load_counts("unused", component_types)

        self.assertEqual(set(counts), {"space_logger", "abs_signal", "psd", "strip"})
        self.assertEqual(counts["space_logger"].kind, "Union_logger_2D_space")
        self.assertEqual(counts["abs_signal"].kind, "Union_abs_logger_2D_space")
        self.assertEqual(counts["psd"].kind, "PSD_monitor")
        self.assertEqual(counts["strip"].kind, "Monitor_nD")
        self.assertIsNone(counts["strip"].axis2)
        self.assertEqual(counts["space_logger"].total, 4.0)


class PlacementTest(unittest.TestCase):
    def test_local_corners_fill_the_named_axes_and_zero_the_third(self):
        corners = logger_output.local_corners("z", "x", (-1, 2, -3, 4))
        self.assertEqual(corners.shape, (4, 3))
        # y (index 1) is the unlisted axis - always zero.
        np.testing.assert_array_equal(corners[:, 1], np.zeros(4))
        np.testing.assert_array_equal(sorted(corners[:, 2]), [-1, -1, 2, 2])  # z
        np.testing.assert_array_equal(sorted(corners[:, 0]), [-3, -3, 4, 4])  # x

    def test_local_line_points_span_limits_along_the_named_axis(self):
        points = logger_output.local_line_points("y", (-1, 3), n_bins=4)
        self.assertEqual(points.shape, (5, 3))  # 4 bins -> 5 edges
        # x, z (indices 0, 2) are the unlisted axes - always zero.
        np.testing.assert_array_equal(points[:, 0], np.zeros(5))
        np.testing.assert_array_equal(points[:, 2], np.zeros(5))
        np.testing.assert_allclose(points[:, 1], [-1, 0, 1, 2, 3])

    def test_world_points_applies_translation_and_rotation(self):
        local = np.array([[1.0, 0.0, 0.0]])
        # 90 degree rotation about z, plus a translation.
        world_matrix = np.array([
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 1.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        world = logger_output.world_points(local, world_matrix)
        np.testing.assert_allclose(world, [[10.0, 21.0, 30.0]])


class TextureImageTest(unittest.TestCase):
    def test_shape_and_dtype(self):
        grid = np.array([[0.0, 1.0], [2.0, 3.0]])
        image = logger_output.texture_image(grid)
        self.assertEqual(image.shape, (2, 2, 4))
        self.assertEqual(image.dtype, np.uint8)

    def test_min_and_max_bins_reach_the_ends_of_the_colour_scale(self):
        grid = np.array([[0.0, 5.0], [10.0, 2.0]])
        image = logger_output.texture_image(grid, vmin=0, vmax=10)
        # viridis: low end is dark purple-ish, high end is yellow-ish -
        # just check the extremes actually differ and alpha is opaque.
        self.assertFalse(np.array_equal(image[0, 0], image[1, 0]))
        np.testing.assert_array_equal(image[..., 3], 255)


class LineVertexColorsTest(unittest.TestCase):
    def test_each_bin_gets_a_pair_of_identical_vertex_colours(self):
        grid = np.array([0.0, 5.0, 10.0])
        colors = logger_output.line_vertex_colors(grid, vmin=0, vmax=10)
        self.assertEqual(colors.shape, (6, 4))
        np.testing.assert_array_equal(colors[0], colors[1])  # bin 0's two endpoints
        np.testing.assert_array_equal(colors[2], colors[3])  # bin 1's two endpoints
        np.testing.assert_array_equal(colors[4], colors[5])  # bin 2's two endpoints
        self.assertFalse(np.array_equal(colors[0], colors[4]))  # different bins differ


if __name__ == "__main__":
    unittest.main()
