"""Tests for the Ruler dock's point bookkeeping. Uses Qt's offscreen
platform, so no display is needed."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qtpy import QtWidgets  # noqa: E402

from panels import RulerPanel  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class RulerPanelTest(unittest.TestCase):
    def setUp(self):
        self.panel = RulerPanel()
        self.emitted = 0
        self.panel.points_edited.connect(self._count)

    def _count(self):
        self.emitted += 1

    def test_clicks_fill_both_points_then_restart(self):
        self.panel.add_point((1, 2, 3))
        self.panel.add_point((4, 5, 6))
        np.testing.assert_allclose(self.panel.points(), [(1, 2, 3), (4, 5, 6)])
        self.panel.add_point((7, 8, 9))
        np.testing.assert_allclose(self.panel.points(), [(7, 8, 9)])
        self.assertEqual(self.emitted, 3)

    def test_clicked_point_fills_the_boxes(self):
        self.panel.add_point((0.1, -0.25, 10.5))
        self.assertEqual([s.value() for s in self.panel.point_spins[0]], [0.1, -0.25, 10.5])

    def test_typing_a_coordinate_moves_the_point(self):
        self.panel.add_point((1, 2, 3))
        self.panel.add_point((4, 5, 6))
        self.emitted = 0
        self.panel.point_spins[1][2].setValue(3.0)
        np.testing.assert_allclose(self.panel.points(), [(1, 2, 3), (4, 5, 3)])
        self.assertEqual(self.emitted, 1)

    def test_points_can_be_typed_from_scratch(self):
        self.assertEqual(self.panel.points(), [])
        self.panel.point_spins[0][0].setValue(1.0)
        self.panel.point_spins[1][1].setValue(2.0)
        np.testing.assert_allclose(self.panel.points(), [(1, 0, 0), (0, 2, 0)])

    def test_clear_unsets_both_points(self):
        self.panel.add_point((1, 2, 3))
        self.panel.add_point((4, 5, 6))
        self.panel.clear()
        self.assertEqual(self.panel.points(), [])
        self.assertEqual([s.value() for s in self.panel.point_spins[0]], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
