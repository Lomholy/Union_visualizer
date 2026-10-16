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



if __name__ == "__main__":
    unittest.main()
