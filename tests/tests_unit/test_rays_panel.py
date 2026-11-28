"""Tests for the Neutron Rays dock's layout. Uses Qt's offscreen platform,
so no display is needed."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from qtpy import QtWidgets  # noqa: E402

from unviz.panels import RaysPanel, SettingsPanel  # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class RaysPanelTest(unittest.TestCase):
    def test_show_checkbox_is_in_the_rays_panel_not_settings(self):
        rays = RaysPanel()
        self.assertTrue(rays.isAncestorOf(rays.show_checkbox))
        self.assertEqual(rays.show_checkbox.text(), "Show neutron rays")
        self.assertFalse(rays.show_checkbox.isChecked())
        settings = SettingsPanel()
        labels = [c.text() for c in settings.findChildren(QtWidgets.QCheckBox)]
        self.assertNotIn("Show neutron rays", labels)

    def test_disabled_options_leave_the_show_checkbox_clickable(self):
        rays = RaysPanel()
        rays.options.setEnabled(False)
        self.assertTrue(rays.show_checkbox.isEnabled())
        for widget in (rays.count_spin, rays.rerun_button, rays.line_checkbox):
            self.assertFalse(widget.isEnabled())


if __name__ == "__main__":
    unittest.main()
