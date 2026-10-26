"""Tests for qt_helpers' widget logic. Uses Qt's offscreen platform, so no
window is shown and no display is needed."""

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from qtpy import QtCore, QtWidgets  # noqa: E402

from qt_helpers import (  # noqa: E402
    add_labeled_row,
    clear_layout,
    make_checkbox,
    repopulate_combo,
    set_option_enabled,
)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class RepopulateComboTest(unittest.TestCase):
    def setUp(self):
        self.combo = QtWidgets.QComboBox()
        self.emitted = []
        self.combo.currentIndexChanged.connect(self.emitted.append)

    def items(self):
        return [(self.combo.itemText(i), self.combo.itemData(i)) for i in range(self.combo.count())]

    def test_first_item_then_names_with_tooltips(self):
        repopulate_combo(self.combo, "World", ["a", "b"])
        self.assertEqual(self.items(), [("World", None), ("a", "a"), ("b", "b")])
        self.assertEqual(self.combo.itemData(2, QtCore.Qt.ItemDataRole.ToolTipRole), "b")
        self.assertEqual(self.emitted, [])

    def test_keeps_a_selection_that_is_still_offered(self):
        repopulate_combo(self.combo, "World", ["a", "b"])
        self.combo.setCurrentIndex(2)
        self.assertFalse(repopulate_combo(self.combo, "World", ["b", "c"]))
        self.assertEqual(self.combo.currentData(), "b")

    def test_falls_back_to_the_first_item_and_reports_the_change(self):
        repopulate_combo(self.combo, "World", ["a", "b"])
        self.combo.setCurrentIndex(1)
        self.emitted.clear()
        self.assertTrue(repopulate_combo(self.combo, "World", ["c"]))
        self.assertIsNone(self.combo.currentData())
        self.assertEqual(self.emitted, [])


class LayoutHelpersTest(unittest.TestCase):
    def test_clear_layout_empties_nested_rows(self):
        parent = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(parent)
        add_labeled_row(layout, "Axis", QtWidgets.QComboBox())
        layout.addWidget(make_checkbox("x"))
        clear_layout(layout)
        self.assertEqual(layout.count(), 0)

    def test_set_option_enabled_applies_to_every_widget(self):
        widgets = (QtWidgets.QLabel("Resolution"), QtWidgets.QComboBox())
        set_option_enabled(widgets, False, "Not used")
        self.assertEqual([(w.isEnabled(), w.toolTip()) for w in widgets], [(False, "Not used")] * 2)


if __name__ == "__main__":
    unittest.main()
