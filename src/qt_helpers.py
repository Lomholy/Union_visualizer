"""Builders for the small Qt widgets and layouts the viewer's docks repeat."""

from qtpy import QtCore, QtGui, QtWidgets


def add_labeled_row(layout, text, widget):
    """Append a row [QLabel(text), widget] to layout; returns the label."""
    row = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel(text)
    row.addWidget(label)
    row.addWidget(widget)
    layout.addLayout(row)
    return label


def add_row(layout, *widgets, stretch=False):
    """Append a row of widgets to layout, optionally followed by a stretch."""
    row = QtWidgets.QHBoxLayout()
    for widget in widgets:
        row.addWidget(widget)
    if stretch:
        row.addStretch()
    layout.addLayout(row)
    return row


def make_checkbox(text, checked=False, tooltip=None):
    checkbox = QtWidgets.QCheckBox(text)
    checkbox.setChecked(checked)
    if tooltip:
        checkbox.setToolTip(tooltip)
    return checkbox


def make_button(text, tooltip=None, on_click=None):
    button = QtWidgets.QPushButton(text)
    if tooltip:
        button.setToolTip(tooltip)
    if on_click is not None:
        button.clicked.connect(on_click)
    return button


def make_note_label(text="", italic=False):
    """A word-wrapped grey label for status and help text."""
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: gray;" + (" font-style: italic;" if italic else ""))
    return label


def make_double_spin(minimum, maximum, value, decimals, step=None, tooltip=None):
    spin = QtWidgets.QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(minimum, maximum)
    if step is not None:
        spin.setSingleStep(step)
    spin.setValue(value)
    if tooltip:
        spin.setToolTip(tooltip)
    return spin


def make_elided_combo(min_chars=12):
    """A combo box capped at min_chars characters wide, eliding longer
    entries. A long component name would otherwise widen the combo box, and
    with it the whole left-hand column; repopulate_combo() keeps each full
    name available as its item's tooltip."""
    combo = QtWidgets.QComboBox()
    combo.setSizeAdjustPolicy(
        QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
    )
    combo.setMinimumContentsLength(min_chars)
    return combo


def repopulate_combo(combo, first_text, names):
    """Replace combo's items with first_text (data None) followed by one
    item per name (data and tooltip both the name), without emitting
    signals. Keeps the previous selection if it is still offered, else
    selects first_text. Returns whether the selected data changed."""
    current = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem(first_text, None)
    for name in names:
        combo.addItem(name, name)
        combo.setItemData(combo.count() - 1, name, QtCore.Qt.ItemDataRole.ToolTipRole)
    combo.setCurrentIndex(max(combo.findData(current), 0))
    combo.blockSignals(False)
    return combo.currentData() != current


def set_swatch_color(button, hex_color):
    button.setStyleSheet(f"background-color: {hex_color}; border: 1px solid #888;")


def make_swatch_button(size, hex_color, tooltip, on_click):
    """A small square button filled with hex_color."""
    button = QtWidgets.QPushButton()
    button.setFixedSize(size, size)
    button.setToolTip(tooltip)
    set_swatch_color(button, hex_color)
    button.clicked.connect(lambda checked=False: on_click())
    return button


def ask_color(parent, current, title):
    """Open a colour dialog starting at hex colour current. Returns the
    chosen hex colour, or None if cancelled."""
    color = QtWidgets.QColorDialog.getColor(QtGui.QColor(current), parent, title)
    return color.name() if color.isValid() else None


def set_option_enabled(widgets, enabled, tooltip):
    """Enable or disable widgets together and give them all tooltip."""
    for widget in widgets:
        widget.setEnabled(enabled)
        widget.setToolTip(tooltip)


def clear_layout(layout):
    """Remove every item from layout, deleting its widgets and emptying
    nested layouts."""
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            clear_layout(item.layout())
