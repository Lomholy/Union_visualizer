"""Reusable widgets for the viewer window: collapsible dock title bars, a
colour bar and the status bar's loading indicator."""

import time

from qtpy import QtCore, QtGui, QtWidgets

from gui_helpers import VIRIDIS_STOPS


class CollapsibleTitleBar(QtWidgets.QWidget):
    """Dock title bar whose arrow (or a double-click on the title)
    collapses the dock to just this bar, so the other docks get the room.
    A dock starts as collapsed_by_default until the user toggles it; after
    that, their choice is remembered in settings."""

    def __init__(self, dock, settings, on_toggled=None, collapsed_by_default=True):
        super().__init__(dock)
        self.dock = dock
        self.settings = settings
        self.on_toggled = on_toggled

        # Hiding the dock's own widget would also cap the dock's width at
        # this title bar's, so the content is hidden inside a wrapper that
        # stays visible instead.
        self.content = dock.widget()
        wrapper = QtWidgets.QWidget()
        wrapper_layout = QtWidgets.QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(self.content)
        dock.setWidget(wrapper)
        self.settings_key = f"panel_collapsed/{dock.windowTitle()}"

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        self.toggle_button = QtWidgets.QToolButton()
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setToolTip("Collapse or expand this panel")
        self.toggle_button.clicked.connect(lambda: self.set_collapsed(not self.collapsed))
        layout.addWidget(self.toggle_button)

        title = QtWidgets.QLabel(dock.windowTitle())
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title, 1)

        float_button = QtWidgets.QToolButton()
        float_button.setAutoRaise(True)
        float_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_TitleBarNormalButton)
        )
        float_button.setToolTip("Undock or re-dock this panel")
        float_button.clicked.connect(lambda: dock.setFloating(not dock.isFloating()))
        layout.addWidget(float_button)

        self.collapsed = False
        self.set_collapsed(
            settings.value(self.settings_key, collapsed_by_default, type=bool), notify=False
        )

    def set_collapsed(self, collapsed, notify=True):
        self.collapsed = collapsed
        self.content.setVisible(not collapsed)
        self.dock.setMaximumHeight(
            self.sizeHint().height() if collapsed else QtWidgets.QWIDGETSIZE_MAX
        )
        self.toggle_button.setArrowType(
            QtCore.Qt.ArrowType.RightArrow if collapsed else QtCore.Qt.ArrowType.DownArrow
        )
        if notify:
            self.settings.setValue(self.settings_key, collapsed)
            if self.on_toggled is not None:
                self.on_toggled()

    def mouseDoubleClickEvent(self, event):
        self.set_collapsed(not self.collapsed)


class ColorBarWidget(QtWidgets.QWidget):
    """A horizontal viridis gradient with its low/high value labelled at
    each end, for the ray colouring modes. Paints the same VIRIDIS_STOPS
    gui_helpers.colormap() interpolates, so the bar always matches the
    colours actually drawn on the rays."""

    def __init__(self):
        super().__init__()
        self.setFixedHeight(28)
        self.low_text = ""
        self.high_text = ""

    def set_range(self, low_text, high_text):
        self.low_text = low_text
        self.high_text = high_text
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        bar_rect = self.rect().adjusted(0, 14, 0, 0)

        gradient = QtGui.QLinearGradient(bar_rect.left(), 0, bar_rect.right(), 0)
        for i, rgb in enumerate(VIRIDIS_STOPS):
            gradient.setColorAt(
                i / (len(VIRIDIS_STOPS) - 1),
                QtGui.QColor.fromRgbF(*(float(c) for c in rgb)),
            )
        painter.fillRect(bar_rect, gradient)
        painter.setPen(QtGui.QColor("#888"))
        painter.drawRect(bar_rect.adjusted(0, 0, -1, -1))

        painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.WindowText))
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -bar_rect.height()),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop,
            self.low_text,
        )
        painter.drawText(
            self.rect().adjusted(0, 0, 0, -bar_rect.height()),
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop,
            self.high_text,
        )


class LoadingIndicator(QtCore.QObject):
    """A spinning icon and an elapsed-time label, added to window's status
    bar, shown while set_loading(True) is in effect."""

    ICON_SIZE = 24

    def __init__(self, window, text="Building meshes"):
        super().__init__(window)
        self.text = text
        self.base_pixmap = window.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_BrowserReload
        ).pixmap(16, 16)
        self.spin_angle = 0
        self.start_time = None

        # A rotated 16x16 pixmap's bounding box grows to ~22x22 for any
        # angle that isn't a multiple of 90 degrees (QPixmap.transformed()
        # enlarges the pixmap to fit the rotated content). Painting into a
        # fixed-size canvas instead of using that pixmap's own size keeps
        # the label's geometry constant every tick - otherwise the label
        # (and the status bar layout around it) resizes ~17 times/sec as
        # the icon spins, which is what actually caused the flicker.
        self.icon_label = QtWidgets.QLabel()
        self.icon_label.setFixedSize(self.ICON_SIZE, self.ICON_SIZE)
        self.icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.icon_label.setPixmap(self.base_pixmap)
        self.text_label = QtWidgets.QLabel(f"{text}...")
        # Fixed width sized for the longest plausible elapsed time, so the
        # growing digit count doesn't resize the label (and reflow the
        # status bar) every tick, for the same reason as the icon's canvas.
        self.text_label.setFixedWidth(
            self.text_label.fontMetrics().horizontalAdvance(f"{text} for 9999.99 seconds")
        )

        window.statusBar().addWidget(self.icon_label)
        window.statusBar().addWidget(self.text_label)
        self.icon_label.hide()
        self.text_label.hide()

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(60)
        self.timer.timeout.connect(self._spin)

    def set_loading(self, is_loading):
        self.icon_label.setVisible(is_loading)
        self.text_label.setVisible(is_loading)
        if is_loading:
            self.start_time = time.time()
            self.text_label.setText(f"{self.text} for 0.00 seconds")
            self.timer.start()
        else:
            self.timer.stop()
            self.icon_label.setPixmap(self.base_pixmap)

    def _spin(self):
        self.spin_angle = (self.spin_angle + 30) % 360

        size = self.ICON_SIZE
        canvas = QtGui.QPixmap(size, size)
        canvas.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(size / 2, size / 2)
        painter.rotate(self.spin_angle)
        base = self.base_pixmap
        painter.drawPixmap(QtCore.QPointF(-base.width() / 2, -base.height() / 2), base)
        painter.end()

        self.icon_label.setPixmap(canvas)

        elapsed = time.time() - self.start_time
        self.text_label.setText(f"{self.text} for {elapsed:.2f} seconds")
