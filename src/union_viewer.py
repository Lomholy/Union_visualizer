import argparse
import sys
import time
from pathlib import Path
import traceback
import numpy as np
import trimesh
import pygfx as gfx
from qtpy import QtWidgets, QtCore, QtGui
from rendercanvas.qt import QRenderWidget
from pygfx.utils.viewport import Viewport
from clipping import resolve_clip_frame
from meshing import MESHER_CAPABILITIES
from gui_helpers import (
    component_color_key,
    clip_planes,
    instrument_param_args,
    clip_mesh,
    RAY_COLOR_MODES,
    rays_reaching,
    wrap_label,
    VIRIDIS_STOPS,
)
import logger_output
from pipeline import NO_CLIP, compute_counts_data, compute_mesh_data, compute_trace_data
from scene_objects import (
    DEFAULT_MARKER_SIZES,
    DEFAULT_RAY_COLORS,
    GRID_SPACING,
    MAX_GRID_DIVISIONS,
    add_default_lights,
    build_component_group,
    build_counts_group,
    build_gfx_group,
    build_ray_group,
    grid_size_for_bbox,
    make_coordinate_axes,
)
from background import JobRunner
from qt_helpers import (
    add_labeled_row,
    add_row,
    ask_color,
    clear_layout,
    make_button,
    make_checkbox,
    make_double_spin,
    make_elided_combo,
    make_note_label,
    make_swatch_button,
    repopulate_combo,
    set_option_enabled,
    set_swatch_color,
)
from camera import fit_camera_to_scene, recentre_controller, update_camera_depth_range

# The meshers offered in the dock, in display order.
MESHER_KEYS = ("mc", "dc", "brep")

MESHER_DESCRIPTIONS = {
    "mc": (
        "Marching Cubes: samples the signed distance field on a uniform grid. "
        "Fast, but curved surfaces are limited by the chosen resolution."
    ),
    "dc": (
        "Dual Contouring: builds an adaptive octree from the point cloud. "
        "Better at preserving sharp edges/corners than Marching Cubes."
    ),
    "brep": (
        "BREP (exact CAD): uses OpenCASCADE boundary representation for exact, "
        "resolution-independent geometry. Most accurate, but can be slower for "
        "complex boolean operations."
    ),
}

# The marching-cubes grid resolutions offered in the dock.
RESOLUTIONS = (16, 32, 64, 128, 256, 512)
DEFAULT_RESOLUTION = 64

DEFLECTION_TOOLTIP = (
    "How closely the brep mesher's triangles must follow the exact curved "
    "surface, in metres. Smaller values hug curves more closely (more "
    "triangles, slower to build); larger values are coarser but faster."
)


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


# ============================================================
# Main window
# ============================================================


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, input_file=None, settings=None):
        super().__init__()
        self.settings = settings or QtCore.QSettings("unviz", "union_viewer")
        self.setWindowTitle("Union Viewer")
        self.resize(1400, 900)
        self.colors = {}
        self.input_file = input_file
        self.start_input = bool(input_file)
        self.last_mtime = None
        self.current_group = None
        self.dependencies = None
        self.meshes = None
        self.mesher = "brep"

        # ----------------------------------------------------
        # Background jobs
        # ----------------------------------------------------
        # Meshing, mcrun --trace and counts loading each get their own
        # worker process, so a slow mcrun compile never delays the meshes,
        # a mesher/clip change never reruns McStas, and loading a large
        # logger file never delays either.
        self.mesh_jobs = JobRunner(
            self, self._on_reload_finished, self._on_reload_failed, self._on_mesh_jobs_idle
        )
        self.trace_jobs = JobRunner(self, self._on_trace_finished, self._on_trace_failed)
        self.counts_jobs = JobRunner(
            self, self._on_counts_finished, self._on_counts_failed, self._on_counts_jobs_idle
        )
        self._reload_pending_force = False
        self._pending_fit_camera = False

        self.trace_group = None
        self.trace_rays = None
        self.ray_group = None

        # ----------------------------------------------------
        # Detector counts
        # ----------------------------------------------------
        self.counts = {}              # {name: logger_output.LoggerCounts}
        self.counts_group = None
        self.counts_meshes = {}       # {name: gfx.Mesh}
        self.counts_textures = {}     # {name: gfx.Texture}
        self.counts_visibility = {}   # {name: bool}
        self.counts_checkboxes = {}
        self.counts_run_folder = None
        self._loading_counts_folder = None

        # ----------------------------------------------------
        # Render widget
        # ----------------------------------------------------
        self.canvas = QRenderWidget()
        self.setCentralWidget(self.canvas)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        # ----------------------------------------------------
        # Scene
        # ----------------------------------------------------
        self.scene = gfx.Scene()
        add_default_lights(self.scene)
        # ----------------------------------------------------
        # Camera
        # ----------------------------------------------------
        self.camera = gfx.PerspectiveCamera(35)
        self.camera.local.position = (0, 1, 10)
        self.camera.look_at((0, 0, 0))
        self.controller = gfx.OrbitController(
            self.camera,
            register_events=self.renderer,
        )
        self.controller.target = (0, 0, 0)

        update_camera_depth_range(self.camera, self.controller.target)
        self.scene.add(make_coordinate_axes(length=1000, tick_spacing=1000))

        self.grid = None
        self._resize_grid(None)

        self.gizmo_viewport = Viewport(self.renderer, (0, 0, 120, 120))
        self.gizmo_scene = gfx.Scene()
        self.gizmo = make_coordinate_axes(length=1.0, tick_spacing=0.5, tick_size=0.05)
        self.gizmo_scene.add(self.gizmo)
        self.gizmo_camera = gfx.PerspectiveCamera(50, 1)
        self.gizmo_camera.local.position = (0, 0, 4)

        # ----------------------------------------------------
        # Open file shortcut (the "Open File..." button lives in the
        # Settings dock; this just keeps Ctrl+O working)
        # ----------------------------------------------------
        self.open_file_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Open, self
        )
        self.open_file_shortcut.activated.connect(self.open_file)
        # ----------------------------------------------------
        # Loading indicator (status bar)
        # ----------------------------------------------------
        self._loading_base_pixmap = self.style().standardIcon(
            QtWidgets.QStyle.StandardPixmap.SP_BrowserReload
        ).pixmap(16, 16)
        self._loading_spin_angle = 0
        # A rotated 16x16 pixmap's bounding box grows to ~22x22 for any
        # angle that isn't a multiple of 90 degrees (QPixmap.transformed()
        # enlarges the pixmap to fit the rotated content). Painting into a
        # fixed-size canvas instead of using that pixmap's own size keeps
        # the label's geometry constant every tick - otherwise the label
        # (and the status bar layout around it) resizes ~17 times/sec as
        # the icon spins, which is what actually caused the flicker.
        self._loading_icon_size = 24

        self.loading_icon_label = QtWidgets.QLabel()
        self.loading_icon_label.setFixedSize(
            self._loading_icon_size, self._loading_icon_size
        )
        self.loading_icon_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.loading_icon_label.setPixmap(self._loading_base_pixmap)
        self._loading_start_time = None
        self.loading_text_label = QtWidgets.QLabel("Building meshes...")
        # Fixed width sized for the longest plausible elapsed time, so the
        # growing digit count doesn't resize the label (and reflow the
        # status bar) every tick the way the un-fixed-size spinner icon
        # used to - see _spin_loading_icon()'s canvas-painting comment.
        self.loading_text_label.setFixedWidth(
            self.loading_text_label.fontMetrics().horizontalAdvance(
                "Building meshes for 9999.99 seconds"
            )
        )

        self.statusBar().addWidget(self.loading_icon_label)
        self.statusBar().addWidget(self.loading_text_label)
        self.loading_icon_label.hide()
        self.loading_text_label.hide()

        self.loading_spin_timer = QtCore.QTimer(self)
        self.loading_spin_timer.setInterval(60)
        self.loading_spin_timer.timeout.connect(self._spin_loading_icon)
        # ----------------------------------------------------
        # Render timer
        # ----------------------------------------------------
        self.render_timer = QtCore.QTimer()
        self.render_timer.timeout.connect(self.animate)
        self.render_timer.start(16)
        # ----------------------------------------------------
        # File watcher timer
        # ----------------------------------------------------
        self.watch_timer = QtCore.QTimer()
        self.watch_timer.timeout.connect(self.check_file_update)
        self.watch_timer.start(100)

        # ----------------------------------------------------
        # Clipping state
        # ----------------------------------------------------

        self.clip = {
            "enable": False,
            "axis": "Z",
            "mode": "Above",
            "position": 0.0,
            "frame": None,
        }
        self.world_matrices = {}

        # Docks are added in this order (Settings, Instrument Parameters,
        # Mesher Options, Clipping, Neutron Rays, Detector Counts, Visible
        # Geometries) so they stack top-to-bottom in the left panel.

        # ----------------------------------------------------
        # Settings dock
        # ----------------------------------------------------

        settings_layout = self._add_dock("Settings", collapsed=False)

        self.open_file_button = make_button(
            "Open File...", "Open a McStas instrument (shortcut: Ctrl+O)", self.open_file
        )
        self.material_group_checkbox = make_checkbox(
            "Group by material",
            True,
            "Combine components that share a material into a single "
            "rendered object (trimesh concatenation, not a boolean fusion).",
        )
        self.vacuum_checkbox = make_checkbox(
            "Hide vacuum",
            True,
            "Hides volumes whose material is 'vacuum'/'Vacuum' or "
            "'exit'/'Exit' (McStas treats 'exit' as vacuum too).",
        )
        self.pygen_checkbox = make_checkbox(
            "Force mcstas-pygen preprocessing",
            False,
            "Translate a .instr input through the real McStas front-end "
            "(mcstas-pygen) instead of mcstasscript's lightweight .instr "
            "reader. mcstas-pygen already runs automatically whenever that "
            "lightweight reader raises an error - this instead forces it "
            "for every load, for instruments the lightweight reader "
            "mis-parses without raising an error. Requires mcstas-pygen "
            "on PATH (ships with the McStas install).",
        )
        self.components_checkbox = make_checkbox(
            "Show McStas components",
            True,
            "Draw every non-Union component the way McStas's own MCDISPLAY "
            "draws it (no priority cutting). Compiles and runs the "
            "instrument with mcrun --trace, so it needs a working McStas "
            "install and compiler.",
        )
        self.arms_checkbox = make_checkbox(
            "Show Arms", False, "Arms only mark coordinate frames."
        )
        self.rays_checkbox = make_checkbox(
            "Show neutron rays",
            False,
            "Trace neutrons through the instrument with mcrun --trace and "
            "draw their paths. Options are in the Neutron Rays panel.",
        )
        self.trace_status_label = make_note_label()
        self.reset_view_button = make_button(
            "Reset view", "Refit the camera (shortcut: R)", self.reset_view
        )
        self.fit_instrument_button = make_button(
            "Fit whole instrument",
            "Fit the camera to every McStas component, not just the Union geometry.",
            self.fit_whole_instrument,
        )
        self.export_stl_button = make_button(
            "Export STL...",
            "Export the visible Union meshes and McStas components as a "
            "single .stl file, cut by the clipping plane if enabled. "
            "McStas lines are exported as thin tubes.",
            self.export_stl,
        )
        for widget in (
            self.open_file_button,
            self.material_group_checkbox,
            self.vacuum_checkbox,
            self.pygen_checkbox,
            self.components_checkbox,
            self.arms_checkbox,
            self.rays_checkbox,
            self.trace_status_label,
            self.reset_view_button,
            self.fit_instrument_button,
            self.export_stl_button,
        ):
            settings_layout.addWidget(widget)
        self.reset_view_shortcut = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        self.reset_view_shortcut.activated.connect(self.reset_view)

        # ----------------------------------------------------
        # Instrument parameters dock
        # ----------------------------------------------------

        params_layout = self._add_dock("Instrument Parameters", stretch=False)
        self.param_values = {}
        self.param_edits = {}
        self.param_names = None

        params_widget = QtWidgets.QWidget()
        self.params_form = QtWidgets.QFormLayout(params_widget)
        self.params_empty_label = make_note_label("No instrument loaded.")
        params_layout.addWidget(self.params_empty_label)

        # Scrollable, and capped at a fixed height rather than growing with
        # the parameter count - an instrument can have many parameters, and
        # this keeps the rest of the panels from being pushed off-screen.
        params_scroll = QtWidgets.QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setWidget(params_widget)
        params_scroll.setMaximumHeight(220)
        params_layout.addWidget(params_scroll)

        # ----------------------------------------------------
        # Mesher options dock
        # ----------------------------------------------------

        mesher_options_layout = self._add_dock("Mesher Options")

        self.mesher_box = QtWidgets.QComboBox()
        for key in MESHER_KEYS:
            label = key
            if not MESHER_CAPABILITIES[key]["incremental_rebuild"]:
                label += " (full rebuild only)"
            self.mesher_box.addItem(label, key)
        self.mesher_box.setCurrentIndex(self.mesher_box.findData(self.mesher))
        add_labeled_row(mesher_options_layout, "Mesher", self.mesher_box)

        self.mesher_description_label = make_note_label(
            MESHER_DESCRIPTIONS.get(self.mesher, ""), italic=True
        )
        mesher_options_layout.addWidget(self.mesher_description_label)

        self.res_val = QtWidgets.QComboBox()
        for res in RESOLUTIONS:
            self.res_val.addItem(str(res), res)
        self.res_val.setCurrentIndex(self.res_val.findData(DEFAULT_RESOLUTION))
        self.resolution_label = add_labeled_row(
            mesher_options_layout, "Resolution", self.res_val
        )

        self.deflection_val = make_double_spin(
            0.0001, 10.0, 0.01, decimals=4, step=0.001, tooltip=DEFLECTION_TOOLTIP
        )
        self.deflection_label = add_labeled_row(
            mesher_options_layout, "Surface deflection", self.deflection_val
        )
        self.deflection_label.setToolTip(DEFLECTION_TOOLTIP)

        # ----------------------------------------------------
        # Clipping dock
        # ----------------------------------------------------

        clipping_layout = self._add_dock("Clipping")

        self.clip_checkbox = make_checkbox("Enable clipping")
        clipping_layout.addWidget(self.clip_checkbox)

        self.clip_frame_combo = make_elided_combo()
        self.clip_frame_combo.addItem("World", None)
        self.clip_frame_combo.setToolTip(
            "Axis and position are taken in this component's own coordinate "
            "system (its AT/ROTATED frame), e.g. the sample's Arm."
        )
        add_labeled_row(clipping_layout, "Coordinate system", self.clip_frame_combo)

        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["X", "Y", "Z"])
        add_labeled_row(clipping_layout, "Axis", self.axis_combo)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Above", "Below"])
        add_labeled_row(clipping_layout, "Mode", self.mode_combo)

        self.slice_val = make_double_spin(-1e6, 1e6, 0.0, decimals=5, step=0.01)
        add_labeled_row(clipping_layout, "Position", self.slice_val)

        # ----------------------------------------------------
        # Neutron rays dock
        # ----------------------------------------------------

        rays_layout = self._add_dock("Neutron Rays")
        self.rays_options = QtWidgets.QWidget()
        options_layout = QtWidgets.QVBoxLayout(self.rays_options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        rays_layout.addWidget(self.rays_options)

        self.ray_count_val = QtWidgets.QSpinBox()
        self.ray_count_val.setRange(1, 100_000_000)
        self.ray_count_val.setValue(50)
        self.ray_count_val.setToolTip(
            "Trace mode is single-threaded and verbose - keep this small; "
            "McStas itself has no lower limit worth mentioning."
        )
        add_labeled_row(options_layout, "Number of rays", self.ray_count_val)

        self.ray_seed_val = QtWidgets.QSpinBox()
        self.ray_seed_val.setRange(0, 2**31 - 1)
        self.ray_seed_val.setSpecialValueText("random")
        self.ray_seed_val.setToolTip("0 picks a new random seed on every run.")
        add_labeled_row(options_layout, "Seed", self.ray_seed_val)

        self.rerun_rays_button = make_button(
            "Re-run", "Trace a new set of rays.", self.rerun_rays
        )
        add_row(options_layout, self.rerun_rays_button)

        self.ray_color_combo = QtWidgets.QComboBox()
        self.ray_color_combo.addItems(list(RAY_COLOR_MODES))
        add_labeled_row(options_layout, "Colour by", self.ray_color_combo)

        self.ray_colorbar = ColorBarWidget()
        self.ray_colorbar.hide()
        options_layout.addWidget(self.ray_colorbar)

        # Label above the combo, not beside it, so a long selected/eliding
        # entry doesn't push the row (and the panel) wider.
        options_layout.addWidget(QtWidgets.QLabel("Only rays reaching"))
        self.ray_reaching_combo = make_elided_combo()
        self.ray_reaching_combo.addItem("any component", None)
        options_layout.addWidget(self.ray_reaching_combo)

        self.ray_colors = dict(DEFAULT_RAY_COLORS)
        self.ray_marker_sizes = dict(DEFAULT_MARKER_SIZES)
        self.ray_color_buttons = {}
        self.ray_size_spins = {}

        self.rays_line_checkbox = make_checkbox(
            "Show rays",
            True,
            "The ordinary path each ray follows. The colour applies when "
            "colouring by Uniform.",
        )
        add_row(options_layout, self.rays_line_checkbox, self._ray_color_button("ray"), stretch=True)

        self.teleport_line_checkbox = make_checkbox(
            "Show teleports",
            True,
            "The jump a restore_neutron monitor (e.g. PSD_monitor) causes: "
            "it detects a ray, then restores its pre-detection state.",
        )
        add_row(
            options_layout,
            self.teleport_line_checkbox,
            self._ray_color_button("teleport"),
            stretch=True,
        )

        self.scatter_points_checkbox = make_checkbox(
            "Mark scatterings",
            True,
            "Points where a ray changed direction. Union volume boundary "
            "crossings are not marked.",
        )
        add_row(
            options_layout,
            self.scatter_points_checkbox,
            self._ray_color_button("scatter"),
            self._marker_size_spin("scatter"),
            stretch=True,
        )

        self.absorb_points_checkbox = make_checkbox("Mark absorptions", True)
        add_row(
            options_layout,
            self.absorb_points_checkbox,
            self._ray_color_button("absorb"),
            self._marker_size_spin("absorb"),
            stretch=True,
        )

        self.ray_info_label = make_note_label()
        options_layout.addWidget(self.ray_info_label)

        self.ray_rerun_timer = QtCore.QTimer(self)
        self.ray_rerun_timer.setSingleShot(True)
        self.ray_rerun_timer.setInterval(700)
        self.ray_rerun_timer.timeout.connect(self.rerun_rays)

        # ----------------------------------------------------
        # Detector counts dock
        # ----------------------------------------------------

        counts_layout = self._add_dock("Detector Counts")
        self.counts_options = QtWidgets.QWidget()
        counts_options_layout = QtWidgets.QVBoxLayout(self.counts_options)
        counts_options_layout.setContentsMargins(0, 0, 0, 0)
        counts_layout.addWidget(self.counts_options)

        self.load_counts_button = make_button(
            "Load counts folder...",
            "A McStas run's output folder (containing mccode.sim). Every "
            "detector in it that can be placed in 3D - a Union logger/"
            "abs_logger, an ordinary monitor (PSD_monitor, Monitor_nD, ...), "
            "or an event-list logger with x/y/z columns - is drawn at its "
            "own position: a colour-mapped plane for 2D data, a coloured "
            "line for 1D.",
            self.load_counts_folder,
        )
        counts_options_layout.addWidget(self.load_counts_button)

        self.counts_checkbox = make_checkbox("Show detector counts", True)
        self.counts_checkbox.stateChanged.connect(self.apply_counts_visibility)
        counts_options_layout.addWidget(self.counts_checkbox)

        range_tooltip = "Editable - overrides the automatic min/max for every visible logger's plane."
        self.counts_min_val = make_double_spin(-1e12, 1e12, 0.0, decimals=4, tooltip=range_tooltip)
        self.counts_max_val = make_double_spin(-1e12, 1e12, 0.0, decimals=4, tooltip=range_tooltip)
        add_row(
            counts_options_layout,
            QtWidgets.QLabel("Colour range"),
            self.counts_min_val,
            self.counts_max_val,
        )
        self.counts_min_val.valueChanged.connect(self.on_counts_range_changed)
        self.counts_max_val.valueChanged.connect(self.on_counts_range_changed)

        self.counts_auto_range_button = make_button(
            "Auto range",
            "Reset the colour range to the min/max of the currently visible loggers.",
            self.reset_counts_range,
        )
        counts_options_layout.addWidget(self.counts_auto_range_button)

        self.counts_colorbar = ColorBarWidget()
        self.counts_colorbar.hide()
        counts_options_layout.addWidget(self.counts_colorbar)

        self.counts_panel_layout = QtWidgets.QVBoxLayout()
        counts_options_layout.addLayout(self.counts_panel_layout)

        self.counts_info_label = make_note_label("No counts loaded.")
        counts_options_layout.addWidget(self.counts_info_label)

        # ----------------------------------------------------
        # Geometry visibility dock
        # ----------------------------------------------------

        self.geometry_checkboxes = {}
        self.geometry_color_buttons = {}
        self.geometry_visibility = {}

        self.geometry_dock = QtWidgets.QDockWidget("Visible Geometries", self)
        geometry_dock_widget = QtWidgets.QWidget()
        geometry_dock_layout = QtWidgets.QVBoxLayout(geometry_dock_widget)

        self.geometry_filter = QtWidgets.QLineEdit()
        self.geometry_filter.setPlaceholderText("Filter...")
        self.geometry_filter.setClearButtonEnabled(True)
        self.geometry_filter.textChanged.connect(self.on_geometry_filter_changed)
        geometry_dock_layout.addWidget(self.geometry_filter)

        self.show_all_button = make_button(
            "Show all", on_click=lambda: self.set_all_geometry_visibility(True)
        )
        self.hide_all_button = make_button(
            "Hide all", on_click=lambda: self.set_all_geometry_visibility(False)
        )
        add_row(geometry_dock_layout, self.show_all_button, self.hide_all_button)

        self.component_checkboxes = {}
        self.component_color_buttons = {}
        self.component_visibility = {}

        self.geometry_widget = QtWidgets.QWidget()
        panel_layout = QtWidgets.QVBoxLayout(self.geometry_widget)
        self.union_header = QtWidgets.QLabel("<b>Union components</b>")
        self.union_header.hide()
        panel_layout.addWidget(self.union_header)
        self.geometry_layout = QtWidgets.QVBoxLayout()
        panel_layout.addLayout(self.geometry_layout)
        self.component_header = QtWidgets.QLabel("<b>McStas components</b>")
        self.component_header.hide()
        panel_layout.addWidget(self.component_header)
        self.component_layout = QtWidgets.QVBoxLayout()
        panel_layout.addLayout(self.component_layout)
        panel_layout.addStretch()

        geometry_scroll = QtWidgets.QScrollArea()
        geometry_scroll.setWidgetResizable(True)
        geometry_scroll.setWidget(self.geometry_widget)
        geometry_dock_layout.addWidget(geometry_scroll)

        self.geometry_dock.setWidget(geometry_dock_widget)
        geometry_dock_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )

        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            self.geometry_dock,
        )
        self.geometry_dock.setTitleBarWidget(
            CollapsibleTitleBar(self.geometry_dock, self.settings, self.rebalance_docks)
        )
        QtCore.QTimer.singleShot(0, self.rebalance_docks)

        # ----------------------------------------------------
        # Signals
        # ----------------------------------------------------

        self.clip_checkbox.stateChanged.connect(self.on_clip_changed)
        self.material_group_checkbox.stateChanged.connect(self.on_material_group_changed)
        self.vacuum_checkbox.stateChanged.connect(self.on_vacuum_changed)
        self.mesher_box.currentIndexChanged.connect(self.on_mesher_changed)
        self.pygen_checkbox.stateChanged.connect(self.on_pygen_changed)
        self.components_checkbox.stateChanged.connect(self.on_components_changed)
        self.arms_checkbox.stateChanged.connect(self.apply_component_visibility)
        self.rays_checkbox.stateChanged.connect(self.on_rays_changed)
        self.ray_count_val.valueChanged.connect(self.ray_rerun_timer.start)
        self.ray_seed_val.valueChanged.connect(self.ray_rerun_timer.start)
        self.ray_color_combo.currentIndexChanged.connect(self.rebuild_ray_group)
        self.ray_reaching_combo.currentIndexChanged.connect(self.rebuild_ray_group)
        self.rays_line_checkbox.stateChanged.connect(self.apply_ray_visibility)
        self.teleport_line_checkbox.stateChanged.connect(self.apply_ray_visibility)
        self.scatter_points_checkbox.stateChanged.connect(self.apply_ray_visibility)
        self.absorb_points_checkbox.stateChanged.connect(self.apply_ray_visibility)
        self.axis_combo.currentTextChanged.connect(self.on_clip_changed)
        self.clip_frame_combo.currentIndexChanged.connect(self.on_clip_changed)
        self.mode_combo.currentTextChanged.connect(self.on_clip_changed)
        self.slice_val.valueChanged.connect(self.on_clip_changed)
        self.res_val.currentIndexChanged.connect(self.on_res_changed)
        self.deflection_val.valueChanged.connect(self.on_deflection_changed)

        self.update_mesher_capability_ui()
        self.rays_options.setEnabled(self.rays_checkbox.isChecked())

    def _add_dock(self, title, collapsed=True, stretch=True):
        """Create a collapsible, left-docked QDockWidget titled `title`
        (collapsed at first unless collapsed=False) that only takes the
        height its contents need, and return the layout for the caller to
        populate. With stretch=True, extra space goes below the contents."""
        dock = QtWidgets.QDockWidget(title, self)
        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )
        widget = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(widget)
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        outer.addLayout(layout)
        if stretch:
            outer.addStretch()
        dock.setWidget(widget)
        dock.setTitleBarWidget(
            CollapsibleTitleBar(dock, self.settings, self.rebalance_docks, collapsed)
        )
        dock.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Maximum,
        )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        return layout

    def rebalance_docks(self):
        """Hand the height collapsed docks free up to the expanded ones:
        each expanded dock gets its natural height and the Visible
        Geometries dock takes whatever is left."""
        docks = [
            d for d in self.findChildren(QtWidgets.QDockWidget)
            if d.isVisibleTo(self)
            and not d.isFloating()
            and self.dockWidgetArea(d) == QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        ]
        if not docks:
            return
        sizes = [
            d.titleBarWidget().sizeHint().height() if d.titleBarWidget().collapsed
            else d.sizeHint().height()
            for d in docks
        ]
        if self.geometry_dock in docks and not self.geometry_dock.titleBarWidget().collapsed:
            sizes[docks.index(self.geometry_dock)] = self.height()
        self.resizeDocks(docks, sizes, QtCore.Qt.Orientation.Vertical)

    # ========================================================
    # Open file dialog
    # ========================================================

    def open_file(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open McStas File",
            "",
            "McStas Files (*.instr *.py);;All Files (*)",
        )
        if not filename:
            return
        self.input_file = filename
        self.last_mtime = Path(filename).stat().st_mtime
        self._pending_fit_camera = True
        self.reload_meshes()
        self.reload_trace()

    def apply_geometry_visibility(self):
        """Recompute mesh.visible for every row from the per-row checkbox
        state and the "Hide vacuum" filter. Never rebuilds geometry - safe
        to call on its own whenever either input changes."""
        if self.current_group is None:
            return
        hide_vacuum = self.vacuum_checkbox.isChecked()
        is_vacuum = self.current_group.geometry_is_vacuum
        for key, mesh in self.current_group.geometry_meshes.items():
            visible = self.geometry_visibility.get(key, True)
            if hide_vacuum and is_vacuum.get(key, False):
                visible = False
            mesh.visible = visible

    def on_geometry_visibility_changed(self, name, checked):
        self.geometry_visibility[name] = checked
        self.apply_geometry_visibility()

    def _add_panel_row(self, layout, name, visible, on_toggled, on_color, color):
        row = QtWidgets.QHBoxLayout()

        cb = make_checkbox(wrap_label(name), visible, name)
        cb.toggled.connect(on_toggled)
        row.addWidget(cb, 1)

        color_button = make_swatch_button(20, color, f"Change colour for '{name}'", on_color)
        row.addWidget(color_button)

        layout.addLayout(row)
        return cb, color_button

    def rebuild_geometry_panel(self):
        clear_layout(self.geometry_layout)

        self.geometry_checkboxes.clear()
        self.geometry_color_buttons.clear()

        if self.current_group is None:
            self.union_header.hide()
            return

        self.union_header.setVisible(bool(self.current_group.geometry_meshes))
        for name in self.current_group.geometry_meshes:
            cb, color_button = self._add_panel_row(
                self.geometry_layout,
                name,
                self.geometry_visibility.get(name, True),
                lambda checked, n=name: self.on_geometry_visibility_changed(n, checked),
                lambda n=name: self.pick_color(n),
                self.colors.get(name, "#b6b6b6"),
            )
            self.geometry_checkboxes[name] = cb
            self.geometry_color_buttons[name] = color_button

        self.on_geometry_filter_changed(self.geometry_filter.text())

    def rebuild_component_panel(self):
        clear_layout(self.component_layout)

        self.component_checkboxes.clear()
        self.component_color_buttons.clear()

        names = [] if self.trace_group is None else list(self.trace_group.component_objects)
        self.component_header.setVisible(bool(names) and self.components_checkbox.isChecked())
        for name in names:
            cb, color_button = self._add_panel_row(
                self.component_layout,
                name,
                self.component_visibility.get(name, True),
                lambda checked, n=name: self.on_component_visibility_changed(n, checked),
                lambda n=name: self.pick_component_color(n),
                self.colors.get(component_color_key(name), "#b6b6b6"),
            )
            self.component_checkboxes[name] = cb
            self.component_color_buttons[name] = color_button

        self.on_geometry_filter_changed(self.geometry_filter.text())

    def on_geometry_filter_changed(self, text):
        text = text.strip().lower()
        show_components = self.components_checkbox.isChecked()
        for checkboxes, buttons, section_visible in (
            (self.geometry_checkboxes, self.geometry_color_buttons, True),
            (self.component_checkboxes, self.component_color_buttons, show_components),
        ):
            for name, cb in checkboxes.items():
                match = section_visible and text in name.lower()
                cb.setVisible(match)
                button = buttons.get(name)
                if button is not None:
                    button.setVisible(match)

    def set_all_geometry_visibility(self, visible):
        for cb in self.geometry_checkboxes.values():
            cb.setChecked(visible)
        for cb in self.component_checkboxes.values():
            cb.setChecked(visible)

    def on_component_visibility_changed(self, name, checked):
        self.component_visibility[name] = checked
        self.apply_component_visibility()

    def apply_component_visibility(self):
        if self.trace_group is None:
            return
        self.trace_group.visible = self.components_checkbox.isChecked()
        show_arms = self.arms_checkbox.isChecked()
        for name, obj in self.trace_group.component_objects.items():
            visible = self.component_visibility.get(name, True)
            if self.trace_group.component_is_arm[name] and not show_arms:
                visible = False
            obj.visible = visible

    def apply_clipping(self):
        planes = clip_planes(self.resolved_clip())
        for group in (self.current_group, self.trace_group, self.ray_group, self.counts_group):
            if group is None:
                continue
            for obj in group.iter():
                if getattr(obj, "material", None) is not None:
                    obj.material.clipping_planes = planes

    def _pick_panel_color(self, name, color_key, objects, button):
        """Ask for a new colour for the panel row `name`, then store it in
        self.colors[color_key] and apply it to objects' materials and the
        row's swatch button."""
        hex_color = ask_color(self, self.colors.get(color_key, "#b6b6b6"), f"Colour for '{name}'")
        if hex_color is None:
            return
        self.colors[color_key] = hex_color
        for obj in objects:
            obj.material.color = hex_color
        if button is not None:
            set_swatch_color(button, hex_color)

    def pick_component_color(self, name):
        if self.trace_group is None or name not in self.trace_group.component_objects:
            return
        self._pick_panel_color(
            name,
            component_color_key(name),
            self.trace_group.component_objects[name].children,
            self.component_color_buttons.get(name),
        )

    def _ray_color_button(self, key):
        """A small colour button that picks the colour of one object in the
        Neutron Rays panel ("ray", "teleport", "scatter" or "absorb")."""
        button = make_swatch_button(
            14, self.ray_colors[key], "Click to change the colour.", lambda: self.pick_ray_color(key)
        )
        self.ray_color_buttons[key] = button
        return button

    def _marker_size_spin(self, key):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(1, 50)
        spin.setSingleStep(1)
        spin.setDecimals(1)
        spin.setSuffix(" px")
        spin.setValue(self.ray_marker_sizes[key])
        spin.setToolTip("Marker size.")
        spin.valueChanged.connect(lambda value: self.set_ray_marker_size(key, value))
        self.ray_size_spins[key] = spin
        return spin

    def pick_color(self, key):
        if self.current_group is None or key not in self.current_group.geometry_meshes:
            return
        self._pick_panel_color(
            key,
            key,
            [self.current_group.geometry_meshes[key]],
            self.geometry_color_buttons.get(key),
        )

    # ========================================================
    # Instrument parameters and clip frame
    # ========================================================

    def rebuild_params_form(self, parameters):
        """One labelled field per instrument parameter, showing its default
        as the placeholder. Values typed earlier are kept by name."""
        names = [name for name, _, _ in parameters]
        if names == self.param_names:
            return
        self.param_names = names
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self.param_edits = {}
        self.param_values = {n: v for n, v in self.param_values.items() if n in names}
        for name, param_type, default in parameters:
            edit = QtWidgets.QLineEdit(self.param_values.get(name, ""))
            edit.setPlaceholderText(default if default is not None else "required")
            edit.setToolTip(
                f"{param_type} {name}"
                + (f" (default {default})" if default is not None else " (no default)")
                + ". Leave empty to use the default."
            )
            edit.editingFinished.connect(
                lambda n=name, e=edit: self.on_param_edited(n, e.text())
            )
            self.params_form.addRow(name, edit)
            self.param_edits[name] = edit
        self.params_empty_label.setText("This instrument has no parameters.")
        self.params_empty_label.setVisible(not names)

    def rebuild_clip_frame_combo(self):
        names = list(self.world_matrices)
        combo = self.clip_frame_combo
        if names == [combo.itemData(i) for i in range(1, combo.count())]:
            return
        if repopulate_combo(combo, "World", names):
            self.on_clip_changed()

    def resolved_clip(self):
        return resolve_clip_frame(self.clip, self.world_matrices)

    def on_param_edited(self, name, text):
        if self.param_values.get(name, "").strip() == text.strip():
            return
        self.param_values[name] = text.strip()
        self.reload_meshes()
        self.reload_trace()

    # ========================================================
    # Export STL
    # ========================================================

    def _export_parts(self):
        """(Union meshes, McStas component meshes, names that failed) for
        what is currently visible, each cut by the clip plane as shown."""
        clip = self.resolved_clip()
        failed = []

        def clipped(name, mesh):
            try:
                return clip_mesh(mesh, clip)
            except Exception:
                traceback.print_exc()
                failed.append(name)
                return None

        union_parts = []
        if self.current_group is not None:
            trimeshes = self.current_group.geometry_trimeshes
            for key, obj in self.current_group.geometry_meshes.items():
                if obj.visible and trimeshes.get(key) is not None:
                    mesh = clipped(key, trimeshes[key])
                    if mesh is not None:
                        union_parts.append(mesh)
        component_parts = []
        if self.trace_group is not None and self.trace_group.visible:
            for name, obj in self.trace_group.component_objects.items():
                if not obj.visible:
                    continue
                mesh = self.trace_group.components[name].export_mesh()
                if mesh is not None:
                    mesh = clipped(name, mesh)
                if mesh is not None:
                    component_parts.append(mesh)
        return union_parts, component_parts, failed

    def export_stl(self):
        if self.current_group is None and self.trace_group is None:
            QtWidgets.QMessageBox.warning(
                self, "Export STL", "No geometry loaded to export."
            )
            return

        try:
            union_parts, component_parts, failed = self._export_parts()
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Export STL", f"Failed to prepare the export:\n{e}"
            )
            return
        parts = union_parts + component_parts
        if not parts:
            QtWidgets.QMessageBox.warning(
                self, "Export STL", "No visible geometry to export."
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export STL", "", "STL Files (*.stl)"
        )
        if not filename:
            return
        if not filename.lower().endswith(".stl"):
            filename += ".stl"

        try:
            combined = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
            combined.export(filename)
        except Exception as e:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(
                self, "Export STL", f"Failed to export STL:\n{e}"
            )
            return

        QtWidgets.QMessageBox.information(
            self,
            "Export STL",
            f"Exported {len(union_parts)} Union mesh(es) and "
            f"{len(component_parts)} McStas component(s) to:\n{filename}"
            + (f"\n\nLeft out (could not be clipped): {', '.join(failed)}" if failed else ""),
        )

    # ========================================================
    # Reload geometry (asynchronous)
    # ========================================================

    def reload_meshes(self, force_reload=False):
        if self.input_file is None:
            return
        # dc has no working incremental rebuild path (build_mesh returns
        # None for it), so always fully rebuild while it's selected.
        if not MESHER_CAPABILITIES[self.mesher]["incremental_rebuild"]:
            force_reload = True

        if self.mesh_jobs.busy:
            self._reload_pending_force = self._reload_pending_force or force_reload
            self.mesh_jobs.run_when_idle(self._reload_pending_meshes)
            return

        self._reload_pending_force = False
        self._set_loading(True)
        print("Building meshes...")

        colors = self.colors
        self.mesh_jobs.start(
            compute_mesh_data,
            self.input_file,
            NO_CLIP,
            meshes=self.meshes,
            dependencies=self.dependencies,
            mesher=self.mesher,
            force_remesh=force_reload,
            res=self.res_val.currentData(),
            group_by_material=self.material_group_checkbox.isChecked(),
            deflection=self.deflection_val.value(),
            force_pygen=self.pygen_checkbox.isChecked(),
            param_values=dict(self.param_values),
            # Building the pygfx objects is cheap, so it stays on the job's
            # QThread instead of crossing the process boundary.
            post=lambda data: (build_gfx_group(data[0], data[1], colors), *data[2:]),
        )

    def _reload_pending_meshes(self):
        self.reload_meshes(force_reload=self._reload_pending_force)

    def _resize_grid(self, group):
        """(Re)build the floor grid to cover the loaded geometry's footprint.

        GridHelper bakes its size and division count into its vertex data at
        construction time, so there's no in-place resize - the old grid is
        replaced instead.
        """
        bbox = group.get_world_bounding_box() if group is not None else None
        size = grid_size_for_bbox(bbox)
        divisions = min(MAX_GRID_DIVISIONS, max(10, round(size / GRID_SPACING)))
        if self.grid is not None:
            self.scene.remove(self.grid)
        self.grid = gfx.GridHelper(size=size, divisions=divisions, thickness=1)
        self.scene.add(self.grid)

    def _on_reload_finished(self, result, elapsed):
        new_group, meshes, dependencies, instrument_info = result
        self.rebuild_params_form(instrument_info["parameters"])
        self.world_matrices = {
            name: np.array(M) for name, M in instrument_info["world_matrices"].items()
        }
        self.rebuild_clip_frame_combo()
        if self.current_group is not None:
            self.scene.remove(self.current_group)
        self.dependencies = dependencies
        self.meshes = meshes

        self.current_group = new_group

        self.scene.add(self.current_group)
        self.apply_clipping()

        self.rebuild_geometry_panel()
        self.apply_geometry_visibility()
        self._resize_grid(self.current_group)
        recentre_controller(self.controller, self.current_group)
        if self.counts:
            self.rebuild_counts_group()
        print("Reload complete")

        if self._pending_fit_camera:
            self._pending_fit_camera = False
            fit_camera_to_scene(
                self.camera,
                self.controller,
                self.current_group,
            )

    def _on_reload_failed(self, traceback_text, summary):
        print("Mesh rebuild failed:")
        print(traceback_text)

    def _on_mesh_jobs_idle(self):
        self._set_loading(False)

    # ========================================================
    # McStas trace (components and rays)
    # ========================================================

    def rays_enabled(self):
        return self.rays_checkbox.isChecked()

    def trace_enabled(self):
        return self.components_checkbox.isChecked() or self.rays_enabled()

    def reload_trace(self):
        if self.input_file is None or not self.trace_enabled():
            return
        if self.trace_jobs.busy:
            self.trace_jobs.run_when_idle(self.reload_trace)
            return
        params = instrument_param_args(self.param_values)

        self._set_trace_status("Running mcrun --trace...")

        colors = self.colors
        self.trace_jobs.start(
            compute_trace_data,
            self.input_file,
            self.pygen_checkbox.isChecked(),
            params,
            self.ray_count_val.value() if self.rays_enabled() else 0,
            self.ray_seed_val.value() or None,
            post=lambda data: (build_component_group(data[0], colors), data[1]),
        )

    def _on_trace_finished(self, result, elapsed):
        group, rays = result
        if self.trace_group is not None:
            self.scene.remove(self.trace_group)
        self.trace_group = group
        self.scene.add(group)
        self.rebuild_component_panel()
        self.apply_component_visibility()
        status = f"{len(group.component_objects)} McStas components"
        self.set_trace_rays(rays)
        if rays is not None:
            status += f", {rays.n_rays} rays"
        self.apply_clipping()
        self._set_trace_status(f"{status} ({elapsed:.1f} s)")

    def _on_trace_failed(self, traceback_text, summary):
        print(traceback_text, end="", file=sys.stderr)
        print("McStas trace failed:")
        print(summary)
        self._set_trace_status(
            "mcrun FAILED. ENSURE THAT INSTRUMENT COMPILES IN ORDER TO "
            "VISUALIZE THE MCSTAS COMPONENTS",
            error=True,
        )

    def _set_trace_status(self, text, error=False):
        if error:
            text = "\n".join(text.strip().splitlines()[-6:])
        self.trace_status_label.setStyleSheet(
            "color: #c0392b;" if error else "color: gray;"
        )
        self.trace_status_label.setText(text)

    def on_components_changed(self):
        self.component_header.setVisible(
            self.components_checkbox.isChecked() and bool(self.component_checkboxes)
        )
        self.on_geometry_filter_changed(self.geometry_filter.text())
        if self.trace_group is None:
            self.reload_trace()
        else:
            self.apply_component_visibility()

    def on_rays_changed(self):
        self.rays_options.setEnabled(self.rays_checkbox.isChecked())
        if not self.rays_checkbox.isChecked():
            self.apply_ray_visibility()
        elif self.trace_rays is None:
            self.reload_trace()
        else:
            self.rebuild_ray_group()

    def rerun_rays(self):
        self.ray_rerun_timer.stop()
        if self.rays_enabled():
            self.reload_trace()

    def set_trace_rays(self, rays):
        self.trace_rays = rays
        repopulate_combo(
            self.ray_reaching_combo,
            "any component",
            [] if rays is None else rays.component_names,
        )
        self.rebuild_ray_group()

    def rebuild_ray_group(self):
        if self.ray_group is not None:
            self.scene.remove(self.ray_group)
            self.ray_group = None
        rays = self.trace_rays
        if rays is None:
            self.ray_info_label.setText("")
            self.ray_colorbar.hide()
            self.rebalance_docks()
            return
        chosen = rays_reaching(rays, self.ray_reaching_combo.currentData())
        mode = self.ray_color_combo.currentText()
        self.ray_group, value_range = build_ray_group(
            rays, chosen, mode, self.ray_colors, self.ray_marker_sizes
        )
        self.scene.add(self.ray_group)
        self.apply_ray_visibility()
        self.apply_clipping()

        self.ray_info_label.setText(f"Showing {len(chosen)} of {rays.n_rays} rays.")
        if value_range is not None:
            label, unit = RAY_COLOR_MODES[mode]
            suffix = f" {unit}" if unit else ""
            self.ray_colorbar.set_range(
                f"{label} {value_range[0]:.4g}{suffix}",
                f"{value_range[1]:.4g}{suffix}",
            )
            self.ray_colorbar.show()
        else:
            self.ray_colorbar.hide()
        # The colorbar appearing/disappearing changes this panel's natural
        # height - give the docks a chance to reclaim or yield that space.
        self.rebalance_docks()

    def pick_ray_color(self, key):
        hex_color = ask_color(self, self.ray_colors[key], "Colour")
        if hex_color is None:
            return
        self.ray_colors[key] = hex_color
        set_swatch_color(self.ray_color_buttons[key], hex_color)
        group = self.ray_group
        if group is None:
            return
        obj = {
            "teleport": group.teleport_line,
            "scatter": group.scatter_points,
            "absorb": group.absorb_points,
        }.get(key)
        if key == "ray":
            if self.ray_color_combo.currentText() == "Uniform":
                self.rebuild_ray_group()
        elif obj is not None:
            obj.material.color = hex_color

    def set_ray_marker_size(self, key, size):
        self.ray_marker_sizes[key] = size
        points = getattr(self.ray_group, f"{key}_points", None)
        if points is not None:
            points.material.size = size

    def apply_ray_visibility(self):
        if self.ray_group is None:
            return
        self.ray_group.visible = self.rays_checkbox.isChecked()
        for obj, checkbox in (
            (self.ray_group.ray_line, self.rays_line_checkbox),
            (self.ray_group.teleport_line, self.teleport_line_checkbox),
            (self.ray_group.scatter_points, self.scatter_points_checkbox),
            (self.ray_group.absorb_points, self.absorb_points_checkbox),
        ):
            if obj is not None:
                obj.visible = checkbox.isChecked()

    # ========================================================
    # Detector counts
    # ========================================================

    def load_counts_folder(self):
        if self.input_file is None:
            QtWidgets.QMessageBox.warning(
                self, "Detector Counts",
                "Open an instrument first - counts are matched to it by component name.",
            )
            return
        if self.counts_jobs.busy:
            QtWidgets.QMessageBox.information(
                self, "Detector Counts", "Already loading a counts folder - please wait."
            )
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a McStas run folder (containing mccode.sim)"
        )
        if not folder:
            return

        self.load_counts_button.setEnabled(False)
        self.counts_info_label.setText(f"Loading counts from '{folder}'...")

        self._loading_counts_folder = folder
        self.counts_jobs.start(
            compute_counts_data, folder, self.input_file, self.pygen_checkbox.isChecked()
        )

    def _on_counts_finished(self, counts, elapsed):
        folder = self._loading_counts_folder
        self.counts = counts
        self.counts_run_folder = folder
        self.counts_visibility = {name: True for name in counts}
        if counts:
            self.counts_info_label.setText(f"{len(counts)} detector(s) loaded from '{folder}'.")
        else:
            self.counts_info_label.setText(
                f"No spatially-placeable detector output found in '{folder}'."
            )
        self.rebuild_counts_panel()
        self.reset_counts_range()

    def _on_counts_failed(self, traceback_text, error_text):
        folder = self._loading_counts_folder
        print(traceback_text, end="", file=sys.stderr)
        print("Loading detector counts failed:")
        print(error_text)
        self.counts_info_label.setText(f"Could not load counts from '{folder}': {error_text}")
        QtWidgets.QMessageBox.warning(
            self, "Detector Counts", f"Could not load counts from '{folder}':\n{error_text}"
        )

    def _on_counts_jobs_idle(self):
        self.load_counts_button.setEnabled(True)

    def rebuild_counts_panel(self):
        clear_layout(self.counts_panel_layout)
        self.counts_checkboxes.clear()

        for name, lc in self.counts.items():
            axes = lc.axis1 if lc.axis2 is None else f"{lc.axis1}, {lc.axis2}"
            bins = "x".join(str(n) for n in lc.grid.shape)
            cb = make_checkbox(
                wrap_label(f"{name} ({lc.kind})"),
                self.counts_visibility.get(name, True),
                f"{name}: axes ({axes}), {bins} bins, total {lc.total:.4g}",
            )
            cb.toggled.connect(lambda checked, n=name: self.on_counts_visibility_changed(n, checked))
            add_row(self.counts_panel_layout, cb)
            self.counts_checkboxes[name] = cb

    def on_counts_visibility_changed(self, name, checked):
        self.counts_visibility[name] = checked
        mesh = self.counts_meshes.get(name)
        if mesh is not None:
            mesh.visible = checked

    def apply_counts_visibility(self):
        if self.counts_group is None:
            return
        self.counts_group.visible = self.counts_checkbox.isChecked()
        for name, mesh in self.counts_meshes.items():
            mesh.visible = self.counts_visibility.get(name, True)

    def reset_counts_range(self):
        """Set the colour range to the min/max of the currently visible
        loggers (or every loaded logger, if none are individually toggled
        off yet), then rebuild the planes to match."""
        visible = [lc for name, lc in self.counts.items() if self.counts_visibility.get(name, True)]
        grids = [lc.grid for lc in (visible or self.counts.values())]
        vmax = max((float(grid.max()) for grid in grids), default=1.0)
        vmin = 0.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        for spin, value in ((self.counts_min_val, vmin), (self.counts_max_val, vmax)):
            spin.blockSignals(True)
            spin.setRange(min(-1e12, value * 2 - 1), max(1e12, value * 2 + 1))
            spin.setValue(value)
            spin.blockSignals(False)
        self.on_counts_range_changed()

    def on_counts_range_changed(self, *_args):
        vmin, vmax = self.counts_min_val.value(), self.counts_max_val.value()
        if vmax <= vmin or not self.counts:
            self.counts_colorbar.hide()
            return
        if not self.counts_meshes:
            self.rebuild_counts_group()
            return
        for name, mesh in self.counts_meshes.items():
            lc = self.counts[name]
            texture = self.counts_textures.get(name)
            if texture is not None:
                texture.set_data(logger_output.texture_image(lc.grid, vmin, vmax))
            else:
                mesh.geometry.colors.set_data(logger_output.line_vertex_colors(lc.grid, vmin, vmax))
        self.counts_colorbar.set_range(f"{vmin:.4g}", f"{vmax:.4g}")
        self.counts_colorbar.show()
        self.rebalance_docks()

    def rebuild_counts_group(self):
        if self.counts_group is not None:
            self.scene.remove(self.counts_group)
        self.counts_group, self.counts_meshes, self.counts_textures = None, {}, {}
        if not self.counts:
            return
        vmin, vmax = self.counts_min_val.value(), self.counts_max_val.value()
        self.counts_group, self.counts_meshes, self.counts_textures = build_counts_group(
            self.counts, self.world_matrices, vmin, vmax
        )
        self.scene.add(self.counts_group)
        self.apply_counts_visibility()
        self.apply_clipping()
        self.counts_colorbar.set_range(f"{vmin:.4g}", f"{vmax:.4g}")
        self.counts_colorbar.show()
        self.rebalance_docks()

    def fit_whole_instrument(self):
        if self.trace_group is None or not self.trace_group.visible:
            self.reset_view()
            return
        fit_camera_to_scene(self.camera, self.controller, self.trace_group)

    # ========================================================
    # Loading indicator
    # ========================================================

    def _spin_loading_icon(self):
        self._loading_spin_angle = (self._loading_spin_angle + 30) % 360

        size = self._loading_icon_size
        canvas = QtGui.QPixmap(size, size)
        canvas.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(canvas)
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.translate(size / 2, size / 2)
        painter.rotate(self._loading_spin_angle)
        base = self._loading_base_pixmap
        painter.drawPixmap(
            QtCore.QPointF(-base.width() / 2, -base.height() / 2), base
        )
        painter.end()

        self.loading_icon_label.setPixmap(canvas)

        elapsed = time.time() - self._loading_start_time
        self.loading_text_label.setText(f"Building meshes for {elapsed:.2f} seconds")

    def _set_loading(self, is_loading):
        self.loading_icon_label.setVisible(is_loading)
        self.loading_text_label.setVisible(is_loading)
        if is_loading:
            self._loading_start_time = time.time()
            self.loading_text_label.setText("Building meshes for 0.00 seconds")
            self.loading_spin_timer.start()
        else:
            self.loading_spin_timer.stop()
            self.loading_icon_label.setPixmap(self._loading_base_pixmap)

    # ========================================================
    # Watch file changes
    # ========================================================

    def check_file_update(self):
        if self.input_file is None:
            return
        try:
            new_mtime = Path(self.input_file).stat().st_mtime
            if new_mtime != self.last_mtime:
                self.last_mtime = new_mtime
                print("File changed -> rebuilding")
                if self.start_input:
                    self._pending_fit_camera = True
                    self.start_input = False
                self.reload_meshes()
                self.reload_trace()

        except Exception as e:
            print(e)

    # ========================================================
    # Clip settings changed
    # ========================================================

    def on_clip_changed(self):
        self.clip["enable"] = self.clip_checkbox.isChecked()
        self.clip["axis"] = self.axis_combo.currentText()
        self.clip["mode"] = self.mode_combo.currentText()
        self.clip["position"] = self.slice_val.value()
        self.clip["frame"] = self.clip_frame_combo.currentData()
        self.apply_clipping()

    def on_mesher_changed(self):
        self.mesher = self.mesher_box.currentData()
        self.mesher_description_label.setText(
            MESHER_DESCRIPTIONS.get(self.mesher, "")
        )
        self.update_mesher_capability_ui()
        self.reload_meshes(force_reload=True)

    def on_material_group_changed(self):
        self.reload_meshes()

    def on_pygen_changed(self):
        # Switches which parser builds the McStas_instr entirely, so
        # nothing from a previous load can be trusted as unchanged.
        self.reload_meshes(force_reload=True)
        self.reload_trace()

    def on_vacuum_changed(self):
        self.apply_geometry_visibility()

    def on_res_changed(self):
        # Global mesher setting, not part of any dependency signature.
        self.reload_meshes(force_reload=True)

    def on_deflection_changed(self):
        self.reload_meshes(force_reload=True)

    # ========================================================
    # Mesher capability reflection
    # ========================================================

    def update_mesher_capability_ui(self):
        caps = MESHER_CAPABILITIES[self.mesher]
        unused_tip = f"Not used by the '{self.mesher}' mesher."
        set_option_enabled(
            (self.res_val, self.resolution_label),
            caps["resolution"],
            "" if caps["resolution"] else unused_tip,
        )
        set_option_enabled(
            (self.deflection_val, self.deflection_label),
            caps["deflection"],
            DEFLECTION_TOOLTIP if caps["deflection"] else unused_tip,
        )

    # ========================================================
    # Reset view
    # ========================================================

    def reset_view(self):
        target = self.current_group if self.current_group is not None else self.scene
        fit_camera_to_scene(self.camera, self.controller, target)

    # ========================================================
    # Render loop
    # ========================================================

    def animate(self):
        self.gizmo.local.rotation = self.camera.local.rotation
        self.renderer.render(self.scene, self.camera)
        _, h = self.canvas.get_logical_size()
        s = 160
        self.gizmo_viewport.rect = (10, h - s - 10, s, s)
        self.gizmo_viewport.render(self.gizmo_scene, self.gizmo_camera)
        update_camera_depth_range(self.camera, self.controller.target)

        self.canvas.request_draw()

    # ========================================================
    # Shutdown
    # ========================================================

    def closeEvent(self, event):
        for jobs in (self.mesh_jobs, self.trace_jobs, self.counts_jobs):
            jobs.shutdown()
        super().closeEvent(event)


# ============================================================
# Main
# ============================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        "--input_file",
        dest="input_file",
        help="Input mcstas file, can either be mcstasscript or mcstas",
    )
    return parser


def launch(input_file=None):
    """Create the application and open the interactive Union viewer."""
    app = QtWidgets.QApplication([sys.argv[0]])

    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)

    viewer = Viewer(input_file=input_file)
    viewer.show()
    return app.exec()


def main(argv=None):
    parser = parse()
    args = parser.parse_args(argv)
    return launch(args.input_file)


if __name__ == "__main__":
    raise SystemExit(main())
