import sys
import random
from pathlib import Path
import traceback
import numpy as np
import pygfx as gfx
from qtpy import QtWidgets, QtCore, QtGui
from rendercanvas.qt import QRenderWidget
from pygfx.utils.viewport import Viewport
from preprocess import preprocess
from signed_distance_functions import build_sdfs
from meshing import build_all_meshes, build_mesh, MESHER_CAPABILITIES, DEFAULT_BREP_DEFLECTION
from plot_union_cloud import generate_points
from bounding_box import compute_world_bbox
from gui_helpers import (
    group_components_by_material,
    group_meshes_by_material,
    is_vacuum_material,
)
import argparse

# The meshers offered in the dock, in display order. Kept as a module-level
# constant (rather than only inline inside Viewer.__init__) so tests can
# check it against MESHER_CAPABILITIES without constructing a QApplication.
MESHER_KEYS = ("mc", "dc", "brep")

# ============================================================
# Geometry generation
# ============================================================


def rebuild_mesh(
    meshes,
    name,
    union_geometries,
    world_matrices,
    sdfs,
    final_sdfs,
    res,
    clip,
    mesher,
    deflection=DEFAULT_BREP_DEFLECTION,
):
    meshes[name] = build_mesh(
        union_geometries[name],
        union_geometries,
        world_matrices,
        sdfs,
        final_sdfs,
        res,
        clip,
        mesher=mesher,
        deflection=deflection,
    )
    return meshes


def generate_group(
    input_file,
    clip,
    colors={},
    meshes=None,
    points=None,
    mesher="brep",
    use_colors=False,
    force_remesh=False,
    res=64,
    verbose=True,
    group_by_material=False,
    hide_vacuum=False,
    deflection=DEFAULT_BREP_DEFLECTION,
):
    instr, world_matrices, union_geometries = preprocess(
        input_file,
        verbose=False,
    )
    print("Building sdfs")
    final_sdfs, sdfs = build_sdfs(union_geometries, world_matrices, clip)
    print("Building meshes")

    new_points = generate_points(
        union_geometries, sdfs, final_sdfs, world_matrices, 1000, verbose=False
    )
    if points == None or force_remesh == True:
        meshes = build_all_meshes(
            union_geometries,
            world_matrices,
            sdfs,
            final_sdfs,
            res,
            clip,
            export=False,
            verbose=False,
            mesher=mesher,
            deflection=deflection,
        )
        points = new_points

    print("Defining meshes")
    print(meshes.keys(), points.keys())
    for name in new_points.keys():
        rebuild = 0
        if name not in points.keys():
            rebuild = 1
        elif points[name].shape != new_points[name].shape:
            rebuild = 1
        elif np.any(abs(points[name] - new_points[name]) > 1e-10):
            rebuild = 1
        if rebuild:
            meshes = rebuild_mesh(
                meshes,
                name,
                union_geometries,
                world_matrices,
                sdfs,
                final_sdfs,
                res,
                clip,
                mesher,
                deflection=deflection,
            )
            if verbose:
                print(f"Rebuilding {name}")
    points = new_points

    # Decide what actually gets rendered: one entry per component, or one
    # per material if group_by_material collapses same-material components
    # together (a visual grouping via trimesh.util.concatenate of the
    # already-built meshes, not a boolean fusion), with vacuum/exit
    # materials dropped first if hide_vacuum is set. This is a presentation
    # step over the already-built `meshes` dict - it never triggers or
    # skips a rebuild, so the cache above stays keyed on component names
    # regardless of how the result is displayed.
    visible_geometries = union_geometries
    if hide_vacuum:
        visible_geometries = {
            name: comp
            for name, comp in union_geometries.items()
            if not is_vacuum_material(getattr(comp, "material_string", None))
        }

    if group_by_material:
        render_meshes = group_meshes_by_material(visible_geometries, meshes)
        geometry_components = {
            material: names
            for material, names in group_components_by_material(
                visible_geometries
            ).items()
            if material in render_meshes
        }
    else:
        render_meshes = {
            name: meshes[name]
            for name in visible_geometries
            if meshes.get(name) is not None
        }
        geometry_components = {name: [name] for name in render_meshes}

    print("Defining group")
    group = gfx.Group()
    group.geometry_meshes = {}
    # {displayed key -> [underlying component names]}, so a caller like
    # "centre view on this row" works the same way whether the row is one
    # component or (under group_by_material) several sharing a material.
    group.geometry_components = geometry_components
    group.union_geometries = union_geometries
    group.world_matrices = world_matrices

    for key, mesh in render_meshes.items():
        if key not in colors and use_colors:
            color = random.randrange(0, 2**24)
            colors[key] = f"#{color:06x}"
        elif not use_colors:
            colors[key] = "#b6b6b6"
        elif colors[key] == "#b6b6b6" and use_colors:
            color = random.randrange(0, 2**24)
            colors[key] = f"#{color:06x}"

        gfx_mesh = gfx.Mesh(
            gfx.geometry_from_trimesh(mesh),
            gfx.MeshStandardMaterial(
                color=colors[key],
                metalness=0,
                roughness=0.8,
            ),
        )

        group.add(gfx_mesh)
        group.geometry_meshes[key] = gfx_mesh

    return (group, meshes, points)


def make_coordinate_axes(
    length=10.0,
    tick_spacing=1.0,
    tick_size=0.1,
    show_labels=True,
):
    group = gfx.Group()

    axes = [
        ("X", (1, 0, 0), np.array([1, 0, 0])),
        ("Y", (0, 1, 0), np.array([0, 1, 0])),
        ("Z", (0, 0, 1), np.array([0, 0, 1])),
    ]

    for label, color, axis in axes:
        # main axis line
        positions = np.array([[0, 0, 0], axis * length], dtype=np.float32)
        line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=2.0, color=color),
        )
        group.add(line)

        # ticks
        n_ticks = int(length / tick_spacing)
        for i in range(1, n_ticks + 1):
            p = axis * i * tick_spacing
            perp = np.array([0, 1, 0]) if axis[0] else np.array([1, 0, 0])

            tick = gfx.Line(
                gfx.Geometry(
                    positions=np.array(
                        [p - perp * tick_size, p + perp * tick_size],
                        dtype=np.float32,
                    )
                ),
                gfx.LineMaterial(thickness=1.0, color=color),
            )
            group.add(tick)

        # label
        if show_labels:
            text = gfx.Text(
                text=label,
                font_size=0.5,
                material=gfx.materials.TextMaterial(color=color),
            )
            text.local.position = axis * (length + tick_size * 4)
            group.add(text)

    return group


# ============================================================
# Camera fitting
# ============================================================


def fit_camera_to_scene(camera, controller, scene, scale=2.0):
    print(scene)
    bbox = scene.get_world_bounding_box()

    if bbox is None:
        return
    bmin = bbox[0]
    bmax = bbox[1]
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin
    radius = np.linalg.norm(extent) * 0.5
    direction = np.array([1.0, 1.0, 0.7])
    direction /= np.linalg.norm(direction)
    distance = radius * scale
    position = center + direction * distance
    camera.local.position = position
    camera.look_at(center)
    camera.depth_range = (
        max(0.01, distance - radius * 4),
        1e6,
    )
    controller.target = center


def recentre_controller(controller, group):
    """Keep the orbit pivot on the current geometry's centre, without
    touching the camera's position or zoom.

    fit_camera_to_scene also sets controller.target, but only ever ran once
    per file (on first load / on File > Open) - every later reload (a clip
    tweak, a resolution change, a mesher switch) left the pivot wherever it
    was set at that first load. For geometry that isn't centred on the
    world origin (a real instrument's Union components routinely sit tens
    of metres downstream), that makes the camera orbit around empty space
    instead of the model: rotating reads as swinging the model around
    rather than turning it. This is the narrow fix - only the pivot moves,
    so a parameter tweak never yanks the framing/zoom the way calling
    fit_camera_to_scene on every reload would."""
    if group is None:
        return
    bbox = group.get_world_bounding_box()
    if bbox is None:
        return
    controller.target = (bbox[0] + bbox[1]) / 2.0


# ============================================================
# Main window
# ============================================================


class Viewer(QtWidgets.QMainWindow):
    def __init__(self, input_file=None):
        super().__init__()
        self.setWindowTitle("Union Viewer")
        self.resize(1400, 900)
        self.colors = {}
        self.input_file = input_file
        # Previously only set inside `if input_file:`, so launching with no
        # --input_file left this attribute missing entirely; the first tick
        # of check_file_update's 100ms timer (before any file is opened via
        # the menu) then raised AttributeError on self.start_input, silently
        # swallowed by that method's bare `except`. Unconditional now so the
        # no-argument launch path behaves the same as the fix already makes
        # every other path behave.
        self.start_input = bool(input_file)
        self.last_mtime = None
        self.current_group = None
        self.points = None
        self.meshes = None
        self.mesher = "brep"

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

        # soft global light
        ambient = gfx.AmbientLight(intensity=0.6)
        self.scene.add(ambient)
        # main light (like sun)
        key = gfx.DirectionalLight(intensity=1.5)
        key.local.position = (1, 1, 1)
        key.look_at((0, 0, 0))
        self.scene.add(key)

        # main light (like sun)
        key2 = gfx.DirectionalLight(intensity=1.5)
        key2.local.position = (-1, -1, -1)
        key2.look_at((0, 0, 0))
        self.scene.add(key2)
        # optional fill light (soft opposite side)
        fill = gfx.DirectionalLight(intensity=0.5)
        fill.local.position = (-1, 1, -1)
        fill.look_at((0, 0, 0))
        self.scene.add(fill)
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
        self.scene.add(make_coordinate_axes(length=1000, tick_spacing=1000))

        self.grid = gfx.GridHelper(size=100, divisions=100, thickness=1)
        self.scene.add(self.grid)

        self.gizmo_viewport = Viewport(self.renderer, (0, 0, 120, 120))
        self.gizmo_scene = gfx.Scene()
        self.gizmo = make_coordinate_axes(length=1.0, tick_spacing=0.5, tick_size=0.05)
        self.gizmo_scene.add(self.gizmo)
        self.gizmo_camera = gfx.PerspectiveCamera(50, 1)
        self.gizmo_camera.local.position = (0, 0, 4)

        # ----------------------------------------------------
        # Menu
        # ----------------------------------------------------
        menubar = QtWidgets.QMenuBar(self)
        self.setMenuBar(menubar)
        file_menu = menubar.addMenu("File")
        open_action = QtGui.QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
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
        # Clipping dock widget
        # ----------------------------------------------------

        self.clip_enable = False
        self.clip_axis = "Z"
        self.clip_mode = "Above"
        self.clip_position = 0.0
        self.clip = {
            "enable": self.clip_enable,
            "axis": self.clip_axis,
            "mode": self.clip_mode,
            "position": self.clip_position,
        }

        dock = QtWidgets.QDockWidget("Clipping", self)

        dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        )

        dock_widget = QtWidgets.QWidget()

        dock_layout = QtWidgets.QVBoxLayout(dock_widget)

        # ----------------------------------------
        # Enable checkbox
        # ----------------------------------------

        self.clip_checkbox = QtWidgets.QCheckBox("Enable clipping")
        dock_layout.addWidget(self.clip_checkbox)

        self.color_checkbox = QtWidgets.QCheckBox("Color individual component")
        dock_layout.addWidget(self.color_checkbox)

        self.material_group_checkbox = QtWidgets.QCheckBox("Group by material")
        self.material_group_checkbox.setToolTip(
            "Combine components that share a material into a single "
            "object (trimesh concatenation, not a boolean fusion - "
            "adjacent same-material volumes still keep their internal "
            "walls, they just render and toggle together)."
        )
        dock_layout.addWidget(self.material_group_checkbox)

        self.vacuum_checkbox = QtWidgets.QCheckBox("Hide vacuum")
        self.vacuum_checkbox.setChecked(True)
        self.vacuum_checkbox.setToolTip(
            "Hides Union volumes whose material_string is 'vacuum'/'Vacuum' "
            "or 'exit'/'Exit' (McStas treats 'exit' as vacuum with the "
            "ray marked as having left the instrument). Volumes with no "
            "material set at all are not affected - only these two exact "
            "spellings mean vacuum to McStas."
        )
        dock_layout.addWidget(self.vacuum_checkbox)

        # ----------------------------------------
        # Mesher selector
        # ----------------------------------------

        self.mesher_box = QtWidgets.QComboBox()
        for key in MESHER_KEYS:
            label = key
            if not MESHER_CAPABILITIES[key]["incremental_rebuild"]:
                label += " (full rebuild only)"
            self.mesher_box.addItem(label, key)
        # self.mesher default (below) is "brep", which used to disagree
        # with the combo's own default selection (index 0, "mc") until the
        # user touched it - so the very first render used a mesher the
        # dock never actually showed as selected. Align them.
        self.mesher_box.setCurrentIndex(self.mesher_box.findData(self.mesher))

        dock_layout.addWidget(self.mesher_box)
        # ----------------------------------------
        # Axis selector
        # ----------------------------------------

        axis_layout = QtWidgets.QHBoxLayout()
        axis_label = QtWidgets.QLabel("Axis")
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["X", "Y", "Z"])
        axis_layout.addWidget(axis_label)
        axis_layout.addWidget(self.axis_combo)
        dock_layout.addLayout(axis_layout)

        # ----------------------------------------
        # Mode selector
        # ----------------------------------------

        mode_layout = QtWidgets.QHBoxLayout()
        mode_label = QtWidgets.QLabel("Mode")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(
            [
                "Above",
                "Below",
            ]
        )
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        dock_layout.addLayout(mode_layout)

        # ----------------------------------------
        # Slice position slider
        # ----------------------------------------

        position_layout = QtWidgets.QHBoxLayout()
        position_label = QtWidgets.QLabel("Position")
        self.slice_val = QtWidgets.QDoubleSpinBox()
        self.slice_val.setDecimals(5)
        self.slice_val.setRange(-1e6, 1e6)
        self.slice_val.setSingleStep(0.01)
        self.slice_val.setValue(0.0)
        position_layout.addWidget(position_label)
        position_layout.addWidget(self.slice_val)
        dock_layout.addLayout(position_layout)

        # ----------------------------------------
        # Resolution value
        # ----------------------------------------

        resolution_layout = QtWidgets.QHBoxLayout()
        self.resolution_label = QtWidgets.QLabel("Resolution")
        self.res_val = QtWidgets.QComboBox()
        self.res_val.addItem("16", 16)
        self.res_val.addItem("32", 32)
        self.res_val.addItem("64", 64)
        self.res_val.addItem("128", 128)
        self.res_val.addItem("256", 256)
        self.res_val.addItem("512", 512)
        self.res_val.setCurrentIndex(2)
        resolution_layout.addWidget(self.resolution_label)
        resolution_layout.addWidget(self.res_val)
        dock_layout.addLayout(resolution_layout)

        # ----------------------------------------
        # Surface deflection (brep only - the marching-cubes-style
        # Resolution combo above doesn't apply to an exact BRep solid;
        # BRepMesh_IncrementalMesh's linear deflection is the mesher's own
        # equivalent, previously hard-coded to 0.01)
        # ----------------------------------------

        deflection_layout = QtWidgets.QHBoxLayout()
        self.deflection_label = QtWidgets.QLabel("Surface deflection")
        self.deflection_val = QtWidgets.QDoubleSpinBox()
        self.deflection_val.setDecimals(4)
        self.deflection_val.setRange(0.0001, 10.0)
        self.deflection_val.setSingleStep(0.001)
        self.deflection_val.setValue(0.01)
        deflection_layout.addWidget(self.deflection_label)
        deflection_layout.addWidget(self.deflection_val)
        dock_layout.addLayout(deflection_layout)

        self.reset_visibility_button = QtWidgets.QPushButton(
            "Reset all hidden geometries"
        )
        self.reset_visibility_button.clicked.connect(self.reset_geometry_visibility)
        visibility_layout = QtWidgets.QHBoxLayout()
        visibility_layout.addWidget(self.reset_visibility_button)
        dock_layout.addLayout(visibility_layout)

        # ----------------------------------------
        # Reset view
        # ----------------------------------------

        self.reset_view_button = QtWidgets.QPushButton("Reset view")
        self.reset_view_button.setToolTip("Refit the camera (shortcut: R)")
        self.reset_view_button.clicked.connect(self.reset_view)
        dock_layout.addWidget(self.reset_view_button)
        self.reset_view_shortcut = QtGui.QShortcut(QtGui.QKeySequence("R"), self)
        self.reset_view_shortcut.activated.connect(self.reset_view)

        # ----------------------------------------
        # Position value display
        # ----------------------------------------

        dock_layout.addStretch()
        dock.setWidget(dock_widget)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            dock,
        )

        # ----------------------------------------------------
        # Geometry visibility dock
        # ----------------------------------------------------

        self.geometry_checkboxes = {}
        self.geometry_center_buttons = {}
        self.geometry_visibility = {}

        self.geometry_dock = QtWidgets.QDockWidget("Visible Geometries", self)
        geometry_dock_widget = QtWidgets.QWidget()
        geometry_dock_layout = QtWidgets.QVBoxLayout(geometry_dock_widget)

        self.geometry_filter = QtWidgets.QLineEdit()
        self.geometry_filter.setPlaceholderText("Filter...")
        self.geometry_filter.setClearButtonEnabled(True)
        self.geometry_filter.textChanged.connect(self.on_geometry_filter_changed)
        geometry_dock_layout.addWidget(self.geometry_filter)

        show_hide_layout = QtWidgets.QHBoxLayout()
        self.show_all_button = QtWidgets.QPushButton("Show all")
        self.hide_all_button = QtWidgets.QPushButton("Hide all")
        self.show_all_button.clicked.connect(
            lambda: self.set_all_geometry_visibility(True)
        )
        self.hide_all_button.clicked.connect(
            lambda: self.set_all_geometry_visibility(False)
        )
        show_hide_layout.addWidget(self.show_all_button)
        show_hide_layout.addWidget(self.hide_all_button)
        geometry_dock_layout.addLayout(show_hide_layout)

        # The per-component/per-material checkbox list itself, in a scroll
        # area: with dozens of Union volumes (a real instrument routinely
        # has that many) the list used to run off the bottom of the dock
        # with no way to reach the rest of it.
        self.geometry_widget = QtWidgets.QWidget()
        self.geometry_layout = QtWidgets.QVBoxLayout(self.geometry_widget)
        self.geometry_layout.addStretch()

        geometry_scroll = QtWidgets.QScrollArea()
        geometry_scroll.setWidgetResizable(True)
        geometry_scroll.setWidget(self.geometry_widget)
        geometry_dock_layout.addWidget(geometry_scroll)

        self.geometry_dock.setWidget(geometry_dock_widget)

        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            self.geometry_dock,
        )

        # ----------------------------------------------------
        # Clipping signals
        # ----------------------------------------------------

        self.clip_checkbox.stateChanged.connect(self.on_clip_changed)
        self.color_checkbox.stateChanged.connect(self.on_color_changed)
        self.material_group_checkbox.stateChanged.connect(self.on_material_group_changed)
        self.vacuum_checkbox.stateChanged.connect(self.on_vacuum_changed)
        self.mesher_box.currentIndexChanged.connect(self.on_mesher_changed)
        self.axis_combo.currentTextChanged.connect(self.on_clip_changed)
        self.mode_combo.currentTextChanged.connect(self.on_clip_changed)
        self.slice_val.valueChanged.connect(self.on_clip_changed)
        self.res_val.currentIndexChanged.connect(self.on_res_changed)
        self.deflection_val.valueChanged.connect(self.on_deflection_changed)

        # Reflect what the mesher just selected in the combo (brep, by
        # default - see the comment where the combo is built) actually
        # supports, before anything has a chance to render with the wrong
        # controls enabled.
        self.update_mesher_capability_ui()

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
        self.reload_meshes()
        fit_camera_to_scene(
            self.camera,
            self.controller,
            self.current_group,
        )

    def on_geometry_visibility_changed(self, name, mesh, checked):
        self.geometry_visibility[name] = checked
        mesh.visible = checked

    def reset_geometry_visibility(self):
        self.geometry_visibility.clear()
        if self.current_group is None:
            return
        for name, mesh in self.current_group.geometry_meshes.items():
            mesh.visible = True
            if name in self.geometry_checkboxes:
                self.geometry_checkboxes[name].blockSignals(True)
                self.geometry_checkboxes[name].setChecked(True)
                self.geometry_checkboxes[name].blockSignals(False)

    def _clear_geometry_layout_item(self, item):
        """Delete whatever a takeAt() handed back - a bare widget (the old
        one-checkbox-per-row layout) or, now that each row is its own
        QHBoxLayout of [checkbox, centre button], a nested layout whose own
        widgets need deleting one level down."""
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            row = item.layout()
            while row.count():
                self._clear_geometry_layout_item(row.takeAt(0))

    def rebuild_geometry_panel(self):
        while self.geometry_layout.count() > 1:
            self._clear_geometry_layout_item(self.geometry_layout.takeAt(0))

        self.geometry_checkboxes.clear()
        self.geometry_center_buttons.clear()

        if self.current_group is None:
            return

        for name, mesh in self.current_group.geometry_meshes.items():
            visible = self.geometry_visibility.get(name, True)
            mesh.visible = visible

            row = QtWidgets.QHBoxLayout()

            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(visible)
            cb.toggled.connect(
                lambda checked, n=name, m=mesh: self.on_geometry_visibility_changed(
                    n, m, checked
                )
            )
            row.addWidget(cb, 1)

            center_button = QtWidgets.QToolButton()
            center_button.setText("⊙")  # circled dot, reads as a target
            center_button.setToolTip(f"Centre the view on '{name}'")
            center_button.clicked.connect(
                lambda checked=False, n=name: self.centre_on_geometry(n)
            )
            row.addWidget(center_button)

            self.geometry_layout.insertLayout(
                self.geometry_layout.count() - 1,
                row,
            )

            self.geometry_checkboxes[name] = cb
            self.geometry_center_buttons[name] = center_button

        self.on_geometry_filter_changed(self.geometry_filter.text())

    def on_geometry_filter_changed(self, text):
        text = text.strip().lower()
        for name, cb in self.geometry_checkboxes.items():
            match = text in name.lower()
            cb.setVisible(match)
            button = self.geometry_center_buttons.get(name)
            if button is not None:
                button.setVisible(match)

    def set_all_geometry_visibility(self, visible):
        """Show all / Hide all. Goes through each checkbox's own toggled
        signal (on_geometry_visibility_changed) rather than writing
        mesh.visible directly, so the checked state, self.geometry_visibility
        and the actual mesh stay in sync the same single way they always do -
        applies to every row regardless of what the text filter currently
        shows, matching what the two button labels say."""
        for cb in self.geometry_checkboxes.values():
            cb.setChecked(visible)

    def centre_on_geometry(self, key):
        """Move the orbit pivot (not the camera position/zoom - see
        recentre_controller) to the world-space bounding-box centre of
        whichever original components this row represents. Ungrouped, that
        is just the one component; under "Group by material" it is every
        component sharing that material, unioned together."""
        if self.current_group is None:
            return
        member_names = self.current_group.geometry_components.get(key, [key])
        union_geometries = self.current_group.union_geometries
        world_matrices = self.current_group.world_matrices

        mins, maxs = [], []
        for name in member_names:
            comp = union_geometries.get(name)
            if comp is None:
                continue
            try:
                bmin, bmax = compute_world_bbox(comp, world_matrices)
            except Exception as e:
                print(f"Could not compute a bounding box for '{name}': {e}")
                continue
            mins.append(bmin)
            maxs.append(bmax)

        if not mins:
            return

        bmin = np.min(mins, axis=0)
        bmax = np.max(maxs, axis=0)
        self.controller.target = (bmin + bmax) / 2.0

    # ========================================================
    # Reload geometry
    # ========================================================

    def reload_meshes(self, force_reload=False):
        if self.input_file is None:
            return
        # Dual contouring has no working single-component rebuild path
        # (meshing.build_mesh unconditionally returns None for it - see
        # MESHER_CAPABILITIES' "incremental_rebuild" comment); without this,
        # any dock control changed while "dc" is selected would silently
        # blank out whichever components the point-cloud diff thought had
        # changed, instead of visibly re-running the (working) full build.
        if not MESHER_CAPABILITIES[self.mesher]["incremental_rebuild"]:
            force_reload = True
        print("Rebuilding meshes...")
        try:
            new_group, meshes, points = generate_group(
                self.input_file,
                self.clip,
                self.colors,
                meshes=self.meshes,
                points=self.points,
                mesher=self.mesher,
                use_colors=self.color_checkbox.isChecked(),
                res=self.res_val.currentData(),
                force_remesh=force_reload,
                group_by_material=self.material_group_checkbox.isChecked(),
                hide_vacuum=self.vacuum_checkbox.isChecked(),
                deflection=self.deflection_val.value(),
            )
            for name, mesh in new_group.geometry_meshes.items():
                mesh.visible = self.geometry_visibility.get(name, True)
            if self.current_group is not None:
                self.scene.remove(self.current_group)
            self.points = points
            self.meshes = meshes

            self.current_group = new_group

            self.scene.add(self.current_group)

            self.rebuild_geometry_panel()
            # Keep the orbit pivot on the geometry's centre on every reload,
            # not just the first one - see recentre_controller's docstring
            # for why leaving this to fit_camera_to_scene's one-shot calls
            # left the camera orbiting empty space after the first load.
            recentre_controller(self.controller, self.current_group)
            print("Reload complete")
        except Exception as e:
            print("Mesh rebuild failed:")
            traceback.print_exc()
            print(e)

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
                self.reload_meshes()
                if self.start_input:
                    fit_camera_to_scene(
                        self.camera,
                        self.controller,
                        self.current_group,
                    )
                    self.start_input = False

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
        self.reload_meshes()

    def on_mesher_changed(self):
        self.mesher = self.mesher_box.currentData()
        self.update_mesher_capability_ui()
        self.reload_meshes(force_reload=True)

    def on_color_changed(self):
        self.reload_meshes()

    def on_material_group_changed(self):
        self.reload_meshes()

    def on_vacuum_changed(self):
        self.reload_meshes()

    def on_res_changed(self):
        self.reload_meshes()

    def on_deflection_changed(self):
        self.reload_meshes()

    # ========================================================
    # Mesher capability reflection (6d): grey out / explain a control
    # that the currently selected mesher does not use, rather than
    # leaving it interactive and silently doing nothing.
    # ========================================================

    def update_mesher_capability_ui(self):
        caps = MESHER_CAPABILITIES[self.mesher]

        res_used = caps["resolution"]
        self.res_val.setEnabled(res_used)
        self.resolution_label.setEnabled(res_used)
        tip = "" if res_used else f"Not used by the '{self.mesher}' mesher."
        self.res_val.setToolTip(tip)
        self.resolution_label.setToolTip(tip)

        deflection_used = caps["deflection"]
        self.deflection_val.setEnabled(deflection_used)
        self.deflection_label.setEnabled(deflection_used)
        tip = "" if deflection_used else f"Not used by the '{self.mesher}' mesher."
        self.deflection_val.setToolTip(tip)
        self.deflection_label.setToolTip(tip)

    # ========================================================
    # Reset view (6e)
    # ========================================================

    def reset_view(self):
        # Refit to the actual loaded geometry when there is any; falling
        # back to the (otherwise empty) scene - which is only the 1000-unit
        # coordinate axes and grid - so the button and its shortcut don't
        # raise before a file has ever been opened, rather than requiring
        # self.current_group to be set the way the pre-existing call sites
        # (open_file, the first-load branch of check_file_update) do.
        target = self.current_group if self.current_group is not None else self.scene
        fit_camera_to_scene(self.camera, self.controller, target)

    # ========================================================
    # Render loop
    # ========================================================

    def animate(self):
        self.gizmo.local.rotation = self.camera.local.rotation
        self.renderer.render(self.scene, self.camera)
        w, h = self.canvas.get_logical_size()
        s = 160
        self.gizmo_viewport.rect = (10, h - s - 10, s, s)
        self.gizmo_viewport.render(self.gizmo_scene, self.gizmo_camera)

        self.canvas.request_draw()


# ============================================================
# Main
# ============================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", help="Input mcstas file, can either be mcstasscript or mcstas"
    )
    return parser


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    input_file = args.input_file
    app = QtWidgets.QApplication(sys.argv)

    app.setAttribute(QtCore.Qt.ApplicationAttribute.AA_DontUseNativeMenuBar)

    viewer = Viewer(input_file=input_file)
    viewer.show()
    sys.exit(app.exec())
