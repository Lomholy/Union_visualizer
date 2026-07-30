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
from meshing import build_all_meshes, build_mesh
from plot_union_cloud import generate_points
import argparse

# ============================================================
# Geometry generation
# ============================================================


def rebuild_mesh(
    meshes,
    mesh,
    name,
    union_geometries,
    world_matrices,
    sdfs,
    final_sdfs,
    res,
):
    meshes[name] = build_mesh(
        union_geometries[name], world_matrices, sdfs, final_sdfs, res
    )
    mesh = meshes[name]
    return mesh, meshes


def generate_group(
    input_file,
    clip,
    colors={},
    meshes=None,
    points=None,
    use_colors=False,
    force_remesh=False,
    res=64,
    verbose=True,
):
    instr, world_matrices, union_geometries = preprocess(
        input_file,
        verbose=False,

    )
    print("Building sdfs")
    print(clip)
    final_sdfs, sdfs = build_sdfs(union_geometries, world_matrices, clip)
    print("Building meshes")

    new_points = generate_points(
        union_geometries, sdfs, final_sdfs, world_matrices, 1000, verbose=False
    )
    if points == None:
        meshes = build_all_meshes(
            union_geometries,
            world_matrices,
            sdfs,
            final_sdfs,
            res,
            export=False,
            verbose=False,
        )
        points = new_points

    print("Defining group")
    group = gfx.Group()
    group.geometry_meshes = {}
    for name, mesh in meshes.items():
        rebuild = 0
        if name not in points.keys():
            rebuild = 1
        if points[name].shape != new_points[name].shape:
            rebuild = 1
        elif np.any(abs(points[name] - new_points[name]) > 1e-10):
            rebuild = 1
        if rebuild:
            mesh, meshes = rebuild_mesh(
                meshes,
                mesh,
                name,
                union_geometries,
                world_matrices,
                sdfs,
                final_sdfs,
                res,
            )
            if verbose:
                print(f"Rebuilding {name}")

        if name not in colors and use_colors:
            color = random.randrange(0, 2**24)
            colors[name] = f"#{color:06x}"
        elif not use_colors:
            colors[name] = "#b6b6b6"
        elif colors[name] == "#b6b6b6" and use_colors:
            color = random.randrange(0, 2**24)
            colors[name] = f"#{color:06x}"
        if mesh is None:
            continue

        gfx_mesh = gfx.Mesh(
            gfx.geometry_from_trimesh(mesh),
            gfx.MeshStandardMaterial(
                color=colors[name],
                metalness=0,
                roughness=0.8,
            ),
        )

        group.add(gfx_mesh)
        group.geometry_meshes[name] = gfx_mesh
    points = new_points

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
        if input_file:
            self.start_input = True
        self.last_mtime = None
        self.current_group = None
        self.points = None
        self.meshes = None

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
        resolution_label = QtWidgets.QLabel("Resolution")
        self.res_val = QtWidgets.QComboBox()
        self.res_val.addItem("16", 16)
        self.res_val.addItem("32", 32)
        self.res_val.addItem("64", 64)
        self.res_val.addItem("128", 128)
        self.res_val.addItem("256", 256)
        self.res_val.addItem("512", 512)
        self.res_val.setCurrentIndex(2)
        resolution_layout.addWidget(resolution_label)
        resolution_layout.addWidget(self.res_val)
        dock_layout.addLayout(resolution_layout)

        self.reset_visibility_button = QtWidgets.QPushButton(
            "Reset all hidden geometries"
        )
        self.reset_visibility_button.clicked.connect(self.reset_geometry_visibility)
        visibility_layout = QtWidgets.QHBoxLayout()
        visibility_layout.addWidget(self.reset_visibility_button)
        dock_layout.addLayout(visibility_layout)
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
        self.geometry_visibility = {}

        self.geometry_dock = QtWidgets.QDockWidget("Visible Geometries", self)
        self.geometry_widget = QtWidgets.QWidget()
        self.geometry_layout = QtWidgets.QVBoxLayout(self.geometry_widget)

        self.geometry_layout.addStretch()

        self.geometry_dock.setWidget(self.geometry_widget)

        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea,
            self.geometry_dock,
        )

        # ----------------------------------------------------
        # Clipping signals
        # ----------------------------------------------------

        self.clip_checkbox.stateChanged.connect(self.on_clip_changed)
        self.color_checkbox.stateChanged.connect(self.on_color_changed)
        self.axis_combo.currentTextChanged.connect(self.on_clip_changed)
        self.mode_combo.currentTextChanged.connect(self.on_clip_changed)
        self.slice_val.valueChanged.connect(self.on_clip_changed)
        self.res_val.currentIndexChanged.connect(self.on_res_changed)

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

    def rebuild_geometry_panel(self):
        while self.geometry_layout.count() > 1:
            item = self.geometry_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.geometry_checkboxes.clear()

        if self.current_group is None:
            return

        for name, mesh in self.current_group.geometry_meshes.items():
            visible = self.geometry_visibility.get(name, True)
            mesh.visible = visible

            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(visible)

            cb.toggled.connect(
                lambda checked, n=name, m=mesh: self.on_geometry_visibility_changed(
                    n, m, checked
                )
            )
            self.geometry_layout.insertWidget(
                self.geometry_layout.count() - 1,
                cb,
            )

            self.geometry_checkboxes[name] = cb

    # ========================================================
    # Reload geometry
    # ========================================================

    def reload_meshes(self):
        if self.input_file is None:
            return
        print("Rebuilding meshes...")
        try:
            new_group, meshes, points = generate_group(
                self.input_file,
                self.clip,
                self.colors,
                meshes=self.meshes,
                points=self.points,
                use_colors=self.color_checkbox.isChecked(),
                res=self.res_val.currentData(),
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

    def on_color_changed(self):
        self.reload_meshes()

    def on_res_changed(self):
        self.reload_meshes()

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
