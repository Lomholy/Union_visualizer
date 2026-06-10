import pygfx as gfx
from PySide6 import QtWidgets, QtGui, QtCore
from rendercanvas.qt import QRenderWidget
import time
import random
from pathlib import Path
import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import argparse
from preprocess import preprocess
from signed_distance_functions import build_sdfs
from meshing import build_meshes


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", help="Input mcstas file, can either be mcstasscript or mcstas"
    )
    parser.add_argument("--out_file", help="Name of output file.", default="")
    parser.add_argument(
        "--resolution", help="Grid fineness for marching cubes algorithm", default=64
    )
    parser.add_argument("--dont_save_vacuum", action="store_true", default=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser


def generate_group(input_file, colors={}, res=64, verbose=False):
    instr, world_matrices, union_geometries = preprocess(input_file, verbose)

    final_sdfs, sdfs = build_sdfs(
        union_geometries,
        world_matrices,
    )

    meshes = build_meshes(
        union_geometries,
        world_matrices,
        final_sdfs,
        res,
        dont_save_vacuum=True,
        export=False,
        verbose=False,
    )

    group = gfx.Group()

    for name, mesh in meshes.items():
        if name not in colors:
            color = random.randrange(0, 2**24)
            std_color = f"#{color:06x}"
            colors[name] = std_color

        gfx_mesh = gfx.Mesh(
            gfx.geometry_from_trimesh(mesh),
            gfx.MeshStandardMaterial(color=colors[name], metalness=0.6, roughness=0.3),
        )

        group.add(gfx_mesh)

    return group


def fit_camera_to_scene(camera, scene, scale=2.0):
    bbox = scene.get_world_bounding_box()

    if bbox is None:
        return

    # bbox shape:
    # [[xmin, ymin, zmin],
    #  [xmax, ymax, zmax]]

    bmin = bbox[0]
    bmax = bbox[1]

    center = (bmin + bmax) / 2.0

    extent = bmax - bmin
    radius = np.linalg.norm(extent) * 0.5

    # Camera direction
    direction = np.array([1, 1, 0.7])
    direction /= np.linalg.norm(direction)

    distance = radius * scale

    position = center + direction * distance

    camera.local.position = position
    camera.look_at(center)

    # Near/far planes
    camera.depth_range = (
        max(0.01, distance - radius * 4),
        distance + radius * 4,
    )
    return center


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    input_file = args.input_file
    out_file = args.out_file
    dont_save_vacuum = args.dont_save_vacuum
    verbose = args.verbose
    res = args.resolution

    canvas = RenderCanvas()
    renderer = gfx.WgpuRenderer(canvas)

    scene = gfx.Scene()

    ambient = gfx.AmbientLight(intensity=10)
    scene.add(ambient)

    colors = {}
    current_group = generate_group(input_file, colors)
    scene.add(current_group)

    camera = gfx.PerspectiveCamera(35)
    center = fit_camera_to_scene(camera, scene)
    controller = gfx.OrbitController(camera, register_events=renderer)
    controller.target = center

    path = Path(input_file)
    last_mtime = path.stat().st_mtime
    last_check = 0.0

    def animate():
        global current_group
        global last_mtime
        global last_check
        now = time.time()
        # Only check once per millisecond
        if now - last_check > 0.001:
            last_check = now
            new_mtime = path.stat().st_mtime
            if new_mtime != last_mtime:
                print("File changed -> rebuilding")
                try:
                    new_group = generate_group(input_file, colors)
                    scene.remove(current_group)
                    current_group = new_group
                    scene.add(current_group)
                    last_mtime = new_mtime
                    print("Reload complete")
                except Exception as e:
                    print(e)
        renderer.render(scene, camera)
        canvas.request_draw()
    canvas.request_draw(animate)
    loop.run()
