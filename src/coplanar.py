"""Find Union geometries whose flat faces lie in the same plane and overlap.

Coincident faces between two Union volumes make the ray tracing in
Union_master ambiguous: a ray hitting the shared plane can end up in the
wrong volume. This reports every pair of volumes with overlapping faces in
a common plane, whether the faces point the same way or opposite ways.

Flat faces checked: all faces of Union_box (tapered too), the end caps of
Union_cylinder and Union_cone, and every triangle of Union_mesh. Curved
surfaces (cylinder and cone sides, spheres) are not checked.

Usage:
    python -m coplanar my_instrument.instr
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

from preprocess import box_dimensions

CIRCLE_SEGMENTS = 64


def _box_faces(comp):
    x1, y1, x2, y2 = box_dimensions(comp)
    z = float(comp.zdepth) / 2
    corners = np.array(
        [
            [-x1 / 2, -y1 / 2, -z], [x1 / 2, -y1 / 2, -z],
            [x1 / 2, y1 / 2, -z], [-x1 / 2, y1 / 2, -z],
            [-x2 / 2, -y2 / 2, z], [x2 / 2, -y2 / 2, z],
            [x2 / 2, y2 / 2, z], [-x2 / 2, y2 / 2, z],
        ]
    )
    quads = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    return [corners[q] for q in quads]


def _disc(radius, y):
    angles = np.linspace(0, 2 * np.pi, CIRCLE_SEGMENTS, endpoint=False)
    return np.column_stack(
        [radius * np.cos(angles), np.full_like(angles, y), radius * np.sin(angles)]
    )


def _local_faces(comp):
    """Flat faces of one Union geometry, each an (N, 3) array of vertices."""
    t = comp.component_name.lower()
    if t == "union_box":
        return _box_faces(comp)
    if t == "union_cylinder":
        h = float(comp.yheight) / 2
        r = float(comp.radius)
        return [_disc(r, -h), _disc(r, h)]
    if t == "union_cone":
        h = float(comp.yheight) / 2
        caps = [(float(comp.radius_bottom), -h), (float(comp.radius_top), h)]
        return [_disc(r, y) for r, y in caps if r > 0]
    if t == "union_mesh":
        scale = float(comp.coordinate_scale) if comp.coordinate_scale is not None else 1e-3
        mesh = trimesh.load(comp.filename, force="mesh")
        return list(mesh.vertices[mesh.faces] * scale)
    return []


def _plane(points):
    """Unit normal with a fixed sign convention, and offset d = n . p."""
    centered = points - points.mean(axis=0)
    normal = np.linalg.svd(centered)[2][-1]
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    return normal, float(normal @ points[0])


def _in_plane_polygon(points, normal):
    u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return Polygon(np.column_stack([points @ u, points @ v])).buffer(0)


def find_coplanar_faces(union_geometries, world_matrices, tol=1e-9):
    """Return [(name_a, name_b, normal, offset, overlap_area)] for every pair
    of volumes that have overlapping flat faces in a common plane.

    tol is the distance (m) within which two planes count as the same plane,
    and also the tolerance on the components of their unit normals."""
    buckets = defaultdict(lambda: defaultdict(list))
    for name, comp in union_geometries.items():
        matrix = world_matrices[name]
        for face in _local_faces(comp):
            world = face @ matrix[:3, :3].T + matrix[:3, 3]
            normal, offset = _plane(world)
            # ponytail: planes are bucketed by rounding, so two planes that
            # straddle a rounding boundary are missed; compare neighbouring
            # buckets if that ever matters.
            key = tuple(np.round(np.append(normal, offset) / tol).astype(np.int64))
            buckets[key][name].append(_in_plane_polygon(world, normal))

    found = []
    for key, per_volume in buckets.items():
        names = sorted(per_volume)
        if len(names) < 2:
            continue
        shapes = {n: unary_union(per_volume[n]) for n in names}
        normal = np.array(key[:3]) * tol
        offset = key[3] * tol
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                area = shapes[a].intersection(shapes[b]).area
                if area > 1e-9 * min(shapes[a].area, shapes[b].area):
                    found.append((a, b, normal, offset, area))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Report Union volumes with overlapping coplanar faces."
    )
    parser.add_argument("input_file", help="McStas .instr or mcstasscript .py file")
    parser.add_argument(
        "--tol", type=float, default=1e-9, help="plane distance tolerance in m"
    )
    args = parser.parse_args(argv)

    from preprocess import preprocess

    _instr, world_matrices, union_geometries = preprocess(args.input_file, verbose=False)
    found = find_coplanar_faces(union_geometries, world_matrices, tol=args.tol)
    for a, b, normal, offset, area in found:
        print(
            f"{a} and {b}: coplanar faces, normal {np.round(normal, 6)}, "
            f"offset {offset:.6g} m, overlap area {area:.3g} m^2"
        )
    if not found:
        print("No overlapping coplanar faces found.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
