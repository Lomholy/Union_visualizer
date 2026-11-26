"""Find Union geometries whose flat faces lie in the same plane and overlap.

Coincident faces between two Union volumes make the ray tracing in
Union_master ambiguous: a ray hitting the shared plane can end up in the
wrong volume. This reports every pair of volumes with overlapping triangles
in a common plane, whether they point the same way or opposite ways.

The triangles come from Union's own mesher: the polyhedra Union_master draws
for each volume under mcrun --trace=2, which keep only the parts of a
volume's surface where it has the highest priority. Only triangles in a
volume's flat faces are compared (box faces, cylinder and cone end caps,
mesh faces), so tessellated curved surfaces cannot match by accident.
Volumes with visualize=0 and masks are not drawn by Union, so not checked.

Usage:
    python -m coplanar my_instrument.instr
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

from preprocess import box_dimensions


def _is_drawn(comp):
    if getattr(comp, "mask_string", None) is not None:
        return False
    try:
        return float(getattr(comp, "visualize", None)) != 0
    except (TypeError, ValueError):
        return True


def union_mesher_meshes(input_file, instr, union_geometries, world_matrices):
    """{volume name: world-space trimesh} from the polyhedra Union_master
    draws. They carry no names; each is matched to the next drawn volume,
    in instrument order, whose world bbox holds all its vertices, since a
    fully hidden volume draws nothing."""
    from bounding_box import compute_all_world_bboxes
    from mcstas_trace import _polyhedron, instr_file_for, read_positions, run_trace

    masters = {c.name for c in instr.component_list if c.component_name == "Union_master"}
    text = run_trace(
        instr_file_for(input_file), cwd=os.path.dirname(os.path.abspath(input_file))
    )
    matrices = read_positions(text)
    polyhedra = []
    drawing = None
    for line in text.splitlines():
        if line.startswith("MCDISPLAY: component "):
            drawing = line[len("MCDISPLAY: component "):].strip()
        elif line.startswith("MCDISPLAY: polyhedron") and drawing in masters:
            mesh = _polyhedron(line[len("MCDISPLAY: polyhedron"):].strip())
            if mesh is not None:
                polyhedra.append(mesh.apply_transform(matrices[drawing]))
        elif line.startswith("ENTER:"):
            break

    volumes = {n: c for n, c in union_geometries.items() if _is_drawn(c)}
    bboxes = compute_all_world_bboxes(volumes, world_matrices)
    names = list(volumes)
    meshes = {}
    for mesh in polyhedra:
        while names:
            name = names.pop(0)
            low, high = bboxes[name]
            if np.all(mesh.bounds[0] >= low - 1e-6) and np.all(mesh.bounds[1] <= high + 1e-6):
                meshes[name] = mesh
                break
    return meshes


def flat_planes(comp, matrix):
    """(normals, points) of a volume's flat faces in world space, or None
    when every face is flat (Union_mesh)."""
    t = comp.component_name.lower()
    R, c = matrix[:3, :3], matrix[:3, 3]
    if t == "union_mesh":
        return None
    if t in ("union_cylinder", "union_cone"):
        axis = R[:, 1]
        h = float(comp.yheight) / 2
        return np.array([axis, axis]), np.array([c - h * axis, c + h * axis])
    if t == "union_box":
        x1, y1, x2, y2 = box_dimensions(comp)
        z = float(comp.zdepth) / 2
        corners = np.array(
            [[sx * x / 2, sy * y / 2, sz * z]
             for sz, x, y in ((-1, x1, y1), (1, x2, y2))
             for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        ) @ R.T + c
        faces = [[0, 1, 2], [4, 5, 6], [0, 1, 5], [3, 2, 6], [0, 3, 7], [1, 2, 6]]
        tri = corners[faces]
        return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri[:, 0]
    return np.zeros((0, 3)), np.zeros((0, 3))


def _plane_keys(normals, points, tol):
    """Unit normals with a fixed sign convention, and the integer bucket key
    of each (normal, offset) plane."""
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    flip = normals[np.arange(len(normals)), np.argmax(np.abs(normals), axis=1)] < 0
    normals[flip] *= -1
    offsets = np.einsum("ij,ij->i", normals, points)
    # ponytail: planes are bucketed by rounding, so two planes that
    # straddle a rounding boundary are missed; compare neighbouring
    # buckets if that ever matters.
    return np.round(np.column_stack([normals, offsets]) / tol).astype(np.int64)


def _plane_basis(normal):
    u = np.cross(normal, [1.0, 0, 0] if abs(normal[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(normal, u)])


def _triangles_overlap(a, b, eps):
    """Do two 2D triangles (3, 2) overlap with positive area? Separating axis
    test over the six edge normals; touching along an edge is not overlap."""
    for tri in (a, b):
        edges = np.roll(tri, -1, axis=0) - tri
        for axis in np.column_stack([-edges[:, 1], edges[:, 0]]):
            pa, pb = a @ axis, b @ axis
            gap = eps * np.linalg.norm(axis)
            if pa.max() <= pb.min() + gap or pb.max() <= pa.min() + gap:
                return False
    return True


def find_coplanar_faces(meshes, planes=None, tol=1e-9):
    """Return [(name_a, name_b, normal, offset, n_triangle_pairs)] for every
    pair of meshes with overlapping triangles in a common plane.

    planes: {name: (normals, points)} restricting a mesh to triangles in
    those planes; a missing name or None keeps all its triangles.
    tol is the distance (m) within which two planes count as the same plane,
    and also the tolerance on the components of their unit normals."""
    planes = planes or {}
    buckets = defaultdict(lambda: defaultdict(list))
    for name, mesh in meshes.items():
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        keep = np.linalg.norm(normals, axis=1) > 0
        triangles, normals = triangles[keep], normals[keep]
        keys = _plane_keys(normals, triangles[:, 0], tol)
        allowed = planes.get(name)
        if allowed is not None:
            allowed = set(map(tuple, _plane_keys(*allowed, tol))) if len(allowed[0]) else set()
        for key, tri in zip(map(tuple, keys), triangles):
            if allowed is None or key in allowed:
                buckets[key][name].append(tri)

    found = []
    for key, per_volume in buckets.items():
        names = sorted(per_volume)
        if len(names) < 2:
            continue
        normal = np.array(key[:3]) * tol
        basis = _plane_basis(normal / np.linalg.norm(normal))
        flat = {n: [t @ basis.T for t in per_volume[n]] for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                # ponytail: all triangle pairs in one plane, O(n*m); add a 2D
                # grid if huge flat meshes get slow.
                hits = sum(
                    _triangles_overlap(ta, tb, tol) for ta in flat[a] for tb in flat[b]
                )
                if hits:
                    found.append((a, b, normal, key[3] * tol, hits))
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

    instr, world_matrices, union_geometries = preprocess(args.input_file, verbose=False)
    meshes = union_mesher_meshes(args.input_file, instr, union_geometries, world_matrices)
    planes = {n: flat_planes(union_geometries[n], world_matrices[n]) for n in meshes}
    found = find_coplanar_faces(meshes, planes, tol=args.tol)
    for a, b, normal, offset, hits in found:
        print(
            f"{a} and {b}: coplanar faces, normal {np.round(normal, 6)}, "
            f"offset {offset:.6g} m, {hits} overlapping triangle pairs"
        )
    if not found:
        print("No overlapping coplanar faces found.")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
