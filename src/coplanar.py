"""Find Union geometries with coincident surfaces.

Coincident surfaces between two Union volumes make the ray tracing in
Union_master ambiguous: a ray hitting the shared surface can end up in the
wrong volume. Two kinds are reported, whichever way the surfaces face:

Flat faces that lie in a common plane and overlap. The triangles come from
Union's own mesher: the polyhedra Union_master draws for each volume under
mcrun --trace=2, which keep only the parts of a volume's surface where it
has the highest priority. Only triangles in a volume's flat faces are
compared (box faces, cylinder and cone end caps, mesh faces), so
tessellated curved surfaces cannot match by accident. Masks and
visualize=0 volumes are not drawn by Union; their uncarved shapes are
meshed here instead.

Curved surfaces that coincide over a positive area: cylinder and cone
sides on the same axis line with the same radius along their shared
length, and spheres with the same centre and radius. These are compared
on the geometry parameters, for every volume and mask, so a surface hidden
by a higher priority volume is still reported.

Usage:
    python -m coplanar my_instrument.instr
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import trimesh

from preprocess import box_dimensions


def _is_drawn(comp):
    if getattr(comp, "mask_string", None) is not None:
        return False
    try:
        return float(getattr(comp, "visualize", None)) != 0
    except (TypeError, ValueError):
        return True


def union_mesher_meshes(input_file, instr, union_geometries, world_matrices):
    """({volume name: world-space trimesh}, {volume name: vertex precision
    in m}) from the polyhedra Union_master draws. Union prints vertices with
    %g, 6 significant digits, so a vertex is only good to 5e-6 times the
    largest coordinate in the master frame. The polyhedra carry no names;
    each is matched to the next drawn volume, in instrument order, whose
    world bbox holds all its vertices, since a fully hidden volume draws
    nothing."""
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
                precision = 5e-6 * np.abs(mesh.vertices).max() + 1e-12
                polyhedra.append((mesh.apply_transform(matrices[drawing]), precision))
        elif line.startswith("ENTER:"):
            break

    volumes = {n: c for n, c in union_geometries.items() if _is_drawn(c)}
    bboxes = compute_all_world_bboxes(volumes, world_matrices)
    names = list(volumes)
    meshes, precisions = {}, {}
    for mesh, precision in polyhedra:
        fits = [
            i for i, n in enumerate(names)
            if np.all(mesh.bounds[0] >= bboxes[n][0] - precision)
            and np.all(mesh.bounds[1] <= bboxes[n][1] + precision)
        ]
        if not fits:
            print("Warning: a polyhedron drawn by Union_master matches no volume; skipped.")
            continue
        name = names[fits[0]]
        names = names[fits[0] + 1:]
        meshes[name], precisions[name] = mesh, precision
    return meshes, precisions


def _box_corners(comp):
    """Local corners of a Union_box: the -z face, then the +z face."""
    x1, y1, x2, y2 = box_dimensions(comp)
    z = float(comp.zdepth) / 2
    return np.array(
        [[sx * x / 2, sy * y / 2, sz * z]
         for sz, x, y in ((-1, x1, y1), (1, x2, y2))
         for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    )


def _end_radii(comp):
    """(radius at local y = -h/2, radius at y = +h/2) of a cylinder or cone."""
    if comp.component_name.lower() == "union_cylinder":
        return float(comp.radius), float(comp.radius)
    return float(comp.radius_bottom), float(comp.radius_top)


def primitive_mesh(comp, matrix, segments=64):
    """World-space mesh of a volume's uncarved shape, for volumes Union does
    not draw. None for a sphere, which has no flat faces."""
    t = comp.component_name.lower()
    if t == "union_box":
        mesh = trimesh.convex.convex_hull(_box_corners(comp))
    elif t in ("union_cylinder", "union_cone"):
        h = float(comp.yheight) / 2
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        rings = [
            np.column_stack([r * np.cos(angles), np.full(segments, y), r * np.sin(angles)])
            for r, y in zip(_end_radii(comp), (-h, h))
        ]
        mesh = trimesh.convex.convex_hull(np.vstack(rings))
    elif t == "union_mesh":
        scale = float(comp.coordinate_scale) if comp.coordinate_scale is not None else 1e-3
        mesh = trimesh.load(comp.filename, force="mesh")
        mesh.apply_scale(scale)
    else:
        return None
    return mesh.apply_transform(matrix)


def flat_planes(comp, matrix):
    """(normals, points) of a volume's flat faces in world space, taken from
    its geometry parameters (or mesh file) rather than from drawn
    triangles, so they are exact."""
    t = comp.component_name.lower()
    R, c = matrix[:3, :3], matrix[:3, 3]
    if t == "union_mesh":
        tri = primitive_mesh(comp, matrix).triangles
        return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri[:, 0]
    if t in ("union_cylinder", "union_cone"):
        axis = R[:, 1]
        h = float(comp.yheight) / 2
        return np.array([axis, axis]), np.array([c - h * axis, c + h * axis])
    if t == "union_box":
        corners = _box_corners(comp) @ R.T + c
        faces = [[0, 1, 2], [4, 5, 6], [0, 1, 5], [3, 2, 6], [0, 3, 7], [1, 2, 6]]
        tri = corners[faces]
        return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri[:, 0]
    return np.zeros((0, 3)), np.zeros((0, 3))


def _shared_side_length(a, b, tol):
    """Length along the axis over which the sides of two cylinders or cones
    coincide, or 0. Each is (comp, world matrix)."""
    (ca, ma), (cb, mb) = a, b
    axis_a, axis_b = ma[:3, 1], mb[:3, 1]
    if np.linalg.norm(np.cross(axis_a, axis_b)) > tol:
        return 0.0
    d = mb[:3, 3] - ma[:3, 3]
    s_b = d @ axis_a
    if np.linalg.norm(d - s_b * axis_a) > tol:
        return 0.0
    ha, hb = float(ca.yheight) / 2, float(cb.yheight) / 2
    lo, hi = max(-ha, s_b - hb), min(ha, s_b + hb)
    if hi - lo <= tol:
        return 0.0
    ra0, ra1 = _end_radii(ca)
    rb0, rb1 = _end_radii(cb)
    sign = 1.0 if axis_a @ axis_b > 0 else -1.0

    def radius_a(s):
        return ra0 + (ra1 - ra0) * (s + ha) / (2 * ha)

    def radius_b(s):
        return rb0 + (rb1 - rb0) * (sign * (s - s_b) + hb) / (2 * hb)

    if all(abs(radius_a(s) - radius_b(s)) <= tol for s in (lo, hi)):
        return hi - lo
    return 0.0


def find_coincident_curved(union_geometries, world_matrices, tol=1e-9):
    """Return [(name_a, name_b, description)] for every pair of volumes or
    masks whose curved surfaces coincide over a positive area."""
    lathes = ("union_cylinder", "union_cone")
    items = [(n, c, c.component_name.lower()) for n, c in union_geometries.items()]
    found = []
    for i, (na, ca, ta) in enumerate(items):
        for nb, cb, tb in items[i + 1:]:
            ma, mb = world_matrices[na], world_matrices[nb]
            if ta in lathes and tb in lathes:
                length = _shared_side_length((ca, ma), (cb, mb), tol)
                if length:
                    found.append((na, nb, f"sides coincide over {length:.6g} m of axis"))
            elif ta == tb == "union_sphere":
                same_centre = np.linalg.norm(ma[:3, 3] - mb[:3, 3]) <= tol
                if same_centre and abs(float(ca.radius) - float(cb.radius)) <= tol:
                    found.append((na, nb, "spheres coincide"))
    return found


def _planes(normals, points, tol):
    """Unit normals with a fixed sign convention, offsets n . p, and the
    integer bucket key of each plane; zero-length normals are dropped."""
    lengths = np.linalg.norm(normals, axis=1)
    normals, points = normals[lengths > 0] / lengths[lengths > 0, None], points[lengths > 0]
    flip = normals[np.arange(len(normals)), np.argmax(np.abs(normals), axis=1)] < 0
    normals[flip] *= -1
    offsets = np.einsum("ij,ij->i", normals, points)
    # ponytail: planes are bucketed by rounding, so two planes that
    # straddle a rounding boundary are missed; compare neighbouring
    # buckets if that ever matters.
    keys = np.round(np.column_stack([normals, offsets]) / tol).astype(np.int64)
    return normals, offsets, keys


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


def find_coplanar_faces(meshes, planes=None, tol=1e-9, precisions=None):
    """Return [(name_a, name_b, normal, offset)] for every pair of meshes
    with overlapping triangles in a common plane.

    planes: {name: (normals, points)}, the exact planes of each mesh's flat
    faces; only triangles lying in them are compared, and the planes, not
    the triangles, decide which faces share a plane. A missing name uses
    the planes of its own triangles.
    precisions: {name: m}, how far a mesh's vertices may lie from their
    true plane (default tol).
    tol is the distance (m) within which two planes count as the same plane,
    and also the tolerance on the components of their unit normals."""
    planes, precisions = planes or {}, precisions or {}
    buckets = defaultdict(lambda: defaultdict(list))
    plane_of = {}
    for name, mesh in meshes.items():
        triangles = np.asarray(mesh.triangles)
        own = (np.asarray(mesh.face_normals), triangles[:, 0])
        normals, offsets, keys = _planes(*planes.get(name, own), tol)
        snap = precisions.get(name, tol)
        free = np.ones(len(triangles), dtype=bool)
        # ponytail: one pass over the triangles per plane, O(planes * triangles);
        # fine for Union's primitives, slow only for huge Union_mesh files.
        for i in np.unique(keys, axis=0, return_index=True)[1]:
            on = free & np.all(np.abs(triangles @ normals[i] - offsets[i]) <= snap, axis=1)
            free &= ~on
            if on.any():
                key = tuple(keys[i])
                buckets[key][name].extend(triangles[on])
                plane_of[key] = (normals[i], offsets[i])

    found = []
    for key, per_volume in buckets.items():
        names = sorted(per_volume)
        if len(names) < 2:
            continue
        normal, offset = plane_of[key]
        basis = _plane_basis(normal)
        flat = {n: [t @ basis.T for t in per_volume[n]] for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                eps = max(precisions.get(a, tol), precisions.get(b, tol))
                # ponytail: all triangle pairs in one plane, O(n*m); add a 2D
                # grid if huge flat meshes get slow.
                if any(_triangles_overlap(ta, tb, eps) for ta in flat[a] for tb in flat[b]):
                    found.append((a, b, normal, offset))
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
    meshes, precisions = union_mesher_meshes(
        args.input_file, instr, union_geometries, world_matrices
    )
    for name, comp in union_geometries.items():
        if not _is_drawn(comp):
            mesh = primitive_mesh(comp, world_matrices[name])
            if mesh is not None:
                meshes[name] = mesh
    planes = {n: flat_planes(union_geometries[n], world_matrices[n]) for n in meshes}
    flat = find_coplanar_faces(meshes, planes, tol=args.tol, precisions=precisions)
    for a, b, normal, offset in flat:
        print(f"{a} and {b}: coplanar faces, normal {np.round(normal, 6)}, offset {offset:.9g} m")
    curved = find_coincident_curved(union_geometries, world_matrices, tol=args.tol)
    for a, b, description in curved:
        print(f"{a} and {b}: curved {description}")
    if not flat and not curved:
        print("No coincident surfaces found.")
    return 1 if flat or curved else 0


if __name__ == "__main__":
    sys.exit(main())
