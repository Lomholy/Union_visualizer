"""Find Union components whose meshes have coinciding faces.

Coincident surfaces between two Union volumes make the ray tracing in
Union_master ambiguous: a ray hitting the shared surface can end up in the
wrong volume.
"""

from collections import defaultdict

import numpy as np


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


def find_coincident_faces(meshes, tol=1e-9):
    """Return [(name_a, name_b, [(normal, offset), ...])], one entry per
    pair of meshes with overlapping triangles in a common plane, whichever
    way they face, listing every such plane.

    meshes: {component name: world-space trimesh or None}, as built by the
    meshers. tol is the distance (m) within which two planes count as the
    same plane, and also the tolerance on the components of their unit
    normals."""
    buckets = defaultdict(lambda: defaultdict(list))
    planes = {}
    for name, mesh in meshes.items():
        if mesh is None or len(mesh.faces) == 0:
            continue
        triangles = np.asarray(mesh.triangles)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        triangles, normals = triangles[lengths > 0], normals[lengths > 0] / lengths[lengths > 0, None]
        flip = normals[np.arange(len(normals)), np.argmax(np.abs(normals), axis=1)] < 0
        normals[flip] *= -1
        offsets = np.einsum("ij,ij->i", normals, triangles[:, 0])
        # ponytail: planes are bucketed by rounding, so two planes that
        # straddle a rounding boundary are missed; compare neighbouring
        # buckets if that ever matters.
        keys = np.round(np.column_stack([normals, offsets]) / tol).astype(np.int64)
        for key, tri, normal, offset in zip(map(tuple, keys), triangles, normals, offsets):
            buckets[key][name].append(tri)
            planes.setdefault(key, (normal, offset))

    found = defaultdict(list)
    for key, per_mesh in buckets.items():
        names = sorted(per_mesh)
        if len(names) < 2:
            continue
        normal, offset = planes[key]
        basis = _plane_basis(normal)
        flat = {n: [t @ basis.T for t in per_mesh[n]] for n in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                # ponytail: all triangle pairs in one plane, O(n*m); add a 2D
                # grid if huge flat meshes get slow.
                if any(_triangles_overlap(ta, tb, tol) for ta in flat[a] for tb in flat[b]):
                    found[a, b].append((normal, offset))
    return [(a, b, shared) for (a, b), shared in sorted(found.items())]
