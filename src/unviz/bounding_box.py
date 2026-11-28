
import os

import numpy as np
import trimesh

from .preprocess import box_dimensions


def compute_local_bbox(comp):
    """Return local AABB (min, max) in local coordinates."""

    t = comp.component_name.lower()

    if t == "union_box":
        # A tapered box (xwidth2/yheight2) is widest at whichever of its two
        # faces is larger, so the bounding box takes the max of the pair
        # rather than the -z face alone.
        x1, y1, x2, y2 = box_dimensions(comp)
        b = np.array([max(x1, x2), max(y1, y2), float(comp.zdepth)]) / 2
        return -b, b

    elif t == "union_sphere":
        r = comp.radius
        b = np.array([r, r, r])
        return -b, b

    elif t == "union_cylinder":
        r = comp.radius
        h = comp.yheight / 2
        b = np.array([r, h, r])
        return -b, b

    elif t == "union_cone":
        r = max(comp.radius_bottom, comp.radius_top)
        h = comp.yheight / 2
        b = np.array([r, h, r])
        return -b, b

    elif t == "union_mesh":
        # union_clip.py applies coordinate_scale (defaulting to
        # 1e-3) before using a Union_mesh's geometry; without it here the
        # box is ~1000x too large for a millimetre-unit STL.
        scale = float(comp.coordinate_scale) if comp.coordinate_scale is not None else 1e-3
        mesh = trimesh.load(comp.filename)
        bounds = mesh.bounds * scale  # (min, max)
        return bounds[0], bounds[1]

    else:
        raise ValueError(f"Unknown geometry: {t}")


def transform_bbox(min_corner, max_corner, world_matrix):
    """Transform AABB to world space AABB."""

    corners = np.array(
        [
            [min_corner[0], min_corner[1], min_corner[2], 1],
            [min_corner[0], min_corner[1], max_corner[2], 1],
            [min_corner[0], max_corner[1], min_corner[2], 1],
            [min_corner[0], max_corner[1], max_corner[2], 1],
            [max_corner[0], min_corner[1], min_corner[2], 1],
            [max_corner[0], min_corner[1], max_corner[2], 1],
            [max_corner[0], max_corner[1], min_corner[2], 1],
            [max_corner[0], max_corner[1], max_corner[2], 1],
        ]
    )

    world_corners = (world_matrix @ corners.T).T[:, :3]

    return world_corners.min(axis=0), world_corners.max(axis=0)


def compute_world_bbox(comp, world_matrices, margin=0.05):
    """Compute slightly expanded world-space bounding box."""

    local_min, local_max = compute_local_bbox(comp)
    world_min, world_max = transform_bbox(
        local_min, local_max, world_matrices[comp.name]
    )

    # Expand a bit such that marching cubes can get gradient correctly
    size = world_max - world_min
    world_min -= margin * size
    world_max += margin * size

    return world_min, world_max


def compute_all_world_bboxes(union_geometries, world_matrices, margin=0.0):
    """{name: (world_min, world_max)} for every component, computed once so
    callers doing many overlap checks don't repeat mesh loads/transforms."""
    return {
        name: compute_world_bbox(comp, world_matrices, margin=margin)
        for name, comp in union_geometries.items()
    }


def boxes_overlap(a_min, a_max, b_min, b_max, tol=1e-9):
    """Do two world-space AABBs overlap (or touch, within tol)?"""
    return bool(np.all(a_min <= b_max + tol) and np.all(b_min <= a_max + tol))


def overlapping(name, candidate_names, world_bboxes, tol=1e-9):
    """candidate_names whose world bbox overlaps world_bboxes[name]."""
    a_min, a_max = world_bboxes[name]
    return [
        other
        for other in candidate_names
        if other != name
        and boxes_overlap(a_min, a_max, *world_bboxes[other], tol=tol)
    ]


# The parameters each Union shape's geometry is built from
SHAPE_PARAMETERS = {
    "union_box": ("xwidth", "yheight", "zdepth", "xwidth2", "yheight2"),
    "union_sphere": ("radius",),
    "union_cylinder": ("radius", "yheight"),
    "union_cone": ("radius", "radius_top", "radius_bottom", "yheight"),
    "union_mesh": ("filename", "coordinate_scale"),
}


def shape_signature(comp, world_matrices=None):
    """A comparable value for one component's own geometry: its type, the
    parameters its shape is built from, its placement and, for a
    Union_mesh, the modification time of its mesh file."""
    kind = comp.component_name.lower()
    params = tuple(getattr(comp, p, None) for p in SHAPE_PARAMETERS.get(kind, ()))
    placement = None
    if world_matrices is not None:
        placement = tuple(np.asarray(world_matrices[comp.name], float).ravel())
    mtime = None
    if kind == "union_mesh":
        try:
            mtime = os.path.getmtime(str(comp.filename).strip('"'))
        except OSError:
            mtime = None
    return (kind, params, placement, mtime, getattr(comp, "priority", None))


def component_dependency_signature(name, union_geometries, world_bboxes, world_matrices=None):
    """A comparable value for what this component's mesh depends on: its
    own geometry, that of every higher-priority component whose bbox
    currently overlaps it together with that component's masks, and that
    of whatever currently masks this one (masks are matched by name, not
    bbox). Geometry means each component's bbox and its shape_signature,
    so a shape change that leaves the bbox unchanged (a taper, a rotation
    of a symmetric outline, an edited mesh file) is caught too. Equal
    values across two calls mean this component does not need remeshing."""
    comp = union_geometries[name]
    own_min, own_max = world_bboxes[name]

    def geometry(c):
        return (c.name, tuple(world_bboxes[c.name][0]), tuple(world_bboxes[c.name][1]),
                shape_signature(c, world_matrices))

    def masks_of(target):
        return tuple(
            sorted(
                (geometry(c), getattr(c, "mask_setting", None))
                for c in union_geometries.values()
                if getattr(c, "mask_string", None) and target in c.mask_string
            )
        )

    higher_priority_names = [
        c.name for c in union_geometries.values() if c.priority > comp.priority
    ]
    cutters = tuple(
        sorted(
            (geometry(union_geometries[n]), masks_of(n))
            for n in overlapping(name, higher_priority_names, world_bboxes)
        )
    )

    mask_string = getattr(comp, "mask_string", None)
    masks = masks_of(name)

    return (
        tuple(own_min),
        tuple(own_max),
        shape_signature(comp, world_matrices),
        cutters,
        masks,
        mask_string,
        getattr(comp, "mask_setting", None),
    )
