
import numpy as np
import trimesh

from preprocess import box_dimensions


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
        # meshing.py/brep.py both apply coordinate_scale (defaulting to
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


def component_dependency_signature(name, union_geometries, world_bboxes):
    """A comparable value for what this component's mesh depends on: its
    own world bbox, the (name, bbox) of every higher-priority component
    whose bbox currently overlaps it, and the same for whatever currently
    masks it (masks are matched by name, not bbox). Equal values across two
    calls mean this component does not need remeshing.

    Geometry-only: a shape parameter change is caught via its effect on
    the bbox, not compared directly. Does not detect a Union_mesh's file
    changing to different geometry with the same bounding box."""
    comp = union_geometries[name]
    own_min, own_max = world_bboxes[name]

    higher_priority_names = [
        c.name for c in union_geometries.values() if c.priority > comp.priority
    ]
    cutters = tuple(
        sorted(
            (n, tuple(world_bboxes[n][0]), tuple(world_bboxes[n][1]))
            for n in overlapping(name, higher_priority_names, world_bboxes)
        )
    )

    mask_string = getattr(comp, "mask_string", None)
    masking_names = [
        c.name
        for c in union_geometries.values()
        if getattr(c, "mask_string", None) and name in c.mask_string
    ]
    masks = tuple(
        sorted(
            (n, tuple(world_bboxes[n][0]), tuple(world_bboxes[n][1]))
            for n in masking_names
        )
    )

    return (
        tuple(own_min),
        tuple(own_max),
        cutters,
        masks,
        mask_string,
        getattr(comp, "mask_setting", None),
    )
