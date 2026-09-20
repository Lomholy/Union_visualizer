
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
        mesh = trimesh.load(comp.filename)
        bounds = mesh.bounds  # (min, max)
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
