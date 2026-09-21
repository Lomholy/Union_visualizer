import numpy as np
import trimesh

from preprocess import box_dimensions
from bounding_box import compute_all_world_bboxes, overlapping


# =============================================================================
# =========================== BASIC SDFS FOR EACH GEOM ========================
# =============================================================================


def sdf_cylinder(comp, p):
    # p: (N, 3) or (3,)
    r = comp.radius
    h = comp.yheight / 2

    d_xz = np.linalg.norm(p[..., 0:3:2], axis=-1) - r
    d_y = np.abs(p[..., 1]) - h

    outside = np.maximum(d_xz, 0) ** 2 + np.maximum(d_y, 0) ** 2
    inside = np.minimum(np.maximum(d_xz, d_y), 0)

    return np.sqrt(outside) + inside


def sdf_box(comp, p):
    x1, y1, x2, y2 = box_dimensions(comp)
    zdepth = float(comp.zdepth)

    if x1 == x2 and y1 == y2:
        # Plain cuboid: the exact Euclidean box SDF. Worth keeping as its own
        # branch because marching cubes' level-set crossing and sdf_normal's
        # finite-difference gradient both behave best on a true distance
        # field, and this is by far the common case.
        b = np.array([x1, y1, zdepth]) / 2
        d = np.abs(p) - b

        outside = np.maximum(d, 0)
        inside = np.minimum(np.maximum.reduce(d, axis=-1), 0)

        return np.linalg.norm(outside, axis=-1) + inside

    # Tapered box (xwidth2/yheight2): a rectangular frustum whose half-extents
    # interpolate linearly from the -z face to the +z face. This branch
    # returns a bounded rather than exact distance - the same approximation
    # sdf_cone already makes for the analogous circular frustum - so the
    # gradient is not quite unit length on the slanted faces and the
    # `verts += normal * 1e-4` nudge in meshing.py is correspondingly
    # approximate there.
    t = np.clip((p[..., 2] + zdepth / 2) / zdepth, 0, 1)

    bx = 0.5 * (x1 + t * (x2 - x1))
    by = 0.5 * (y1 + t * (y2 - y1))

    d_x = np.abs(p[..., 0]) - bx
    d_y = np.abs(p[..., 1]) - by
    d_z = np.abs(p[..., 2]) - zdepth / 2

    return np.maximum(np.maximum(d_x, d_y), d_z)


def sdf_sphere(comp, p):
    return np.linalg.norm(p, axis=-1) - comp.radius


def sdf_cone(comp, p):
    r1 = comp.radius_bottom
    r2 = comp.radius_top
    h = comp.yheight

    y = p[..., 1] + h / 2
    t = np.clip(y / h, 0, 1)

    r = r1 * (1 - t) + r2 * t
    d_xz = np.linalg.norm(p[..., 0:3:2], axis=-1) - r
    d_y = np.maximum(-y, y - h)

    return np.maximum(d_xz, d_y)


def sdf_mesh(mesh, p):
    sdf = trimesh.proximity.signed_distance(mesh, p)
    return sdf


def sdf_halfspace(axis, threshold, keep_above=True):
    """
    axis:
        0 -> X
        1 -> Y
        2 -> Z
    """

    def f(p):
        coord = p[:, axis]

        if keep_above:
            # Keep coord > threshold
            return threshold - coord

        else:
            # Keep coord < threshold
            return coord - threshold

    return f


def make_sdf(comp, sdf_func, inv_world):
    mesh = None

    if comp.component_name == "Union_mesh":
        mesh = trimesh.load(comp.filename.strip('"'))

    def f(x_world):
        x_local = (inv_world @ x_world.T).T

        if mesh is not None:
            return sdf_func(mesh, x_local[:, :3])
        else:
            return sdf_func(comp, x_local[:, :3])

    return f


def sdf_difference(f, g):
    return lambda x: np.maximum(f(x), -g(x))


def sdf_intersection(f, g):
    return lambda x: np.maximum(f(x), g(x))


def sdf_subtract_all(f_i, higher_priority_fs):
    if not higher_priority_fs:
        return f_i

    def f_block(x):
        return np.min([f(x) for f in higher_priority_fs], axis=0)

    return lambda x: np.maximum(f_i(x), -f_block(x))


GEOMETRY_SDF = {
    "Union_cylinder": {
        "sdf": sdf_cylinder,
    },
    "Union_box": {
        "sdf": sdf_box,
    },
    "Union_sphere": {
        "sdf": sdf_sphere,
    },
    "Union_cone": {
        "sdf": sdf_cone,
    },
    "Union_mesh": {
        "sdf": sdf_mesh,
    },
}


def build_sdfs(
    union_geometries,
    world_matrices,
    clip={
        "enable": False,
        "position": 0,
        "mode": "Above",
        "axis": "X",
    },
    world_bboxes=None,
):
    if world_bboxes is None:
        world_bboxes = compute_all_world_bboxes(union_geometries, world_matrices)

    sdfs = {}
    final_sdfs = {}
    for name, comp in union_geometries.items():
        comp_type = comp.component_name
        sdf = GEOMETRY_SDF[comp_type]["sdf"]
        inv_mat = np.linalg.inv(world_matrices[comp.name])
        sdfs[comp.name] = make_sdf(comp, sdf, inv_mat)

    for name, comp in union_geometries.items():
        candidates = [
            x.name
            for x in union_geometries.values()
            if x.priority > comp.priority and x.component_name != "Union_mesh"
        ]
        # A non-overlapping cutter can't affect the subtraction result.
        higher_comps = [sdfs[n] for n in overlapping(name, candidates, world_bboxes)]
        final_sdfs[name] = sdf_subtract_all(sdfs[name], higher_comps)

    if clip.get("enable", True):
        axis_map = {
            "X": 0,
            "Y": 1,
            "Z": 2,
        }
        axis = axis_map[clip["axis"]]
        threshold = clip["position"]
        keep_above = clip["mode"] == "Above"
        clip_sdf = sdf_halfspace(
            axis,
            threshold,
            keep_above,
        )
        for name in final_sdfs:
            final_sdfs[name] = sdf_intersection(
                final_sdfs[name],
                clip_sdf,
            )

    return final_sdfs, sdfs
