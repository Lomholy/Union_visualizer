import numpy as np
import trimesh


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
    b = np.array([comp.xwidth, comp.yheight, comp.zdepth]) / 2
    d = np.abs(p) - b

    outside = np.maximum(d, 0)
    inside = np.minimum(np.maximum.reduce(d, axis=-1), 0)

    return np.linalg.norm(outside, axis=-1) + inside


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


def sdf_subtract_all(f_i, higher_priority_fs):
    if not higher_priority_fs:
        return f_i

    def f_block(x):
        return np.min([f(x) for f in higher_priority_fs], axis=0)

    return lambda x: np.maximum(f_i(x), -f_block(x))


GEOMETRY_SDF = {
    "union_cylinder": {
        "sdf": sdf_cylinder,
    },
    "union_box": {
        "sdf": sdf_box,
    },
    "union_sphere": {
        "sdf": sdf_sphere,
    },
    "union_cone": {
        "sdf": sdf_cone,
    },
    "union_mesh": {
        "sdf": sdf_mesh,
    },
}


def build_sdfs(union_geometries, world_matrices):
    sdfs = {}
    final_sdfs = {}
    for comp in union_geometries:
        comp_type = comp.component_name.lower()
        sdf = GEOMETRY_SDF[comp_type]["sdf"]
        inv_mat = np.linalg.inv(world_matrices[comp.name])
        sdfs[comp.name] = make_sdf(comp, sdf, inv_mat)

    for comp in union_geometries:
        higher_comps = [
            sdfs[x.name]
            for x in union_geometries
            if x.priority > comp.priority and x.component_name != "Union_mesh"
        ]
        final_sdfs[comp.name] = sdf_subtract_all(sdfs[comp.name], higher_comps)
    return final_sdfs, sdfs


