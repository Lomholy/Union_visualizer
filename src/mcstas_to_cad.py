# Author: Daniel Lomholt Christensen @NBI Spring 2026
#
# The point of this python script is to load in a mcstas file,
# or a mcstasscript python script, and then create a CAD model of their Union
# environments.
#
# The general algorithm is then this:
#  0. Load in the mcstas file, and preprocess to extract the union geometries.
#  1. For each geometry create a point cloud.
#  2. For each point in the point clouds, calculate which geometries it lies within.
#  3. Assign and discard points from geometries according to the priority algorithm
#  4. Use the poisson algorithm from a mesh from the point clouds
#  5. Export the entire environment as a .stl file

import numpy as np
import ast
import trimesh
import plotly.graph_objects as go
import argparse
import mcstasscript as ms
import mcstasscript.helper.mcstas_objects as mshelp
from skimage.measure import marching_cubes
import operator
import math



# ==============================================================================
# ============================ PARSE ARGUMENTS =================================
# ==============================================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", help="Input mcstas file, can either be mcstasscript or mcstas"
    )
    parser.add_argument("--out_file", help="Name of output file.", default="union_env")
    parser.add_argument(
        "--n_points", help="Number of points on geometries", default=10_000
    )
    parser.add_argument("--plot_point_cloud", action="store_true", default=False)
    parser.add_argument("--dont_save_vacuum", action="store_true", default=False)
    return parser


# ==============================================================================
# ============================ Load in McStas file =============================
# ==============================================================================


def execute_mcstasscript_file(input_file):
    # Execute the input python file and extract instrument objects
    namespace = {}
    with open(input_file, "r") as f:
        code = f.read()

    exec(compile(code, input_file, "exec"), namespace)

    # Collect all mcstasscript instruments
    instruments = [
        obj for obj in namespace.values() if isinstance(obj, ms.McStas_instr)
    ]

    if len(instruments) != 1:
        raise ValueError(f"Expected exactly one instrument, found {len(instruments)}.")

    return instruments[0]


def get_union_geometries(instr: ms.McStas_instr):
    union_geometries = []
    for comp in instr.component_list:
        if comp.component_name == "Union_cylinder":
            union_geometries.append(comp)
        elif comp.component_name == "Union_box":
            union_geometries.append(comp)
        elif comp.component_name == "Union_sphere":
            union_geometries.append(comp)
        elif comp.component_name == "Union_cone":
            union_geometries.append(comp)
        elif comp.component_name == "Union_mesh":
            union_geometries.append(comp)
    return union_geometries


def load_McStas_file(input_file):
    if input_file.endswith(".py"):
        instr = execute_mcstasscript_file(input_file)
    elif input_file.endswith(".instr"):
        file = ms.McStas_file(input_file)
        instr = ms.McStas_instr("union_cad")
        file.add_to_instr(instr)
    return instr


# =============================================================================
# ====================== CONVERT PARAMETERS TO FLOATS =========================
# =============================================================================



# Supported operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

# Supported unary operators
UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed math functions/constants
MATH_ENV = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}


def eval_expr(expr, var_map=None):
    if var_map is None:
        var_map = {}

    def _eval(node):
        if isinstance(node, ast.Constant):  # numbers
            return node.value

        elif isinstance(node, ast.BinOp):  # x + y
            return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))

        elif isinstance(node, ast.UnaryOp):  # -x
            return UNARY[type(node.op)](_eval(node.operand))

        elif isinstance(node, ast.Name):  # variables
            if node.id in var_map:
                return var_map[node.id]
            elif node.id in MATH_ENV:
                return MATH_ENV[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.Call):  # function calls
            func = _eval(node.func)
            args = [_eval(arg) for arg in node.args]
            return func(*args)

        else:
            raise TypeError(f"Unsupported expression: {expr}")

    tree = ast.parse(expr, mode='eval')
    return _eval(tree.body)


def parse_param(expr, var_map):
    try:
        return eval_expr(expr, var_map)
    except Exception:
        return expr



def attempt_conversion(comp: mshelp.Component, instr: ms.McStas_instr):
    all_vars = list(instr.declare_list) + list(instr.user_var_list) + list(instr.parameters)
    var_map = {v.name: v.value for v in all_vars}

    for name in (a for a in dir(comp) if not a.startswith("__")):
        value = getattr(comp, name)
        if isinstance(value, (str, int, float)):
            val = parse_param(value, var_map)
            setattr(comp, name, val)
        elif isinstance(value, (list, tuple, set)):
            converted = []

            for val in value:
                res = parse_param(val, var_map)
                converted.append(res)

            setattr(comp, name, type(value)(converted))
    return comp


# =============================================================================
# =========================== MAKE POINT CLOUDS ===============================
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


def sample_union_cylinder(comp: mshelp.Component, n_points):
    # Sample a fraction on the sides, and a fraction on the ends
    radius = comp.radius
    height = comp.yheight
    frac_side = int(n_points * 0.6)
    frac_ends = int(n_points * 0.4)

    theta = np.random.uniform(0, 2 * np.pi, frac_side)
    y = np.random.uniform(-height / 2, height / 2, frac_side)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    tmp = np.column_stack((x, y, z))

    theta = np.random.uniform(0, 2 * np.pi, frac_ends)
    rand_rad = np.random.rand(frac_ends) * radius
    y = height / 2 * np.random.choice((-1, 1), size=frac_ends)
    x = np.cos(theta) * rand_rad
    z = np.sin(theta) * rand_rad
    tmp_2 = np.column_stack((x, y, z))

    return np.row_stack((tmp, tmp_2))


def sample_union_box(comp: mshelp.Component, n_points):
    size_x = comp.xwidth
    size_y = comp.yheight
    size_z = comp.zdepth
    points = []
    faces = [
        (np.array([1, 0, 0]), size_x / 2),
        (np.array([-1, 0, 0]), size_x / 2),
        (np.array([0, 1, 0]), size_y / 2),
        (np.array([0, -1, 0]), size_y / 2),
        (np.array([0, 0, 1]), size_z / 2),
        (np.array([0, 0, -1]), size_z / 2),
    ]
    for _ in range(n_points):
        normal, d = faces[np.random.randint(6)]
        if normal[0]:
            y = np.random.uniform(-size_y / 2, size_y / 2)
            z = np.random.uniform(-size_z / 2, size_z / 2)
            points.append([normal[0] * d, y, z])
        elif normal[1]:
            x = np.random.uniform(-size_x / 2, size_x / 2)
            z = np.random.uniform(-size_z / 2, size_z / 2)
            points.append([x, normal[1] * d, z])
        else:
            x = np.random.uniform(-size_x / 2, size_x / 2)
            y = np.random.uniform(-size_y / 2, size_y / 2)
            points.append([x, y, normal[2] * d])
    return np.array(points)


def sample_union_sphere(comp: mshelp.Component, n_points):
    radius = comp.radius
    phi = np.random.uniform(0, 2 * np.pi, n_points)
    costheta = np.random.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack((x, y, z))


def sample_union_cone(comp: mshelp.Component, n_points):
    radius_bottom = comp.radius_bottom
    radius_top = comp.radius_top
    height = comp.yheight
    frac_side = int(n_points * 0.6)
    frac_ends = int(n_points * 0.4)
    # Truncated cone (frustum) surface sampling
    y = np.random.uniform(-height / 2, height / 2, frac_side)
    t = (y + height / 2) / height  # 0 → bottom, 1 → top
    r = radius_bottom * (1 - t) + radius_top * t

    theta = np.random.uniform(0, 2 * np.pi, frac_side)
    x = r * np.cos(theta)
    z = r * np.sin(theta)
    tmp_pts = np.column_stack((x, y, z))

    theta = np.random.uniform(0, 2 * np.pi, frac_ends)
    y = np.random.choice([-height / 2, height / 2], frac_ends)
    rand_rad = np.where(y > 0, radius_top, radius_bottom)
    rand_rad *= np.random.rand(frac_ends)

    x = rand_rad * np.cos(theta)
    z = rand_rad * np.sin(theta)
    tmp_pts_2 = np.column_stack((x, y, z))

    return np.concatenate((tmp_pts, tmp_pts_2))


def sample_union_mesh(comp: mshelp.Component, n_points):
    mesh_file = comp.filename
    if mesh_file.endswith(".stl") or mesh_file.endswith(".off"):
        mesh = trimesh.load(mesh_file)
    else:
        raise ValueError("Unsupported mesh format")
    if n_points < len(mesh.vertices):
        print(
            "Warning! Loading mesh with more vertices than requested number of points."
            + "Using number of vertices instead!"
        )
        n_points = len(mesh.vertices)
    pcd = trimesh.sample.sample_surface(mesh, n_points)
    return np.asarray(pcd.points)





GEOMETRY_FUNCS = {
    "union_cylinder": {
        "sdf": sdf_cylinder,
        "generate_surface_points": sample_union_cylinder,
    },
    "union_box": {
        "sdf": sdf_box,
        "generate_surface_points": sample_union_box,
    },
    "union_sphere": {
        "sdf": sdf_sphere,
        "generate_surface_points": sample_union_sphere,
    },
    "union_cone": {
        "sdf": sdf_cone,
        "generate_surface_points": sample_union_cone,
    },
    "union_mesh": {
        "sdf": sdf_mesh,
        "generate_surface_points": sample_union_mesh,
    },
}


def sample_sdf_surfaces(geometries, final_sdfs, world_matrices, n_points, steps=10):
    """
    For each geometry:
    - generate initial samples
    - project onto sdf=0 surface (Newton-style)
    - filter valid points

    Returns: list of arrays (points per geometry)
    """

    clouds = []

    for comp in geometries:
        print(f"SAMPLING {comp.name}")

        # initial samples (world space)
        sampler = GEOMETRY_FUNCS[comp.component_name.lower()]["generate_surface_points"]

        pts = sampler(comp, n_points)
        print(pts)
        pts = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        pts = (world_matrices[comp.name] @ pts.T).T

        f = final_sdfs[comp.name]

        # --- project to surface ---
        for _ in range(steps):
            sdf_vals = f(pts)

            grads = sdf_normal(f, pts)

            pts -= sdf_vals[:, None] * grads  # Newton projection

        # --- keep only near-surface points ---
        sdf_vals = f(pts)
        mask = np.abs(sdf_vals) < 1e-6

        points = pts[mask]
        # points = pts

        clouds.append(points)

    return clouds

#
# =============================================================================
# =========================== MARCHING CUBES PIPELINE =========================
# =============================================================================


def compute_local_bbox(comp):
    """Return local AABB (min, max) in local coordinates."""

    t = comp.component_name.lower()

    if t == "union_box":
        b = np.array([comp.xwidth, comp.yheight, comp.zdepth]) / 2
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


def sdf_to_mesh(sdf_func, bbox_min, bbox_max, resolution=64):
    xs = np.linspace(bbox_min[0], bbox_max[0], resolution)
    ys = np.linspace(bbox_min[1], bbox_max[1], resolution)
    zs = np.linspace(bbox_min[2], bbox_max[2], resolution)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    pts = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)

    sdf_vals = sdf_func(pts).reshape((resolution, resolution, resolution))

    verts, faces, normals, _ = marching_cubes(sdf_vals, level=0.0)

    # map voxel coords to world coords
    scale = (bbox_max - bbox_min) / (resolution - 1)
    verts = verts * scale + bbox_min

    return verts, faces


# =============================================================================
# =========================== GRADIENT OF SIGNED DIST FNC =====================
# =============================================================================


def sdf_normal(sdf_func, pts, eps=1e-5):
    grads = np.zeros((pts.shape[0], 4))
    dp = np.zeros_like(pts)

    for i in range(3):
        dp = np.zeros_like(pts)
        dp[:, i] = eps

        f_plus = sdf_func(pts + dp)
        f_minus = sdf_func(pts - dp)

        grads[:, i] = (f_plus - f_minus) / (2 * eps)

    # normalize per point
    norms = np.linalg.norm(grads, axis=1, keepdims=True) + 1e-12
    grads = grads / norms
    return grads


# =============================================================================
# =========================== CALCULATE WORLD MATRICES ========================
# =============================================================================


def compute_world_matrices(instr):
    """
    Returns:
        dict: {component_name_lower: 4x4 world matrix}
    """

    def rotation_matrix(rx, ry, rz):
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1],
            ]
        )
        return Rx @ Ry @ Rz

    def find_relative(comp, instr):
        AT_rel = comp.AT_relative
        ROT_rel = comp.ROTATED_relative

        if AT_rel.startswith("RELATIVE"):
            AT_rel = AT_rel.split(" ")[1]
        if ROT_rel.startswith("RELATIVE"):
            ROT_rel = ROT_rel.split(" ")[1]

        if AT_rel.lower().startswith("previous"):
            idx = instr.component_list.index(comp)
            AT_rel = instr.component_list[idx - 1].name

        if ROT_rel.lower() != "absolute":
            return ROT_rel.lower()

        return AT_rel.lower()

    def local_matrix(comp):
        M = np.eye(4)
        rx, ry, rz = np.array(comp.ROTATED_data) * np.pi / 180
        M[:3, :3] = rotation_matrix(rx, ry, rz)
        M[:3, 3] = comp.AT_data
        return M

    world = {}
    remaining = list(instr.component_list)

    while remaining:
        progressed = False

        for comp in remaining[:]:
            rel = find_relative(comp, instr)

            # Absolute → no dependency
            if rel == "absolute":
                world[comp.name] = local_matrix(comp)
                remaining.remove(comp)
                progressed = True
                continue

            # Relative → need parent first
            if rel in world:
                world[comp.name] = world[rel] @ local_matrix(comp)
                print(f"COMPONENT {comp.name} MATRIX IS : \n{world[comp.name]}")
                remaining.remove(comp)
                progressed = True

        if not progressed:
            missing = [c.name for c in remaining]
            raise RuntimeError(f"Unresolved relative transforms: {missing}")

    return world


# =============================================================================
# =========================== PLOT OF POINT CLOUD WITH PLOTLY =================
# =============================================================================

def plot_multiple_clouds(cloud_list, geoms, size=3):
    fig = go.Figure()

    for i in range(len(cloud_list)):
        # if geoms[i].component_name != "Union_cone":
        #     continue
        pts = cloud_list[i]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=size),
                name=f"Cloud {i}",
                opacity=0.7,
            )
        )

    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    fig.show()


# =============================================================================
# =========================== PRIORITY CHANGE OF POINTS========================
# =============================================================================


def prioritise_points(point_clouds, sdfs, final_sdfs, geometries, world_matrices):
    K = len(point_clouds)

    P = 0
    for x in point_clouds:
        if x.shape[0] > P:
            P = x.shape[0]
    final_points = np.zeros((P * K, 4, K))
    final_points_tracker = np.zeros(K, dtype=int)

    for i in range(K):
        comp_i = geometries[i]
        point_world = point_clouds[i]
        p = point_clouds[i]
        reassigned = np.zeros(point_world.shape[0])

        for j in range(K):
            if i == j:
                continue

            comp_j = geometries[j]
            print(comp_j.name)
            f_j = sdfs[comp_j.name]

            val_j = f_j(p)
            print(val_j)
            mask = np.where(val_j < 0, True, False)
            print(mask.sum())
            reassigned += mask
            print(reassigned.sum())
            added_pts = point_world[mask]

            if comp_i.priority > comp_j.priority:
                final_points[
                    final_points_tracker[j] : final_points_tracker[j]
                    + added_pts.shape[0],
                    :,
                    j,
                ] = added_pts
                final_points_tracker[j] += added_pts.shape[0]
        mask = np.where(reassigned == 0, True, False)
        print(f"Points not added are being added {mask.sum()}")
        final_points[
            final_points_tracker[i] : final_points_tracker[i] + point_world.shape[0],
            :,
            i,
        ] = point_world
        final_points_tracker[i] += point_world.shape[0]
        print(f"Processed geometry {i}")
    clouds = []
    for i in range(K):
        # Do A final wipe, to remove any points not on the edge of the final sdf
        sdf_fin = final_sdfs[geometries[i].name]
        p = final_points[: final_points_tracker[i], :, i]
        vals = sdf_fin(p)
        mask = np.where(abs(vals) < 1e-3, True, False)
        cloud = p[mask]
        clouds.append(cloud)

    return clouds


# =============================================================================
# =========================== MAIN CODE EXECUTION =============================
# =============================================================================

def convert_instrument_to_stl(input_file, out_file, res=64, dont_save_vacuum=True, plot_point_cloud=False, n_points = 10000):
    instr = load_McStas_file(input_file)
    for comp in instr.component_list:
        comp = attempt_conversion(comp, instr)

    world_matrices = compute_world_matrices(instr)
    union_geometries = get_union_geometries(instr)
    sdfs = {}
    final_sdfs = {}
    for comp in union_geometries:
        comp_type = comp.component_name.lower()
        sdf = GEOMETRY_FUNCS[comp_type]["sdf"]
        inv_mat = np.linalg.inv(world_matrices[comp.name])
        sdfs[comp.name] = make_sdf(comp, sdf, inv_mat)

    for comp in union_geometries:
        higher_comps = [
            sdfs[x.name] for x in union_geometries if x.priority > comp.priority and x.component_name != "Union_mesh"
        ]
        final_sdfs[comp.name] = sdf_subtract_all(sdfs[comp.name], higher_comps)

    if plot_point_cloud:
        point_clouds = sample_sdf_surfaces(
            union_geometries, final_sdfs, world_matrices, n_points
        )
        point_clouds = prioritise_points(
            point_clouds, sdfs, final_sdfs, union_geometries, world_matrices
        )
        plot_multiple_clouds(point_clouds, union_geometries)
    meshes = []
    for i, comp in enumerate(union_geometries):
        name = comp.name
        print(name, i)
        if comp.material_string == "Vacuum" and not dont_save_vacuum:
            continue
        if comp.component_name == "Union_mesh":
            mesh = trimesh.load_mesh(comp.filename.strip('"'))
            if comp.coordinate_scale is None:
                comp.coordinate_scale = 1e-3
            print(comp.coordinate_scale)
            mesh.apply_scale(float(comp.coordinate_scale))
            mesh.apply_transform(world_matrices[comp.name])
            mesh.export(f"{out_file}_{comp.name}.stl")
            meshes.append(mesh)
            continue
        
        sdf_func = final_sdfs[name]
        bmin, bmax = compute_world_bbox(comp, world_matrices)
        print(f"BBOX {name}: {bmin} → {bmax}")

        verts, faces = sdf_to_mesh(sdf_func, bmin, bmax, resolution=res)
        if verts is None:
            print("verts is none")
            continue
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(f"{out_file}_{comp.name}.stl")
        meshes.append(mesh)
    comb_mesh = trimesh.util.concatenate(meshes)
    comb_mesh.export(f"{out_file}.stl")


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    convert_instrument_to_stl(args.input_file, args.out_file)

