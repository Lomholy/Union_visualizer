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
import trimesh
import open3d as o3d
import plotly.graph_objects as go
import argparse
import mcstasscript as ms
import mcstasscript.helper.mcstas_objects as mshelp


# ==============================================================================
# ============================ PARSE ARGUMENTS =================================
# ==============================================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", help="Input mcstas file, can either be mcstasscript or mcstas"
    )
    parser.add_argument("--out", help="Name of output file.", default="union_env")
    parser.add_argument("--n_points", help="Number of points on geometries", default=10_000)
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
    instr = execute_mcstasscript_file(input_file)
    return instr


# =============================================================================
# ====================== CONVERT PARAMETERS TO FLOATS =========================
# =============================================================================


def attempt_single_param_conversion(var_map, name, value, comp, instr):
    # Check dictionary of parameters
    if value in instr.parameters and instr.parameters[value] is not None:
        print(value)
        param_value = instr.parameters[value].value
        setattr(comp, name, param_value)
    elif name in var_map:
        setattr(comp, name, var_map[value])
    if type(value) == str:
        try:
            setattr(comp, name, float(value))
        except Exception:
            setattr(comp, name, value)


def attempt_iterable_conversion(var_map, name, param_iter, comp, instr):
    for i, val in enumerate(param_iter):
        # overwrite param iter, and then set comp name to param iter
        if val in instr.parameters and instr.parameters[val] is not None:
            param_iter[i] = float(instr.parameters[val].value)
        elif type(val) == str:
            # attempt to just convert to number
            try:
                param_iter[i] = float(val)
            except ValueError:
                # Value is not numeric, so keep as string
                continue
    setattr(comp, name, param_iter)


def attempt_conversion(comp: mshelp.Component, instr: ms.McStas_instr):
    all_vars = list(instr.declare_list) + list(instr.user_var_list)
    var_map = {v.name: v.value for v in all_vars}

    for name in (a for a in dir(comp) if not a.startswith("__")):
        value = getattr(comp, name)
        if isinstance(value, (str, int, float)):
            attempt_single_param_conversion(var_map, name, value, comp, instr)

        if isinstance(value, (list, tuple, set)):
            attempt_iterable_conversion(var_map, name, value, comp, instr)

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

    outside = np.maximum(d_xz, 0)**2 + np.maximum(d_y, 0)**2
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

    y = p[..., 2] + h / 2
    t = np.clip(y / h, 0, 1)

    r = r1 * (1 - t) + r2 * t
    d_xy = np.linalg.norm(p[..., :2], axis=-1) - r
    d_y = np.maximum(np.maximum(-y, y - h), 0)

    return np.maximum(d_xy, d_y)


def sdf_mesh(comp, p):
    if not hasattr(comp, "_mesh"):
        comp._mesh = trimesh.load(comp.filename)

    sdf = trimesh.proximity.signed_distance(comp._mesh, p)
    return sdf


def make_sdf(comp, sdf_func, inv_world):
    def f(x_world):
        x_local = (inv_world @ x_world.T).T
        return sdf_func(comp, x_local[:,:3])

    return f


def sdf_difference(f, g):
    return lambda x: np.maximum(f(x), -g(x))


def sdf_subtract_all(f_i, higher_priority_fs):
    if not higher_priority_fs:
        return f_i

    f_block = lambda x: np.min([f(x) for f in higher_priority_fs], axis=0)
    return lambda x: np.maximum(f_i(x), -f_block(x))


def sample_union_cylinder(comp: mshelp.Component, n_points):
    # Sample a fraction on the sides, and a fraction on the ends
    radius = comp.radius
    height = comp.yheight
    frac_side = int(n_points*0.6)
    frac_ends = int(n_points*0.4)

    theta = np.random.uniform(0, 2 * np.pi, frac_side)
    y = np.random.uniform(-height / 2, height / 2, frac_side)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    tmp = np.column_stack((x, y, z))

    theta = np.random.uniform(0, 2 * np.pi, frac_ends)
    rand_rad = np.random.rand(frac_ends)*radius
    y = height/2 * np.random.choice((-1, 1), size=frac_ends)
    x = np.cos(theta)*rand_rad
    z = np.sin(theta)*rand_rad
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
    # Truncated cone (frustum) surface sampling
    z = np.random.uniform(-height / 2, height / 2, n_points)
    t = (z + height / 2) / height  # 0 → bottom, 1 → top
    r = radius_bottom * (1 - t) + radius_top * t

    theta = np.random.uniform(0, 2 * np.pi, n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return np.column_stack((x, y, z))


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
        pts = (world_matrices[comp.name.lower()] @ pts.T).T

        # make homogeneous → world transform

        f = final_sdfs[comp.name]

        # --- project to surface ---
        for _ in range(steps):
            sdf_vals = f(pts)

            grads = sdf_normal(f, pts)

            pts -= sdf_vals[:, None] * grads  # Newton projection

        # --- keep only near-surface points ---
        sdf_vals = f(pts)
        mask = np.abs(sdf_vals) < 1e-3

        points = pts[mask]
        # points = pts

        clouds.append(points)

    return clouds

# =============================================================================
# =========================== GRADIENT OF SIGNED DIST FNC =====================
# =============================================================================


def sdf_normal(sdf_func, pts, eps=1e-5):
    grads = np.zeros((pts.shape[0],4))
    dp = np.zeros_like(pts)

    for i in range(3):
        dp = np.zeros_like(pts)
        dp[:, i] = eps

        f_plus  = sdf_func(pts + dp)
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
                world[comp.name.lower()] = local_matrix(comp)
                remaining.remove(comp)
                progressed = True
                continue

            # Relative → need parent first
            if rel in world:
                world[comp.name.lower()] = world[rel] @ local_matrix(comp)
                print(f"COMPONENT {comp.name} MATRIX IS : \n{world[comp.name.lower()]}")
                remaining.remove(comp)
                progressed = True

        if not progressed:
            missing = [c.name for c in remaining]
            raise RuntimeError(f"Unresolved relative transforms: {missing}")

    return world


# =============================================================================
# =========================== PLOT OF POINT CLOUD WITH PLOTLY =================
# =============================================================================


def plot_point_cloud(points, colors=None, size=2, title="3D Point Cloud"):
    """
    Plot a 3D point cloud using Plotly.

    Parameters
    ----------
    points : np.ndarray
        Shape (N, 3) array of XYZ coordinates.
    colors : np.ndarray or None
        Shape (N,) or (N, 3). Optional color per point.
    size : int
        Marker size.
    title : str
        Plot title.
    """

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    marker_kwargs = dict(size=size)

    if colors is not None:
        marker_kwargs["color"] = colors
        marker_kwargs["colorscale"] = "Viridis"
        marker_kwargs["colorbar"] = dict(title="Color")

    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker_kwargs)]
    )

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.show()


def plot_multiple_clouds(cloud_list, size=3):
    fig = go.Figure()

    for i in range(len(cloud_list)):
        pts = cloud_list[i]
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                marker=dict(size=size),
                name=f"Cloud {i}",
                opacity = 0.7
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
        f_i = sdfs[comp_i.name]

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
            mask = np.where(val_j<0, True, False)
            print(mask.sum())
            reassigned += mask
            print(reassigned.sum())
            added_pts = point_world[mask]

            if comp_i.priority > comp_j.priority:
                final_points[final_points_tracker[j]:final_points_tracker[j] + added_pts.shape[0], :, j] = added_pts
                final_points_tracker[j] += added_pts.shape[0]
        mask = np.where(reassigned==0, True, False)
        print(f"Points not added are being added {mask.sum()}")
        test = point_world[mask]
        print(test.shape)
        final_points[final_points_tracker[i]:final_points_tracker[i] + point_world.shape[0], :, i] = point_world
        final_points_tracker[i] += point_world.shape[0]





        print(f"Processed geometry {i}")
    
    clouds = []
    for i in range(K):
        
        # Do A final wipe, to remove any points not on the edge of the final sdf
        sdf_fin = final_sdfs[geometries[i].name]
        p = final_points[:final_points_tracker[i], :, i]
        vals = sdf_fin(p)
        mask = np.where(abs(vals)<1e-3, True, False)
        cloud = p[mask]
        clouds.append(cloud)

    return clouds

# =============================================================================
# =========================== MAIN CODE EXECUTION =============================
# =============================================================================


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    instr = load_McStas_file(args.input_file)
    for comp in instr.component_list:
        comp = attempt_conversion(comp, instr)

    world_matrices = compute_world_matrices(instr)
    union_geometries = get_union_geometries(instr)
    sdfs = {}
    final_sdfs = {}
    for comp in union_geometries:
        comp_type = comp.component_name.lower()
        sdf = GEOMETRY_FUNCS[comp_type]["sdf"]
        inv_mat = np.linalg.inv(world_matrices[comp.name.lower()])
        sdfs[comp.name] = make_sdf(comp, sdf, inv_mat)

    for comp in union_geometries:
        higher_comps = [sdfs[x.name] for x in union_geometries if x.priority > comp.priority]
        final_sdfs[comp.name] = sdf_subtract_all(sdfs[comp.name], higher_comps)
        print(final_sdfs[comp.name], comp.name)
    print(final_sdfs)
    # visualize_sdf_field(union_geometries, final_sdfs, bounds=(-1, 1))
    point_clouds = sample_sdf_surfaces(union_geometries, final_sdfs, world_matrices, args.n_points)
    # point_clouds = prioritise_points(point_clouds, sdfs, final_sdfs, union_geometries, world_matrices)
    plot_multiple_clouds(point_clouds)
    
    meshes = []
    
    for i, cloud in enumerate(point_clouds):
        if cloud.shape[0] == 0:
            continue
    
        pts = cloud[:, :3]
    
        # normals from SDF
        f = final_sdfs[union_geometries[i].name]
        normals = sdf_normal(f, cloud)
    
        # build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(normals[:, :3])
    
        # Poisson
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=6
        )
    
        mesh.compute_vertex_normals()
        meshes.append(mesh)
    
        o3d.io.write_triangle_mesh(f"{args.out}_{i}.stl", mesh)
        # NOTE!!! FAILS ON CRYOSTAT TEST!!!
    
    # print(vertices.shape)
    # print(faces.shape)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # mesh.export(f'{args.out}.stl')
