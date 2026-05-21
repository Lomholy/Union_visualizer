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
#

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
    parser.add_argument("--n_points", help="Number of points on geometries", default=30_000)
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


def sample_union_cylinder(comp: mshelp.Component, n_points):
    radius = comp.radius
    height = comp.yheight

    theta = np.random.uniform(0, 2 * np.pi, n_points)
    z = np.random.uniform(-height / 2, height / 2, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y, z))


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


def generate_point_clouds(geometries, instr, world_matrices, n_points):
    point_clouds = np.zeros((n_points, 4, len(geometries)))
    for i, geometry in enumerate(geometries):
        sample_func = GEOMETRY_FUNCS[geometry.component_name.lower()]["generate_surface_points"]
        point_clouds[:, :3, i] = sample_func(geometry, n_points)
        point_clouds[:, 3, i] = 1
        print(world_matrices[geometry.name.lower()].shape, point_clouds[:, :, i].shape)
        point_clouds[:, :, i] = (
            world_matrices[geometry.name.lower()] @ point_clouds[:, :, i].T
        ).T
    return point_clouds


def cylinder_point_within(comp, point):
    x, y, z = point
    r2 = x**2 + y**2

    return (
        r2 <= comp.radius**2 and
        -comp.yheight / 2 <= z <= comp.yheight / 2
    )


def box_point_within(comp, point):
    x, y, z = point

    return (
        -comp.xwidth / 2 <= x <= comp.xwidth / 2 and
        -comp.yheight / 2 <= y <= comp.yheight / 2 and
        -comp.zdepth / 2 <= z <= comp.zdepth / 2
    )


def sphere_point_within(comp, point):
    x, y, z = point
    return x**2 + y**2 + z**2 <= comp.radius**2


def cone_point_within(comp, point):
    x, y, z = point

    if not (-comp.yheight / 2 <= z <= comp.yheight / 2):
        return False

    # interpolation factor
    t = (z + comp.yheight / 2) / comp.yheight

    # radius at this height
    r = comp.radius_bottom * (1 - t) + comp.radius_top * t

    return x**2 + y**2 <= r**2


def mesh_point_within(comp, point):
    if not hasattr(comp, "_mesh"):
        comp._mesh = trimesh.load(comp.filename)

    return comp._mesh.contains([point])[0]


GEOMETRY_FUNCS = {
    "union_cylinder": {
        "point_within": cylinder_point_within,
        "generate_surface_points": sample_union_cylinder,
    },
    "union_box": {
        "point_within": box_point_within,
        "generate_surface_points": sample_union_box,
    },
    "union_sphere": {
        "point_within": sphere_point_within,
        "generate_surface_points": sample_union_sphere,
    },
    "union_cone": {
        "point_within": cone_point_within,
        "generate_surface_points": sample_union_cone,
    },
    "union_mesh": {
        "point_within": mesh_point_within,
        "generate_surface_points": sample_union_mesh,
    },
}
# =============================================================================
# =========================== PRIORITY CHANGE OF POINTS========================
# =============================================================================

def prioritise_points(point_clouds, geometries, world_matrices):
    """
    Re-assign points between geometries based on containment and priority.

    Parameters
    ----------
    point_clouds : np.ndarray
        Shape (N, 4, K)
    geometries : list
        List of component objects
    world_matrices : dict
        Mapping from geometry.name.lower() -> 4x4 transform

    Returns
    -------
    np.ndarray
        Updated point_clouds
    """

    P, D, K = point_clouds.shape

    # Precompute inverse transforms for efficiency
    inv_world_matrices = {
        name: np.linalg.inv(mat)
        for name, mat in world_matrices.items()
    }
    # Make a P*K X 4 X K matrix to store the points in
    final_points = np.zeros((P*K, D, K))

    final_points_tracker = np.zeros(K, dtype=int)
    # Loop over each geometry's points
    for i in range(K):
        comp_i = geometries[i]
        for p_idx in range(point_clouds.shape[0]):
            point_world = point_clouds[p_idx, :, i]
            reassigned = False

            # Check against ALL other geometries
            for j in range(K):
                if i == j:
                    continue

                comp_j = geometries[j]
                name_j = comp_j.name.lower()
                ops_j = GEOMETRY_FUNCS[comp_j.component_name.lower()]

                # Transform point into geometry j's LOCAL frame
                point_local_j = inv_world_matrices[name_j] @ point_world
                # print(point_local_j)
                # Check containment
                if ops_j["point_within"](comp_j, point_local_j[:3]):
                    reassigned = True
                    if comp_i.priority > comp_j.priority:
                        #  The original priority is higher: keep point 
                        # and add it to the other geometry as well
                        final_points[final_points_tracker[i], :, i] = point_world
                        final_points[final_points_tracker[j], :, j] = point_world
                        final_points_tracker[i] += 1
                        final_points_tracker[j] += 1

            if reassigned is False:
                final_points[final_points_tracker[i], :, i] = point_world
                final_points_tracker[i] += 1


        print(f"Processed geometry {i}")

    # Make final points into a list of arrays
    clouds = []
    for i in range(K):
        cloud = final_points[:final_points_tracker[i],:,i]
        clouds.append(cloud)

    return clouds


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
                print(f"COMPONENT {comp.name} MATRIX IS : {world[comp.name.lower()]}")
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


def plot_multiple_clouds(cloud_list, size=2):
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
            )
        )

    fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))

    fig.show()


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
    point_clouds = generate_point_clouds(union_geometries, instr, world_matrices, args.n_points)
    point_clouds = prioritise_points(point_clouds, union_geometries, world_matrices)
    plot_multiple_clouds(point_clouds)

    for i in range(len(point_clouds)):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_clouds[i][:, :3])
        point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30)
        )
        point_cloud.orient_normals_consistent_tangent_plane(100)
        # o3d.visualization.draw_geometries([point_cloud], point_show_normal=True)
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud, depth=6, linear_fit=True
        )
        # radii = [0.005, 0.01, 0.02, 0.04]
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(point_cloud,o3d.utility.DoubleVector(radii))

        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        if i == 0:
            combined_mesh = mesh
        else:
            combined_mesh += mesh

        o3d.io.write_triangle_mesh(f"{args.out}_{i}.stl", mesh)

    o3d.io.write_triangle_mesh(f"{args.out}.stl", combined_mesh)
    # print(vertices.shape)
    # print(faces.shape)
    # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # mesh.export(f'{args.out}.stl')
