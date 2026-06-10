import numpy as np
import trimesh
import plotly.graph_objects as go
import mcstasscript.helper.mcstas_objects as mshelp
from signed_distance_functions import *



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
            f_j = sdfs[comp_j.name]

            val_j = f_j(p)
            mask = np.where(val_j < 0, True, False)
            reassigned += mask
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


def plot_point_clouds(union_geometries, sdfs, final_sdfs, world_matrices, n_points):
    point_clouds = sample_sdf_surfaces(
        union_geometries, final_sdfs, world_matrices, n_points
    )
    point_clouds = prioritise_points(
        point_clouds, sdfs, final_sdfs, union_geometries, world_matrices
    )
    plot_multiple_clouds(point_clouds, union_geometries)
