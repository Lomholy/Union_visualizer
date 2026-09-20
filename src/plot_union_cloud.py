from signed_distance_functions import *
from preprocess import box_dimensions
import mcstasscript.helper.mcstas_objects as mshelp
import trimesh
import numpy as np

np.random.seed(42)


def sample_union_cylinder(comp: mshelp.Component, n_points):
    # Sample a fraction on the sides, and a fraction on the ends
    radius = comp.radius
    height = comp.yheight
    frac_side = int(np.sqrt(n_points * 0.6))
    frac_ends = int(np.sqrt(n_points * 0.4))
    # Sample sides:

    theta = np.linspace(0, 2 * np.pi, frac_side)
    y = np.linspace(-height / 2, height / 2, frac_side)
    theta, y = np.meshgrid(theta, y)
    theta = np.ravel(theta)
    y = np.ravel(y)
    x = radius * np.cos(theta)
    z = radius * np.sin(theta)
    tmp = np.column_stack((x, y, z))

    # Sample top and bottom

    theta = np.linspace(0, 2 * np.pi, frac_ends)
    rand_rad = np.linspace(0, radius, frac_ends)
    theta, rand_rad = np.meshgrid(theta, rand_rad)
    theta = theta.ravel()
    rand_rad = rand_rad.ravel()
    x = np.cos(theta) * rand_rad
    z = np.sin(theta) * rand_rad

    y = np.ones_like(theta)
    tmp_2 = np.column_stack((x, y, z))
    tmp_3 = np.column_stack((x, -y, z))

    return np.vstack((tmp, tmp_2, tmp_3))


def sample_union_box(comp: mshelp.Component, n_points):
    """Sample the six faces of a Union_box. The box may be tapered
    (xwidth2/yheight2), in which case the four side faces are slanted quads
    and the two z-caps have different sizes, so each face is parameterised by
    its own (u, v) rather than by one shared set of half-extents."""
    x1, y1, x2, y2 = box_dimensions(comp)
    zdepth = float(comp.zdepth)

    # At least 2 samples per direction, so a face is spanned rather than
    # collapsed to a single point for small n_points.
    per_side = max(int(np.sqrt(n_points / 6)), 2)
    u = np.linspace(0.0, 1.0, per_side)
    u, v = np.meshgrid(u, u, indexing="ij")
    u = u.ravel()
    v = v.ravel()

    # u runs from the -z face to the +z face along the four side faces, so
    # the half-extents at that depth interpolate with it.
    z = (u - 0.5) * zdepth
    half_x = 0.5 * (x1 + u * (x2 - x1))
    half_y = 0.5 * (y1 + u * (y2 - y1))
    span = 2 * v - 1  # -1 .. +1 across the face

    faces = [
        np.column_stack((half_x, span * half_y, z)),
        np.column_stack((-half_x, span * half_y, z)),
        np.column_stack((span * half_x, half_y, z)),
        np.column_stack((span * half_x, -half_y, z)),
    ]

    # The two z-caps, each at its own cross-section.
    for width, height, z_face in ((x1, y1, -zdepth / 2), (x2, y2, zdepth / 2)):
        faces.append(
            np.column_stack(
                (
                    (2 * u - 1) * width / 2,
                    span * height / 2,
                    np.full_like(u, z_face),
                )
            )
        )

    return np.vstack(faces)


def sample_union_sphere(comp: mshelp.Component, n_points):
    points = int(np.sqrt(n_points))
    radius = comp.radius
    phi = np.linspace(0, 2 * np.pi, points)
    costheta = np.linspace(-1, 1, points)
    theta = np.arccos(costheta)
    theta, phi = np.meshgrid(theta, phi)
    theta = theta.ravel()
    phi = phi.ravel()
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.column_stack((x, y, z))


def sample_union_cone(comp: mshelp.Component, n_points):
    radius_bottom = comp.radius_bottom
    radius_top = comp.radius_top
    height = comp.yheight
    frac_side = int(np.sqrt(n_points * 0.6))
    frac_ends = int(np.sqrt(n_points * 0.4))
    # Truncated cone (frustum) surface sampling
    y = np.linspace(-height / 2, height / 2, frac_side)

    theta = np.linspace(0, 2 * np.pi, frac_side)
    y, theta = np.meshgrid(y, theta)
    y = y.ravel()
    theta = theta.ravel()
    t = (y + height / 2) / height  # 0 → bottom, 1 → top
    r = radius_bottom * (1 - t) + radius_top * t

    x = r * np.cos(theta)
    z = r * np.sin(theta)
    tmp_pts = np.column_stack((x, y, z))

    rad = np.linspace(0, 1, frac_ends)
    theta = np.linspace(0, 2 * np.pi, frac_ends)
    rad, theta = np.meshgrid(rad, theta)
    rad = rad.ravel()
    theta = theta.ravel()

    y = np.ones_like(rad) * height / 2
    radtop = rad * radius_top
    radbot = rad * radius_bottom

    x = radbot * np.cos(theta)
    z = radbot * np.sin(theta)
    tmp_pts_2 = np.column_stack((x, -y, z))
    x = radtop * np.cos(theta)
    z = radtop * np.sin(theta)
    tmp_pts_3 = np.column_stack((x, y, z))

    return np.concatenate((tmp_pts, tmp_pts_2, tmp_pts_3))


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
    n = pts.shape[0]

    offsets = np.array(
        [
            [eps, 0.0, 0.0, 0.0],
            [-eps, 0.0, 0.0, 0.0],
            [0.0, eps, 0.0, 0.0],
            [0.0, -eps, 0.0, 0.0],
            [0.0, 0.0, eps, 0.0],
            [0.0, 0.0, -eps, 0.0],
        ]
    )

    samples = (pts[:, None, :] + offsets[None, :, :]).reshape(-1, 4)

    vals = sdf_func(samples).reshape(n, 6)

    grads = np.empty((n, 4))
    grads[:, 0] = vals[:, 0] - vals[:, 1]
    grads[:, 1] = vals[:, 2] - vals[:, 3]
    grads[:, 2] = vals[:, 4] - vals[:, 5]
    grads[:, 3] = 0.0

    grads[:, :3] /= np.linalg.norm(grads[:, :3], axis=1, keepdims=True) + 1e-12

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


def sample_sdf_surfaces(
    geometries, final_sdfs, world_matrices, n_points, steps=10, verbose=False
):
    """
    For each geometry:
    - generate initial samples
    - project onto sdf=0 surface (Newton-style)
    - filter valid points

    Returns: list of arrays (points per geometry)
    """

    clouds = {}

    for name, comp in geometries.items():
        if verbose:
            print(f"SAMPLING {comp.name}")

        # initial samples (world space)
        sampler = GEOMETRY_FUNCS[comp.component_name.lower()]["generate_surface_points"]

        pts = sampler(comp, n_points)
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

        clouds[name] = points

    return clouds


def prioritise_points(
    point_clouds, sdfs, final_sdfs, geometries, world_matrices, verbose=False
):
    K = len(point_clouds.keys())
    final_points = {}
    final_points_tracker = {}
    P = 0
    for x in point_clouds.values():
        if x.shape[0] > P:
            P = x.shape[0]

    for name, comp in geometries.items():
        final_points[name] = np.zeros((P * K, 4))
        final_points_tracker[name] = 0
    pairs = []

    for name, comp in geometries.items():
        comp = geometries[name]
        point_world = point_clouds[name]
        p = point_clouds[name]
        reassigned = np.zeros(point_world.shape[0])

        for name2 in geometries.keys():
            if (name, name2) in pairs:
                continue
            pairs.append((name, name2))
            comp_j = geometries[name2]
            f_j = sdfs[name2]

            val_j = f_j(p)
            mask = np.where(val_j < 0, True, False)
            reassigned += mask
            added_pts = point_world[mask]

            if comp.priority > comp_j.priority:
                final_points[name2][
                    final_points_tracker[name2] : final_points_tracker[name2]
                    + added_pts.shape[0],
                    :,
                ] = added_pts
                final_points_tracker[name2] += added_pts.shape[0]
        mask = np.where(reassigned == 0, True, False)
        final_points[name][
            final_points_tracker[name] : final_points_tracker[name]
            + point_world.shape[0],
            :,
        ] = point_world
        final_points_tracker[name] += point_world.shape[0]
        if verbose:
            print(f"Processed geometry {name}")
    clouds = {}
    for name in geometries.keys():
        # Do A final wipe, to remove any points not on the edge of the final sdf
        sdf_fin = final_sdfs[name]
        p = final_points[name][: final_points_tracker[name], :]
        vals = sdf_fin(p)
        mask = np.where(abs(vals) < 1e-3, True, False)
        cloud = p[mask]
        clouds[name] = cloud

    return clouds


def generate_points(
    union_geometries, sdfs, final_sdfs, world_matrices, n_points, verbose=False
):
    point_clouds = sample_sdf_surfaces(
        union_geometries, final_sdfs, world_matrices, n_points, verbose
    )
    point_clouds = prioritise_points(
        point_clouds, sdfs, final_sdfs, union_geometries, world_matrices, verbose
    )
    return point_clouds
