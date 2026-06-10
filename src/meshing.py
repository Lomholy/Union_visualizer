import trimesh
import numpy as np
from skimage.measure import marching_cubes
from plot_union_cloud import sdf_normal


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


def build_meshes(
    union_geometries,
    world_matrices,
    final_sdfs,
    res,
    out_file = "",
    dont_save_vacuum = True,
    export = True,
    verbose = False,
):
    meshes = []
    meshes_dict = {}
    for i, comp in enumerate(union_geometries):
        name = comp.name
        if verbose:
            print(name, i)
        if comp.material_string == "Vacuum" and not dont_save_vacuum:
            continue
        if comp.component_name == "Union_mesh":
            mesh = trimesh.load_mesh(comp.filename.strip('"'))
            if comp.coordinate_scale is None:
                comp.coordinate_scale = 1e-3
            if verbose:
                print(comp.coordinate_scale)
            mesh.apply_scale(float(comp.coordinate_scale))
            mesh.apply_transform(world_matrices[comp.name])
            mesh.export(f"{out_file}_{comp.name}.stl")
            meshes.append(mesh)
            continue
        sdf_func = final_sdfs[name]
        bmin, bmax = compute_world_bbox(comp, world_matrices)
        if verbose:
            print(f"BBOX {name}: {bmin} → {bmax}")

        verts, faces = sdf_to_mesh(sdf_func, bmin, bmax, resolution=res)
        # Calculate the normal of each vert
        vert_norms = sdf_normal(sdf_func, np.concatenate([verts, np.ones((len(verts), 1))], axis=1))[:, :3]
        verts += vert_norms * 1e-4
        if verts is None:
            print("verts is none")
            continue
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if export:
            mesh.export(f"{out_file}_{comp.name}.stl")
        meshes.append(mesh)
        meshes_dict[comp.name.lower()] = mesh
    if export:
        comb_mesh = trimesh.util.concatenate(meshes)
        comb_mesh.export(f"{out_file}.stl")
    return meshes_dict
