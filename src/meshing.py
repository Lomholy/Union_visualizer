import trimesh
import numpy as np
from skimage.measure import marching_cubes
from plot_union_cloud import sdf_normal
from bounding_box import compute_world_bbox
from dual_cont import build_mesh_dual


def make_grid(sdf, bbox_min, bbox_max, resolution):
    xs = np.linspace(bbox_min[0], bbox_max[0], resolution)
    ys = np.linspace(bbox_min[1], bbox_max[1], resolution)
    zs = np.linspace(bbox_min[2], bbox_max[2], resolution)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    pts = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return pts


def sdf_to_mesh(sdf_func, bbox_min, bbox_max, resolution=64):
    pts = make_grid(sdf_func, bbox_min, bbox_max, resolution)
    sdf_vals = sdf_func(pts).reshape((resolution, resolution, resolution))
    print(pts.min(), pts.max())

    verts, faces, normals, _ = marching_cubes(sdf_vals, level=0.0)

    # map voxel coords to world coords
    scale = (bbox_max - bbox_min) / (resolution - 1)
    verts = verts * scale + bbox_min

    return verts, faces


def build_mesh(
    comp,
    world_matrices,
    sdfs,
    final_sdfs,
    res,
    out_file="",
    export=True,
    verbose=False,
    use_dual_contouring=False,
):
    print(f"build single mesh! Building {comp.name}!")
    name = comp.name
    if comp.component_name == "Union_mesh":
        mesh = trimesh.load_mesh(comp.filename.strip('"'))
        if comp.coordinate_scale is None:
            comp.coordinate_scale = 1e-3
        if verbose:
            print(comp.coordinate_scale)
        mesh.apply_scale(float(comp.coordinate_scale))
        mesh.apply_transform(world_matrices[comp.name])
        mesh.export(f"{out_file}_{comp.name}.stl")
        return comp.name, mesh
    sdf_func = final_sdfs[name]
    bmin, bmax = compute_world_bbox(comp, world_matrices)
    if verbose:
        print(f"BBOX {name}: {bmin} → {bmax}")
    if use_dual_contouring:
        return
    try:
        verts, faces = sdf_to_mesh(
            sdf_func,
            bmin,
            bmax,
            resolution=res,
        )

        if len(verts) == 0 or len(faces) == 0:
            print(f"FUCK {comp.name}")
            return
        # Calculate the normal of each vert
        vert_norms = sdf_normal(
            sdf_func, np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
        )[:, :3]
        verts += vert_norms * 1e-4
        if verts is None:
            print("verts is none")
            return
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        if export:
            mesh.export(f"{out_file}_{comp.name}.stl")
        return mesh
    except Exception as e:
        print(e)
        exit()


def build_all_meshes(
    union_geometries,
    world_matrices,
    sdfs,
    final_sdfs,
    res,
    out_file="",
    export=True,
    verbose=True,
    use_dual_contouring=False,
):
    meshes = []
    meshes_dict = {}
    for name, comp in union_geometries.items():
        if verbose is True:
            print(f"Building {name}!")
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
            meshes_dict[name] = mesh
            continue
        sdf_func = final_sdfs[name]
        bmin, bmax = compute_world_bbox(comp, world_matrices)
        if verbose is True:
            print(f"BBOX {name}: {bmin} → {bmax}")
        if use_dual_contouring is True:
            continue
        try:
            verts, faces = sdf_to_mesh(
                sdf_func,
                bmin,
                bmax,
                resolution=res,
            )

            if len(verts) == 0 or len(faces) == 0:
                print(f"FUCK {comp.name}")
                continue

            # Calculate the normal of each vert
            vert_norms = sdf_normal(
                sdf_func, np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
            )[:, :3]
            verts += vert_norms * 1e-4
            if verts is None:
                print("verts is none")
                continue
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            if export:
                mesh.export(f"{out_file}_{comp.name}.stl")
            meshes.append(mesh)
            meshes_dict[comp.name] = mesh
        except Exception as e:
            print("Error building mesh:")
            print(e)
    if export:
        comb_mesh = trimesh.util.concatenate(meshes)
        comb_mesh.export(f"{out_file}.stl")
    if use_dual_contouring:
        build_mesh_dual(union_geometries, sdfs, final_sdfs, world_matrices, out_file)
    #
    return meshes_dict
