import trimesh
import numpy as np
from skimage.measure import marching_cubes
from plot_union_cloud import sdf_normal
from bounding_box import compute_world_bbox
from dual_cont import build_mesh_dual
from brep import build_brep_meshes, build_single_brep_mesh


# What each mesher's dock controls actually do.
MESHER_CAPABILITIES = {
    "mc": {
        "resolution": True,
        "deflection": False,
        "clip": True,
        "incremental_rebuild": True,
    },
    "dc": {
        "resolution": False,
        "deflection": False,
        "clip": True,
        "incremental_rebuild": False,
    },
    "brep": {
        "resolution": False,
        "deflection": True,
        "clip": True,
        "incremental_rebuild": True,
    },
}

# The deflection BRepMesh_IncrementalMesh used before it became a parameter.
DEFAULT_BREP_DEFLECTION = 0.01


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

    verts, faces, normals, _ = marching_cubes(sdf_vals, level=0.0)

    # map voxel coords to world coords
    scale = (bbox_max - bbox_min) / (resolution - 1)
    verts = verts * scale + bbox_min

    return verts, faces


def build_mesh(
    comp,
    union_geometries,
    world_matrices,
    sdfs,
    final_sdfs,
    res,
    clip,
    out_file="",
    export=True,
    verbose=False,
    mesher="mc",
    deflection=DEFAULT_BREP_DEFLECTION,
    world_bboxes=None,
):
    if verbose:
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
    bmin, bmax = compute_world_bbox(comp, world_matrices)
    if verbose:
        print(f"BBOX {name}: {bmin} → {bmax}")
    if mesher == "dc":
        return
    if mesher == "brep":
        mesh = build_single_brep_mesh(
            comp, union_geometries, world_matrices, clip, verbose,
            deflection=deflection, world_bboxes=world_bboxes,
        )
        return mesh

    # Only the "mc" mesher below needs final_sdfs, so callers building for
    # "brep"/"dc" can leave it unpopulated.
    sdf_func = final_sdfs[name]
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
        return


def build_all_meshes(
    union_geometries,
    world_matrices,
    sdfs,
    final_sdfs,
    res,
    clip,
    out_file="",
    export=True,
    verbose=False,
    mesher="brep",
    deflection=DEFAULT_BREP_DEFLECTION,
    world_bboxes=None,
):
    meshes_dict = {}
    if mesher == "dc":
        meshes_dict = build_mesh_dual(
            union_geometries,
            sdfs,
            final_sdfs,
            world_matrices,
            out_file,
            meshes_dict,
        )
    if mesher == "brep":
        meshes_dict = build_brep_meshes(
            union_geometries, world_matrices, clip, verbose, deflection=deflection,
            world_bboxes=world_bboxes,
        )
    for name, comp in union_geometries.items():
        print(mesher)
        if mesher != "mc":
            break
        if verbose:
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
            meshes_dict[name] = mesh
            continue
        if mesher != "mc":
            continue
        sdf_func = final_sdfs[name]
        bmin, bmax = compute_world_bbox(comp, world_matrices)
        if verbose:
            print(f"BBOX {name}: {bmin} → {bmax}")

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
            meshes_dict[comp.name] = mesh
        except Exception as e:
            print(f"Error building mesh on {name}:")
            print(e)

    if export:
        for name, mesh in meshes_dict.items():
            if mesh is None:
                continue
            mesh.export(f"{out_file}_{name}.stl")
        comb_mesh = trimesh.util.concatenate(list(meshes_dict.values()))
        comb_mesh.export(f"{out_file}.stl")

    return meshes_dict
