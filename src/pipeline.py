"""The worker-process side of the viewer: preprocessing and meshing a
Union instrument, tracing it with mcrun, and loading detector counts.
Returns plain picklable data and never touches Qt or pygfx."""

from preprocess import preprocess, instrument_parameters
from clipping import resolve_clip_frame
from signed_distance_functions import build_sdfs
from meshing import build_all_meshes, build_mesh, DEFAULT_BREP_DEFLECTION
from brep import build_many_brep_meshes
from bounding_box import compute_all_world_bboxes, component_dependency_signature
from gui_helpers import group_meshes_by_material, is_vacuum_material
from mcstas_trace import trace_instrument, component_types
import logger_output


def compute_mesh_data(
    input_file,
    clip,
    meshes=None,
    dependencies=None,
    mesher="brep",
    force_remesh=False,
    res=64,
    verbose=True,
    group_by_material=False,
    deflection=DEFAULT_BREP_DEFLECTION,
    force_pygen=False,
    param_values=None,
):
    """Everything a mesh reload needs to do that doesn't touch pygfx/Qt:
    preprocessing, SDF/BREP mesh building (with dependency-based incremental
    rebuild), and material grouping/vacuum tagging. Returns plain,
    picklable data (trimesh objects + dicts), so this is safe to run in a
    worker *process* rather than just a thread - some of the geometry
    kernel calls it makes (OpenCASCADE, via pythonocc-core) don't release
    the GIL, so running them on a QThread still freezes the GUI."""
    instr, world_matrices, union_geometries = preprocess(
        input_file,
        verbose=False,
        force_pygen=force_pygen,
        param_values=param_values,
    )
    clip = resolve_clip_frame(clip, world_matrices)
    world_bboxes = compute_all_world_bboxes(union_geometries, world_matrices)
    # The "brep" mesher never touches sdfs/final_sdfs, so skip building them.
    if mesher == "brep":
        final_sdfs, sdfs = {}, {}
    else:
        print("Building sdfs")
        final_sdfs, sdfs = build_sdfs(
            union_geometries, world_matrices, clip, world_bboxes=world_bboxes
        )
    print("Building meshes")

    new_dependencies = {
        name: component_dependency_signature(name, union_geometries, world_bboxes)
        for name in union_geometries
    }

    if meshes is None or force_remesh:
        meshes = build_all_meshes(
            union_geometries,
            world_matrices,
            sdfs,
            final_sdfs,
            res,
            clip,
            export=False,
            verbose=False,
            mesher=mesher,
            deflection=deflection,
            world_bboxes=world_bboxes,
        )
    else:
        changed_names = [
            name for name in new_dependencies
            if new_dependencies[name] != (dependencies or {}).get(name)
        ]
        if verbose:
            for name in changed_names:
                print(f"Rebuilding {name}")
        if mesher == "brep":
            # Each changed component's boolean-cut chain is independent -
            # fan them out across worker processes instead of rebuilding
            # one at a time (see build_many_brep_meshes).
            meshes.update(
                build_many_brep_meshes(
                    changed_names, union_geometries, world_matrices, clip,
                    verbose, deflection=deflection, world_bboxes=world_bboxes,
                )
            )
        else:
            for name in changed_names:
                meshes[name] = build_mesh(
                    union_geometries[name],
                    union_geometries,
                    world_matrices,
                    sdfs,
                    final_sdfs,
                    res,
                    clip,
                    mesher=mesher,
                    deflection=deflection,
                    world_bboxes=world_bboxes,
                )
    dependencies = new_dependencies

    # One entry per component, or per material if grouped (trimesh
    # concatenation, not a boolean fusion). Vacuum is not filtered here -
    # Viewer.apply_geometry_visibility hides it client-side instead.
    if group_by_material:
        render_meshes = group_meshes_by_material(union_geometries, meshes)
        geometry_is_vacuum = {material: is_vacuum_material(material) for material in render_meshes}
    else:
        render_meshes = {
            name: meshes[name]
            for name in union_geometries
            if meshes.get(name) is not None
        }
        geometry_is_vacuum = {
            name: is_vacuum_material(getattr(union_geometries[name], "material_string", None))
            for name in render_meshes
        }

    instrument_info = {
        "parameters": instrument_parameters(instr),
        "world_matrices": {name: M.tolist() for name, M in world_matrices.items()},
    }
    return (render_meshes, geometry_is_vacuum, meshes, dependencies, instrument_info)


# The viewer clips Union meshes with pygfx clipping planes like everything
# else, so its meshers never cut and a clip change never remeshes.
NO_CLIP = {"enable": False, "axis": "X", "mode": "Above", "position": 0.0}


def compute_trace_data(input_file, force_pygen, params, ncount, seed):
    """Run input_file through mcrun --trace. Worker-process side of
    TraceWorker, like compute_mesh_data for MeshBuildWorker."""
    return trace_instrument(
        input_file,
        params=params,
        force_pygen=force_pygen,
        ncount=ncount,
        seed=seed,
        # trace_instrument's own max_rays default (1000) exists to bound a
        # file loaded independently of any particular run; here ncount is
        # exactly how many rays the user asked mcrun to simulate, so all of
        # them should be kept rather than silently truncated at 1000.
        max_rays=max(ncount, 1),
    )


def compute_counts_data(run_folder, input_file, force_pygen):
    """Load run_folder's spatially-placeable detector output (Union
    loggers/abs_loggers and ordinary McStas monitors alike) as a
    {name: LoggerCounts} dict. Worker-process side of CountsWorker, like
    compute_trace_data for TraceWorker."""
    types = component_types(input_file, force_pygen)
    return logger_output.load_counts(run_folder, types)
