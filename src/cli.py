"""Command-line entry point for the Union visualizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unviz",
        description=(
            "Open the interactive Union visualizer, or export a whole McStas "
            "instrument - its Union sample environment and every other "
            "component - as a CAD mesh."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        metavar="INPUT_FILE",
        help="McStas .instr file or mcstasscript Python file to open",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        dest="input_file",
        help="McStas .instr file or mcstasscript Python file to open",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the input instead of opening the interactive viewer",
    )
    parser.add_argument(
        "-o",
        "--out-file",
        dest="out_file",
        default="union_env",
        help="Exported file name (default: union_env)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
        help="Grid fineness used by the marching-cubes mesher (default: 64)",
    )
    parser.add_argument(
        "--n-points",
        dest="n_points",
        type=int,
        default=1_000,
        help="Number of points used when plotting point clouds (default: 1000)",
    )
    parser.add_argument(
        "--mesher",
        choices=("mc", "dc", "brep"),
        default="brep",
        help="Mesher used for export (default: brep)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")
    parser.add_argument(
        "--force-pygen",
        dest="force_pygen",
        action="store_true",
        help="Always translate .instr input through mcstas-pygen",
    )
    return parser


def _resolve_input(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str | None:
    if args.input and args.input_file and args.input != args.input_file:
        parser.error("input file was provided twice with different values")
    return args.input_file or args.input


def launch_viewer(input_file: str | None) -> int:
    # Import lazily so that `unviz --help` does not require the GUI stack to
    # initialise and the headless exporter does not import Qt at all.
    from union_viewer import launch

    return launch(input_file)


def export_model(args: argparse.Namespace, input_file: str) -> None:
    export_instrument(
        input_file=input_file,
        out_file=args.out_file,
        resolution=args.resolution,
        mesher=args.mesher,
        verbose=args.verbose,
        force_pygen=args.force_pygen,
    )


def export_instrument(
    input_file: str,
    out_file: str = "union_env",
    resolution: int = 64,
    mesher: str = "brep",
    verbose: bool = False,
    force_pygen: bool = False,
) -> dict:
    """Build a CAD mesh of the *whole* instrument - the Union sample
    environment plus every other McStas component along the beamline
    (sources, guides, slits, monitors, ...) - and write it to out_file.

    Writes one f"{out_file}_{name}.stl" per part plus a combined
    f"{out_file}.stl", and returns {name: trimesh.Trimesh} for every part
    that was written.

    This supersedes the old, Union-only mcstas_to_cad.py script: import
    lazily, so `unviz --help` and the interactive viewer never pull in the
    geometry-kernel/meshing stack just to parse arguments.
    """
    import trimesh
    from preprocess import preprocess
    from signed_distance_functions import build_sdfs
    from meshing import build_all_meshes
    from bounding_box import compute_all_world_bboxes
    from mcstas_trace import trace_instrument, McrunNotFoundError, TraceError

    clip = {"enable": False, "axis": "X", "mode": "Above", "position": 0}

    instr, world_matrices, union_geometries = preprocess(
        input_file, verbose, force_pygen=force_pygen
    )
    world_bboxes = compute_all_world_bboxes(union_geometries, world_matrices)
    final_sdfs, sdfs = build_sdfs(union_geometries, world_matrices, world_bboxes=world_bboxes)
    all_meshes = build_all_meshes(
        union_geometries,
        world_matrices,
        sdfs,
        final_sdfs,
        resolution,
        clip,
        out_file=out_file,
        export=False,
        verbose=verbose,
        mesher=mesher,
        world_bboxes=world_bboxes,
    )

    # Every other component along the beamline - drawn via mcrun --trace,
    # the same way the interactive viewer's "Show McStas components" does.
    # ncount=0: geometry only, no rays needed for an export. A component
    # that's part of the Union environment (Union_box, Union_cylinder, ...)
    # doesn't draw anything of its own via MCDISPLAY, so there's no overlap
    # with the meshes built above.
    try:
        components, _rays = trace_instrument(input_file, force_pygen=force_pygen, ncount=0)
        for name, component in components.items():
            mesh = component.export_mesh()
            if mesh is not None:
                all_meshes[name] = mesh
    except (McrunNotFoundError, TraceError) as e:
        print(
            f"Warning: could not draw the instrument's other McStas components "
            f"({type(e).__name__}: {e}); exporting the Union sample environment only."
        )

    if not all_meshes:
        raise RuntimeError(f"'{input_file}' has no exportable geometry.")

    for name, mesh in all_meshes.items():
        mesh.export(f"{out_file}_{name}.stl")
    combined = (
        next(iter(all_meshes.values()))
        if len(all_meshes) == 1
        else trimesh.util.concatenate(list(all_meshes.values()))
    )
    combined.export(f"{out_file}.stl")
    return all_meshes


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    input_file = _resolve_input(parser, args)

    if args.export:
        if input_file is None:
            parser.error("--export requires an input file")
        export_model(args, input_file)
        return 0

    export_only_options = (
        args.out_file != "union_env"
        or args.resolution != 64
        or args.n_points != 1_000
        or args.mesher != "brep"
        or args.verbose
        or args.force_pygen
    )
    if export_only_options:
        parser.error("export options require --export")

    return launch_viewer(input_file)


if __name__ == "__main__":
    raise SystemExit(main())
