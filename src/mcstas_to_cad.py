# Author: Daniel Lomholt Christensen @NBI Spring 2026
#
# The point of this python script is to load in a mcstas file,
# or a mcstasscript python script, and then create a CAD model of their Union
# environments.
#

import argparse
from preprocess import preprocess
from signed_distance_functions import build_sdfs
from meshing import build_all_meshes
from bounding_box import compute_all_world_bboxes

# ==============================================================================
# ============================ PARSE ARGUMENTS =================================
# ==============================================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", help="Input mcstas file, can either be mcstasscript or mcstas", type=str
    )
    parser.add_argument("--out_file", help="Name of output file.", default="union_env")
    parser.add_argument(
        "--resolution", help="Grid fineness for marching cubes algorithm", default=64
    )
    parser.add_argument(
        "--n_points",
        help="Number of points on geometries when plotting point clouds",
        default=1_000,
    )
    parser.add_argument(
        "--mesher",
        default="brep",
        help="Chosen mesher. Possibilities are: mc (Marching cubes), dc (Dual contouring), brep (Boundary representations)",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--export", action="store_true", default=False)
    parser.add_argument(
        "--force_pygen",
        action="store_true",
        default=False,
        help=(
            "Always translate a .instr input through mcstas-pygen instead of "
            "mcstasscript's lightweight .instr reader (that reader's parse "
            "failures already fall back to mcstas-pygen automatically)."
        ),
    )
    return parser


# =============================================================================
# =========================== MAIN CODE EXECUTION =============================
# =============================================================================


def convert(
    input_file,
    out_file="union_env",
    resolution=64,
    n_points=1_000,
    mesher="brep",
    verbose=False,
    export=False,
    force_pygen=False,
):
    """Build (and optionally export) the Union geometry for an instrument."""
    # n_points remains part of the public interface for compatibility with the
    # original script, even though the current mesh builders do not use it.
    del n_points
    clip = {
        "enable": False,
        "axis": "X",
        "mode": "Above",
        "position": 0,
    }

    instr, world_matrices, union_geometries = preprocess(
        input_file, verbose, force_pygen=force_pygen
    )
    world_bboxes = compute_all_world_bboxes(union_geometries, world_matrices)
    final_sdfs, sdfs = build_sdfs(
        union_geometries, world_matrices, world_bboxes=world_bboxes
    )
    return build_all_meshes(
        union_geometries,
        world_matrices,
        sdfs,
        final_sdfs,
        resolution,
        clip,
        out_file=out_file,
        export=export,
        mesher=mesher,
        world_bboxes=world_bboxes,
    )


def main(argv=None):
    parser = parse()
    args = parser.parse_args(argv)
    return convert(
        input_file=args.input_file,
        out_file=args.out_file,
        resolution=args.resolution,
        n_points=args.n_points,
        mesher=args.mesher,
        verbose=args.verbose,
        export=args.export,
        force_pygen=args.force_pygen,
    )


if __name__ == "__main__":
    main()
