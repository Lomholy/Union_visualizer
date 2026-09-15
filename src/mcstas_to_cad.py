# Author: Daniel Lomholt Christensen @NBI Spring 2026
#
# The point of this python script is to load in a mcstas file,
# or a mcstasscript python script, and then create a CAD model of their Union
# environments.
#

import argparse
from preprocess import preprocess
from signed_distance_functions import build_sdfs
from plot_union_cloud import plot_point_clouds
from meshing import build_all_meshes

# ==============================================================================
# ============================ PARSE ARGUMENTS =================================
# ==============================================================================


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", help="Input mcstas file, can either be mcstasscript or mcstas"
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
    parser.add_argument("--plot_point_cloud", action="store_true", default=False)
    parser.add_argument(
        "--mesher",
        default="brep",
        help="Chosen mesher. Possibilities are: mc (Marching cubes), dc (Dual contouring), brep (Boundary representations)",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--export", action="store_true", default=False)
    return parser


# =============================================================================
# =========================== MAIN CODE EXECUTION =============================
# =============================================================================


if __name__ == "__main__":
    parser = parse()
    args = parser.parse_args()
    input_file = args.input_file
    out_file = args.out_file
    plot_point_cloud = args.plot_point_cloud
    n_points = args.n_points
    verbose = args.verbose
    res = args.resolution
    export = args.export
    mesher = args.mesher
    clip = {
            "enable": False,
            "axis": "X",
            "mode": "Above",
            "position": 0,
        }

    instr, world_matrices, union_geometries = preprocess(input_file, verbose)
    final_sdfs, sdfs = build_sdfs(union_geometries, world_matrices)
    if plot_point_cloud:
        plot_point_clouds(union_geometries, sdfs, final_sdfs, world_matrices, n_points)

    build_all_meshes(
        union_geometries,
        world_matrices,
        sdfs,
        final_sdfs,
        res,
        clip,
        out_file=out_file,
        export=export,
        mesher=mesher,

    )
