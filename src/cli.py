"""Command-line entry point for the Union visualizer."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unviz",
        description=(
            "Open the interactive Union visualizer, or export a McStas "
            "Union environment as a CAD mesh."
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
    # Keep the implementation in mcstas_to_cad so its historical script entry
    # point and this installed command exercise exactly the same code path.
    from mcstas_to_cad import convert

    convert(
        input_file=input_file,
        out_file=args.out_file,
        resolution=args.resolution,
        n_points=args.n_points,
        mesher=args.mesher,
        verbose=args.verbose,
        export=True,
        force_pygen=args.force_pygen,
    )


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
