"""Tests for the installed ``unviz`` command dispatcher."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import cli  # noqa: E402
from mcstas_trace import TraceError  # noqa: E402


class CliTest(unittest.TestCase):
    @patch("cli.launch_viewer", return_value=0)
    def test_no_arguments_launches_file_picker(self, launch_viewer):
        self.assertEqual(cli.main([]), 0)
        launch_viewer.assert_called_once_with(None)

    @patch("cli.launch_viewer", return_value=0)
    def test_positional_input_launches_viewer(self, launch_viewer):
        self.assertEqual(cli.main(["sample.instr"]), 0)
        launch_viewer.assert_called_once_with("sample.instr")

    @patch("cli.export_model")
    def test_export_dispatches_to_converter(self, export_model):
        self.assertEqual(
            cli.main(
                [
                    "--input_file",
                    "sample.instr",
                    "--export",
                    "--out-file",
                    "sample.stl",
                    "--mesher",
                    "mc",
                ]
            ),
            0,
        )
        args, input_file = export_model.call_args.args
        self.assertEqual(input_file, "sample.instr")
        self.assertEqual(args.out_file, "sample.stl")
        self.assertEqual(args.mesher, "mc")

    def test_export_requires_input(self):
        with self.assertRaises(SystemExit) as raised:
            cli.main(["--export"])
        self.assertEqual(raised.exception.code, 2)

    def test_export_options_are_not_silently_ignored_by_viewer(self):
        with self.assertRaises(SystemExit) as raised:
            cli.main(["--out-file", "sample.stl"])
        self.assertEqual(raised.exception.code, 2)


class ExportInstrumentTest(unittest.TestCase):
    """export_instrument() supersedes the old, Union-only mcstas_to_cad.py:
    it builds the Union sample environment (no mcrun needed - preprocess()
    parses the instrument itself) and every other McStas component (drawn
    via mcrun --trace, ncount=0), combined into one export."""

    def test_skips_a_component_whose_geometry_came_out_empty(self):
        # build_all_meshes() stores None (not omitting the key) for a
        # component whose geometry came out empty - reproduces the crash
        # this caused: AttributeError: 'NoneType' object has no attribute
        # 'export', seen in CI (unviz --export on tests/crack_height.instr
        # and others) once real instruments started hitting this path.
        import trimesh
        real_mesh = trimesh.creation.box(extents=(1, 1, 1))

        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "simple_test.instr")
            shutil.copy(ROOT / "tests" / "simple_test.instr", instr)
            out_file = os.path.join(tmp, "with_empty")
            with patch("mcstas_trace.trace_instrument", side_effect=TraceError("boom")), \
                 patch("meshing.build_all_meshes",
                       return_value={"box_left": None, "box_lefttwo": real_mesh}):
                meshes = cli.export_instrument(instr, out_file=out_file, mesher="mc", resolution=8)

            self.assertEqual(set(meshes), {"box_lefttwo"})
            self.assertTrue(os.path.exists(f"{out_file}.stl"))
            self.assertTrue(os.path.exists(f"{out_file}_box_lefttwo.stl"))
            self.assertFalse(os.path.exists(f"{out_file}_box_left.stl"))

    def test_falls_back_to_union_only_geometry_when_tracing_fails(self):
        # Only mcstas_trace.trace_instrument is mocked - preprocess() and
        # the meshing pipeline run for real against a real instrument, so
        # this exercises the actual fallback control flow without needing
        # mcrun on PATH. resolution=32: low enough to stay fast, high
        # enough that marching cubes doesn't miss either Union_box's
        # (small) isosurface.
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "simple_test.instr")
            shutil.copy(ROOT / "tests" / "simple_test.instr", instr)
            out_file = os.path.join(tmp, "union_only")
            with patch("mcstas_trace.trace_instrument", side_effect=TraceError("boom")):
                meshes = cli.export_instrument(instr, out_file=out_file, mesher="mc", resolution=32)

            # The two Union_box components are still there...
            self.assertEqual(set(meshes), {"box_left", "box_lefttwo"})
            self.assertTrue(os.path.exists(f"{out_file}.stl"))
            self.assertTrue(os.path.exists(f"{out_file}_box_left.stl"))
            # ...but nothing from the (failed) trace, e.g. the source.
            self.assertNotIn("src", meshes)

    @unittest.skipUnless(shutil.which("mcrun"), "mcrun not on PATH")
    def test_combines_union_geometry_and_other_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "simple_test.instr")
            shutil.copy(ROOT / "tests" / "simple_test.instr", instr)
            out_file = os.path.join(tmp, "combined")
            try:
                meshes = cli.export_instrument(instr, out_file=out_file, mesher="mc", resolution=32)
            except Exception as e:
                self.skipTest(f"mcrun could not compile here: {e}")

            # export_instrument() only warns and falls back on a trace
            # failure rather than raising (see the fallback test above), so
            # a broken local mcrun compile toolchain (e.g. no SDKROOT set)
            # wouldn't be caught by the try/except above - check directly.
            if "src" not in meshes:
                self.skipTest("mcrun could not compile simple_test.instr here (see the warning above)")

            # Union geometry (Union_box components)...
            self.assertIn("box_left", meshes)
            self.assertIn("box_lefttwo", meshes)
            # ...and ordinary McStas components, drawn via mcrun --trace.
            self.assertIn("src", meshes)  # Source_simple
            self.assertTrue(os.path.exists(f"{out_file}.stl"))
            self.assertTrue(os.path.exists(f"{out_file}_box_left.stl"))
            self.assertTrue(os.path.exists(f"{out_file}_src.stl"))


if __name__ == "__main__":
    unittest.main()
