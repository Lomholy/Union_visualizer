"""Tests for the installed ``unviz`` command dispatcher."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import cli  # noqa: E402


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
                    "--out_file",
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


if __name__ == "__main__":
    unittest.main()
