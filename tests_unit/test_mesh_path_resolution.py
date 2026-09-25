# Regression test for src/preprocess.py's resolution of Union_mesh
# `filename` parameters.
#
# tests/crack_height.instr references its mesh as `filename =
# "./crack_cryst.stl"`. CI (.github/workflows/run-tests.yml) invokes
# unviz --export from the repo root over every file in tests/*, so the
# process's current working directory is the repo root, not tests/ - a
# filename resolved against cwd would point at a nonexistent
# <repo-root>/crack_cryst.stl instead of the real tests/crack_cryst.stl.
# resolve_mesh_filenames fixes this by resolving relative filenames against
# the instrument file's own directory instead, matching how real McStas
# resolves such paths.
#
# Not wired into CI's tests/* loop (that loop feeds every file in tests/ to
# unviz --export as an instrument input, so a plain unittest module can't
# live there). Run directly:
#
#   python tests_unit/test_mesh_path_resolution.py

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import preprocess as pp  # noqa: E402


class FakeMeshComponent:
    def __init__(self, filename):
        self.filename = filename


class MeshPathResolutionTest(unittest.TestCase):
    def test_relative_path_is_resolved_against_instrument_directory(self):
        # Mirrors tests/crack_height.instr: a Union_mesh filename written
        # relative to the .instr file, parsed with its literal quotes still
        # attached (the .instr text parser doesn't strip them).
        comp = FakeMeshComponent('"./crack_cryst.stl"')
        union_geometries = {"cryst": comp}

        pp.resolve_mesh_filenames(
            union_geometries, os.path.join(REPO_ROOT, "tests", "crack_height.instr")
        )

        self.assertEqual(
            comp.filename, os.path.join(REPO_ROOT, "tests", "crack_cryst.stl")
        )

    def test_resolution_does_not_depend_on_current_working_directory(self):
        comp = FakeMeshComponent('"./crack_cryst.stl"')
        union_geometries = {"cryst": comp}

        original_cwd = os.getcwd()
        os.chdir(REPO_ROOT)
        try:
            pp.resolve_mesh_filenames(
                union_geometries, os.path.join("tests", "crack_height.instr")
            )
        finally:
            os.chdir(original_cwd)

        self.assertEqual(
            comp.filename, os.path.join(REPO_ROOT, "tests", "crack_cryst.stl")
        )

    def test_absolute_path_is_left_unchanged(self):
        absolute = os.path.join(REPO_ROOT, "tests", "crack_cryst.stl")
        comp = FakeMeshComponent(f'"{absolute}"')
        union_geometries = {"cryst": comp}

        pp.resolve_mesh_filenames(
            union_geometries, os.path.join(REPO_ROOT, "tests", "crack_height.instr")
        )

        self.assertEqual(comp.filename, absolute)

    def test_non_mesh_components_without_a_filename_are_skipped(self):
        class FakeCylinderComponent:
            pass

        union_geometries = {"sample": FakeCylinderComponent()}

        # Must not raise, e.g. from getattr(comp, "filename") on a
        # component type that has no such parameter.
        pp.resolve_mesh_filenames(
            union_geometries, os.path.join(REPO_ROOT, "tests", "crack_height.instr")
        )


if __name__ == "__main__":
    unittest.main()
