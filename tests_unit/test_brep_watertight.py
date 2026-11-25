"""BRep meshes of closed Union geometry must be watertight.

OpenCASCADE's sphere triangulation leaves zero-area triangles at the poles;
kept, they make the mesh non-watertight, and clip_mesh then exports the cut
of such a component without a cap. tests/test.instr has a sphere carved out
of a box, so both the sphere and the box show the problem.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import preprocess  # noqa: E402
from brep import build_brep_meshes  # noqa: E402
from gui_helpers import clip_mesh, _can_cap_slices  # noqa: E402
from pipeline import NO_CLIP  # noqa: E402


class BrepWatertightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, world_matrices, union_geometries = preprocess(str(ROOT / "tests" / "test.instr"), False)
        cls.meshes = build_brep_meshes(union_geometries, world_matrices, NO_CLIP, False, max_workers=1)

    def test_every_mesh_is_watertight_without_degenerate_faces(self):
        for name, mesh in self.meshes.items():
            with self.subTest(name=name):
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(np.all(mesh.area_faces > 0))

    def test_clipped_sphere_is_capped(self):
        if not _can_cap_slices():
            self.skipTest("trimesh's capping dependencies (shapely, mapbox_earcut) are not installed")
        clip = {"enable": True, "axis": "X", "mode": "Above", "position": 0.0, "frame": None}
        for name in ("sphere_mid", "box_schlong"):
            with self.subTest(name=name):
                self.assertTrue(clip_mesh(self.meshes[name], clip).is_watertight)


if __name__ == "__main__":
    unittest.main()
