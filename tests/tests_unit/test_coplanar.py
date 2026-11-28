import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from unviz.coplanar import find_coincident_faces  # noqa: E402


def box(x=0.0, y=0.0, z=0.0, size=0.1):
    return trimesh.creation.box(extents=[size] * 3).apply_translation([x, y, z])


def cylinder_along_y(y=0.0, radius=0.02, height=0.1):
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return mesh.apply_translation([0, y, 0])


def pairs(meshes):
    return [(f[0], f[1]) for f in find_coincident_faces(meshes)]


class TestCoincidentFaces(unittest.TestCase):
    def test_touching_boxes_are_flagged(self):
        found = find_coincident_faces({"a": box(), "b": box(x=0.1)})
        self.assertEqual([(f[0], f[1]) for f in found], [("a", "b")])
        [(normal, offset)] = found[0][2]
        np.testing.assert_allclose(normal, [1, 0, 0])
        self.assertAlmostEqual(offset, 0.05)

    def test_offset_boxes_are_not_flagged(self):
        self.assertEqual(pairs({"a": box(), "b": box(x=0.1 + 1e-6)}), [])

    def test_coplanar_but_separate_faces_are_not_flagged(self):
        self.assertEqual(pairs({"a": box(), "b": box(x=0.5)}), [])

    def test_edge_contact_is_not_flagged(self):
        self.assertEqual(pairs({"a": box(), "b": box(x=0.1, y=0.1)}), [])

    def test_cylinder_cap_on_box_is_flagged(self):
        self.assertEqual(pairs({"a": box(), "c": cylinder_along_y(y=0.1)}), [("a", "c")])

    def test_overlapping_coaxial_cylinder_sides_are_flagged(self):
        self.assertEqual(pairs({"low": cylinder_along_y(), "high": cylinder_along_y(y=0.05)}),
                         [("high", "low")])

    def test_empty_and_missing_meshes_are_skipped(self):
        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        self.assertEqual(pairs({"a": box(), "b": None, "c": empty}), [])


if __name__ == "__main__":
    unittest.main()
