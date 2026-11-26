import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from coplanar import find_coplanar_faces, flat_planes  # noqa: E402


class Comp:
    def __init__(self, component_name, **params):
        self.component_name = component_name
        self.name = component_name
        self.__dict__.update(params)


def at(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def box(x=0.0, y=0.0, z=0.0, size=0.1):
    return trimesh.creation.box(extents=[size] * 3).apply_translation([x, y, z])


def cylinder_along_y(matrix, radius=0.02, height=0.1):
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return mesh.apply_transform(matrix)


def pairs(found):
    return [(f[0], f[1]) for f in found]


class TestCoplanar(unittest.TestCase):
    def test_touching_boxes_are_flagged(self):
        found = find_coplanar_faces({"a": box(), "b": box(x=0.1)})
        self.assertEqual(pairs(found), [("a", "b")])

    def test_offset_boxes_are_not_flagged(self):
        self.assertEqual(find_coplanar_faces({"a": box(), "b": box(x=0.1 + 1e-6)}), [])

    def test_coplanar_but_separate_faces_are_not_flagged(self):
        self.assertEqual(find_coplanar_faces({"a": box(), "b": box(x=0.5)}), [])

    def test_edge_contact_is_not_flagged(self):
        self.assertEqual(find_coplanar_faces({"a": box(), "b": box(x=0.1, y=0.1)}), [])

    def test_cylinder_cap_on_box_is_flagged(self):
        matrix = at(y=0.1)
        comp = Comp("Union_cylinder", radius=0.02, yheight=0.1)
        planes = {"a": flat_planes(Comp("Union_box", xwidth=0.1, yheight=0.1, zdepth=0.1), at()),
                  "c": flat_planes(comp, matrix)}
        found = find_coplanar_faces({"a": box(), "c": cylinder_along_y(matrix)}, planes)
        self.assertEqual(pairs(found), [("a", "c")])

    def test_tessellated_cylinder_sides_are_ignored(self):
        comp = Comp("Union_cylinder", radius=0.02, yheight=0.1)
        low, high = at(), at(y=0.05)
        meshes = {"low": cylinder_along_y(low), "high": cylinder_along_y(high)}
        self.assertTrue(find_coplanar_faces(meshes))
        planes = {"low": flat_planes(comp, low), "high": flat_planes(comp, high)}
        self.assertEqual(find_coplanar_faces(meshes, planes), [])


if __name__ == "__main__":
    unittest.main()
