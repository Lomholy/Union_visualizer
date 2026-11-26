import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from coplanar import find_coplanar_faces  # noqa: E402


class Comp:
    def __init__(self, component_name, name, **params):
        self.component_name = component_name
        self.name = name
        self.__dict__.update(params)


def at(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    return m


def box(name, w=0.1):
    return Comp("Union_box", name, xwidth=w, yheight=w, zdepth=w)


class TestCoplanar(unittest.TestCase):
    def test_touching_boxes_are_flagged(self):
        geoms = {"a": box("a"), "b": box("b")}
        found = find_coplanar_faces(geoms, {"a": at(), "b": at(x=0.1)})
        self.assertEqual([(f[0], f[1]) for f in found], [("a", "b")])
        self.assertAlmostEqual(found[0][4], 0.01)

    def test_offset_boxes_are_not_flagged(self):
        geoms = {"a": box("a"), "b": box("b")}
        self.assertEqual(find_coplanar_faces(geoms, {"a": at(), "b": at(x=0.1 + 1e-6)}), [])

    def test_coplanar_but_separate_faces_are_not_flagged(self):
        geoms = {"a": box("a"), "b": box("b")}
        self.assertEqual(find_coplanar_faces(geoms, {"a": at(), "b": at(x=0.5)}), [])

    def test_cylinder_on_box(self):
        geoms = {
            "a": box("a"),
            "c": Comp("Union_cylinder", "c", radius=0.02, yheight=0.1),
        }
        found = find_coplanar_faces(geoms, {"a": at(), "c": at(y=0.1)})
        self.assertEqual([(f[0], f[1]) for f in found], [("a", "c")])


if __name__ == "__main__":
    unittest.main()
