import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from coplanar import (  # noqa: E402
    find_coincident_curved,
    find_coplanar_faces,
    flat_planes,
    primitive_mesh,
)


class Comp:
    def __init__(self, component_name, **params):
        self.component_name = component_name
        self.name = component_name
        self.__dict__.update(params)


def at(x=0.0, y=0.0, z=0.0, rot_x=0.0, rot_z=0.0):
    m = trimesh.transformations.euler_matrix(np.radians(rot_x), 0, np.radians(rot_z))
    m[:3, 3] = [x, y, z]
    return m


def curved(geoms):
    """find_coincident_curved on {name: (comp, matrix)}, as name pairs."""
    comps = {n: c for n, (c, _m) in geoms.items()}
    matrices = {n: m for n, (_c, m) in geoms.items()}
    return [(f[0], f[1]) for f in find_coincident_curved(comps, matrices)]


def cyl(radius=0.02, yheight=0.1):
    return Comp("Union_cylinder", radius=radius, yheight=yheight)


def cone(radius_bottom, radius_top, yheight):
    return Comp("Union_cone", radius_bottom=radius_bottom, radius_top=radius_top, yheight=yheight)


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


class TestCurved(unittest.TestCase):
    def test_overlapping_coaxial_cylinders_are_flagged(self):
        self.assertEqual(curved({"a": (cyl(), at()), "b": (cyl(), at(y=0.05))}), [("a", "b")])

    def test_flipped_coaxial_cylinder_is_flagged(self):
        self.assertEqual(curved({"a": (cyl(), at()), "b": (cyl(), at(y=0.05, rot_x=180))}),
                         [("a", "b")])

    def test_end_to_end_cylinders_are_not_flagged(self):
        self.assertEqual(curved({"a": (cyl(), at()), "b": (cyl(), at(y=0.1))}), [])

    def test_different_radius_is_not_flagged(self):
        self.assertEqual(curved({"a": (cyl(), at()), "b": (cyl(radius=0.021), at())}), [])

    def test_parallel_offset_axis_is_not_flagged(self):
        self.assertEqual(curved({"a": (cyl(), at()), "b": (cyl(), at(x=1e-6))}), [])

    def test_crossing_cylinders_are_not_flagged(self):
        self.assertEqual(curved({"a": (cyl(), at()), "b": (cyl(), at(rot_z=90))}), [])

    def test_cone_continuing_another_cone_is_flagged(self):
        # radius 0.1 -> 0.05 over y in [-0.1, 0.1]; the second cone covers
        # y in [0, 0.2] with the same slope, so the sides share y in [0, 0.1].
        self.assertEqual(
            curved({"a": (cone(0.1, 0.05, 0.2), at()), "b": (cone(0.075, 0.025, 0.2), at(y=0.1))}),
            [("a", "b")],
        )

    def test_cone_with_other_slope_is_not_flagged(self):
        self.assertEqual(
            curved({"a": (cone(0.1, 0.05, 0.2), at()), "b": (cone(0.075, 0.05, 0.2), at(y=0.1))}),
            [],
        )

    def test_cylinder_matching_equal_radius_cone_is_flagged(self):
        self.assertEqual(curved({"a": (cyl(), at()), "b": (cone(0.02, 0.02, 0.1), at())}),
                         [("a", "b")])

    def test_identical_spheres_are_flagged(self):
        s = Comp("Union_sphere", radius=0.05)
        self.assertEqual(curved({"a": (s, at()), "b": (s, at())}), [("a", "b")])

    def test_shifted_spheres_are_not_flagged(self):
        s = Comp("Union_sphere", radius=0.05)
        self.assertEqual(curved({"a": (s, at()), "b": (s, at(z=1e-6))}), [])


class TestUndrawnVolumes(unittest.TestCase):
    """Masks and visualize=0 volumes are not drawn by Union, so they are
    checked on their uncarved primitive meshes."""

    def test_mask_box_touching_volume_box_is_flagged(self):
        mask = Comp("Union_box", xwidth=0.1, yheight=0.1, zdepth=0.1, mask_string="v")
        vol = Comp("Union_box", xwidth=0.1, yheight=0.1, zdepth=0.1)
        m_mask, m_vol = at(), at(x=0.1)
        meshes = {"mask": primitive_mesh(mask, m_mask), "vol": primitive_mesh(vol, m_vol)}
        planes = {"mask": flat_planes(mask, m_mask), "vol": flat_planes(vol, m_vol)}
        self.assertEqual(pairs(find_coplanar_faces(meshes, planes)), [("mask", "vol")])

    def test_tapered_box_primitive_has_its_faces(self):
        comp = Comp("Union_box", xwidth=0.2, yheight=0.1, zdepth=0.1, xwidth2=0.1, yheight2=0.05)
        mesh = primitive_mesh(comp, at())
        self.assertTrue(mesh.is_watertight)
        self.assertAlmostEqual(mesh.bounds[1][0], 0.1)

    def test_cylinder_cap_primitive_on_box_is_flagged(self):
        box_comp = Comp("Union_box", xwidth=0.1, yheight=0.1, zdepth=0.1)
        c, m = cyl(), at(y=0.1)
        meshes = {"a": primitive_mesh(box_comp, at()), "c": primitive_mesh(c, m)}
        planes = {"a": flat_planes(box_comp, at()), "c": flat_planes(c, m)}
        self.assertEqual(pairs(find_coplanar_faces(meshes, planes)), [("a", "c")])

    def test_sphere_has_no_primitive(self):
        self.assertIsNone(primitive_mesh(Comp("Union_sphere", radius=0.05), at()))


if __name__ == "__main__":
    unittest.main()
