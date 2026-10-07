"""Tests for cavity-wall tagging: brep.build_single_brep_mesh tags each
triangle a higher-priority cut left behind with that component's name
(brep.CARVED_BY), gui_helpers maps those names to render keys, and
scene_objects.build_gfx_group splits the cavity walls into their own
Meshes so the viewer can draw them only while their carver is hidden.
Drawing a cavity wall together with the surface it lies on is what made
an enclosing volume, such as surrounding air, z-fight with its contents.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import gui_helpers  # noqa: E402
from bounding_box import compute_all_world_bboxes  # noqa: E402


class FakeComp:
    """Stand-in for a mcstasscript Union_box/Union_cylinder component, with
    only the attributes the brep mesher reads."""

    mask_string = None

    def __init__(self, component_name, name, priority, material="Al", **dims):
        self.component_name = component_name
        self.name = name
        self.priority = priority
        self.material_string = material
        for key, value in dims.items():
            setattr(self, key, value)


def box(name, priority, size, material="Al"):
    return FakeComp("Union_box", name, priority, material, xwidth=size, yheight=size, zdepth=size)


def translated(x=0.0, y=0.0, z=0.0):
    matrix = np.eye(4)
    matrix[:3, 3] = (x, y, z)
    return matrix


class BrepCarvedByTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import brep  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise unittest.SkipTest(f"pythonocc-core unavailable: {exc}")

    def mesh(self, comps, world, name):
        import brep

        geometries = {c.name: c for c in comps}
        return brep.build_single_brep_mesh(
            geometries[name],
            geometries,
            world,
            {"enable": False},
            verbose=False,
            deflection=0.001,
            world_bboxes=compute_all_world_bboxes(geometries, world),
        )

    def tagged_area(self, mesh, carver):
        import brep

        return mesh.area_faces[mesh.face_attributes[brep.CARVED_BY] == carver].sum()

    def test_enclosed_component_leaves_a_cavity_tagged_with_its_name(self):
        r, h = 0.2, 0.4
        comps = [
            box("outer", 1, 1.0),
            FakeComp("Union_cylinder", "inner", 2, radius=r, yheight=h),
        ]
        world = {"outer": np.eye(4), "inner": np.eye(4)}
        outer = self.mesh(comps, world, "outer")

        self.assertEqual(set(outer.face_attributes["carved_by"]), {"", "inner"})
        self.assertAlmostEqual(self.tagged_area(outer, ""), 6.0, places=6)
        cylinder_area = 2 * np.pi * r * h + 2 * np.pi * r**2
        self.assertAlmostEqual(self.tagged_area(outer, "inner") / cylinder_area, 1.0, delta=0.01)

    def test_higher_priority_component_is_not_tagged(self):
        comps = [box("outer", 1, 1.0), box("inner", 2, 0.4)]
        world = {"outer": np.eye(4), "inner": np.eye(4)}
        inner = self.mesh(comps, world, "inner")
        self.assertEqual(set(inner.face_attributes["carved_by"]), {""})

    def test_partial_overlap_tags_only_the_walls_inside(self):
        # inner spans x in [0.2, 0.8]; outer keeps x in [-0.5, 0.2] beside it,
        # walled by inner's -x face (0.6 x 0.6) and the parts of inner's four
        # side faces between x = 0.2 and 0.5 (4 x 0.3 x 0.6).
        comps = [box("outer", 1, 1.0), box("inner", 2, 0.6)]
        world = {"outer": np.eye(4), "inner": translated(x=0.5)}
        outer = self.mesh(comps, world, "outer")
        self.assertAlmostEqual(self.tagged_area(outer, "inner"), 0.36 + 4 * 0.3 * 0.6, places=6)

    def test_each_carver_tags_its_own_walls(self):
        comps = [box("outer", 1, 1.0), box("a", 2, 0.2), box("b", 3, 0.1)]
        world = {"outer": np.eye(4), "a": translated(x=-0.25), "b": translated(x=0.25)}
        outer = self.mesh(comps, world, "outer")
        self.assertAlmostEqual(self.tagged_area(outer, "a"), 6 * 0.2**2, places=6)
        self.assertAlmostEqual(self.tagged_area(outer, "b"), 6 * 0.1**2, places=6)

    def test_overlapping_carvers_tag_the_wall_of_whichever_bounds_the_cavity(self):
        # a spans x in [-0.3, 0.1] and b (higher priority) x in [-0.1, 0.3],
        # so the cavity is a 0.6 x 0.4 x 0.4 box. Its walls for x < -0.1 lie
        # on a's surface; the rest on b's, including where a's side faces are
        # coplanar with b's, since b's surface is the one drawn there.
        comps = [box("outer", 1, 1.0), box("a", 2, 0.4), box("b", 3, 0.4)]
        world = {"outer": np.eye(4), "a": translated(x=-0.1), "b": translated(x=0.1)}
        outer = self.mesh(comps, world, "outer")
        self.assertAlmostEqual(self.tagged_area(outer, "a"), 0.4 * 0.4 + 4 * 0.2 * 0.4, places=6)
        self.assertAlmostEqual(self.tagged_area(outer, "b"), 0.4 * 0.4 + 4 * 0.4 * 0.4, places=6)

    def test_face_touching_a_higher_priority_component_is_tagged(self):
        # Two cubes sharing the plane x = 0.5: outer's +x face lies on
        # neighbour's -x face, and only neighbour's should be drawn there.
        comps = [box("outer", 1, 1.0), box("neighbour", 2, 1.0)]
        world = {"outer": np.eye(4), "neighbour": translated(x=1.0)}
        outer = self.mesh(comps, world, "outer")
        self.assertAlmostEqual(self.tagged_area(outer, "neighbour"), 1.0, places=6)
        self.assertAlmostEqual(self.tagged_area(outer, ""), 5.0, places=6)


class CarverKeysTest(unittest.TestCase):
    def geometries(self):
        return {
            "air": FakeComp("Union_box", "air", 0, '"Air"'),
            "pocket": FakeComp("Union_box", "pocket", 1, '"Air"'),
            "sample": FakeComp("Union_box", "sample", 2, '"Al"'),
            "mask": FakeComp("Union_box", "mask", 3, '"Al"'),
        }

    def test_render_keys_are_names_or_materials(self):
        meshes = {"air": 1, "pocket": 1, "sample": 1, "mask": None}
        self.assertEqual(
            gui_helpers.render_keys_of(self.geometries(), meshes, False),
            {"air": "air", "pocket": "pocket", "sample": "sample"},
        )
        self.assertEqual(
            gui_helpers.render_keys_of(self.geometries(), meshes, True),
            {"air": "Air", "pocket": "Air", "sample": "Al"},
        )

    def test_carvers_without_a_render_key_become_own_surface(self):
        render_key_of = {"air": "Air", "pocket": "Air", "sample": "Al"}
        keys = gui_helpers.carver_keys(["", "pocket", "sample", "mask"], render_key_of)
        self.assertEqual(list(keys), ["", "Air", "Al", ""])


class BuildGfxGroupCavityTest(unittest.TestCase):
    def tagged_box(self, keys):
        mesh = trimesh.creation.box()
        return trimesh.Trimesh(
            mesh.vertices,
            mesh.faces,
            face_attributes={gui_helpers.CARVER_KEY: np.array(keys)},
        )

    def test_cavity_walls_get_their_own_meshes(self):
        from scene_objects import build_gfx_group

        keys = [""] * 4 + ["Al"] * 4 + ["Air"] * 4
        group = build_gfx_group({"Air": self.tagged_box(keys)}, {"Air": False}, {})
        obj = group.geometry_meshes["Air"]

        # Walls carved by "Air" itself are dropped; "Al" walls are separate.
        self.assertEqual(set(group.cavity_meshes["Air"]), {"Al"})
        self.assertEqual(len(obj.children), 2)
        triangles = sorted(len(child.geometry.indices.data) for child in obj.children)
        self.assertEqual(triangles, [4, 4])
        self.assertIs(obj.children[0].material, obj.children[1].material)

    def test_untagged_mesh_is_one_mesh(self):
        from scene_objects import build_gfx_group

        group = build_gfx_group({"Al": trimesh.creation.box()}, {"Al": False}, {})
        self.assertEqual(len(group.geometry_meshes["Al"].children), 1)
        self.assertEqual(group.cavity_meshes["Al"], {})


if __name__ == "__main__":
    unittest.main()
