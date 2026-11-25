"""Tests for the "union" mesher: a port of McStas Union's priority display
clipping (union_mcdisplay_polyhedron in McCode's union-lib.c).

The fake components carry only the attributes the mesher reads, as in
test_tapered_box.py. TestMcStasReference compares against areas McStas
itself drew with mcdisplay --trace=2 for tests_unit/data/union_clip_shapes.instr.
"""

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import meshing  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"
NO_CLIP = {"enable": False, "axis": "X", "mode": "Above", "position": 0.0}


class FakeComp:
    def __init__(self, component_name, name, priority=1, mask_string=None,
                 mask_setting=None, **params):
        self.component_name = component_name
        self.name = name
        self.priority = priority
        self.mask_string = mask_string
        self.mask_setting = mask_setting
        self.material_string = "inc"
        for key, value in params.items():
            setattr(self, key, value)


def box(name, w, h, d, **kw):
    return FakeComp("Union_box", name, xwidth=w, yheight=h, zdepth=d, **kw)


def translation(x=0.0, y=0.0, z=0.0):
    m = np.eye(4)
    m[:3, 3] = (x, y, z)
    return m


def build(comps, matrices=None, clip=NO_CLIP):
    geoms = {c.name: c for c in comps}
    matrices = matrices or {}
    world = {name: matrices.get(name, np.eye(4)) for name in geoms}
    with contextlib.redirect_stdout(io.StringIO()):
        return meshing.build_all_meshes(
            geoms, world, {}, {}, 64, clip, export=False, mesher="union",
        )


def area(mesh):
    return 0.0 if mesh is None else float(mesh.area)


class TestPriorityClipping(unittest.TestCase):
    def test_higher_priority_hides_lower(self):
        # R1: A is shown only outside the higher-priority B (x > 0.1). Boundaries in
        # these tests stay off the 0.5 vertex lattice the box pre-split makes, where
        # McStas's algorithm drops slivers along an edge lying in the boundary.
        a = box("a", 2, 2, 2, priority=1)
        b = box("b", 2, 2.5, 2.5, priority=2)
        meshes = build([a, b], {"b": translation(x=1.1)})
        self.assertAlmostEqual(area(meshes.get("a")), 4 + 4 * 2 * 1.1, places=6)
        self.assertAlmostEqual(area(meshes.get("b")), 2 * 2 * 2.5 * 2 + 2 * 2.5 * 2.5, places=6)

    def test_no_boolean_operations(self):
        # R1: the mesher never calls the OpenCASCADE Boolean kernel
        import brep
        with mock.patch.object(brep, "BRepAlgoAPI_Cut", side_effect=AssertionError), \
                mock.patch.object(brep, "BRepAlgoAPI_Common", side_effect=AssertionError):
            meshes = build([box("a", 2, 2, 2, priority=1), box("b", 1, 1, 1, priority=2)])
        self.assertGreater(area(meshes.get("a")), 0)

    def test_unknown_component_type_is_skipped_with_warning(self):
        # R1 error case
        odd = FakeComp("Union_torus", "odd", priority=5)
        geoms = {"odd": odd, "a": box("a", 2, 2, 2)}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            meshes = meshing.build_all_meshes(
                geoms, {"odd": np.eye(4), "a": np.eye(4)}, {}, {}, 64, NO_CLIP,
                export=False, mesher="union",
            )
        self.assertIsNone(meshes.get("odd"))
        self.assertIn("Union_torus", out.getvalue())
        self.assertAlmostEqual(area(meshes.get("a")), 24.0, places=6)


class TestShapes(unittest.TestCase):
    # R2: every Union shape, alone, keeps its whole surface
    def test_box(self):
        self.assertAlmostEqual(area(build([box("a", 1, 2, 3)]).get("a")), 22.0, places=6)

    def test_tapered_box(self):
        x1, y1, x2, y2, d = 1.0, 2.0, 3.0, 1.0, 2.0
        comp = box("a", x1, y1, d, xwidth2=x2, yheight2=y2)
        sides = (y1 + y2) * math.hypot(d, (x2 - x1) / 2) + (x1 + x2) * math.hypot(d, (y2 - y1) / 2)
        self.assertAlmostEqual(area(build([comp]).get("a")), x1 * y1 + x2 * y2 + sides, places=6)

    def test_cylinder(self):
        r, h, n = 0.5, 2.0, 32
        comp = FakeComp("Union_cylinder", "a", radius=r, yheight=h)
        polygon = n * 2 * r * math.sin(math.pi / n) * h + 2 * n * 0.5 * r * r * math.sin(2 * math.pi / n)
        self.assertAlmostEqual(area(build([comp]).get("a")), polygon, places=6)

    def test_cone(self):
        rb, rt, h = 1.0, 0.25, 2.0
        comp = FakeComp("Union_cone", "a", radius_bottom=rb, radius_top=rt, yheight=h)
        exact = math.pi * (rb + rt) * math.hypot(h, rb - rt) + math.pi * (rb * rb + rt * rt)
        self.assertAlmostEqual(area(build([comp]).get("a")) / exact, 1.0, delta=0.01)

    def test_sphere(self):
        comp = FakeComp("Union_sphere", "a", radius=0.7)
        self.assertAlmostEqual(area(build([comp]).get("a")) / (4 * math.pi * 0.49), 1.0, delta=0.02)

    def test_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cube.stl"
            trimesh.creation.box(extents=(1, 2, 3)).export(path)
            comp = FakeComp("Union_mesh", "a", filename=str(path), coordinate_scale=1.0)
            self.assertAlmostEqual(area(build([comp]).get("a")), 22.0, places=6)

    def test_missing_mesh_file_raises_like_brep(self):
        # R2 error case: same exception type as the brep mesher
        import brep
        comp = FakeComp("Union_mesh", "a", filename="/nonexistent/x.stl", coordinate_scale=1.0)
        with self.assertRaises(Exception) as from_brep:
            brep.build_comp_brep(comp, {"a": np.eye(4)})
        with self.assertRaises(type(from_brep.exception)):
            build([comp])


class TestMasks(unittest.TestCase):
    # R3: A (cube of side 2 at the origin) masked by M1 (x < -0.1) and M2 (y < -0.1)
    ALL_AREA = 2 * 0.9 * 2 + 2 * 0.9 * 0.9
    ANY_AREA = 24 - (2 * 1.1 * 2 + 2 * 1.1 * 1.1)

    def masks(self, setting):
        a = box("a", 2, 2, 2, priority=1)
        m1 = box("m1", 2, 3, 3, mask_string="a", mask_setting=setting)
        m2 = box("m2", 3, 2, 3, mask_string="a", mask_setting=setting)
        return build([a, m1, m2], {"m1": translation(x=-1.1), "m2": translation(y=-1.1)})

    def test_all(self):
        meshes = self.masks("All")
        self.assertAlmostEqual(area(meshes.get("a")), self.ALL_AREA, places=6)
        self.assertIsNone(meshes.get("m1"))
        self.assertIsNone(meshes.get("m2"))

    def test_any(self):
        self.assertAlmostEqual(area(self.masks('"ANY"').get("a")), self.ANY_AREA, places=6)

    def test_unset_setting_is_all(self):
        self.assertAlmostEqual(area(self.masks(None).get("a")), self.ALL_AREA, places=6)

    def test_mask_name_must_match_whole_name(self):
        a = box("a", 2, 2, 2, priority=1)
        ab = box("ab", 2, 2, 2, priority=2)
        mask = box("m", 2, 3, 3, mask_string="ab")
        meshes = build([a, ab, mask], {"ab": translation(z=5), "m": translation(z=5, x=-1.1)})
        self.assertAlmostEqual(area(meshes.get("a")), 24.0, places=6)
        self.assertAlmostEqual(area(meshes.get("ab")), 4 + 4 * 2 * 0.9, places=6)


class TestIncrementalRebuild(unittest.TestCase):
    def test_moving_a_cutters_mask_changes_the_lower_components_signature(self):
        # A (priority 1) is cut by B (priority 2), which mask M limits
        from bounding_box import compute_all_world_bboxes, component_dependency_signature

        geoms = {c.name: c for c in (box("a", 2, 2, 2, priority=1), box("b", 1, 1, 1, priority=2),
                                     box("m", 1, 1, 1, mask_string="b"))}
        signatures = []
        for x in (0.0, 0.3):
            world = {"a": np.eye(4), "b": np.eye(4), "m": translation(x=x)}
            signatures.append(component_dependency_signature("a", geoms, compute_all_world_bboxes(geoms, world)))
        self.assertNotEqual(*signatures)

    def test_union_rebuild_builds_shapes_once(self):
        import union_clip
        import union_viewer

        calls = []
        real = union_clip.build_shapes

        def counting(*args):
            calls.append(1)
            return real(*args)

        inp = str(DATA / "union_clip_shapes.instr")
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, meshes, deps, _ = union_viewer.compute_mesh_data(inp, NO_CLIP, mesher="union")
            stale = {name: "stale" for name in deps}
            with mock.patch.object(union_clip, "build_shapes", counting):
                union_viewer.compute_mesh_data(inp, NO_CLIP, meshes=meshes, dependencies=stale, mesher="union")
        self.assertEqual(len(calls), 1)


class TestClipBox(unittest.TestCase):
    # R4: the clip plane removes its side of every component
    def test_clip_removes_side(self):
        clip = {"enable": True, "axis": "X", "mode": "Above", "position": 0.1}
        self.assertAlmostEqual(area(build([box("a", 2, 2, 2)], clip=clip).get("a")), 4 + 4 * 2 * 0.9, places=6)

    def test_clip_below(self):
        clip = {"enable": True, "axis": "Z", "mode": "Below", "position": 0.3}
        self.assertAlmostEqual(area(build([box("a", 2, 2, 2)], clip=clip).get("a")), 4 + 4 * 2 * 1.3, places=6)

    def test_clip_disabled(self):
        self.assertAlmostEqual(area(build([box("a", 2, 2, 2)]).get("a")), 24.0, places=6)


class TestMcStasReference(unittest.TestCase):
    # R7: per-component area within 0.2% of McStas's own --trace=2 polyhedra
    def test_shapes_instrument(self):
        import preprocess
        from clipping import resolve_clip_frame

        reference = json.loads((DATA / "union_clip_shapes_mcstas_areas.json").read_text())
        with contextlib.redirect_stdout(io.StringIO()):
            _, world, geoms = preprocess.preprocess(str(DATA / "union_clip_shapes.instr"), False)
            meshes = meshing.build_all_meshes(
                geoms, world, {}, {}, 64, resolve_clip_frame(NO_CLIP, world),
                export=False, mesher="union",
            )
        self.assertEqual(set(reference), {n for n, m in meshes.items() if m is not None})
        for name, mcstas_area in reference.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(area(meshes[name]) / mcstas_area, 1.0, delta=2e-3)


if __name__ == "__main__":
    unittest.main()
