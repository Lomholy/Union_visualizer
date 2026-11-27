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


class TestCoincidentSurfaces(unittest.TestCase):
    # CAD parts often share faces exactly; a surface point takes the status of the material just inside it
    def mesh_comp(self, tmp, name, priority, offset=(0, 0, 0)):
        path = Path(tmp) / f"{name}.stl"
        cube = trimesh.creation.box(extents=(1, 1, 1))
        cube.apply_translation(offset)
        cube.export(path)
        return FakeComp("Union_mesh", name, priority=priority, filename=str(path), coordinate_scale=1.0)

    def test_identical_higher_priority_copy_hides_the_lower(self):
        with tempfile.TemporaryDirectory() as tmp:
            meshes = build([self.mesh_comp(tmp, "low", 1), self.mesh_comp(tmp, "high", 2)])
        self.assertIsNone(meshes.get("low"))
        self.assertAlmostEqual(area(meshes.get("high")), 6.0, places=6)

    def test_touching_face_stays_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            meshes = build([self.mesh_comp(tmp, "low", 1), self.mesh_comp(tmp, "high", 2, offset=(1, 0, 0))])
        self.assertAlmostEqual(area(meshes.get("low")), 6.0, places=6)


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


class TestCloseCuts(unittest.TestCase):
    # Mesher Options > Close cut surfaces: walls where higher priorities and masks cut a component
    def build_closed(self, comps, matrices=None, close_cuts=True):
        geoms = {c.name: c for c in comps}
        world = {name: (matrices or {}).get(name, np.eye(4)) for name in geoms}
        with contextlib.redirect_stdout(io.StringIO()):
            return meshing.build_all_meshes(
                geoms, world, {}, {}, 64, NO_CLIP, export=False, mesher="union", close_cuts=close_cuts,
            )

    def test_cut_box_is_closed_by_a_wall(self):
        # A (cube of side 2) with x > 0.1 taken by B: the wall at x = 0.1 closes it
        a = box("a", 2, 2, 2, priority=1)
        b = box("b", 2, 2.5, 2.5, priority=2)
        mesh = self.build_closed([a, b], {"b": translation(x=1.1)}).get("a")
        self.assertAlmostEqual(area(mesh) / (4 + 4 * 2 * 1.1 + 4), 1, delta=5e-4)
        self.assertAlmostEqual(mesh.volume / (2 * 2 * 1.1), 1, delta=5e-4)

    def test_masked_box_is_closed_by_its_masks(self):
        # A masked by M1 (x < -0.1) and M2 (y < -0.1), ALL: a closed 0.9 x 0.9 x 2 block
        a = box("a", 2, 2, 2, priority=1)
        m1 = box("m1", 2, 3, 3, mask_string="a")
        m2 = box("m2", 3, 2, 3, mask_string="a")
        mesh = self.build_closed([a, m1, m2], {"m1": translation(x=-1.1), "m2": translation(y=-1.1)}).get("a")
        self.assertAlmostEqual(mesh.volume / (0.9 * 0.9 * 2), 1, delta=5e-4)
        self.assertAlmostEqual(area(mesh) / (2 * 0.9 * 0.9 + 4 * 0.9 * 2), 1, delta=5e-4)

    def test_off_by_default(self):
        a = box("a", 2, 2, 2, priority=1)
        b = box("b", 2, 2.5, 2.5, priority=2)
        self.assertAlmostEqual(area(build([a, b], {"b": translation(x=1.1)}).get("a")), 4 + 4 * 2 * 1.1, places=6)


class TestClosedCut(unittest.TestCase):
    # Close cut surfaces on convex cutters: the kept surface and the walls share their cut vertices
    CUTTERS = {
        "box": (box("b", 2, 2.5, 2.5, priority=2), translation(x=1.1)),
        "cylinder_side": (FakeComp("Union_cylinder", "b", priority=2, radius=0.5, yheight=3), translation(x=0.93, z=0.07)),
        "cylinder_cap": (FakeComp("Union_cylinder", "b", priority=2, radius=0.5, yheight=1),
                         translation(x=0.03, y=1.07, z=0.11)),
        "cone": (FakeComp("Union_cone", "b", priority=2, radius_bottom=0.5, radius_top=0.2, yheight=1),
                 translation(x=0.05, y=0.93, z=-0.12)),
        "sphere": (FakeComp("Union_sphere", "b", priority=2, radius=0.5), translation(x=0.91, y=0.13, z=0.07)),
    }

    def build_pair(self, cutter, matrix, close_cuts=True):
        geoms = {"a": box("a", 2, 2, 2, priority=1), "b": cutter}
        with contextlib.redirect_stdout(io.StringIO()):
            return meshing.build_all_meshes(geoms, {"a": np.eye(4), "b": matrix}, {}, {}, 64, NO_CLIP,
                                            export=False, mesher="union", close_cuts=close_cuts)

    def assert_closed(self, mesh):
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)
        edges = np.sort(mesh.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        self.assertEqual(set(np.unique(edges, axis=0, return_counts=True)[1]), {2})
        faces = np.sort(mesh.faces, axis=1)
        self.assertEqual(len(np.unique(faces, axis=0)), len(faces))
        self.assertFalse(np.any((faces[:, 0] == faces[:, 1]) | (faces[:, 1] == faces[:, 2])))

    def facet_volume_inside_a(self, cutter, matrix):
        # the cutter's display polyhedron cut by the six faces of a (a cube of side 2 at the origin)
        import union_clip
        solid = trimesh.Trimesh(union_clip.make_shape(cutter, matrix).triangles().reshape(-1, 3)).convex_hull
        for axis in range(3):
            for sign in (1, -1):
                normal = np.zeros(3)
                normal[axis] = -sign
                solid = trimesh.intersections.slice_mesh_plane(solid, normal, -normal, cap=True)
        return solid.volume

    def test_convex_cutters_close_watertight(self):
        for name, (cutter, matrix) in self.CUTTERS.items():
            with self.subTest(cutter=name):
                meshes = self.build_pair(cutter, matrix)
                for mesh in meshes.values():
                    self.assert_closed(mesh)
                expected = 2 * 2 * 1.1 if name == "box" else 8 - self.facet_volume_inside_a(cutter, matrix)
                self.assertAlmostEqual(meshes["a"].volume / expected, 1, delta=5e-4)

    def test_kept_area_matches_the_open_cut(self):
        # R6: the faceted cut keeps the open cut's area of a's own surface within 1%
        for name in ("cylinder_side", "cylinder_cap", "cone", "sphere"):
            with self.subTest(cutter=name):
                closed = self.build_pair(*self.CUTTERS[name])["a"]
                on_surface = np.abs(closed.triangles_center).max(1) > 1 - 1e-9
                kept = closed.area_faces[on_surface].sum()
                self.assertAlmostEqual(kept / area(self.build_pair(*self.CUTTERS[name], close_cuts=False)["a"]), 1,
                                       delta=0.01)

    def test_cutters_with_flush_faces_close_once(self):
        # b and c share their y and z face planes, facing the same way: their walls overlap there
        geoms = {c.name: c for c in (box("a", 2, 2, 2, priority=1), box("b", 1, 1, 1, priority=2),
                                     box("c", 1, 1, 1, priority=3))}
        world = {"a": np.eye(4), "b": translation(x=0.5), "c": translation(x=0.8)}
        with contextlib.redirect_stdout(io.StringIO()):
            mesh = meshing.build_all_meshes(geoms, world, {}, {}, 64, NO_CLIP, export=False, mesher="union",
                                            close_cuts=True)["a"]
        self.assert_closed(mesh)
        self.assertAlmostEqual(mesh.volume, 7.0, places=9)

    def test_boxes_sharing_a_face_both_close(self):
        # b touches a's +x face with its -x face: each keeps its own face, neither gets a wall there
        meshes = self.build_pair(box("b", 2, 1.5, 1.5, priority=2), translation(x=2))
        for name, volume, surface in (("a", 8.0, 24.0), ("b", 4.5, 4 * 2 * 1.5 + 2 * 1.5 * 1.5)):
            with self.subTest(name=name):
                self.assert_closed(meshes[name])
                self.assertAlmostEqual(meshes[name].volume, volume, places=9)
                self.assertAlmostEqual(meshes[name].area, surface, places=9)


def rotation_y(degrees, **shift):
    m = translation(**shift)
    a = math.radians(degrees)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    return m


class TestClosedCutGeneral(unittest.TestCase):
    # Close cut surfaces with masks, meshes, meeting walls and degenerate positions
    ICOSPHERE = str(DATA / "union_clip_icosphere.stl")
    assert_closed = TestClosedCut.assert_closed
    build_pair = TestClosedCut.build_pair
    facet_volume_inside_a = TestClosedCut.facet_volume_inside_a

    def build_closed(self, comps, matrices, close_cuts=True):
        geoms = {c.name: c for c in comps}
        world = {name: matrices.get(name, np.eye(4)) for name in geoms}
        with contextlib.redirect_stdout(io.StringIO()):
            return meshing.build_all_meshes(geoms, world, {}, {}, 64, NO_CLIP, export=False, mesher="union",
                                            close_cuts=close_cuts)

    def masked(self, setting, close_cuts=True):
        # a (cube of side 2) masked by m1 (x < -0.1) and m2 (y < -0.1)
        comps = [box("a", 2, 2, 2), box("m1", 2, 3, 3, mask_string="a", mask_setting=setting),
                 box("m2", 3, 2, 3, mask_string="a", mask_setting=setting)]
        return self.build_closed(comps, {"m1": translation(x=-1.1), "m2": translation(y=-1.1)}, close_cuts)["a"]

    def test_masks_close_watertight(self):
        for setting, volume in (("All", 0.9 * 0.9 * 2), ("ANY", 8 - 1.1 * 1.1 * 2)):
            with self.subTest(setting=setting):
                mesh = self.masked(setting)
                self.assert_closed(mesh)
                self.assertAlmostEqual(mesh.volume, volume, places=9)

    def test_masked_kept_area_matches_the_open_cut(self):
        # R6
        for setting in ("All", "ANY"):
            with self.subTest(setting=setting):
                closed = self.masked(setting)
                kept = closed.area_faces[np.abs(closed.triangles_center).max(1) > 1 - 1e-9].sum()
                self.assertAlmostEqual(kept / area(self.masked(setting, close_cuts=False)), 1, delta=0.01)

    def test_walls_meeting_close_watertight(self):
        # b's and c's walls in a meet along a line inside a
        comps = [box("a", 2, 2, 2), box("b", 1, 1.5, 3, priority=2),
                 FakeComp("Union_cylinder", "c", priority=3, radius=0.6, yheight=3)]
        meshes = self.build_closed(comps, {"b": translation(x=0.9, y=0.7), "c": translation(x=0.5, z=0.85)})
        for name, mesh in meshes.items():
            with self.subTest(name=name):
                self.assert_closed(mesh)

    def test_icosphere_cutter_closes_watertight(self):
        cutter = FakeComp("Union_mesh", "b", priority=2, filename=self.ICOSPHERE, coordinate_scale=0.02)
        matrix = translation(x=0.93, y=0.11, z=0.07)
        meshes = self.build_pair(cutter, matrix)
        for mesh in meshes.values():
            self.assert_closed(mesh)
        self.assertAlmostEqual(meshes["a"].volume / (8 - self.facet_volume_inside_a(cutter, matrix)), 1, delta=5e-4)

    def test_icosphere_cut_by_a_box_closes_watertight(self):
        comps = [FakeComp("Union_mesh", "a", filename=self.ICOSPHERE, coordinate_scale=0.02),
                 box("b", 1, 1, 1, priority=2)]
        for mesh in self.build_closed(comps, {"b": translation(x=0.6, y=0.1)}).values():
            self.assert_closed(mesh)

    def test_mesh_with_flat_faces_closes_watertight(self):
        # own is an STL cube (side 2) whose flat faces hold 2, or after subdividing 8, coplanar triangles
        cases = {"flush": (0, translation(x=1.5), 8.0),
                 "crossing": (1, translation(x=0.93, y=0.31, z=-0.27), 8 - 0.57)}
        for name, (subdivisions, matrix, volume) in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "cube.stl"
                cube = trimesh.creation.box(extents=(2, 2, 2))
                for _ in range(subdivisions):
                    cube = cube.subdivide()
                cube.export(path)
                own = FakeComp("Union_mesh", "a", filename=str(path), coordinate_scale=1.0)
                mesh = self.build_closed([own, box("b", 1, 1, 1, priority=2)], {"b": matrix})["a"]
                self.assert_closed(mesh)
                self.assertAlmostEqual(mesh.volume, volume, places=9)

    def test_cut_through_pole_and_along_facet_edges(self):
        # the plane x = 0 holds the sphere's poles and two meridians of its facets
        sphere = FakeComp("Union_sphere", "a", radius=0.7)
        whole = self.build_closed([sphere], {})["a"].volume
        mesh = self.build_closed([sphere, box("b", 1, 2, 2, priority=2)], {"b": translation(x=0.5)})["a"]
        self.assert_closed(mesh)
        self.assertAlmostEqual(mesh.volume, whole / 2, places=9)

    def test_cut_through_cone_apex(self):
        cone = FakeComp("Union_cone", "a", radius_bottom=0.6, radius_top=0.0, yheight=1)
        tilt = math.radians(3)
        mesh = self.build_closed([cone, box("b", 1, 2, 2, priority=2)],
                                 {"b": rotation_y(3, x=0.5 * math.cos(tilt), z=-0.5 * math.sin(tilt))})["a"]
        self.assert_closed(mesh)

    def test_cutter_edge_lying_on_a_face(self):
        # b's edge touches a's top face from outside: a stays whole
        mesh = self.build_closed([box("a", 2, 2, 2), box("b", 1, 1, 1, priority=2)],
                                 {"b": rotation_y(45, x=0.3, z=1 + 0.5 * math.sqrt(2))})["a"]
        self.assert_closed(mesh)
        self.assertAlmostEqual(mesh.area, 24.0, places=9)

    def test_sphere_axis_lying_on_a_face(self):
        # each band of b's facets has its planes meet at one point of b's axis, on a's face x = 1
        meshes = self.build_pair(FakeComp("Union_sphere", "b", priority=2, radius=0.5), translation(x=1, y=0.13, z=0.07))
        for mesh in meshes.values():
            self.assert_closed(mesh)

    def test_tapered_box_with_a_zero_width_face(self):
        mesh = self.build_closed([box("a", 1, 1, 1, xwidth2=0.0, yheight2=1.0), box("b", 1, 1, 1, priority=2)],
                                 {"b": translation(x=0.6)})["a"]
        self.assert_closed(mesh)

    def test_buried_component_has_no_mesh(self):
        self.assertIsNone(self.build_closed([box("a", 1, 1, 1), box("b", 2, 2, 2, priority=2)], {})["a"])

    def test_cavity_keeps_its_wall(self):
        meshes = self.build_closed([box("a", 2, 2, 2), box("b", 0.5, 0.5, 0.5, priority=2)],
                                   {"b": translation(x=0.13, y=0.07)})
        self.assert_closed(meshes["a"])
        self.assertAlmostEqual(meshes["a"].volume, 8 - 0.125, places=9)
        self.assertAlmostEqual(meshes["a"].area, 24 + 1.5, places=9)

    def test_open_mesh_cutter_builds(self):
        # R1 error case: an open Union_mesh cutter never fails the build
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "open.stl"
            mesh = trimesh.load_mesh(self.ICOSPHERE)
            mesh.update_faces(np.arange(len(mesh.faces)) >= 3)
            mesh.export(path)
            cutter = FakeComp("Union_mesh", "b", priority=2, filename=str(path), coordinate_scale=0.02)
            meshes = self.build_pair(cutter, translation(x=0.93, y=0.11, z=0.07))
        self.assertGreater(area(meshes["a"]), 0)


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


class TestWatertightInstruments(unittest.TestCase):
    # brep's watertight components per instrument, frozen while brep existed: a lower bound
    def test_brep_watertight_components_are_closed(self):
        import preprocess
        import union_clip

        expected = json.loads((DATA / "union_watertight_expected.json").read_text())
        root = DATA.parent.parent
        for inp, names in expected.items():
            with contextlib.redirect_stdout(io.StringIO()):
                _, world, geoms = preprocess.preprocess(str(root / inp), False)
                meshes = union_clip.build_union_meshes(geoms, world, NO_CLIP, close_cuts=True)
            for name in names:
                with self.subTest(instrument=inp, component=name):
                    mesh = meshes[name]
                    self.assertIsNotNone(mesh)
                    self.assertTrue(mesh.is_watertight)
                    self.assertTrue(mesh.is_winding_consistent)
                    self.assertGreater(mesh.volume, 0)


class TestClosedIncrementalRebuild(unittest.TestCase):
    # moving the cylinder that cuts touch_a rebuilds only the components that depend on it
    DRILL = """
COMPONENT drill = Union_cylinder(priority = 30, radius = 0.03, yheight = 0.2, material_string = "mat")
AT ({x},0.01,0.013) RELATIVE sample

COMPONENT master"""

    def write(self, tmp, x):
        text = (DATA / "coincident_faces_test.instr").read_text()
        path = Path(tmp) / "drilled.instr"
        path.write_text(text.replace("\nCOMPONENT master", self.DRILL.format(x=x), 1))
        return str(path)

    def test_rebuilt_mesh_equals_full_rebuild(self):
        import pipeline
        import union_clip

        rebuilt = []
        real = union_clip.build_union_meshes

        def recording(*args, **kwargs):
            rebuilt.append(kwargs.get("names"))
            return real(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            kw = dict(mesher="union", close_cuts=True)
            _, _, meshes, deps, _ = pipeline.compute_mesh_data(self.write(tmp, -0.04), NO_CLIP, **kw)
            moved = self.write(tmp, -0.03)
            with mock.patch.object(pipeline, "build_union_meshes", recording):
                _, _, incremental, _, _ = pipeline.compute_mesh_data(moved, NO_CLIP, meshes=dict(meshes),
                                                                     dependencies=deps, **kw)
            _, _, full, _, _ = pipeline.compute_mesh_data(moved, NO_CLIP, **kw)
        self.assertEqual(sorted(rebuilt[0]), ["drill", "touch_a"])
        self.assertTrue(full["touch_a"].is_watertight)
        for name in full:
            with self.subTest(component=name):
                if full[name] is None:
                    self.assertIsNone(incremental[name])
                    continue
                np.testing.assert_array_equal(incremental[name].vertices, full[name].vertices)
                np.testing.assert_array_equal(incremental[name].faces, full[name].faces)


class TestKernelPlanes(unittest.TestCase):
    def test_coincident_planes_in_one_call_share_an_id(self):
        from union_facets import Kernel
        kernel = Kernel(1.0)
        # offsets in neighbouring rounding bins, closer than eps
        ids, sides = kernel.planes(np.array([[0, 1.0, 0], [0, 1.0, 0], [0, -1.0, 0]]),
                                   np.array([0.49, 0.51, -0.5]) * kernel.eps)
        self.assertEqual(len(set(ids.tolist())), 1)
        self.assertEqual(sides.tolist()[2], -sides.tolist()[0])



class TestNumbaDependency(unittest.TestCase):
    # R3 error case: without numba the mesher does not import
    def test_import_without_numba_names_numba(self):
        import importlib

        with mock.patch.dict(sys.modules, {"numba": None}):
            del sys.modules["union_clip"]
            with self.assertRaises(ImportError) as ctx:
                importlib.import_module("union_clip")
        self.assertIn("numba", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
