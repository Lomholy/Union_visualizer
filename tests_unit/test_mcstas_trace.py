"""Tests for mcstas_trace.py: parsing mcrun --trace=2 output into
world-space McStas component geometry and neutron rays.

The fixtures in tests_unit/data/ are real `mcrun <instr> --trace=2
--no-output-files -n 0 -y` output for tests/simple_test.instr and
tests/rotated_previous_test.instr, plus a 3-ray `-n 3 --seed=1234` run of
tests/simple_test.instr, so the parser tests need no McStas installation. The end-to-end test at the bottom runs mcrun itself and is
skipped when it isn't on PATH.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import mcstas_trace  # noqa: E402
from mcstas_trace import (  # noqa: E402
    ABSORB, PASS, SCATTER, STATE, _drawcall_geometry, parse_rays, parse_trace,
)

DATA = Path(__file__).resolve().parent / "data"

SIMPLE_TYPES = {
    "Origin": "Progress_bar", "src": "Source_simple", "sample": "Arm",
    "right": "Arm", "left": "Arm", "init": "Union_init", "non": "Non_process",
    "mat": "Union_make_material", "box_left": "Union_box",
    "box_lefttwo": "Union_box", "master": "Union_master", "stop": "Union_stop",
    "psd_det": "PSD_monitor",
}


def read(name):
    return (DATA / name).read_text()


class ComponentParsingTest(unittest.TestCase):
    def setUp(self):
        self.components = parse_trace(
            read("simple_test_geometry_trace.txt"), SIMPLE_TYPES
        )

    def test_only_drawn_non_union_components(self):
        # Union_master's wireframe is skipped; components with no MCDISPLAY
        # output (Progress_bar, the Union_* helpers) are left out.
        self.assertEqual(
            list(self.components), ["src", "sample", "right", "left", "psd_det"]
        )

    def test_master_is_drawn_without_types(self):
        components = parse_trace(read("simple_test_geometry_trace.txt"), {})
        self.assertIn("master", components)

    def test_arms_are_flagged(self):
        arms = [name for name, c in self.components.items() if c.is_arm]
        self.assertEqual(arms, ["sample", "right", "left"])

    def test_source_circle_in_world_space(self):
        # Source_simple draws its circle first, then dashed lines towards
        # its focus target.
        src = self.components["src"]
        points = src.segments[: mcstas_trace.CIRCLE_SEGMENTS].reshape(-1, 3)
        np.testing.assert_allclose(np.linalg.norm(points[:, :2], axis=1), 0.1, atol=1e-9)
        np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-12)

    def test_monitor_rectangle_is_at_its_position(self):
        psd = self.components["psd_det"]
        np.testing.assert_allclose(psd.segments[:, :, 2], 11.0, atol=1e-9)

    def test_arm_cones_follow_the_arm_rotation(self):
        # sample is unrotated at (0,0,10); right is the same Arm drawing at
        # (0.2,0,10), ROTATED (0,0,90), so its local x arrow points along y.
        sample, right = self.components["sample"], self.components["right"]
        local = sample.solid.vertices.mean(axis=0) - (0, 0, 10)
        rotated = right.solid.vertices.mean(axis=0) - (0.2, 0, 10)
        np.testing.assert_allclose(rotated, (-local[1], local[0], local[2]), atol=1e-5)


class WorldMatrixTest(unittest.TestCase):
    def test_matches_preprocess_world_matrices(self):
        """POS lines must use the same convention as
        preprocess.compute_world_matrices, or McStas components and Union
        meshes would not line up."""
        try:
            from preprocess import preprocess
            _, world, _ = preprocess(str(ROOT / "tests" / "rotated_previous_test.instr"), False)
        except Exception as e:
            self.skipTest(f"preprocess needs a McStas installation: {e}")
        text = read("rotated_previous_test_geometry_trace.txt")
        matrices = {}
        name = None
        for line in text.splitlines():
            if line.startswith("COMPONENT:"):
                name = line.split('"')[1]
            elif line.startswith("POS:"):
                matrices[name] = mcstas_trace._pos_to_matrix(mcstas_trace._floats(line[4:]))
        self.assertTrue(any(not np.allclose(M[:3, :3], np.eye(3)) for M in matrices.values()))
        for name, M in matrices.items():
            np.testing.assert_allclose(M, world[name], atol=1e-5, err_msg=name)


class DrawcallTest(unittest.TestCase):
    def test_box_extents_and_axis(self):
        _, box = _drawcall_geometry("mcdisbox(1,2,3, 0.2,0.4,0.6, 0, 0,1,0)")
        np.testing.assert_allclose(box.extents, (0.2, 0.4, 0.6), atol=1e-9)
        np.testing.assert_allclose(box.bounds.mean(axis=0), (1, 2, 3), atol=1e-9)
        self.assertAlmostEqual(box.volume, 0.2 * 0.4 * 0.6)

    def test_hollow_box_is_open_along_z(self):
        _, box = _drawcall_geometry("mcdisbox(0,0,0, 0.2,0.2,1, 0.02, 0,1,0)")
        np.testing.assert_allclose(box.extents, (0.2, 0.2, 1), atol=1e-9)
        self.assertAlmostEqual(box.volume, (0.2 * 0.2 - 0.16 * 0.16) * 1, places=9)

    def test_cylinder_along_tilted_axis(self):
        _, cyl = _drawcall_geometry("mcdiscylinder(0,0,0, 0.1,2, 0, 1,0,0)")
        self.assertAlmostEqual(cyl.extents[0], 2.0, places=6)
        self.assertAlmostEqual(cyl.extents[1], 0.2, places=3)

    def test_hollow_cylinder(self):
        _, cyl = _drawcall_geometry("mcdiscylinder(0,0,0, 0.1,1, 0.02, 0,1,0)")
        expected = np.pi * (0.1**2 - 0.08**2)
        self.assertAlmostEqual(cyl.volume, expected, delta=expected * 0.01)

    def test_cone_is_centred(self):
        _, cone = _drawcall_geometry("mcdiscone(0,0,0, 0.1,0.4, 0,0,1)")
        np.testing.assert_allclose(cone.bounds[:, 2], (-0.2, 0.2), atol=1e-9)

    def test_sphere(self):
        _, sphere = _drawcall_geometry("mcdissphere(0,0,5, 0.5)")
        np.testing.assert_allclose(sphere.bounds.mean(axis=0), (0, 0, 5), atol=1e-9)

    def test_polyhedron(self):
        text = (
            'polyhedron {"vertices": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], '
            '"faces": [{"face": [0,1,2,3]}]}'
        )
        _, mesh = _drawcall_geometry(text)
        self.assertEqual(len(mesh.faces), 2)

    def test_rectangle_and_circle_are_closed_loops(self):
        segs, _ = _drawcall_geometry("mcdisrectangle('xy',0,0,0,2,1)")
        self.assertEqual(len(segs), 4)
        np.testing.assert_allclose(segs[0, 0], segs[-1, 1])
        segs, _ = _drawcall_geometry("mcdiscircle('yz',0,0,0,1)")
        np.testing.assert_allclose(segs[:, :, 0], 0.0, atol=1e-12)

    def test_export_mesh_turns_lines_into_tubes(self):
        comp = mcstas_trace.TraceComponent(
            "slit", "Slit", np.eye(4),
            segments=_drawcall_geometry("mcdisrectangle('xy',0,0,0,0.2,0.1)")[0],
        )
        mesh = comp.export_mesh()
        self.assertTrue(mesh.is_watertight)
        np.testing.assert_allclose(
            mesh.bounds, [[-0.101, -0.051, -0.001], [0.101, 0.051, 0.001]], atol=1e-9
        )
        self.assertIsNone(mcstas_trace.TraceComponent("arm", "Arm", np.eye(4)).export_mesh())

    def test_segments_to_tubes_volume(self):
        tube = mcstas_trace.segments_to_tubes(np.array([[[0, 0, 0], [0, 0, 2.0]]]), radius=0.1)
        octagon_area = 0.5 * 8 * 0.1**2 * np.sin(2 * np.pi / 8)
        self.assertAlmostEqual(tube.volume, 2 * octagon_area)

    def test_degenerate_and_unknown_calls_draw_nothing(self):
        self.assertEqual(_drawcall_geometry("mcdissphere(0,0,0,0)"), (None, None))
        self.assertEqual(_drawcall_geometry("not_a_drawcall(1,2)"), (None, None))


HAND_WRITTEN_TRACE = """\
COMPONENT: "a"
POS: 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1
COMPONENT: "b"
POS: 0, 0, 5, 0, 1, 0, -1, 0, 0, 0, 0, 1
COMPONENT: "c"
POS: 0, 0, 20, 1, 0, 0, 0, 1, 0, 0, 0, 1
COMPONENT: "i"
POS: 0, 0, 50, 1, 0, 0, 0, 1, 0, 0, 0, 1
MCDISPLAY: start
MCDISPLAY: end
INSTRUMENT END:
ENTER:
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
COMP: "a"
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
COMP: "b"
STATE: 0, 0, -1, 0, 0, 1000, 0.004, 0, 0, 0, 1
SCATTER: 0, 0, 0, 0, 0, 1000, 0.005, 0, 0, 0, 0.5
SCATTER: 0, 0, 0.1, 1000, 0, 0, 0.0051, 0, 0, 0, 0.5
STATE: 0.2, 0, 0.1, 1000, 0, 0, 0.0053, 0, 0, 0, 0.5
ABSORB:
LEAVE:
STATE: 9, 9, 9, 0, 0, 1000, 0, 0, 0, 0, 1
ENTER:
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
COMP: "a"
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
STATE: 0, 0, 2, 0, 0, 1000, 0.002, 0, 0, 0, 1
LEAVE:
STATE: 0, 0, 3, 0, 0, 1000, 0.003, 0, 0, 0, 1
ENTER:
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
COMP: "a"
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
STATE: 0, 0, 1, 0, 0, 1000, 0.001, 0, 0, 0, 1
COMP: "c"
STATE: 0, 0, -1, 0, 0, 1000, 0.02, 0, 0, 0, 1
SCATTER: 0, 0, 0, 0, 0, 1000, 0.021, 0, 0, 0, 1
STATE: 0, 0, -1, 0, 0, 1000, 0.02, 0, 0, 0, 1
LEAVE:
STATE: 5, 5, 5, 0, 0, 1000, 0, 0, 0, 0, 1
ENTER:
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
COMP: "a"
STATE: 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 1
STATE: 0, 0, 49.9999996, 0, 0, 1000, 0.05, 0, 0, 0, 1
COMP: "i"
STATE: 0, 0, -0.0000003, 0, 0, 1000, 0.05, 0, 0, 0, 1
SCATTER: 0, 0, 5, 100, 0, 900, 0.055, 0, 0, 0, 1
STATE: 0, 0, -0.0000003, 0, 0, 1000, 0.05, 0, 0, 0, 1
LEAVE:
STATE: 0, 0, -0.0000003, 0, 0, 1000, 0.05, 0, 0, 0, 1
"""


class RayParsingTest(unittest.TestCase):
    def test_hand_written_rays(self):
        rays = parse_rays(HAND_WRITTEN_TRACE)
        self.assertEqual(rays.n_rays, 4)
        first = slice(rays.ray_offsets[0], rays.ray_offsets[1])
        # b is at z=5, rotated 90 degrees about z: local x is world y.
        np.testing.assert_allclose(
            rays.points[first],
            [(0, 0, 0), (0, 0, 4), (0, 0, 5), (0, 0, 5.1), (0, 0.2, 5.1)],
            atol=1e-12,
        )
        # Unchanged velocity: a boundary crossing. Changed velocity: a real
        # scattering. The restored STATE after an absorbed ray's LEAVE: is
        # dropped, and the absorption is marked on the last real point.
        self.assertEqual(list(rays.kind[first]), [0, 0, PASS, SCATTER, ABSORB])
        second = rays.points[rays.ray_offsets[1]:rays.ray_offsets[2]]
        np.testing.assert_allclose(second[-1], (0, 0, 3))

    def test_max_rays(self):
        self.assertEqual(parse_rays(HAND_WRITTEN_TRACE, max_rays=1).n_rays, 1)

    def test_restore_neutron_duplicate_is_dropped(self):
        # A monitor with restore_neutron=1 (e.g. PSD_monitor) prints:
        # approach STATE, a SCATTER/PASS at its detection plane, then a
        # STATE identical to the approach point (the neutron's state is
        # restored so later code sees it unperturbed). Drawing that
        # restored point would make the path jump forward to the plane
        # and then snap back - it must be dropped, leaving the path
        # ending at the detection point and the final LEAVE STATE.
        rays = parse_rays(HAND_WRITTEN_TRACE)
        third = rays.points[rays.ray_offsets[2]:rays.ray_offsets[3]]
        third_kinds = rays.kind[rays.ray_offsets[2]:rays.ray_offsets[3]]
        np.testing.assert_allclose(
            third, [(0, 0, 0), (0, 0, 1), (0, 0, 19), (0, 0, 20), (5, 5, 25)], atol=1e-12
        )
        self.assertEqual(list(third_kinds), [STATE, STATE, STATE, PASS, STATE])
        # The path never moves backward along its own direction.
        d = third[-1] - third[0]
        d = d / np.linalg.norm(d)
        self.assertTrue(np.all(np.diff((third - third[0]) @ d) >= -1e-9))

    def test_near_duplicate_points_across_frames_stay_distinct(self):
        # Two points computed through DIFFERENT components' matrices can
        # land within numpy's default relative tolerance of each other
        # (~1e-4 for values around 50) while still being genuinely
        # different physical points - e.g. the last point recorded in one
        # component and the entry point of the next, printed via two
        # different rotation matrices with %g's ~6-significant-figure
        # precision. They must not be merged into one point (that both
        # throws away real path detail and can make a later exact
        # restore-neutron duplicate fail to match its true entry point).
        rays = parse_rays(HAND_WRITTEN_TRACE)
        fourth = rays.points[rays.ray_offsets[3]:rays.ray_offsets[4]]
        fourth_kinds = rays.kind[rays.ray_offsets[3]:rays.ray_offsets[4]]
        np.testing.assert_allclose(
            fourth,
            [(0, 0, 0), (0, 0, 49.9999996), (0, 0, 49.9999997), (0, 0, 55)],
            atol=1e-7,
        )
        # The near-duplicate a/i points are distinct, not merged away, and
        # the exact restore (matching i's own entry point) is dropped -
        # the ray ends at the real scattering, not snapping back to 50.
        self.assertEqual(list(fourth_kinds), [STATE, STATE, STATE, SCATTER])
        d = fourth[-1] - fourth[0]
        d = d / np.linalg.norm(d)
        self.assertTrue(np.all(np.diff((fourth - fourth[0]) @ d) >= -1e-9))

    def test_captured_rays_are_in_world_space(self):
        rays = parse_rays(read("simple_test_ray_trace.txt"))
        self.assertEqual(rays.n_rays, 3)
        master = rays.component_names.index("master")
        # simple_test's Union_master sits at z=10; points traced in its
        # frame must land there, not near the origin.
        inside = rays.points[rays.component == master]
        self.assertTrue(np.all(np.abs(inside[:, 2] - 10) < 0.5))
        # McStas's pre-source placeholder state (at the origin, 1 m/s) is
        # dropped: every ray starts on the source disc (radius 0.1, z=0).
        starts = rays.points[rays.ray_offsets[:-1]]
        self.assertTrue(np.all(np.linalg.norm(starts[:, :2], axis=1) <= 0.1))
        np.testing.assert_allclose(starts[:, 2], 0, atol=1e-12)
        self.assertGreater(rays.speed.min(), 100)
        self.assertEqual(int((rays.kind == ABSORB).sum()), 2)

    def test_geometry_only_trace_has_no_rays(self):
        self.assertEqual(parse_rays(read("simple_test_geometry_trace.txt")).n_rays, 0)


@unittest.skipUnless(shutil.which("mcrun"), "mcrun not on PATH")
class EndToEndTest(unittest.TestCase):
    def test_trace_simple_instrument(self):
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "simple_test.instr")
            shutil.copy(ROOT / "tests" / "simple_test.instr", instr)
            try:
                components, rays = mcstas_trace.trace_instrument(instr, ncount=2, seed=7)
            except mcstas_trace.TraceError as e:
                self.skipTest(f"mcrun could not compile here: {e}")
        self.assertIn("src", components)
        self.assertNotIn("master", components)
        self.assertEqual(rays.n_rays, 2)

    def test_parameter_values_reach_mcrun(self):
        text = (ROOT / "tests" / "simple_test.instr").read_text()
        text = text.replace(
            "DEFINE INSTRUMENT ODIN (", "DEFINE INSTRUMENT ODIN (psd_w = 0.2, double nodef", 1
        ).replace("xwidth = 1,", "xwidth = psd_w,", 1)
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "param_test.instr")
            Path(instr).write_text(text)
            try:
                components, _ = mcstas_trace.trace_instrument(
                    instr, params=["psd_w=0.5", "nodef=1"]
                )
            except mcstas_trace.TraceError as e:
                self.skipTest(f"mcrun could not compile here: {e}")
            width = np.ptp(components["psd_det"].segments[:, :, 0])
            self.assertAlmostEqual(width, 0.5)
            # A parameter without a default must fail with McStas's own
            # message, not wait for input.
            with self.assertRaisesRegex(mcstas_trace.TraceError, "nodef left unset"):
                mcstas_trace.trace_instrument(instr, params=["psd_w=0.5"])


if __name__ == "__main__":
    unittest.main()
