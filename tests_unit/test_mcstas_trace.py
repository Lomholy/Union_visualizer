"""Tests for mcstas_trace.py: parsing mcrun --trace=2 output into
world-space McStas component geometry.

The fixtures in tests_unit/data/ are real `mcrun <instr> --trace=2
--no-output-files -n 0 -y` output for tests/simple_test.instr and
tests/rotated_previous_test.instr, so the parser tests need no McStas
installation. The end-to-end test at the bottom runs mcrun itself and is
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
from mcstas_trace import _drawcall_geometry, parse_trace  # noqa: E402

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


@unittest.skipUnless(shutil.which("mcrun"), "mcrun not on PATH")
class EndToEndTest(unittest.TestCase):
    def test_trace_simple_instrument(self):
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "simple_test.instr")
            shutil.copy(ROOT / "tests" / "simple_test.instr", instr)
            try:
                components = mcstas_trace.trace_instrument(instr)
            except mcstas_trace.TraceError as e:
                self.skipTest(f"mcrun could not compile here: {e}")
        self.assertIn("src", components)
        self.assertNotIn("master", components)

    def test_parameter_values_reach_mcrun(self):
        text = (ROOT / "tests" / "simple_test.instr").read_text()
        text = text.replace(
            "DEFINE INSTRUMENT ODIN (", "DEFINE INSTRUMENT ODIN (psd_w = 0.2, double nodef", 1
        ).replace("xwidth = 1,", "xwidth = psd_w,", 1)
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "param_test.instr")
            Path(instr).write_text(text)
            try:
                components = mcstas_trace.trace_instrument(
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
