"""Numeric regression tests for independent AT/ROTATED parent resolution.

These build small instruments directly through mcstasscript's Python API
(bypassing var_map/attempt_conversion, since AT_data/ROTATED_data are
already numeric) and assert exact expected world matrices, derived by hand
from the formula documented in Desktop/Visualizer_separate_AT_ROT_plan.md
Section 1 (itself grounded in McCode's cogen_comp_init_position()):

    R[comp] = R[ROT_parent] @ R_local(comp)
    t[comp] = t[AT_parent] + R[AT_parent] @ AT_data(comp)

This directory is intentionally NOT under tests/, since
.github/workflows/run-tests.yml feeds every file in tests/* to
src/mcstas_to_cad.py as a pipeline smoke test; a plain assertion script
there would break that loop.
"""

import io
import sys
import unittest
import contextlib
from pathlib import Path

import numpy as np
import mcstasscript as ms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import preprocess  # noqa: E402


def expected_rotation_matrix(rx_deg, ry_deg, rz_deg):
    """Independent re-implementation of preprocess.py's rotation_matrix(),
    intentionally duplicated: what's under test here is parent composition
    (which world matrix gets multiplied with which), not this Euler
    convention, which the plan explicitly leaves untouched."""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def make_matrix(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


TOL = 1e-7


class TestATRotSeparation(unittest.TestCase):
    def setUp(self):
        self.instr = ms.McStas_instr("unit_test")
        # origin: pinned to world, no rotation.
        self.instr.add_component("origin", "Arm", AT=[0, 0, 0], AT_RELATIVE="ABSOLUTE")
        # A: relative to origin, both AT and ROTATED explicit and equal
        # (the common case), rotated 90 degrees about Y so its inherited
        # rotation is easy to distinguish from identity in later assertions.
        self.instr.add_component(
            "A", "Arm",
            AT=[1, 0, 0], AT_RELATIVE="origin",
            ROTATED=[0, 90, 0], ROTATED_RELATIVE="origin",
        )
        self.R_A = expected_rotation_matrix(0, 90, 0)
        self.t_A = np.array([1.0, 0.0, 0.0])

    def world(self):
        return preprocess.compute_world_matrices(self.instr)

    def test_at_and_rotated_relative_to_two_different_named_components(self):
        # B: AT relative to A, ROTATED relative to origin (a different,
        # named component) - the core case this plan fixes: PR #8's code
        # would have used origin's rotation AND origin's position (wrongly
        # discarding A as the AT parent) for B.
        self.instr.add_component(
            "B", "Arm",
            AT=[0, 0, 2], AT_RELATIVE="A",
            ROTATED=[0, 0, 45], ROTATED_RELATIVE="origin",
        )
        world = self.world()

        expected_R = expected_rotation_matrix(0, 0, 45)  # R[origin] is identity
        expected_t = self.t_A + self.R_A @ np.array([0.0, 0.0, 2.0])
        np.testing.assert_allclose(world["B"], make_matrix(expected_R, expected_t), atol=TOL)

    def test_rotated_omitted_mirrors_at_reference(self):
        # C: AT relative to A, ROTATED omitted entirely (never call
        # set_ROTATED -> ROTATED_specified stays False). Must inherit A's
        # rotation, not fall back to ABSOLUTE (which is what
        # mcstasscript's own unset default would naively give).
        self.instr.add_component("C", "Arm", AT=[0, 0, 1], AT_RELATIVE="A")
        self.assertFalse(self.instr.component_list[-1].ROTATED_specified)

        world = self.world()

        expected_R = self.R_A
        expected_t = self.t_A + self.R_A @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(world["C"], make_matrix(expected_R, expected_t), atol=TOL)

    def test_rotated_absolute_is_not_composed_with_at_parent_rotation(self):
        # D: AT relative to A (which is rotated), ROTATED explicitly
        # ABSOLUTE. D's world rotation must be exactly its own local
        # angles, with NO contribution from A's rotation - PR #8's code
        # wrongly folded A's rotation in here because it used AT_rel as
        # the sole parent whenever ROT_rel == "ABSOLUTE".
        self.instr.add_component(
            "D", "Arm",
            AT=[0, 0, 1], AT_RELATIVE="A",
            ROTATED=[0, 0, 30], ROTATED_RELATIVE="ABSOLUTE",
        )
        world = self.world()

        expected_R = expected_rotation_matrix(0, 0, 30)
        expected_t = self.t_A + self.R_A @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(world["D"], make_matrix(expected_R, expected_t), atol=TOL)

    def test_at_absolute_is_not_affected_by_rotated_parent(self):
        # E: AT ABSOLUTE, ROTATED relative to A. World position must be
        # exactly the literal AT coordinates, regardless of A's transform.
        self.instr.add_component(
            "E", "Arm",
            AT=[5, 6, 7], AT_RELATIVE="ABSOLUTE",
            ROTATED=[0, 0, 60], ROTATED_RELATIVE="A",
        )
        world = self.world()

        expected_R = self.R_A @ expected_rotation_matrix(0, 0, 60)
        expected_t = np.array([5.0, 6.0, 7.0])
        np.testing.assert_allclose(world["E"], make_matrix(expected_R, expected_t), atol=TOL)

    def test_rotated_previous_with_at_relative_to_a_different_named_component(self):
        # Numeric re-assertion of PR #8's original fix, sharper than the
        # existing tests/rotated_previous_test.instr pipeline smoke test:
        # spacer sits between A and F in declaration order, so PREVIOUS
        # (for F's ROTATED) resolves to "spacer", while F's AT is relative
        # to "A" directly - two different, non-PREVIOUS-spelled names.
        self.instr.add_component("spacer", "Arm", AT=[0, 0, 0.3], AT_RELATIVE="A")
        R_spacer = self.R_A  # ROTATED omitted -> mirrors AT's parent (A)
        t_spacer = self.t_A + self.R_A @ np.array([0.0, 0.0, 0.3])

        self.instr.add_component(
            "F", "Arm",
            AT=[0, 0, 0.5], AT_RELATIVE="A",
            ROTATED=[0, 0, 10], ROTATED_RELATIVE="PREVIOUS",
        )
        world = self.world()

        expected_R = R_spacer @ expected_rotation_matrix(0, 0, 10)
        expected_t = self.t_A + self.R_A @ np.array([0.0, 0.0, 0.5])
        np.testing.assert_allclose(world["F"], make_matrix(expected_R, expected_t), atol=TOL)
        # Sanity: spacer itself resolved as expected too.
        np.testing.assert_allclose(world["spacer"], make_matrix(R_spacer, t_spacer), atol=TOL)

    def test_previous_on_first_component_warns_and_falls_back_to_absolute(self):
        # Real McStas's own grammar (instrument.y: compref -> PREVIOUS)
        # warns and falls back to ABSOLUTE for a first-component PREVIOUS
        # reference rather than erroring - and real corpus instruments
        # (e.g. ILL_H53_D16.instr) rely on exactly this. Build a fresh
        # instrument whose only/first component does this.
        instr = ms.McStas_instr("first_previous_test")
        instr.add_component(
            "first", "Arm", AT=[1, 2, 3], AT_RELATIVE="PREVIOUS",
            ROTATED=[4, 5, 6], ROTATED_RELATIVE="PREVIOUS",
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            world = preprocess.compute_world_matrices(instr)

        self.assertIn("Warning:", buf.getvalue())
        self.assertIn("first", buf.getvalue())

        expected_R = expected_rotation_matrix(4, 5, 6)
        expected_t = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(world["first"], make_matrix(expected_R, expected_t), atol=TOL)


if __name__ == "__main__":
    unittest.main()
