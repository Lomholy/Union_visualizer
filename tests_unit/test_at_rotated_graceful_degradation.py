# Regression tests for src/preprocess.py's AT/ROTATED graceful-degradation
# fix: parse_param() now warns (instead of silently swallowing) when it
# can't evaluate a component parameter/AT/ROTATED expression, COMP_GETPAR
# is resolved against instr.component_list, struct member access gets a
# specific "unsupported" message instead of a generic one, and
# compute_world_matrices() defaults an unresolved component's own local
# transform to zero and keeps going instead of aborting the whole
# instrument. Run directly:
#
#   python tests_unit/test_at_rotated_graceful_degradation.py

import io
import os
import sys
import unittest
import contextlib

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
import mcstasscript as ms
import preprocess as pp  # noqa: E402


class ParseParamWarningTest(unittest.TestCase):
    def test_genuine_failure_prints_a_warning_and_returns_the_raw_expression(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.parse_param("some_undefined_var*2", {})
        self.assertEqual(result, "some_undefined_var*2")
        self.assertIn("Warning: Failed to evaluate", buf.getvalue())

    def test_already_numeric_value_is_passed_through_silently(self):
        # This used to rely on eval_expr(5.0, ...) raising a TypeError that
        # got silently swallowed - now it's a fast, explicit, silent path.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.parse_param(5.0, {})
        self.assertEqual(result, 5.0)
        self.assertEqual(buf.getvalue(), "")

    def test_successful_evaluation_prints_nothing(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.parse_param("2*x", {"x": 3.0})
        self.assertEqual(result, 6.0)
        self.assertEqual(buf.getvalue(), "")

    def test_opaque_runtime_call_prints_info_not_warning(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.parse_param("mcget_ncount()", {})
        self.assertEqual(result, "mcget_ncount()")
        output = buf.getvalue()
        self.assertIn("Info:", output)
        self.assertNotIn("Warning:", output)

    def test_struct_member_access_gets_a_specific_message(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.parse_param("machine_hkl.l1", {})
        self.assertEqual(result, "machine_hkl.l1")
        self.assertIn("Unsupported: struct member access", buf.getvalue())


class CompGetparTest(unittest.TestCase):
    def setUp(self):
        self.instr = ms.McStas_instr("unit_test")
        # Guide has a real "l" (length) parameter - COMP_GETPAR reads a
        # genuine McStas component parameter, so this needs a component
        # type that actually declares one, unlike the bare Arm used
        # elsewhere in this file for pure AT/ROTATED geometry tests.
        self.instr.add_component(
            "mono", "Guide", AT=[0, 0, 0], AT_RELATIVE="ABSOLUTE"
        )
        self.instr.component_list[-1].l = 3.5  # noqa: E741 - real Guide param
        self.instr.add_component(
            "sample", "Arm", AT=[0, 0, 1], AT_RELATIVE="ABSOLUTE"
        )

    def test_comp_getpar_previous_resolves_the_prior_component(self):
        comp_context = (self.instr, self.instr.component_list[-1])
        value = pp.parse_param("COMP_GETPAR(PREVIOUS,l)", {}, comp_context)
        self.assertEqual(value, 3.5)

    def test_comp_getpar_by_name_resolves_a_named_component(self):
        comp_context = (self.instr, self.instr.component_list[-1])
        value = pp.parse_param("COMP_GETPAR(mono,l)", {}, comp_context)
        self.assertEqual(value, 3.5)

    def test_comp_getpar_previous_on_first_component_warns_and_stays_unresolved(self):
        comp_context = (self.instr, self.instr.component_list[0])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            value = pp.parse_param("COMP_GETPAR(PREVIOUS,l)", {}, comp_context)
        self.assertEqual(value, "COMP_GETPAR(PREVIOUS,l)")
        self.assertIn("Warning:", buf.getvalue())

    def test_comp_getpar_unknown_component_warns_and_stays_unresolved(self):
        comp_context = (self.instr, self.instr.component_list[-1])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            value = pp.parse_param("COMP_GETPAR(nonexistent,l)", {}, comp_context)
        self.assertEqual(value, "COMP_GETPAR(nonexistent,l)")
        self.assertIn("Warning:", buf.getvalue())

    def test_comp_getpar_used_without_a_comp_context_warns(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            value = pp.parse_param("COMP_GETPAR(PREVIOUS,l)", {})
        self.assertEqual(value, "COMP_GETPAR(PREVIOUS,l)")
        self.assertIn("Warning:", buf.getvalue())


class ComputeWorldMatricesGracefulDegradationTest(unittest.TestCase):
    def test_unresolved_component_defaults_to_zero_and_the_rest_still_computes(self):
        instr = ms.McStas_instr("unit_test")
        instr.add_component("origin", "Arm", AT=[0, 0, 0], AT_RELATIVE="ABSOLUTE")
        # broken: AT still holds an unresolved expression string (as if
        # eval_expr had failed on it upstream), instead of a float.
        instr.add_component(
            "broken", "Arm",
            AT=["some_undefined_var", 0, 0], AT_RELATIVE="origin",
        )
        instr.add_component(
            "downstream", "Arm",
            AT=[0, 0, 5], AT_RELATIVE="broken",
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            world = pp.compute_world_matrices(instr)

        self.assertIn("Warning: Component 'broken': could not resolve AT", buf.getvalue())
        # broken's own AT defaulted to zero, so its world position is
        # exactly origin's (identity offset).
        np.testing.assert_allclose(world["broken"][:3, 3], [0.0, 0.0, 0.0], atol=1e-7)
        # downstream is still computed at all (not aborted) and correctly
        # offset from broken's (zeroed) position.
        self.assertIn("downstream", world)
        np.testing.assert_allclose(world["downstream"][:3, 3], [0.0, 0.0, 5.0], atol=1e-7)

    def test_fully_resolved_instrument_prints_no_at_rotated_warning(self):
        instr = ms.McStas_instr("unit_test")
        instr.add_component("origin", "Arm", AT=[0, 0, 0], AT_RELATIVE="ABSOLUTE")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pp.compute_world_matrices(instr)
        self.assertEqual(buf.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
