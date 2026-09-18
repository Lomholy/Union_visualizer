# Regression test for src/preprocess.py's handling of opaque runtime calls
# (malloc, create_darr1d, mcget_ncount): these genuinely have no static
# value at preprocessing time and shouldn't be reported identically to a
# typo'd variable name. Run directly:
#
#   python tests_unit/test_opaque_runtime_calls.py

import io
import os
import sys
import unittest
import contextlib

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
sys.path.insert(0, SRC_DIR)

import preprocess as pp  # noqa: E402


class EvalExprOpaqueCallTest(unittest.TestCase):
    def test_mcget_ncount_raises_opaque_runtime_call(self):
        with self.assertRaises(pp.OpaqueRuntimeCall):
            pp.eval_expr("mcget_ncount()")

    def test_malloc_raises_opaque_runtime_call_without_evaluating_its_args(self):
        # sizeof(char) isn't valid Python and isn't in MATH_ENV either - if
        # eval_expr tried to evaluate the args first, this would instead
        # raise "Unknown variable: sizeof", masking the real category.
        with self.assertRaises(pp.OpaqueRuntimeCall):
            pp.eval_expr("malloc(150*sizeof(char))")

    def test_create_darr1d_raises_opaque_runtime_call(self):
        with self.assertRaises(pp.OpaqueRuntimeCall):
            pp.eval_expr("create_darr1d(10)")

    def test_a_real_unknown_name_still_raises_plain_value_error(self):
        # Guards against accidentally widening OPAQUE_RUNTIME_CALLS or the
        # Call-node check into swallowing genuine parser gaps/typos.
        with self.assertRaises(ValueError):
            pp.eval_expr("some_typoed_function(1)")

    def test_opaque_call_nested_in_a_larger_expression_still_propagates(self):
        # Mirrors the real corpus pattern: p0 = 1.0/mcget_ncount(); - the
        # call isn't the whole expression, but the surrounding expression
        # still has no static value either, so it should still propagate.
        with self.assertRaises(pp.OpaqueRuntimeCall):
            pp.eval_expr("1.0/mcget_ncount()")


class CreateVarMapOpaqueCallTest(unittest.TestCase):
    def test_opaque_call_prints_info_not_warning_and_leaves_variable_unresolved(self):
        import mcstasscript as ms

        instr = ms.McStas_instr("unit_test")
        instr.append_initialize("p0=1.0/mcget_ncount();")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            var_map = pp.create_var_map(instr)
        output = buf.getvalue()
        self.assertNotIn("p0", var_map)
        self.assertIn("Info:", output)
        self.assertIn("mcget_ncount", output)
        self.assertNotIn("Warning:", output)

    def test_unrelated_unknown_variable_still_prints_a_warning(self):
        import mcstasscript as ms

        instr = ms.McStas_instr("unit_test")
        instr.append_initialize("foo=some_typo_var*2;")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            var_map = pp.create_var_map(instr)
        output = buf.getvalue()
        self.assertNotIn("foo", var_map)
        self.assertIn("Warning: Failed to evaluate", output)
        self.assertNotIn("Info:", output)


if __name__ == "__main__":
    unittest.main()
