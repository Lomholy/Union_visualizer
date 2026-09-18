# Regression test for src/preprocess.py's eval_expr division-by-zero
# handling: a division (or mod, or negative-power-of-zero) by zero
# shouldn't abort the whole expression with an unhandled ZeroDivisionError -
# it should warn and default that sub-result to 0, since the parameter
# hitting it may not even be essential to the Union geometry being
# visualized. Run directly:
#
#   python tests_unit/test_division_by_zero.py

import io
import os
import sys
import unittest
import contextlib

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
sys.path.insert(0, SRC_DIR)

import preprocess as pp  # noqa: E402


class DivisionByZeroTest(unittest.TestCase):
    def test_plain_division_by_zero_defaults_to_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.eval_expr("5.0/cnum", {"cnum": 0.0})
        self.assertEqual(result, 0)
        self.assertIn("Warning: Division by zero", buf.getvalue())

    def test_no_warning_printed_when_denominator_is_nonzero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.eval_expr("5.0/cnum", {"cnum": 2.0})
        self.assertEqual(result, 2.5)
        self.assertEqual(buf.getvalue(), "")

    def test_division_by_zero_nested_in_a_larger_expression(self):
        # Mirrors the real corpus pattern: polz_w1 = 0.05/cnum; where cnum
        # defaulted to 0.0 - only the failing sub-expression should default
        # to 0, the rest of the expression still evaluates normally.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.eval_expr("1.0 + 5.0/cnum", {"cnum": 0.0})
        self.assertEqual(result, 1.0)
        self.assertIn("Warning: Division by zero", buf.getvalue())

    def test_mod_by_zero_also_defaults_to_zero(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            result = pp.eval_expr("5.0 % z", {"z": 0.0})
        self.assertEqual(result, 0)
        self.assertIn("Warning: Division by zero", buf.getvalue())

    def test_create_var_map_records_zero_instead_of_dropping_the_variable(self):
        # Before this fix, create_var_map's except-Exception fallback would
        # print a "Failed to evaluate" warning and leave the variable out of
        # var_map entirely, which then cascaded into every later expression
        # referencing it. Now the assignment should succeed with value 0.
        import mcstasscript as ms

        instr = ms.McStas_instr("unit_test")
        instr.append_initialize("cnum = 0.0;")
        instr.append_initialize("polz_w1 = 0.05/cnum;")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            var_map = pp.create_var_map(instr)
        self.assertIn("polz_w1", var_map)
        self.assertEqual(var_map["polz_w1"], 0)
        self.assertIn("Warning: Division by zero", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
