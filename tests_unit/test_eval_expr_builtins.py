# Regression test for src/preprocess.py's eval_expr environment: McStas's
# built-in physical constants (MCSTAS_CONSTANTS) and the C-callable
# builtins that aren't part of Python's math module (MATH_ENV). Run
# directly:
#
#   python tests_unit/test_eval_expr_builtins.py

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
sys.path.insert(0, SRC_DIR)

import preprocess as pp  # noqa: E402


class McStasConstantsTest(unittest.TestCase):
    def test_k2v_and_v2k_are_reciprocal_conversions(self):
        # K2V/V2K are #define aliases of AA2MS/MS2AA in mcstas-r.h - lock
        # that relationship in, not just their standalone values.
        self.assertEqual(pp.eval_expr("K2V"), pp.eval_expr("AA2MS"))
        self.assertEqual(pp.eval_expr("V2K"), pp.eval_expr("MS2AA"))

    def test_se2v_and_vs2e_are_usable_in_a_real_expression(self):
        # Mirrors the corpus pattern this constant set exists to fix:
        # vi = K2V*fabs(Ki); Ei = VS2E*vi*vi;
        var_map = {"Ki": 2.5}
        vi = pp.eval_expr("K2V*fabs(Ki)", var_map)
        var_map["vi"] = vi
        ei = pp.eval_expr("VS2E*vi*vi", var_map)
        self.assertGreater(ei, 0)

    def test_null_and_na_are_defined(self):
        self.assertEqual(pp.eval_expr("NULL"), 0)
        self.assertGreater(pp.eval_expr("NA"), 6e23)


class MathEnvBuiltinsTest(unittest.TestCase):
    def test_abs_is_available(self):
        # abs() is a Python builtin, not a math module member, so
        # dir(math) alone doesn't provide it - this is what 2.4 fixes.
        self.assertEqual(pp.eval_expr("abs(-5.0)"), 5.0)

    def test_min_and_max_are_available(self):
        var_map = {"a": 3.0, "b": 7.0}
        self.assertEqual(pp.eval_expr("min(a, b)", var_map), 3.0)
        self.assertEqual(pp.eval_expr("max(a, b)", var_map), 7.0)

    def test_abs_matches_the_real_corpus_pattern(self):
        # det_angle = abs(ang_fin - ang_ini) / 2.0 + ang_ini;
        var_map = {"ang_fin": 10.0, "ang_ini": 30.0}
        det_angle = pp.eval_expr(
            "abs(ang_fin-ang_ini)/2.0 + ang_ini", var_map
        )
        self.assertEqual(det_angle, 40.0)


if __name__ == "__main__":
    unittest.main()
