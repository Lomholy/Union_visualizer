# Regression tests for src/preprocess.py's INITIALIZE parsing of braced
# if / else if / else chains and strcmp conditions. create_var_map used to
# walk INITIALIZE line by line and only understood one-line
# "if (cond) x = a;", so every assignment inside a braced branch ran
# whether or not its branch was taken, and the last one won. Run directly:
#
#   python tests_unit/test_initialize_blocks.py

import io
import os
import sys
import unittest
import contextlib

import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import mcstasscript as ms
import preprocess as pp  # noqa: E402

# The setup selection of McCode's Test_inhomogeneous_process.instr
SAMPLE_CHAIN = """
rot_samples = 0;
width = 0.03;
if (!strcmp("thin", sample)){
  is_thin = 1;
} else if (!strcmp("inc_trans", sample)) {
  rot_samples = 90;   // transmission geometry
  activate_inc_linear = 1;
  width = 0.01;
} else if (!strcmp("gauss_heavy", sample)){
  rot_samples = 90;
  activate_gauss = 1;
  width = 0.01;
} else {
  fprintf (stderr, "Unsupported instrument parameter value sample=\\"%s\\".\\n", sample);
  exit (-1);
}
"""


def var_map_for(initialize, params=(), param_values=None):
    instr = ms.McStas_instr("unit_test")
    for name, value in params:
        instr.add_parameter("string", name, value=value)
    instr.append_initialize(initialize)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        var_map = pp.create_var_map(instr, param_values)
    return var_map, buf.getvalue()


class StrcmpTest(unittest.TestCase):
    def test_equal_strings(self):
        self.assertEqual(pp.eval_expr('strcmp("a", "a")'), 0)

    def test_order(self):
        self.assertLess(pp.eval_expr('strcmp("a", "b")'), 0)
        self.assertGreater(pp.eval_expr('strcmp("b", "a")'), 0)

    def test_quoted_parameter_value_matches_literal(self):
        self.assertEqual(pp.eval_expr('strcmp("thin", sample)', {"sample": '"thin"'}), 0)

    def test_strstr_and_strcasestr(self):
        self.assertTrue(pp.eval_expr('strstr(pol, "y")', {"pol": '"xy"'}))
        self.assertFalse(pp.eval_expr('strstr(pol, "z")', {"pol": '"xy"'}))
        self.assertTrue(pp.eval_expr('strcasestr(sector, "n")', {"sector": '"N"'}))


class PythonKeywordNameTest(unittest.TestCase):
    def test_lambda_is_an_ordinary_variable(self):
        self.assertEqual(pp.eval_expr("2*lambda", {"lambda": 1.5}), 3.0)

    def test_keyword_inside_a_string_literal_is_kept(self):
        self.assertEqual(pp.eval_expr('strcmp("lambda", s)', {"s": "lambda"}), 0)

    def test_lambda_condition_in_initialize(self):
        var_map, output = var_map_for("lambda = 0; TOA = 30;\nif (lambda == 0 && TOA != 0)\n  lambda = 2*TOA;")
        self.assertEqual(var_map["lambda"], 60)
        self.assertEqual(output, "")


class BracedChainTest(unittest.TestCase):
    def test_default_parameter_takes_only_its_branch(self):
        var_map, output = var_map_for(SAMPLE_CHAIN, [("sample", '"thin"')])
        self.assertEqual(var_map["is_thin"], 1)
        self.assertEqual(var_map["rot_samples"], 0)
        self.assertEqual(var_map["width"], 0.03)
        self.assertNotIn("activate_inc_linear", var_map)
        self.assertNotIn("activate_gauss", var_map)
        self.assertEqual(output, "")

    def test_overridden_parameter_takes_its_else_if_branch(self):
        var_map, output = var_map_for(SAMPLE_CHAIN, [("sample", '"thin"')], {"sample": "inc_trans"})
        self.assertEqual(var_map["rot_samples"], 90)
        self.assertEqual(var_map["width"], 0.01)
        self.assertEqual(var_map["activate_inc_linear"], 1)
        self.assertNotIn("is_thin", var_map)
        self.assertNotIn("activate_gauss", var_map)
        self.assertEqual(output, "")

    def test_no_match_takes_the_else_branch(self):
        var_map, output = var_map_for(SAMPLE_CHAIN, [("sample", '"thin"')], {"sample": "other"})
        self.assertEqual(var_map["rot_samples"], 0)
        self.assertNotIn("is_thin", var_map)
        self.assertEqual(output, "")

    def test_unbraced_else_if_chain(self):
        var_map, output = var_map_for("x = 2; if (x == 1) y = 1; else if (x == 2) y = 2; else y = 3;")
        self.assertEqual(var_map["y"], 2)
        self.assertEqual(output, "")

    def test_nested_if_inside_a_branch(self):
        var_map, output = var_map_for("a = 1; b = 0; if (a) { if (b) y = 1; else y = 2; z = 3; }")
        self.assertEqual(var_map["y"], 2)
        self.assertEqual(var_map["z"], 3)

    def test_failing_condition_runs_no_branch(self):
        var_map, output = var_map_for("if (some_typo_var) { y = 1; } else { y = 2; }")
        self.assertNotIn("y", var_map)
        self.assertIn("Warning: Failed to evaluate if (some_typo_var) { ... }", output)

    def test_if_without_assignments_is_skipped_silently(self):
        var_map, output = var_map_for('if (!mcdotrace) { printf("no trace\\n"); exit(1); }')
        self.assertEqual(output, "")

    def test_local_declaration_with_initializer(self):
        var_map, output = var_map_for("A1 = 2;\nchar   anglemode = 0;\nif (A1) anglemode=1;")
        self.assertEqual(var_map["anglemode"], 1)
        self.assertEqual(output, "")

    def test_condition_over_several_lines(self):
        var_map, output = var_map_for("a = 0; b = 0;\nif (a == 0\n    || b == 1) { y = 1; }")
        self.assertEqual(var_map["y"], 1)
        self.assertEqual(output, "")

    def test_statement_over_several_lines(self):
        var_map, output = var_map_for("x = 1 +\n    2;")
        self.assertEqual(var_map["x"], 3)
        self.assertEqual(output, "")

    def test_comments_are_ignored(self):
        var_map, output = var_map_for("/* x = 5; */ x = 1; // x = 7; { }\ny = x;")
        self.assertEqual(var_map["x"], 1)
        self.assertEqual(var_map["y"], 1)
        self.assertEqual(output, "")


class DemonstrationFixtureTest(unittest.TestCase):
    """tests/initialize_blocks_test.instr rotates its sample arm by
    rot_samples, which only the branch of the selected setup sets."""

    def rotation_of_box(self, sample):
        instr_path = os.path.join(REPO_ROOT, "tests", "initialize_blocks_test.instr")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            instr, world, _ = pp.preprocess(instr_path, verbose=False, param_values={"sample": sample})
        return world["box"][:3, :3], buf.getvalue()

    def test_default_setup_is_not_rotated(self):
        rotation, output = self.rotation_of_box('"thin"')
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        self.assertEqual(output, "")

    def test_transmission_setup_is_rotated_90_degrees_about_y(self):
        rotation, output = self.rotation_of_box('"inc_trans"')
        np.testing.assert_allclose(rotation, [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
        self.assertEqual(output, "")


if __name__ == "__main__":
    unittest.main()
