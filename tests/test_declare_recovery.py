# Regression test for src/preprocess.py's DECLARE raw-string recovery.
#
# Not a pytest suite (this project has no test framework dependency) - run
# directly:
#
#   python tests/test_declare_recovery.py
#
# Covers the PR #7 review comment on create_var_map: a blanket
# "isinstance(v, str): continue" skip over mcstasscript's raw-string
# DECLARE/USERVARS entries would silently discard variables mcstasscript
# misclassifies as non-variable content (its "is this a function?" heuristic
# misfires on an ordinary multi-line initializer like
# "double result = fabs(\n  -5.0\n);"), even though preprocess.py's own
# evaluator could compute them. _recover_declare_assignments rejoins a run
# of raw-string entries and pulls out any real "name = expr;" before giving
# up on the rest.
#
# This is tested directly against create_var_map/_recover_declare_assignments
# rather than via a full .instr fixture: embedding this exact mcstasscript
# misclassification pattern inside a real instrument file was found to trip
# a *different*, unrelated mcstasscript reader crash (an IndexError inside
# DeclareVariable's constructor, from the same heuristic corrupting the
# DECLARE/INITIALIZE section boundary) before preprocess.py's own code ever
# runs - see tests/rotated_previous_test.instr's docstring for the details.

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
sys.path.insert(0, SRC_DIR)

import preprocess as pp  # noqa: E402


class DeclareRecoveryTest(unittest.TestCase):
    def test_define_line_is_silently_skipped(self):
        # A genuine #define has no "name = expr" shape at all - nothing to
        # recover, and no warning should be printed for it.
        var_map = {}
        pp._recover_declare_assignments(['#define VERSION "1.7"'], var_map)
        self.assertEqual(var_map, {})

    def test_misclassified_multiline_expression_is_recovered(self):
        # mcstasscript dumps "double result = fabs(\n  -5.0\n);" as three
        # separate raw-string entries (append_declare is called once per
        # physical source line). Rejoining them should recover the value.
        var_map = {}
        pp._recover_declare_assignments(
            ["double result = fabs(", "    -5.0", ");"], var_map
        )
        self.assertEqual(var_map, {"result": 5.0})

    def test_function_body_is_not_treated_as_top_level_assignments(self):
        # A genuine multi-line C function is also dumped as raw strings, and
        # can easily contain an assignment-shaped internal statement (here,
        # "result = a + b;"). That's function-local, not an instrument-global
        # DECLARE variable, and must not be recovered - the presence of a
        # brace pair is what distinguishes this from the case above.
        var_map = {}
        pp._recover_declare_assignments(
            [
                "double calcAlpha(double a, double b) {",
                "  double result;",
                "  result = a + b;",
                "  return result;",
                "}",
            ],
            var_map,
        )
        self.assertEqual(var_map, {})

    def test_add_declare_vars_to_map_groups_consecutive_raw_strings(self):
        # End-to-end through the grouping pass: a normal DeclareVariable,
        # then a run of raw strings representing the misclassified
        # expression, then another normal DeclareVariable - all three
        # should end up correctly represented in var_map.
        class FakeDeclareVar:
            def __init__(self, name, value, type_="double"):
                self.name = name
                self.value = value
                self.type = type_

        entries = [
            FakeDeclareVar("angle", 30.0),
            "double result = fabs(",
            "    -5.0",
            ");",
            FakeDeclareVar("gap", ""),  # declared but never initialized
        ]
        var_map = {}
        pp._add_declare_vars_to_map(entries, var_map)
        self.assertEqual(var_map["angle"], 30.0)
        self.assertEqual(var_map["result"], 5.0)
        self.assertEqual(var_map["gap"], 0.0)


if __name__ == "__main__":
    unittest.main()
