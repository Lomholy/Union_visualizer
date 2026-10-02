# Regression tests for src/preprocess.py's multi-statement-line handling
# (2.1): create_var_map's INITIALIZE loop used to match a whole line
# against one "name = expr;" regex, so several ';'-terminated C statements
# on one line (extremely common in real McStas INITIALIZE sections) made
# the entire line fail with "invalid syntax" and none of its assignments
# happened. It now splits each line on top-level ';' and evaluates each
# statement independently, including minimal single-line
# "if (cond) name = expr; [else name2 = expr2;]" conditional support. Run
# directly:
#
#   python tests_unit/test_multi_statement_lines.py

import io
import os
import sys
import unittest
import contextlib

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
SRC_DIR = os.path.join(REPO_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import mcstasscript as ms
import preprocess as pp  # noqa: E402


def var_map_for(*initialize_lines):
    instr = ms.McStas_instr("unit_test")
    for line in initialize_lines:
        instr.append_initialize(line)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        var_map = pp.create_var_map(instr)
    return var_map, buf.getvalue()


class PlainMultiStatementLineTest(unittest.TestCase):
    def test_three_semicolon_separated_assignments_on_one_line(self):
        # The exact corpus pattern: 'SM = 1; SS = -1; SA = 1;'
        var_map, output = var_map_for("SM = 1; SS = -1; SA = 1;")
        self.assertEqual(var_map["SM"], 1)
        self.assertEqual(var_map["SS"], -1)
        self.assertEqual(var_map["SA"], 1)
        self.assertEqual(output, "")

    def test_tab_separated_assignments(self):
        # The exact corpus pattern: 'sT1=313.;\tsI1=8.05e12;'
        var_map, output = var_map_for("sT1=313.;\tsI1=8.05e12;")
        self.assertEqual(var_map["sT1"], 313.0)
        self.assertEqual(var_map["sI1"], 8.05e12)
        self.assertEqual(output, "")

    def test_second_assignment_can_reference_a_variable_from_earlier_in_the_corpus(self):
        # The exact corpus pattern: 'A4 = A2; A6 = A2;' - both reference a
        # variable set on an earlier, separate line, not by the split.
        var_map, output = var_map_for("A2 = 3.2;", "A4 = A2; A6 = A2;")
        self.assertEqual(var_map["A4"], 3.2)
        self.assertEqual(var_map["A6"], 3.2)
        self.assertEqual(output, "")

    def test_one_failing_statement_does_not_block_its_neighbors_on_the_same_line(self):
        # Before this fix, one bad statement failed the whole line - now
        # each split statement is independent.
        var_map, output = var_map_for("good1 = 1; bad = some_typo_var; good2 = 2;")
        self.assertEqual(var_map["good1"], 1)
        self.assertEqual(var_map["good2"], 2)
        self.assertNotIn("bad", var_map)
        self.assertIn("Warning: Failed to evaluate bad = some_typo_var;", output)


class ConditionalAssignmentTest(unittest.TestCase):
    def test_assignment_followed_by_conditional_reassignment(self):
        # The exact corpus pattern: 'Emin = EI-20; if (Emin < EI/3) Emin=EI/3;'
        var_map, output = var_map_for(
            "EI = 5.0;", "Emin = EI-20; if (Emin < EI/3) Emin=EI/3;"
        )
        self.assertAlmostEqual(var_map["Emin"], 5.0 / 3)
        self.assertEqual(output, "")

    def test_if_true_branch_taken(self):
        var_map, output = var_map_for("x = 1;", "if (x > 0) y = 10;")
        self.assertEqual(var_map["y"], 10)
        self.assertEqual(output, "")

    def test_if_false_with_no_else_leaves_target_unresolved_and_prints_nothing(self):
        # Real C semantics: an untaken branch is never executed, so it must
        # not be evaluated or warned about even if its expression would
        # have failed.
        var_map, output = var_map_for("x = 1;", "if (x < 0) never = some_typo_var;")
        self.assertNotIn("never", var_map)
        self.assertEqual(output, "")

    def test_if_else_true_branch(self):
        var_map, output = var_map_for("x = 1;", "if (x > 0) y = 10; else y = -10;")
        self.assertEqual(var_map["y"], 10)
        self.assertEqual(output, "")

    def test_if_else_false_branch(self):
        var_map, output = var_map_for("x = -1;", "if (x > 0) y = 10; else y = -10;")
        self.assertEqual(var_map["y"], -10)
        self.assertEqual(output, "")

    def test_untaken_branch_failure_is_never_evaluated_or_warned_about(self):
        # Only the taken branch (else, here) is attempted - the failing
        # true-branch expression must be silently skipped, not warned about.
        var_map, output = var_map_for(
            "x = -1;", "if (x > 0) y = some_typo_var; else y = -10;"
        )
        self.assertEqual(var_map["y"], -10)
        self.assertEqual(output, "")

    def test_condition_with_its_own_nested_parens(self):
        # Regression guard for the naive "if\\s*\\((.+)\\)" trap: a naive
        # regex would stop at the FIRST ')', truncating the condition to
        # "a<b" and leaving ") && c) x=1" as unparseable garbage.
        var_map, output = var_map_for(
            "a = 1;", "b = 2;", "c = 1;",
            "if ((a < b) && c) x = 42;",
        )
        self.assertEqual(var_map["x"], 42)
        self.assertEqual(output, "")

    def test_c_and_operator(self):
        var_map, output = var_map_for(
            "a = 1;", "b = 1;", "if (a > 0 && b > 0) both = 1;"
        )
        self.assertEqual(var_map["both"], 1)

    def test_c_or_operator(self):
        var_map, output = var_map_for(
            "a = -1;", "b = 1;", "if (a > 0 || b > 0) either = 1;"
        )
        self.assertEqual(var_map["either"], 1)

    def test_c_negation_operator(self):
        var_map, output = var_map_for("flag = 0;", "if (!flag) q = 1;")
        self.assertEqual(var_map["q"], 1)
        self.assertEqual(output, "")

    def test_negation_at_the_very_start_of_the_condition_does_not_crash(self):
        # Regression guard: "!flag" naively substitutes to " not flag"
        # (leading space), which ast.parse(mode="eval") rejects as an
        # IndentationError unless the result is stripped before parsing.
        var_map, output = var_map_for("flag = 1;", "if (!flag) q = 1;")
        self.assertNotIn("q", var_map)
        self.assertEqual(output, "")

    def test_chained_comparison(self):
        var_map, output = var_map_for("x = 5;", "if (0 < x < 10) in_range = 1;")
        self.assertEqual(var_map["in_range"], 1)

    def test_bang_equal_is_left_untouched_by_the_negation_substitution(self):
        # "!=" must not become " not =" - _NEGATION_RE's lookahead guards
        # against that.
        var_map, output = var_map_for("x = 1;", "if (x != 0) nonzero = 1;")
        self.assertEqual(var_map["nonzero"], 1)

    def test_condition_failure_prints_a_warning_and_leaves_both_branches_unresolved(self):
        var_map, output = var_map_for(
            "if (some_typo_var > 0) y = 1; else y = 2;"
        )
        self.assertNotIn("y", var_map)
        self.assertIn("Warning: Failed to evaluate if (some_typo_var > 0) y = 1; else y = 2;", output)


class DemonstrationFixtureTest(unittest.TestCase):
    """End-to-end regression test tied to
    tests/multi_statement_lines_test.instr - the demonstration fixture for
    this fix. box_angle = Emin * A4 there only comes out right if every
    multi-statement/conditional INITIALIZE line above it resolved
    correctly; before this fix, DECLARE's zero-init silently masked the
    failure (Emin/A4 both read back as 0.0), so box_angle silently
    computed to 0.0 with no warning at all rather than the correct
    ~11.67 - a wrong-but-plausible-looking result, not just a noisy one."""

    def test_box_angle_resolves_correctly_and_reaches_the_rotated_vector(self):
        instr_path = os.path.join(REPO_ROOT, "tests", "multi_statement_lines_test.instr")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            instr, world, _ = pp.preprocess(instr_path, verbose=False)
        self.assertEqual(buf.getvalue(), "")

        box = next(c for c in instr.component_list if c.name == "box")
        self.assertAlmostEqual(box.ROTATED_data[1], 5.0 / 3 * 7.0, places=6)
        self.assertNotAlmostEqual(box.ROTATED_data[1], 0.0)


if __name__ == "__main__":
    unittest.main()
