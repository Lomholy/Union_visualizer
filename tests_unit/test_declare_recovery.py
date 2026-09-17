# Regression test for src/preprocess.py's handling of mcstasscript's
# raw-string fallback entries in declare_list/user_var_list.
#
# Not wired into CI's tests/* loop (that loop feeds every file in tests/ to
# mcstas_to_cad.py as an instrument input, so a plain unittest module can't
# live there). Run directly:
#
#   python tests_unit/test_declare_recovery.py

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
sys.path.insert(0, SRC_DIR)

import preprocess as pp  # noqa: E402


class DeclareRecoveryTest(unittest.TestCase):
    def test_pragma_line_is_left_unresolved(self):
        # A #pragma has no "name = expr" shape - nothing to recover, and no
        # warning should be printed for it.
        var_map = {}
        pp._recover_raw_declare_block(
            ["  #pragma acc declare create(smi,filt,tof)"], var_map
        )
        self.assertEqual(var_map, {})

    def test_misclassified_multiline_initializer_is_recovered(self):
        # mcstasscript's "is this a function?" heuristic dumps
        # "double result = fabs(\n  -5.0\n);" as three raw-string entries
        # (one per physical source line). Rejoining them should recover the
        # real value instead of losing it silently.
        var_map = {}
        pp._recover_raw_declare_block(
            ["double result = fabs(", "  -5.0", ");"], var_map
        )
        self.assertEqual(var_map, {"result": 5.0})

    def test_function_body_internals_are_not_promoted_to_var_map(self):
        # A genuine multi-line function is also dumped as raw strings, and
        # can contain an assignment-shaped internal statement. That's
        # function-local C, not an instrument-global DECLARE variable, and
        # the brace pair is what must stop it from being recovered.
        var_map = {}
        pp._recover_raw_declare_block(
            [
                "double add_two(double a, double b) {",
                "  double total;",
                "  total = a + b;",
                "  return total;",
                "}",
            ],
            var_map,
        )
        self.assertEqual(var_map, {})

    def test_populate_declare_vars_handles_mixed_entries_in_order(self):
        class FakeDeclareVar:
            def __init__(self, name, value, type_="double"):
                self.name = name
                self.value = value
                self.type = type_

        entries = [
            FakeDeclareVar("angle", 30.0),
            "double result = fabs(",
            "  -5.0",
            ");",
            FakeDeclareVar("gap", ""),  # declared but never initialized
        ]
        var_map = {}
        pp._populate_declare_vars(entries, var_map)
        self.assertEqual(var_map["angle"], 30.0)
        self.assertEqual(var_map["result"], 5.0)
        self.assertEqual(var_map["gap"], 0.0)

    def test_string_typed_parameter_entry_is_skipped_not_recovered(self):
        # instr.parameters entries are never raw strings in practice, but
        # create_var_map must not route one through the DECLARE-recovery
        # machinery above if it ever were - that machinery is for a
        # different source list and could misparse a parameter's raw text.
        class FakeInstr:
            declare_list = []
            user_var_list = []
            parameters = ["not_a_real_parameter_object"]
            initialize_section = ""

        var_map = pp.create_var_map(FakeInstr())
        self.assertEqual(var_map, {})

    def test_real_parameter_objects_are_never_dropped(self):
        # Guards against reintroducing the trap of filtering by
        # "isinstance(v, DeclareVariable)" as an allowlist, which would
        # silently discard every instrument parameter (a different,
        # unrelated object type) along with the raw strings.
        class FakeParameter:
            def __init__(self, name, value, type_="double"):
                self.name = name
                self.value = value
                self.type = type_

        class FakeInstr:
            declare_list = []
            user_var_list = []
            parameters = [FakeParameter("wavelength", 2.5)]
            initialize_section = ""

        var_map = pp.create_var_map(FakeInstr())
        self.assertEqual(var_map["wavelength"], 2.5)


if __name__ == "__main__":
    unittest.main()
