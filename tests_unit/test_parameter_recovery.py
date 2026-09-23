# Regression test for src/preprocess.py's handling of a raw-string entry
# in instr.parameters.
#
# Every case we've reproduced so far (mcstasscript's DEFINE INSTRUMENT(...)
# reader always constructs a proper Parameter object, never a bare string -
# see _recover_raw_parameter's docstring) says this can't happen. This
# suite exists anyway, because relying on "it never happens" instead of
# handling it is exactly the mistake the DECLARE raw-string fix was written
# to avoid. Run directly:
#
#   python tests_unit/test_parameter_recovery.py

import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(TESTS_DIR), "src")
sys.path.insert(0, SRC_DIR)

import preprocess as pp  # noqa: E402


class ParameterRecoveryTest(unittest.TestCase):
    def test_int_parameter_default_is_recovered(self):
        var_map = {}
        pp._recover_raw_parameter("int height=3", var_map)
        self.assertEqual(var_map, {"height": 3})

    def test_double_parameter_default_is_recovered(self):
        var_map = {}
        pp._recover_raw_parameter("double angle=1.5", var_map)
        self.assertEqual(var_map, {"angle": 1.5})

    def test_string_parameter_default_has_quotes_stripped(self):
        # Unlike mcstasscript's own object path, which leaves the quotes
        # embedded in the value (verified against a real Parameter object:
        # value == '"myfile.txt"'), the recovered value should be the
        # literal filename with no quote characters.
        var_map = {}
        pp._recover_raw_parameter('string filename="myfile.txt"', var_map)
        self.assertEqual(var_map, {"filename": "myfile.txt"})

    def test_default_referencing_a_builtin_constant_is_evaluated(self):
        var_map = {}
        pp._recover_raw_parameter("double full_turn=2*PI", var_map)
        self.assertAlmostEqual(var_map["full_turn"], 2 * 3.14159265358979)

    def test_bare_parameter_without_default_is_left_unresolved(self):
        # A required parameter with no default has no computable value at
        # preprocessing time - expected, not a parser gap, so nothing
        # should be added and nothing should raise.
        var_map = {}
        pp._recover_raw_parameter("height", var_map)
        self.assertEqual(var_map, {})

    def test_create_var_map_recovers_a_raw_string_parameter_entry(self):
        class FakeInstr:
            declare_list = []
            user_var_list = []
            parameters = ["int height=3", 'string filename="myfile.txt"']
            initialize_section = ""

        var_map = pp.create_var_map(FakeInstr())
        self.assertEqual(var_map["height"], 3)
        self.assertEqual(var_map["filename"], "myfile.txt")


    def test_create_var_map_uses_given_parameter_values(self):
        class Param:
            def __init__(self, name, type, value):
                self.name, self.type, self.value = name, type, value

        class FakeInstr:
            declare_list = []
            user_var_list = []
            parameters = [
                Param("width", "double", 0.1),
                Param("depth", "double", 0.2),
                Param("filename", "string", '"a.dat"'),
            ]
            initialize_section = "area = width * depth;"

        var_map = pp.create_var_map(
            FakeInstr(),
            {"width": "2*0.25", "depth": " ", "filename": "b.dat", "unknown": "1"},
        )
        self.assertEqual(var_map["width"], 0.5)
        self.assertEqual(var_map["depth"], 0.2)
        self.assertEqual(var_map["filename"], "b.dat")
        self.assertNotIn("unknown", var_map)
        # INITIALIZE runs after the override, so derived values follow it.
        self.assertAlmostEqual(var_map["area"], 0.1)

    def test_instrument_parameters_lists_defaults(self):
        class Param:
            def __init__(self, name, type, value):
                self.name, self.type, self.value = name, type, value

        class FakeInstr:
            parameters = [Param("a", "double", 1.5), Param("b", "int", None), "raw x=1"]

        self.assertEqual(
            pp.instrument_parameters(FakeInstr()),
            [("a", "double", "1.5"), ("b", "int", None)],
        )


if __name__ == "__main__":
    unittest.main()
