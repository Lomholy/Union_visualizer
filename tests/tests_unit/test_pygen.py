"""preprocess(force_pygen=True): the mcstas-pygen path loads an instrument."""

import contextlib
import io
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import preprocess  # noqa: E402


@unittest.skipUnless(shutil.which("mcstas-pygen"), "mcstas-pygen not on PATH")
class PygenTest(unittest.TestCase):
    def test_pygen_matches_the_quick_reader(self):
        inp = str(ROOT / "tests" / "test_instruments" / "cryostat_test.instr")
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, quick = preprocess(inp, False)
            _, _, pygen = preprocess(inp, False, force_pygen=True)
        self.assertEqual(sorted(pygen), sorted(quick))


if __name__ == "__main__":
    unittest.main()
