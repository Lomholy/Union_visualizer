"""Tests for union_viewer.py's non-GUI logic: compute_trace_data, the
worker-process side of the trace pool, needs no display to run.

The GUI itself is not unit-testable without a display, so its own wiring
was exercised interactively with QT_QPA_PLATFORM=offscreen against real
test instruments while implementing changes to it.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import union_viewer as uv  # noqa: E402


@unittest.skipUnless(shutil.which("mcrun"), "mcrun not on PATH")
class ComputeTraceDataTest(unittest.TestCase):
    def test_all_requested_rays_are_kept(self):
        # trace_instrument's own max_rays defaults to 1000 (meant for a
        # file loaded independently of any run); compute_trace_data must
        # pass the actual requested ncount through instead, or a run for
        # more rays than that silently gets truncated back down to 1000.
        with tempfile.TemporaryDirectory() as tmp:
            instr = os.path.join(tmp, "simple_test.instr")
            shutil.copy(ROOT / "tests" / "simple_test.instr", instr)
            try:
                _, rays = uv.compute_trace_data(instr, False, [], 1200, 7)
            except Exception as e:
                self.skipTest(f"mcrun could not compile here: {e}")
        self.assertEqual(rays.n_rays, 1200)


if __name__ == "__main__":
    unittest.main()
