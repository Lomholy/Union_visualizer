"""The brep mesher and Union masks: a mask only limits the volume it masks
(McStas never lets a mask hide anything), and a masked higher-priority
component only cuts where its masks let it exist."""

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import brep  # noqa: E402

NO_CLIP = {"enable": False, "axis": "X", "mode": "Above", "position": 0.0}


class FakeBox:
    component_name = "Union_box"

    def __init__(self, name, w, h, d, priority=1, mask_string=None, mask_setting=None):
        self.name, self.priority = name, priority
        self.xwidth, self.yheight, self.zdepth = w, h, d
        self.mask_string, self.mask_setting = mask_string, mask_setting
        self.material_string = "inc"


def translation(z=0.0):
    m = np.eye(4)
    m[2, 3] = z
    return m


def area(comps, name, matrices=None):
    geoms = {c.name: c for c in comps}
    world = {n: (matrices or {}).get(n, np.eye(4)) for n in geoms}
    with contextlib.redirect_stdout(io.StringIO()):
        mesh = brep.build_single_brep_mesh(geoms[name], geoms, world, NO_CLIP, False)
    return 0.0 if mesh is None else float(mesh.area)


class TestBrepMasks(unittest.TestCase):
    def test_a_mask_does_not_cut_other_components(self):
        # M masks a component elsewhere; its priority is above A's, but it must not hide A
        a = FakeBox("a", 2, 2, 2, priority=1)
        m = FakeBox("m", 3, 3, 3, priority=100, mask_string="elsewhere")
        self.assertAlmostEqual(area([a, m], "a"), 24.0, places=4)

    def test_a_masked_cutter_only_cuts_inside_its_mask(self):
        # B (1 x 1 x 4) runs through A along z, but its mask M keeps only z > 0.
        # A loses B's upper half: top hole 1, four cavity walls of 1 and a floor of 1
        a = FakeBox("a", 2, 2, 2, priority=1)
        b = FakeBox("b", 1, 1, 4, priority=2)
        m = FakeBox("m", 3, 3, 3, priority=100, mask_string="b", mask_setting="All")
        self.assertAlmostEqual(area([a, b, m], "a", {"m": translation(z=1.5)}), 24 - 1 + 4 + 1, places=4)


if __name__ == "__main__":
    unittest.main()
