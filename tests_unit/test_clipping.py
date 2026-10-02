"""Tests for clipping.py: the viewer's clip plane, optionally taken in a
component's coordinate system, as one world-space plane shared by the
meshers, the pygfx view and STL export."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from clipping import clip_plane, resolve_clip_frame  # noqa: E402
from signed_distance_functions import sdf_plane  # noqa: E402


def clip(axis="Z", mode="Above", position=0.0, frame=None):
    return {"enable": True, "axis": axis, "mode": mode, "position": position, "frame": frame}


class ClipPlaneTest(unittest.TestCase):
    def test_world_frame_matches_old_axis_behaviour(self):
        normal, point = clip_plane(resolve_clip_frame(clip("Y", "Above", 2.0), {}))
        np.testing.assert_allclose(normal, (0, 1, 0))
        np.testing.assert_allclose(point, (0, 2, 0))
        normal, _ = clip_plane(clip("Y", "Below", 2.0))
        np.testing.assert_allclose(normal, (0, -1, 0))

    def test_component_frame(self):
        frame = np.eye(4)
        frame[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]  # local z is world -x
        frame[:3, 3] = (1, 2, 3)
        resolved = resolve_clip_frame(clip("Z", "Above", 0.5, "sample"), {"sample": frame})
        normal, point = clip_plane(resolved)
        np.testing.assert_allclose(normal, (-1, 0, 0), atol=1e-12)
        np.testing.assert_allclose(point, (0.5, 2, 3), atol=1e-12)

    def test_unknown_frame_falls_back_to_world(self):
        resolved = resolve_clip_frame(clip(frame="gone"), {})
        np.testing.assert_allclose(resolved["matrix"], np.eye(4))

    def test_sdf_plane_is_negative_on_kept_side(self):
        normal, point = clip_plane(clip("X", "Above", 1.0))
        f = sdf_plane(normal, point)
        values = f(np.array([[2.0, 0, 0, 1], [0.0, 0, 0, 1]]))
        self.assertLess(values[0], 0)
        self.assertGreater(values[1], 0)


if __name__ == "__main__":
    unittest.main()
