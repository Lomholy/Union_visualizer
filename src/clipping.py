"""The viewer's clipping plane: an axis-aligned cut in the coordinate system
of any component (or the world), resolved into one world-space plane that
the meshers, the pygfx view and STL export all share."""

import numpy as np


def resolve_clip_frame(clip, world_matrices):
    """A copy of clip with "matrix" set to the world matrix of the component
    named by clip["frame"] (identity for the world frame)."""
    clip = dict(clip)
    frame = clip.get("frame")
    matrix = world_matrices.get(frame) if frame else None
    if frame and matrix is None:
        print(f"Warning: clip coordinate system '{frame}' not found; using world axes.")
    clip["matrix"] = (np.eye(4) if matrix is None else np.asarray(matrix, dtype=float)).tolist()
    return clip


def clip_plane(clip):
    """(unit normal, point) in world space. The kept side is
    normal . (p - point) >= 0: "Above" keeps positive coordinates along the
    frame's clip axis, measured from position along that axis."""
    matrix = np.asarray(clip.get("matrix", np.eye(4)), dtype=float)
    axis = "XYZ".index(clip["axis"].upper())
    normal = matrix[:3, axis] / np.linalg.norm(matrix[:3, axis])
    point = matrix[:3, 3] + float(clip["position"]) * normal
    if clip["mode"] != "Above":
        normal = -normal
    return normal, point
