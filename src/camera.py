"""Camera placement and clipping range for the viewer."""

import numpy as np


# Near clipping distance in metres: as close to 0 as a perspective projection
# allows (exactly 0 divides by zero in its depth maths), so even a small
# detector can be viewed up close without being clipped away.
CAMERA_NEAR = 1e-2
CAMERA_FAR = 1e4


def update_camera_depth_range(camera, target):
    """Set camera's explicit near/far clipping range from its distance to
    target. Without an explicit range pygfx derives the planes from the
    camera's fov and depth, which clips whatever is close to the camera."""
    distance = np.linalg.norm(np.asarray(camera.local.position) - np.asarray(target))
    near = max(CAMERA_NEAR, distance * 1e-2)
    if camera.depth_range != (near, CAMERA_FAR):
        camera.depth_range = (near, CAMERA_FAR)


def fit_camera_to_scene(camera, controller, scene, scale=2.0):
    bbox = scene.get_world_bounding_box()

    if bbox is None:
        return
    bmin = bbox[0]
    bmax = bbox[1]
    center = (bmin + bmax) / 2.0
    extent = bmax - bmin
    radius = np.linalg.norm(extent) * 0.5
    direction = np.array([1.0, 1.0, 0.7])
    direction /= np.linalg.norm(direction)
    distance = radius * scale
    position = center + direction * distance
    camera.local.position = position
    camera.look_at(center)
    controller.target = center
    update_camera_depth_range(camera, controller.target)


def recentre_controller(controller, group):
    """Move the orbit pivot to the group's centre, without touching the
    camera's position or zoom."""
    if group is None:
        return
    bbox = group.get_world_bounding_box()
    if bbox is None:
        return
    controller.target = (bbox[0] + bbox[1]) / 2.0
