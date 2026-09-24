"""Load Union logger/abs_logger detector-counts output (via mcstasscript)
and turn it into world-placed, texture-ready heatmap planes for the viewer.

Only loggers whose data has a meaningful position in 3D space can be shown
this way: the *_space loggers (Union_logger_2D_space/3D_space,
Union_abs_logger_1D_space/2D_space), whose files are already a 2D histogram
along two of the logger's own local x/y/z axes, and event-list loggers
(Union_abs_logger_event, Union_abs_logger_nD in list mode) whose recorded
x/y/z columns we histogram ourselves - McStas records those in the same
local (AT/ROTATED) frame as the binned loggers (confirmed against
Union_abs_logger_event.comp's record_to_perm: `coords_sub` then
`rot_apply`). Non-spatial loggers (time, Q, kf) have no 3D placement and are
skipped.

Placement reuses this codebase's own preprocess.compute_world_matrices():
every *_space logger's histogram is a rectangle in its own local frame, so
the same world matrix that places a Union volume places a logger's plane.
"""

import numpy as np
import mcstasscript as ms

from gui_helpers import colormap

SPATIAL_AXES = ("x", "y", "z")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Bin count used to histogram an event-list logger ourselves (it has no
# built-in binning to fall back on).
DEFAULT_EVENT_BINS = 200


class LoggerCounts:
    """One logger's counts, ready to place and render.

    axis1/axis2 are which of the logger's own local x/y/z axes grid's rows
    and columns run along: grid[i, j] is the bin at axis1 index i, axis2
    index j. limits = (min1, max1, min2, max2) along axis1/axis2, in the
    logger's own local (AT/ROTATED) frame.
    """

    def __init__(self, name, kind, axis1, axis2, limits, grid):
        self.name = name
        self.kind = kind  # "logger" or "abs_logger" - for UI grouping/labels
        self.axis1 = axis1
        self.axis2 = axis2
        self.limits = limits
        self.grid = grid

    @property
    def total(self):
        return float(self.grid.sum())


def _binned_axes(dataset):
    """(axis1, axis2) for a binned 2D dataset whose xvar/yvar are both
    spatial, or None (e.g. a Union_logger_2DQ/2D_kf's reciprocal-space axes,
    or Union_logger_1D's time/q)."""
    info = dataset.metadata.info
    xvar = info.get("xvar", "").strip()
    yvar = info.get("yvar", "").strip()
    if xvar in SPATIAL_AXES and yvar in SPATIAL_AXES:
        return xvar, yvar
    return None


def _event_axes(dataset):
    """Two spatial axes to histogram an event-list logger against, or None
    if it doesn't record at least two of x/y/z. Prefers the horizontal
    plane (x, y), then (x, z), else whichever two spatial axes are present."""
    present = [axis for axis in SPATIAL_AXES if axis in dataset.variables]
    if len(present) < 2:
        return None
    if "x" in present and "y" in present:
        return "x", "y"
    if "x" in present and "z" in present:
        return "x", "z"
    return present[0], present[1]


def _histogram_events(dataset, axis1, axis2, bins):
    a = dataset.get_data_column(axis1)
    b = dataset.get_data_column(axis2)
    weight = dataset.get_data_column("p")
    grid, edges1, edges2 = np.histogram2d(a, b, bins=bins, weights=weight)
    return grid, (edges1[0], edges1[-1], edges2[0], edges2[-1])


def spatial_grid(dataset, event_bins=DEFAULT_EVENT_BINS):
    """(axis1, axis2, limits, grid) for an mcstasscript McStasData object
    that can be spatially histogrammed in the logger's own local frame, or
    None if it can't (e.g. a time/Q logger, or event data with under 2
    spatial columns)."""
    if dataset.data_type == "Events":
        axes = _event_axes(dataset)
        if axes is None:
            return None
        axis1, axis2 = axes
        grid, limits = _histogram_events(dataset, axis1, axis2, event_bins)
        return axis1, axis2, limits, grid

    axes = _binned_axes(dataset)
    if axes is None or np.ndim(dataset.Intensity) != 2:
        return None
    axis1, axis2 = axes
    limits = tuple(dataset.metadata.limits)
    return axis1, axis2, limits, np.asarray(dataset.Intensity, dtype=float)


def _logger_kind(component_type):
    return "abs_logger" if "abs_logger" in component_type.lower() else "logger"


def load_counts(run_folder, component_types, event_bins=DEFAULT_EVENT_BINS):
    """Every Union logger/abs_logger in run_folder that can be spatially
    histogrammed, as a {component name: LoggerCounts} dict.

    component_types: {component name: McStas component type}, e.g. from
    mcstas_trace.component_types(input_file) for the instrument this run
    came from - used to tell loggers apart from ordinary monitors also
    present in the folder, and to label each as "logger" vs "abs_logger".
    """
    datasets = ms.load_data(run_folder)
    counts = {}
    for dataset in datasets:
        name = dataset.metadata.component_name
        comp_type = component_types.get(name, "")
        if "logger" not in comp_type.lower():
            continue
        grid_info = spatial_grid(dataset, event_bins=event_bins)
        if grid_info is None:
            continue
        axis1, axis2, limits, grid = grid_info
        counts[name] = LoggerCounts(name, _logger_kind(comp_type), axis1, axis2, limits, grid)
    return counts


def local_corners(axis1, axis2, limits):
    """The 4 corner points (winding order, a flat quad) of a logger's
    histogrammed rectangle, in the logger's own local (AT/ROTATED) frame -
    the third, unlisted axis is 0."""
    min1, max1, min2, max2 = limits
    i1, i2 = _AXIS_INDEX[axis1], _AXIS_INDEX[axis2]
    corners = np.zeros((4, 3))
    for row, (v1, v2) in enumerate(((min1, min2), (max1, min2), (max1, max2), (min1, max2))):
        corners[row, i1] = v1
        corners[row, i2] = v2
    return corners


def world_corners(local_points, world_matrix):
    """local_points (n, 3) transformed by a 4x4 world matrix."""
    homogeneous = np.hstack([local_points, np.ones((len(local_points), 1))])
    return (homogeneous @ world_matrix.T)[:, :3]


def texture_image(grid, vmin=None, vmax=None):
    """(h, w, 4) uint8 RGBA viridis image for a 2D counts grid, scaled to
    [vmin, vmax] (grid's own min/max when not given) - the same colour scale
    gui_helpers.colormap() uses for rays, so a logger's plane and the ray
    colourbar read consistently if ever shown together."""
    flat_rgba = colormap(grid.ravel(), vmin, vmax)
    return (flat_rgba.reshape(grid.shape + (4,)) * 255).astype(np.uint8)
