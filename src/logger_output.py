"""Load spatially-placeable detector output (via mcstasscript) and turn it
into world-placed, render-ready counts for the viewer: a 2D histogram plane,
or a 1D histogram line.

Any component whose output has a position in 3D space qualifies - Union
loggers/abs_loggers (Union_logger_2D_space/3D_space,
Union_abs_logger_1D_space/2D_space) as well as ordinary McStas detector
components (PSD_monitor, Monitor_nD, ...): whatever the component, its file
is already a 1D or 2D histogram along one or two of its own local x/y/z axes
(from the file's xvar/yvar) - used directly. Event-list loggers
(Union_abs_logger_event, Union_abs_logger_nD in list mode) record x/y/z
columns in that same local frame (confirmed against
Union_abs_logger_event.comp's record_to_perm: `coords_sub` then
`rot_apply`), so they're histogrammed into an equivalent 2D grid ourselves.
Non-spatial output (time, wavelength, Q, kf axes) has no 3D placement and is
skipped.

Placement reuses this codebase's own preprocess.compute_world_matrices():
every detector's histogram is a point/line/rectangle in its own local frame,
so the same world matrix that places a Union volume places a detector's
line or plane - a detector is just another AT/ROTATED component.
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
    """One detector's counts, ready to place and render.

    2D (axis2 is not None): grid[i, j] is the bin at axis1 index i, axis2
    index j; limits = (min1, max1, min2, max2) - rendered as a textured
    plane. 1D (axis2 is None): grid[i] is the bin at axis1 index i; limits
    = (min1, max1) - rendered as a coloured line, since there's no second
    axis to give it any width. axis1/axis2 are always local (AT/ROTATED)
    x/y/z axes.
    """

    def __init__(self, name, kind, axis1, axis2, limits, grid):
        self.name = name
        self.kind = kind  # the detector's own McStas component type - for UI labels
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
    or an energy/wavelength monitor's non-spatial axes)."""
    info = dataset.metadata.info
    xvar = info.get("xvar", "").strip()
    yvar = info.get("yvar", "").strip()
    if xvar in SPATIAL_AXES and yvar in SPATIAL_AXES:
        return xvar, yvar
    return None


def _binned_1d_axis(dataset):
    """axis1 for a binned 1D dataset whose xvar is spatial, or None (e.g. a
    time-of-flight or wavelength monitor)."""
    xvar = dataset.metadata.info.get("xvar", "").strip()
    return xvar if xvar in SPATIAL_AXES else None


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
    that can be spatially histogrammed in its own local frame (axis2 is None
    for a 1D result - see LoggerCounts), or None if it can't (e.g. a
    time/wavelength/Q monitor, or event data with under 2 spatial columns)."""
    if dataset.data_type == "Events":
        axes = _event_axes(dataset)
        if axes is None:
            return None
        axis1, axis2 = axes
        grid, limits = _histogram_events(dataset, axis1, axis2, event_bins)
        return axis1, axis2, limits, grid

    if np.ndim(dataset.Intensity) == 2:
        axes = _binned_axes(dataset)
        if axes is None:
            return None
        axis1, axis2 = axes
        limits = tuple(dataset.metadata.limits)
        return axis1, axis2, limits, np.asarray(dataset.Intensity, dtype=float)

    if np.ndim(dataset.Intensity) == 1:
        axis1 = _binned_1d_axis(dataset)
        if axis1 is None:
            return None
        limits = tuple(dataset.metadata.limits)
        return axis1, None, limits, np.asarray(dataset.Intensity, dtype=float)

    return None


def load_counts(run_folder, component_types, event_bins=DEFAULT_EVENT_BINS):
    """Every component in run_folder whose output can be spatially placed
    (a 1D or 2D position histogram, or an event list with x/y/z columns),
    as a {component name: LoggerCounts} dict - Union loggers/abs_loggers as
    well as ordinary McStas detector components (PSD_monitor, Monitor_nD,
    ...). Non-spatial output is skipped automatically by spatial_grid(), so
    nothing here needs to know which components are "detectors" up front.

    component_types: {component name: McStas component type}, e.g. from
    mcstas_trace.component_types(input_file) for the instrument this run
    came from - used only to label each entry's `kind` for the UI.
    """
    datasets = ms.load_data(run_folder)
    counts = {}
    for dataset in datasets:
        name = dataset.metadata.component_name
        grid_info = spatial_grid(dataset, event_bins=event_bins)
        if grid_info is None:
            continue
        axis1, axis2, limits, grid = grid_info
        counts[name] = LoggerCounts(name, component_types.get(name, ""), axis1, axis2, limits, grid)
    return counts


def local_corners(axis1, axis2, limits):
    """The 4 corner points (winding order, a flat quad) of a 2D detector's
    histogrammed rectangle, in its own local (AT/ROTATED) frame - the
    third, unlisted axis is 0."""
    min1, max1, min2, max2 = limits
    i1, i2 = _AXIS_INDEX[axis1], _AXIS_INDEX[axis2]
    corners = np.zeros((4, 3))
    for row, (v1, v2) in enumerate(((min1, min2), (max1, min2), (max1, max2), (min1, max2))):
        corners[row, i1] = v1
        corners[row, i2] = v2
    return corners


def local_line_points(axis1, limits, n_bins):
    """The n_bins+1 bin-edge points of a 1D detector's histogram along
    axis1, in its own local (AT/ROTATED) frame - the other two axes are 0."""
    min1, max1 = limits
    edges = np.linspace(min1, max1, n_bins + 1)
    points = np.zeros((n_bins + 1, 3))
    points[:, _AXIS_INDEX[axis1]] = edges
    return points


def world_points(local_points, world_matrix):
    """local_points (n, 3) - quad corners, line edges, or any other local
    points - transformed by a 4x4 world matrix."""
    homogeneous = np.hstack([local_points, np.ones((len(local_points), 1))])
    return (homogeneous @ world_matrix.T)[:, :3]


def texture_image(grid, vmin=None, vmax=None):
    """(h, w, 4) uint8 RGBA viridis image for a 2D counts grid, scaled to
    [vmin, vmax] (grid's own min/max when not given) - the same colour scale
    gui_helpers.colormap() uses for rays, so a detector's plane and the ray
    colourbar read consistently if ever shown together."""
    flat_rgba = colormap(grid.ravel(), vmin, vmax)
    return (flat_rgba.reshape(grid.shape + (4,)) * 255).astype(np.uint8)


def line_vertex_colors(grid, vmin=None, vmax=None):
    """(2n, 4) float32 RGBA viridis colours for a 1D counts grid of n bins,
    scaled to [vmin, vmax] - one pair of identical colours per bin's two
    segment endpoints, ready for a pygfx LineSegmentMaterial(color_mode=
    "vertex")."""
    per_bin = colormap(grid, vmin, vmax)
    return np.repeat(per_bin, 2, axis=0)
