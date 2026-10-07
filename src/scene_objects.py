"""pygfx objects for the viewer's scene: Union meshes, McStas
components, neutron rays, detector counts, coordinate axes and the
floor grid sizing."""

import numpy as np
import pygfx as gfx
from gui_helpers import (
    assign_default_color,
    component_color_key,
    ray_segment_indices,
    ray_color_values,
    colormap,
    hex_to_rgba,
    UNIFORM_RAY_COLOR,
    SCATTER_MARKER_COLOR,
    ABSORB_MARKER_COLOR,
    TELEPORT_COLOR,
    SCATTER_MARKER_SIZE,
    ABSORB_MARKER_SIZE,
)
from mcstas_trace import SCATTER, ABSORB, TELEPORT
import logger_output


def build_gfx_group(render_meshes, geometry_is_vacuum, colors):
    """Wrap compute_mesh_data()'s plain trimesh output into pygfx objects.
    Cheap (no geometry kernel calls) - safe to run on the GUI thread or a
    plain QThread."""
    print("Defining group")
    group = gfx.Group()
    group.geometry_meshes = {}
    group.geometry_is_vacuum = geometry_is_vacuum
    group.geometry_trimeshes = render_meshes

    for key, mesh in render_meshes.items():
        assign_default_color(colors, key)

        gfx_mesh = gfx.Mesh(
            gfx.geometry_from_trimesh(mesh),
            gfx.MeshStandardMaterial(
                color=colors[key],
                metalness=0,
                roughness=0.8,
            ),
        )

        group.add(gfx_mesh)
        group.geometry_meshes[key] = gfx_mesh

    return group


def build_component_group(components, colors):
    """pygfx objects for every McStas component: one Line for all of its
    MCDISPLAY lines and one translucent Mesh for all of its solids."""
    group = gfx.Group()
    group.components = components
    group.component_objects = {}
    group.component_is_arm = {}
    for name, comp in components.items():
        color = assign_default_color(colors, component_color_key(name))
        comp_group = gfx.Group()
        if len(comp.segments):
            comp_group.add(
                gfx.Line(
                    gfx.Geometry(positions=comp.segments.reshape(-1, 3).astype(np.float32)),
                    gfx.LineSegmentMaterial(thickness=1.5, color=color),
                )
            )
        if comp.solid is not None:
            comp_group.add(
                gfx.Mesh(
                    gfx.geometry_from_trimesh(comp.solid),
                    gfx.MeshStandardMaterial(
                        color=color,
                        opacity=0.5,
                        alpha_mode="blend",
                        side="both",
                        roughness=0.8,
                    ),
                )
            )
        group.add(comp_group)
        group.component_objects[name] = comp_group
        group.component_is_arm[name] = comp.is_arm
    return group


DEFAULT_RAY_COLORS = {
    "ray": UNIFORM_RAY_COLOR,
    "teleport": TELEPORT_COLOR,
    "scatter": SCATTER_MARKER_COLOR,
    "absorb": ABSORB_MARKER_COLOR,
}
DEFAULT_MARKER_SIZES = {
    "scatter": SCATTER_MARKER_SIZE,
    "absorb": ABSORB_MARKER_SIZE,
}


def build_ray_group(rays, ray_indices, color_mode, colors=None, marker_sizes=None):
    """Two separate Lines for the chosen rays, so the ordinary path and the
    restore_neutron jumps can be shown/hidden independently:
    group.ray_line for ordinary segments (vertex-coloured unless color_mode
    is "Uniform") and group.teleport_line for a segment whose end point is
    a TELEPORT. Plus Points marking scatterings and absorptions.
    colors maps "ray", "teleport", "scatter" and "absorb" to a hex colour,
    marker_sizes maps "scatter" and "absorb" to a point size; missing keys
    fall back to the gui_helpers defaults. Returns
    (group, (vmin, vmax) or None)."""
    colors = {**DEFAULT_RAY_COLORS, **(colors or {})}
    marker_sizes = {**DEFAULT_MARKER_SIZES, **(marker_sizes or {})}
    group = gfx.Group()
    pairs = ray_segment_indices(rays, ray_indices)
    # A segment "teleports" if the point it ends on is a restore_neutron
    # duplicate.
    is_teleport = rays.kind[pairs[:, 1]] == TELEPORT
    regular_pairs, teleport_pairs = pairs[~is_teleport], pairs[is_teleport]

    values = ray_color_values(rays, color_mode)
    value_range = None
    group.ray_line = None
    if len(regular_pairs):
        positions = rays.points[regular_pairs].reshape(-1, 3).astype(np.float32)
        if values is None:
            vertex_colors = np.tile(hex_to_rgba(colors["ray"]), (len(positions), 1))
        else:
            used = values[regular_pairs.ravel()]
            value_range = (float(used.min()), float(used.max()))
            vertex_colors = colormap(used, *value_range)
        group.ray_line = gfx.Line(
            gfx.Geometry(positions=positions, colors=vertex_colors),
            gfx.LineSegmentMaterial(thickness=1.5, color_mode="vertex"),
        )
        group.add(group.ray_line)

    group.teleport_line = None
    if len(teleport_pairs):
        positions = rays.points[teleport_pairs].reshape(-1, 3).astype(np.float32)
        group.teleport_line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineSegmentMaterial(thickness=1.5, color=colors["teleport"]),
        )
        group.add(group.teleport_line)

    in_chosen = np.zeros(len(rays.points), dtype=bool)
    for i in ray_indices:
        in_chosen[rays.ray_offsets[i]:rays.ray_offsets[i + 1]] = True
    group.scatter_points = group.absorb_points = None
    for attr, kind, key in (
        ("scatter_points", SCATTER, "scatter"),
        ("absorb_points", ABSORB, "absorb"),
    ):
        points = rays.points[in_chosen & (rays.kind == kind)]
        if len(points):
            obj = gfx.Points(
                gfx.Geometry(positions=points.astype(np.float32)),
                gfx.PointsMaterial(size=marker_sizes[key], color=colors[key]),
            )
            group.add(obj)
            setattr(group, attr, obj)
    return group, value_range


def _build_counts_plane(lc, world_matrix, vmin, vmax):
    """A textured quad for a 2D LoggerCounts. Returns (mesh, texture)."""
    local_pts = logger_output.local_corners(lc.axis1, lc.axis2, lc.limits)
    world_pts = logger_output.world_points(local_pts, world_matrix).astype(np.float32)
    texture = gfx.Texture(logger_output.texture_image(lc.grid, vmin, vmax), dim=2)
    mesh = gfx.Mesh(
        gfx.Geometry(
            positions=world_pts,
            indices=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            texcoords=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
        ),
        # side="both": a logger's own local-frame winding has no meaningful
        # "outward" direction to get right (same reasoning as MCDISPLAY-drawn
        # components; see build_component_group).
        gfx.MeshBasicMaterial(map=gfx.TextureMap(texture), color_mode="vertex_map", side="both"),
    )
    return mesh, texture


def _build_counts_line(lc, world_matrix, vmin, vmax):
    """A coloured line for a 1D LoggerCounts: one segment per bin, its two
    endpoints both coloured by that bin's value."""
    n_bins = len(lc.grid)
    local_edges = logger_output.local_line_points(lc.axis1, lc.limits, n_bins)
    world_edges = logger_output.world_points(local_edges, world_matrix).astype(np.float32)
    positions = np.empty((2 * n_bins, 3), dtype=np.float32)
    positions[0::2] = world_edges[:-1]
    positions[1::2] = world_edges[1:]
    colors = logger_output.line_vertex_colors(lc.grid, vmin, vmax)
    return gfx.Line(
        gfx.Geometry(positions=positions, colors=colors),
        gfx.LineSegmentMaterial(thickness=4, color_mode="vertex"),
    )


def build_counts_group(counts, world_matrices, vmin, vmax):
    """One plane (2D LoggerCounts) or coloured line (1D LoggerCounts) per
    entry in counts (see logger_output.py), placed with world_matrices[name]
    - skipping any detector not found there (the loaded instrument doesn't
    match the run the counts came from). Returns (group, meshes, textures):
    meshes/textures are {name: obj} (textures only has an entry for the 2D,
    plane-shaped ones), so a colour-range change can restyle in place -
    texture.set_data() for a plane, geometry.colors.set_data() for a line -
    instead of rebuilding."""
    group = gfx.Group()
    meshes = {}
    textures = {}
    for name, lc in counts.items():
        world_matrix = world_matrices.get(name)
        if world_matrix is None:
            print(f"Warning: logger '{name}' not found in the loaded instrument - skipping.")
            continue
        if lc.axis2 is None:
            obj = _build_counts_line(lc, world_matrix, vmin, vmax)
        else:
            obj, texture = _build_counts_plane(lc, world_matrix, vmin, vmax)
            textures[name] = texture
        group.add(obj)
        meshes[name] = obj
    return group, meshes, textures


def make_coordinate_axes(
    length=10.0,
    tick_spacing=1.0,
    tick_size=0.1,
    show_labels=True,
):
    group = gfx.Group()

    axes = [
        ("X", (1, 0, 0), np.array([1, 0, 0])),
        ("Y", (0, 1, 0), np.array([0, 1, 0])),
        ("Z", (0, 0, 1), np.array([0, 0, 1])),
    ]

    for label, color, axis in axes:
        # main axis line
        positions = np.array([[0, 0, 0], axis * length], dtype=np.float32)
        line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=2.0, color=color),
        )
        group.add(line)

        # ticks
        n_ticks = int(length / tick_spacing)
        for i in range(1, n_ticks + 1):
            p = axis * i * tick_spacing
            perp = np.array([0, 1, 0]) if axis[0] else np.array([1, 0, 0])

            tick = gfx.Line(
                gfx.Geometry(
                    positions=np.array(
                        [p - perp * tick_size, p + perp * tick_size],
                        dtype=np.float32,
                    )
                ),
                gfx.LineMaterial(thickness=1.0, color=color),
            )
            group.add(tick)

        # label
        if show_labels:
            text = gfx.Text(
                text=label,
                font_size=0.5,
                material=gfx.materials.TextMaterial(color=color),
            )
            text.local.position = axis * (length + tick_size * 4)
            group.add(text)

    return group


# (intensity, position) of each directional light, all aimed at the origin:
# two main lights from opposite corners and a soft fill light.
DIRECTIONAL_LIGHTS = (
    (1.5, (1, 1, 1)),
    (1.5, (-1, -1, -1)),
    (0.5, (-1, 1, -1)),
)


def add_default_lights(scene):
    """A soft ambient light plus DIRECTIONAL_LIGHTS."""
    scene.add(gfx.AmbientLight(intensity=0.6))
    for intensity, position in DIRECTIONAL_LIGHTS:
        light = gfx.DirectionalLight(intensity=intensity)
        light.local.position = position
        light.look_at((0, 0, 0))
        scene.add(light)


# ============================================================
# Floor grid
# ============================================================

# The floor grid defaults to this footprint (with unit-spaced divisions)
# when nothing is loaded yet, or the loaded geometry is smaller than it.
DEFAULT_GRID_SIZE = 100.0
GRID_SPACING = 1.0
MAX_GRID_DIVISIONS = 500


def grid_size_for_bbox(bbox, minimum=DEFAULT_GRID_SIZE, margin=1.2):
    """Pick a floor-grid size that comfortably covers a scene's footprint.

    The grid lives in the xz-plane, so only the x and z extents of the
    bounding box matter; instruments taller than the default grid (e.g. a
    tall detector tank) don't need a larger footprint on their own.
    """
    if bbox is None:
        return minimum
    bmin, bmax = bbox
    extent = bmax - bmin
    footprint = max(extent[0], extent[2])
    return max(minimum, footprint * margin)
