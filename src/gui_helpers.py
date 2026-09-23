"""Display-independent helpers for the union_viewer GUI: material name
normalisation/grouping, vacuum detection, default colour assignment, and
neutron ray selection/colouring."""

import importlib.util

import numpy as np
import trimesh

from clipping import clip_plane


def normalize_material_string(material_string):
    """Strip a surrounding pair of double quotes from material_string, and
    normalise a missing one to ""."""
    if material_string is None:
        return ""
    text = str(material_string).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text


# The four spellings McStas's Union geometry components treat as vacuum
# (case-sensitive strcmp).
_VACUUM_MATERIAL_STRINGS = {"vacuum", "Vacuum", "exit", "Exit"}


def is_vacuum_material(material_string):
    """Is this material one of McStas's vacuum/exit spellings?"""
    return normalize_material_string(material_string) in _VACUUM_MATERIAL_STRINGS


def group_components_by_material(union_geometries):
    """{material: [component names]}, in order of first appearance."""
    groups = {}
    for name, comp in union_geometries.items():
        material = normalize_material_string(getattr(comp, "material_string", None))
        groups.setdefault(material, []).append(name)
    return groups


def group_meshes_by_material(union_geometries, meshes):
    """{material: trimesh}, concatenating every component's mesh that
    shares a material. Components with no mesh are skipped; a material left
    with none is dropped from the result."""
    groups = group_components_by_material(union_geometries)
    grouped_meshes = {}
    for material, names in groups.items():
        parts = [meshes[name] for name in names if meshes.get(name) is not None]
        if not parts:
            continue
        grouped_meshes[material] = (
            parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
        )
    return grouped_meshes


# Default colour cycle. The first 11 are the requested starting colours;
# the rest extend the cycle so it doesn't repeat until ~30 keys are in use.
DEFAULT_COLOR_CYCLE = [
    "#E40303",  # Red
    "#FF8C00",  # Orange
    "#FFED00",  # Yellow
    "#008026",  # Green
    "#004DFF",  # Blue
    "#750787",  # Violet
    "#000000",  # Black
    "#613915",  # Brown
    "#74D7EE",  # Light Blue
    "#FFAFC8",  # Pink
    "#FFFFFF",  # White
    "#008080",  # Teal
    "#800000",  # Maroon
    "#808000",  # Olive
    "#000080",  # Navy
    "#32CD32",  # Lime
    "#00FFFF",  # Cyan
    "#FF00FF",  # Magenta
    "#FFD700",  # Gold
    "#4B0082",  # Indigo
    "#FF7F50",  # Coral
    "#40E0D0",  # Turquoise
    "#FA8072",  # Salmon
    "#808080",  # Gray
    "#7FFF00",  # Chartreuse
    "#DC143C",  # Crimson
    "#6A5ACD",  # Slate Blue
    "#F0E68C",  # Khaki
    "#DA70D6",  # Orchid
    "#4682B4",  # Steel Blue
]


_CYCLE_INDEX_KEY = "__cycle_index__"


def assign_default_color(colors, key):
    """Ensure colors[key] exists, assigning the next unused colour from
    DEFAULT_COLOR_CYCLE if not. Leaves an existing entry untouched, and
    returns the resulting colour. The cycle position is tracked separately
    (colors[_CYCLE_INDEX_KEY]) rather than derived from len(colors), so it
    still advances correctly after a delete-and-reassign."""
    if key not in colors:
        index = colors.get(_CYCLE_INDEX_KEY, 0)
        colors[key] = DEFAULT_COLOR_CYCLE[index % len(DEFAULT_COLOR_CYCLE)]
        colors[_CYCLE_INDEX_KEY] = index + 1
    return colors[key]


LABEL_WRAP_WIDTH = 18


def wrap_label(text, width=LABEL_WRAP_WIDTH):
    """text broken onto lines of at most width characters, for panel rows.
    McStas names rarely contain spaces, so a line preferably ends just
    after a separator (_ - . / or space) and is cut mid-word only when it
    has none."""
    lines = []
    while len(text) > width:
        cut = max(text.rfind(sep, 1, width) for sep in " _-./") + 1
        if cut <= 1:
            cut = width
        lines.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    lines.append(text)
    return "\n".join(lines)


def component_color_key(name):
    """Key into the shared colours dict for a McStas component, kept apart
    from Union component and material keys."""
    return f"comp:{name}"


def clip_planes(clip):
    """The viewer's clip settings as pygfx clipping planes (pygfx keeps
    a*x + b*y + c*z + d >= 0), cutting the same side as the meshers."""
    if not clip["enable"]:
        return []
    normal, point = clip_plane(clip)
    return [(*(float(n) for n in normal), -float(normal @ point))]


def instrument_param_args(param_values):
    """mcrun name=value args for the parameters given a value; the rest
    keep the instrument's defaults."""
    return [f"{name}={text.strip()}" for name, text in param_values.items() if text.strip()]


def _can_cap_slices():
    """trimesh can only cap a cut mesh with shapely and mapbox_earcut."""
    try:
        return all(importlib.util.find_spec(m) for m in ("shapely", "mapbox_earcut"))
    except (ImportError, ValueError):
        return False


def clip_mesh(mesh, clip):
    """mesh cut by the viewer's clip plane, keeping the same side as the
    view. The cut is capped when mesh is closed and trimesh's capping
    dependencies are installed, and left open otherwise. None if nothing
    is left."""
    if not clip["enable"]:
        return mesh
    normal, origin = clip_plane(clip)
    if mesh.is_watertight and _can_cap_slices():
        try:
            clipped = trimesh.intersections.slice_mesh_plane(mesh, normal, origin, cap=True)
            return clipped if clipped is not None and len(clipped.faces) else None
        except Exception as e:
            print(f"Warning: could not cap the clipped mesh ({e}); exporting it open.")
    vertices, faces = trimesh.intersections.slice_faces_plane(
        mesh.vertices, mesh.faces, normal, origin
    )[:2]
    if len(faces) == 0:
        return None
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# ---------------------------------------------------------------------------
# Neutron rays
# ---------------------------------------------------------------------------

# {mode: (label, unit)}; "Uniform" draws every ray in one colour.
RAY_COLOR_MODES = {
    "Uniform": ("", ""),
    "Speed": ("Speed", "m/s"),
    "Weight": ("log10 weight", ""),
    "Time": ("Time", "ms"),
}

_VIRIDIS = np.array([
    [0.267, 0.005, 0.329],
    [0.230, 0.322, 0.546],
    [0.128, 0.567, 0.551],
    [0.369, 0.789, 0.383],
    [0.993, 0.906, 0.144],
])


def rays_reaching(rays, component_name=None):
    """Indices of the rays with at least one point in component_name's
    frame, or of every ray when component_name is None."""
    if component_name is None:
        return np.arange(rays.n_rays)
    if component_name not in rays.component_names:
        return np.arange(0)
    index = rays.component_names.index(component_name)
    hits = rays.component == index
    return np.array([
        i for i in range(rays.n_rays)
        if hits[rays.ray_offsets[i]:rays.ray_offsets[i + 1]].any()
    ], dtype=int)


def ray_segment_indices(rays, ray_indices):
    """(k, 2) indices into rays.points of every consecutive point pair
    within the chosen rays, never joining the end of one ray to the start
    of the next."""
    starts = [
        np.arange(rays.ray_offsets[i], rays.ray_offsets[i + 1] - 1)
        for i in ray_indices
    ]
    if not starts:
        return np.zeros((0, 2), dtype=int)
    first = np.concatenate(starts)
    return np.stack([first, first + 1], axis=1)


def ray_color_values(rays, mode):
    """Per-point value to colour by for a RAY_COLOR_MODES mode, or None for
    "Uniform"."""
    if mode == "Speed":
        return rays.speed
    if mode == "Weight":
        positive = rays.weight[rays.weight > 0]
        floor = positive.min() if len(positive) else 1e-300
        return np.log10(np.maximum(rays.weight, floor))
    if mode == "Time":
        return rays.time * 1e3
    return None


def colormap(values, vmin=None, vmax=None):
    """(n, 4) float32 viridis RGBA for values, scaled to [vmin, vmax]."""
    values = np.asarray(values, dtype=float)
    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    t = np.zeros_like(values) if vmax <= vmin else (values - vmin) / (vmax - vmin)
    t = np.clip(t, 0, 1) * (len(_VIRIDIS) - 1)
    stops = np.arange(len(_VIRIDIS))
    rgb = np.stack([np.interp(t, stops, _VIRIDIS[:, c]) for c in range(3)], axis=1)
    return np.concatenate([rgb, np.ones((len(values), 1))], axis=1).astype(np.float32)
