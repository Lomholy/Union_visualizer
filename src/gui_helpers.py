"""Display-independent helpers for the union_viewer GUI: material name
normalisation/grouping, vacuum detection, and default colour assignment."""

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