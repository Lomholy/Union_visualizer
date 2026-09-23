"""Display-independent helpers for the union_viewer GUI: material name
normalisation/grouping, vacuum detection, and default colour assignment."""

import trimesh


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


def component_color_key(name):
    """Key into the shared colours dict for a McStas component, kept apart
    from Union component and material keys."""
    return f"comp:{name}"


def clip_planes(clip):
    """The viewer's clip settings as pygfx clipping planes. "Above" keeps
    the side where the axis coordinate is greater than the position, like
    brep.clip_component."""
    if not clip["enable"]:
        return []
    normal = [0.0, 0.0, 0.0]
    normal["XYZ".index(clip["axis"].upper())] = 1.0
    position = float(clip["position"])
    if clip["mode"] == "Above":
        return [(*normal, -position)]
    return [(*(-n for n in normal), position)]


def parse_instrument_params(text):
    """Split the "Instrument parameters" field into mcrun name=value args."""
    params = text.replace(",", " ").split()
    bad = [p for p in params if "=" not in p]
    if bad:
        raise ValueError(f"Expected name=value, got: {' '.join(bad)}")
    return params
