"""Pure, display-independent helpers for the union_viewer GUI: material
name normalisation/grouping and vacuum detection.

Kept out of union_viewer.py, which needs a Qt application to import its
widget-construction code meaningfully, so this logic can be unit tested
without a display."""

import trimesh


def normalize_material_string(material_string):
    """Strip one layer of surrounding double quotes from a component's
    material_string, and normalise a missing one to "".

    The .instr text parser sometimes leaves the quotes from the source
    literal embedded in the value (e.g. material_string='"sample_0"'
    rather than 'sample_0') and sometimes does not, depending on how the
    instrument wrote it - the same wrinkle preprocess._resolve_relative_path
    already handles for Union_mesh filenames (confirmed on
    tests/crack_height.instr, where one component's material_string comes
    back quoted and the other's does not). Every consumer that groups or
    matches on material_string should go through this first."""
    if material_string is None:
        return ""
    text = str(material_string).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text


# McStas's Union_box.comp (and its sibling geometry components) recognise
# exactly these four spellings via a plain C strcmp - case sensitive, no
# other casing - to mean "non-scattering, don't build a material for this
# volume". "exit" is a vacuum that also marks the ray as having left the
# instrument (is_exit_volume set), which is why it counts here too.
_VACUUM_MATERIAL_STRINGS = {"vacuum", "Vacuum", "exit", "Exit"}


def is_vacuum_material(material_string):
    """Is this Union geometry's material one of the vacuum/exit spellings
    McStas itself treats as non-scattering?"""
    return normalize_material_string(material_string) in _VACUUM_MATERIAL_STRINGS


def group_components_by_material(union_geometries):
    """{material: [component names]}, in the order components first appear
    in union_geometries. A component with no material_string set groups
    under "" alongside any other component that also has none, rather than
    under a per-component key - they are indistinguishable by material."""
    groups = {}
    for name, comp in union_geometries.items():
        material = normalize_material_string(getattr(comp, "material_string", None))
        groups.setdefault(material, []).append(name)
    return groups


def group_meshes_by_material(union_geometries, meshes):
    """{material: trimesh}, concatenating every component's mesh that
    shares a material. A component with no mesh (a mask component, or one
    a mesher failed to build) contributes nothing; a material left with no
    surviving mesh is dropped entirely rather than included as None, so
    callers never need to null-check the result."""
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
