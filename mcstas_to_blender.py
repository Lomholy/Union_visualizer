# Author: Daniel Lomholt Christensen @NBI & UCPH 05/03/2026
# This script takes a mcstasscript instrument as an input, and
# writes a new file, that creates the union geometry in Blender
#

import mcstasscript as ms
import mcstasscript.helper.mcstas_objects as mshelp
import argparse
import numpy as np

# ===============================================================================
# =================== I/O Helpers
# ===============================================================================


def execute_mcstasscript_file(input_file):
    # Execute the input python file and extract instrument objects
    namespace = {}
    with open(input_file, "r") as f:
        code = f.read()

    exec(compile(code, input_file, "exec"), namespace)

    # Collect all mcstasscript instruments
    instruments = [
        obj for obj in namespace.values() if isinstance(obj, ms.McStas_instr)
    ]

    if len(instruments) != 1:
        raise ValueError(f"Expected exactly one instrument, found {
                         len(instruments)}.")

    return instruments[0]


def init_build_blender(instr):
    lines = []
    lines.append("import bpy")
    lines.append("import math")
    lines.append("from mathutils import Matrix")
    lines.append("")
    lines.append("# Auto-generated from mcstasscript instrument")
    lines.append("# Author: Daniel Lomholt Christensen @NBI @UCPH 05/03/2026")
    lines.append(f"# Instrument name: {instr.name}")
    lines.append("")
    lines.append("# Clear scene")
    lines.append("bpy.ops.object.select_all(action='SELECT')")
    lines.append("bpy.ops.object.delete(use_global=False)")
    lines.append("for block in bpy.data.meshes: bpy.data.meshes.remove(block)")
    lines.append(
        "for block in bpy.data.objects: bpy.data.objects.remove(block)")
    lines.append("")
    return "\n".join(lines)


# def priority_cuts():
#     return '''
# # Collect all mesh objects that have a priority property
# objs = [o for o in bpy.data.objects if "priority" in o and o.type == "MESH"]

# # Sort by priority (lowest → highest)
# objs_sorted = sorted(objs, key=lambda o: o["priority"], reverse=False)

# for i, obj in enumerate(objs_sorted):
#     p = obj["priority"]
#     if p is None:
#         continue

#     # Find all higher-priority mesh cutters
#     cutters_src = [o for o in objs_sorted if o.type == "MESH" and o["priority"] > p]
#     if not cutters_src:
#         continue

#     # --- Duplicate cutters so originals remain intact ---
#     dupes = []
#     for c in cutters_src:
#         d = c.copy()
#         d.data = c.data.copy()
#         d.hide_set(False)
#         bpy.context.scene.collection.objects.link(d)
#         dupes.append(d)

#     # --- Create a fresh receiver mesh to JOIN INTO ---
#     recv_mesh = bpy.data.meshes.new(f"_cut_p{int(p)}_mesh")
#     combined = bpy.data.objects.new(f"_cut_p{int(p)}", recv_mesh)
#     bpy.context.scene.collection.objects.link(combined)

#     # Ensure Object mode, then select receiver + dupes and JOIN
#     bpy.ops.object.mode_set(mode="OBJECT", toggle=False)
#     bpy.ops.object.select_all(action="DESELECT")
#     combined.select_set(True)
#     for d in dupes:
#         d.select_set(True)
#     bpy.context.view_layer.objects.active = combined

#     if dupes:  # Only join if there is at least one duplicate
#         bpy.ops.object.join()  # Blender deletes the dupes here; only 'combined' survives

#     # --- Boolean DIFFERENCE with FLOAT solver ---
#     mod = obj.modifiers.new(name=f"bool_cut_{obj.name}", type="BOOLEAN")
#     mod.operation = "DIFFERENCE"
#     mod.solver = "FLOAT"
#     mod.object = combined

#     bpy.ops.object.select_all(action="DESELECT")
#     obj.hide_set(False)
#     obj.select_set(True)
#     bpy.context.view_layer.objects.active = obj
#     bpy.ops.object.modifier_apply(modifier=mod.name)

#     # --- Cleanup: ONLY remove the combined receiver (dupes were deleted by join) ---
#     if combined.name in bpy.data.objects:
#         bpy.data.objects.remove(combined, do_unlink=True)'''


def priority_cuts():
    return """
# Watertight priority cuts: boolean (EXACT) → weld → voxel remesh → recalc normals
objs = [o for o in bpy.data.objects if "priority" in o and o.type == "MESH"]
objs_sorted = sorted(objs, key=lambda o: o["priority"])

for obj in objs_sorted:
    p = obj["priority"]
    if p is None:
        continue

    cutters = [o for o in objs_sorted if o["priority"]>p]
    if not cutters:
        continue

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Boolean DIFFERENCE with manifold solver
    for cutter in cutters:
        if not cutter.data or cutter == obj:
            continue
        mod = obj.modifiers.new(name=f"bool_cut_{cutter.name}", type="BOOLEAN")
        mod.operation = "DIFFERENCE"
        mod.solver = "MANIFOLD"
        mod.object = cutter

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)

    # Weld tiny cracks (relative threshold)
    mx = max(obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]) if obj.dimensions.length else 1.0
    weld = obj.modifiers.new(name="weld_fix", type="WELD")
    weld.merge_threshold = max(1e-6, mx * 1e-5)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=weld.name)

    # Voxel remesh to guarantee watertight mesh
    mx = max(obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]) if obj.dimensions.length else 1.0
    vs = max(mx / 512.0, 1e-4)  # adaptive voxel size
    rem = obj.modifiers.new(name="remesh_voxel", type="REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = vs
    rem.adaptivity = 0.0
    rem.use_smooth_shade = False
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=rem.name)

    # Recalculate normals outside
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
"""


def write_blender_script(result, args):
    # Write the result to the given file
    output = args.output_file
    with open(output, "w") as f:
        f.write(result)


# ===============================================================================
# =================== Add in for components
# ===============================================================================

def attempt_single_param_conversion(var_map, name, comp, instr):
    # Check dictionary of parameters
    if name in instr.parameters and instr.parameters[name] is not None:
        param_value = instr.parameters[name].value
        setattr(comp, name, param_value)
    elif name in var_map:
        setattr(comp, name, var_map[name])

def attempt_conversion(comp: mshelp.Component, instr: ms.McStas_instr):
    all_vars = list(instr.declare_list) + list(instr.user_var_list)
    var_map = {v.name: v.value for v in all_vars}

    for name in (a for a in dir(comp) if not a.startswith('__')):
        value = getattr(comp, name)
        if isinstance(value, (str, int, float)):
            attempt_single_param_conversion(var_map, name, comp, instr)

    return comp


def build_cylinder(comp, instr):
    name = comp.name
    rad = comp.radius
    yheight = comp.yheight
    px, py, pz = comp.AT_data[0], comp.AT_data[2], comp.AT_data[1]
    rx, ry, rz = comp.ROTATED_data[0], comp.ROTATED_data[2], comp.ROTATED_data[1]
    rel = comp.AT_relative
    if rel.startswith("RELATIVE"):
        rel = rel.split(" ")[1]

    lines = []
    lines.append(f"# Create cylinder for {name}")
    lines.append(
        f"bpy.ops.mesh.primitive_cylinder_add(radius={rad}, depth={
            yheight
        }, enter_editmode=False)"
    )
    lines.append("obj = bpy.context.active_object")
    lines.append(f"obj.name = '{name}'")
    lines.append(f"obj['priority'] = {comp.priority}")
    lines.append(f"obj.location = ({px}, {py}, {pz})")
    lines.append(
        f"obj.rotation_euler = (math.radians({rx}), math.radians({ry}), math.radians({
            rz
        }))"
    )
    if rel and str(rel).lower() != "absolute":
        lines.append(f"parent_obj = bpy.data.objects.get('{rel}')")
        lines.append("if parent_obj: obj.parent = parent_obj")
    material = comp.material_string
    if material.startswith('"'):
        material = material[1:-1]
    print(comp.material_string)
    print(material)
    mat_name = f"Union_Make_Material_{material}"
    lines.append(
        f"mat = bpy.data.materials.get('{
            mat_name
        }') or bpy.data.materials.get('Union_Vacuum')"
    )
    lines.append("if mat:")
    lines.append("    if obj.data.materials:")
    lines.append("        obj.data.materials[0] = mat")
    lines.append("    else:")
    lines.append("        obj.data.materials.append(mat)")

    # print(comp.material_string)

    return lines


def build_secret_comp(comp, instr):
    name = comp.name
    px, py, pz = comp.AT_data[0], comp.AT_data[2], comp.AT_data[1]
    rx, ry, rz = comp.ROTATED_data[0], comp.ROTATED_data[2], comp.ROTATED_data[1]
    # Check through that the data is in numbers, and attempt conversion
    rel = comp.AT_relative
    if rel.startswith("RELATIVE"):
        rel = rel.split(" ")[1]

    lines = []
    lines.append(f"# Empty for {name} (coordinate system only)")
    lines.append(f"empty = bpy.data.objects.new('{name}', None)")
    lines.append("bpy.context.scene.collection.objects.link(empty)")
    lines.append(f"empty.location = ({px}, {py}, {pz})")
    lines.append(
        f"empty.rotation_euler = (math.radians({rx}), math.radians({ry}), math.radians({
            rz
        }))"
    )
    if rel and str(rel).lower() != "absolute":
        lines.append(f"parent_obj = bpy.data.objects.get('{rel}')")
        lines.append("if parent_obj: empty.parent = parent_obj")
    return lines


def add_single_comp(comp: mshelp.Component, instr: ms.McStas_instr):
    supported_components = ["Union_cylinder"]
    # Attempt to convert parameters
    attempt_conversion(comp, instr)
    print(comp.component_name)
    if comp.component_name in supported_components:
        print("Now building the geometry")
        if comp.component_name == supported_components[0]:
            lines = build_cylinder(comp, instr)
        else:
            raise RuntimeError()
    else:
        print("Adding in a secret object, that merely serves as a coordinate system")
        lines = build_secret_comp(comp, instr)
    # print(lines)
    comp_bpy = "\n".join(lines) + "\n"
    return comp_bpy


def build_materials(instr, result):
    # First build materials for each make material component
    result.append("# Build materials")

    for comp in instr.component_list:
        if comp.component_name == "Union_make_material":
            mat_var = f"Union_Make_Material_{comp.name}"
            result.append(
                f"{mat_var} = bpy.data.materials.get('{
                    mat_var
                }') or bpy.data.materials.new(name='{mat_var}')"
            )
            result.append(
                f"nodes = {mat_var}.node_tree.nodes; links = {
                    mat_var}.node_tree.links"
            )
            result.append("nodes.clear()")
            result.append(
                "out = nodes.new('ShaderNodeOutputMaterial'); out.location = (300,0)"
            )
            result.append(
                "bsdf = nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (0,0)"
            )
            result.append(
                f"r,g,b = {np.random.uniform()}, {np.random.uniform()}, {
                    np.random.uniform()
                }"
            )
            result.append(
                "bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)")
            result.append("bsdf.inputs['Roughness'].default_value = 0.4")
            result.append(
                "links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])")
    # Vacuum material (invisible)
    result.append(
        "Union_Vacuum = bpy.data.materials.get('Union_Vacuum') or bpy.data.materials.new(name='Union_Vacuum')"
    )
    result.append(
        "nodes = Union_Vacuum.node_tree.nodes; links = Union_Vacuum.node_tree.links"
    )
    result.append(
        "nodes = Union_Vacuum.node_tree.nodes; links = Union_Vacuum.node_tree.links"
    )
    result.append("nodes.clear()")

    result.append(
        "out = nodes.new('ShaderNodeOutputMaterial'); out.location = (300,0)")
    result.append(
        "mix = nodes.new('ShaderNodeMixShader'); mix.location = (100,0)")
    result.append(
        "transp = nodes.new('ShaderNodeBsdfTransparent'); transp.location = (-100,50)"
    )
    result.append(
        "bsdf = nodes.new('ShaderNodeBsdfPrincipled'); bsdf.location = (-100,-100)"
    )

    # Set fully transparent BSDF
    result.append("bsdf.inputs['Alpha'].default_value = 0.0")

    # Mix: 100% transparent
    result.append("mix.inputs['Fac'].default_value = 1.0")

    result.append("links.new(transp.outputs['BSDF'], mix.inputs[2])")
    result.append("links.new(bsdf.outputs['BSDF'], mix.inputs[1])")
    result.append("links.new(mix.outputs['Shader'], out.inputs['Surface'])")

    # Material settings
    result.append("Union_Vacuum.blend_method = 'BLEND'")
    result.append("Union_Vacuum.use_backface_culling = True")
    result.append(
        "if hasattr(Union_Vacuum, 'use_shadow_cast'): Union_Vacuum.use_shadow_cast = False"
    )
    result.append(
        "if hasattr(Union_Vacuum, 'use_transparent_shadows'): Union_Vacuum.use_transparent_shadows = False"
    )

    return "\n".join(result) + "\n"


def add_all_components(result: list, instr: ms.McStas_instr):
    # we add components in the following algorithm:
    # First add all components relative to absolute.
    # Then add all components relative to one of the components we have already made.
    # Continue to do this until no more components are being added with a new pass.
    # Then if more are missing, raise an error.
    remaining = instr.component_list

    # Components successfully added
    added = set()
    # First make all materials
    build_materials(instr, result)

    # Continue making passes until no progress is possible
    while True:
        start_count = len(added)

        for comp in remaining[:]:
            AT_rel = comp.AT_relative
            ROT_rel = comp.ROTATED_relative
            if AT_rel.startswith("RELATIVE"):
                AT_rel = AT_rel.split(" ")[1]
            if ROT_rel.startswith("RELATIVE"):
                ROT_rel = ROT_rel.split(" ")[1]

            if AT_rel != ROT_rel and AT_rel != None and ROT_rel.lower() != "absolute":
                print(
                    f"Error, {
                        comp.name} has differing relatives in AT and ROT not yet supported"
                )
                continue
            # Case 1: relative to ABSOLUTE
            if AT_rel is None or AT_rel.lower() == "absolute":
                result.append(add_single_comp(comp, instr))
                added.add(comp.name)
                remaining.remove(comp)
                continue

            # Case 2: AT_relative to previously-added component
            if AT_rel in added:
                result.append(add_single_comp(comp, instr))
                added.add(comp.name)
                remaining.remove(comp)
                continue

        # Stop if no new components were added in this pass
        if len(added) == start_count:
            break
    # If anything is still missing, dependency graph is broken
    if remaining:
        names = ", ".join(c.name for c in remaining)
        raise ValueError(
            f"Could not place all components. Unresolved: {names}")
    return "\n".join(result)


def main():
    print("Running main ")
    parser = argparse.ArgumentParser(
        description="Export Union_* components from a mcstasscript instrument to Blender bpy script."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input mcstasscript file that will be converted to blender python",
    )

    parser.add_argument(
        "--output_file", type=str, required=True, help="Output blender python file"
    )

    args = parser.parse_args()
    instr = execute_mcstasscript_file(args.input_file)
    result = init_build_blender(instr)
    result = add_all_components([result], instr)
    end_string = priority_cuts()
    result += "\n" + end_string

    write_blender_script(result, args)


if __name__ == "__main__":
    main()
