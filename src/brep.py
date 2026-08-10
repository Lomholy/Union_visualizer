from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCone
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.gp import gp_Pln
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeHalfSpace
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
import trimesh
import numpy as np


def build_comp_brep(comp, world_matrices):
    name = comp.name
    c_type = comp.component_name
    mat = world_matrices[name]
    pnt = mat[:3, 3]
    x_dir = mat[:3, 0]
    y_dir = mat[:3, 1]
    z_dir = mat[:3, 2]
    axis = gp_Ax2(gp_Pnt(*pnt), gp_Dir(*x_dir), gp_Dir(*z_dir))
    if c_type == "Union_sphere":
        brep = BRepPrimAPI_MakeSphere(axis, comp.radius).Shape()
    elif c_type == "Union_box":
        corner = (
            pnt
            - 0.5 * comp.xwidth * x_dir
            - 0.5 * comp.yheight * y_dir
            - 0.5 * comp.zdepth * z_dir
        )

        axis = gp_Ax2(
            gp_Pnt(*corner),
            gp_Dir(*z_dir),  # local Z
            gp_Dir(*x_dir),  # local X
        )

        brep = BRepPrimAPI_MakeBox(axis, comp.xwidth, comp.yheight, comp.zdepth).Shape()

    elif c_type == "Union_cylinder":
        pnt = pnt - np.array([0, comp.yheight / 2, 0])
        axis = gp_Ax2(gp_Pnt(*pnt), gp_Dir(*y_dir))
        brep = BRepPrimAPI_MakeCylinder(axis, comp.radius, comp.yheight).Shape()
    elif c_type == "Union_cone":
        pnt = pnt - comp.yheight / 2 * y_dir
        axis = gp_Ax2(gp_Pnt(*pnt), gp_Dir(*y_dir))
        brep = BRepPrimAPI_MakeCone(
            axis, comp.radius_bottom, comp.radius_top, comp.yheight
        ).Shape()
    return brep


def build_higher_priorities(higher_priorities, world_matrices):
    breps = []
    for i, comp in enumerate(higher_priorities):
        brep = build_comp_brep(comp, world_matrices)
        breps.append(brep)
    return breps


def subtract_higher_priorities(comp, prio_breps):
    for prio in prio_breps:
        cut = BRepAlgoAPI_Cut(comp, prio)
        cut.Build()
        if not cut.IsDone():
            raise RuntimeError("Boolean cut failed")
        comp = cut.Shape()
    return comp


def clip_component(shape, clip):
    if not clip["enable"]:
        return shape

    axis_name = clip["axis"].upper()
    position = float(clip["position"])
    mode = clip["mode"]

    if axis_name == "X":
        normal = np.array([1.0, 0.0, 0.0])
    elif axis_name == "Y":
        normal = np.array([0.0, 1.0, 0.0])
    elif axis_name == "Z":
        normal = np.array([0.0, 0.0, 1.0])
    else:
        raise ValueError(f"Unknown clip axis: {axis_name}")

    # plane point
    plane_point = np.zeros(3)

    if axis_name == "X":
        plane_point[0] = position
    elif axis_name == "Y":
        plane_point[1] = position
    elif axis_name == "Z":
        plane_point[2] = position

    # select side to keep
    if mode == "Above":
        keep_point = plane_point + normal
    elif mode == "Below":
        keep_point = plane_point - normal
    else:
        raise ValueError(f"Unknown clip mode: {mode}")

    plane = gp_Pln(
        gp_Pnt(*plane_point),
        gp_Dir(*normal)
    )

    plane_face = BRepBuilderAPI_MakeFace(plane).Face()

    halfspace = BRepPrimAPI_MakeHalfSpace(
        plane_face,
        gp_Pnt(*keep_point)
    ).Solid()

    result = BRepAlgoAPI_Common(
        shape,
        halfspace
    )

    result.Build()

    if not result.IsDone():
        raise RuntimeError("Clip operation failed")

    return result.Shape()


def build_single_brep_mesh(comp, union_geometries, world_matrices, clip, verbose):
    higher_priority = [
        x for n, x in union_geometries.items() if x.priority > comp.priority
    ]
    res_comp = build_comp_brep(comp, world_matrices)
    prio_breps = build_higher_priorities(higher_priority, world_matrices)
    res_comp = subtract_higher_priorities(res_comp, prio_breps)
    res_comp = clip_component(res_comp, clip)

    BRepMesh_IncrementalMesh(res_comp, 0.01).Perform()
    vertices = []
    faces = []

    vertex_offset = 0

    explorer = TopExp_Explorer(res_comp, TopAbs_FACE)

    while explorer.More():
        face = explorer.Current()

        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)

        if triangulation is None:
            explorer.Next()
            continue

        trsf = loc.Transformation()

        for i in range(1, triangulation.NbNodes() + 1):
            p = triangulation.Node(i).Transformed(trsf)

            vertices.append([p.X(), p.Y(), p.Z()])

        # triangles
        for i in range(1, triangulation.NbTriangles() + 1):
            tri = triangulation.Triangle(i)

            n1, n2, n3 = tri.Get()
            if face.Orientation() == TopAbs_REVERSED:
                n2, n3 = n3, n2

            faces.append(
                [
                    vertex_offset + n1 - 1,
                    vertex_offset + n2 - 1,
                    vertex_offset + n3 - 1,
                ]
            )

        vertex_offset += triangulation.NbNodes()

        explorer.Next()
    if len(vertices) == 0 or len(faces) == 0:
        print(f"WARNING: empty mesh for {comp.name}")
        return None
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.fix_normals()

    return mesh


def build_brep_meshes(union_geometries, world_matrices, clip, verbose):
    meshes_dict = {}

    for name, comp in union_geometries.items():
        mesh = build_single_brep_mesh(
            comp, union_geometries, world_matrices, clip, verbose
        )
        meshes_dict[name] = mesh
    return meshes_dict
