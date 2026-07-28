from plot_union_cloud import sdf_normal
import trimesh
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
from plot_union_cloud import generate_points
import plotly.graph_objects as go


@dataclass
class OctreeNode:
    """
    A single node in the octree.

    Each node represents an axis-aligned bounding box from bmin to bmax.
    If the node has children, it is an internal octree node.
    If children is empty, the node is a leaf cell.
    Once the octree has been built, the children of a leaf cube are populated
    with the cube itself.

    The dual contouring data is stored only for leaf nodes that contain
    a sign change of the SDF.
    """

    bmin: np.ndarray
    bmax: np.ndarray
    center: np.ndarray
    depth: int
    children: list
    is_leaf: bool = False

    # dual contouring data.
    corner_signs: np.ndarray | None = None
    hermite_points: list | None = None
    hermite_normals: list | None = None
    vertex: np.ndarray | None = None


def build_uniform_octree_from_points(
    points,
    bbox_min,
    bbox_max,
    max_depth=6,
    min_points=20,
):
    """
    Build a uniform octree over the given bounding box.

    Every cell is subdivided until max_depth is reached.

    This gives a regular grid of cells with resolution:

        n = 2 ** max_depth

    along each axis, for a total of:

        n^3 = 8 ** max_depth

    leaf cells.

    Parameters
    ----------
    points:
        Input point cloud. This argument is kept for compatibility with
        the old function signature, but is not used for subdivision.
        In the future it will be used to determine the necessary depth
        of the octree
    bbox_min, bbox_max:
        Root bounding box of the octree.
    max_depth:
        Uniform subdivision depth.
    min_points:
        Parameter determining the minimum amount of points inside a box for
        it to be split again.
    Returns
    -------
    OctreeNode
        Root node of the uniformly subdivided octree.
    """

    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)
    return recurse(
        bbox_min,
        bbox_max,
        depth=0,
        max_depth=max_depth,
        points=points,
        min_points=min_points,
    )


def recurse(bmin, bmax, depth, max_depth, points, min_points):
    """
    Recursively subdivide every cell until max_depth, or less than min points
    are contained within the box


    """
    center = 0.5 * (bmin + bmax)

    node = OctreeNode(
        bmin=np.asarray(bmin, dtype=float),
        bmax=np.asarray(bmax, dtype=float),
        center=center,
        depth=depth,
        children=[],
    )
    npts = len(points)
    # Stop once all cells have reached the same target depth.
    if depth >= max_depth or npts < min_points:
        return node

    # Determine which side of the center each point lies on
    octant_ids = (
        (points[:, 0] >= center[0]).astype(np.int32) * 4
        + (points[:, 1] >= center[1]).astype(np.int32) * 2
        + (points[:, 2] >= center[2]).astype(np.int32)
    )

    # At each subdivision, the grid index is doubled and offset by
    # the local child octant index.
    #
    # Example:
    # parent index (i, j, k)
    # child index = (2*i + ix, 2*j + iy, 2*k + iz)
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
                octant = 4 * ix + 2 * iy + iz
                child_points = points[octant_ids == octant]

                child_min = np.array(
                    [
                        bmin[0] if ix == 0 else center[0],
                        bmin[1] if iy == 0 else center[1],
                        bmin[2] if iz == 0 else center[2],
                    ],
                    dtype=float,
                )

                child_max = np.array(
                    [
                        center[0] if ix == 0 else bmax[0],
                        center[1] if iy == 0 else bmax[1],
                        center[2] if iz == 0 else bmax[2],
                    ],
                    dtype=float,
                )
                child = recurse(
                    child_min, child_max, depth + 1, max_depth, child_points, min_points
                )

                node.children.append(child)

    return node


def get_octree_leaves(node):
    """
    Collect all leaf nodes in the octree.

    A leaf node is any node with no children.
    These are the cells where dual contouring vertices will be computed.
    """

    leaves = []

    def recurse(n):
        if n is None:
            return

        # No children means this node is a leaf cell.
        if len(n.children) == 0:
            leaves.append(n)
            n.is_leaf = True
            return

        # Otherwise continue descending through the tree.
        n.is_leaf = False
        for child in n.children:
            recurse(child)

    recurse(node)

    return leaves


# Edge list for a hexahedral cell.
#
# The corners are assumed to be ordered as:
#
#   7 ----- 6
#  /|      /|
# 4 ----- 5 |
# | 3 ---|- 2
# |/     |/
# 0 ----- 1
#
# Edges are pairs of corner indices.
EDGE_VERTS = [
    (0, 1),
    (1, 3),
    (2, 3),
    (0, 2),
    (4, 5),
    (5, 7),
    (6, 7),
    (4, 6),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]

flag_fix_verts = [0]


def calc_vert(leaf, sdf):
    """
    Compute the dual contouring vertex for a single octree leaf cell.

    The method is:

    1. Evaluate the SDF at the eight cell corners.
    2. If all corners have the same sign, the surface does not cross
       this cell, so no vertex is created.
    3. For every edge with a sign change, find an approximate zero
       crossing of the SDF.
    4. Estimate the SDF normal at each crossing. These point-normal pairs
       are the Hermite data.
    5. Solve a regularized least-squares system, similar to a QEF solve,
       to place one representative dual contouring vertex inside the cell.
    """

    # Extract the axis-aligned bounds of the current leaf cell.
    x0, y0, z0 = leaf.bmin
    x1, y1, z1 = leaf.bmax

    # Coordinates of the eight cell corners.
    #
    # The fourth coordinate is set to 1, presumably because the SDF expects
    # homogeneous coordinates or points compatible with 4D transform logic.
    corners = np.array(
        [
            [x0, y0, z0, 1],
            [x0, y0, z1, 1],
            [x0, y1, z0, 1],
            [x0, y1, z1, 1],
            [x1, y0, z0, 1],
            [x1, y0, z1, 1],
            [x1, y1, z0, 1],
            [x1, y1, z1, 1],
        ]
    )

    # Evaluate the signed distance field at the cell corners.
    values = sdf(corners)

    # If all corners lie on the same side of the implicit surface,
    # there is no detected surface crossing in this cell.
    if np.all(values > 0) or np.all(values < 0):
        return None

    # Hermite data consists of:
    # - surface intersection points on cell edges,
    # - estimated surface normals at those points.
    hermite_points = []
    hermite_normals = []

    # Small offset used later when constructing p.
    # Be careful: adding eps * pa biases the point slightly in the pa direction.
    eps = 1e-8

    # Check every cell edge for an SDF sign change.
    for a, b in EDGE_VERTS:
        va = values[a]
        vb = values[b]

        # If the edge endpoints have the same sign, the implicit surface
        # does not cross this edge.
        if va * vb >= 0:
            continue
        # This code uses a few bisection iterations, which can be
        # more robust if the SDF is nonlinear along the edge.

        pa = corners[a].copy()
        pb = corners[b].copy()

        # Refine the edge intersection using bisection.
        #
        # Four iterations gives a coarse approximation. Increasing this
        # number gives more accurate Hermite points at modest cost.
        for _ in range(4):
            pm = 0.5 * (pa + pb)
            vm = sdf(pm[None])[0]

            # Keep the sub-interval that contains the sign change.
            if va * vm <= 0:
                pb = pm
                vb = vm
            else:
                pa = pm
                va = vm

        p = 0.5 * (pa + pb) + eps * pa
        p = p[:, None]
        n = sdf_normal(sdf, p.T)
        hermite_points.append(np.squeeze(p)[:3])
        hermite_normals.append(np.squeeze(n)[:3])

    # If we got here because the corners had mixed signs, at least one edge
    # crossing should normally have been found. If not, something is
    # inconsistent between the signs and edge checks.
    if len(hermite_points) == 0:
        print("ERROR: No hermite points. Exiting")
        exit()
        return

    # Build the linear system for the QEF-like objective:
    #
    #     minimize sum_i (n_i dot x - n_i dot p_i)^2
    #
    # Each Hermite point p_i with normal n_i defines a tangent plane:
    #
    #     n_i dot x = n_i dot p_i
    #
    # The solution x is the point that best fits all tangent planes.
    A = np.squeeze(np.asarray(hermite_normals))
    b = np.asarray([np.dot(n, p) for p, n in zip(hermite_points, hermite_normals)])
    center = np.mean(hermite_points, axis=0)
    lam = 1e-5
    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(3)])
    b_aug = np.concatenate([b, np.sqrt(lam) * center])

    # Solve the regularized least-squares problem.
    # x, *_ = np.linalg.lstsq(A_aug, b_aug)
    res = lsq_linear(A_aug, b_aug, bounds=(leaf.bmin, leaf.bmax))
    x = res.x
    # If the sdf value is large, then the QEF did not work.
    # Instead, we march the vertices, by using the sdf normal
    x = np.array([[x[0], x[1], x[2], 1]])
    x_sdf = sdf(x)

    if abs(x_sdf) > 1e-3:
        # Move the vertices to the closest surface
        # using the sdf normal at that point
        flag_fix_verts[0] += 1
        # x = np.array(hermite_points[0])
        for i in range(4):
            sdf_vals = sdf(x)

            grads = sdf_normal(sdf, x)

            x -= sdf_vals[:, None] * grads  # Newton projection

    # Clamp the dual contouring vertex to the current cell.
    x = x.squeeze()[:3]
    x = np.clip(x, leaf.bmin, leaf.bmax)

    # Store data on the leaf for later visualization or meshing.
    leaf.corner_signs = values
    leaf.hermite_points = hermite_points
    leaf.hermite_normals = hermite_normals
    leaf.vertex = x


def compute_vertices(node, sdf):
    """
    Traverse the octree and compute dual contouring vertices for all leaves.
    """

    # Leaf cell: compute its dual contouring vertex if it contains the surface.
    if node.is_leaf:
        calc_vert(node, sdf)
        return

    # Internal node: recurse into children.
    for child in node.children:
        compute_vertices(
            child,
            sdf,
        )


def self_or_child(node, idx):
    if node.is_leaf:
        return node
    return node.children[idx]


def cellProc(
    node: OctreeNode,
    faces: list,
):
    if node.is_leaf:
        return

    for child in node.children:
        cellProc(child, faces)

    face_pairs = [  # first two indices is choice of nodes. Last idx is coordinate not shared
        (0, 1, 2),
        (0, 2, 1),
        (0, 4, 0),
        (1, 3, 1),
        (1, 5, 0),
        (2, 3, 2),
        (2, 6, 0),
        (3, 7, 0),
        (4, 5, 2),
        (4, 6, 1),
        (5, 7, 1),
        (6, 7, 2),
    ]

    for i, j, k in face_pairs:
        faceProc(node.children[i], node.children[j], faces, k)

    edge_pairs = [
        (0, 1, 3, 2, 0),  # Same x
        (4, 5, 7, 6, 0),  # Same x
        (0, 4, 6, 2, 2),  # Same z
        (1, 5, 7, 3, 2),  # Same z
        (0, 1, 5, 4, 1),  # Same y
        (2, 3, 7, 6, 1),  # Same y
    ]
    for i, j, k, l, h in edge_pairs:
        edgeProc(
            [node.children[i], node.children[j], node.children[k], node.children[l]],
            faces,
            h,
        )
    return faces


def faceProc(node1: OctreeNode, node2: OctreeNode, faces: list, coord: int):
    if node1.is_leaf and node2.is_leaf:
        return

    # Depending on the plane shared, call faceproc
    low = []
    high = []
    edges = []
    nodes = [node1, node2]
    if coord == 0:
        low = [4, 5, 6, 7]
        high = [0, 1, 2, 3]

        edges = [  # Edges are always from (low, low, high, high)
            ((0, 4), (0, 5), (1, 1), (1, 0), 1),
            ((0, 6), (0, 7), (1, 3), (1, 2), 1),
            ((0, 5), (1, 1), (1, 3), (0, 7), 2),
            ((0, 4), (1, 0), (1, 2), (0, 6), 2),
        ]
        # return
    if coord == 1:
        low = [2, 3, 6, 7]
        high = [0, 1, 4, 5]
        edges = [  # Edges are always from (low, low, high, high)
            ((0, 2), (0, 3), (1, 1), (1, 0), 0),
            ((0, 6), (0, 7), (1, 5), (1, 4), 0),
            ((0, 3), (0, 7), (1, 5), (1, 1), 2),
            ((0, 2), (0, 6), (1, 4), (1, 0), 2),
        ]
        # return
    if coord == 2:
        low = [1, 3, 5, 7]
        high = [0, 2, 4, 6]
        edges = [  # Edges are always from (low, low, high, high)
            ((0, 1), (1, 0), (1, 2), (0, 3), 0),
            ((0, 5), (1, 4), (1, 6), (0, 7), 0),
            ((0, 1), (1, 0), (1, 4), (0, 5), 1),
            ((0, 3), (1, 2), (1, 6), (0, 7), 1),
        ]
        # return

    for i, j in zip(low, high):
        faceProc(self_or_child(node1, i), self_or_child(node2, j), faces, coord)
    for i, j, k, l, h in edges:
        edgeProc(
            [
                self_or_child(nodes[i[0]], i[1]),
                self_or_child(nodes[j[0]], j[1]),
                self_or_child(nodes[k[0]], k[1]),
                self_or_child(nodes[l[0]], l[1]),
            ],
            faces,
            h,
        )
    return


def edgeProc(nodes: list(OctreeNode), faces: list, coord: int):
    # If all boxes are not leafs, we must split the edge, and apply edgeproc
    # to the new cubes sharing the two edges.
    if nodes[0].is_leaf and nodes[1].is_leaf and nodes[2].is_leaf and nodes[3].is_leaf:
        for a in nodes:
            if a.vertex is None:
                return
        Generate_polygon(*nodes, faces, coord)
        return

    if coord == 0:
        edge_same_x(*nodes, faces)
    elif coord == 1:
        edge_same_y(*nodes, faces)
    elif coord == 2:
        edge_same_z(*nodes, faces)
    return


def edge_same_x(
    a: OctreeNode, b: OctreeNode, c: OctreeNode, d: OctreeNode, faces: list
):
    edgeProc(
        [
            self_or_child(a, 3),
            self_or_child(b, 2),
            self_or_child(c, 0),
            self_or_child(d, 1),
        ],
        faces,
        0,
    )
    edgeProc(
        [
            self_or_child(a, 7),
            self_or_child(b, 6),
            self_or_child(c, 4),
            self_or_child(d, 5),
        ],
        faces,
        0,
    )
    return


def edge_same_y(
    a: OctreeNode, b: OctreeNode, c: OctreeNode, d: OctreeNode, faces: list
):
    edgeProc(
        [
            self_or_child(a, 7),
            self_or_child(b, 6),
            self_or_child(c, 2),
            self_or_child(d, 3),
        ],
        faces,
        1,
    )
    edgeProc(
        [
            self_or_child(a, 5),
            self_or_child(b, 4),
            self_or_child(c, 0),
            self_or_child(d, 1),
        ],
        faces,
        1,
    )
    return


def edge_same_z(
    a: OctreeNode, b: OctreeNode, c: OctreeNode, d: OctreeNode, faces: list
):
    edgeProc(
        [
            self_or_child(a, 6),
            self_or_child(b, 2),
            self_or_child(c, 0),
            self_or_child(d, 4),
        ],
        faces,
        2,
    )
    edgeProc(
        [
            self_or_child(a, 7),
            self_or_child(b, 3),
            self_or_child(c, 1),
            self_or_child(d, 5),
        ],
        faces,
        2,
    )
    return


def get_edge_corners(a, b, c, d, coord):
    if coord == 0:
        a_corn = [3, 7]
        b_corn = [2, 6]
        c_corn = [0, 4]
        d_corn = [1, 5]
    if coord == 1:
        a_corn = [5, 7]
        b_corn = [4, 6]
        c_corn = [0, 2]
        d_corn = [1, 3]
    if coord == 2:
        a_corn = [6, 7]
        b_corn = [2, 3]
        c_corn = [0, 1]
        d_corn = [4, 5]
    a_corn = [a.corner_signs[x] for x in a_corn]
    b_corn = [b.corner_signs[x] for x in b_corn]
    c_corn = [c.corner_signs[x] for x in c_corn]
    d_corn = [d.corner_signs[x] for x in d_corn]
    corn = [a_corn, b_corn, c_corn, d_corn]
    s = [a, b, c, d]
    s = sorted(range(len(s)), key=lambda k: s[k].depth, reverse=True)
    return corn[s[0]]


def Generate_polygon(
    a: OctreeNode, b: OctreeNode, c: OctreeNode, d: OctreeNode, faces: list, coord: int
):
    # Check if the edge actually has a sign change
    # Note! This does not work when a is bigger than the others!
    edge_corners = get_edge_corners(a, b, c, d, coord)
    if (edge_corners[0] * edge_corners[1]) > 0:
        return

    face1 = [a.vertex_index, b.vertex_index, c.vertex_index]
    face2 = [a.vertex_index, c.vertex_index, d.vertex_index]

    if edge_corners[1] < 0:
        if coord == 1:
            face1[1], face1[2] = face1[2], face1[1]
            face2[1], face2[2] = face2[2], face2[1]
        if coord == 2:
            face1[1], face1[2] = face1[2], face1[1]
            face2[1], face2[2] = face2[2], face2[1]
    elif coord == 0:
        face1[1], face1[2] = face1[2], face1[1]
        face2[1], face2[2] = face2[2], face2[1]
    faces.append(face1)
    faces.append(face2)
    return


def plot_octree(leaves, root, faces, vertices, clouds=None):
    """
    Visualize the octree leaves, input point clouds, and computed dual
    contouring vertices using Plotly.

    Parameters
    ----------
    leaves:
        List of octree leaf nodes.
    root:
        Root node of the octree. Currently unused in the plotting code,
        but kept as an argument for possible future use.
    clouds:
        Optional list of point clouds to draw together with the octree.
    """

    fig = go.Figure()

    # -------------------------
    # Point clouds
    # -------------------------
    #
    # Draw the sampled input geometry used to build the octree.
    if clouds is not None:
        for i, pts in enumerate(clouds):
            if len(pts) == 0:
                continue

            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="markers",
                    marker=dict(
                        size=2,
                    ),
                    name=f"Point Cloud",
                    opacity=0.7,
                )
            )

    # =============================================================
    # =================== Plot octree boxes =======================
    # =============================================================
    x = []
    y = []
    z = []

    for leaf in leaves:
        x0, y0, z0 = leaf.bmin
        x1, y1, z1 = leaf.bmax

        # Eight corners of the current leaf cell.
        corners = np.array(
            [
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z0],
                [x1, y1, z1],
            ]
        )

        # Add all cube edges as line segments.
        for i, j in EDGE_VERTS:
            x.extend([corners[i, 0], corners[j, 0], None])
            y.extend([corners[i, 1], corners[j, 1], None])
            z.extend([corners[i, 2], corners[j, 2], None])

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(
                width=2,
                color="black",
            ),
            name="Octree boxes",
        )
    )

    # =============================================================
    # Plot the dual contouring vertex of each active leaf cell.
    # =============================================================

    verts = []
    for leaf in leaves:
        if leaf.vertex is not None:
            verts.append(leaf.vertex.tolist())
    x = [i[0] for i in verts]
    y = [i[1] for i in verts]
    z = [i[2] for i in verts]

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker=dict(
                size=5,
                color="red",
            ),
            name="Vertices",
        )
    )

    edge_x = []
    edge_y = []
    edge_z = []

    for face in faces:
        n = len(face)

        for i in range(n):
            v0 = vertices[face[i]]
            v1 = vertices[face[(i + 1) % n]]

            edge_x.extend([v0[0], v1[0], None])
            edge_y.extend([v0[1], v1[1], None])
            edge_z.extend([v0[2], v1[2], None])

    fig.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line=dict(color="pink", width=6),
            name="Connectivity",
        )
    )

    # ==========================================================================
    # =============== PLOT THE CENTERS OF ROOT =================================
    # ==========================================================================

    for i, node in enumerate(root.children):
        fig.add_trace(
            go.Scatter3d(
                x=[node.center[0]],
                y=[node.center[1]],
                z=[node.center[2]],
                name=f"{i} in children",
            )
        )

    # Keep the 3D axes scaled equally so the octree cells are not visually
    # distorted.
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        )
    )

    fig.show()


def build_mesh_dual(union_geometries, sdfs, final_sdfs, world_matrices, out_file):
    """
    High-level driver for building an adaptive octree and computing dual
    contouring vertices for each component in the union geometry.

    Current behavior:
    - Generate point samples for each input component.
    - Build an adaptive octree around each component's sampled points.
    - Compute one dual contouring vertex per active leaf cell.
    - Visualize the octree and vertices.
    """

    # Generate point samples from the input geometries/SDFs.
    clouds = generate_points(
        union_geometries,
        sdfs,
        final_sdfs,
        world_matrices,
        n_points=10000,
    )

    # Process each component independently.
    for i, comp in enumerate(union_geometries):
        cloud = clouds[comp.name]
        sdf = final_sdfs[comp.name]

        # Skip components with no sampled points.
        if len(cloud) == 0:
            continue

        points = cloud[:, :3]

        # Compute a tight bounding box around the sampled points.
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        # Expand the bounding box slightly so the surface is not clipped
        # exactly at the sample bounds.
        span = bbox_max - bbox_min
        bbox_min -= 0.05 * span
        bbox_max += 0.05 * span

        # Build an adaptive octree from the point cloud.
        root = build_uniform_octree_from_points(
            points,
            bbox_min,
            bbox_max,
            max_depth=8,
            min_points=5,
        )

        print("Octree built!")

        # Extract all leaf cells where DC vertices may be computed.
        leaves = get_octree_leaves(root)

        print("Leaves gotten!")

        # Evaluate the SDF in each leaf and compute the dual contouring
        # vertex where the implicit surface crosses the cell.
        compute_vertices(root, sdf)
        print(f"Number of vertices to fix is {flag_fix_verts}")
        vertices = []
        for leaf in leaves:
            if leaf.vertex is not None:
                leaf.vertex_index = len(vertices)
                vertices.append(leaf.vertex)
        vertices = np.asarray(vertices)
        #
        print("Vertices calculated!")

        faces = []
        cellProc(root, faces)
        print("Faces calculated!")
        faces = np.asarray(faces, dtype=int)
        if len(vertices) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
            )
            # mesh.fix_normals()

            mesh.export(f"dc_{out_file}_{comp.name}.stl")

            print(f"Exported dual contouring mesh to: {out_file}")
        else:
            print("No mesh exported because vertices or faces are empty.")

        print(
            comp.name,
            "points:",
            len(points),
            "leaves:",
            len(leaves),
            "dc verts:",
            len(vertices),
            "faces:",
            len(faces),
        )

        # Visualize the sampled point cloud, octree cells, and computed
        # dual contouring vertices.
        # plot_octree(
        #     leaves,
        #     root,
        #     faces,
        #     vertices,
        #     [cloud],
        # )
