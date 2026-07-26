from plot_union_cloud import sdf_normal
import trimesh
from dataclasses import dataclass
import numpy as np
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
    return recurse(bbox_min, bbox_max, depth=0, max_depth=max_depth)


def recurse(bmin, bmax, depth, max_depth):
    """
    Recursively subdivide every cell until max_depth.

    Because this is a uniform octree, every node at depth < max_depth
    gets exactly eight children.
    """
    center = 0.5 * (bmin + bmax)

    node = OctreeNode(
        bmin=np.asarray(bmin, dtype=float),
        bmax=np.asarray(bmax, dtype=float),
        center=center,
        depth=depth,
        children=[],
    )
    # Stop once all cells have reached the same target depth.
    if depth >= max_depth:
        return node

    # At each subdivision, the grid index is doubled and offset by
    # the local child octant index.
    #
    # Example:
    # parent index (i, j, k)
    # child index = (2*i + ix, 2*j + iy, 2*k + iz)
    for ix in range(2):
        for iy in range(2):
            for iz in range(2):
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
                child = recurse(child_min, child_max, depth + 1, max_depth)

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
            n.children = [n, n, n, n, n, n, n, n]
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
        for _ in range(3):
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
    center = 0.5 * (leaf.bmin + leaf.bmax)
    lam = 1e-5
    A_aug = np.vstack([A, np.sqrt(lam) * np.eye(3)])
    b_aug = np.concatenate([b, np.sqrt(lam) * center])

    # Solve the regularized least-squares problem.
    x, *_ = np.linalg.lstsq(A_aug, b_aug)
    # If the sdf value is large, then the QEF did not work.
    # Instead, we march the vertices, by using the sdf normal
    x = np.maximum(x[:3], leaf.bmin)
    x = np.minimum(x[:3], leaf.bmax)
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


def evaluate_uniform_grid_sdf(bbox_min, bbox_max, max_depth, sdf):
    """
    Evaluate the SDF at all vertices of the uniform grid.

    If the octree has depth d, then there are:

        n = 2**d

    cells along each axis, and therefore:

        n + 1

    grid vertices along each axis.

    Returns
    -------
    grid_values:
        Array of shape (n + 1, n + 1, n + 1) containing SDF values.
    """

    bbox_min = np.asarray(bbox_min, dtype=float)
    bbox_max = np.asarray(bbox_max, dtype=float)

    n = 2**max_depth

    xs = np.linspace(bbox_min[0], bbox_max[0], n + 1)
    ys = np.linspace(bbox_min[1], bbox_max[1], n + 1)
    zs = np.linspace(bbox_min[2], bbox_max[2], n + 1)

    # Build all grid vertex coordinates.
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    pts = np.column_stack(
        [
            X.ravel(),
            Y.ravel(),
            Z.ravel(),
            np.ones(X.size),
        ]
    )

    values = sdf(pts)

    return values.reshape((n + 1, n + 1, n + 1))


def orient_faces_with_sdf(vertices, faces, sdf):
    """
    Orient triangle faces consistently using the SDF normal.

    For each triangle, we compare:
    - the triangle normal from the current winding,
    - the SDF normal at the triangle centroid.

    If they point in opposite directions, the triangle winding is flipped.

    Assumption
    ----------
    sdf_normal points toward increasing SDF values.

    If your convention is:
        outside = positive SDF
    then this produces outward-facing triangles.

    If your convention is opposite, change the test from dot < 0 to dot > 0.
    """

    oriented_faces = []

    for face in faces:
        i0, i1, i2 = face

        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]

        # Triangle normal from current winding.
        tri_normal = np.cross(p1 - p0, p2 - p0)

        norm = np.linalg.norm(tri_normal)

        tri_normal = tri_normal / norm

        c = (p0 + p1 + p2) / 3.0

        eps = 1e-3

        s_plus = sdf(np.array([[*(c + eps * tri_normal), 1.0]]))

        s_minus = sdf(np.array([[*(c - eps * tri_normal), 1.0]]))
        print(s_plus, s_minus)

        if s_plus[0] < s_minus[0]:
            oriented_faces.append([i0, i2, i1])
        else:
            oriented_faces.append([i0, i1, i2])

    return np.asarray(oriented_faces, dtype=int)


def self_or_child(node, idx):
    if node.is_leaf:
        return node
    return node.children[idx]


def cellProc(node: OctreeNode, faces: list, ):
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
        (0, 2, 6, 4, 2),  # Same z
        (1, 3, 7, 5, 2),  # Same z
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
    # print("In faceProc")
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
            ((0, 5), (0, 7), (1, 3), (1, 1), 2),
            ((0, 4), (0, 6), (1, 2), (1, 0), 2),
        ]
        # return
    if coord == 1:
        low = [2, 3, 6, 7]
        high = [0, 1, 4, 5]
        edges = [  # Edges are always from (low, low, high, high)
            ((0, 2), (0, 3), (1, 1), (1, 0), 0),
            ((0, 6), (0, 7), (1, 5), (1, 4), 0),
            ((0, 3), (1, 1), (1, 5), (0, 7), 2),
            ((0, 2), (1, 0), (1, 4), (0, 6), 2),
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
            self_or_child(b, 4),
            self_or_child(c, 0),
            self_or_child(d, 2),
        ],
        faces,
        2,
    )
    edgeProc(
        [
            self_or_child(a, 7),
            self_or_child(b, 5),
            self_or_child(c, 1),
            self_or_child(d, 3),
        ],
        faces,
        2,
    )
    return


def Generate_polygon(
    a: OctreeNode, b: OctreeNode, c: OctreeNode, d: OctreeNode, faces: list, coord: int
):
    # Check if the edge actually has a sign change and vertices on all nodes
    if coord == 0:
        if (a.corner_signs[3] * a.corner_signs[7]) > 0:
            return
    elif coord == 1:
        if (a.corner_signs[5] * a.corner_signs[7]) > 0:
            return
    elif coord == 2:
        if (a.corner_signs[6] * a.corner_signs[7]) > 0:
            return

    if a.corner_signs[7] > 0:
        faces.append([a.vertex_index, b.vertex_index, c.vertex_index])
        faces.append([a.vertex_index, c.vertex_index, d.vertex_index])
    else:
        faces.append([c.vertex_index, b.vertex_index, a.vertex_index])
        faces.append([d.vertex_index, c.vertex_index, a.vertex_index])
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
        print(node.center)
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
        n_points=5000,
    )

    max_depth = 4
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
            max_depth=max_depth,
            min_points=10,
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

        # faces = build_dual_contouring_faces_uniform(
        #     leaves,
        #     bbox_min,
        #     bbox_max,
        #     max_depth=max_depth,
        #     sdf=sdf,
        # )
        faces = []
        cellProc(root, faces)
        print("Faces calculated!")
        faces = np.asarray(faces, dtype=int)
        # faces = orient_faces_with_sdf(
        #     vertices,
        #     faces,
        #     sdf,
        # )
        print("Faces Oriented!")

        if len(vertices) > 0 and len(faces) > 0:
            mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                process=False,
            )
            mesh.fix_normals()

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
        plot_octree(
            leaves,
            root,
            faces,
            vertices,
            [cloud],
        )
        if i >= 1:
            break
