from plot_union_cloud import sdf_normal
from dataclasses import dataclass
import numpy as np
import trimesh
from plot_union_cloud import generate_points
import plotly.graph_objects as go


@dataclass
class OctreeNode:
    bmin: np.ndarray
    bmax: np.ndarray
    depth: int
    children: list
    corner_signs: np.ndarray | None = None
    hermite_points: list | None = None
    hermite_normals: list | None = None
    vertex: np.ndarray | None = None


def build_octree_from_points(
    points,
    bbox_min,
    bbox_max,
    max_depth=6,
    min_points=20,
):
    points = points[:, :3]

    def recurse(points, bmin, bmax, depth):
        if len(points) == 0:
            return None

        node = OctreeNode(
            bmin=np.asarray(bmin),
            bmax=np.asarray(bmax),
            depth=depth,
            children=[],
        )

        if depth >= max_depth:
            return node

        if len(points) <= min_points:
            return node

        center = 0.5 * (bmin + bmax)

        for ix in range(2):
            for iy in range(2):
                for iz in range(2):
                    child_min = np.array(
                        [
                            bmin[0] if ix == 0 else center[0],
                            bmin[1] if iy == 0 else center[1],
                            bmin[2] if iz == 0 else center[2],
                        ]
                    )

                    child_max = np.array(
                        [
                            center[0] if ix == 0 else bmax[0],
                            center[1] if iy == 0 else bmax[1],
                            center[2] if iz == 0 else bmax[2],
                        ]
                    )

                    mask = np.all(
                        (points[:, :3] >= child_min) & (points[:, :3] <= child_max),
                        axis=1,
                    )

                    child_points = points[mask]

                    child = recurse(
                        child_points,
                        child_min,
                        child_max,
                        depth + 1,
                    )

                    if child is not None:
                        node.children.append(child)

        return node

    return recurse(
        points,
        np.asarray(bbox_min),
        np.asarray(bbox_max),
        0,
    )


def get_octree_leaves(node):
    leaves = []

    def recurse(n):
        if n is None:
            return

        if len(n.children) == 0:
            leaves.append(n)
            return

        for child in n.children:
            recurse(child)

    recurse(node)

    return leaves


EDGE_VERTS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def calc_vert(leaf, sdf):
    x0, y0, z0 = leaf.bmin
    x1, y1, z1 = leaf.bmax

    corners = np.array(
        [
            [x0, y0, z0, 1],
            [x1, y0, z0, 1],
            [x1, y1, z0, 1],
            [x0, y1, z0, 1],
            [x0, y0, z1, 1],
            [x1, y0, z1, 1],
            [x1, y1, z1, 1],
            [x0, y1, z1, 1],
        ]
    )
    values = sdf(corners)
    if np.all(values > 0) or np.all(values < 0):
        return None  # No sign change

    hermite_points = []
    hermite_normals = []
    eps = 1e-8

    for a, b in EDGE_VERTS:
        va = values[a]
        vb = values[b]

        if va * vb >= 0:
            continue

        # t = va / (va - vb)

        # p = corners[a] + t * (corners[b] - corners[a])
        pa = corners[a].copy()
        pb = corners[b].copy()

        for _ in range(4):
            pm = 0.5 * (pa + pb)

            vm = sdf(pm[None])[0]

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

    if len(hermite_points) == 0:
        print("ERROR: No hermite points. Exiting")
        exit()
        return
    # Append a point in the middle that pulls the vert towards the center
    # BIAS_STRENGTH = 1e-6
    # mass_point = np.mean(hermite_points, axis=0)
    # hermite_normals.append([BIAS_STRENGTH, 0, 0])
    # hermite_points.append(mass_point)
    # hermite_normals.append([0, BIAS_STRENGTH, 0])
    # hermite_points.append(mass_point)
    # hermite_normals.append([0, 0, BIAS_STRENGTH])
    # hermite_points.append(mass_point)

    A = np.squeeze(np.asarray(hermite_normals))
    b = np.asarray([np.dot(n, p) for p, n in zip(hermite_points, hermite_normals)])

    center = 0.5 * (leaf.bmin + leaf.bmax)

    lam = 1e-3

    A_aug = np.vstack(
        [
            A,
            np.sqrt(lam) * np.eye(3),
        ]
    )

    b_aug = np.concatenate(
        [
            b,
            np.sqrt(lam) * center,
        ]
    )

    x, *_ = np.linalg.lstsq(
        A_aug,
        b_aug,
        rcond=None,
    )

    x = np.maximum(x, leaf.bmin)
    x = np.minimum(x, leaf.bmax)

    leaf.corner_values = values
    leaf.hermite_points = hermite_points
    leaf.hermite_normals = hermite_normals
    leaf.dc_vertex = x


def compute_vertices(node, sdf):
    if len(node.children) == 0:
        calc_vert(node, sdf)
        return

    for child in node.children:
        compute_vertices(
            child,
            sdf,
        )


def plot_octree(leaves, root, clouds=None):
    fig = go.Figure()

    # -------------------------
    # Point clouds
    # -------------------------
    #
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
                    name=f"Cloud {i}",
                    opacity=0.7,
                )
            )

    # -------------------------
    # Octree cells
    # -------------------------

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    x = []
    y = []
    z = []
    for leaf in leaves:
        x0, y0, z0 = leaf.bmin
        x1, y1, z1 = leaf.bmax

        corners = np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x1, y1, z0],
                [x0, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x1, y1, z1],
                [x0, y1, z1],
            ]
        )

        for i, j in edges:
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
            showlegend=False,
        )
    )
    hp = []
    hn = []

    for leaf in leaves:
        if hasattr(leaf, "hermite_points") and leaf.hermite_points is not None:
            hp.extend(leaf.hermite_points)
            hn.extend(leaf.hermite_normals)

    for leaf in leaves:
        if hasattr(leaf, "dc_vertex"):
            # print(leaf.dc_vertex[0])
            fig.add_trace(
                go.Scatter3d(
                    x=[leaf.dc_vertex[0]],
                    y=[leaf.dc_vertex[1]],
                    z=[leaf.dc_vertex[2]],
                    mode="markers",
                    marker=dict(
                        size=5,
                        color="red",
                    ),
                    showlegend=False,
                )
            )
            # break

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
    clouds = generate_points(
        union_geometries,
        sdfs,
        final_sdfs,
        world_matrices,
        n_points=5000,
    )
    for i, comp in enumerate(union_geometries):
        cloud = clouds[comp.name]
        sdf = final_sdfs[comp.name]
        if len(cloud) == 0:
            continue

        points = cloud[:, :3]

        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)

        span = bbox_max - bbox_min

        bbox_min -= 0.05 * span
        bbox_max += 0.05 * span

        root = build_octree_from_points(
            points,
            bbox_min,
            bbox_max,
            max_depth=7,
            min_points=50,
        )
        print("Octree built!")
        leaves = get_octree_leaves(root)

        print("Leaves gotten!")
        compute_vertices(root, sdf)
        print("Vertices calculated!")
        print(
            comp.name,
            "points:",
            len(points),
            "leaves:",
            len(leaves),
            "dc verts:",
        )

        plot_octree(
            leaves,
            root,
            [cloud],
        )
