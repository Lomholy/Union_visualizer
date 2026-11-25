"""The "union" mesher: each Union component's surface, kept only where that
component has the highest priority. A Python port of McStas Union's display
clipping (union_mcdisplay_polyhedron, union_tri_add, union_edge_cuts and
r_has_highest_priority in McCode's union-lib.c), so no solids and no Boolean
operations are involved.

Every shape answers two questions in world space: which points are inside
it, and where segments cross its surface. The clip plane is one more shape,
a box on the removed side that takes priority over every component."""

import re

import numpy as np
import trimesh

from preprocess import box_dimensions
from clipping import clip_plane
from bounding_box import compute_all_world_bboxes, overlapping

# Split budget in half levels: a split in 4 costs 2, a split in 2 costs 1
TRI_DEPTH = 12
# Allowed distance between a straight cut and the true boundary, relative to the triangle size
TRI_CUT_TOL = 0.01
EDGE_MAX_CUTS = 8
EPS = 1e-9
LATHE_SEGMENTS = 32
CLIP_BOX_SIZE = 10000.0


# ------------------------------- shapes ---------------------------------------


class Grid:
    """Boxes binned once into cells of size h; query() pairs other boxes with them."""

    def __init__(self, lo, hi, h):
        self.lo, self.hi, self.h = lo, hi, h
        self.origin = lo.min(0) - h
        box, key = self._cells(lo, hi)
        order = np.argsort(key, kind="stable")
        self.key, self.box = key[order], box[order]

    def _cells(self, lo, hi):
        i0 = np.floor((lo - self.origin) / self.h).astype(np.int64)
        i1 = np.floor((hi - self.origin) / self.h).astype(np.int64)
        ext = i1 - i0 + 1
        n = ext.prod(1)
        box = np.repeat(np.arange(len(lo)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        key = np.zeros(len(box), np.int64)
        for d in range(lo.shape[1] - 1, -1, -1):
            e = ext[box, d]
            key += (i0[box, d] + k % e) * 1_000_003 ** (lo.shape[1] - 1 - d)
            k //= e
        return box, key

    def query(self, lo, hi):
        box, key = self._cells(lo, hi)
        start = np.searchsorted(self.key, key, "left")
        n = np.searchsorted(self.key, key, "right") - start
        i = np.repeat(box, n)
        j = self.box[np.repeat(start, n) + np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)]
        if len(i) == 0:
            return i, j
        nb = len(self.lo)
        pair = np.unique(i * nb + j)
        i, j = pair // nb, pair % nb
        keep = np.all((lo[i] <= self.hi[j]) & (self.lo[j] <= hi[i]), axis=1)
        return i[keep], j[keep]


def ray_triangle(origin, direction, tri, t_max):
    """Moller-Trumbore for paired rays and triangles: (hit mask, t)."""
    v0 = tri[:, 0]
    e1, e2 = tri[:, 1] - v0, tri[:, 2] - v0
    p = np.cross(direction, e2)
    det = (e1 * p).sum(1)
    ok = np.abs(det) > 1e-300
    f = np.where(ok, 1 / np.where(ok, det, 1), 0)
    s = origin - v0
    u = f * (s * p).sum(1)
    q = np.cross(s, e1)
    v = f * (direction * q).sum(1)
    t = f * (e2 * q).sum(1)
    return ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0) & (t < t_max), t


def orient_outward(tri, center):
    """Flip triangles of a convex surface whose normal points towards center."""
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = (normal * (tri.mean(1) - center)).sum(1) < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    return tri


class Shape:
    """A shape placed by a 4x4 world matrix; local frames follow the McStas components.
    triangles() keeps McStas's vertex order, which the clipping depends on; a convex
    shape's output is turned outward afterwards."""

    depth = TRI_DEPTH
    convex = True

    def __init__(self, matrix):
        self.rot = np.asarray(matrix, float)[:3, :3]
        self.center = np.asarray(matrix, float)[:3, 3]

    def to_local(self, p):
        return (p - self.center) @ self.rot

    def to_world(self, p):
        return p @ self.rot.T + self.center


class MeshShape(Shape):
    convex = False

    def __init__(self, mesh):
        super().__init__(np.eye(4))
        self.mesh = mesh
        self.tri = mesh.triangles
        lo, hi = self.tri.min(1), self.tri.max(1)
        h = 2 * np.median(hi - lo) + 1e-12
        self.grid3 = Grid(lo, hi, h)
        self.grid_yz = Grid(lo[:, 1:], hi[:, 1:], h)

    def triangles(self):
        return self.tri.copy()

    def contains(self, p):
        """Parity of the +x ray from each point."""
        inside = np.zeros(len(p), bool)
        i, j = self.grid_yz.query(p[:, 1:], p[:, 1:])
        if len(i):
            direction = np.broadcast_to([1.0, 0.0, 0.0], (len(i), 3))
            hit, _ = ray_triangle(p[i], direction, self.tri[j], np.inf)
            np.logical_xor.at(inside, i[hit], True)
        return inside

    def crossings(self, a, b):
        i, j = self.grid3.query(np.minimum(a, b), np.maximum(a, b))
        if len(i) == 0:
            return i, np.zeros(0)
        hit, t = ray_triangle(a[i], b[i] - a[i], self.tri[j], 1.0)
        return i[hit], t[hit]


class ConvexShape(Shape):
    """Intersection of the half-spaces normal . p <= offset (world space)."""

    def __init__(self, normals, offsets):
        self.normals, self.offsets = normals, offsets

    def contains(self, p):
        return np.all(p @ self.normals.T <= self.offsets, axis=1)

    def crossings(self, a, b):
        d = (b - a) @ self.normals.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self.offsets - a @ self.normals.T) / d
        seg, plane = np.nonzero((d != 0) & (t > 0) & (t < 1))
        return seg, t[seg, plane]


class BoxShape(ConvexShape):
    """Union_box, tapered by xwidth2/yheight2 towards its +z face."""

    depth = TRI_DEPTH + 4

    def __init__(self, comp, matrix):
        Shape.__init__(self, matrix)
        x1, y1, x2, y2 = box_dimensions(comp)
        z = 0.5 * float(comp.zdepth)
        local = np.array([[sx * w / 2, sy * h / 2, sz]
                          for sz, w, h in ((-z, x1, y1), (z, x2, y2))
                          for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
        s = self.to_world(local).reshape(2, 4, 3)
        # two triangles per face, in McStas's order, so tri[::2] holds one per face
        tri = []
        for j in range(2):
            tri += [(s[j][0], s[j][1], s[j][2]), (s[j][0], s[j][2], s[j][3])]
        for i in range(4):
            tri.append((s[0][i], s[0][(i + 1) % 4], s[1][(i + 1) % 4]))
            tri.append((s[0][i], s[1][(i + 1) % 4], s[1][i]))
        self.tri = np.array(tri)
        face = self.tri[::2]
        normals = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals *= np.sign((normals * (face.mean(1) - self.center)).sum(1))[:, None]
        super().__init__(normals, (normals * face[:, 0]).sum(1))

    def triangles(self):
        return self.tri.copy()


class ClipBox(ConvexShape):
    """The clipped side of the clip plane, CLIP_BOX_SIZE along the plane and across it."""

    def __init__(self, clip):
        normal, point = clip_plane(clip)
        u = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        normals = np.array([normal, -normal, u, -u, v, -v])
        half = CLIP_BOX_SIZE / 2
        offsets = normals @ point + np.array([0, CLIP_BOX_SIZE, half, half, half, half])
        super().__init__(normals, offsets)


class LatheShape(Shape):
    """Union_cylinder, Union_cone and Union_sphere: surfaces of revolution about local y,
    triangulated as McStas's union_tri_lathe does."""

    def __init__(self, comp, matrix):
        super().__init__(matrix)
        if comp.component_name == "Union_sphere":
            self.radius = float(comp.radius)
            k = np.arange(17)
            self.profile_y = -self.radius * np.cos(np.pi * k / 16)
            self.profile_r = np.where((k == 0) | (k == 16), 0.0, self.radius * np.sin(np.pi * k / 16))
            return
        self.radius = None
        self.half = 0.5 * float(comp.yheight)
        if comp.component_name == "Union_cylinder":
            r_bottom = r_top = float(comp.radius)
        else:
            r_bottom, r_top = float(comp.radius_bottom), float(comp.radius_top)
        # radius along local y: k0 + k1 * y
        self.k1 = (r_top - r_bottom) / (2 * self.half)
        self.k0 = r_bottom + self.k1 * self.half
        self.profile_y = np.array([-self.half, self.half])
        self.profile_r = np.array([r_bottom, r_top])

    def triangles(self):
        y, r, n = self.profile_y, self.profile_r, LATHE_SEGMENTS
        angle = 2 * np.pi * np.arange(n + 1) / n
        # McStas's point_on_circle: rings start at axis x (0,1,0), or axis x (1,0,0) near y;
        # a sphere's axis is the world y axis whatever its rotation
        axis = np.array([0.0, 1.0, 0.0]) if self.radius is not None else self.rot[:, 1]
        e1 = np.cross(axis, [1.0, 0.0, 0.0] if axis[1] > 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)

        def ring(k, i):
            return (y[k] * axis + r[k] * (np.cos(angle[i])[:, None] * e1 + np.sin(angle[i])[:, None] * e2))

        i, i1 = np.arange(n), np.arange(1, n + 1)
        tri = []
        for k in range(len(y) - 1):
            p00, p01, p10, p11 = ring(k, i), ring(k, i1), ring(k + 1, i), ring(k + 1, i1)
            if r[k] > 0:
                tri.append(np.stack([p00, p01, p11], 1))
            if r[k + 1] > 0:
                tri.append(np.stack([p00, p11, p10], 1))
        for k, (first, second) in ((0, (i1, i)), (len(y) - 1, (i, i1))):
            if r[k] > 0:
                axis_point = np.tile(y[k] * axis, (n, 1))
                tri.append(np.stack([axis_point, ring(k, first), ring(k, second)], 1))
        return np.concatenate(tri) + self.center

    def contains(self, p):
        q = self.to_local(p)
        if self.radius is not None:
            return (q * q).sum(1) <= self.radius ** 2
        r = self.k0 + self.k1 * q[:, 1]
        return (np.abs(q[:, 1]) <= self.half) & (q[:, 0] ** 2 + q[:, 2] ** 2 <= r * r)

    def crossings(self, a, b):
        """Crossings with the side surface between the caps, and with the cap discs."""
        qa, qd = self.to_local(a), self.to_local(b) - self.to_local(a)
        if self.radius is not None:
            A = (qd * qd).sum(1)
            B = 2 * (qa * qd).sum(1)
            C = (qa * qa).sum(1) - self.radius ** 2
            roots = [*quadratic_roots(A, B, C)]
        else:
            ra = self.k0 + self.k1 * qa[:, 1]
            A = qd[:, 0] ** 2 + qd[:, 2] ** 2 - (self.k1 * qd[:, 1]) ** 2
            B = 2 * (qa[:, 0] * qd[:, 0] + qa[:, 2] * qd[:, 2] - self.k1 * qd[:, 1] * ra)
            C = qa[:, 0] ** 2 + qa[:, 2] ** 2 - ra * ra
            roots = []
            with np.errstate(divide="ignore", invalid="ignore"):
                for t in quadratic_roots(A, B, C):
                    roots.append(np.where(np.abs(qa[:, 1] + t * qd[:, 1]) <= self.half, t, np.nan))
                for s in (-1, 1):
                    t = (s * self.half - qa[:, 1]) / qd[:, 1]
                    p = qa + t[:, None] * qd
                    r = self.k0 + self.k1 * s * self.half
                    roots.append(np.where(p[:, 0] ** 2 + p[:, 2] ** 2 <= r * r, t, np.nan))
        t = np.stack(roots, 1)
        seg, k = np.nonzero(np.isfinite(t) & (t > 0) & (t < 1))
        return seg, t[seg, k]


def quadratic_roots(A, B, C):
    """Both real roots of A t^2 + B t + C (NaN where none), the linear root where A is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = B * B - 4 * A * C
        sq = np.sqrt(np.where(disc >= 0, disc, np.nan))
        linear = np.abs(A) < 1e-14 * (np.abs(B) + np.abs(C) + 1e-300)
        t1 = np.where(linear, -C / B, (-B - sq) / (2 * A))
        t2 = np.where(linear, np.nan, (-B + sq) / (2 * A))
    return t1, t2


def make_shape(comp, matrix):
    """The shape of a Union component, or None for a type the mesher does not know."""
    kind = comp.component_name
    if kind == "Union_box":
        return BoxShape(comp, matrix)
    if kind in ("Union_cylinder", "Union_cone", "Union_sphere"):
        return LatheShape(comp, matrix)
    if kind == "Union_mesh":
        mesh = trimesh.load_mesh(comp.filename.strip('"'))
        if comp.coordinate_scale is None:
            comp.coordinate_scale = 1e-3
        mesh.apply_scale(float(comp.coordinate_scale))
        mesh.apply_transform(matrix)
        return MeshShape(mesh)
    return None


# -------------------------------- masks ---------------------------------------


def _unquote(value):
    return str(value).strip().strip('"').strip("'").strip()


def is_mask(comp):
    value = getattr(comp, "mask_string", None)
    return value is not None and _unquote(value) not in ("", "0", "NULL")


def masks_of(name, union_geometries):
    """The masks naming this component, matched by whole name as McStas does."""
    return [
        c for c in union_geometries.values()
        if is_mask(c) and name in re.split(r"[\s,]+", _unquote(c.mask_string))
    ]


def mask_mode_any(masks):
    """McStas's rule: ALL unless any of the masks asks for ANY."""
    return any(_unquote(getattr(m, "mask_setting", None) or "").upper() == "ANY" for m in masks)


class Masked:
    """A shape together with the masks that limit it (no masks: the shape alone)."""

    def __init__(self, shape, masks, any_mode):
        self.shape, self.masks, self.any_mode = shape, masks, any_mode

    def mask_status(self, p):
        inside = [m.contains(p) for m in self.masks]
        return np.any(inside, 0) if self.any_mode else np.all(inside, 0)

    def contains(self, p):
        inside = self.shape.contains(p)
        if self.masks and inside.any():
            idx = np.flatnonzero(inside)
            inside[idx] = self.mask_status(p[idx])
        return inside


# ------------------------------- clipping -------------------------------------


class Clipper:
    """union_tri_add for one component against everything that can hide it."""

    def __init__(self, own_masks, cutters, sources):
        self.own_masks = own_masks
        self.cutters = cutters
        self.sources = sources
        self.out = []

    def highest(self, p):
        """r_has_highest_priority: in the component's masks, and inside no cutter."""
        p = np.atleast_2d(p)
        ok = np.ones(len(p), bool)
        if self.own_masks is not None:
            ok = self.own_masks.mask_status(p)
        for cutter in self.cutters:
            idx = np.flatnonzero(ok)
            if len(idx) == 0:
                break
            ok[idx] = ~cutter.contains(p[idx])
        return ok

    def edge_cuts(self, a, b):
        """union_edge_cuts per segment: (fractions where the status changes, status at a)."""
        n = len(a)
        seg, ts = [np.arange(n), np.arange(n)], [np.zeros(n), np.ones(n)]
        for shape in self.sources:
            i, t = shape.crossings(a, b)
            seg.append(i)
            ts.append(t)
        seg, ts = np.concatenate(seg), np.concatenate(ts)
        order = np.lexsort((ts, seg))
        seg, ts = seg[order], ts[order]
        same = seg[1:] == seg[:-1]
        s0, s1, sid = ts[:-1][same], ts[1:][same], seg[:-1][same]
        keep = self.highest(a[sid] + (b[sid] - a[sid]) * (0.5 * (s0 + s1))[:, None])
        cuts = [[] for _ in range(n)]
        start = np.zeros(n, bool)
        bounds = np.flatnonzero(np.diff(np.concatenate([[-1], sid, [n]])))
        single = np.diff(bounds) == 1
        start[sid[bounds[:-1][single]]] = keep[bounds[:-1][single]]
        # Kept pieces that touch merge into intervals; their inner ends are the cuts
        for k in np.flatnonzero(~single):
            s = sid[bounds[k]]
            x, y = 0.0, -1.0
            for m in range(bounds[k], bounds[k + 1] + 1):
                last = m == bounds[k + 1]
                if not last and not keep[m]:
                    continue
                if not last and y >= 0 and s0[m] - y <= EPS:
                    y = s1[m]
                    continue
                if y >= 0:
                    if x < EPS:
                        start[s] = True
                    else:
                        cuts[s].append(x)
                    if y < 1 - EPS:
                        cuts[s].append(y)
                if not last:
                    x, y = s0[m], s1[m]
        return cuts, start

    def emit(self, a, b, c):
        self.out.append((a, b, c))

    def add(self, a, b, c, depth, pre=None):
        """union_tri_add; pre = (cuts, start) per edge when already known."""
        v = (a, b, c)
        if depth <= TRI_DEPTH:
            cuts, st = pre or self.edge_cuts(np.array(v), np.array((b, c, a)))
            n = [len(x) for x in cuts]
            total = sum(n)
            if total == 0:
                if depth <= 1 or self.highest((a + b + c) / 3)[0] == st[0]:
                    if st[0]:
                        self.emit(a, b, c)
                    return
            elif total == 2 and max(n) <= 1:
                k = next(k for k in range(3) if n[k] == 1 and n[(k + 2) % 3] == 1)
                w1, w2 = v[(k + 1) % 3], v[(k + 2) % 3]
                sp, sq = cuts[k][0], 1 - cuts[(k + 2) % 3][0]
                p = v[k] + (w1 - v[k]) * sp
                q = v[k] + (w2 - v[k]) * sq
                # a straight boundary crosses the median v[k] -> (w1+w2)/2 at 2 sp sq / (sp + sq)
                m = 0.5 * (w1 + w2)
                size = max(np.linalg.norm(w1 - v[k]), np.linalg.norm(w2 - v[k]), np.linalg.norm(w2 - w1))
                (sm,), _ = self.edge_cuts(v[k][None], m[None])
                median = np.linalg.norm(m - v[k])
                if depth == 0 or (len(sm) == 1 and abs(sm[0] - 2 * sp * sq / (sp + sq)) * median < TRI_CUT_TOL * size):
                    if st[k]:
                        self.emit(v[k], p, q)
                    else:
                        self.emit(p, w1, w2)
                        self.emit(p, w2, q)
                    return
            elif total == 1:
                # boundary from the vertex o opposite the cut edge (left on the boundary by a split in 2)
                k = next(k for k in range(3) if n[k] == 1)
                o, w1, w2 = v[(k + 2) % 3], v[k], v[(k + 1) % 3]
                p = w1 + (w2 - w1) * cuts[k][0]
                m1, m2 = 0.5 * (w1 + o), 0.5 * (w2 + o)
                size = max(np.linalg.norm(w2 - w1), np.linalg.norm(o - w1), np.linalg.norm(o - w2))
                (sm,), _ = self.edge_cuts(m1[None], m2[None])
                midline = 0.5 * np.linalg.norm(w2 - w1)
                if depth == 0 or (len(sm) == 1 and abs(sm[0] - cuts[k][0]) * midline < TRI_CUT_TOL * size):
                    if st[k]:
                        self.emit(w1, p, o)
                    else:
                        self.emit(p, w2, o)
                    return
            elif depth > 0:
                # split in 2 through the crossing nearest the middle of the most crossed edge
                e = 0
                for k in (1, 2):
                    if n[k] > n[e]:
                        e = k
                best = 0
                for i in range(1, min(n[e], EDGE_MAX_CUTS)):
                    if abs(cuts[e][i] - 0.5) < abs(cuts[e][best] - 0.5):
                        best = i
                x = v[e] + (v[(e + 1) % 3] - v[e]) * cuts[e][best]
                self.add(v[e], x, v[(e + 2) % 3], depth - 1)
                self.add(x, v[(e + 1) % 3], v[(e + 2) % 3], depth - 1)
                return
            if depth <= 1:
                if self.highest((a + b + c) / 3)[0]:
                    self.emit(a, b, c)
                return
        ab, bc, ca = 0.5 * (a + b), 0.5 * (b + c), 0.5 * (c + a)
        self.add(a, ab, ca, depth - 2)
        self.add(ab, b, bc, depth - 2)
        self.add(ca, bc, c, depth - 2)
        self.add(ab, bc, ca, depth - 2)

    def run(self, tri, depth):
        """Clip triangles (T, 3, 3) at the given split budget. The first level runs
        vectorized over the shared edges; only triangles a boundary touches recurse."""
        while depth > TRI_DEPTH:
            ab, bc, ca = (tri[:, 0] + tri[:, 1]) / 2, (tri[:, 1] + tri[:, 2]) / 2, (tri[:, 2] + tri[:, 0]) / 2
            tri = np.concatenate([
                np.stack([tri[:, 0], ab, ca], 1), np.stack([ab, tri[:, 1], bc], 1),
                np.stack([ca, bc, tri[:, 2]], 1), np.stack([ab, bc, ca], 1),
            ])
            depth -= 2
        if not self.sources and self.own_masks is None:
            return tri
        mesh = trimesh.Trimesh(tri.reshape(-1, 3), np.arange(3 * len(tri)).reshape(-1, 3), process=False)
        mesh.merge_vertices()
        faces, edges, fe = mesh.faces, mesh.edges_unique, mesh.faces_unique_edges
        cuts, st = self.edge_cuts(mesh.vertices[edges[:, 0]], mesh.vertices[edges[:, 1]])
        n_cuts = np.array([len(c) for c in cuts])
        forward = edges[fe, 0] == faces
        # status at the start vertex of each face edge
        st_face = np.where(forward, st[fe], st[fe] ^ (n_cuts[fe] % 2 == 1))
        centroid_ok = self.highest(tri.mean(1))
        easy = (n_cuts[fe].sum(1) == 0) & (centroid_ok == st_face[:, 0])
        kept = tri[easy & st_face[:, 0]]
        for f in np.flatnonzero(~easy):
            pre = [cuts[fe[f, k]] if forward[f, k] else sorted(1 - x for x in cuts[fe[f, k]]) for k in range(3)]
            self.add(*tri[f], depth, pre=(pre, list(st_face[f])))
        return np.concatenate([kept, np.array(self.out).reshape(-1, 3, 3)])


def _masked(comp, shapes, union_geometries):
    masks = [m for m in masks_of(comp.name, union_geometries) if shapes.get(m.name) is not None]
    return Masked(shapes[comp.name], [shapes[m.name] for m in masks], mask_mode_any(masks))


def clip_component(comp, union_geometries, shapes, world_bboxes, clip):
    """Triangles (T, 3, 3) of comp's surface where it has the highest priority."""
    candidates = [
        c.name for c in union_geometries.values()
        if not is_mask(c) and shapes.get(c.name) is not None and c.priority > comp.priority
    ]
    cutters = [_masked(union_geometries[n], shapes, union_geometries)
               for n in overlapping(comp.name, candidates, world_bboxes)]
    own = _masked(comp, shapes, union_geometries)
    sources = list(own.masks)
    for cutter in cutters:
        sources += [cutter.shape, *cutter.masks]
    if clip.get("enable"):
        box = ClipBox(clip)
        cutters.append(Masked(box, [], False))
        sources.append(box)
    clipper = Clipper(own if own.masks else None, cutters, sources)
    tri = clipper.run(own.shape.triangles(), own.shape.depth)
    return orient_outward(tri, own.shape.center) if own.shape.convex else tri


def _to_mesh(tri):
    if len(tri) == 0:
        return None
    mesh = trimesh.Trimesh(tri.reshape(-1, 3), np.arange(3 * len(tri)).reshape(-1, 3), process=False)
    mesh.merge_vertices()
    return mesh


def build_shapes(union_geometries, world_matrices):
    shapes = {}
    for name, comp in union_geometries.items():
        shapes[name] = make_shape(comp, world_matrices[name])
        if shapes[name] is None:
            print(f"Warning: the union mesher skips {name} ({comp.component_name}), an unknown Union shape.")
    return shapes


def build_union_meshes(union_geometries, world_matrices, clip, verbose=False, world_bboxes=None, names=None):
    """{name: trimesh or None} for names (default: every component). Masks, unknown
    shapes and fully hidden components map to None."""
    if world_bboxes is None:
        world_bboxes = compute_all_world_bboxes(
            {n: c for n, c in union_geometries.items() if c.component_name in KNOWN_SHAPES},
            world_matrices,
        )
    shapes = build_shapes(union_geometries, world_matrices)
    meshes = {}
    for name in union_geometries if names is None else names:
        comp = union_geometries[name]
        if is_mask(comp) or shapes[name] is None:
            if verbose and is_mask(comp):
                print(f"{name} is a mask, and is therefore not meshed.")
            meshes[name] = None
            continue
        meshes[name] = _to_mesh(clip_component(comp, union_geometries, shapes, world_bboxes, clip))
    return meshes


KNOWN_SHAPES = ("Union_box", "Union_cylinder", "Union_cone", "Union_sphere", "Union_mesh")
