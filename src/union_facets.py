"""Closed cuts on faceted shapes: a component's kept surface and the walls that
close it, cut along one shared set of vertices so the result is watertight.

Every shape is its faceted display polyhedron: convex planar facets for boxes and
lathes, triangles for meshes; the clip box is half-spaces only. Coincident planes
share one plane id. A vertex is known by the planes it lies on, or by its position
where they do not fix a point, so a cut surface and the wall meeting it are built
from the same vertices; each vertex is classified against each plane once."""

from collections import defaultdict
from itertools import chain, combinations

import numpy as np
from numba import njit, typed, types
from numba.extending import intrinsic

# distances relative to the own shape's size
ON_PLANE = 1e-9
SAME_POINT = 1e-8
# largest 1 - |cos| between the normals of coincident planes
PARALLEL = 1e-12


class Kernel:
    def __init__(self, scale):
        self.eps, self.merge = ON_PLANE * scale, SAME_POINT * scale
        # a vertex gains a label on a plane only within merge of it, so its sign against a
        # plane it is farther than this from and not labelled on never changes
        self.margin = 2 * self.merge
        self.normal, self.offset = np.zeros((0, 3)), np.zeros(0)
        self.pos, self.on = [], []
        # pos as one array, rows past len(pos) unused
        self.xyz = np.zeros((256, 3))
        self.ids, self.lines, self.signs = {}, defaultdict(set), {}
        # by position, in cells of size merge: the cell's newest vertex id, and each vertex's
        # next older one in its cell (-1 at the end)
        self.grid = typed.Dict.empty(types.UniTuple(types.int64, 3), types.int64)
        self.older = np.full(256, -1)

    def planes(self, normal, offset):
        """(plane ids, sides) for planes normal . p = offset; side is -1 where the
        stored plane of that id faces the other way."""
        # coincident planes within the call: normals turned to point along their largest component
        flip = np.where(np.take_along_axis(normal, np.abs(normal).argmax(1)[:, None], 1)[:, 0] < 0, -1, 1)
        rounded = np.column_stack([np.round(normal * flip[:, None] * 1e8), np.round(offset * flip / self.eps)])
        _, first, group = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
        group = group.ravel()
        ids, sides = self.new_planes(normal[first], offset[first])
        return ids[group], sides[group] * flip * flip[first][group]

    def new_planes(self, normal, offset):
        ids, sides = np.zeros(len(offset), int), np.ones(len(offset), int)
        fresh = np.ones(len(offset), bool)
        if len(self.offset):
            hit, old, dot = self.coincident(normal, offset, self.normal, self.offset)
            fresh[hit] = False
            ids[hit] = old
            sides[hit] = np.sign(dot)
        # fresh planes coincident with an earlier fresh one take its id
        f = np.flatnonzero(fresh)
        rep, side = np.arange(len(f)), np.ones(len(f), int)
        for i, j, d in zip(*self.coincident(normal[f], offset[f], normal[f], offset[f], earlier=True)):
            rep[i], side[i] = rep[j], side[j] * np.sign(d)
        own = rep == np.arange(len(f))
        ids[f] = (len(self.offset) + np.cumsum(own) - 1)[rep]
        sides[f] = side
        self.normal = np.vstack([self.normal, normal[f[own]]])
        self.offset = np.concatenate([self.offset, offset[f[own]]])
        return ids, sides

    def coincident(self, normal, offset, s_normal, s_offset, earlier=False):
        """(i, j, dot): each plane i matched to the first plane j of the s set coincident with
        it (j < i only if earlier), and the dot product of their normals."""
        # coincident planes have |offset| within eps: pair each plane with the s planes in
        # that band of the |offset|-sorted list
        # ponytail: many planes at one |offset| make the band O(n*m), a normal hash if that bites
        order = np.argsort(np.abs(s_offset), kind="stable")
        size = np.abs(s_offset[order])
        lo = np.searchsorted(size, np.abs(offset) - self.eps, "left")
        hi = np.searchsorted(size, np.abs(offset) + self.eps, "right")
        count = hi - lo
        new = np.repeat(np.arange(len(offset)), count)
        old = order[np.arange(count.sum()) - np.repeat(np.cumsum(count) - count - lo, count)]
        dot = np.einsum("ij,ij->i", normal[new], s_normal[old])
        same = (1 - np.abs(dot) < PARALLEL) & (np.abs(offset[new] - np.sign(dot) * s_offset[old]) < self.eps)
        if earlier:
            same &= old < new
        new, old, dot = new[same], old[same], dot[same]
        first = np.lexsort((old, new))
        new, old, dot = new[first], old[first], dot[first]
        keep = np.r_[True, new[1:] != new[:-1]][:len(new)]
        return new[keep], old[keep], np.sign(dot[keep]).astype(int)

    def vertex(self, p, planes):
        """The vertex on these planes at p: an existing one within merge of p, or a new one.
        Only planes that fix a point are kept as the vertex's key."""
        key = frozenset(planes)
        if key in self.ids:
            return self.ids[key]
        point = len(key) >= 3 and np.linalg.svd(self.normal[list(key)], compute_uv=False)[2] > 1e-9
        v = grid_find(self.grid, self.older, self.xyz, p, self.merge)
        if v >= 0:
            self.add_planes(v, planes)
            if point:
                self.ids[key] = v
            return v
        v = self.new_point(p)
        grid_add(self.grid, self.older, p, self.merge, v)
        if point:
            self.ids[key] = v
        self.add_planes(v, planes)
        return v

    def new_point(self, p):
        v = len(self.pos)
        if v == len(self.xyz):
            self.xyz = np.vstack([self.xyz, np.zeros_like(self.xyz)])
            self.older = np.concatenate([self.older, np.full(v, -1)])
        self.xyz[v] = p
        self.pos.append(p)
        self.on.append(set())
        return v

    def add_planes(self, v, planes):
        for p in planes:
            if p not in self.on[v]:
                for q in self.on[v]:
                    self.lines[(min(p, q), max(p, q))].add(v)
                self.on[v].add(p)

    def on_line(self, planes):
        found = set()
        for p, q in combinations(sorted(planes), 2):
            found |= self.lines.get((p, q), set())
        return found

    def sign(self, v, p):
        s = self.signs.get((v, p))
        if s is None:
            if p in self.on[v]:
                s = 0
            else:
                dist = self.pos[v] @ self.normal[p] - self.offset[p]
                s = 0 if abs(dist) < self.eps else (1 if dist > 0 else -1)
                if s == 0:
                    self.add_planes(v, (p,))
            self.signs[(v, p)] = s
        return s

    def sign_matrix(self, poly, planes):
        """sign(v, p) for every vertex of poly (rows) and plane (columns), computed in one
        numpy pass where the vertex is farther than margin from a plane it is not labelled on."""
        planes = np.asarray(planes)
        dist = self.xyz[poly] @ self.normal[planes].T - self.offset[planes]
        s = np.sign(dist).astype(int)
        near = np.abs(dist) < self.margin
        col = {int(p): c for c, p in enumerate(planes)}
        for r, v in enumerate(poly):
            for p in self.on[v] & col.keys():
                near[r, col[p]] = True
        for r, c in zip(*np.nonzero(near)):
            s[r, c] = self.sign(poly[r], int(planes[c]))
        return s

    def crossing(self, a, b, p):
        """The vertex where edge a-b crosses plane p, shared by every edge through that point."""
        line = self.on[a] & self.on[b]
        key = frozenset(line | {p})
        if key in self.ids:
            return self.ids[key]
        da = self.pos[a] @ self.normal[p] - self.offset[p]
        db = self.pos[b] @ self.normal[p] - self.offset[p]
        return self.vertex(self.pos[a] + (self.pos[b] - self.pos[a]) * (da / (da - db)), line | {p})

    def split(self, poly, p, side):
        """(inside, outside) parts of a polygon that straddles the half-space."""
        inside, outside = [], []
        s = [side * self.sign(v, p) for v in poly]
        for i, a in enumerate(poly):
            j = (i + 1) % len(poly)
            if s[i] <= 0:
                inside.append(a)
            if s[i] >= 0:
                outside.append(a)
            if s[i] * s[j] < 0:
                c = self.crossing(a, poly[j], p)
                inside.append(c)
                outside.append(c)
        return distinct(inside), distinct(outside)

    def refine(self, poly, plane, half):
        """poly as (piece, inside) pieces: each lies inside every half-space (on them at
        most) or outside one. A half-space on poly's own plane is skipped."""
        half = [(p, s) for p, s in half if p != plane]
        if not half:
            return [(poly, True)]
        ids, sides = np.array(half).T
        signs = self.sign_matrix(poly, ids) * sides
        lo, hi = signs.min(0), signs.max(0)
        if np.any((lo >= 0) & (hi > 0)):
            return [(poly, False)]
        if not np.any(hi > 0):
            return [(poly, True)]
        out = []
        for p, s in half:
            signs = [s * self.sign(v, p) for v in poly]
            if max(signs) > 0:
                if min(signs) >= 0:
                    return out + [(poly, False)]
                poly, outside = self.split(poly, p, s)
                if len(outside) >= 3:
                    out.append((outside, False))
                if len(poly) < 3:
                    return out
        return out + [(poly, True)]

    def boxes(self, work):
        """(lo, hi) bounds of each piece of work, one row per piece."""
        if not work:
            return np.zeros((0, 3)), np.zeros((0, 3))
        pts = [self.xyz[q] for q, _ in work]
        return np.array([x.min(0) for x in pts]), np.array([x.max(0) for x in pts])

    def cut_pieces(self, work, box, plane, tri):
        """(work, box) with the (piece, label) pieces of work on plane split along the
        planes carrying the trace of a mesh triangle (vertex ids, plane id, its neighbours'
        plane ids) on that plane: the triangle's own plane where it meets that plane in a
        segment, its neighbours' planes where it lies on it. box is boxes(work). Pieces
        whose box, or whose vertices, lie clearly on one side of every such plane are
        passed through unchanged."""
        vids, tp, around = tri
        if not work:
            return work, box
        if tp == plane:
            cuts = around
        else:
            st = [self.sign(v, plane) for v in vids]
            cuts = [tp] if min(st) < 0 < max(st) or st.count(0) >= 2 else []
        if not cuts:
            return work, box
        normal, offset = self.normal[cuts], self.offset[cuts]
        lo, hi = box
        test = box_test(lo, hi, normal, offset, self.margin).tolist()
        if not test:
            return work, box
        sub = [work[i] for i in test]
        lens = [len(q) for q, _ in sub]
        ids = np.fromiter(chain.from_iterable(q for q, _ in sub), np.int64, sum(lens))
        clear = vertex_clear(self.xyz, ids, np.cumsum([0] + lens), normal, offset, self.margin).tolist()
        # a vertex labelled on a cut plane keeps its piece
        on = self.on
        clear = [c and all(on[v].isdisjoint(cuts) for v in q) for c, (q, _) in zip(clear, sub)]
        changed = []
        for i, (q, lab), skip in zip(test, sub, clear):
            if skip:
                continue
            pieces = [q]
            for p in cuts:
                split = []
                for r in pieces:
                    signs = [self.sign(v, p) for v in r]
                    split += [x for x in self.split(r, p, 1) if len(x) >= 3] if min(signs) < 0 < max(signs) else [r]
                pieces = split
            if pieces != [q]:
                changed.append((i, [(r, lab) for r in pieces]))
        if not changed:
            return work, box
        out, los, his, prev = [], [], [], 0
        for i, pieces in changed:
            out += work[prev:i]
            out += pieces
            new_lo, new_hi = self.boxes(pieces)
            los += [lo[prev:i], new_lo]
            his += [hi[prev:i], new_hi]
            prev = i + 1
        out += work[prev:]
        return out, (np.concatenate(los + [lo[prev:]]), np.concatenate(his + [hi[prev:]]))

    def repair(self, poly):
        """poly with every vertex lying on one of its edges inserted into that edge."""
        out = []
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            out.append(a)
            between = self.on_line(self.on[a] & self.on[b]) - {a, b}
            if between:
                vs = np.fromiter(between, np.int64, len(between))
                out += on_edge(self.xyz, a, b, vs, self.merge, self.merge ** 2).tolist()
        return out

    def fan(self, poly):
        """Triangles of a convex polygon; a fan from its centre when vertices lie on its sides."""
        n = len(poly)
        if not any(len(self.on[poly[i - 1]] & self.on[poly[i]] & self.on[poly[(i + 1) % n]]) > 1 for i in range(n)):
            return [(poly[0], poly[i], poly[i + 1]) for i in range(1, n - 1)]
        c = self.new_point(np.mean([self.pos[v] for v in poly], 0))
        return [(c, poly[i], poly[(i + 1) % n]) for i in range(n)]


@intrinsic
def fma(typingctx, x, y, z):
    """x * y + z rounded once."""
    def codegen(context, builder, signature, args):
        return builder.fma(*args)
    return types.float64(types.float64, types.float64, types.float64), codegen


@njit(cache=True)
def on_edge(xyz, a, b, vs, merge, merge2):
    """The vertices vs lying on edge a-b strictly between its ends, within merge of it,
    sorted by position along it then id; merge2 is merge ** 2. Dot products round as
    numpy's BLAS does on arm64 (x0 y0, then two fused steps), the squared distance as its
    einsum does there (z fused onto x, plus y); other platforms may differ in the last bit."""
    d0, d1, d2 = xyz[b, 0] - xyz[a, 0], xyz[b, 1] - xyz[a, 1], xyz[b, 2] - xyz[a, 2]
    length = fma(d2, d2, fma(d1, d1, d0 * d0))
    tol = merge / np.sqrt(length)
    t = np.empty(len(vs))
    keep = np.zeros(len(vs), np.bool_)
    for k in range(len(vs)):
        r0, r1, r2 = xyz[vs[k], 0] - xyz[a, 0], xyz[vs[k], 1] - xyz[a, 1], xyz[vs[k], 2] - xyz[a, 2]
        t[k] = fma(r2, d2, fma(r1, d1, r0 * d0)) / length
        # near-parallel planes share vertices off any one line
        o0, o1, o2 = r0 - t[k] * d0, r1 - t[k] * d1, r2 - t[k] * d2
        keep[k] = tol < t[k] and t[k] < 1 - tol and fma(o2, o2, o0 * o0) + o1 * o1 < merge2
    vs, t = vs[keep], t[keep]
    order = np.argsort(vs)
    vs, t = vs[order], t[order]
    return vs[np.argsort(t, kind="mergesort")]


@njit(cache=True)
def piece_geometry(xyz, ids, start):
    """(sum of the cross products of consecutive points, mean point, perimeter) of each
    polygon ids[start[i]:start[i + 1]] of points xyz, in numpy's rounding: rows summed in
    order, edge lengths summed pairwise."""
    m = len(start) - 1
    n, centre, perimeter = np.zeros((m, 3)), np.zeros((m, 3)), np.zeros(m)
    edge = np.empty(len(ids))
    for i in range(m):
        lo, k = start[i], start[i + 1] - start[i]
        for r in range(lo, lo + k):
            p, q = ids[r], ids[lo + (r - lo + 1) % k]
            c0 = xyz[p, 1] * xyz[q, 2] - xyz[p, 2] * xyz[q, 1]
            c1 = xyz[p, 2] * xyz[q, 0] - xyz[p, 0] * xyz[q, 2]
            c2 = xyz[p, 0] * xyz[q, 1] - xyz[p, 1] * xyz[q, 0]
            if r == lo:
                n[i, 0], n[i, 1], n[i, 2] = c0, c1, c2
                centre[i, 0], centre[i, 1], centre[i, 2] = xyz[p, 0], xyz[p, 1], xyz[p, 2]
            else:
                n[i, 0] += c0
                n[i, 1] += c1
                n[i, 2] += c2
                centre[i, 0] += xyz[p, 0]
                centre[i, 1] += xyz[p, 1]
                centre[i, 2] += xyz[p, 2]
            e0, e1, e2 = xyz[p, 0] - xyz[q, 0], xyz[p, 1] - xyz[q, 1], xyz[p, 2] - xyz[q, 2]
            edge[r] = np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
        centre[i] /= k
        perimeter[i] = pairwise_sum(edge, lo, k)
    return n, centre, perimeter


@njit(cache=True)
def pairwise_sum(a, lo, n):
    """a[lo:lo + n].sum() as numpy computes it."""
    if n < 8:
        res = -0.0
        for i in range(lo, lo + n):
            res += a[i]
        return res
    if n <= 128:
        r = a[lo:lo + 8].copy()
        i = 8
        while i < n - n % 8:
            for j in range(8):
                r[j] += a[lo + i + j]
            i += 8
        res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for j in range(lo + i, lo + n):
            res += a[j]
        return res
    half = n // 2
    half -= half % 8
    return pairwise_sum(a, lo, half) + pairwise_sum(a, lo + half, n - half)


@njit(cache=True)
def grid_find(grid, older, xyz, p, merge):
    """The vertex within merge of p, or -1: the lowest id in the first of the 27 cells
    around p's (in dx, dy, dz order) holding one. Distances round as numpy's norm does on
    arm64."""
    x, y, z = int(np.floor(p[0] / merge)), int(np.floor(p[1] / merge)), int(np.floor(p[2] / merge))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = (x + dx, y + dy, z + dz)
                v = grid[cell] if cell in grid else -1
                found = -1
                while v >= 0:
                    d0, d1, d2 = xyz[v, 0] - p[0], xyz[v, 1] - p[1], xyz[v, 2] - p[2]
                    if np.sqrt(fma(d2, d2, fma(d1, d1, d0 * d0))) < merge:
                        found = v
                    v = older[v]
                if found >= 0:
                    return found
    return -1


@njit(cache=True)
def grid_add(grid, older, p, merge, v):
    cell = (int(np.floor(p[0] / merge)), int(np.floor(p[1] / merge)), int(np.floor(p[2] / merge)))
    older[v] = grid[cell] if cell in grid else -1
    grid[cell] = v


@njit(cache=True)
def box_test(lo, hi, normal, offset, margin):
    """Indices of the boxes lo..hi that some plane normal . p = offset passes within
    margin of; products round as numpy's matmul does on arm64."""
    out = np.empty(len(lo), np.int64)
    n = 0
    for i in range(len(lo)):
        s0, s1, s2 = lo[i, 0] + hi[i, 0], lo[i, 1] + hi[i, 1], lo[i, 2] + hi[i, 2]
        w0, w1, w2 = hi[i, 0] - lo[i, 0], hi[i, 1] - lo[i, 1], hi[i, 2] - lo[i, 2]
        for j in range(len(offset)):
            mid = fma(s2, normal[j, 2], fma(s1, normal[j, 1], s0 * normal[j, 0])) / 2 - offset[j]
            reach = fma(w2, abs(normal[j, 2]), fma(w1, abs(normal[j, 1]), w0 * abs(normal[j, 0]))) / 2
            if not (mid - reach > margin or mid + reach < -margin):
                out[n] = i
                n += 1
                break
    return out[:n]


@njit(cache=True)
def vertex_clear(xyz, ids, start, normal, offset, margin):
    """Whether each polygon ids[start[i]:start[i + 1]] has all its vertices farther than
    margin on one side of each plane normal . p = offset; products round as numpy's matmul
    does on arm64."""
    m = len(start) - 1
    clear = np.ones(m, np.bool_)
    for i in range(m):
        for j in range(len(offset)):
            lo, hi = np.inf, -np.inf
            for r in range(start[i], start[i + 1]):
                v = ids[r]
                d = fma(xyz[v, 2], normal[j, 2], fma(xyz[v, 1], normal[j, 1], xyz[v, 0] * normal[j, 0])) - offset[j]
                lo, hi = min(lo, d), max(hi, d)
            if not (lo > margin or hi < -margin):
                clear[i] = False
                break
    return clear


def distinct(loop):
    """loop without vertices repeating the one before them."""
    return [v for k, v in enumerate(loop) if v != loop[k - 1]]


def facet_geometry(shape):
    """(points, loops, unit normals, offsets, faces at each edge or None, loop index of
    each input polygon or -1) of a shape's faceted surface, without degenerate facets;
    a convex shape's loops are wound outward."""
    points, polygons = shape.facets()
    points, inverse = np.unique(np.asarray(points, float), axis=0, return_inverse=True)
    inverse = inverse.ravel()
    size = points.max(0) - points.min(0)
    tiny = 1e-12 * (size @ size)
    loops, normals = [], []
    loop_of = np.full(len(polygons), -1)
    for g, poly in enumerate(polygons):
        loop = distinct([int(inverse[i]) for i in poly])
        p = points[loop]
        n = np.cross(p, np.roll(p, -1, 0)).sum(0) if len(loop) >= 3 else np.zeros(3)
        if np.linalg.norm(n) > tiny:
            loop_of[g] = len(loops)
            loops.append(loop)
            normals.append(n / np.linalg.norm(n))
    normals = np.array(normals).reshape(-1, 3)
    if shape.convex:
        interior = points.mean(0)
        for f, loop in enumerate(loops):
            if normals[f] @ (points[loop].mean(0) - interior) < 0:
                loops[f], normals[f] = loop[::-1], -normals[f]
    offsets = np.array([n @ points[loop].mean(0) for n, loop in zip(normals, loops)])
    faces_at = None
    if not shape.convex:
        faces_at = defaultdict(set)
        for f, loop in enumerate(loops):
            for k in range(3):
                faces_at[frozenset((loop[k], loop[k - 1]))].add(f)
    return points, loops, normals, offsets, faces_at, loop_of


class Faceted:
    """A shape's facets in a kernel as (vertex loop, plane id, side), side -1 where the
    facet faces against its plane's stored normal, and its bounds. A convex shape gives
    the half-spaces (plane id, side) whose intersection is the shape; a mesh gives its
    triangles as (vertex ids, plane id, the plane ids of the triangles sharing an edge
    with it). The clip box gives its half-spaces only. geometry is facet_geometry(shape)."""

    def __init__(self, kernel, shape, geometry):
        self.mesh = None if shape.convex else shape
        self.facets, self.tris, self.half = [], [], None
        if not hasattr(shape, "facets"):
            ids, sides = kernel.planes(shape.normals, shape.offsets)
            self.half = [(int(p), int(s)) for p, s in zip(ids, sides)]
            self.half_map = dict(self.half)
            self.lo, self.hi = np.full(3, -np.inf), np.full(3, np.inf)
            return
        points, loops, normals, offsets, faces_at, self.loop_of = geometry
        self.lo, self.hi = points.min(0), points.max(0)
        self.tri_lo = self.tri_hi = np.zeros((0, 3))
        if not loops:
            return
        ids, sides = kernel.planes(normals, offsets)
        incident = defaultdict(set)
        for f, loop in enumerate(loops):
            for i in loop:
                incident[i].add(int(ids[f]))
        if self.mesh is not None:
            # an edge between coplanar triangles gets the plane through it across their plane
            flat = [(tuple(e), next(iter(fs))) for e, fs in faces_at.items() if len({int(ids[f]) for f in fs}) == 1]
            if flat:
                across = np.array([np.cross(points[b] - points[a], normals[f]) for (a, b), f in flat])
                across /= np.linalg.norm(across, axis=1)[:, None]
                edge_ids, _ = kernel.planes(across, np.array([n @ points[a] for n, ((a, _), _) in zip(across, flat)]))
                for ((a, b), _), q in zip(flat, edge_ids):
                    incident[a].add(int(q))
                    incident[b].add(int(q))
        vid = {i: kernel.vertex(points[i], planes) for i, planes in incident.items()}
        self.facets = [([vid[i] for i in loop], int(ids[f]), int(sides[f])) for f, loop in enumerate(loops)]
        if self.mesh is None:
            self.half = [(int(p), int(s)) for p, s in zip(ids, sides)]
            self.half_map = dict(self.half)
        else:
            self.tris = []
            for f, (poly, plane, _) in enumerate(self.facets):
                loop = loops[f]
                around = {int(ids[g]) for k in range(3) for g in faces_at[frozenset((loop[k], loop[k - 1]))]} - {plane}
                self.tris.append((poly, plane, sorted(around)))
            corners = points[np.array(loops)]
            self.tri_lo, self.tri_hi = corners.min(1), corners.max(1)


def closed_cut(own, cutters, clip_box=None, geometry=None):
    """(vertices, faces) of the region inside own and its masks and outside every
    cutter present (inside its masks): the parts of each surface bounding it, wound
    outward. own and cutters give .shape, .masks and .any_mode; clip_box is a
    ConvexShape that removes its side and gets no wall. geometry caches
    facet_geometry by shape across calls."""
    geometry = {} if geometry is None else geometry

    def facets_of(shape):
        if shape not in geometry:
            geometry[shape] = facet_geometry(shape)
        return geometry[shape]

    points = facets_of(own.shape)[0]
    scale = np.linalg.norm(points.max(0) - points.min(0))
    kernel = Kernel(scale)
    shapes, index = [], {}

    def add(shape):
        if id(shape) not in index:
            index[id(shape)] = len(shapes)
            shapes.append(Faceted(kernel, shape, facets_of(shape) if hasattr(shape, "facets") else None))
        return index[id(shape)]

    own_i, own_masks = add(own.shape), [add(m) for m in own.masks]
    cuts = [(add(c.shape), [add(m) for m in c.masks], c.any_mode) for c in cutters]
    owners = len(shapes)
    clip_i = add(clip_box) if clip_box is not None else None
    body = shapes[own_i]
    plain_cutters = {i for i, masks, _ in cuts if not masks} | {clip_i}
    all_masks = set(own_masks) if not own.any_mode else set()

    def present(st, i, masks, any_mode):
        return st[i] and (not masks or (any if any_mode else all)(st[m] for m in masks))

    def region(st):
        return present(st, own_i, own_masks, own.any_mode) and not any(present(st, *c) for c in cuts) \
            and not (clip_i is not None and st[clip_i])

    def dead(j, inside, plane):
        if not inside:
            return j == own_i or j in all_masks
        return j in plain_cutters and plane not in shapes[j].half_map

    pieces = []
    for k in range(owners):
        bounds = [np.array([kernel.pos[v] for v in poly]) for poly, _, _ in shapes[k].facets]
        facet_lo = np.array([b.min(0) for b in bounds]).reshape(-1, 3)
        facet_hi = np.array([b.max(0) for b in bounds]).reshape(-1, 3)
        near = {j: triangles_near(other, facet_lo, facet_hi, kernel.eps)
                for j, other in enumerate(shapes) if j != k and other.mesh is not None}
        for f, (poly, plane, _) in enumerate(shapes[k].facets):
            lo, hi = facet_lo[f], facet_hi[f]
            if k != own_i and (np.any(lo > body.hi) or np.any(hi < body.lo)):
                continue
            work = [(poly, {k: True})]
            for j, other in enumerate(shapes):
                if not work:
                    break
                if j == k or np.any(lo > other.hi + kernel.eps) or np.any(hi < other.lo - kernel.eps):
                    continue
                if other.mesh is None:
                    work = [(r, {**lab, j: inside}) for q, lab in work
                            for r, inside in kernel.refine(q, plane, other.half) if not dead(j, inside, plane)]
                    continue
                box = kernel.boxes(work)
                for t in near[j][f]:
                    work, box = kernel.cut_pieces(work, box, plane, other.tris[t])
            pieces += [(q, plane, k, lab) for q, lab in work]
    loops = classify_pieces(kernel, shapes, pieces, region, scale)
    faces = [t for poly in loops for t in kernel.fan(kernel.repair(poly))]
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    faces = np.array(faces)
    used, faces = np.unique(faces, return_inverse=True)
    return np.array(kernel.pos)[used], faces.reshape(-1, 3)


def triangles_near(mesh, lo, hi, eps):
    """For each box lo..hi, the ascending indices of mesh's triangles whose boxes
    overlap it within eps."""
    q, t = mesh.mesh.bvh.overlaps(lo - eps, hi + eps)
    t = mesh.loop_of[t]
    keep = t >= 0
    q, t = q[keep], t[keep]
    keep = np.all((mesh.tri_lo[t] <= hi[q] + eps) & (mesh.tri_hi[t] >= lo[q] - eps), axis=1)
    q, t = q[keep], t[keep]
    order = np.lexsort((t, q))
    q, t = q[order], t[order]
    return np.split(t, np.searchsorted(q, np.arange(1, len(lo))))


def classify_pieces(kernel, shapes, pieces, region, scale):
    """The pieces that bound the region, wound outward. Each piece's status in every
    shape is taken just above and just below its plane: exactly from its labels for a
    convex shape, by the mesh's own inside test for a mesh. A piece lying on the
    surface of a shape listed before its own is left to that shape."""
    if not pieces:
        return []
    lens = [len(q) for q, _, _, _ in pieces]
    ids = np.fromiter(chain.from_iterable(q for q, _, _, _ in pieces), np.int64, sum(lens))
    n, centre, perimeter = piece_geometry(kernel.xyz, ids, np.concatenate([[0], np.cumsum(lens)]))
    area = 0.5 * np.sqrt(np.vecdot(n, n))
    keep = np.flatnonzero(~(area <= kernel.merge ** 2))
    if not len(keep):
        return []
    kept = [(*pieces[i][:4], n[i]) for i in keep.tolist()]
    centre = centre[keep]
    step = np.minimum(1e-6 * scale, 0.2 * area[keep] / perimeter[keep])
    offset = step[:, None] * kernel.normal[[pieces[i][1] for i in keep.tolist()]]
    statuses = []
    for j, shape in enumerate(shapes):
        if shape.mesh is None:
            statuses.append(None)
            continue
        above, below = np.zeros(len(kept), bool), np.zeros(len(kept), bool)
        near = np.flatnonzero(np.all((centre >= shape.lo - scale * 1e-6) & (centre <= shape.hi + scale * 1e-6), axis=1))
        if len(near):
            above[near] = shape.mesh.contains(centre[near] + offset[near])
            below[near] = shape.mesh.contains(centre[near] - offset[near])
        statuses.append((above, below))
    loops = []
    for i, (q, plane, k, lab, n) in enumerate(kept):
        above, below = [], []
        for j, shape in enumerate(shapes):
            if statuses[j] is not None:
                above.append(statuses[j][0][i])
                below.append(statuses[j][1][i])
            elif not lab.get(j, False):
                above.append(False)
                below.append(False)
            else:
                s = shape.half_map.get(plane, 0)
                above.append(s <= 0)
                below.append(s >= 0)
        if any(above[j] != below[j] for j in range(k)):
            continue
        ra, rb = region(above), region(below)
        if ra != rb:
            loops.append(q if (n @ kernel.normal[plane] > 0) == rb else q[::-1])
    return loops
