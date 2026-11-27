"""Closed cuts on faceted shapes: a component's kept surface and the walls that
close it, cut along one shared set of vertices so the result is watertight.

Every shape is its faceted display polyhedron: convex planar facets for boxes and
lathes, triangles for meshes; the clip box is half-spaces only. Coincident planes
share one plane id. A vertex is known by the planes it lies on, or by its position
where they do not fix a point, so a cut surface and the wall meeting it are built
from the same vertices; each vertex is classified against each plane once."""

from collections import defaultdict
from itertools import chain

import numpy as np
from numba import njit, typed, types

# distances relative to the own shape's size
ON_PLANE = 1e-9
SAME_POINT = 1e-8
# largest 1 - |cos| between the normals of coincident planes
PARALLEL = 1e-12
# the kernel's entry lists
ON, LINE, KEY = 0, 1, 2


class Kernel:
    def __init__(self, scale):
        self.eps, self.merge = ON_PLANE * scale, SAME_POINT * scale
        # a vertex gains a label on a plane only within merge of it, so its sign against a
        # plane it is farther than this from and not labelled on never changes
        self.margin = 2 * self.merge
        self.normal, self.offset = np.zeros((0, 3)), np.zeros(0)
        # vertex positions, rows past count[0] unused
        self.xyz, self.count = np.zeros((256, 3)), np.zeros(1, np.int64)
        # entry lists, used[k] of store[k] filled: the planes each vertex lies on, (next entry,
        # plane) in store[ON] from each vertex's newest in head (-1 at the end), with is_on
        # holding pair(v, p) for each; the vertices on each line pair(p, q), p < q, (next
        # entry, vertex) in store[LINE] from its newest in lines; the vertex of each key of
        # planes fixing a point, (next entry, vertex, number of planes, planes ascending) in
        # store[KEY] from the newest of its key_hash in ids
        self.store = typed.List([np.zeros(1024, np.int64) for _ in range(3)])
        self.used = np.zeros(3, np.int64)
        self.head = np.full(256, -1)
        self.is_on = typed.Dict.empty(types.int64, types.boolean)
        self.lines = typed.Dict.empty(types.int64, types.int64)
        self.ids = typed.Dict.empty(types.int64, types.int64)
        # the first sign given for each pair(v, p)
        self.signs = typed.Dict.empty(types.int64, types.int8)
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
        self.reserve()
        return vertex(self.xyz, self.older, self.head, self.count, self.grid, self.merge, self.normal, self.store,
                      self.used, self.is_on, self.lines, self.ids, np.asarray(p, float),
                      np.fromiter(planes, np.int64, len(planes)))

    def reserve(self):
        """Room in xyz for one more vertex."""
        v = self.count[0]
        if v == len(self.xyz):
            self.xyz = np.vstack([self.xyz, np.zeros_like(self.xyz)])
            self.older = np.concatenate([self.older, np.full(v, -1)])
            self.head = np.concatenate([self.head, np.full(v, -1)])

    def new_point(self, p):
        self.reserve()
        v = int(self.count[0])
        self.xyz[v] = p
        self.count[0] += 1
        return v

    def common(self, vs):
        """The planes every vertex of vs is labelled on, ascending."""
        return common(self.head, self.store, self.is_on, np.asarray(vs, np.int64))

    def on_line(self, planes):
        """The vertices on the line of some two of planes (ascending ids), each once."""
        return on_line(self.store, self.lines, np.asarray(planes, np.int64))

    def sign(self, vs, p):
        """sign(v, p) for each vertex v of vs: 0 where v is labelled on p or within eps of
        it (labelling it), the first answer for (v, p) once given."""
        return sign_of(self.xyz, self.normal, self.offset, self.eps, self.head, self.store, self.used, self.is_on,
                       self.lines, self.signs, np.asarray(vs, np.int64), p)

    def sign_matrix(self, poly, planes):
        """sign(v, p) for every vertex of poly (rows) and plane (columns), computed directly
        where the vertex is farther than margin from a plane it is not labelled on."""
        return sign_matrix(self.xyz, self.normal, self.offset, self.eps, self.margin, self.head, self.store,
                           self.used, self.is_on, self.lines, self.signs, np.asarray(poly, np.int64),
                           np.asarray(planes, np.int64))

    def crossing(self, a, b, p):
        """The vertex where edge a-b crosses plane p, shared by every edge through that point."""
        self.reserve()
        return crossing(self.xyz, self.older, self.head, self.count, self.grid, self.merge, self.normal,
                        self.offset, self.store, self.used, self.is_on, self.lines, self.ids, a, b, p)

    def split(self, poly, p, side):
        """(inside, outside) parts of a polygon that straddles the half-space."""
        inside, outside = [], []
        s = (side * self.sign(poly, p)).tolist()
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
        half = [(p, s) for p, s in half.items() if p != plane]
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
            signs = s * self.sign(poly, p)
            if signs.max() > 0:
                if signs.min() >= 0:
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
        lens = [len(q) for q, _ in work]
        ids = np.fromiter(chain.from_iterable(q for q, _ in work), np.int64, sum(lens))
        return piece_bounds(self.xyz, ids, np.cumsum([0] + lens))

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
            st = self.sign(vids, plane).tolist()
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
        start = np.cumsum([0] + lens)
        # a vertex labelled on a cut plane keeps its piece
        clear = (vertex_clear(self.xyz, ids, start, normal, offset, self.margin)
                 & ~any_on(self.is_on, ids, start, np.asarray(cuts, np.int64))).tolist()
        changed = []
        for i, (q, lab), skip in zip(test, sub, clear):
            if skip:
                continue
            pieces = [q]
            for p in cuts:
                split = []
                for r in pieces:
                    signs = self.sign(r, p)
                    split += [x for x in self.split(r, p, 1) if len(x) >= 3] if signs.min() < 0 < signs.max() else [r]
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
            vs = self.on_line(self.common((a, b)))
            vs = vs[(vs != a) & (vs != b)]
            if len(vs):
                out += on_edge(self.xyz, a, b, vs, self.merge).tolist()
        return out

    def fan(self, poly):
        """Triangles of a convex polygon; a fan from its centre when vertices lie on its sides."""
        n = len(poly)
        if not any(len(self.common((poly[i - 1], poly[i], poly[(i + 1) % n]))) > 1 for i in range(n)):
            return [(poly[0], poly[i], poly[i + 1]) for i in range(1, n - 1)]
        c = self.new_point(self.xyz[poly].mean(0))
        return [(c, poly[i], poly[(i + 1) % n]) for i in range(n)]


@njit(cache=True)
def pair(a, b):
    """One int64 key for a pair of non-negative ids below 2**31."""
    return a << 32 | b


@njit(cache=True)
def push(store, used, k, x):
    """Append x to the entry list store[k], doubling it when full; its index."""
    n = used[k]
    if n == len(store[k]):
        grown = np.empty(2 * n, np.int64)
        grown[:n] = store[k]
        store[k] = grown
    store[k][n] = x
    used[k] = n + 1
    return n


@njit(cache=True)
def add_planes(head, store, used, is_on, lines, v, planes):
    """Label vertex v on planes, adding it to the line of each new pair of its planes."""
    for p in planes:
        if pair(v, p) in is_on:
            continue
        on = store[ON]
        e = head[v]
        while e >= 0:
            q = on[e + 1]
            line = pair(min(p, q), max(p, q))
            lines[line] = push(store, used, LINE, lines[line] if line in lines else -1)
            push(store, used, LINE, v)
            e = on[e]
        head[v] = push(store, used, ON, head[v])
        push(store, used, ON, p)
        is_on[pair(v, p)] = True


@njit(cache=True)
def common(head, store, is_on, vs):
    """The planes every vertex of vs is labelled on, ascending."""
    on = store[ON]
    out = []
    e = head[vs[0]]
    while e >= 0:
        p = on[e + 1]
        shared = True
        for k in range(1, len(vs)):
            if pair(vs[k], p) not in is_on:
                shared = False
                break
        if shared:
            out.append(p)
        e = on[e]
    return np.sort(np.array(out, np.int64))


@njit(cache=True)
def on_line(store, lines, planes):
    """The vertices on the line of some two of planes (ascending ids), each once."""
    entries = store[LINE]
    heads = []
    n = 0
    for i in range(len(planes)):
        for j in range(i + 1, len(planes)):
            line = pair(planes[i], planes[j])
            if line in lines:
                e = lines[line]
                heads.append(e)
                while e >= 0:
                    n += 1
                    e = entries[e]
    out = np.empty(n, np.int64)
    n = 0
    for e in heads:
        while e >= 0:
            out[n] = entries[e + 1]
            n += 1
            e = entries[e]
    # a vertex is on each line once, so only several lines can repeat it
    return np.unique(out) if len(heads) > 1 else out


@njit(cache=True)
def sign(xyz, normal, offset, eps, head, store, used, is_on, lines, signs, v, p):
    """The side of plane p vertex v lies on: 0 if labelled on it or within eps of it
    (labelling it), the first answer for (v, p) once given."""
    key = pair(v, p)
    if key in signs:
        return np.int64(signs[key])
    s = 0
    if key not in is_on:
        dist = np.dot(xyz[v], normal[p]) - offset[p]
        s = 0 if abs(dist) < eps else (1 if dist > 0 else -1)
        if s == 0:
            add_planes(head, store, used, is_on, lines, v, np.array([p]))
    signs[key] = np.int8(s)
    return s


@njit(cache=True)
def sign_of(xyz, normal, offset, eps, head, store, used, is_on, lines, signs, vs, p):
    out = np.empty(len(vs), np.int64)
    for k in range(len(vs)):
        out[k] = sign(xyz, normal, offset, eps, head, store, used, is_on, lines, signs, vs[k], p)
    return out


@njit(cache=True)
def sign_matrix(xyz, normal, offset, eps, margin, head, store, used, is_on, lines, signs, poly, planes):
    """sign of every vertex of poly (rows) against every plane (columns), taken from the
    distance where the vertex is farther than margin from a plane it is not labelled on;
    the others are given by sign in row-major order."""
    s = np.empty((len(poly), len(planes)), np.int64)
    near = np.zeros((len(poly), len(planes)), np.bool_)
    for r in range(len(poly)):
        for c in range(len(planes)):
            dist = np.dot(xyz[poly[r]], normal[planes[c]]) - offset[planes[c]]
            s[r, c] = 1 if dist > 0 else (-1 if dist < 0 else 0)
            near[r, c] = abs(dist) < margin or pair(poly[r], planes[c]) in is_on
    for r in range(len(poly)):
        for c in range(len(planes)):
            if near[r, c]:
                s[r, c] = sign(xyz, normal, offset, eps, head, store, used, is_on, lines, signs, poly[r], planes[c])
    return s


@njit(cache=True)
def key_hash(planes):
    h = len(planes)
    for p in planes:
        h = h * 1000003 + p
    return h


@njit(cache=True)
def vertex(xyz, older, head, count, grid, merge, normal, store, used, is_on, lines, ids, p, planes):
    """The vertex on planes at p: the one keyed by them, else an existing one within merge
    of p, else a new one at xyz[count[0]] (a free row). It gets labelled on planes; planes
    that fix a point become its key."""
    planes = np.unique(planes)
    h = key_hash(planes)
    keys = store[KEY]
    e = ids[h] if h in ids else -1
    while e >= 0:
        if keys[e + 2] == len(planes) and np.all(keys[e + 3:e + 3 + len(planes)] == planes):
            return keys[e + 1]
        e = keys[e]
    point = len(planes) >= 3 and fixes_point(normal, planes)
    v = grid_find(grid, older, xyz, p, merge)
    if v < 0:
        v = count[0]
        count[0] += 1
        xyz[v] = p
        grid_add(grid, older, p, merge, v)
    add_planes(head, store, used, is_on, lines, v, planes)
    if point:
        ids[h] = push(store, used, KEY, ids[h] if h in ids else -1)
        push(store, used, KEY, v)
        push(store, used, KEY, len(planes))
        for q in planes:
            push(store, used, KEY, q)
    return v


@njit(cache=True)
def crossing(xyz, older, head, count, grid, merge, normal, offset, store, used, is_on, lines, ids, a, b, p):
    """The vertex where edge a-b crosses plane p, on p and the planes a and b share."""
    planes = np.append(common(head, store, is_on, np.array([a, b])), p)
    pa, pb = xyz[a], xyz[b]
    da, db = np.dot(pa, normal[p]) - offset[p], np.dot(pb, normal[p]) - offset[p]
    return vertex(xyz, older, head, count, grid, merge, normal, store, used, is_on, lines, ids,
                  pa + (pb - pa) * (da / (da - db)), planes)


@njit(cache=True)
def any_on(is_on, ids, start, planes):
    """Whether some vertex of each polygon ids[start[i]:start[i + 1]] is labelled on one of planes."""
    m = len(start) - 1
    out = np.zeros(m, np.bool_)
    for i in range(m):
        for r in range(start[i], start[i + 1]):
            for p in planes:
                if pair(ids[r], p) in is_on:
                    out[i] = True
    return out


@njit(cache=True)
def fixes_point(normal, ids):
    """Whether the planes ids meet in one point: the smallest singular value of their
    normals above 1e-9, taken as e3 / e2 of the normals' Gram matrix (the squared smallest
    singular value up to a relative error of order its own square over the second
    smallest's), with e2 and e3 summed from squared 2x2 and 3x3 minors (Cauchy-Binet)."""
    e2 = e3 = 0.0
    k = len(ids)
    for a in range(k):
        p = normal[ids[a]]
        for b in range(a + 1, k):
            q = normal[ids[b]]
            c0, c1, c2 = p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]
            e2 += c0 * c0 + c1 * c1 + c2 * c2
            for c in range(b + 1, k):
                r = normal[ids[c]]
                d = c0 * r[0] + c1 * r[1] + c2 * r[2]
                e3 += d * d
    return e3 > 1e-18 * e2


@njit(cache=True)
def piece_bounds(xyz, ids, start):
    """(lo, hi) bounds of each polygon ids[start[i]:start[i + 1]] of points xyz."""
    m = len(start) - 1
    lo, hi = np.empty((m, 3)), np.empty((m, 3))
    for i in range(m):
        for d in range(3):
            lo[i, d], hi[i, d] = np.inf, -np.inf
            for r in range(start[i], start[i + 1]):
                lo[i, d] = min(lo[i, d], xyz[ids[r], d])
                hi[i, d] = max(hi[i, d], xyz[ids[r], d])
    return lo, hi


@njit(cache=True)
def on_edge(xyz, a, b, vs, merge):
    """The vertices vs lying on edge a-b strictly between its ends, within merge of it,
    sorted by position along it then id."""
    d0, d1, d2 = xyz[b, 0] - xyz[a, 0], xyz[b, 1] - xyz[a, 1], xyz[b, 2] - xyz[a, 2]
    length = d0 * d0 + d1 * d1 + d2 * d2
    tol = merge / np.sqrt(length)
    t = np.empty(len(vs))
    keep = np.zeros(len(vs), np.bool_)
    for k in range(len(vs)):
        r0, r1, r2 = xyz[vs[k], 0] - xyz[a, 0], xyz[vs[k], 1] - xyz[a, 1], xyz[vs[k], 2] - xyz[a, 2]
        t[k] = (r0 * d0 + r1 * d1 + r2 * d2) / length
        # near-parallel planes share vertices off any one line
        o0, o1, o2 = r0 - t[k] * d0, r1 - t[k] * d1, r2 - t[k] * d2
        keep[k] = tol < t[k] < 1 - tol and o0 * o0 + o1 * o1 + o2 * o2 < merge * merge
    vs, t = vs[keep], t[keep]
    order = np.argsort(vs)
    vs, t = vs[order], t[order]
    return vs[np.argsort(t, kind="mergesort")]


@njit(cache=True)
def piece_geometry(xyz, ids, start):
    """(sum of the cross products of consecutive points, mean point, perimeter) of each
    polygon ids[start[i]:start[i + 1]] of points xyz."""
    m = len(start) - 1
    n, centre, perimeter = np.zeros((m, 3)), np.zeros((m, 3)), np.zeros(m)
    for i in range(m):
        lo, k = start[i], start[i + 1] - start[i]
        for r in range(lo, lo + k):
            p, q = ids[r], ids[lo + (r - lo + 1) % k]
            n[i, 0] += xyz[p, 1] * xyz[q, 2] - xyz[p, 2] * xyz[q, 1]
            n[i, 1] += xyz[p, 2] * xyz[q, 0] - xyz[p, 0] * xyz[q, 2]
            n[i, 2] += xyz[p, 0] * xyz[q, 1] - xyz[p, 1] * xyz[q, 0]
            centre[i] += xyz[p]
            e0, e1, e2 = xyz[p, 0] - xyz[q, 0], xyz[p, 1] - xyz[q, 1], xyz[p, 2] - xyz[q, 2]
            perimeter[i] += np.sqrt(e0 * e0 + e1 * e1 + e2 * e2)
        centre[i] /= k
    return n, centre, perimeter


@njit(cache=True)
def grid_find(grid, older, xyz, p, merge):
    """The vertex within merge of p, or -1: the lowest id in the first of the 27 cells
    around p's (in dx, dy, dz order) holding one."""
    x, y, z = int(np.floor(p[0] / merge)), int(np.floor(p[1] / merge)), int(np.floor(p[2] / merge))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cell = (x + dx, y + dy, z + dz)
                v = grid[cell] if cell in grid else -1
                found = -1
                while v >= 0:
                    d0, d1, d2 = xyz[v, 0] - p[0], xyz[v, 1] - p[1], xyz[v, 2] - p[2]
                    if np.sqrt(d0 * d0 + d1 * d1 + d2 * d2) < merge:
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
    margin of."""
    out = np.empty(len(lo), np.int64)
    n = 0
    for i in range(len(lo)):
        s0, s1, s2 = lo[i, 0] + hi[i, 0], lo[i, 1] + hi[i, 1], lo[i, 2] + hi[i, 2]
        w0, w1, w2 = hi[i, 0] - lo[i, 0], hi[i, 1] - lo[i, 1], hi[i, 2] - lo[i, 2]
        for j in range(len(offset)):
            n0, n1, n2 = normal[j, 0], normal[j, 1], normal[j, 2]
            mid = (s0 * n0 + s1 * n1 + s2 * n2) / 2 - offset[j]
            reach = (w0 * abs(n0) + w1 * abs(n1) + w2 * abs(n2)) / 2
            if not (mid - reach > margin or mid + reach < -margin):
                out[n] = i
                n += 1
                break
    return out[:n]


@njit(cache=True)
def vertex_clear(xyz, ids, start, normal, offset, margin):
    """Whether each polygon ids[start[i]:start[i + 1]] has all its vertices farther than
    margin on one side of each plane normal . p = offset."""
    m = len(start) - 1
    clear = np.ones(m, np.bool_)
    for i in range(m):
        for j in range(len(offset)):
            lo, hi = np.inf, -np.inf
            for r in range(start[i], start[i + 1]):
                v = ids[r]
                d = xyz[v, 0] * normal[j, 0] + xyz[v, 1] * normal[j, 1] + xyz[v, 2] * normal[j, 2] - offset[j]
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
    if isinstance(polygons, np.ndarray) and polygons.ndim == 2 and polygons.shape[1] == 3:
        # triangles: a repeated corner gives a zero normal, as distinct() would
        tri = inverse[polygons]
        p = points[tri]
        cross = np.cross(p, np.roll(p, -1, 1)).sum(1)
        for g, n in enumerate(cross):
            length = np.linalg.norm(n)
            if length > tiny:
                loop_of[g] = len(loops)
                loops.append(tri[g].tolist())
                normals.append(n / length)
    else:
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
    the half-spaces as {plane id: side}, whose intersection is the shape; a mesh gives its
    triangles as (vertex ids, plane id, the plane ids of the triangles sharing an edge
    with it). The clip box gives its half-spaces only. geometry is facet_geometry(shape)."""

    def __init__(self, kernel, shape, geometry):
        self.mesh = None if shape.convex else shape
        self.facets, self.tris, self.half = [], [], None
        if not hasattr(shape, "facets"):
            ids, sides = kernel.planes(shape.normals, shape.offsets)
            self.half = dict(zip(ids.tolist(), sides.tolist()))
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
            self.half = dict(zip(ids.tolist(), sides.tolist()))
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
        return j in plain_cutters and plane not in shapes[j].half

    pieces = []
    for k in range(owners):
        facet_lo, facet_hi = kernel.boxes([(poly, None) for poly, _, _ in shapes[k].facets])
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
    return kernel.xyz[used], faces.reshape(-1, 3)


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
                s = shape.half.get(plane, 0)
                above.append(s <= 0)
                below.append(s >= 0)
        if any(above[j] != below[j] for j in range(k)):
            continue
        ra, rb = region(above), region(below)
        if ra != rb:
            loops.append(q if (n @ kernel.normal[plane] > 0) == rb else q[::-1])
    return loops
