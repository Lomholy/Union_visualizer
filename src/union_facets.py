"""Closed cuts on faceted convex shapes: a component's kept surface and the walls
that close it, cut along one shared set of vertices so the result is watertight.

Every shape is its faceted display polyhedron, a list of convex planar facets.
Coincident planes of different shapes share one plane id. A vertex is known by
the planes it lies on, so a cut surface and the wall meeting it are built from
the same vertices; each vertex is classified against each plane once."""

from collections import defaultdict
from itertools import combinations

import numpy as np

# distances relative to the own shape's size
ON_PLANE = 1e-9
SAME_POINT = 1e-8
# largest 1 - |cos| between the normals of coincident planes
PARALLEL = 1e-12


class Kernel:
    def __init__(self, scale):
        self.eps, self.merge = ON_PLANE * scale, SAME_POINT * scale
        self.normal, self.offset = np.zeros((0, 3)), np.zeros(0)
        self.pos, self.on = [], []
        self.ids, self.lines, self.signs = {}, defaultdict(set), {}
        # vertex ids by position, in cells of size merge
        self.grid = defaultdict(list)

    def planes(self, normal, offset):
        """(plane ids, sides) for planes normal . p = offset; side is -1 where the
        stored plane of that id faces the other way."""
        ids, sides = np.zeros(len(offset), int), np.ones(len(offset), int)
        fresh = np.ones(len(offset), bool)
        if len(self.offset):
            dot = normal @ self.normal.T
            same = (1 - np.abs(dot) < PARALLEL) & (np.abs(offset[:, None] - np.sign(dot) * self.offset) < self.eps)
            fresh = ~same.any(1)
            hit = np.flatnonzero(~fresh)
            ids[hit] = same[hit].argmax(1)
            sides[hit] = np.sign(dot[hit, ids[hit]])
        ids[fresh] = len(self.offset) + np.arange(fresh.sum())
        self.normal = np.vstack([self.normal, normal[fresh]])
        self.offset = np.concatenate([self.offset, offset[fresh]])
        return ids, sides

    def vertex(self, p, planes):
        """The vertex on these planes at p: an existing one within merge of p, or a new one."""
        key = frozenset(planes)
        if key in self.ids:
            return self.ids[key]
        cell = np.floor(p / self.merge).astype(int)
        for d in np.ndindex(3, 3, 3):
            for v in self.grid.get(tuple(cell + d - 1), ()):
                if np.linalg.norm(self.pos[v] - p) < self.merge:
                    self.add_planes(v, planes)
                    self.ids[key] = v
                    return v
        v = len(self.pos)
        self.pos.append(p)
        self.on.append(set())
        self.grid[tuple(cell)].append(v)
        self.ids[key] = v
        self.add_planes(v, planes)
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
        for p, s in half:
            if p != plane and min(s * self.sign(v, p) for v in poly) >= 0 and max(s * self.sign(v, p) for v in poly) > 0:
                return [(poly, False)]
        out = []
        for p, s in half:
            if p != plane:
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

    def cut_by_triangle(self, poly, plane, tri):
        """poly split along the planes carrying the trace of a mesh triangle (vertex ids,
        plane id, its neighbours' plane ids) on poly's plane: the triangle's own plane
        where it meets that plane in a segment, its neighbours' planes where it lies on it."""
        vids, tp, around = tri
        if tp == plane:
            cuts = around
        else:
            st = [self.sign(v, plane) for v in vids]
            cuts = [tp] if min(st) < 0 < max(st) or st.count(0) >= 2 else []
        pieces = [poly]
        for p in cuts:
            out = []
            for q in pieces:
                signs = [self.sign(v, p) for v in q]
                out += [r for r in self.split(q, p, 1) if len(r) >= 3] if min(signs) < 0 < max(signs) else [q]
            pieces = out
        return pieces

    def repair(self, poly):
        """poly with every vertex lying on one of its edges inserted into that edge."""
        out = []
        for i, a in enumerate(poly):
            b = poly[(i + 1) % len(poly)]
            out.append(a)
            between = self.on_line(self.on[a] & self.on[b]) - {a, b}
            if between:
                pa, d = self.pos[a], self.pos[b] - self.pos[a]
                length = d @ d
                tol = self.merge / np.sqrt(length)
                ts = sorted(((self.pos[v] - pa) @ d / length, v) for v in between)
                out += [v for t, v in ts if tol < t < 1 - tol]
        return out

    def fan(self, poly):
        """Triangles of a convex polygon; a fan from its centre when vertices lie on its sides."""
        n = len(poly)
        if not any(len(self.on[poly[i - 1]] & self.on[poly[i]] & self.on[poly[(i + 1) % n]]) > 1 for i in range(n)):
            return [(poly[0], poly[i], poly[i + 1]) for i in range(1, n - 1)]
        c = len(self.pos)
        self.pos.append(np.mean([self.pos[v] for v in poly], 0))
        self.on.append(set())
        return [(c, poly[i], poly[(i + 1) % n]) for i in range(n)]


def distinct(loop):
    """loop without vertices repeating the one before them."""
    return [v for k, v in enumerate(loop) if v != loop[k - 1]]


class Faceted:
    """A shape's facets in a kernel as (vertex loop, plane id, side), side -1 where the
    facet faces against its plane's stored normal, and its bounds. A convex shape gives
    the half-spaces (plane id, side) whose intersection is the shape; a mesh gives its
    triangles as (vertex ids, plane id, the plane ids of the triangles sharing an edge
    with it). The clip box gives its half-spaces only."""

    def __init__(self, kernel, shape):
        self.mesh = None if shape.convex else shape
        self.facets, self.tris, self.half = [], [], None
        if not hasattr(shape, "facets"):
            ids, sides = kernel.planes(shape.normals, shape.offsets)
            self.half = [(int(p), int(s)) for p, s in zip(ids, sides)]
            self.half_map = dict(self.half)
            self.lo, self.hi = np.full(3, -np.inf), np.full(3, np.inf)
            return
        points, polygons = shape.facets()
        points, inverse = np.unique(np.asarray(points, float), axis=0, return_inverse=True)
        inverse = inverse.ravel()
        self.lo, self.hi = points.min(0), points.max(0)
        tiny = 1e-12 * ((self.hi - self.lo) @ (self.hi - self.lo))
        loops, normals = [], []
        for poly in polygons:
            loop = distinct([int(inverse[i]) for i in poly])
            p = points[loop]
            n = np.cross(p, np.roll(p, -1, 0)).sum(0) if len(loop) >= 3 else np.zeros(3)
            if np.linalg.norm(n) > tiny:
                loops.append(loop)
                normals.append(n / np.linalg.norm(n))
        normals = np.array(normals)
        if self.mesh is None:
            interior = points.mean(0)
            for f, loop in enumerate(loops):
                if normals[f] @ (points[loop].mean(0) - interior) < 0:
                    loops[f], normals[f] = loop[::-1], -normals[f]
        offsets = np.array([n @ points[loop].mean(0) for n, loop in zip(normals, loops)])
        ids, sides = kernel.planes(normals, offsets)
        incident = defaultdict(set)
        for f, loop in enumerate(loops):
            for i in loop:
                incident[i].add(int(ids[f]))
        vid = {i: kernel.vertex(points[i], planes) for i, planes in incident.items()}
        self.facets = [([vid[i] for i in loop], int(ids[f]), int(sides[f])) for f, loop in enumerate(loops)]
        if self.mesh is None:
            self.half = [(int(p), int(s)) for p, s in zip(ids, sides)]
            self.half_map = dict(self.half)
        else:
            faces_at = defaultdict(set)
            for f, loop in enumerate(loops):
                for k in range(3):
                    faces_at[frozenset((loop[k], loop[k - 1]))].add(f)
            self.tris = []
            for f, (poly, plane, _) in enumerate(self.facets):
                loop = loops[f]
                around = {int(ids[g]) for k in range(3) for g in faces_at[frozenset((loop[k], loop[k - 1]))]} - {plane}
                self.tris.append((poly, plane, sorted(around)))
            corners = points[np.array(loops)]
            self.tri_lo, self.tri_hi = corners.min(1), corners.max(1)


def closed_cut(own, cutters, clip_box=None):
    """(vertices, faces) of the region inside own and its masks and outside every
    cutter present (inside its masks): the parts of each surface bounding it, wound
    outward. own and cutters give .shape, .masks and .any_mode; clip_box is a
    ConvexShape that removes its side and gets no wall."""
    points, _ = own.shape.facets()
    points = np.asarray(points, float)
    scale = np.linalg.norm(points.max(0) - points.min(0))
    kernel = Kernel(scale)
    shapes, index = [], {}

    def add(shape):
        if id(shape) not in index:
            index[id(shape)] = len(shapes)
            shapes.append(Faceted(kernel, shape))
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
        for poly, plane, _ in shapes[k].facets:
            pts = np.array([kernel.pos[v] for v in poly])
            lo, hi = pts.min(0), pts.max(0)
            if k != own_i and (np.any(lo > body.hi) or np.any(hi < body.lo)):
                continue
            work = [(poly, {k: True})]
            for j, other in enumerate(shapes):
                if j == k or np.any(lo > other.hi + kernel.eps) or np.any(hi < other.lo - kernel.eps):
                    continue
                if other.mesh is None:
                    work = [(r, {**lab, j: inside}) for q, lab in work
                            for r, inside in kernel.refine(q, plane, other.half) if not dead(j, inside, plane)]
                    continue
                near = np.flatnonzero(np.all((other.tri_lo <= hi + kernel.eps) & (other.tri_hi >= lo - kernel.eps), axis=1))
                for t in near:
                    tri = other.tris[t]
                    work = [(r, lab) for q, lab in work for r in kernel.cut_by_triangle(q, plane, tri)]
            pieces += [(q, plane, k, lab) for q, lab in work]
    loops = classify_pieces(kernel, shapes, pieces, region, scale)
    faces = [t for poly in loops for t in kernel.fan(kernel.repair(poly))]
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    faces = np.array(faces)
    used, faces = np.unique(faces, return_inverse=True)
    return np.array(kernel.pos)[used], faces.reshape(-1, 3)


def classify_pieces(kernel, shapes, pieces, region, scale):
    """The pieces that bound the region, wound outward. Each piece's status in every
    shape is taken just above and just below its plane: exactly from its labels for a
    convex shape, by the mesh's own inside test for a mesh. A piece lying on the
    surface of a shape listed before its own is left to that shape."""
    kept, centre, step, normal = [], [], [], []
    for q, plane, k, lab in pieces:
        pts = np.array([kernel.pos[v] for v in q])
        n = np.cross(pts, np.roll(pts, -1, 0)).sum(0)
        area = 0.5 * np.linalg.norm(n)
        if area <= kernel.merge ** 2:
            continue
        perimeter = np.linalg.norm(pts - np.roll(pts, -1, 0), axis=1).sum()
        kept.append((q, plane, k, lab, n))
        centre.append(pts.mean(0))
        step.append(min(1e-6 * scale, 0.2 * area / perimeter))
        normal.append(kernel.normal[plane])
    if not kept:
        return []
    centre, offset = np.array(centre), np.array(step)[:, None] * np.array(normal)
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
