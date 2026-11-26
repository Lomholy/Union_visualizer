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
        key = frozenset(planes)
        if key in self.ids:
            return self.ids[key]
        v = len(self.pos)
        self.pos.append(p)
        self.on.append(set())
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
        """The vertex where edge a-b crosses plane p, shared by every edge on that line."""
        line = self.on[a] & self.on[b]
        key = frozenset(line | {p})
        if key in self.ids:
            return self.ids[key]
        da = self.pos[a] @ self.normal[p] - self.offset[p]
        db = self.pos[b] @ self.normal[p] - self.offset[p]
        x = self.pos[a] + (self.pos[b] - self.pos[a]) * (da / (da - db))
        for v in self.on_line(line):
            if np.linalg.norm(self.pos[v] - x) < self.merge:
                self.add_planes(v, (p,))
                self.ids[key] = v
                return v
        return self.vertex(x, line | {p})

    def classify(self, poly, plane, p, side, probe):
        """+1 if polygon poly (on plane) lies outside the half-space side * (n.x - d) <= 0
        of plane p, -1 if inside, 0 if it straddles. A polygon lying on p takes the side
        that probe (a direction off the polygon) points to."""
        if plane != p:
            s = [side * self.sign(v, p) for v in poly]
            lo, hi = min(s), max(s)
            if lo < 0 < hi:
                return 0
            if lo < 0 or hi > 0:
                return 1 if hi > 0 else -1
        return 1 if side * (probe @ self.normal[p]) > 0 else -1

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
        return inside, outside

    def subtract(self, poly, plane, probe, cutter):
        """Convex pieces of poly outside the convex cutter (half-spaces, bounding box)."""
        half, lo, hi = cutter
        if lo is not None:
            pts = np.array([self.pos[v] for v in poly])
            if np.any(pts.min(0) > hi + self.eps) or np.any(pts.max(0) < lo - self.eps):
                return [poly]
        if any(self.classify(poly, plane, p, s, probe) > 0 for p, s in half):
            return [poly]
        kept = []
        for p, s in half:
            c = self.classify(poly, plane, p, s, probe)
            if c > 0:
                kept.append(poly)
                break
            if c == 0:
                poly, outside = self.split(poly, p, s)
                kept.append(outside)
        return kept

    def clip(self, poly, plane, probe, half):
        """The part of poly inside every half-space, or None."""
        for p, s in half:
            c = self.classify(poly, plane, p, s, probe)
            if c > 0:
                return None
            if c == 0:
                poly = self.split(poly, p, s)[0]
        return poly

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


class Faceted:
    """A convex shape's facets in a kernel: (vertex loop, plane id, side) facing outward,
    the half-spaces (plane id, side) whose intersection is the shape, and its bounds."""

    def __init__(self, kernel, points, polygons):
        interior = points.mean(0)
        normals, loops = [], []
        for poly in polygons:
            p = points[poly]
            n = np.cross(p, np.roll(p, -1, 0)).sum(0)
            n /= np.linalg.norm(n)
            if n @ (p.mean(0) - interior) < 0:
                poly, n = poly[::-1], -n
            normals.append(n)
            loops.append(list(poly))
        normals = np.array(normals)
        offsets = np.array([n @ points[poly].mean(0) for n, poly in zip(normals, loops)])
        ids, sides = kernel.planes(normals, offsets)
        incident = defaultdict(set)
        for f, poly in enumerate(loops):
            for i in poly:
                incident[i].add(int(ids[f]))
        vid = {i: kernel.vertex(points[i], planes) for i, planes in incident.items()}
        self.facets = [([vid[i] for i in poly], int(ids[f]), int(sides[f])) for f, poly in enumerate(loops)]
        self.half = [(int(p), int(s)) for p, s in zip(ids, sides)]
        self.lo, self.hi = points.min(0), points.max(0)


def closed_cut(own, cutters, clip_box=None):
    """(vertices, faces) of the region inside own and outside every cutter: own's kept
    surface and the cutters' walls, wound outward. own and cutters give facets();
    clip_box is a ConvexShape that removes its side and gets no wall."""
    points, polygons = own.facets()
    kernel = Kernel(np.linalg.norm(points.max(0) - points.min(0)))
    body = Faceted(kernel, points, polygons)
    shapes = [Faceted(kernel, *c.facets()) for c in cutters]
    blockers = [(f.half, f.lo, f.hi) for f in shapes]
    if clip_box is not None:
        ids, sides = kernel.planes(clip_box.normals, clip_box.offsets)
        blockers.append(([(int(p), int(s)) for p, s in zip(ids, sides)], None, None))
    loops = []
    for poly, plane, side in body.facets:
        probe = -side * kernel.normal[plane]
        pieces = [poly]
        for cutter in blockers:
            pieces = [q for piece in pieces for q in kernel.subtract(piece, plane, probe, cutter)]
        loops += pieces
    own_planes = {p for p, _ in body.half}
    for i, shape in enumerate(shapes):
        others = blockers[:i] + blockers[i + 1:]
        for poly, plane, side in shape.facets:
            pts = np.array([kernel.pos[v] for v in poly])
            if plane in own_planes or np.any(pts.min(0) > body.hi) or np.any(pts.max(0) < body.lo):
                continue
            probe = side * kernel.normal[plane]
            piece = kernel.clip(poly, plane, probe, body.half)
            if piece is None:
                continue
            pieces = [piece]
            for cutter in others:
                pieces = [q for p in pieces for q in kernel.subtract(p, plane, probe, cutter)]
            loops += [q[::-1] for q in pieces]
    faces = [t for poly in loops for t in kernel.fan(kernel.repair(poly))]
    if not faces:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    faces = np.array(faces)
    used, faces = np.unique(faces, return_inverse=True)
    return np.array(kernel.pos)[used], faces.reshape(-1, 3)
