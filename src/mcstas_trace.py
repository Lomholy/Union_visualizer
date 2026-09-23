"""Run a McStas instrument in trace mode (mcrun --trace) and turn every
component's MCDISPLAY drawing into world-space geometry. Everything returned
is plain numpy/trimesh data, so it can be built in a worker process and sent
back to the GUI."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import trimesh

from preprocess import execute_mcstasscript_file, load_McStas_file


class McrunNotFoundError(RuntimeError):
    """Raised when the mcrun binary can't be located."""


class TraceError(RuntimeError):
    """Raised when mcrun fails to compile or run the instrument."""


# Union_master draws a wireframe of every Union volume, which unviz already
# meshes properly itself.
SKIPPED_COMPONENT_TYPES = {"Union_master"}

CIRCLE_SEGMENTS = 48


@dataclass
class TraceComponent:
    name: str
    comp_type: str
    world_matrix: np.ndarray
    segments: np.ndarray = field(default_factory=lambda: np.zeros((0, 2, 3)))
    solid: trimesh.Trimesh = None

    @property
    def is_arm(self):
        return self.comp_type == "Arm"


# ==============================================================================
# ============================== Running mcrun ==================================
# ==============================================================================


def find_mcrun():
    override = os.environ.get("MCRUN")
    if override:
        if shutil.which(override) or os.path.isfile(override):
            return override
        raise McrunNotFoundError(
            f"MCRUN is set to '{override}', but no executable was found there."
        )
    found = shutil.which("mcrun")
    if found:
        return found
    raise McrunNotFoundError(
        "mcrun not found on PATH. It ships with the McStas install (the "
        "'mcstas' conda package in unviz_env.yml) - activate that environment, "
        "or set MCRUN to its path."
    )


def instr_file_for(input_file):
    """The .instr file mcrun should run for input_file. A McStasScript .py
    file is written out to a per-input cache directory first."""
    if input_file.endswith(".instr"):
        return os.path.abspath(input_file)
    instr = execute_mcstasscript_file(input_file)
    key = hashlib.sha1(os.path.abspath(input_file).encode()).hexdigest()[:12]
    out_dir = os.path.join(tempfile.gettempdir(), "unviz_trace", key)
    os.makedirs(out_dir, exist_ok=True)
    instr.input_path = out_dir
    instr.write_full_instrument()
    return os.path.join(out_dir, instr.name + ".instr")


def run_trace(instr_file, ncount=0, seed=None, params=(), cwd=None, timeout=300):
    """Run instr_file with mcrun --trace=2 and return its stdout. cwd
    defaults to the instrument's own directory, like mcrun/mcdisplay, so
    instrument-local .comp and data files resolve."""
    cmd = [
        find_mcrun(), instr_file, "--trace=2", "--no-output-files",
        f"--ncount={int(ncount)}", "-y", *params,
    ]
    if seed:
        cmd.append(f"--seed={int(seed)}")
    if cwd is None:
        cwd = os.path.dirname(instr_file) or "."
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        raise TraceError(f"mcrun did not finish within {timeout} s")
    if result.returncode != 0 or "INSTRUMENT END:" not in result.stdout:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-15:])
        if "symbol(s) not found" in tail:
            tail += (
                "\n\nLinking failed - on macOS this usually means SDKROOT is "
                "not set; launch from an activated 'unviz' environment."
            )
        raise TraceError(f"mcrun failed on '{instr_file}':\n{tail}")
    return result.stdout


def component_types(input_file, force_pygen=False):
    """{component name: component type}. The trace output only has names."""
    try:
        instr = load_McStas_file(input_file, force_pygen=force_pygen)
    except Exception as e:
        print(f"Warning: could not read component types from '{input_file}': {e}")
        return {}
    return {comp.name: comp.component_name for comp in instr.component_list}


def trace_instrument(input_file, params=(), force_pygen=False):
    """Entry point: run input_file in trace mode and return its drawn
    components as {name: TraceComponent}."""
    instr_file = instr_file_for(input_file)
    cwd = os.path.dirname(os.path.abspath(input_file))
    trace_text = run_trace(instr_file, params=params, cwd=cwd)
    return parse_trace(trace_text, component_types(input_file, force_pygen))


# ==============================================================================
# ============================== Parsing ========================================
# ==============================================================================


_QUOTED_NAME_RE = re.compile(r'"(.*)"')
_DRAWCALL_RE = re.compile(r"^(\w+)\s*\((.*)\)\s*$")


def _floats(text):
    return [float(v) for v in text.split(",")]


def _pos_to_matrix(values):
    """POS: prints McStas's own rotation matrix, the transpose of the
    local-to-world rotation preprocess.compute_world_matrices uses."""
    x, y, z, *rot = values
    M = np.eye(4)
    M[:3, :3] = np.array(rot).reshape(3, 3).T
    M[:3, 3] = (x, y, z)
    return M


def parse_trace(text, comp_types=None):
    comp_types = comp_types or {}
    matrices = {}
    order = []
    drawcalls = {}
    current = None
    drawing = None

    for line in text.splitlines():
        if line.startswith("COMPONENT:"):
            current = _QUOTED_NAME_RE.search(line).group(1)
        elif line.startswith("POS:") and current is not None:
            matrices[current] = _pos_to_matrix(_floats(line[4:]))
            order.append(current)
            current = None
        elif line.startswith("MCDISPLAY: component "):
            drawing = line[len("MCDISPLAY: component "):].strip()
            drawcalls.setdefault(drawing, [])
        elif line.startswith("MCDISPLAY: ") and drawing is not None:
            drawcalls[drawing].append(line[len("MCDISPLAY: "):].strip())
        elif line.startswith("ENTER:"):
            break

    components = {}
    for name in order:
        comp_type = comp_types.get(name, "")
        if comp_type in SKIPPED_COMPONENT_TYPES:
            continue
        segments, solid = _build_component_geometry(drawcalls.get(name, []), name)
        if len(segments) == 0 and solid is None:
            continue
        M = matrices[name]
        if len(segments):
            flat = segments.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
            segments = flat.reshape(-1, 2, 3)
        if solid is not None:
            solid.apply_transform(M)
        components[name] = TraceComponent(name, comp_type, M, segments, solid)
    return components


# ------------------------------ MCDISPLAY -------------------------------------


def _loop_segments(points):
    points = np.asarray(points, dtype=float)
    return np.stack([points[:-1], points[1:]], axis=1)


def _circle_points(center, radius, normal):
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    helper = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(normal, helper)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    theta = np.linspace(0, 2 * np.pi, CIRCLE_SEGMENTS + 1)
    return np.asarray(center) + radius * (
        np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)
    )


_PLANE_NORMALS = {
    "xy": (0, 0, 1), "yx": (0, 0, 1),
    "xz": (0, 1, 0), "zx": (0, 1, 0),
    "yz": (1, 0, 0), "zy": (1, 0, 0),
}


def _place(mesh, center, axis, mesh_axis):
    """Rotate mesh_axis of mesh onto axis, then move it to center. A zero
    axis leaves the orientation alone."""
    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) > 0:
        mesh.apply_transform(trimesh.geometry.align_vectors(mesh_axis, axis))
    mesh.apply_translation(center)
    return mesh


def _flat_ring(outer, inner):
    theta = np.linspace(0, 2 * np.pi, CIRCLE_SEGMENTS, endpoint=False)
    ring = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    n = len(theta)
    idx = np.arange(n)
    nxt = (idx + 1) % n
    if inner <= 0:
        vertices = np.vstack([[0, 0, 0], outer * ring])
        faces = np.stack([np.zeros(n, int), idx + 1, nxt + 1], axis=1)
    else:
        vertices = np.vstack([outer * ring, inner * ring])
        faces = np.vstack([
            np.stack([idx, nxt, nxt + n], axis=1),
            np.stack([idx, nxt + n, idx + n], axis=1),
        ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _hollow_box(w, h, l, t):
    """A box open along local z with walls of thickness t, like a guide."""
    t = min(t, w / 2, h / 2)
    walls = [
        trimesh.creation.box(extents=(t, h, l)).apply_translation(((w - t) / 2, 0, 0)),
        trimesh.creation.box(extents=(t, h, l)).apply_translation((-(w - t) / 2, 0, 0)),
        trimesh.creation.box(extents=(w - 2 * t, t, l)).apply_translation((0, (h - t) / 2, 0)),
        trimesh.creation.box(extents=(w - 2 * t, t, l)).apply_translation((0, -(h - t) / 2, 0)),
    ]
    return trimesh.util.concatenate(walls)


def _polyhedron(json_text):
    data = json.loads(json_text)
    vertices = np.asarray(data["vertices"], dtype=float)
    faces = []
    for entry in data["faces"]:
        face = entry["face"] if isinstance(entry, dict) else entry
        faces.extend([face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1))
    if len(vertices) == 0 or not faces:
        return None
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _drawcall_geometry(call):
    """(segments or None, solid or None) for one MCDISPLAY draw call, in the
    component's local frame. Solids follow mcdisplay-webgl's --trace=2
    conventions: centred on (x,y,z) with their local y axis along (nx,ny,nz)."""
    if call.startswith("polyhedron"):
        return None, _polyhedron(call[len("polyhedron"):].strip())

    match = _DRAWCALL_RE.match(call)
    if not match:
        return None, None
    name, arg_text = match.groups()
    args = [a.strip().strip("'\"") for a in arg_text.split(",")]

    if name == "multiline":
        values = np.array(args[1:], dtype=float).reshape(-1, 3)
        return (_loop_segments(values) if len(values) > 1 else None), None
    if name in ("mcdisline", "line", "dashed_line"):
        values = np.array(args[:6], dtype=float).reshape(2, 3)
        return values[None], None
    if name == "mcdisrectangle":
        plane = args[0]
        x, y, z, w, h = (float(a) for a in args[1:6])
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2],
                            [-w / 2, h / 2], [-w / 2, -h / 2]])
        axes = {"xy": (0, 1), "xz": (0, 2), "yz": (2, 1)}.get(plane, (0, 1))
        points = np.zeros((5, 3))
        points[:, axes[0]] = corners[:, 0]
        points[:, axes[1]] = corners[:, 1]
        return _loop_segments(points + (x, y, z)), None
    if name == "mcdiscircle":
        center = [float(a) for a in args[1:4]]
        radius = float(args[4])
        normal = _PLANE_NORMALS.get(args[0], (0, 0, 1))
        return _loop_segments(_circle_points(center, radius, normal)), None
    if name == "mcdisnew_circle":
        x, y, z, r, nx, ny, nz = (float(a) for a in args[:7])
        return _loop_segments(_circle_points((x, y, z), r, (nx, ny, nz))), None

    values = [float(a) for a in args]
    up = (0, 1, 0)
    if name == "mcdisbox":
        x, y, z, w, h, l, t, nx, ny, nz = values
        if min(w, h, l) <= 0:
            return None, None
        box = _hollow_box(w, h, l, t) if t > 0 else trimesh.creation.box(extents=(w, h, l))
        return None, _place(box, (x, y, z), (nx, ny, nz), up)
    if name == "mcdiscylinder":
        x, y, z, r, h, t, nx, ny, nz = values
        if r <= 0 or h <= 0:
            return None, None
        if 0 < t < r:
            mesh = trimesh.creation.annulus(r_min=r - t, r_max=r, height=h, sections=CIRCLE_SEGMENTS)
        else:
            mesh = trimesh.creation.cylinder(radius=r, height=h, sections=CIRCLE_SEGMENTS)
        return None, _place(mesh, (x, y, z), (nx, ny, nz), (0, 0, 1))
    if name == "mcdiscone":
        x, y, z, r, h, nx, ny, nz = values
        if r <= 0 or h == 0:
            return None, None
        cone = trimesh.creation.cone(radius=r, height=abs(h), sections=CIRCLE_SEGMENTS)
        cone.apply_translation((0, 0, -abs(h) / 2))
        return None, _place(cone, (x, y, z), (nx, ny, nz), (0, 0, 1))
    if name == "mcdissphere":
        x, y, z, r = values[:4]
        if r <= 0:
            return None, None
        return None, trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation((x, y, z))
    if name == "mcdisdisc":
        x, y, z, r, nx, ny, nz = values
        if r <= 0:
            return None, None
        return None, _place(_flat_ring(r, 0), (x, y, z), (nx, ny, nz), (0, 0, 1))
    if name == "mcdisannulus":
        x, y, z, outer, inner, nx, ny, nz = values
        if outer <= 0:
            return None, None
        return None, _place(_flat_ring(outer, inner), (x, y, z), (nx, ny, nz), (0, 0, 1))
    return None, None


def _build_component_geometry(calls, comp_name):
    segments = []
    solids = []
    for call in calls:
        if call.startswith("magnify"):
            continue
        try:
            segs, solid = _drawcall_geometry(call)
        except Exception as e:
            print(f"Warning: Component '{comp_name}': could not draw '{call[:60]}': {e}")
            continue
        if segs is not None:
            segments.append(segs)
        if solid is not None:
            solids.append(solid)
    segments = np.concatenate(segments) if segments else np.zeros((0, 2, 3))
    solid = None
    if solids:
        solid = solids[0] if len(solids) == 1 else trimesh.util.concatenate(solids)
    return segments, solid
