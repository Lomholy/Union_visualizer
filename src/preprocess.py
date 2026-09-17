import mcstasscript as ms
import mcstasscript.helper.mcstas_objects as mshelp
import operator
import ast
import re
import math
import numpy as np


# ==============================================================================
# ============================ Load in McStas file =============================
# ==============================================================================


def execute_mcstasscript_file(input_file):
    # Execute the input python file and extract instrument objects
    namespace = {}
    with open(input_file, "r") as f:
        code = f.read()

    exec(compile(code, input_file, "exec"), namespace)

    # Collect all mcstasscript instruments
    instruments = [
        obj for obj in namespace.values() if isinstance(obj, ms.McStas_instr)
    ]

    if len(instruments) != 1:
        raise ValueError(f"Expected exactly one instrument, found {len(instruments)}.")

    return instruments[0]


def get_union_geometries(instr: ms.McStas_instr):
    union_geometries = {}
    union_names = ["Union_cylinder",
                   "Union_box",
                   "Union_sphere",
                   "Union_cone",
                   "Union_mesh"]
    for comp in instr.component_list:
        if comp.component_name in union_names:
            union_geometries[comp.name] = comp
    return union_geometries


def load_McStas_file(input_file):
    if input_file.endswith(".py"):
        instr = execute_mcstasscript_file(input_file)
    elif input_file.endswith(".instr"):
        file = ms.McStas_file(input_file)
        instr = ms.McStas_instr("union_cad")
        file.add_to_instr(instr)
    return instr



# =============================================================================
# ====================== CONVERT PARAMETERS TO FLOATS =========================
# =============================================================================


# Supported operators
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}

# Supported unary operators
UNARY = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Allowed math functions/constants
MATH_ENV = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}

# McStas's own built-in constants (defined in its C runtime headers), which
# aren't part of Python's math module.
MCSTAS_CONSTANTS = {
    "PI": math.pi,
    "DEG2RAD": math.pi / 180,
    "RAD2DEG": 180 / math.pi,
}


def eval_expr(expr, var_map=None):
    if var_map is None:
        var_map = {}

    def _eval(node):
        if isinstance(node, ast.Constant):  # numbers
            return node.value

        elif isinstance(node, ast.BinOp):  # x + y
            return OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))

        elif isinstance(node, ast.UnaryOp):  # -x
            return UNARY[type(node.op)](_eval(node.operand))

        elif isinstance(node, ast.Name):  # variables
            if node.id in var_map:
                return var_map[node.id]
            elif node.id in MCSTAS_CONSTANTS:
                return MCSTAS_CONSTANTS[node.id]
            elif node.id in MATH_ENV:
                return MATH_ENV[node.id]
            else:
                raise ValueError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.Call):  # function calls
            func = _eval(node.func)
            args = [_eval(arg) for arg in node.args]
            return func(*args)

        else:
            raise TypeError(f"Unsupported expression: {expr}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def parse_param(expr, var_map):
    if type(expr) is str:
        expr = "".join(expr.split())
    try:
        return eval_expr(expr, var_map)
    except Exception:
        return expr


# C types for which mcstasscript reports an empty-string .value when a
# DECLARE/USERVARS variable has no initializer. McStas generates these as
# plain C globals, which zero-initialize, so we mirror that instead of
# treating the placeholder "" as a literal (string) value.
NUMERIC_DECLARE_TYPES = {"double", "float", "int", "long"}

# mcstasscript stores DECLARE/USERVARS content it can't parse into a typed
# variable as plain strings (one per physical source line) rather than
# DeclareVariable objects: function bodies, struct/union definitions, and
# #define/#pragma lines. Most of that genuinely has no computable value -
# it's "declared" (a function, a type, a macro) but never "initialized" in
# the sense we care about. But mcstasscript's own "is this a function?"
# heuristic (a line with "(" and no ";" yet) can also misfire on an ordinary
# multi-line variable initializer, e.g.:
#   double result = fabs(
#       -5.0
#   );
# which gets dumped as three raw-string entries even though eval_expr could
# trivially compute it. _recover_declare_assignments rejoins a run of these
# raw lines and tries to pull a real "name = expr;" out of it before giving
# up on the rest.
_DECLARE_TYPE_KEYWORDS = {
    "double", "float", "int", "long", "char", "short", "unsigned", "signed", "const",
}


def _strip_leading_declare_type(stmt):
    parts = stmt.split(None, 1)
    while len(parts) == 2 and parts[0] in _DECLARE_TYPE_KEYWORDS:
        stmt = parts[1]
        parts = stmt.split(None, 1)
    return stmt


def _iter_top_level_statements(text):
    """Split C-ish text on top-level ';' (i.e. not inside (), [], or a
    string/char literal), yielding each stripped statement (without the
    trailing ';'). Doesn't need to understand {}/if/for - it's only used to
    isolate individual statements for a plain assignment check."""
    depth = 0
    in_str = None
    start = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in ("'", '"'):
            in_str = c
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == ";" and depth <= 0:
            stmt = text[start:i].strip()
            if stmt:
                yield stmt
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        yield tail


_DECLARE_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", re.S)


def _recover_declare_assignments(raw_lines, var_map):
    text = "\n".join(raw_lines)
    if "{" in text or "}" in text:
        # A genuine function or struct body always has a brace pair (that's
        # how mcstasscript itself knows where the block ends - see
        # read_declare.py's bracket-counting loop). Statements *inside* one
        # are function-local, not instrument-global variables, so treating
        # them as top-level assignments would be wrong, not just noisy (e.g.
        # a helper function's "result = a + b;" is not a DECLARE variable).
        # The misclassified-multi-line-expression case this function exists
        # to recover (see module comment above) never has braces at all, so
        # this cleanly tells the two apart.
        return
    for stmt in _iter_top_level_statements(text):
        stmt = _strip_leading_declare_type(stmt)
        match = _DECLARE_ASSIGNMENT_RE.match(stmt)
        if not match:
            # No "name = expr" shape at all - a function signature/body line,
            # a struct definition, a #define/#pragma, a bare declaration with
            # no initializer, etc. This is the expected common case, not a
            # parser gap, so stay silent.
            continue
        name, expr = match.groups()
        expr = "".join(expr.split())
        try:
            var_map[name] = eval_expr(expr, var_map)
        except Exception as e:
            print(f"Warning: Failed to evaluate {stmt};: {e}")


def _add_declare_vars_to_map(entries, var_map):
    """Populate var_map from one DECLARE/USERVARS source list, recovering
    any evaluable assignment hiding in a run of mcstasscript's raw-string
    entries instead of silently discarding the whole run."""
    i = 0
    n = len(entries)
    while i < n:
        v = entries[i]
        if isinstance(v, str):
            j = i
            raw_lines = []
            while j < n and isinstance(entries[j], str):
                raw_lines.append(entries[j])
                j += 1
            _recover_declare_assignments(raw_lines, var_map)
            i = j
            continue
        value = v.value
        if value == "" and getattr(v, "type", None) in NUMERIC_DECLARE_TYPES:
            value = 0.0
        var_map[v.name] = value
        i += 1


def create_var_map(instr: ms.McStas_instr):
    var_map = {}
    _add_declare_vars_to_map(list(instr.declare_list), var_map)
    _add_declare_vars_to_map(list(instr.user_var_list), var_map)
    for v in instr.parameters:
        # Defensive: parameters are resolved from the DEFINE INSTRUMENT(...)
        # line, not the DECLARE-raw-string mechanism above, so this isn't
        # expected to ever be a bare string - but skip rather than crash if
        # it ever is.
        if isinstance(v, str):
            continue
        var_map[v.name] = v.value
    ASSIGNMENT_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+?)\s*;$")

    lines = instr.initialize_section.splitlines()

    for line in lines:
        line = line.strip()

        if not line or line.startswith("//"):
            continue

        match = ASSIGNMENT_RE.match(line)

        if not match:
            continue

        name, expr = match.groups()

        try:
            value = eval_expr(expr, var_map)
            var_map[name] = value
        except Exception as e:
            print(f"Warning: Failed to evaluate {line}: {e}")
    return var_map


def attempt_conversion(comp: mshelp.Component, instr: ms.McStas_instr, var_map: dict):
    # Parameters exist in the following spaces in each component:
    # AT vector, the ROT vector,
    # the component_parameters
    # First loop over comp params:
    for name in comp.parameter_names:
        value = getattr(comp, name)
        if isinstance(value, (str, int, float)):
            val = parse_param(value, var_map)
            setattr(comp, name, val)
        elif isinstance(value, (list, tuple, set)):
            converted = []

            for val in value:
                res = parse_param(val, var_map)
                converted.append(res)

            setattr(comp, name, type(value)(converted))
    # Then go over AT and ROT vector
    for i, val in enumerate(comp.AT_data):
        value = parse_param(val, var_map)
        comp.AT_data[i] = value
    for i, val in enumerate(comp.ROTATED_data):
        value = parse_param(val, var_map)
        comp.ROTATED_data[i] = value

    return comp


# =============================================================================
# =========================== CALCULATE WORLD MATRICES ========================
# =============================================================================


def compute_world_matrices(instr, verbose=False):
    """
    Returns:
        dict: {component_name_lower: 4x4 world matrix}
    """

    def rotation_matrix(rx, ry, rz):
        Rx = np.array(
            [
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)],
            ]
        )
        Ry = np.array(
            [
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)],
            ]
        )
        Rz = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1],
            ]
        )
        return Rx @ Ry @ Rz

    def find_relative(comp, instr):
        AT_rel = comp.AT_relative
        ROT_rel = comp.ROTATED_relative

        if AT_rel.startswith("RELATIVE"):
            AT_rel = AT_rel.split(" ")[1]
        if ROT_rel.startswith("RELATIVE"):
            ROT_rel = ROT_rel.split(" ")[1]

        if AT_rel.startswith("PREVIOUS") or ROT_rel.startswith("PREVIOUS"):
            idx = instr.component_list.index(comp)
            previous_name = instr.component_list[idx - 1].name
            if AT_rel.startswith("PREVIOUS"):
                AT_rel = previous_name
            if ROT_rel.startswith("PREVIOUS"):
                ROT_rel = previous_name

        if ROT_rel != "ABSOLUTE":
            return ROT_rel

        return AT_rel

    def local_matrix(comp):
        unresolved = [v for v in comp.ROTATED_data if isinstance(v, str)]
        unresolved += [v for v in comp.AT_data if isinstance(v, str)]
        if unresolved:
            raise ValueError(
                f"Component '{comp.name}': could not resolve AT/ROTATED "
                f"expression(s) to numeric values: {unresolved}. Check for "
                f"undefined variables or missing constants in "
                f"eval_expr/MCSTAS_CONSTANTS."
            )

        M = np.eye(4)
        rx, ry, rz = np.array(comp.ROTATED_data) * np.pi / 180
        M[:3, :3] = rotation_matrix(rx, ry, rz)
        M[:3, 3] = comp.AT_data
        return M

    world = {}
    remaining = list(instr.component_list)

    while remaining:
        progressed = False

        for comp in remaining[:]:
            rel = find_relative(comp, instr)

            # Absolute → no dependency
            if rel == "ABSOLUTE":
                world[comp.name] = local_matrix(comp)
                remaining.remove(comp)
                progressed = True
                continue

            # Relative → need parent first
            if rel in world:
                world[comp.name] = world[rel] @ local_matrix(comp)
                if verbose:
                    print(f"COMPONENT {comp.name} MATRIX IS : \n{world[comp.name]}")
                remaining.remove(comp)
                progressed = True

        if not progressed:
            missing = [c.name for c in remaining]
            raise RuntimeError(f"Unresolved relative transforms: {missing}")

    return world


def preprocess(input_file: str, verbose: bool):
    """
    Function to preprocess the input file.

    Returns:
        McStas_instr containing the processed instrument
        dict: {component_name_lower: 4x4 world matrix}
        list: Each union geometry in the instrument.
    """
    instr = load_McStas_file(input_file)
    var_map = create_var_map(instr)
    for comp in instr.component_list:
        comp = attempt_conversion(comp, instr, var_map)

    world_matrices = compute_world_matrices(instr, verbose)
    union_geometries = get_union_geometries(instr)
    return instr, world_matrices, union_geometries
