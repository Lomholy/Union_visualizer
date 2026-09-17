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

# C type keywords that can prefix a DECLARE/USERVARS statement before the
# actual "name = expr" assignment, e.g. "double result = 1.0".
_C_TYPE_KEYWORDS = {
    "double", "float", "int", "long", "short", "char", "unsigned", "signed", "const",
}

# McStas instrument-parameter type keywords, a superset of the C ones above
# since "string" is a valid McStas parameter type despite not being a C type.
_PARAM_TYPE_KEYWORDS = _C_TYPE_KEYWORDS | {"string"}

_RAW_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_]\w*)\s*=\s*(.+)$", re.S)
_QUOTED_STRING_RE = re.compile(r'^"(.*)"$', re.S)


def _resolved_value(entry):
    value = entry.value
    if value == "" and getattr(entry, "type", None) in NUMERIC_DECLARE_TYPES:
        return 0.0
    return value


def _split_top_level_statements(text):
    """Split C-ish source text on ';' characters that aren't nested inside
    ()/[] or a string/char literal, yielding each trimmed statement with
    its trailing ';' removed. Doesn't need to understand braces/control
    flow - it only has to isolate individual statements well enough to
    check each one against a plain assignment pattern."""
    statements = []
    depth = 0
    quote = None
    start = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "'\"":
            quote = c
        elif c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        elif c == ";" and depth <= 0:
            piece = text[start:i].strip()
            if piece:
                statements.append(piece)
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _strip_c_type_prefix(statement, type_keywords=_C_TYPE_KEYWORDS):
    while True:
        head, sep, rest = statement.partition(" ")
        if sep and head in type_keywords:
            statement = rest.strip()
        else:
            return statement


def _recover_raw_declare_block(raw_lines, var_map):
    """mcstasscript reports DECLARE/USERVARS content it can't parse into a
    typed variable as one raw string per physical source line - functions,
    structs, #define/#pragma lines (see append_declare() in mcstasscript's
    interface/instr.py). Most of that genuinely has no computable value.
    But mcstasscript's own "is this a function?" heuristic (a line with "("
    and no ";" yet) can also misfire on an ordinary multi-line initializer,
    e.g. "double result = fabs(\n  -5.0\n);", which becomes three raw-string
    entries even though it's a real, evaluable assignment. Rejoin the block
    and try to recover a "name = expr;" statement from it before giving up
    on the rest, instead of silently discarding a computable value."""
    joined = "\n".join(raw_lines)
    if "{" in joined or "}" in joined:
        # A genuine function/struct body always has a brace pair. Anything
        # assignment-shaped inside one is function-local C, not an
        # instrument-global DECLARE variable, so leave the block alone.
        return
    for statement in _split_top_level_statements(joined):
        statement = _strip_c_type_prefix(statement)
        match = _RAW_ASSIGNMENT_RE.match(statement)
        if not match:
            # No "name = expr" shape at all: a #define/#pragma line, a bare
            # declaration with no initializer, a function signature. This
            # is expected, non-computable content, not a parser gap.
            continue
        name, expr = match.groups()
        expr = "".join(expr.split())
        try:
            var_map[name] = eval_expr(expr, var_map)
        except Exception as e:
            print(f"Warning: Failed to evaluate {statement};: {e}")


def _populate_declare_vars(entries, var_map):
    """Populate var_map from one DECLARE/USERVARS source list (declare_list
    or user_var_list), recovering any evaluable assignment hiding in a run
    of mcstasscript's raw-string fallback entries rather than crashing on
    them (they don't have a '.value' attribute) or silently dropping them."""
    i = 0
    n = len(entries)
    while i < n:
        entry = entries[i]
        if isinstance(entry, str):
            j = i
            while j < n and isinstance(entries[j], str):
                j += 1
            _recover_raw_declare_block(entries[i:j], var_map)
            i = j
            continue
        var_map[entry.name] = _resolved_value(entry)
        i += 1


def _recover_raw_parameter(raw_text, var_map):
    """instr.parameters entries come from mcstasscript's DEFINE
    INSTRUMENT(...) parser (read_definition.py), a different code path
    from DECLARE's raw-string fallback: every parameter it finds is turned
    into a proper Parameter object, confirmed by reading that reader and
    reproducing typed defaults like "int height=3" and
    "string filename=\"myfile.txt\"" directly. But treating "this never
    happens" as the actual safety net is exactly the mistake the DECLARE
    fix above exists to avoid, so a raw-string parameter entry is parsed
    the same way as a DECLARE default instead of being dropped.

    A quoted string default (McStas's "string" parameter type) is taken
    as a literal value rather than run through eval_expr, and has its
    surrounding quotes stripped - unlike mcstasscript's own object path,
    which leaves them embedded in the value."""
    statement = _strip_c_type_prefix(raw_text.strip(), _PARAM_TYPE_KEYWORDS)
    match = _RAW_ASSIGNMENT_RE.match(statement)
    if not match:
        # A bare parameter name with no default value has no computable
        # value at preprocessing time - expected, not a parser gap.
        return
    name, expr = match.groups()
    expr = expr.strip()
    quoted = _QUOTED_STRING_RE.match(expr)
    if quoted:
        var_map[name] = quoted.group(1)
        return
    expr = "".join(expr.split())
    try:
        var_map[name] = eval_expr(expr, var_map)
    except Exception as e:
        print(f"Warning: Failed to evaluate parameter {statement};: {e}")


def create_var_map(instr: ms.McStas_instr):
    var_map = {}
    _populate_declare_vars(list(instr.declare_list), var_map)
    _populate_declare_vars(list(instr.user_var_list), var_map)

    for param in instr.parameters:
        # Parameters come from mcstasscript's DEFINE INSTRUMENT(...) parser,
        # a different code path from the freeform DECLARE/USERVARS
        # fallback above - see _recover_raw_parameter for why a raw-string
        # entry here is recovered rather than assumed impossible.
        if isinstance(param, str):
            _recover_raw_parameter(param, var_map)
            continue
        var_map[param.name] = _resolved_value(param)

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

    def local_rotation(comp):
        rx, ry, rz = np.array(comp.ROTATED_data) * np.pi / 180
        return rotation_matrix(rx, ry, rz)

    def find_relative_parents(comp, instr):
        # AT and ROTATED are resolved against independent reference
        # components in real McStas (see cogen_comp_init_position() in
        # McCode's cogen.c.in): a component's world position always uses
        # its AT reference, and its world rotation always uses its ROTATED
        # reference, even when the two differ.
        AT_rel = comp.AT_relative
        # When ROTATED is omitted in the source, mcstasscript leaves
        # ROTATED_relative at its "ABSOLUTE" default, but real McStas
        # instead defaults an omitted ROTATED to the *same* reference as
        # AT (instrument.y: orientation_rel = isdefault ? place_rel : ...).
        # comp.ROTATED_specified tells us whether ROTATED was actually
        # written in the source, so we can apply the correct default
        # ourselves instead of trusting mcstasscript's "ABSOLUTE".
        ROT_rel = comp.ROTATED_relative if comp.ROTATED_specified else comp.AT_relative

        if AT_rel.startswith("RELATIVE"):
            AT_rel = AT_rel.split(" ")[1]
        if ROT_rel.startswith("RELATIVE"):
            ROT_rel = ROT_rel.split(" ")[1]

        if AT_rel == "PREVIOUS" or ROT_rel == "PREVIOUS":
            idx = instr.component_list.index(comp)
            if idx == 0:
                # Real McStas's own grammar (instrument.y: compref -> PREVIOUS)
                # handles a first-component PREVIOUS reference the same way:
                # print a warning and fall back to ABSOLUTE, rather than a
                # hard error. This isn't just theoretical - several real
                # instruments in the McCode corpus do this (e.g. ILL_H53_D16,
                # ILL_H22_D1B), presumably as a copy-paste artifact, and McStas
                # itself still compiles and runs them.
                print(
                    f"Warning: Component '{comp.name}' is RELATIVE PREVIOUS "
                    f"but is the first component in the instrument; no "
                    f"previous component exists. Using ABSOLUTE, matching "
                    f"McStas's own fallback for this case."
                )
                if AT_rel == "PREVIOUS":
                    AT_rel = "ABSOLUTE"
                if ROT_rel == "PREVIOUS":
                    ROT_rel = "ABSOLUTE"
            else:
                previous_name = instr.component_list[idx - 1].name
                if AT_rel == "PREVIOUS":
                    AT_rel = previous_name
                if ROT_rel == "PREVIOUS":
                    ROT_rel = previous_name

        return AT_rel, ROT_rel

    def compute_matrix(comp, AT_parent, ROT_parent, world):
        unresolved = [v for v in comp.ROTATED_data if isinstance(v, str)]
        unresolved += [v for v in comp.AT_data if isinstance(v, str)]
        if unresolved:
            raise ValueError(
                f"Component '{comp.name}': could not resolve AT/ROTATED "
                f"expression(s) to numeric values: {unresolved}. Check for "
                f"undefined variables or missing constants in "
                f"eval_expr/MCSTAS_CONSTANTS."
            )

        R_parent = np.eye(3) if ROT_parent == "ABSOLUTE" else world[ROT_parent][:3, :3]
        R = R_parent @ local_rotation(comp)

        AT_data = np.array(comp.AT_data, dtype=float)
        if AT_parent == "ABSOLUTE":
            t = AT_data
        else:
            t = world[AT_parent][:3, 3] + world[AT_parent][:3, :3] @ AT_data

        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        return M

    world = {}
    remaining = list(instr.component_list)

    while remaining:
        progressed = False

        for comp in remaining[:]:
            AT_parent, ROT_parent = find_relative_parents(comp, instr)

            at_ready = AT_parent == "ABSOLUTE" or AT_parent in world
            rot_ready = ROT_parent == "ABSOLUTE" or ROT_parent in world

            if at_ready and rot_ready:
                world[comp.name] = compute_matrix(comp, AT_parent, ROT_parent, world)
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
