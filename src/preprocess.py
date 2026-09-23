import mcstasscript as ms
import mcstasscript.helper.mcstas_objects as mshelp
import operator
import ast
import os
import re
import math
import shutil
import subprocess
import tempfile
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


class PygenNotFoundError(RuntimeError):
    """Raised when the mcstas-pygen binary can't be located."""


def find_mcstas_pygen():
    """Locate the mcstas-pygen binary. It ships alongside mcstas/mcrun (e.g.
    as part of the same conda package that provides mcstasscript's backend),
    so PATH lookup is normally enough once that environment is active;
    MCSTAS_PYGEN lets it be pointed at explicitly otherwise."""
    override = os.environ.get("MCSTAS_PYGEN")
    if override:
        if shutil.which(override) or os.path.isfile(override):
            return override
        raise PygenNotFoundError(
            f"MCSTAS_PYGEN is set to '{override}', but no executable was found there."
        )
    found = shutil.which("mcstas-pygen")
    if found:
        return found
    raise PygenNotFoundError(
        "mcstas-pygen not found on PATH. It ships with the McStas install "
        "(e.g. via the 'mcstas' conda package used by unviz_env.yml) - "
        "activate that environment, or set MCSTAS_PYGEN to its path."
    )


def run_mcstas_pygen(instr_file, verbose=False):
    """Translate a .instr file into a McStasScript module via mcstas-pygen,
    the McStas-provided code generator that runs the instrument through the
    real McStas front-end (component definitions and all) instead of
    mcstasscript's own lightweight regex-based .instr reader. Useful for
    instruments the lightweight reader can't parse. Returns the path to the
    generated .py file, written into a fresh temp directory.

    Component search: standard/contrib components (including Union) come
    from the MCSTAS environment variable, same as mcrun/mcstas. mcstas-pygen
    3.8.5's own -I flag errors out instead of extending that path (verified
    against the installed binary - its --help text lists -I, but passing it
    always falls straight to the usage banner), so the only way left to pick
    up instrument-local .comp files is to run with the instrument's own
    directory as cwd, which is what mcrun does too.
    """
    pygen = find_mcstas_pygen()
    out_dir = tempfile.mkdtemp(prefix="union_viewer_pygen_")
    abs_instr_file = os.path.abspath(instr_file)
    instr_dir = os.path.dirname(abs_instr_file) or "."
    base = os.path.splitext(os.path.basename(instr_file))[0]
    out_file = os.path.join(out_dir, f"{base}_generated.py")
    cmd = [pygen, "-o", out_file, abs_instr_file]

    if verbose:
        print("Running:", " ".join(cmd), "(cwd=%s)" % instr_dir)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=instr_dir)
    if verbose and result.stdout:
        print(result.stdout)

    if result.returncode != 0 or not os.path.isfile(out_file):
        message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"mcstas-pygen failed to translate '{instr_file}':\n{message}"
        )
    return out_file


def _clear_pygen_unset_string_defaults(instr):
    """Undo a mismatch pygen's output otherwise creates against the rest of
    this codebase's expectations: McStas represents an omitted string
    parameter with the literal, unquoted token 0 (e.g. `string
    mask_string = 0` in a .comp DEFINE - a real string value is always
    quoted, e.g. '"non"', so the bare '0' is unambiguous). mcstasscript's
    lightweight .instr reader only sets attributes that are actually
    written in the .instr text, so an omitted one simply stays absent and
    reads back as the Component class's own default of None. pygen's
    generated code instead assigns every declared parameter explicitly,
    including that literal '0' sentinel for the ones the instrument never
    set - and code elsewhere (e.g. brep.py's `hasattr(comp, 'mask_string')
    and comp.mask_string != None` mask check) relies on the "still None"
    state to know a parameter was never given a real value. Rewriting
    each such sentinel back to None makes pygen-loaded components behave
    like natively-read ones for every one of those checks."""
    for comp in instr.component_list:
        for pname, ptype in comp.parameter_types.items():
            if ptype == "string" and getattr(comp, pname, None) == "0":
                setattr(comp, pname, None)


def execute_pygen_file(generated_file):
    """Load the McStasScript instrument built by a mcstas-pygen-generated
    module. Unlike a hand-written McStasScript script (which instantiates
    McStas_instr at module scope, what execute_mcstasscript_file scans
    for), pygen's output wraps construction in a make() factory function
    and guards its demo CLI behind `if __name__ == "__main__"` - so this
    calls make() directly rather than scanning module globals, and sets
    __name__ to something other than "__main__" so exec() doesn't run
    into that guard reading an undefined name."""
    with open(generated_file, "r") as f:
        code = f.read()

    namespace = {"__name__": "mcstas_pygen_module"}
    exec(compile(code, generated_file, "exec"), namespace)

    make = namespace.get("make")
    if make is None:
        raise ValueError(
            f"mcstas-pygen output '{generated_file}' has no make() function; "
            "unexpected mcstas-pygen output format."
        )
    instr = make()
    _clear_pygen_unset_string_defaults(instr)
    return instr


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


def load_McStas_file(input_file, force_pygen=False, verbose=False):
    """Load a .py (McStasScript) or .instr (native McStas) file into an
    McStas_instr. For .instr inputs there are two ways to reach the
    mcstas-pygen path instead of mcstasscript's own lightweight .instr
    reader: force_pygen=True always takes it (e.g. because a user enabled
    it in the GUI settings for an instrument they know the lightweight
    reader mishandles), and otherwise it's used automatically as a
    fallback if the lightweight reader raises."""
    if input_file.endswith(".py"):
        instr = execute_mcstasscript_file(input_file)
    elif input_file.endswith(".instr"):
        if force_pygen:
            instr = execute_pygen_file(run_mcstas_pygen(input_file, verbose=verbose))
        else:
            try:
                file = ms.McStas_file(input_file)
                instr = ms.McStas_instr("union_cad")
                file.add_to_instr(instr)
            except Exception as exc:
                print(
                    f"mcstasscript's built-in .instr reader failed on "
                    f"'{input_file}' ({exc}); falling back to mcstas-pygen."
                )
                instr = execute_pygen_file(
                    run_mcstas_pygen(input_file, verbose=verbose)
                )
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
    ast.Not: operator.not_,   # the "!" -> " not " substitution ahead of a
                               # single-line "if (cond) ..." turns C's "!"
                               # into an ast.UnaryOp(ast.Not, ...) node,
                               # which the existing UNARY dispatch handles
                               # for free once it's in this table.
}

# Comparison operators for "if (cond) name = expr;" conditions (2.1 part 2).
# Chained comparisons (a < b < c) are walked pairwise in eval_expr, mirroring
# how OPERATORS/UNARY are already node-type -> callable lookup tables.
COMPARE_OPERATORS = {
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

# Allowed math functions/constants
MATH_ENV = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}

# C-callable builtins that are Python builtins rather than math module
# members, so dir(math) above doesn't pick them up (e.g. abs() is fabs()
# in C/McStas terms, but McStas source uses "abs" directly).
MATH_ENV.update({
    "abs": abs,
    "min": min,
    "max": max,
})

# McStas's own built-in constants (defined in its C runtime headers), which
# aren't part of Python's math module. Values pulled directly from
# McCode/mccode/nlib/share/mcstas-r.h (lines 43-50) and general.h (line 21),
# not re-derived, so they match bit-for-bit what the real simulator uses.
MCSTAS_CONSTANTS = {
    "PI": math.pi,
    "DEG2RAD": math.pi / 180,
    "RAD2DEG": 180 / math.pi,
    "AA2MS": 629.622368,       # mcstas-r.h:43 - convert k[1/AA] to v[m/s]
    "MS2AA": 1.58825361e-3,    # mcstas-r.h:44 - convert v[m/s] to k[1/AA]
    "K2V": 629.622368,         # mcstas-r.h:45 - #define K2V AA2MS
    "V2K": 1.58825361e-3,      # mcstas-r.h:46 - #define V2K MS2AA
    "SE2V": 437.393377,        # mcstas-r.h:49 - convert sqrt(E)[meV] to v[m/s]
    "VS2E": 5.22703725e-6,     # mcstas-r.h:50 - convert (v[m/s])**2 to E[meV]
    "NA": 6.022137e23,         # general.h:21 - Avogadro's number (McStas's own value, not current CODATA)
    "NULL": 0,
}


class OpaqueRuntimeCall(Exception):
    """Raised by eval_expr when an expression calls a function that
    genuinely has no static value at preprocessing time - not a parser
    gap, just something only known once a simulation actually runs."""


# Calls that are fundamentally uncomputable ahead of time, verified against
# the mcstas-comps corpus (not speculative): dynamic-allocation calls
# (malloc, McStas's create_darr1d) and runtime-state reads (mcget_ncount() -
# the neutron count, only known once a simulation runs). These used to be
# reported identically to a typo'd variable name ("Unknown variable: X"),
# which is misleading - they're not bugs to fix, they're a fundamentally
# different category, so eval_expr raises OpaqueRuntimeCall for them
# instead of the generic ValueError a real unknown name gets.
OPAQUE_RUNTIME_CALLS = {"malloc", "create_darr1d", "mcget_ncount"}


def _resolve_comp_getpar(node, comp_context):
    """COMP_GETPAR(CompName, param) / COMP_GETPAR(PREVIOUS, param) is a
    McStas macro meaning "read param off the named (or previous)
    component" - directly answerable from instr.component_list, unlike a
    generic unsupported function call. comp_context is (instr, comp), comp
    being the component whose AT/ROTATED/parameters are currently being
    resolved (needed to find "the previous component")."""
    if comp_context is None:
        raise ValueError(f"COMP_GETPAR used outside a component context: {ast.unparse(node)}")
    instr, comp = comp_context
    if len(node.args) != 2 or not all(isinstance(a, ast.Name) for a in node.args):
        raise ValueError(f"Unsupported COMP_GETPAR usage: {ast.unparse(node)}")
    comp_token, param_token = node.args[0].id, node.args[1].id
    if comp_token == "PREVIOUS":
        idx = instr.component_list.index(comp)
        if idx == 0:
            raise ValueError(
                f"COMP_GETPAR(PREVIOUS, {param_token}) used on '{comp.name}', "
                f"the first component in the instrument; no previous "
                f"component exists"
            )
        target = instr.component_list[idx - 1]
    else:
        matches = [c for c in instr.component_list if c.name == comp_token]
        if not matches:
            raise ValueError(f"COMP_GETPAR: no component named '{comp_token}'")
        target = matches[0]
    if not hasattr(target, param_token):
        raise ValueError(
            f"COMP_GETPAR: component '{target.name}' has no parameter '{param_token}'"
        )
    return getattr(target, param_token)


def eval_expr(expr, var_map=None, comp_context=None):
    if var_map is None:
        var_map = {}

    def _eval(node):
        if isinstance(node, ast.Constant):  # numbers
            return node.value

        elif isinstance(node, ast.BinOp):  # x + y
            left, right = _eval(node.left), _eval(node.right)
            try:
                return OPERATORS[type(node.op)](left, right)
            except ZeroDivisionError:
                # A parameter hitting a division by zero here doesn't mean
                # the instrument is broken - it may not even be essential to
                # the Union geometry being visualized - so we warn and let
                # the rest of preprocessing continue rather than aborting
                # this assignment (and cascading into every later expression
                # that depends on it) the way an unhandled exception would.
                print(
                    f"Warning: Division by zero evaluating '{ast.unparse(node)}' "
                    f"in '{expr}'; defaulting to 0"
                )
                return 0

        elif isinstance(node, ast.UnaryOp):  # -x, or "not x" from a translated C "!x"
            return UNARY[type(node.op)](_eval(node.operand))

        elif isinstance(node, ast.Compare):  # a < b, chained a < b < c, etc.
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                if type(op) not in COMPARE_OPERATORS:
                    raise TypeError(f"Unsupported comparison operator: {type(op).__name__}")
                right = _eval(comparator)
                if not COMPARE_OPERATORS[type(op)](left, right):
                    return False
                left = right
            return True

        elif isinstance(node, ast.BoolOp):  # translated C "&&"/"||"
            if isinstance(node.op, ast.And):
                result = True
                for value in node.values:
                    result = _eval(value)
                    if not result:
                        return result
                return result
            elif isinstance(node.op, ast.Or):
                result = False
                for value in node.values:
                    result = _eval(value)
                    if result:
                        return result
                return result
            else:
                raise TypeError(f"Unsupported boolean operator: {type(node.op).__name__}")

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
            if isinstance(node.func, ast.Name) and node.func.id == "COMP_GETPAR":
                return _resolve_comp_getpar(node, comp_context)
            if isinstance(node.func, ast.Name) and node.func.id in OPAQUE_RUNTIME_CALLS:
                # Bail out before evaluating func/args - the args are often
                # not even valid Python (e.g. malloc(150*sizeof(char))),
                # and none of that matters since the call itself has no
                # static value regardless of what its arguments evaluate to.
                raise OpaqueRuntimeCall(node.func.id)
            func = _eval(node.func)
            args = [_eval(arg) for arg in node.args]
            return func(*args)

        elif isinstance(node, ast.Attribute):  # obj.field - struct member access
            # DECLARE-time C structs aren't modeled as Python objects here;
            # per the plan this is a deliberate, visible "left unresolved"
            # rather than an attempt to model arbitrary structs.
            raise ValueError(f"Unsupported: struct member access '{ast.unparse(node)}'")

        else:
            raise TypeError(f"Unsupported expression: {expr}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def parse_param(expr, var_map, comp_context=None):
    if type(expr) is not str:
        # Already numeric (mcstasscript resolved it itself) - nothing to
        # evaluate, and critically nothing to warn about either.
        return expr
    stripped = "".join(expr.split())
    try:
        return eval_expr(stripped, var_map, comp_context)
    except OpaqueRuntimeCall as e:
        print(f"Info: {expr} depends on runtime state ({e}()), left unresolved")
        return expr
    except Exception as e:
        # Unlike create_var_map, this used to fail completely silently -
        # invisible until it blew up later in compute_world_matrices, a
        # worse failure mode than a warning here and now.
        print(f"Warning: Failed to evaluate {expr}: {e}")
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
_ELSE_ASSIGNMENT_RE = re.compile(r"^else\s+([A-Za-z_]\w*)\s*=\s*(.+)$", re.S)
_NEGATION_RE = re.compile(r"!(?!=)")  # a bare "!", not part of "!="


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


def _c_bool_ops_to_python(text):
    """Textually translate C's boolean operators to Python's before handing
    a condition to eval_expr (ast.parse rejects "&&"/"||"/bare "!" outright,
    they aren't Python syntax at all). "!=" is left untouched - it's already
    valid Python and _NEGATION_RE's negative lookahead skips it."""
    text = text.replace("&&", " and ").replace("||", " or ")
    return _NEGATION_RE.sub(" not ", text)


def _split_if_statement(statement):
    """Split a statement starting with "if (" into (cond, rest), where rest
    is everything after the condition's *matching* close-paren. A naive
    regex like r"if\\s*\\((.+)\\)" stops at the first ")", which breaks on
    any condition that has parens of its own, e.g. "if ((a<b) && c) x=1" -
    so this walks the text counting paren depth instead, the same technique
    _split_top_level_statements already uses for ';'. Returns None if the
    statement doesn't start with "if (" or the parens never balance."""
    match = re.match(r"^if\s*\(", statement)
    if not match:
        return None
    i = match.end()
    depth = 1
    start = i
    n = len(statement)
    while i < n and depth > 0:
        if statement[i] == "(":
            depth += 1
        elif statement[i] == ")":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    cond = statement[start:i - 1].strip()
    rest = statement[i:].strip()
    return cond, rest


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
        except OpaqueRuntimeCall as e:
            print(f"Info: {statement}; depends on runtime state ({e}()), left unresolved")
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
    except OpaqueRuntimeCall as e:
        print(f"Info: parameter {statement}; depends on runtime state ({e}()), left unresolved")
    except Exception as e:
        print(f"Warning: Failed to evaluate parameter {statement};: {e}")


def _assign_or_warn(var_map, name, expr, display_text):
    """Evaluate expr and store it under name in var_map, or print the
    appropriate Info:/Warning: message and leave it unresolved - shared by
    a plain "name = expr;" statement and whichever branch of an
    "if (cond) name = expr; [else name2 = expr2;]" actually gets taken."""
    try:
        var_map[name] = eval_expr(expr, var_map)
    except OpaqueRuntimeCall as e:
        print(f"Info: {display_text} depends on runtime state ({e}()), left unresolved")
    except Exception as e:
        print(f"Warning: Failed to evaluate {display_text}: {e}")


def _apply_conditional_statement(cond, name, expr, else_name, else_expr, var_map):
    """Handle one "if (cond) name = expr;", optionally paired with
    "else name2 = expr2;". Real C semantics: only the taken branch is ever
    executed, so the untaken one must never be evaluated or warned about -
    a failing expression that's never reached isn't a bug."""
    display = f"if ({cond}) {name} = {expr};"
    if else_name is not None:
        display += f" else {else_name} = {else_expr};"

    try:
        # ast.parse(mode="eval") treats leading whitespace as an
        # (invalid) indent, not just insignificant padding - and "!x" at
        # the very start of a condition becomes " not x" after the
        # substitution below, so this can't skip the strip().
        cond_value = eval_expr(_c_bool_ops_to_python(cond).strip(), var_map)
    except OpaqueRuntimeCall as e:
        print(f"Info: {display} depends on runtime state ({e}()), left unresolved")
        return
    except Exception as e:
        print(f"Warning: Failed to evaluate {display}: {e}")
        return

    if cond_value:
        _assign_or_warn(var_map, name, expr, display)
    elif else_name is not None:
        _assign_or_warn(var_map, else_name, else_expr, display)


def instrument_parameters(instr: ms.McStas_instr):
    """[(name, type, default or None)] for every instrument parameter."""
    params = []
    for param in instr.parameters:
        if isinstance(param, str):
            continue
        default = param.value
        params.append((param.name, param.type or "double", None if default in (None, "") else str(default)))
    return params


def _override_parameter(name, text, param_type, var_map):
    """Use a user-given value (e.g. from the viewer's parameter form) for
    instrument parameter name instead of its default."""
    text = text.strip()
    quoted = _QUOTED_STRING_RE.match(text)
    if param_type == "string" or quoted:
        var_map[name] = quoted.group(1) if quoted else text
        return
    try:
        var_map[name] = eval_expr("".join(text.split()), var_map)
    except Exception as e:
        print(f"Warning: Failed to evaluate parameter value {name}={text}: {e}")


def create_var_map(instr: ms.McStas_instr, param_values=None):
    var_map = {}
    _populate_declare_vars(list(instr.declare_list), var_map)
    _populate_declare_vars(list(instr.user_var_list), var_map)

    param_types = {}
    for param in instr.parameters:
        # Parameters come from mcstasscript's DEFINE INSTRUMENT(...) parser,
        # a different code path from the freeform DECLARE/USERVARS
        # fallback above - see _recover_raw_parameter for why a raw-string
        # entry here is recovered rather than assumed impossible.
        if isinstance(param, str):
            _recover_raw_parameter(param, var_map)
            continue
        var_map[param.name] = _resolved_value(param)
        param_types[param.name] = param.type

    for name, text in (param_values or {}).items():
        if name in param_types and text.strip():
            _override_parameter(name, text, param_types[name], var_map)

    lines = instr.initialize_section.splitlines()

    for line in lines:
        line = line.strip()

        if not line or line.startswith("//"):
            continue

        # C allows several ';'-terminated statements on one line (McStas
        # INITIALIZE sections use this constantly, e.g. "SM=1; SS=-1;
        # SA=1;"), so split on top-level ';' first and process each
        # resulting statement independently, instead of matching (and
        # requiring) exactly one assignment for the whole line.
        statements = _split_top_level_statements(line)
        idx = 0
        n = len(statements)
        while idx < n:
            statement = statements[idx]

            if_split = _split_if_statement(statement)
            if if_split is not None:
                cond, rest = if_split
                match = _RAW_ASSIGNMENT_RE.match(rest)
                if match:
                    name, expr = match.groups()
                    else_name = else_expr = None
                    # An "if (cond) x=a;" and its "else x=b;" become two
                    # separate chunks after the ';'-split above - look ahead
                    # one chunk to re-associate them before evaluating.
                    if idx + 1 < n:
                        else_match = _ELSE_ASSIGNMENT_RE.match(statements[idx + 1])
                        if else_match:
                            else_name, else_expr = else_match.groups()
                            idx += 1
                    _apply_conditional_statement(
                        cond, name, expr.strip(), else_name,
                        else_expr.strip() if else_expr is not None else None,
                        var_map,
                    )
                idx += 1
                continue

            match = _RAW_ASSIGNMENT_RE.match(statement)
            if match:
                name, expr = match.groups()
                _assign_or_warn(var_map, name, expr, f"{statement};")
            idx += 1
    return var_map


def attempt_conversion(comp: mshelp.Component, instr: ms.McStas_instr, var_map: dict):
    # (instr, comp) context so parse_param can resolve COMP_GETPAR(...) -
    # it needs to look up other components by name/position, not just the
    # plain var_map of DECLARE/INITIALIZE values.
    comp_context = (instr, comp)
    # Parameters exist in the following spaces in each component:
    # AT vector, the ROT vector,
    # the component_parameters
    # First loop over comp params:
    for name in comp.parameter_names:
        value = getattr(comp, name)
        if isinstance(value, (str, int, float)):
            val = parse_param(value, var_map, comp_context)
            setattr(comp, name, val)
        elif isinstance(value, (list, tuple, set)):
            converted = []

            for val in value:
                res = parse_param(val, var_map, comp_context)
                converted.append(res)

            setattr(comp, name, type(value)(converted))
    # Then go over AT and ROT vector
    for i, val in enumerate(comp.AT_data):
        value = parse_param(val, var_map, comp_context)
        comp.AT_data[i] = value
    for i, val in enumerate(comp.ROTATED_data):
        value = parse_param(val, var_map, comp_context)
        comp.ROTATED_data[i] = value

    return comp


# =============================================================================
# ======================== COMPONENT GEOMETRY PARAMETERS ======================
# =============================================================================


def _taper_dimension(value, untapered, comp_name, param_name):
    """Normalise one optional Union_box taper parameter (xwidth2/yheight2)
    into an actual dimension, falling back to the untapered one.

    Union_box.comp declares these with a -1 default and treats any negative
    value as "same as xwidth/yheight" (`if (xwidth2 < 0) xwidth2 = xwidth;`),
    while rejecting a value that is <= 0 but not exactly -1. mcstasscript
    reports a setting parameter the instrument never wrote as None rather
    than as the component's own -1 default, so both spellings of "unset"
    reach us here and both have to mean "untapered".

    Anything McStas itself would reject (0, or a negative that isn't the -1
    sentinel) warns and falls back to the untapered dimension, matching how
    the rest of this module prefers a visible warning plus a still-usable
    instrument over aborting the whole run."""
    if value is None:
        return untapered

    try:
        value = float(value)
    except (TypeError, ValueError):
        # parse_param leaves an expression it could not evaluate as the
        # original string, and has already warned about it.
        print(
            f"Warning: Component '{comp_name}': could not resolve {param_name} "
            f"to a number ({value!r}); treating the box as untapered."
        )
        return untapered

    if value == -1:
        return untapered

    if value <= 0:
        print(
            f"Warning: Component '{comp_name}': {param_name}={value} is not a "
            f"usable dimension (McStas requires it to be > 0 or the -1 "
            f"default); treating the box as untapered."
        )
        return untapered

    return value


def box_dimensions(comp):
    """Return (x1, y1, x2, y2): a Union_box's cross-section at its -z face
    and at its +z face.

    McStas's Union_box takes optional xwidth2/yheight2 giving a different
    width and height at the +z face, which turns the box into a rectangular
    frustum whose cross-section varies linearly along local z. Every
    consumer of box dimensions goes through this one helper so the "what
    counts as unset" rule lives in a single place."""
    x1 = float(comp.xwidth)
    y1 = float(comp.yheight)
    x2 = _taper_dimension(getattr(comp, "xwidth2", None), x1, comp.name, "xwidth2")
    y2 = _taper_dimension(getattr(comp, "yheight2", None), y1, comp.name, "yheight2")
    return x1, y1, x2, y2


def box_is_tapered(comp):
    """Does this Union_box actually need frustum handling? Kept next to
    box_dimensions so the exact-equality test is written once: every caller
    has a cheaper and more accurate untapered path worth preserving."""
    x1, y1, x2, y2 = box_dimensions(comp)
    return x1 != x2 or y1 != y2


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

    def default_unresolved_to_zero(comp, values, kind):
        # A component with a genuinely unresolved AT/ROTATED expression
        # doesn't mean the whole instrument is broken - abort just this
        # component's own local transform (defaulting it to zero) rather
        # than the entire preprocessing run, so the rest of the instrument
        # is still visualizable. Mutates values in place (comp.AT_data /
        # comp.ROTATED_data), matching how attempt_conversion already
        # treats these lists as mutable.
        unresolved = [v for v in values if isinstance(v, str)]
        if not unresolved:
            return
        print(
            f"Warning: Component '{comp.name}': could not resolve {kind} "
            f"expression(s) to numeric values: {unresolved}. Check for "
            f"undefined variables or missing constants in "
            f"eval_expr/MCSTAS_CONSTANTS; defaulting to 0 for this "
            f"component's local transform."
        )
        for i, v in enumerate(values):
            if isinstance(v, str):
                values[i] = 0.0

    def compute_matrix(comp, AT_parent, ROT_parent, world):
        default_unresolved_to_zero(comp, comp.ROTATED_data, "ROTATED")
        default_unresolved_to_zero(comp, comp.AT_data, "AT")

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


def _resolve_relative_path(path, base_dir):
    """Strip a raw component-parameter string down to its literal path (the
    .instr text parser leaves surrounding quotes in place, e.g. a filename
    read as '"./mesh.stl"' rather than "./mesh.stl") and, if it's relative,
    resolve it against base_dir rather than leaving it to whatever the
    process's current working directory happens to be."""
    quoted = _QUOTED_STRING_RE.match(path.strip())
    path = quoted.group(1) if quoted else path.strip()
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def resolve_mesh_filenames(union_geometries, input_file):
    """Union_mesh components reference an external mesh file (e.g. an STL)
    by a filename that's typically written relative to the .instr/.py file
    it's declared in - that's how real McStas resolves such paths too. Our
    own pipeline has no equivalent of McStas's search path, so without this
    a relative filename is instead resolved against the process's current
    working directory, which silently depends on where mcstas_to_cad.py was
    invoked from. Rewrite each filename to be resolved against the
    instrument file's own directory instead, up front, so every downstream
    mesh loader (meshing.py, signed_distance_functions.py, bounding_box.py)
    gets a working, unambiguous path regardless of invocation cwd."""
    base_dir = os.path.dirname(os.path.abspath(input_file))
    for comp in union_geometries.values():
        filename = getattr(comp, "filename", None)
        if filename:
            comp.filename = _resolve_relative_path(filename, base_dir)


def preprocess(
    input_file: str, verbose: bool, force_pygen: bool = False, param_values=None
):
    """
    Function to preprocess the input file.

    force_pygen: always translate a .instr input through mcstas-pygen
        instead of mcstasscript's lightweight .instr reader (that reader is
        still used as an automatic fallback on parse failure regardless of
        this flag - see load_McStas_file).
    param_values: {parameter name: value text} used instead of the
        instrument's defaults.

    Returns:
        McStas_instr containing the processed instrument
        dict: {component_name_lower: 4x4 world matrix}
        list: Each union geometry in the instrument.
    """
    instr = load_McStas_file(input_file, force_pygen=force_pygen, verbose=verbose)
    var_map = create_var_map(instr, param_values)
    for comp in instr.component_list:
        comp = attempt_conversion(comp, instr, var_map)

    world_matrices = compute_world_matrices(instr, verbose)
    union_geometries = get_union_geometries(instr)
    resolve_mesh_filenames(union_geometries, input_file)
    return instr, world_matrices, union_geometries
