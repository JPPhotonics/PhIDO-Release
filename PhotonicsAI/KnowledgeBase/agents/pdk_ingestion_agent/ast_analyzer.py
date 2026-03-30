"""AST-based layout composition analyzer for GDSFactory cell functions.

Statically parses ``@gf.cell``-decorated functions to extract:
- Sub-component instantiations  (``c << module.func(...)``)
- Explicit connections           (``route_single``, ``.connect``)
- GDSFactory compound role assignments (``mzi2x2_2x2(..., splitter=X)``)
- External port mappings         (``c.add_port("oN", port=X.ports["oM"])``)

The output is a ``LayoutComposition`` that downstream phases can feed into
the LLM topology completion step (Phase 2b) and ``COMPOSED_OF`` edge creation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

from .docstring_parser import DESIGN_LIBRARY_IMPORT_PREFIX
from .models import (
    ExplicitConnection,
    ExternalPort,
    LayoutComposition,
    LayoutInstantiation,
    RoleAssignment,
)


def analyze_cell_file(file_path: Path) -> LayoutComposition:
    """Perform full AST analysis on a DesignLibrary component file."""
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))
    module_name = file_path.stem

    # 1. Collect DesignLibrary imports
    dl_imports = _collect_dl_imports(tree)
    if not dl_imports:
        return LayoutComposition(
            module_name=module_name,
            imported_modules=[],
            composition_pattern="primitive",
        )

    # 2. Find the @gf.cell-decorated function
    cell_func = _find_cell_function(tree)
    if cell_func is None:
        return LayoutComposition(
            module_name=module_name,
            imported_modules=list(dl_imports.keys()),
            composition_pattern="unknown",
        )

    # 3. Walk the cell function body
    instantiations: list[LayoutInstantiation] = []
    connections: list[ExplicitConnection] = []
    role_assignments: list[RoleAssignment] = []
    external_ports: list[ExternalPort] = []

    # Map local variable names to their DesignLibrary module names
    var_to_module: dict[str, str] = {}

    for stmt in ast.walk(cell_func):
        # --- Instantiation: var = c << module.func(...) ---
        inst = _try_parse_instantiation(stmt, dl_imports)
        if inst:
            instantiations.append(inst)
            var_to_module[inst.var_name] = inst.module_name
            continue

        # --- Explicit connection: route_single / route_bundle / .connect ---
        conn = _try_parse_connection(stmt)
        if conn:
            connections.append(conn)
            continue

        # --- GDSFactory compound with role kwargs ---
        roles = _try_parse_gf_compound_roles(stmt, dl_imports)
        if roles:
            role_assignments.append(roles)
            continue

        # --- External port: c.add_port("oN", port=X.ports["oM"]) ---
        ext = _try_parse_external_port(stmt)
        if ext:
            external_ports.append(ext)
            continue

    # Classify composition pattern
    pattern = _classify_pattern(instantiations, connections, role_assignments, dl_imports)

    return LayoutComposition(
        module_name=module_name,
        instantiations=instantiations,
        explicit_connections=connections,
        role_assignments=role_assignments,
        external_ports=external_ports,
        imported_modules=list(dl_imports.keys()),
        composition_pattern=pattern,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_dl_imports(tree: ast.Module) -> dict[str, str]:
    """Return {alias_name: module_name} for DesignLibrary imports."""
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(DESIGN_LIBRARY_IMPORT_PREFIX) or node.module == DESIGN_LIBRARY_IMPORT_PREFIX:
                for alias in node.names:
                    imports[alias.asname or alias.name] = alias.name
    return imports


def _find_cell_function(tree: ast.Module) -> Optional[ast.FunctionDef]:
    """Find the first function decorated with @gf.cell."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if _is_gf_cell_decorator(dec):
                    return node
    return None


def _is_gf_cell_decorator(dec: ast.expr) -> bool:
    """Check if a decorator is ``gf.cell`` (Attribute) or ``cell`` (Name)."""
    if isinstance(dec, ast.Attribute):
        return (
            isinstance(dec.value, ast.Name) and dec.value.id == "gf" and dec.attr == "cell"
        )
    if isinstance(dec, ast.Name):
        return dec.id == "cell"
    if isinstance(dec, ast.Call):
        return _is_gf_cell_decorator(dec.func)
    return False


def _try_parse_instantiation(
    node: ast.AST, dl_imports: dict[str, str]
) -> Optional[LayoutInstantiation]:
    """Try to match: var = c << module.func(...) or c << module.func(...)"""
    target_name: Optional[str] = None
    rhs: Optional[ast.expr] = None

    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            target_name = target.id
        rhs = node.value
    elif isinstance(node, ast.Expr):
        rhs = node.value

    if rhs is None:
        return None

    # Match c << expr  (BinOp with LShift)
    if not (isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.LShift)):
        return None

    call_node = rhs.right
    if not isinstance(call_node, ast.Call):
        return None

    # Resolve the function being called to a DesignLibrary module
    mod_name = _resolve_dl_call(call_node.func, dl_imports)
    if mod_name is None:
        return None

    # Extract keyword arguments
    call_args: dict = {}
    for kw in call_node.keywords:
        if kw.arg is not None:
            call_args[kw.arg] = _safe_literal(kw.value)

    return LayoutInstantiation(
        var_name=target_name or f"_anon_{id(node)}",
        module_name=mod_name,
        call_args=call_args,
    )


def _try_parse_connection(node: ast.AST) -> Optional[ExplicitConnection]:
    """Match route_single/route_bundle or .connect calls."""
    if not isinstance(node, (ast.Expr, ast.Assign)):
        return None

    call: Optional[ast.Call] = None
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        call = node.value
    elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        call = node.value

    if call is None:
        return None

    # --- gf.routing.route_single(c, port1=X.ports["oN"], port2=Y.ports["oM"]) ---
    func_name = _get_func_name(call.func)
    if func_name in ("route_single", "route_bundle"):
        port1 = port2 = None
        for kw in call.keywords:
            if kw.arg == "port1":
                port1 = _parse_port_ref(kw.value)
            elif kw.arg == "port2":
                port2 = _parse_port_ref(kw.value)
        # Also check positional args (port1, port2 can be args[1], args[2] after component)
        if port1 is None and len(call.args) >= 2:
            port1 = _parse_port_ref(call.args[1])
        if port2 is None and len(call.args) >= 3:
            port2 = _parse_port_ref(call.args[2])
        if port1 and port2:
            return ExplicitConnection(
                from_var=port1[0], from_port=port1[1],
                to_var=port2[0], to_port=port2[1],
            )

    # --- X.connect("oN", Y.ports["oM"]) ---
    if func_name == "connect" and isinstance(call.func, ast.Attribute):
        caller_var = _get_var_name(call.func.value)
        if caller_var and len(call.args) >= 2:
            port_name = _safe_literal(call.args[0])
            other_ref = _parse_port_ref(call.args[1])
            if isinstance(port_name, str) and other_ref:
                return ExplicitConnection(
                    from_var=caller_var, from_port=port_name,
                    to_var=other_ref[0], to_port=other_ref[1],
                )

    return None


def _try_parse_gf_compound_roles(
    node: ast.AST, dl_imports: dict[str, str]
) -> Optional[RoleAssignment]:
    """Match c << gf.components.xxx(..., splitter=A, combiner=B, ...)."""
    rhs: Optional[ast.expr] = None
    if isinstance(node, ast.Assign):
        rhs = node.value
    elif isinstance(node, ast.Expr):
        rhs = node.value

    if rhs is None:
        return None

    if isinstance(rhs, ast.BinOp) and isinstance(rhs.op, ast.LShift):
        rhs = rhs.right

    if not isinstance(rhs, ast.Call):
        return None

    # Check if this is a gf.components.xxx call
    gf_compound = _get_gf_components_name(rhs.func)
    if gf_compound is None:
        return None

    keyword_roles: dict[str, str] = {}
    for kw in rhs.keywords:
        if kw.arg is None:
            continue
        # Value could be: module.func (Attribute), module (Name), or a call
        resolved = _resolve_kwarg_to_dl_module(kw.value, dl_imports)
        if resolved:
            keyword_roles[kw.arg] = resolved

    if not keyword_roles:
        return None

    return RoleAssignment(gf_compound=gf_compound, keyword_roles=keyword_roles)


def _try_parse_external_port(node: ast.AST) -> Optional[ExternalPort]:
    """Match c.add_port("oN", port=X.ports["oM"])."""
    if not isinstance(node, ast.Expr):
        return None
    if not isinstance(node.value, ast.Call):
        return None

    call = node.value
    func_name = _get_func_name(call.func)
    if func_name != "add_port":
        return None

    # First arg is the external port name
    if not call.args:
        return None
    ext_name = _safe_literal(call.args[0])
    if not isinstance(ext_name, str):
        return None

    # port= keyword
    for kw in call.keywords:
        if kw.arg == "port":
            ref = _parse_port_ref(kw.value)
            if ref:
                return ExternalPort(
                    external_port=ext_name,
                    instance_var=ref[0],
                    instance_port=ref[1],
                )

    return None


# ---------------------------------------------------------------------------
# Low-level AST utilities
# ---------------------------------------------------------------------------

def _resolve_dl_call(func: ast.expr, dl_imports: dict[str, str]) -> Optional[str]:
    """Resolve a call target to a DesignLibrary module name, if applicable.

    Matches patterns like: module.func(...)  where module is in dl_imports.
    """
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        alias = func.value.id
        if alias in dl_imports:
            return dl_imports[alias]
    return None


def _get_gf_components_name(func: ast.expr) -> Optional[str]:
    """If func is gf.components.xxx, return 'xxx'."""
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and isinstance(func.value.value, ast.Name)
        and func.value.value.id == "gf"
        and func.value.attr == "components"
    ):
        return func.attr
    return None


def _resolve_kwarg_to_dl_module(
    value: ast.expr, dl_imports: dict[str, str]
) -> Optional[str]:
    """Resolve a keyword argument value to a DesignLibrary module name.

    Handles: module.func, module.func(), module (Name).
    """
    # module.func  (Attribute on a DL import)
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
        if value.value.id in dl_imports:
            return dl_imports[value.value.id]
    # module.func()  (Call wrapping above)
    if isinstance(value, ast.Call):
        return _resolve_kwarg_to_dl_module(value.func, dl_imports)
    # Direct Name reference
    if isinstance(value, ast.Name) and value.id in dl_imports:
        return dl_imports[value.id]
    return None


def _parse_port_ref(node: ast.expr) -> Optional[tuple[str, str]]:
    """Parse X.ports["oN"] -> ("X", "oN")."""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Attribute) and node.value.attr == "ports":
            var = _get_var_name(node.value.value)
            port_key = _safe_literal(node.slice)
            if var and isinstance(port_key, str):
                return (var, port_key)
    return None


def _get_func_name(node: ast.expr) -> Optional[str]:
    """Get the terminal function/method name from a call."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _get_var_name(node: ast.expr) -> Optional[str]:
    """Get a simple variable name from an expression."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _safe_literal(node: ast.expr) -> object:
    """Try to evaluate a constant/literal AST node."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.dump(node)


def _classify_pattern(
    instantiations: list[LayoutInstantiation],
    connections: list[ExplicitConnection],
    role_assignments: list[RoleAssignment],
    dl_imports: dict[str, str],
) -> str:
    """Classify the composition pattern of the cell function."""
    if not dl_imports:
        return "primitive"
    if role_assignments:
        return "delegated"
    if instantiations and connections:
        return "explicit"
    if instantiations and not connections:
        return "wrapper"
    return "unknown"
