"""Parse DesignLibrary component docstrings and introspect GDSFactory ports.

Adapted from mcp_servers/pdk_catalog_server.py::_parse_component_docstring
with extensions for numeric spec extraction, primitive classification, and
flatten detection.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

import yaml

from .models import NumericSpec, PDKCellNode, PortInfo

DESIGN_LIBRARY_IMPORT_PREFIX = "PhotonicsAI.KnowledgeBase.DesignLibrary"

# Regex for numeric values with units, e.g. "2.5 dB", "200 KHz", "1450-1650 nm"
_NUMERIC_RE = re.compile(r"([\d.]+(?:\s*-\s*[\d.]+)?)\s*([a-zA-Zμµ/%]+)")

# Spec fields we look for in the docstring YAML front-matter
_SPEC_FIELDS = {
    "Insertion loss",
    "Extinction ratio",
    "Optical Bandwidth",
    "Bandwidth",
    "Modulation bandwidth/Switching speed",
    "Drive voltage/power",
    "channel spacing",
    "N of channels",
    "Footprint Estimate",
}


def parse_component_file(
    file_path: Path,
    pdk_name: str,
    pdk_version: str = "1.0.0",
) -> PDKCellNode:
    """Parse a single DesignLibrary .py file into a PDKCellNode.

    This reads the module-level docstring, extracts YAML front-matter metadata,
    determines whether the cell function calls `.flatten()`, and checks if the
    module imports from DesignLibrary (primitive detection).
    """
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    raw_docstring = ast.get_docstring(tree) or ""
    module_name = file_path.stem

    # Split on YAML front-matter separator "---"
    parts = raw_docstring.split("---", 1)
    summary = parts[0].strip()
    metadata: dict = {}
    if len(parts) > 1:
        try:
            metadata = yaml.safe_load(parts[1].strip()) or {}
        except yaml.YAMLError:
            metadata = {}

    # Detect DesignLibrary imports
    dl_imports = _extract_design_library_imports(tree)
    is_primitive = len(dl_imports) == 0

    # Detect .flatten() call and get_model function
    is_flattened = ".flatten()" in source
    has_simulation_model = any(
        isinstance(node, ast.FunctionDef) and node.name == "get_model"
        for node in ast.walk(tree)
    )

    # Parse numeric specs from YAML metadata
    numeric_specs = _extract_numeric_specs(metadata)

    # Args can be a dict, a string, or a list depending on docstring formatting
    raw_args = metadata.get("Args", {})
    if not isinstance(raw_args, dict):
        raw_args = {"raw": raw_args} if raw_args else {}

    return PDKCellNode(
        module_name=module_name,
        display_name=metadata.get("Name", module_name),
        description=metadata.get("Description", summary),
        pdk_name=pdk_name,
        pdk_version=pdk_version,
        ports=str(metadata.get("ports", "unknown")),
        labels=metadata.get("NodeLabels", []),
        aka=metadata.get("aka", "") or "",
        technology=metadata.get("Technology", "") or "",
        parameters=raw_args,
        numeric_specs=numeric_specs,
        is_flattened=is_flattened,
        is_primitive=is_primitive,
        has_simulation_model=has_simulation_model,
        source_file=str(file_path),
    )


def introspect_ports(module_name: str) -> tuple[list[PortInfo], Optional[float], Optional[float]]:
    """Instantiate a component via GDSFactory and record port positions + footprint.

    Returns (port_infos, dx_um, dy_um).
    """
    import importlib

    import gdsfactory as gf
    from gdsfactory.generic_tech import get_generic_pdk

    get_generic_pdk().activate()

    full_module = f"{DESIGN_LIBRARY_IMPORT_PREFIX}.{module_name}"
    mod = importlib.import_module(full_module)
    cell_func = getattr(mod, module_name)
    comp: gf.Component = cell_func()

    port_infos: list[PortInfo] = []
    for port in comp.ports:
        port_infos.append(
            PortInfo(
                name=port.name,
                x=float(port.dcenter[0]),
                y=float(port.dcenter[1]),
                orientation=float(port.orientation) if port.orientation is not None else 0.0,
                width=float(port.dwidth) if hasattr(port, "dwidth") else None,
            )
        )

    return port_infos, float(comp.dxsize), float(comp.dysize)


def _extract_design_library_imports(tree: ast.Module) -> list[str]:
    """Return list of DesignLibrary module names imported by this file."""
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith(DESIGN_LIBRARY_IMPORT_PREFIX) or node.module == DESIGN_LIBRARY_IMPORT_PREFIX:
                for alias in node.names:
                    modules.append(alias.name)
    return modules


def _extract_numeric_specs(metadata: dict) -> list[NumericSpec]:
    """Parse numeric specification fields from docstring YAML metadata."""
    specs: list[NumericSpec] = []
    for field_name in _SPEC_FIELDS:
        raw = metadata.get(field_name)
        if raw is None:
            continue
        raw_str = str(raw).strip()
        match = _NUMERIC_RE.search(raw_str)
        if match:
            value_str = match.group(1)
            # Handle range values like "1450-1650" by taking the first number
            if "-" in value_str:
                value_str = value_str.split("-")[0].strip()
            try:
                value = float(value_str)
            except ValueError:
                continue
            specs.append(
                NumericSpec(
                    field_name=field_name,
                    value=value,
                    unit=match.group(2),
                    raw_text=raw_str,
                )
            )
    return specs
