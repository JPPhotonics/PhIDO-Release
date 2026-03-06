"""
MCP Server: PDK Catalog
Exposes the PhIDO DesignLibrary as searchable tools for LLM agents.

Run with: python mcp_servers/pdk_catalog_server.py
Test with: mcp dev mcp_servers/pdk_catalog_server.py
"""

import ast
import glob
import os
import json
import yaml
from pathlib import Path
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# 1. Initialize the MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "PhIDO PDK Catalog",
    instructions="Search and inspect photonic components in the PhIDO DesignLibrary.",
)

# ---------------------------------------------------------------------------
# 2. Load the PDK catalog at startup (once)
# ---------------------------------------------------------------------------
# Resolve path relative to this file → repo root → DesignLibrary
REPO_ROOT = Path(__file__).parent.parent
PDK_DIR = REPO_ROOT / "PhotonicsAI" / "KnowledgeBase" / "DesignLibrary"

def _parse_component_docstring(file_path: str) -> dict:
    """Parse a component docstring and return a dictionary of its metadata."""
    with open(file_path, encoding="utf-8") as f:
        source = f.read()
        
    module = ast.parse(source)
    raw_docstring = ast.get_docstring(module) or ""
    module_name = os.path.basename(file_path).replace(".py", "")
    
    # Split on the YAML front-matter separator "---"
    parts = raw_docstring.split("---", 1)
    summary = parts[0].strip()
    metadata = {}
    
    if len(parts) > 1:
        try:
            metadata = yaml.safe_load(parts[1].strip())
        except yaml.YAMLError:
            metadata = {}
    
    return {
        "module_name": module_name,
        "summary": summary,
        "name": metadata.get("Name", module_name),
        "description": metadata.get("Description", summary),
        "ports": metadata.get("ports", "unknown"),
        "labels": metadata.get("NodeLabels", []),
        "bandwidth": metadata.get("Bandwidth", None),
        "technology": metadata.get("Technology", None),
        "aka": metadata.get("aka", None),
        "args": metadata.get("Args", {}),
        "full_docstring": raw_docstring,
    }
    
def _load_catalog() -> list[dict]:
    """Load all components from the DesignLibrary."""
    catalog = []
    for py_file in sorted(glob.glob(str(PDK_DIR / "*.py"))):
        if os.path.basename(py_file) == "__init__.py":
            continue
        try:
            entry = _parse_component_docstring(py_file)
            catalog.append(entry)
        except Exception as e:
            print(f"Warning: Could not parse {py_file}: {e}")
    return catalog

# Load the catalog at startup
CATALOG = _load_catalog()
print(f"Loaded {len(CATALOG)} components from {PDK_DIR}.")

# ---------------------------------------------------------------------------
# 2b. GDSFactory layer — lazy-loaded on first use
# ---------------------------------------------------------------------------

_gf = None
_demo_pdk = None

def _ensure_pdk():
    """Activate GDSFactory + DemoPDK on first call. Subsequent calls are no-ops."""
    global _gf, _demo_pdk
    if _gf is not None:
        return
    
    import gdsfactory as gf_lib
    from gdsfactory.generic_tech import get_generic_pdk
    
    generic = get_generic_pdk()
    
    # Reuse the same import logic as DemoPDK.py
    import importlib
    cells = {}
    for mod_name in [c["module_name"] for c in CATALOG]:
        full = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{mod_name}"
        module = importlib.import_module(full)
        func = getattr(module, mod_name)
        cells[mod_name] = func
        
    pdk = gf_lib.Pdk(
        name="DemoPDK",
        layers=generic.layers,
        cross_sections=generic.cross_sections,
        cells=cells,
        layer_views=generic.layer_views,
    )
    pdk.activate()
    
    _gf = gf_lib
    _demo_pdk = pdk
    
# ---------------------------------------------------------------------------
# 3. Define MCP Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_all_components() -> str:
    """List all available photonic components in the PDK.
    
    Returns a JSON array of components with their name, module_name, ports configuration, and labels.
    """
    
    summary = [
        {
            "module_name": c["module_name"],
            "name": c["name"],
            "ports": c["ports"],
            "labels": c["labels"],
            "summary": c["summary"],
        }
        for c in CATALOG
    ]
    return json.dumps(summary, indent=2)

@mcp.tool()
def search_components(query: str) -> str:
    """Search the photonic components matching a natural-language description.
    
    Performs keyword matching against component names, descriptions, labels, 
    and aliases (aka). returns the top matches with full metadata.
    
    Args:
        query: Natural language description of the desired component,
        e.g. "1x2 MZI with 10 nm bandwidth" or "grating coupler" or "photodetector"
    """
    query_lower = query.lower()
    query_terms = query_lower.split()
    
    scored = []
    for comp in CATALOG:
        # Build a searchable text blob for this component
        searchable = " ".join([
            comp["module_name"],
            comp["name"],
            comp["description"],
            comp.get("summary", ""),
            comp.get("aka", "") or "",
            " ".join(comp.get("labels", [])),
            comp.get("ports", ""),
        ]).lower()
        
        # Simple term-matching scoring
        score = sum(1 for term in query_terms if term in searchable)
        
        # Boost exact module_name match
        if query_lower in comp["module_name"].lower():
            score += 5
        # Boost ports match (e.g. "1x2" -> "1x2 MZI")
        for term in query_terms:
            if "x" in term and term == comp.get("ports", ""):
                score += 3
                
        if score > 0:
            scored.append((score, comp))
            
    # Sort by score descending and return top matches
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [
        {
            "module_name": c["module_name"],
            "name": c["name"],
            "ports": c["ports"],
            "labels": c["labels"],
            "description": c["description"],
            "technology": c["technology"],
            "bandwidth": c["bandwidth"],
            "args": c["args"],
            "match_score": s,
        }
        for s, c in scored[:5]
    ]
    
    if not results:
        return json.dumps({"message": "No matching components found.", "query": query})
    
    return json.dumps(results, indent=2)

@mcp.tool()
def get_component_details(module_name: str) -> str:
    """Get full details for a specific PDK component by its module name.
    
    Args:
        module_name: The exact module name (filename without .py),
        e.g. "mzi_2x2_heater_tin_cband" or "_mmi1x2"
    """
    for comp in CATALOG:
        if comp["module_name"] == module_name:
            return json.dumps(comp, indent=2)
        
    # Fuzzy fallback: check if query is a substring
    partial = [c for c in CATALOG if module_name.lower() in c["module_name"].lower()]
    if partial:
        return json.dumps(
            {"message": f"Exact match not found. Did you mean one of these?",
             "suggestions": [c["module_name"] for c in partial]},
             indent=2   
        )
    return json.dumps({"error": f"Component '{module_name}' not found in PDK."})

@mcp.tool()
def validate_port_config(component_query: str, port_config: str) -> str:
    """Check whether a port configuration is valid for a given component.
    
    Searches the PDK for components matching the query and reports whether
    the given port config (e.g. "2x2", "1x4") matches any of them.
    
    Args:
        component_query: Component name or description, e.g. "MZI" or "mmi"
        port_config: Desired port configuration, e.g. "2x2", "1x2"
    """
    query_lower = component_query.lower()
    matches = [
        c for c in CATALOG
        if query_lower in c["module_name"].lower()
        or query_lower in c["name"].lower()
        or query_lower in c["description"].lower()
        or query_lower in (c.get("aka", "") or "").lower()
    ]
    
    if not matches:
        return json.dumps({"valid": False, "reason": f"No components found matching '{component_query}'."})
    
    exact = [c for c in matches if c.get("ports", "") == port_config]
    available = sorted(set(c.get("ports", "unknown") for c in matches))

    if exact:
        return json.dumps({
            "valid": True,
            "matching_components": [
                {"module_name": c["module_name"], "name": c["name"], "ports": c["ports"]}
                for c in exact
            ],
        })

    return json.dumps({
        "valid": False,
        "reason": f"'{port_config}' not available for '{component_query}'.",
        "available_port_configs": available,
        "components_found": [
            {"module_name": c["module_name"], "ports": c.get("ports", "unknown")}
            for c in matches
        ],
    }) 
    
@mcp.tool()
def get_module_params(module_name: str) -> str:
    """Get the default GDSFactory settings/parameters for a PDK component.

    Instantiates the component in GDSFactory and returns its resolved
    default settings. Use this to discover what parameters a component
    accepts and their default values.

    Args:
        module_name: Exact PDK module name, e.g. "mzi_2x2_heater_tin_cband"
    """
    _ensure_pdk()

    # Verify the module exists in our catalog first (fast check before heavy GDS work)
    if not any(c["module_name"] == module_name for c in CATALOG):
        return json.dumps({"error": f"Module '{module_name}' not in PDK catalog."})

    try:
        netlist_yaml = yaml.dump({
            "instances": {"tmp": {"component": module_name}}
        })
        component = _gf.read.from_yaml(netlist_yaml)
        resolved = component.get_netlist(recursive=False)
        settings = resolved["instances"]["tmp"]["settings"]

        return json.dumps({
            "module_name": module_name,
            "settings": settings,
        }, indent=2, default=str)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to resolve params for '{module_name}': {e}"
        })
        
@mcp.tool()
def get_component_footprint(module_name: str) -> str:
    """Get the physical bounding-box dimensions of a PDK component in microns.

    Returns dx (width) and dy (height) of the component's layout footprint.
    These values are needed for schematic layout sizing.

    Args:
        module_name: Exact PDK module name, e.g. "mzi_2x2_heater_tin_cband"
    """
    _ensure_pdk()

    if not any(c["module_name"] == module_name for c in CATALOG):
        return json.dumps({"error": f"Module '{module_name}' not in PDK catalog."})

    try:
        comp = _demo_pdk.get_component(module_name)
        return json.dumps({
            "module_name": module_name,
            "dx_um": float(comp.dxsize),
            "dy_um": float(comp.dysize),
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to get footprint for '{module_name}': {e}"
        })        

def get_port_names(module_name: str) -> list[str]:
    """Return the ordered list of port names for a PDK module.

    Port naming follows the convention used in circuit_dsl_to_dot:
    for an NxM port config, inputs are o1..oN (counter-clockwise) and
    outputs are o(N+1)..o(N+M). Falls back to the catalog ``ports``
    field; returns an empty list if the module is not found or has
    no parseable port config.
    """
    entry = next((c for c in CATALOG if c["module_name"] == module_name), None)
    if entry is None:
        return []
    ports_str = entry.get("ports", "")
    if not isinstance(ports_str, str) or "x" not in ports_str:
        return []
    try:
        inp, out = map(int, ports_str.split("x"))
    except (ValueError, TypeError):
        return []
    return [f"o{i}" for i in range(1, inp + out + 1)]


@mcp.tool()
def validate_selection(mappings_json: str) -> str:
    """Validate a batch of component-to-PDK-module mappings.

    For each mapping, checks: (1) module exists in PDK, (2) module can be
    instantiated in GDSFactory, (3) port config matches. Returns a list
    of issues (empty list means all valid).

    Args:
        mappings_json: JSON array of objects, each with "component_id" and
                       "pdk_module" keys. Optionally "expected_ports".
                       Example: [{"component_id": "C1", "pdk_module": "mzi_2x2_heater_tin_cband", "expected_ports": "2x2"}]
    """
    _ensure_pdk()

    try:
        mappings = json.loads(mappings_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    issues = []

    for m in mappings:
        cid = m.get("component_id", "?")
        mod = m.get("pdk_module", "")

        # Check 1: exists in catalog?
        catalog_entry = next((c for c in CATALOG if c["module_name"] == mod), None)
        if catalog_entry is None:
            issues.append({
                "component_id": cid,
                "issue": f"Module '{mod}' not found in PDK catalog",
                "severity": "fundamental",
            })
            continue

        # Check 2: can GDSFactory instantiate it?
        try:
            _demo_pdk.get_component(mod)
        except Exception as e:
            issues.append({
                "component_id": cid,
                "issue": f"GDSFactory cannot instantiate '{mod}': {e}",
                "severity": "fundamental",
            })
            continue

        # Check 3: port config match?
        expected = m.get("expected_ports")
        if expected and catalog_entry.get("ports") != expected:
            issues.append({
                "component_id": cid,
                "issue": f"Expected ports '{expected}' but '{mod}' has '{catalog_entry.get('ports', 'unknown')}'",
                "severity": "major",
            })

    return json.dumps({
        "valid": len(issues) == 0,
        "checked": len(mappings),
        "issues": issues,
    }, indent=2)

# ---------------------------------------------------------------------------
# 4. Run the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()