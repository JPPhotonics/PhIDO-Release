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
    

# ---------------------------------------------------------------------------
# 4. Run the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()