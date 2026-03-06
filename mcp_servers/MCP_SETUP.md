# MCP Servers — Setup & Usage

PhIDO's agentic pipeline is built on a set of MCP (Model Context Protocol) servers and supporting modules that expose photonic design capabilities as callable tools for LLM agents.

---

## Overview

| File | Type | Description |
|------|------|-------------|
| `kg_server.py` | MCP Server | Photonics knowledge graph — ontology schema + Neo4j GraphRAG retrieval |
| `pdk_catalog_server.py` | MCP Server | PDK component catalog — search, inspect, validate DesignLibrary components via GDSFactory |
| `schematic_builder_server.py` | MCP Server | Schematic builder — DOT graph generation, layout computation, circuit DSL assembly |
| `layout_sim_server.py` | Module | GDS layout rendering and SAX circuit simulation (called directly, not via MCP) |
| `interpreter_agent.py` | Agent | Multi-phase interpreter — translates natural language into structured design intent |
| `pipeline_orchestrator.py` | Orchestrator | Deterministic controller chaining interpretation → selection → schematic → layout → simulation |
| `kg_agent.py` | Agent | Standalone conversational Q&A agent over the knowledge graph |
| `models.py` | Models | Pydantic data contracts shared across all agents and tools |
| `streamlit_app.py` | UI | Main PhIDO pipeline Streamlit interface |
| `kg_chat.py` | UI | Standalone KG agent Streamlit chat interface |

## Prerequisites

### Python Dependencies

All servers run in the same virtual environment. Core dependencies:

```bash
pip install mcp rdflib neo4j openai gdsfactory sax pygraphviz pydantic streamlit
```

Ensure you are using the project venv (e.g., `/home/tofu8/venv/bin/python`).

### Neo4j (for KG Server)

The knowledge graph tools in `kg_server.py` require a running Neo4j instance. See [`NEO4J_README.md`](../NEO4J_README.md) for Docker setup. Quick start:

```bash
./start_neo4j.sh
```

If Neo4j is not running, the KG server operates in degraded mode — ontology tools still work, but graph retrieval tools return an error message.

### OpenAI API Key (for Agents)

The interpreter agent, pipeline orchestrator, and KG agent all call the OpenAI API:

```bash
export OPENAI_API_KEY="sk-..."
```

## Running the MCP Servers

### Standalone (for testing)

Each MCP server can be run directly:

```bash
python mcp_servers/kg_server.py
python mcp_servers/pdk_catalog_server.py
python mcp_servers/schematic_builder_server.py
```

To test with the MCP dev inspector:

```bash
mcp dev mcp_servers/kg_server.py
mcp dev mcp_servers/pdk_catalog_server.py
mcp dev mcp_servers/schematic_builder_server.py
```

### As Cursor MCP Servers

Add entries to `.cursor/mcp.json` to make tools available to Cursor's agent:

```json
{
  "mcpServers": {
    "kg-server": {
      "command": "/home/tofu8/venv/bin/python",
      "args": ["mcp_servers/kg_server.py"],
      "cwd": "/home/tofu8/PhIDO-Release"
    },
    "pdk-catalog": {
      "command": "/home/tofu8/venv/bin/python",
      "args": ["mcp_servers/pdk_catalog_server.py"],
      "cwd": "/home/tofu8/PhIDO-Release"
    },
    "schematic-builder": {
      "command": "/home/tofu8/venv/bin/python",
      "args": ["mcp_servers/schematic_builder_server.py"],
      "cwd": "/home/tofu8/PhIDO-Release"
    }
  }
}
```

Adjust the `command` path to match your Python environment.

### Via the PhIDO Pipeline (normal usage)

In typical usage, you don't run MCP servers individually. The interpreter agent and pipeline orchestrator import tools directly as Python functions:

```bash
# Main pipeline UI
streamlit run mcp_servers/streamlit_app.py

# Standalone KG chat UI
streamlit run mcp_servers/kg_chat.py
```

## Server Details

### KG Server (`kg_server.py`)

Exposes 7 tools for querying the photonics knowledge graph. See [`KG_SERVER_README.md`](KG_SERVER_README.md) for full documentation.

**Ontology tools** (always available):
- `get_ontology_schema()` — full OWL ontology: classes, relationships, data properties
- `list_entity_types()` — static + dynamically discovered relationship types

**Graph retrieval tools** (require Neo4j):
- `search_concepts(query, entity_type?)` — semantic vector search
- `get_concept_neighborhood(concept_name, max_hops?)` — graph traversal
- `resolve_function(function_description)` — reverse lookup: function → components
- `get_component_properties(component_name)` — all known properties for a component
- `kg_stats()` — node/edge counts by type

### PDK Catalog Server (`pdk_catalog_server.py`)

Exposes the DesignLibrary as searchable tools backed by GDSFactory.

- `list_components()` — summary of all PDK components
- `search_components(query)` — keyword search across names, descriptions, labels
- `get_component_info(module_name)` — full metadata for a specific component
- `validate_port_config(component_query, port_config)` — check port compatibility
- `get_module_params(module_name)` — ground-truth PCell parameters from GDSFactory
- `get_component_footprint(module_name)` — physical dimensions (dx, dy in um)
- `get_port_names(module_name)` — deterministic port name list
- `validate_selection(mappings_json)` — post-selection validation of component mappings

### Schematic Builder Server (`schematic_builder_server.py`)

Generates and validates photonic circuit schematics.

- `circuit_dsl_to_dot(circuit_dsl_json)` — convert circuit DSL to DOT graph
- `check_planarity(dot_string)` — detect crossing edges
- `compute_layout(dot_string, footprints_json)` — Graphviz layout with node positions
- `find_open_ports(dot_string)` — identify unconnected ports
- `export_gf_netlist(circuit_dsl_json)` — generate GDSFactory-compatible YAML netlist

### Layout & Simulation Module (`layout_sim_server.py`)

Called directly by the pipeline orchestrator (not an MCP server).

- `render_gds_layout(gf_netlist_yaml)` — instantiate GDSFactory component, render layout image
- `run_sax_simulation(gf_netlist_yaml, wl_start, wl_stop, wl_points)` — wavelength sweep via SAX
- `write_gds_file(gf_netlist_yaml, filename)` — export to `.gds` file

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ImportError: No module named 'mcp'` | MCP SDK not installed | `pip install mcp` |
| KG tools return "Neo4j is not available" | Neo4j not running | `./start_neo4j.sh` or `sudo docker compose up -d` |
| PDK catalog loads 0 components | Wrong working directory | Run from repo root or set `cwd` in mcp.json |
| `pygraphviz` import error | System graphviz not installed | `sudo apt install graphviz libgraphviz-dev` then `pip install pygraphviz` |
| OpenAI API errors from agents | Missing or invalid API key | `export OPENAI_API_KEY="sk-..."` |
| Streamlit won't start | Port already in use | `streamlit run ... --server.port 8502` |

## File Reference

```
mcp_servers/
├── MCP_SETUP.md                  # This file
├── KG_SERVER_README.md           # Detailed KG server documentation
├── kg_server.py                  # MCP: Knowledge graph tools
├── pdk_catalog_server.py         # MCP: PDK catalog tools
├── schematic_builder_server.py   # MCP: Schematic builder tools
├── layout_sim_server.py          # Module: GDS layout + SAX simulation
├── interpreter_agent.py          # Agent: NL → DesignIntent
├── pipeline_orchestrator.py      # Orchestrator: full pipeline controller
├── kg_agent.py                   # Agent: standalone KG Q&A
├── models.py                     # Pydantic data contracts
├── streamlit_app.py              # UI: main pipeline
└── kg_chat.py                    # UI: KG agent chat
```
