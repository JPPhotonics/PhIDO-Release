# KG Server — Photonics Knowledge Graph MCP Server

The KG server (`kg_server.py`) exposes PhIDO's photonics knowledge graph as a set of MCP tools that LLM agents can call. It combines two knowledge layers:

- **Ontology layer** — class definitions, relationships, and data properties parsed from the OWL ontology (`pic_ontology.ttl`). Always available, no external dependencies.
- **Graph retrieval layer** — semantic search and neighborhood traversal over a Neo4j knowledge graph populated from research papers. Requires a running Neo4j instance.

---

## Prerequisites

| Dependency | Version | Purpose |
|-----------|---------|---------|
| Python | 3.12+ | Runtime |
| `mcp` (FastMCP) | >= 1.23 | MCP server framework |
| `rdflib` | >= 7.5 | OWL ontology parsing |
| `neo4j` (Python driver) | >= 5.28 | Graph database connectivity |
| Neo4j (Docker) | 5.x | Knowledge graph storage |

Install Python dependencies (if not already in your virtual environment):

```bash
pip install mcp rdflib neo4j
```

## Neo4j Setup

The graph retrieval tools require a running Neo4j instance. See [`NEO4J_README.md`](../NEO4J_README.md) in the repo root for full Docker setup instructions. The quick version:

```bash
# Start Neo4j via the provided script
./start_neo4j.sh

# Or via Docker Compose
sudo docker compose up -d
```

**Connection defaults** (configurable via environment variables):

| Variable | Default |
|----------|---------|
| `NEO4J_URI` | `bolt://localhost:7687` |
| `NEO4J_USER` | `neo4j` |
| `NEO4J_PASSWORD` | `password` |
| `NEO4J_DATABASE` | `neo4j` |

These are read by `PhotonicsAI/KnowledgeBase/Neo4j/config.py`.

### Populating the Knowledge Graph

An empty Neo4j instance won't return useful results. Seed it with the foundational ontology and (optionally) ingest research papers:

```bash
# 1. Seed foundational ontology (components, functions, principles from YAML primitives)
python reinitialize_neo4j_kb.py

# 2. Ingest research papers (optional, adds extracted entities)
python process_papers.py
```

## Running the Server

### Standalone (for testing)

```bash
python mcp_servers/kg_server.py
```

This starts the MCP server on stdio. You can test it with:

```bash
mcp dev mcp_servers/kg_server.py
```

### As a Cursor MCP Server

Add the following to your `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "kg-server": {
      "command": "/path/to/venv/bin/python",
      "args": ["mcp_servers/kg_server.py"],
      "cwd": "/path/to/PhIDO-Release"
    }
  }
}
```

### Used by Other PhIDO Components

The KG server's tools are also imported directly as Python functions by:

- **Interpreter agent** (`interpreter_agent.py`) — for grounding design concepts during interpretation
- **KG chat agent** (`kg_agent.py`) — standalone conversational Q&A agent over the knowledge graph
- **KG chat UI** (`kg_chat.py`) — Streamlit interface for the KG agent

## Available Tools

### Ontology Tools (always available, no Neo4j required)

| Tool | Description |
|------|-------------|
| `get_ontology_schema()` | Returns the full PIC ontology: classes, object properties (relationships), and data properties. Parsed from `pic_ontology.ttl`. |
| `list_entity_types()` | Returns relationship types from both the static ontology and any dynamically discovered types in the KG (via the SchemaRegistry). Falls back to ontology-only if Neo4j is unavailable. |

### Graph Retrieval Tools (require Neo4j)

| Tool | Description |
|------|-------------|
| `search_concepts(query, entity_type?)` | Semantic vector search across the KG. Finds entities matching a natural-language query (e.g., "high speed optical modulator"). Optionally filter by entity type: `Components`, `Architectures`, `Properties`, `Design_Functions`, `Physical_Principles`. |
| `get_concept_neighborhood(concept_name, max_hops?)` | Graph traversal from a named concept. Returns all connected entities within `max_hops` (1–3), grouped by relationship type and direction. |
| `resolve_function(function_description)` | Reverse lookup: given a design function (e.g., "Modulation"), find all components/architectures that perform it, along with their physical principles. Tries exact match first, then semantic fallback. |
| `get_component_properties(component_name)` | Returns all known properties, design functions, physical principles, and sub-components for a named component or architecture. |
| `kg_stats()` | Summary statistics: node counts by type, relationship counts by type, totals. |

## Behavior Without Neo4j

The server starts and operates in **degraded mode** if Neo4j is unavailable:

- Ontology tools (`get_ontology_schema`, `list_entity_types`) work normally
- Graph retrieval tools return a JSON error message with the suggestion to start Neo4j
- Neo4j connection is lazy — it's only attempted when a graph tool is first called
- If `rdflib` is also missing, ontology tools fall back to a hardcoded subset of the schema

## Entity Types in the Knowledge Graph

| Entity Type | Neo4j Label | Description |
|-------------|-------------|-------------|
| Components | `Component` | Photonic devices (e.g., MZM, Waveguide, Photodetector) |
| Architectures | `Architecture` | Composite photonic structures (e.g., MZI, Ring Resonator) |
| Properties | `Property` | Observable/measurable quantities (e.g., Insertion Loss, Bandwidth) |
| Design Functions | `Design_Function` | Functional roles (e.g., Modulation, Wavelength Filtering) |
| Physical Principles | `Physical_Principle` | Underlying physics (e.g., Plasma Dispersion, Thermo-Optic Effect) |
| Documents | `Document` | Source papers from which entities were extracted |

## Relationship Types

| Relationship | Direction | Meaning |
|-------------|-----------|---------|
| `PERFORMS_FUNCTION` | Component → Design_Function | What function a component performs |
| `BASED_ON_PRINCIPLE` | Component → Physical_Principle | What physics a component relies on |
| `HAS_PROPERTY` | Component → Property | What properties a component has |
| `USES_COMPONENT` | Architecture → Component | What sub-components an architecture contains |
| `RELATED_TO` | Any → Any | General semantic relationship |
| `EXTRACTED_FROM` | Entity → Document | Provenance tracking |

## KG Chat Agent

A standalone conversational agent that uses the KG tools to answer photonics questions is available separately:

```bash
# Start the Streamlit chat UI
streamlit run mcp_servers/kg_chat.py
```

This provides a multi-turn chat interface with real-time tool-call visualization and automatic JSONL logging for benchmarking. See `kg_agent.py` for the agent logic and `kg_chat.py` for the UI.

## File Reference

| File | Purpose |
|------|---------|
| `mcp_servers/kg_server.py` | MCP server — ontology parsing + Neo4j graph retrieval tools |
| `mcp_servers/kg_agent.py` | Conversational ReAct agent that calls KG tools via direct import |
| `mcp_servers/kg_chat.py` | Streamlit chat UI for the KG agent |
| `PhotonicsAI/KnowledgeBase/GenerativeOntology/ontology/pic_ontology.ttl` | OWL ontology source |
| `PhotonicsAI/KnowledgeBase/Neo4j/config.py` | Neo4j connection configuration |
| `PhotonicsAI/KnowledgeBase/Neo4j/client.py` | Neo4j client with semantic search and schema registry |
| `logs/kg_agent_log.jsonl` | Auto-generated interaction logs from the KG chat agent |
