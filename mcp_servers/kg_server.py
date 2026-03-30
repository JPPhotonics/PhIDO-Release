"""
MCP Server: Photonics Knowledge Base
Combines ontology schema (from pic_ontology.ttl) with Neo4j GraphRAG retrieval.

The ontology tools are always available (pure Python, no dependencies).
The graph retrieval tools require a running Neo4j instance (bolt://localhost:7687).

Run with: python mcp_servers/kg_server.py
Test with: mcp dev mcp_servers/kg_server.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# 0. Path setup — allow imports from the PhIDO codebase
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# 1. Initialize the MCP server
# ---------------------------------------------------------------------------
mcp = FastMCP(
    "PhIDO Knowledge Graph",
    instructions=(
        "Query the photonics knowledge graph. "
        "Use ontology tools (get_ontology_schema, list_entity_types) for class/relationship definitions. "
        "Use graph tools (search_concepts, get_concept_neighborhood, resolve fundtion, "
        "get_component_properties) for instance-level knowledge from research papers."
    ),
)

# ---------------------------------------------------------------------------
# 2. Load the OWL ontology at startup (always available, no Neo4j needed)
# ---------------------------------------------------------------------------
ONTOLOGY_PATH = (
    REPO_ROOT / "PhotonicsAI" / "KnowledgeBase" / "GenerativeOntology" / "ontology" / "pic_ontology.ttl"
)

_ontology_data: Optional[dict] = None

def _load_ontology() -> dict:
    """Parse pic_ontology.ttl with rdflib and extract classes, object properties, data properties."""
    global _ontology_data
    if _ontology_data is not None:
        return _ontology_data
    try:
        from rdflib import Graph as RDFGraph, RDF, RDFS, OWL, Namespace
        
        g = RDFGraph()
        g.parse(str(ONTOLOGY_PATH), format="turtle")
        
        # --- Classes ---
        classes = []
        for cls in g.subjects(RDF.type, OWL.Class):
            name = str(cls).split("#")[-1] if "#" in str(cls) else str(cls)
            comment = str(g.value(cls, RDFS.comment) or "")
            superclasses = [
                str(sc).split("#")[1]
                for sc in g.objects(cls, RDFS.subClassOF)
                if "#" in str(sc)
            ]
            classes.append({
                "name": name,
                "description": comment,
                "superclasses": superclasses,
            })
        
        # --- Object Properties (relationships) ---
        obj_props = []
        for prop in g.subjects(RDF.type, OWL.ObjectProperty):
            name = str(prop).split("#")[-1] if "#" in str(prop) else str(prop)
            label = str(g.value(prop, RDFS.label) or name)
            domain = g.value(prop, RDFS.domain)
            range_ = g.value(prop, RDFS.range)
            domain_name = str(domain).split("#")[-1] if domain and "#" in str(domain) else str(domain or "Any")
            range_name = str(range_).split("#")[-1] if range_ and "#" in str(range_) else str(range_ or "Any")
            obj_props.append(
                {
                    "property": name,
                    "label": label,
                    "domain": domain_name,
                    "range": range_name,
                }
            )
            
        # --- Data Properties ---
        data_props = []
        for prop in g.subjects(RDF.type, OWL.DatatypeProperty):
            name = str(prop).split("#")[-1] if "#" in str(prop) else str(prop)
            domain = g.value(prop, RDFS.domain)
            range_ = g.value(prop, RDFS.range)
            domain_name = str(domain).split("#")[-1] if domain and "#" in str(domain) else str(domain or "Any")
            range_name = str(range_).split("#")[-1] if range_ and "#" in str(range_) else str(range_ or "Any")
            data_props.append({
                "property": name,
                "domain": domain_name,
                "range": range_name,
            })
            
        _ontology_data = {
            "classes": classes,
            "object_properties": obj_props,
            "data_properties": data_props,
        }
        print(f"Ontology loaded: {len(classes)} classes, {len(obj_props)} object properties, {len(data_props)} data properties.")
        return _ontology_data
    
    except ImportError:
        print("Warning: rdflib not installed. Ontology tools will use fallback. ")
        _ontology_data = _fallback_ontology()
        return _ontology_data
    except Exception as e:
        print(f"Warning: Failed to parse ontology: {e}. Using fallback.")
        _ontology_data = _fallback_ontology()
        return _ontology_data
    
def _fallback_ontology() -> dict:
    """Hardcoded fallback if rdflib is unavailable or parsing fails."""
    return {
        "classes": [
            {"name": "Component", "description": "A photonic component (e.g., MZM, Waveguide)", "superclasses": ["Device", "FeatureOfInterest"]},
            {"name": "Architecture", "description": "A complex photonic architecture (e.g., MZI) composed of components", "superclasses": ["Component"]},
            {"name": "DesignFunction", "description": "The function performed by a component (e.g., Modulation)", "superclasses": ["Function"]},
            {"name": "PhysicalPrinciple", "description": "Underlying physical mechanism (e.g., Plasma Dispersion)", "superclasses": ["Process"]},
            {"name": "Property", "description": "A property concept (e.g., Insertion Loss)", "superclasses": ["QuantityKind", "ObservableProperty"]},
            {"name": "Document", "description": "Source document", "superclasses": ["Entity"]},
        ],
        "object_properties": [
            {"property": "performsFunction", "label": "PERFORMS_FUNCTION", "domain": "Component", "range": "DesignFunction"},
            {"property": "basedOnPrinciple", "label": "BASED_ON_PRINCIPLE", "domain": "Component", "range": "PhysicalPrinciple"},
            {"property": "usesComponent", "label": "USES_COMPONENT", "domain": "Architecture", "range": "Component"},
            {"property": "extractedFrom", "label": "EXTRACTED_FROM", "domain": "Component", "range": "Document"},
            {"property": "relatedTo", "label": "RELATED_TO", "domain": "Thing", "range": "Thing"},
        ],
        "data_properties": [
            {"property": "numericValue", "domain": "ComponentObservation", "range": "double"},
            {"property": "unit", "domain": "ComponentObservation", "range": "string"},
        ],
    }
    
# Preload ontology
_load_ontology()

# ---------------------------------------------------------------------------
# 3. Neo4j connection (lazy — only when graph tools are called)
# ---------------------------------------------------------------------------
_neo4j_client = None
_neo4j_error: Optional[str] = None

def _get_neo4j():
    """Lazily connect to Neo4j. Returns (client, error_message)."""
    global _neo4j_client, _neo4j_error
    
    if _neo4j_client is not None:
        return _neo4j_client, None
    if _neo4j_error is not None:
        return None, _neo4j_error
    
    try:
        from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
        from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
        
        config = Neo4jConfig()
        client = Neo4jClient(config)
        client.connect()
        _neo4j_client = client
        print(f"Connected to Neo4j at {config.uri}")
        return client, None
    except Exception as e:
        _neo4j_error = f"Neo4j unavailable: {e}"
        print(f"Warning: {_neo4j_error}")
        return None, _neo4j_error

def _require_neo4j():
    """Get Neo4j client or return a JSON error string."""
    client, err = _get_neo4j()
    if client is None:
        return None, json.dumps({
            "error": "Neo4j is not available. Start it with: bash start_neo4j.sh",
            "details": err,
        })
    return client, None

# ---------------------------------------------------------------------------
# 4. Ontology Tools (always available)
# ---------------------------------------------------------------------------

@mcp.tool()
def get_ontology_schema() -> str:
    """Get the full PIC ontology schema: classes, object properties (relationships), and data properties.
    
    This describes the *types* and *alled relationships* in the photonics domain.
    Always available - does not require Neo4j.
    """
    onto = _load_ontology()
    return json.dumps(onto, indent=2)

@mcp.tool()
def list_entity_types() -> str:
    """Get all relationship types from both the static ontology and any discovered types in the KG.
    
    If Neo4j is running, includes dynamically discovered relationship types from the SchemaRegistry.
    Falls back to static ontology relationships if Neo4j is unavailable.
    """
    onto = _load_ontology()
    result = {
        "ontology_relationships": onto["object_properties"],
        "discovered_relationships": [],
    }
    
    client, _ = _get_neo4j()
    if client is not None:
        try:
            registry = client.get_schema_registry()
            active_types = registry.get_active_edge_types()
            #Filter to non-seed types (discovered by the GraphRAG pipeline)
            discovered = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "allowed_sources": t["allowed_source_types"],
                    "allowed_targets": t["allowed_target_types"],
                    "observation_count": t.get("observation_count", 0),
                }
                for t in active_types
                if not t.get("is_seed", True)
            ]
            result["discovered_relationships"] = discovered
            # Also include the full active set for reference
            result["all_active_types"] = [t["name"] for t in active_types]
        except Exception as e:
            result["registry_error"] = str(e)
    
    return json.dumps(result, indent=2)

# ---------------------------------------------------------------------------
# 5. Graph Retrieval Tools (require Neo4j)
# ---------------------------------------------------------------------------

@mcp.tool()
def search_concepts(query: str, entity_type: str = "") -> str:
    """Semantic vector search across the knowledge graph to find concepts matching a query.
    
    Uses embeddings to find the closest matching entities in the KG - this is the 
    "Retrieval" in GraphRAG. Works across all entity types or a specific one.
    
    Args:
        query: Natural language description, e.g. "high speed optical modulator"
            or "wavelength selective filter" or "carrier depletion effect"
        entity_type: Optional. One of: Components, Architectures, Properties,
            Design_Functions, Physical_Principles, PDK_Cells. If empty, searches all types.
    """
    client, err = _require_neo4j()
    if client is None:
        return err
    
    collections = (
        [entity_type]
        if entity_type
        else [
            "Components",
            "Architectures",
            "Properties",
            "Design_Functions",
            "Physical_Principles",
            "PDK_Cells",
        ]
    )
    
    all_results = []
    for coll in collections:
        try:
            hits = client.semantic_search(
                query_text=query,
                collection=coll,
                limit=5,
                threshold=0.4,
            )
            for h in hits:
                # Remove embedding vector from output (huge, not useful for LLM)
                h.pop("embedding", None)
                h["entity_type"] = coll
            all_results.extend(hits)
        except Exception as e:
            all_results.append({"entity_type": coll, "error": str(e)})
            
    # Sort by score descending, take top 10
    scored = [r for r in all_results if "score" in r]
    errors = [r for r in all_results if "error" in r]
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    
    return json.dumps({
        "query": query,
        "results": scored[:10],
        "errors": errors
    }, indent=2, default=str)
    
@mcp.tool()
def get_concept_neighborhood(concept_name: str, max_hops: int = 1) -> str:
    """Get a concept and its graph neighborhood - the "Graph" in GraphRAG.
    
    Returns the concept node and all relationships within max_hops, grouped by
    relationship type. This reveals what functions a component performs, what 
    principles it's based on, what properties it has, and what sub-components
    an architecture uses.
    
    Args:
        concept_name: Name of the concept, e.g. "MZI", "Ring_Resonator",
        "Carrier_Depletion", "Insertion_Loss"
        max_hops: How many relationship hops to traverse (1 or 2). Default 1.
    """
    client, err = _require_neo4j()
    if client is None:
        return err
    
    max_hops = min(max(max_hops, 1), 3)
    
    # Find the center node
    query_center = """
    MATCH (n)
    WHERE n.name = $name OR toLower(n.name) = toLower($name)
    RETURN n, labels(n) AS labels
    LIMIT 1
    """
    
    # Traverse neighborhood
    query_neighbors = f"""
    MATCH (center)
    WHERE center.name = $name OR toLower(center.name) = toLower($name)
    WITH center LIMIT 1
    MATCH p = (center)-[r*1..{max_hops}]-(neighbor)
    WITH center, neighbor, labels(neighbor) AS neighbor_labels,
        [rel IN relationships(p) | type(rel)] AS rel_types,
         [rel IN relationships(p) | startNode(rel) = center] AS directions
    RETURN DISTINCT
        neighbor.name AS neighbor_name,
        neighbor.description AS neighbor_description,
        neighbor_labels,
        rel_types,
        directions
    LIMIT 50
    """
    
    try:
        center_node = None
        neighbors = []

        with client.driver.session() as session:
            # Get center
            result = session.run(query_center, name=concept_name)
            record = result.single()
            if not record:
                return json.dumps({
                    "error": f"Concept '{concept_name}' not found in the knowledge graph.",
                    "suggestion": "Try search_concepts to find the correct name.",
                })
            node = record["n"]
            labels = record["labels"]
            center_node = {
                "name": node.get("name"),
                "description": node.get("description", ""),
                "type": labels[0] if labels else "Unknown",
            }

            # Get neighbors
            result = session.run(query_neighbors, name=concept_name)
            for rec in result:
                rel_types = rec["rel_types"] or []
                directions = rec["directions"] or []
                n_labels = rec["neighbor_labels"] or []

                # Determine relationship direction for first hop
                rel_label = rel_types[0] if rel_types else "RELATED_TO"
                is_outgoing = directions[0] if directions else True

                neighbors.append({
                    "name": rec["neighbor_name"],
                    "description": (rec["neighbor_description"] or "")[:200],
                    "type": n_labels[0] if n_labels else "Unknown",
                    "relationship": rel_label,
                    "direction": "outgoing" if is_outgoing else "incoming",
                    "hops": len(rel_types),
                })

        # Group by relationship type for readability
        grouped = {}
        for n in neighbors:
            key = f"{n['relationship']} ({'→' if n['direction'] == 'outgoing' else '←'})"
            grouped.setdefault(key, []).append({
                "name": n["name"],
                "type": n["type"],
                "description": n["description"],
                "hops": n["hops"],
            })

        return json.dumps(
            {"center": center_node, "neighborhood": grouped, "total_neighbors": len(neighbors)},
            indent=2,
            default=str,
        )

    except Exception as e:
        return json.dumps({"error": f"Graph traversal failed: {e}"})
    
@mcp.tool()
def resolve_function(function_description: str) -> str:
    """Find which components or architectures can perform a given design function.

    This is a reverse lookup: given a function like "modulation" or "wavelength filtering",
    find all components/architectures linked via PERFORMS_FUNCTION.

    Args:
        function_description: The design function to look up, e.g. "Modulator",
                              "Wavelength_Filter", "Phase_Shifter", "Photodetector"
    """
    client, err = _require_neo4j()
    if client is None:
        return err

    # First try exact match, then semantic search
    query_exact = """
    MATCH (c)-[:PERFORMS_FUNCTION]->(f:Design_Function)
    WHERE f.name = $name OR toLower(f.name) = toLower($name)
    OPTIONAL MATCH (c)-[:BASED_ON_PRINCIPLE]->(p:Physical_Principle)
    RETURN c.name AS component, labels(c) AS component_labels,
           c.description AS description,
           collect(DISTINCT p.name) AS principles
    ORDER BY c.name
    """

    try:
        results = []
        with client.driver.session() as session:
            records = session.run(query_exact, name=function_description)
            for rec in records:
                labels = rec["component_labels"] or []
                results.append({
                    "component": rec["component"],
                    "type": labels[0] if labels else "Unknown",
                    "description": (rec["description"] or "")[:300],
                    "based_on_principles": rec["principles"],
                })

        if results:
            return json.dumps(
                {"function": function_description, "match_type": "exact", "components": results},
                indent=2,
                default=str,
            )

        # Fallback: semantic search on Design_Functions, then traverse
        hits = client.semantic_search(
            query_text=function_description,
            collection="Design_Functions",
            limit=3,
            threshold=0.4,
        )
        if not hits:
            return json.dumps({
                "function": function_description,
                "match_type": "none",
                "components": [],
                "message": "No matching design function found in the KG.",
            })

        # Use the best semantic match
        best_name = hits[0].get("name", "")
        with client.driver.session() as session:
            records = session.run(query_exact, name=best_name)
            for rec in records:
                labels = rec["component_labels"] or []
                results.append({
                    "component": rec["component"],
                    "type": labels[0] if labels else "Unknown",
                    "description": (rec["description"] or "")[:300],
                    "based_on_principles": rec["principles"],
                })

        return json.dumps(
            {
                "function": function_description,
                "resolved_to": best_name,
                "match_type": "semantic",
                "similarity": hits[0].get("score", 0),
                "components": results,
            },
            indent=2,
            default=str,
        )

    except Exception as e:
        return json.dumps({"error": f"Function resolution failed: {e}"})
    
@mcp.tool()
def get_component_properties(component_name: str) -> str:
    """Get all known properties and related knowledge for a component or architecture.

    Returns properties (HAS_PROPERTY), design functions (PERFORMS_FUNCTION),
    physical principles (BASED_ON_PRINCIPLE), and sub-components (USES_COMPONENT)
    from the knowledge graph.

    Args:
        component_name: Name of the component, e.g. "MZI", "Ring_Resonator",
                        "Directional_Coupler", "Micro-ring_Modulator"
    """
    client, err = _require_neo4j()
    if client is None:
        return err

    query = """
    MATCH (c)
    WHERE c.name = $name OR toLower(c.name) = toLower($name)
    WITH c LIMIT 1
    OPTIONAL MATCH (c)-[:HAS_PROPERTY]->(prop:Property)
    OPTIONAL MATCH (c)-[:PERFORMS_FUNCTION]->(func:Design_Function)
    OPTIONAL MATCH (c)-[:BASED_ON_PRINCIPLE]->(prin:Physical_Principle)
    OPTIONAL MATCH (c)-[:USES_COMPONENT]->(sub:Component)
    RETURN c.name AS name,
           c.description AS description,
           labels(c) AS labels,
           collect(DISTINCT {name: prop.name, description: prop.description}) AS properties,
           collect(DISTINCT {name: func.name, description: func.description}) AS functions,
           collect(DISTINCT {name: prin.name, description: prin.description}) AS principles,
           collect(DISTINCT {name: sub.name, description: sub.description}) AS sub_components
    """

    try:
        with client.driver.session() as session:
            result = session.run(query, name=component_name)
            record = result.single()

            if not record or not record["name"]:
                return json.dumps({
                    "error": f"Component '{component_name}' not found in the knowledge graph.",
                    "suggestion": "Try search_concepts to find the correct name.",
                })

            labels = record["labels"] or []

            # Clean up null entries from OPTIONAL MATCH
            def _clean(items):
                return [i for i in items if i.get("name") is not None]

            return json.dumps(
                {
                    "name": record["name"],
                    "type": labels[0] if labels else "Unknown",
                    "description": record["description"] or "",
                    "properties": _clean(record["properties"]),
                    "design_functions": _clean(record["functions"]),
                    "physical_principles": _clean(record["principles"]),
                    "sub_components": _clean(record["sub_components"]),
                },
                indent=2,
                default=str,
            )

    except Exception as e:
        return json.dumps({"error": f"Property lookup failed: {e}"})
    
@mcp.tool()
def get_pdk_implementations(concept_name: str) -> str:
    """Find all PDK cells that implement a given Component or Architecture concept.

    Returns concrete PDK_Cell nodes linked via IMPLEMENTS, including their metadata,
    port positions, and topology templates. Use this to find real fabrication-ready
    cells for an abstract concept.

    Args:
        concept_name: Name of a Component or Architecture, e.g. "MZI", "Ring_Resonator"
    """
    client, err = _require_neo4j()
    if client is None:
        return err

    query = """
    MATCH (comp)
    WHERE (comp:Component OR comp:Architecture)
      AND (comp.name = $name OR toLower(comp.name) = toLower($name))
    WITH comp LIMIT 1
    OPTIONAL MATCH (pdk:PDK_Cell)-[:IMPLEMENTS]->(comp)
    RETURN comp.name AS concept_name,
           labels(comp)[0] AS concept_type,
           collect({
             module_name: pdk.module_name,
             display_name: pdk.display_name,
             pdk_name: pdk.pdk_name,
             ports: pdk.ports,
             technology: pdk.technology,
             description: pdk.description,
             dx_um: pdk.dx_um,
             dy_um: pdk.dy_um,
             port_details: pdk.port_details,
             topology_template: pdk.topology_template,
             is_primitive: pdk.is_primitive
           }) AS implementations
    """

    try:
        with client.driver.session() as session:
            result = session.run(query, name=concept_name)
            record = result.single()

            if not record or not record["concept_name"]:
                return json.dumps({
                    "error": f"Concept '{concept_name}' not found.",
                    "suggestion": "Try search_concepts to find the correct name.",
                })

            impls = [i for i in record["implementations"] if i.get("module_name")]
            for impl in impls:
                # Parse JSON strings back for readability
                for json_field in ("port_details", "topology_template"):
                    raw = impl.get(json_field)
                    if isinstance(raw, str):
                        try:
                            impl[json_field] = json.loads(raw)
                        except (json.JSONDecodeError, TypeError):
                            pass

            return json.dumps({
                "concept": record["concept_name"],
                "concept_type": record["concept_type"],
                "implementation_count": len(impls),
                "implementations": impls,
            }, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": f"PDK implementation lookup failed: {e}"})


@mcp.tool()
def get_pdk_cell_details(module_name: str, pdk_name: str = "DemoPDK") -> str:
    """Get full details for a specific PDK cell from the knowledge graph.

    Returns the cell's metadata, port positions, topology template, composition
    tree (COMPOSED_OF), and all relationships (IMPLEMENTS, PERFORMS_FUNCTION, etc.).

    Args:
        module_name: Exact PDK module name, e.g. "mzi_2x2_heater_tin_cband"
        pdk_name: PDK identifier. Defaults to "DemoPDK".
    """
    client, err = _require_neo4j()
    if client is None:
        return err

    query = """
    MATCH (pdk:PDK_Cell {module_name: $module_name, pdk_name: $pdk_name})
    OPTIONAL MATCH (pdk)-[:IMPLEMENTS]->(comp:Component)
    OPTIONAL MATCH (pdk)-[:COMPOSED_OF]->(child:PDK_Cell)
    OPTIONAL MATCH (pdk)-[:PERFORMS_FUNCTION]->(func:Design_Function)
    OPTIONAL MATCH (pdk)-[:FABRICATED_WITH]->(prin:Physical_Principle)
    OPTIONAL MATCH (pdk)-[:EXHIBITS]->(prop:Property)
    RETURN pdk,
           collect(DISTINCT comp.name) AS implements,
           collect(DISTINCT {name: child.module_name, display_name: child.display_name}) AS children,
           collect(DISTINCT func.name) AS functions,
           collect(DISTINCT prin.name) AS principles,
           collect(DISTINCT prop.name) AS properties
    """

    try:
        with client.driver.session() as session:
            result = session.run(query, module_name=module_name, pdk_name=pdk_name)
            record = result.single()

            if not record or record["pdk"] is None:
                return json.dumps({
                    "error": f"PDK cell '{module_name}' not found in {pdk_name}.",
                })

            node = dict(record["pdk"])
            node.pop("embedding", None)

            # Parse JSON string fields
            for json_field in ("port_details", "topology_template", "numeric_specs", "parameters"):
                raw = node.get(json_field)
                if isinstance(raw, str):
                    try:
                        node[json_field] = json.loads(raw)
                    except (json.JSONDecodeError, TypeError):
                        pass

            def _clean(items):
                return [i for i in items if i.get("name") is not None]

            return json.dumps({
                "cell": node,
                "implements": record["implements"],
                "composed_of": _clean(record["children"]),
                "design_functions": record["functions"],
                "physical_principles": record["principles"],
                "properties": record["properties"],
            }, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": f"PDK cell details lookup failed: {e}"})


@mcp.tool()
def search_pdk_by_function(function_description: str) -> str:
    """Find PDK cells that can perform a given design function.

    Reverse lookup from Design_Function through PERFORMS_FUNCTION to PDK_Cells.
    Use this to find fabrication-ready components for a specific function.

    Args:
        function_description: The function to look up, e.g. "modulation",
                              "wavelength filtering", "photodetection"
    """
    client, err = _require_neo4j()
    if client is None:
        return err

    # Exact match first
    query_exact = """
    MATCH (pdk:PDK_Cell)-[:PERFORMS_FUNCTION]->(f:Design_Function)
    WHERE f.name = $name OR toLower(f.name) = toLower($name)
    RETURN pdk.module_name AS module_name,
           pdk.display_name AS display_name,
           pdk.pdk_name AS pdk_name,
           pdk.ports AS ports,
           pdk.description AS description,
           f.name AS function_name
    ORDER BY pdk.module_name
    """

    try:
        results = []
        with client.driver.session() as session:
            for rec in session.run(query_exact, name=function_description):
                results.append({
                    "module_name": rec["module_name"],
                    "display_name": rec["display_name"],
                    "pdk_name": rec["pdk_name"],
                    "ports": rec["ports"],
                    "description": (rec["description"] or "")[:200],
                    "matched_function": rec["function_name"],
                })

        if results:
            return json.dumps({
                "function": function_description,
                "match_type": "exact",
                "pdk_cells": results,
            }, indent=2, default=str)

        # Semantic fallback
        hits = client.semantic_search(
            query_text=function_description,
            collection="Design_Functions",
            limit=3,
            threshold=0.4,
        )
        if not hits:
            return json.dumps({
                "function": function_description,
                "match_type": "none",
                "pdk_cells": [],
                "message": "No matching design function found.",
            })

        best_name = hits[0].get("name", "")
        with client.driver.session() as session:
            for rec in session.run(query_exact, name=best_name):
                results.append({
                    "module_name": rec["module_name"],
                    "display_name": rec["display_name"],
                    "pdk_name": rec["pdk_name"],
                    "ports": rec["ports"],
                    "description": (rec["description"] or "")[:200],
                    "matched_function": rec["function_name"],
                })

        return json.dumps({
            "function": function_description,
            "resolved_to": best_name,
            "match_type": "semantic",
            "similarity": hits[0].get("score", 0),
            "pdk_cells": results,
        }, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": f"PDK function search failed: {e}"})


@mcp.tool()
def kg_stats() -> str:
    """Get summary statistics about the knowledge graph: node counts by type, edge counts, etc.

    Useful for understanding how much knowledge is available.
    """
    client, err = _require_neo4j()
    if client is None:
        return err

    query_nodes = """
    MATCH (n)
    WHERE any(label IN labels(n) WHERE label IN
        ['Component', 'Architecture', 'Property', 'Design_Function', 'Physical_Principle',
         'Document', 'PDK_Cell', 'PDK_Cell_History'])
    RETURN labels(n)[0] AS type, count(n) AS count
    ORDER BY count DESC
    """

    query_edges = """
    MATCH ()-[r]->()
    WHERE type(r) IN ['PERFORMS_FUNCTION', 'BASED_ON_PRINCIPLE', 'HAS_PROPERTY',
                       'USES_COMPONENT', 'RELATED_TO', 'EXTRACTED_FROM',
                       'IMPLEMENTS', 'COMPOSED_OF', 'EXHIBITS', 'FABRICATED_WITH', 'SUPERSEDES']
    RETURN type(r) AS relationship, count(r) AS count
    ORDER BY count DESC
    """

    try:
        stats = {"nodes": {}, "edges": {}, "total_nodes": 0, "total_edges": 0}
        with client.driver.session() as session:
            for rec in session.run(query_nodes):
                stats["nodes"][rec["type"]] = rec["count"]
                stats["total_nodes"] += rec["count"]
            for rec in session.run(query_edges):
                stats["edges"][rec["relationship"]] = rec["count"]
                stats["total_edges"] += rec["count"]

        return json.dumps(stats, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Stats query failed: {e}"})


# ---------------------------------------------------------------------------
# 6. Run the server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run()