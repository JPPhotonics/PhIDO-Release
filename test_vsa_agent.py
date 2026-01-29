"""Test script for VSA Agent."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from PhotonicsAI.KnowledgeBase.agents.vsa_agent import VSAAgent
from PhotonicsAI.KnowledgeBase.agents.ppc_agent.models import PPCResult, NormalizedEntity, NewConcept, KBUpdateSuggestion
from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient as KnowledgeBaseClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

def _setup_real_agent():
    """Setup VSA Agent with real Neo4j connection."""
    config = Neo4jConfig()
    
    print(f"Connecting to Neo4j at {config.uri}...")
    try:
        client = KnowledgeBaseClient(config=config)
        client.connect()
        print("✓ Connected to Neo4j")
        return VSAAgent(kb_client=client, llm_model="gemini-2.5-pro")
    except Exception as e:
        print(f"X Failed to connect to Neo4j: {e}")
        print("  Make sure Neo4j is running.")
        raise e

def _visualize_manifest(payload, output_filename="vsa_graph.html"):
    """Generate an interactive PyVis graph from the VSA manifest."""
    try:
        from pyvis.network import Network
    except ImportError:
        print("⚠ PyVis not installed. Skipping visualization.")
        print("  Install with: pip install pyvis")
        return

    # Initialize network with dark theme (matching arangoDB_test style)
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
    
    # Collection colors
    colors = {
        "Components": "#ff9900",
        "Architectures": "#00ccff", 
        "Properties": "#cc00ff",
        "Design_Functions": "#00ff99",
        "Physical_Principles": "#ff0066",
        "Documents": "#aaaaaa"
    }

    # Add Nodes
    for node in payload.nodes:
        color = colors.get(node.collection, "#999999")
        title = f"{node.name}\n({node.collection})\nOp: {node.operation}"
        if node.description:
            title += f"\n\n{node.description[:200]}..."
            
        net.add_node(
            node.name, 
            label=node.name, 
            title=title, 
            color=color,
            shape="dot" if node.collection != "Documents" else "box"
        )
    
    # Ensure Document node exists (it might not be in nodes list but is in edges)
    net.add_node(
        payload.document_key,
        label=f"DOC: {payload.document_key}",
        color=colors["Documents"],
        shape="box"
    )

    # Add Edges
    for edge in payload.edges:
        if edge.edge_collection == "EXTRACTED_FROM":
            continue
            
        # Add source/target nodes if they don't exist (e.g. from existing KB or inference referencing outside scope)
        # Note: PyVis adds them automatically if referenced in add_edge, but we want styling
        for n_name, n_coll in [(edge.from_node, edge.from_collection), (edge.to_node, edge.to_collection)]:
            # Check if node exists in graph
            node_exists = False
            for existing_node in net.nodes:
                if existing_node['id'] == n_name:
                    node_exists = True
                    break
            
            if not node_exists:
                color = colors.get(n_coll, "#555555")
                net.add_node(n_name, label=n_name, title=f"{n_name} ({n_coll})", color=color)

        # Edge label/title
        label = edge.edge_collection
        title = f"{edge.edge_collection}\nWeight: {edge.weight}"
        if edge.description:
            title += f"\n{edge.description}"
            
        net.add_edge(
            edge.from_node, 
            edge.to_node, 
            label=label, 
            title=title,
            arrows="to"
        )

    # Physics options for better layout (matching arangoDB_test style)
    net.force_atlas_2based()
    net.show_buttons(filter_=['physics'])
    
    # Save
    net.save_graph(output_filename)
    print(f"✓ Visualization saved to {output_filename}")


def _run_vsa_pipeline(ppc_result, document_key, test_name):
    """Helper to run the VSA pipeline with real dependencies."""
    
    print(f"\n[{test_name}] Initializing VSA Agent (Real LLM + Real DB)...")
    
    try:
        agent = _setup_real_agent()
        
        # 3. Execution
        print(f"[{test_name}] Executing VSA Agent pipeline...")
        payload = agent.process(ppc_result, document_key)
        
        # 4. Assertions / Validation
        print(f"\n[{test_name}] Validating Output Payload...")
        
        print(f"  - Document Key: {payload.document_key}")
        print(f"  - Generated At: {payload.generated_at}")
        print(f"  - Validation Flags: {len(payload.validation_flags)} flags")
        for flag in payload.validation_flags:
            print(f"    ! {flag}")
        
        # Check Nodes
        print(f"\n  [Nodes]: {len(payload.nodes)}")
        node_counts = {"CREATE": 0, "UPDATE": 0, "MERGE": 0}
        for node in payload.nodes:
            node_counts[node.operation] += 1
            if node.operation == "UPDATE":
                print(f"    - UPDATE {node.name}: Description len={len(node.description or '')}")
            
        print(f"    - Breakdown: {node_counts}")
        
        # Check Edges
        print(f"\n  [Edges]: {len(payload.edges)}")
        edge_types = {}
        for edge in payload.edges:
            edge_types[edge.edge_collection] = edge_types.get(edge.edge_collection, 0) + 1
        
        for etype, count in edge_types.items():
            print(f"    - {etype}: {count}")

        # Basic Sanity Checks
        if len(payload.nodes) > 0:
            print(f"\n  ✓ Success: Generated nodes from {test_name}.")
        else:
             print(f"\n  ! Warning: No nodes generated from {test_name} (might be expected if inputs were filtered).")

        # Save manifest
        output_file = Path(f"vsa_manifest_{test_name.lower().replace(' ', '_')}.json")
        with open(output_file, 'w') as f:
            f.write(payload.model_dump_json(indent=2))
        print(f"\n  Saved manifest to: {output_file}")
        
        # Generate Visualization
        viz_file = f"vsa_viz_{test_name.lower().replace(' ', '_')}.html"
        _visualize_manifest(payload, viz_file)
        
    except Exception as e:
        print(f"\n  X Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()


def test_vsa_with_real_ppc_output():
    """Test VSA Agent with real output from PPC Agent."""
    print("\n" + "=" * 60)
    print("Test: VSA Agent Pipeline (Real PPC Output)")
    print("=" * 60)

    # 1. Check for PPC output file
    ppc_output_file = Path("ppc_agent_results.json")
    if not ppc_output_file.exists():
        print("ℹ No 'ppc_agent_results.json' found.")
        print("  Run 'python3 test_ppc_agent.py' first to generate real PPC output.")
        print("  Skipping this test.")
        return

    print(f"1. Loading real PPC output from {ppc_output_file}...")
    try:
        with open(ppc_output_file, 'r') as f:
            ppc_data = json.load(f)
        
        # 2. Reconstruct PPCResult object
        known = [NormalizedEntity(**e) for e in ppc_data.get("known_entities", [])]
        new = [NewConcept(**c) for c in ppc_data.get("new_concepts", [])]
        
        ppc_result = PPCResult(known_entities=known, new_concepts=new)
        print(f"  - Loaded {len(known)} known entities and {len(new)} new concepts.")
        
        document_key = "real_paper_pdf_test"
        
        # 3. Run Pipeline
        _run_vsa_pipeline(ppc_result, document_key, "Real PPC Output Test")
        
    except Exception as e:
        print(f"X Failed to load or process PPC output: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Neo4j password hint (defaults to "password" in Neo4jConfig)
    if not os.getenv("NEO4J_PASSWORD"):
        print("Note: Using default NEO4J_PASSWORD='password'")
        print("  If your Neo4j instance uses a different password, set NEO4J_PASSWORD.\n")
        
    test_vsa_with_real_ppc_output()
