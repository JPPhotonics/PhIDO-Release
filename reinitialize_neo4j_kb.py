"""
Script to re-import YAML ontology data into Neo4j.
Modeled after reinitialize_kb.py but for Neo4j backend.
"""

import os
import sys
from pathlib import Path

# Ensure package root is in path
sys.path.append(str(Path(__file__).parent))

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

def main():
    print("Initializing Neo4j KB Client...")
    
    # Check if Env vars are set, else warn user
    if not os.getenv("NEO4J_PASSWORD"):
        print("Warning: NEO4J_PASSWORD not set. Using default 'password'.")
    
    try:
        # Load config from env
        config = Neo4jConfig()
        client = Neo4jClient(config=config)
        client.connect()
        print(f"Connected to Neo4j at {config.uri}")
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return

    # Path to YAML files
    # Assuming this script is at repo root
    yaml_dir = Path("PhotonicsAI/KnowledgeBase/GenerativeOntology/Primitives")
    if not yaml_dir.exists():
        print(f"Error: YAML directory not found at {yaml_dir}")
        return

    # Initialize Schema & Indexes
    print("Initializing Vector Indexes and Constraints...")
    # Passing reset=True will wipe existing data to ensure a clean slate
    # Just like ArangoDB test script usually assumes
    client.initialize(reset=True)

    print(f"Importing YAML data from {yaml_dir}...")
    try:
        counts = client.import_yaml_data(yaml_dir)
        print(f"Success! Imported {counts['nodes']} nodes and {counts['edges']} edges.")
        
        # Simple verification
        print("\n--- Verification ---")
        test_entities = [
            ("Mach–Zehnder Interferometer", "Components"), # En-dash
            ("Ring Resonator", "Components"),
            ("Interference", "Physical_Principles")
        ]
        
        for name, collection in test_entities:
            # Try finding with exact name or slight variations handled by client
            node = client.find_by_name(name, collection)
            if node:
                print(f"✓ Found '{name}' ({collection})")
            else:
                # Try fallback with hyphen if en-dash failed
                if "–" in name:
                    name_hyphen = name.replace("–", "-")
                    node = client.find_by_name(name_hyphen, collection)
                    if node:
                        print(f"✓ Found '{name_hyphen}' (fallback for {name})")
                    else:
                        print(f"✗ Could not find '{name}' or '{name_hyphen}'")
                else:
                    print(f"✗ Could not find '{name}'")

        # Create PyVis Visualization
        print("\nGenerating Knowledge Graph Visualization...")
        try:
            from PhotonicsAI.KnowledgeBase.Neo4j.visualization import Neo4jVisualizer
            viz = Neo4jVisualizer(client)
            output_file = "neo4j_kb_graph.html"
            viz.visualize_graph(output_file=output_file)
            print(f"✓ Graph visualization saved to {output_file}")
        except ImportError:
            print("⚠ Could not import Neo4jVisualizer. Skipping visualization.")
        except Exception as e:
            print(f"⚠ Visualization failed: {e}")
            
    except Exception as e:
        print(f"Error during import: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    main()

