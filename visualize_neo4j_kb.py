"""
Script to generate a PyVis visualization of the Neo4j Knowledge Base.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
from PhotonicsAI.KnowledgeBase.Neo4j.visualization import Neo4jVisualizer

def main():
    print("Initializing Neo4j Client...")
    
    # Check for Neo4j connection env vars if needed, though Config handles defaults
    if not os.getenv("NEO4J_PASSWORD"):
        print("Note: Using default NEO4J_PASSWORD='password'")
        print("  If your Neo4j instance uses a different password, set NEO4J_PASSWORD.\n")
    
    try:
        client = Neo4jClient(config=Neo4jConfig())
        client.connect()
        print("✓ Connected to Neo4j")
        
        output_file = "neo4j_kb_graph.html"
        print(f"Generating visualization to {output_file}...")
        
        viz = Neo4jVisualizer(client)
        viz.visualize_graph(output_file=output_file)
        
        print(f"✓ Visualization saved to {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"X Error: {e}")
        if "Connection refused" in str(e) or "Can't connect" in str(e):
             print("\nMake sure the Neo4j container is running:")
             print("  sudo docker start neo4j-phido")

if __name__ == "__main__":
    main()

