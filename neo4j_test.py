"""Test script for Neo4j Knowledge Base retrieval functions."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig

def test_direct_neighbors(client: Neo4jClient):
    """Test direct neighbor retrieval."""
    print("\n" + "="*60)
    print("TEST 1: Direct Neighbor Retrieval")
    print("="*60)
    
    # Test: Get neighbors of Mach-Zehnder Interferometer
    # Neo4j equivalent: MATCH (n)-[r]-(m) WHERE n.name = ...
    print("\n1. Getting neighbors of 'Mach–Zehnder Interferometer':")
    
    query = """
    MATCH (n {name: $name})-[r]-(m)
    RETURN m, type(r) as rel_type
    LIMIT 5
    """
    
    with client.driver.session() as session:
        # Note: Handling en-dash vs hyphen might be needed, but assuming user fixed data or exact match works
        # Try both if first fails
        name = "Mach–Zehnder Interferometer"
        result = session.run(query, name=name)
        records = list(result)
        
        if not records:
             # Fallback
             name = "Mach-Zehnder Interferometer"
             result = session.run(query, name=name)
             records = list(result)
             
        print(f"   Found {len(records)} neighbors for '{name}':")
        for record in records:
            node = record["m"]
            print(f"   - {node.get('name', 'Unknown')} ({list(node.labels)[0]}) via {record['rel_type']}")

    # Test: Get only specific edge types
    print("\n2. Getting only functions performed by 'Grating Coupler':")
    query_func = """
    MATCH (n:Component {name: 'Grating Coupler'})-[:PERFORMS_FUNCTION]->(m)
    RETURN m
    """
    with client.driver.session() as session:
        result = session.run(query_func)
        records = list(result)
        print(f"   Found {len(records)} functions:")
        for record in records:
            print(f"   - {record['m'].get('name')}")

    # Test: Get properties of a component
    print("\n3. Getting properties of 'Ring Resonator':")
    query_prop = """
    MATCH (n:Component {name: 'Ring Resonator'})-[:HAS_PROPERTY]->(m)
    RETURN m
    """
    with client.driver.session() as session:
        result = session.run(query_prop)
        records = list(result)
        print(f"   Found {len(records)} properties:")
        for record in records:
            print(f"   - {record['m'].get('name')}")


def test_semantic_search(client: Neo4jClient):
    """Test vector-based semantic search."""
    print("\n" + "="*60)
    print("TEST 2: Vector-Based Semantic Search")
    print("="*60)
    
    queries = [
        ("optical modulation devices that control light amplitude", "Components"),
        ("components that perform phase shifting and modulation", "Components"),
        ("signal attenuation and insertion loss properties", "Properties"),
        ("bandwidth and frequency response characteristics", "Properties"),
        ("wave interference and phase difference generation in photonic systems", "Physical_Principles"),
        ("evanescent coupling and mode interaction mechanisms", "Physical_Principles"),
        ("photodetectors that convert light to electrical signals", "Components"),
        ("resonant structures and ring resonators for wavelength filtering", "Components")
    ]

    for i, (text, collection) in enumerate(queries, 1):
        print(f"\n{i}. Searching for '{text}' in {collection}:")
        results = client.semantic_search(text, collection, limit=5, threshold=0.2)
        print(f"   Found {len(results)} results:")
        for res in results:
            print(f"   - {res.get('name')} (score: {res.get('score', 0):.4f})")


def test_relationship_traversal(client: Neo4jClient):
    """Test relationship traversal."""
    print("\n" + "="*60)
    print("TEST 3: Relationship Traversal")
    print("="*60)
    
    # Test: Find all components that use a specific principle
    print("\n1. Finding components based on 'Interference' principle:")
    query = """
    MATCH (c)-[:BASED_ON_PRINCIPLE]->(p:Physical_Principle {name: 'Interference'})
    RETURN c
    LIMIT 5
    """
    with client.driver.session() as session:
        result = session.run(query)
        records = list(result)
        print(f"   Found {len(records)} components/architectures:")
        for r in records:
            print(f"   - {r['c'].get('name')}")

    # Test: Find architectures using a specific component
    print("\n2. Finding architectures using 'Directional Coupler':")
    query = """
    MATCH (a:Architecture)-[:USES_COMPONENT]->(c:Component {name: 'Directional Coupler'})
    RETURN a
    """
    with client.driver.session() as session:
        result = session.run(query)
        records = list(result)
        print(f"   Found {len(records)} architectures:")
        for r in records:
            print(f"   - {r['a'].get('name')}")


def test_hybrid_search(client: Neo4jClient):
    """Test hybrid search combining vector and graph queries."""
    print("\n" + "="*60)
    print("TEST 4: Hybrid Search (Simulated)")
    print("="*60)
    
    # Neo4j Client doesn't have a direct 'hybrid_search' method in the interface yet 
    # (unless added recently), but we can simulate it or use semantic search + post-filtering.
    
    print("\n1. Searching for 'modulator' components with 'Insertion_Loss' property:")
    # Step 1: Vector Search
    candidates = client.semantic_search("modulator", "Components", limit=20, threshold=0.2)
    
    # Step 2: Filter by relationship
    filtered = []
    with client.driver.session() as session:
        for cand in candidates:
            # Check for edge
            check_query = """
            MATCH (n:Component {name: $name})-[:HAS_PROPERTY]->(p {name: 'Insertion_Loss'})
            RETURN count(p) > 0 as exists
            """
            res = session.run(check_query, name=cand['name'])
            if res.single()['exists']:
                filtered.append(cand)
    
    print(f"   Found {len(filtered)} results (from {len(candidates)} vector matches):")
    for res in filtered[:5]:
        print(f"   - {res.get('name')}")


def test_convenience_methods(client: Neo4jClient):
    """Test convenience query methods."""
    print("\n" + "="*60)
    print("TEST 5: Convenience Methods")
    print("="*60)
    
    # Test: Find by name
    print("\n1. Finding 'Grating Coupler' by name:")
    comp = client.find_by_name("Grating Coupler", "Components")
    if comp:
        print(f"   Found: {comp.get('name')}")
        print(f"   Type: {comp.get('type')}")
    else:
        print("   Not found")


def test_visualization(client: Neo4jClient):
    """Test PyVis interactive visualization."""
    print("\n" + "="*60)
    print("TEST 6: PyVis Interactive Visualization")
    print("="*60)
    
    try:
        from PhotonicsAI.KnowledgeBase.Neo4j.visualization import Neo4jVisualizer
        viz = Neo4jVisualizer(client)
        output_file = "neo4j_kb_graph_test.html"
        viz.visualize_graph(output_file=output_file)
        print(f"   ✓ Interactive visualization saved to: {output_file}")
    except Exception as e:
        print(f"   ✗ Error creating visualization: {e}")

def main():
    print("="*60)
    print("Neo4j Knowledge Base - Retrieval Tests")
    print("="*60)
    
    try:
        client = Neo4jClient()
        client.connect()
        print("Connected to Neo4j.")
        
        test_direct_neighbors(client)
        test_semantic_search(client)
        test_relationship_traversal(client)
        test_hybrid_search(client)
        test_convenience_methods(client)
        test_visualization(client)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'client' in locals():
            client.close()

if __name__ == "__main__":
    main()

