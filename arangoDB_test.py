"""Test script for ArangoDB Knowledge Base retrieval functions."""

from PhotonicsAI.KnowledgeBase.ArangoDB import KnowledgeBaseClient, ArangoDBConfig


def test_direct_neighbors(client: KnowledgeBaseClient):
    """Test direct neighbor retrieval."""
    print("\n" + "="*60)
    print("TEST 1: Direct Neighbor Retrieval")
    print("="*60)
    
    # Test: Get neighbors of Mach-Zehnder Interferometer
    print("\n1. Getting neighbors of 'Mach–Zehnder Interferometer':")
    neighbors = client.get_neighbors("Mach–Zehnder Interferometer", "Components")
    print(f"   Found {len(neighbors)} neighbors:")
    for neighbor in neighbors[:5]:  # Show first 5
        print(f"   - {neighbor.get('name', 'Unknown')} ({neighbor.get('_id', 'Unknown')})")
    
    # Test: Get only specific edge types
    print("\n2. Getting only functions performed by 'Grating Coupler':")
    neighbors = client.get_neighbors(
        "Grating Coupler",
        "Components",
        edge_types=["PERFORMS_FUNCTION"]
    )
    print(f"   Found {len(neighbors)} functions:")
    for neighbor in neighbors:
        print(f"   - {neighbor.get('name', 'Unknown')}")
    
    # Test: Get properties of a component
    print("\n3. Getting properties of 'Ring Resonator':")
    neighbors = client.get_neighbors(
        "Ring Resonator",
        "Components",
        edge_types=["HAS_PROPERTY"]
    )
    print(f"   Found {len(neighbors)} properties:")
    for neighbor in neighbors:
        print(f"   - {neighbor.get('name', 'Unknown')}")


def test_semantic_search(client: KnowledgeBaseClient):
    """Test vector-based semantic search."""
    print("\n" + "="*60)
    print("TEST 2: Vector-Based Semantic Search")
    print("="*60)
    
    # Test: Search for modulation-related components with longer phrases
    print("\n1. Searching for 'optical modulation devices that control light amplitude' in Components:")
    results = client.semantic_search(
        "optical modulation devices that control light amplitude",
        "Components",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    print("\n2. Searching for 'components that perform phase shifting and modulation' in Components:")
    results = client.semantic_search(
        "components that perform phase shifting and modulation",
        "Components",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    # Test: Search for loss-related properties with descriptive queries
    print("\n3. Searching for 'signal attenuation and insertion loss properties' in Properties:")
    results = client.semantic_search(
        "signal attenuation and insertion loss properties",
        "Properties",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    print("\n4. Searching for 'bandwidth and frequency response characteristics' in Properties:")
    results = client.semantic_search(
        "bandwidth and frequency response characteristics",
        "Properties",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    # Test: Search for interference-related principles with longer descriptions
    print("\n5. Searching for 'wave interference and phase difference generation in photonic systems' in Physical_Principles:")
    results = client.semantic_search(
        "wave interference and phase difference generation in photonic systems",
        "Physical_Principles",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    print("\n6. Searching for 'evanescent coupling and mode interaction mechanisms' in Physical_Principles:")
    results = client.semantic_search(
        "evanescent coupling and mode interaction mechanisms",
        "Physical_Principles",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    # Test: Search for detector-related components
    print("\n7. Searching for 'photodetectors that convert light to electrical signals' in Components:")
    results = client.semantic_search(
        "photodetectors that convert light to electrical signals",
        "Components",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    # Test: Search for resonator-related concepts
    print("\n8. Searching for 'resonant structures and ring resonators for wavelength filtering' in Components:")
    results = client.semantic_search(
        "resonant structures and ring resonators for wavelength filtering",
        "Components",
        limit=5,
        threshold=0.2
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")


def test_relationship_traversal(client: KnowledgeBaseClient):
    """Test relationship traversal."""
    print("\n" + "="*60)
    print("TEST 3: Relationship Traversal")
    print("="*60)
    
    # Test: Find all components that use a specific principle
    print("\n1. Finding components based on 'Interference' principle:")
    # First, get the principle key
    principle = client.find_by_name("Interference", "Physical_Principles")
    if principle:
        print(f"   Found principle: {principle.get('name')}")
        # Traverse from principle to components
        components = client.find_components_by_principle("Interference")
        print(f"   Found {len(components)} components/architectures:")
        for comp in components[:5]:
            print(f"   - {comp.get('name', 'Unknown')} ({comp.get('_id', 'Unknown')})")
    
    # Test: Find architectures using a specific component
    print("\n2. Finding architectures using 'Directional Coupler':")
    architectures = client.find_architectures_using_component("Directional Coupler")
    print(f"   Found {len(architectures)} architectures:")
    for arch in architectures:
        print(f"   - {arch.get('name', 'Unknown')}")
    
    # Test: Traverse multiple relationships
    print("\n3. Traversing from component to its properties:")
    results = client.traverse(
        "Ring Resonator",
        "Components",
        relationship_path=["HAS_PROPERTY"],
        max_depth=1
    )
    print(f"   Found {len(results)} properties:")
    for result in results[:5]:
        print(f"   - {result.get('name', 'Unknown')}")


def test_hybrid_search(client: KnowledgeBaseClient):
    """Test hybrid search combining vector and graph queries."""
    print("\n" + "="*60)
    print("TEST 4: Hybrid Search")
    print("="*60)
    
    # Test: Search for modulators that have insertion loss property
    print("\n1. Searching for 'modulator' components with 'Insertion_Loss' property:")
    results = client.hybrid_search(
        "modulator",
        "Components",
        relationship_filters={
            "must_have_edge": "HAS_PROPERTY",
            "edge_target": "Insertion_Loss"
        },
        limit=5
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")
    
    # Test: Search for components that perform specific function
    print("\n2. Searching for components that perform 'Modulator' function:")
    results = client.hybrid_search(
        "phase shift",
        "Components",
        relationship_filters={
            "must_have_edge": "PERFORMS_FUNCTION",
            "edge_target": "Modulator"
        },
        limit=5
    )
    print(f"   Found {len(results)} results:")
    for result in results:
        print(f"   - {result.get('name', 'Unknown')}")


def test_convenience_methods(client: KnowledgeBaseClient):
    """Test convenience query methods."""
    print("\n" + "="*60)
    print("TEST 5: Convenience Methods")
    print("="*60)
    
    # Test: Find by name
    print("\n1. Finding 'Grating Coupler' by name:")
    component = client.find_by_name("Grating Coupler", "Components")
    if component:
        print(f"   Found: {component.get('name')}")
        print(f"   Type: {component.get('type')}")
        print(f"   Source: {component.get('source')}")
    else:
        print("   Not found")
    
    # Test: Find components by principle
    print("\n2. Finding components based on 'Evanescent_Coupling' principle:")
    components = client.find_components_by_principle("Evanescent_Coupling")
    print(f"   Found {len(components)} components:")
    for comp in components[:5]:
        print(f"   - {comp.get('name', 'Unknown')}")
    
    # Test: Find architectures using component
    print("\n3. Finding architectures using 'Y-Branch':")
    architectures = client.find_architectures_using_component("Y-Branch")
    print(f"   Found {len(architectures)} architectures:")
    for arch in architectures:
        print(f"   - {arch.get('name', 'Unknown')}")


def test_pyvis_visualization(client: KnowledgeBaseClient):
    """Test PyVis interactive visualization."""
    print("\n" + "="*60)
    print("TEST 6: PyVis Interactive Visualization")
    print("="*60)
    
    try:
        from pyvis.network import Network
    except ImportError:
        print("⚠ PyVis not installed. Skipping visualization.")
        print("  Install with: pip install pyvis")
        return

    try:
        output_file = "knowledge_base_graph.html"
        print(f"\n1. Generating PyVis graph visualization: {output_file}...")
        
        # Initialize network
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
        
        # Collection colors (matching VSA viz style)
        colors = {
            "Components": "#ff9900",
            "Architectures": "#00ccff", 
            "Properties": "#cc00ff",
            "Design_Functions": "#00ff99",
            "Physical_Principles": "#ff0066",
            "Documents": "#aaaaaa"
        }

        print("   Fetching ALL nodes and edges from database...")
        
        # Helper to add node
        visited_nodes = set()
        def add_node_to_net(node_doc, collection_name):
            node_id = node_doc["_id"]
            if node_id in visited_nodes:
                return
            
            color = colors.get(collection_name, "#999999")
            title = f"{node_doc.get('name')}\n({collection_name})"
            if node_doc.get('description'):
                title += f"\n\n{node_doc.get('description')[:200]}..."
            
            net.add_node(
                node_id, 
                label=node_doc.get('name'), 
                title=title, 
                color=color,
                shape="dot"
            )
            visited_nodes.add(node_id)

        # 1. Fetch ALL vertices
        vertex_collections = [
            "Components", "Architectures", "Properties", 
            "Design_Functions", "Physical_Principles"
        ]
        
        for coll_name in vertex_collections:
            print(f"   - Fetching {coll_name}...")
            cursor = client.db.collection(coll_name).all()
            for doc in cursor:
                add_node_to_net(doc, coll_name)

        # 2. Fetch ALL edges
        print("   Fetching edges...")
        edge_collections = [
            "PERFORMS_FUNCTION", "BASED_ON_PRINCIPLE", "HAS_PROPERTY",
            "USES_COMPONENT", "RELATED_TO"
        ]
        
        edge_count = 0
        for edge_coll in edge_collections:
            cursor = client.db.collection(edge_coll).all()
            for edge in cursor:
                # Only add edge if both nodes exist in our graph (they should, since we fetched all vertices)
                if edge["_from"] in visited_nodes and edge["_to"] in visited_nodes:
                    net.add_edge(
                        edge["_from"], 
                        edge["_to"], 
                        label=edge_coll, 
                        title=edge_coll,
                        arrows="to"
                    )
                    edge_count += 1
        
        print(f"   Graph stats: {len(visited_nodes)} nodes, {edge_count} edges")

        # Physics options
        net.force_atlas_2based()
        net.show_buttons(filter_=['physics'])
        
        net.save_graph(output_file)
        
        print(f"\n   ✓ Interactive visualization saved to: {output_file}")
        print(f"   Open {output_file} in your browser to explore the graph!")
        print("\n   Features:")
        print("   - Hover over nodes to see details")
        print("   - Click and drag to pan")
        print("   - Use zoom controls")
        print("   - Select node by type via dropdown")
        
    except Exception as e:
        print(f"\n   ✗ Error creating visualization: {e}")
        import traceback
        traceback.print_exc()


def test_graph_exists(client: KnowledgeBaseClient):
    """Test that the named graph 'KnowledgeGraph' exists and is usable."""
    print("\n" + "="*60)
    print("TEST 7: Graph Existence and Basics")
    print("="*60)

    try:
        has_graph = client.db.has_graph("KnowledgeGraph")
        print(f"\nGraph 'KnowledgeGraph' exists: {has_graph}")
        if not has_graph:
            print("  ✗ Named graph missing. Ensure setup.initialize_database creates it.")
            return

        graph = client.db.graph("KnowledgeGraph")
        edge_defs = graph.edge_definitions()
        print(f"  Edge definitions: {[d.get('edge_collection') for d in edge_defs]}")

        # Quick traversal sanity: count vertices and edges
        v_count = sum(client.db.collection(c).count() for c in [
            "Components", "Architectures", "Properties",
            "Design_Functions", "Physical_Principles", "Documents"
        ])
        e_count = sum(client.db.collection(c).count() for c in [
            "PERFORMS_FUNCTION", "BASED_ON_PRINCIPLE", "HAS_PROPERTY",
            "USES_COMPONENT", "RELATED_TO", "EXTRACTED_FROM"
        ])
        print(f"  Vertex count (all collections): {v_count}")
        print(f"  Edge count (all edge collections): {e_count}")
        print("\n  ✓ Graph existence test passed\n")
    except Exception as e:
        print(f"\n   ✗ Error checking graph existence: {e}")
        import traceback
        traceback.print_exc()


def test_ontology_integration(client: KnowledgeBaseClient):
    """Test RDF ontology integration and validation."""
    print("\n" + "="*60)
    print("TEST 8: Ontology Integration and Validation")
    print("="*60)
    
    try:
        from PhotonicsAI.KnowledgeBase.GenerativeOntology.src.graph_manager import GraphManager
        
        # Use existing client config
        db_config = {
            'url': client.hosts[0], # Assuming single host for now
            'username': client.username,
            'password': client.password,
            'db_name': client.db_name
        }
        
        print("\n1. Initializing GraphManager...")
        manager = GraphManager(db_config)
        
        # Test component
        component_name = "Mach–Zehnder Interferometer" 
        ontology_path = "PhotonicsAI/KnowledgeBase/GenerativeOntology/ontology/pic_ontology.ttl"
        
        print(f"\n2. Fetching RDF subgraph for '{component_name}'...")
        rdf_graph = manager.fetch_subgraph_rdf(component_name)
        print(f"   Generated RDF graph with {len(rdf_graph)} triples.")
        
        # Print a few triples to verify
        print("   Sample triples:")
        count = 0
        for s, p, o in rdf_graph:
            print(f"   - {s.n3()} {p.n3()} {o.n3()}")
            count += 1
            if count >= 5: break
            
        print(f"\n3. Exporting to Turtle format...")
        output_file = "mzi_subgraph.ttl"
        manager.export_to_turtle(component_name, output_file)
        print(f"   Exported to {output_file}")
        
        print(f"\n4. Validating against ontology ({ontology_path})...")
        try:
            report = manager.validate_subgraph(component_name, ontology_path)
            print(f"   Conforms: {report['conforms']}")
            print(f"   Report:\n{report['report_text']}")
        except Exception as e:
            print(f"   Validation failed to run: {e}")
            
    except ImportError as e:
        print(f"\n   ✗ Missing dependency: {e}")
        print("   Install with: pip install arango-rdf rdflib pyshacl")
    except Exception as e:
        print(f"\n   ✗ Error during ontology test: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all retrieval tests."""
    print("="*60)
    print("ArangoDB Knowledge Base - Retrieval Tests")
    print("="*60)
    
    # Create client
    client = KnowledgeBaseClient()
    
    # Initialize database schema (skip if already initialized)
    print("\nInitializing database...")
    client.initialize()
    
    # Import YAML data (skip if already imported)
    print("Importing YAML data...")
    yaml_dir = "PhotonicsAI/KnowledgeBase/GenerativeOntology/Primitives"
    try:
        client.import_yaml_data(yaml_dir)
        print("Import complete.")
    except Exception as e:
        print(f"Note: Import may have already been done or encountered error: {e}")
    
    # Run all tests
    try:
        test_direct_neighbors(client)
        test_semantic_search(client)
        test_relationship_traversal(client)
        test_hybrid_search(client)
        test_convenience_methods(client)
        test_pyvis_visualization(client) # Replaced plotly with pyvis
        test_graph_exists(client)
        test_ontology_integration(client) # Add ontology test
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        print("\nNote: If visualization was created, open knowledge_base_graph.html in your browser.")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
