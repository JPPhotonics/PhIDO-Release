"""Example script for visualizing the ArangoDB knowledge base graph."""

from PhotonicsAI.KnowledgeBase.ArangoDB import KnowledgeBaseClient


def main():
    """Demonstrate graph visualization options."""
    # Create client and connect
    client = KnowledgeBaseClient()
    client.connect()
    
    # Get visualizer
    viz = client.get_visualizer()
    
    # Print statistics
    print("Graph Statistics:")
    viz.print_statistics()
    
    # Option 1: Export to GraphML (for Gephi, Cytoscape, etc.)
    print("\nExporting to GraphML...")
    try:
        viz.export_to_graphml("knowledge_base.graphml", max_nodes=200)
        print("✓ GraphML export complete. Open with Gephi or Cytoscape.")
    except ImportError as e:
        print(f"✗ {e}")
    
    # Option 2: Export to JSON (for web visualization)
    print("\nExporting to JSON...")
    try:
        viz.export_to_json("knowledge_base.json", max_nodes=200)
        print("✓ JSON export complete. Use with D3.js or other web tools.")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Option 3: Matplotlib visualization
    print("\nCreating matplotlib visualization...")
    try:
        viz.plot_networkx(
            max_nodes=50,
            figsize=(14, 10),
            layout='spring',
            show_labels=True
        )
        print("✓ Matplotlib visualization displayed.")
    except ImportError as e:
        print(f"✗ {e}")
        print("  Install with: pip install networkx matplotlib")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Option 4: Interactive Plotly visualization
    print("\nCreating interactive Plotly visualization...")
    try:
        viz.plot_plotly(
            max_nodes=100,
            output_file="graph_interactive.html"
        )
        print("✓ Interactive HTML saved to graph_interactive.html")
    except ImportError as e:
        print(f"✗ {e}")
        print("  Install with: pip install plotly networkx")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    print("\n" + "="*60)
    print("Visualization Options Summary:")
    print("="*60)
    print("1. ArangoDB Web UI: http://localhost:8529")
    print("   - Log in and go to Graphs tab")
    print("   - Use built-in graph viewer")
    print("\n2. GraphML Export: knowledge_base.graphml")
    print("   - Open with Gephi (https://gephi.org/)")
    print("   - Or Cytoscape (https://cytoscape.org/)")
    print("\n3. JSON Export: knowledge_base.json")
    print("   - Use with D3.js force-directed graph")
    print("   - Or other web visualization libraries")
    print("\n4. Python Visualization:")
    print("   - Matplotlib: viz.plot_networkx()")
    print("   - Plotly: viz.plot_plotly()")
    print("="*60)


if __name__ == "__main__":
    main()

