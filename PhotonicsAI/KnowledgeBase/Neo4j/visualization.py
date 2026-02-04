"""Neo4j Graph Visualization using PyVis."""

import networkx as nx
from pyvis.network import Network
from .client import Neo4jClient

class Neo4jVisualizer:
    """Visualizes the Neo4j Knowledge Graph using PyVis."""

    def __init__(self, client: Neo4jClient):
        self.client = client

    def visualize_graph(self, output_file: str = "neo4j_graph.html"):
        """
        Fetch the graph from Neo4j and render it to an HTML file.
        
        Args:
            output_file: Path to save the HTML file.
        """
        if not self.client._connected:
            self.client.connect()

        # Initialize network
        net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, filter_menu=True)
        
        # Collection colors (matching ArangoDB viz style)
        colors = {
            "Component": "#ff9900",
            "Architecture": "#00ccff", 
            "Property": "#cc00ff",
            "Design_Function": "#00ff99",
            "Physical_Principle": "#ff0066",
            "Document": "#aaaaaa"
        }

        # Fetch ALL Nodes (including isolated ones)
        node_query = f"""
        MATCH (n)
        RETURN n
        """

        # Fetch ALL Relationships
        rel_query = f"""
        MATCH (n)-[r]->(m)
        RETURN n, r, m
        """

        with self.client.driver.session() as session:
            # 1. Add ALL nodes first
            result_nodes = session.run(node_query)
            
            visited_nodes = set()
            
            for record in result_nodes:
                n = record["n"]
                n_id = n.element_id if hasattr(n, 'element_id') else n.id
                
                # Check if we've already added this node (by name or ID)
                # But here we are iterating unique nodes from DB ideally
                if n_id in visited_nodes:
                    continue
                
                n_label = list(n.labels)[0] if n.labels else "Unknown"
                n_title = f"{n.get('name', 'N/A')}\n({n_label})"
                if n.get('description'):
                    n_title += f"\n\n{n.get('description', '')[:200]}..."
                
                # Use Name as ID for PyVis if available, else element_id
                # This ensures the dropdown works
                viz_id = n.get('name', n_id) 
                
                net.add_node(
                    viz_id, 
                    label=n.get('name', 'N/A'), 
                    title=n_title, 
                    color=colors.get(n_label, "#999999"), 
                    group=n_label,
                    shape="dot"
                )
                visited_nodes.add(n_id)

            # 2. Add edges
            result_rels = session.run(rel_query)
            count = 0
            
            for record in result_rels:
                n = record["n"]
                m = record["m"]
                r = record["r"]
                
                n_viz_id = n.get('name', n.element_id if hasattr(n, 'element_id') else n.id)
                m_viz_id = m.get('name', m.element_id if hasattr(m, 'element_id') else m.id)
                
                # Edges might reference nodes we haven't fetched if LIMIT cut off the node query 
                # but caught the edge query. Safe to add edges if nodes exist in viz.
                # PyVis creates nodes if they don't exist, but we want them styled.
                # Since we ran a node query with limit, we might miss some endpoints if limit is tight.
                # For robustness, we should ensure endpoints exist or add them on the fly.
                
                # Check existence in network
                try:
                    net.get_node(n_viz_id)
                except:
                    # Add N if missing (styling it)
                    n_label = list(n.labels)[0] if n.labels else "Unknown"
                    net.add_node(n_viz_id, label=n.get('name'), color=colors.get(n_label, "#999999"), group=n_label, shape="dot")
                    
                try:
                    net.get_node(m_viz_id)
                except:
                    # Add M if missing
                    m_label = list(m.labels)[0] if m.labels else "Unknown"
                    net.add_node(m_viz_id, label=m.get('name'), color=colors.get(m_label, "#999999"), group=m_label, shape="dot")

                # Add edge
                edge_title = r.type
                if r.get("description"):
                    edge_title += f"\n{r.get('description')}"
                net.add_edge(
                    n_viz_id, 
                    m_viz_id, 
                    title=edge_title, 
                    label=r.type,
                    arrows="to"
                )
                count += 1

        print(f"Visualized {count} relationships.")
        
        # Physics options
        net.force_atlas_2based()
        net.show_buttons(filter_=['physics'])
        net.save_graph(output_file)

