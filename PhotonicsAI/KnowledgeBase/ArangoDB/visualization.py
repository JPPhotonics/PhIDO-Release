"""Graph visualization utilities for ArangoDB knowledge base."""

import json
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from arango.database import StandardDatabase

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .config import ArangoDBConfig


class GraphVisualizer:
    """Visualize ArangoDB graph database."""
    
    def __init__(self, db: StandardDatabase, config: ArangoDBConfig):
        """Initialize visualizer with database connection."""
        self.db = db
        self.config = config
        self.vertex_collections = [
            "Components", "Architectures", "Properties",
            "Design_Functions", "Physical_Principles", "Documents"
        ]
        self.edge_collections = [
            "PERFORMS_FUNCTION", "BASED_ON_PRINCIPLE", "HAS_PROPERTY",
            "USES_COMPONENT", "RELATED_TO", "EXTRACTED_FROM"
        ]
    
    def to_networkx(self, max_nodes: Optional[int] = None,
                    collections: Optional[List[str]] = None,
                    edge_types: Optional[List[str]] = None) -> 'nx.Graph':
        """
        Convert ArangoDB graph to NetworkX graph.
        
        Args:
            max_nodes: Maximum number of nodes to include (None = all)
            collections: Vertex collections to include (None = all)
            edge_types: Edge collections to include (None = all)
        
        Returns:
            NetworkX graph object
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required. Install with: pip install networkx")
        
        G = nx.MultiDiGraph()
        
        # Determine which collections to use
        collections = collections or self.vertex_collections
        edge_types = edge_types or self.edge_collections
        
        # Add nodes
        node_count = 0
        for coll_name in collections:
            if not self.db.has_collection(coll_name):
                continue
            
            coll = self.db.collection(coll_name)
            for doc in coll:
                if max_nodes and node_count >= max_nodes:
                    break
                
                node_id = doc["_id"]
                # Extract fields, excluding system fields and embedding
                node_attrs = {
                    k: v for k, v in doc.items() 
                    if k not in ["_id", "_key", "_rev", "embedding"]
                }
                # Ensure name, collection, and description are set (may override existing)
                node_attrs["name"] = doc.get("name", "")
                node_attrs["collection"] = coll_name
                node_attrs["description"] = doc.get("description", "")[:100]  # Truncate
                
                G.add_node(node_id, **node_attrs)
                node_count += 1
            
            if max_nodes and node_count >= max_nodes:
                break
        
        # Add edges
        for edge_type in edge_types:
            if not self.db.has_collection(edge_type):
                continue
            
            edge_coll = self.db.collection(edge_type)
            for edge in edge_coll:
                from_id = edge["_from"]
                to_id = edge["_to"]
                
                # Only add edges if both nodes are in the graph
                if from_id in G and to_id in G:
                    G.add_edge(
                        from_id,
                        to_id,
                        edge_type=edge_type,
                        **{k: v for k, v in edge.items() if k not in ["_id", "_key", "_rev", "_from", "_to"]}
                    )
        
        return G
    
    def export_to_graphml(self, output_path: str, max_nodes: Optional[int] = None,
                          collections: Optional[List[str]] = None,
                          edge_types: Optional[List[str]] = None):
        """
        Export graph to GraphML format (can be opened in Gephi, Cytoscape, etc.).
        
        Args:
            output_path: Path to save GraphML file
            max_nodes: Maximum number of nodes to export
            collections: Vertex collections to include
            edge_types: Edge collections to include
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required. Install with: pip install networkx")
        
        G = self.to_networkx(max_nodes, collections, edge_types)
        nx.write_graphml(G, output_path)
        print(f"Exported graph to {output_path}")
    
    def export_to_json(self, output_path: str, max_nodes: Optional[int] = None,
                      collections: Optional[List[str]] = None,
                      edge_types: Optional[List[str]] = None):
        """
        Export graph to JSON format for web visualization.
        
        Args:
            output_path: Path to save JSON file
            max_nodes: Maximum number of nodes to export
            collections: Vertex collections to include
            edge_types: Edge collections to include
        """
        nodes = []
        edges = []
        
        # Collect nodes
        node_count = 0
        node_ids = set()
        
        collections = collections or self.vertex_collections
        for coll_name in collections:
            if not self.db.has_collection(coll_name):
                continue
            
            coll = self.db.collection(coll_name)
            for doc in coll:
                if max_nodes and node_count >= max_nodes:
                    break
                
                node_id = doc["_id"]
                node_ids.add(node_id)
                nodes.append({
                    "id": node_id,
                    "name": doc.get("name", ""),
                    "collection": coll_name,
                    "description": doc.get("description", "")[:200],
                    "type": doc.get("type", ""),
                })
                node_count += 1
            
            if max_nodes and node_count >= max_nodes:
                break
        
        # Collect edges
        edge_types = edge_types or self.edge_collections
        for edge_type in edge_types:
            if not self.db.has_collection(edge_type):
                continue
            
            edge_coll = self.db.collection(edge_type)
            for edge in edge_coll:
                from_id = edge["_from"]
                to_id = edge["_to"]
                
                if from_id in node_ids and to_id in node_ids:
                    edges.append({
                        "source": from_id,
                        "target": to_id,
                        "type": edge_type,
                    })
        
        graph_data = {
            "nodes": nodes,
            "edges": edges
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Exported {len(nodes)} nodes and {len(edges)} edges to {output_path}")
    
    def plot_networkx(self, max_nodes: int = 100, figsize: tuple = (12, 8),
                     collections: Optional[List[str]] = None,
                     edge_types: Optional[List[str]] = None,
                     layout: str = "spring",
                     show_labels: bool = True):
        """
        Create a matplotlib visualization of the graph.
        
        Args:
            max_nodes: Maximum number of nodes to visualize
            figsize: Figure size (width, height)
            collections: Vertex collections to include
            edge_types: Edge collections to include
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai', 'random')
            show_labels: Whether to show node labels
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required. Install with: pip install networkx")
        if not HAS_MATPLOTLIB:
            raise ImportError("Matplotlib is required. Install with: pip install matplotlib")
        
        G = self.to_networkx(max_nodes, collections, edge_types)
        
        if len(G.nodes()) == 0:
            print("No nodes to visualize")
            return
        
        # Choose layout
        if layout == "spring":
            pos = nx.spring_layout(G, k=1, iterations=50)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "kamada_kawai":
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                pos = nx.spring_layout(G)
        elif layout == "random":
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Color nodes by collection
        collection_colors = {
            "Components": "lightblue",
            "Architectures": "lightgreen",
            "Properties": "lightyellow",
            "Design_Functions": "lightcoral",
            "Physical_Principles": "lightpink",
            "Documents": "lightgray",
        }
        
        node_colors = []
        for node_id in G.nodes():
            collection = G.nodes[node_id].get("collection", "Unknown")
            node_colors.append(collection_colors.get(collection, "gray"))
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, arrowsize=10, edge_color='gray')
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
        
        # Draw labels
        if show_labels:
            labels = {node_id: G.nodes[node_id].get("name", node_id.split("/")[-1]) 
                     for node_id in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Create legend
        legend_elements = [
            mpatches.Patch(color=color, label=name)
            for name, color in collection_colors.items()
            if any(G.nodes[n].get("collection") == name for n in G.nodes())
        ]
        plt.legend(handles=legend_elements, loc='upper left')
        
        plt.title(f"Knowledge Base Graph ({len(G.nodes())} nodes, {len(G.edges())} edges)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def plot_plotly(self, max_nodes: int = 100,
                   collections: Optional[List[str]] = None,
                   edge_types: Optional[List[str]] = None,
                   output_file: Optional[str] = None):
        """
        Create an interactive Plotly visualization.
        
        Args:
            max_nodes: Maximum number of nodes to visualize
            collections: Vertex collections to include
            edge_types: Edge collections to include
            output_file: Optional path to save HTML file
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX is required. Install with: pip install networkx")
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required. Install with: pip install plotly")
        
        G = self.to_networkx(max_nodes, collections, edge_types)
        
        if len(G.nodes()) == 0:
            print("No nodes to visualize")
            return
        
        # Use spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare edge traces
        edge_traces = []
        for edge_type in (edge_types or self.edge_collections):
            edge_x = []
            edge_y = []
            for edge in G.edges(data=True):
                if edge[2].get("edge_type") == edge_type:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            if edge_x:
                edge_traces.append(go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines',
                    name=edge_type
                ))
        
        # Prepare node traces by collection
        collection_colors = {
            "Components": "lightblue",
            "Architectures": "lightgreen",
            "Properties": "lightyellow",
            "Design_Functions": "lightcoral",
            "Physical_Principles": "lightpink",
            "Documents": "lightgray",
        }
        
        node_traces = []
        for collection in (collections or self.vertex_collections):
            node_x = []
            node_y = []
            node_text = []
            node_info = []
            
            for node_id in G.nodes():
                if G.nodes[node_id].get("collection") == collection:
                    x, y = pos[node_id]
                    node_x.append(x)
                    node_y.append(y)
                    name = G.nodes[node_id].get("name", node_id.split("/")[-1])
                    node_text.append(name)
                    desc = G.nodes[node_id].get("description", "")[:100]
                    node_info.append(f"{name}<br>{desc}")
            
            if node_x:
                node_traces.append(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    name=collection,
                    text=node_text,
                    textposition="middle center",
                    hovertext=node_info,
                    hoverinfo='text',
                    marker=dict(
                        size=10,
                        color=collection_colors.get(collection, "gray"),
                        line=dict(width=2, color='white')
                    )
                ))
        
        # Create figure
        fig = go.Figure(
            data=edge_traces + node_traces,
            layout=go.Layout(
                title=dict(text='Knowledge Base Graph', font=dict(size=16)),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[dict(
                    text=f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(color="black", size=12)
                )],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        if output_file:
            fig.write_html(output_file)
            print(f"Saved interactive visualization to {output_file}")
        else:
            fig.show()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        stats = {
            "vertices": {},
            "edges": {},
            "total_vertices": 0,
            "total_edges": 0
        }
        
        # Count vertices by collection
        for coll_name in self.vertex_collections:
            if self.db.has_collection(coll_name):
                coll = self.db.collection(coll_name)
                count = coll.count()
                stats["vertices"][coll_name] = count
                stats["total_vertices"] += count
        
        # Count edges by type
        for edge_type in self.edge_collections:
            if self.db.has_collection(edge_type):
                edge_coll = self.db.collection(edge_type)
                count = edge_coll.count()
                stats["edges"][edge_type] = count
                stats["total_edges"] += count
        
        return stats
    
    def print_statistics(self):
        """Print graph statistics to console."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("Graph Database Statistics")
        print("="*60)
        print(f"\nTotal Vertices: {stats['total_vertices']}")
        print("\nVertices by Collection:")
        for coll, count in stats["vertices"].items():
            print(f"  {coll}: {count}")
        
        print(f"\nTotal Edges: {stats['total_edges']}")
        print("\nEdges by Type:")
        for edge_type, count in stats["edges"].items():
            print(f"  {edge_type}: {count}")
        print("="*60 + "\n")

