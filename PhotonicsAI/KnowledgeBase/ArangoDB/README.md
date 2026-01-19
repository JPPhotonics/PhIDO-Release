# ArangoDB Knowledge Base

A graph database implementation for the photonic ontology with vector embeddings support for semantic search and retrieval.

## Overview

This module converts YAML ontology files into an ArangoDB graph database, enabling:
- **Graph-based retrieval**: Navigate relationships between components, properties, principles, and functions
- **Vector semantic search**: Find similar entities using embeddings
- **Hybrid search**: Combine semantic similarity with graph structure
- **Incremental updates**: Add new documents and entities as the ontology grows

## Installation

### Prerequisites

1. **ArangoDB**: Install ArangoDB locally or use Docker

   ```bash
   # Using Docker (recommended)
   # Note: ARANGO_ROOT_PASSWORD is the password for ArangoDB's root database user,
   # NOT your Linux system user. Set it to any secure password you choose.
   docker run -d --name arangodb -p 8529:8529 -e ARANGO_ROOT_PASSWORD=my_secure_password arangodb:latest
   
   # Or install locally
   # See: https://www.arangodb.com/download/
   ```

2. **Python Dependencies**

   ```bash
   pip install python-arango sentence-transformers numpy pyyaml
   ```

### Environment Variables

Create a `.env` file or set environment variables:

```bash
ARANGO_HOST=localhost
ARANGO_PORT=8529
ARANGO_USERNAME=root
# ARANGO_PASSWORD should match the ARANGO_ROOT_PASSWORD you set when starting ArangoDB
# This is the ArangoDB database user password, not your Linux system user password
ARANGO_PASSWORD=my_secure_password
ARANGO_DATABASE=photonics_kb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

**Important Note**: The `ARANGO_ROOT_PASSWORD` and `ARANGO_PASSWORD` refer to ArangoDB's database user credentials, not your Linux system user. The ArangoDB `root` user is the default administrative user in ArangoDB, which is completely separate from your Linux username. You can set these to any secure password you choose.

## Quick Start

### 0. Run Tests

A comprehensive test script is available to verify all functionality:

```bash
python arangoDB_test.py
```

This script tests:
- Direct neighbor retrieval
- Vector-based semantic search (with both short and long queries)
- Relationship traversal
- Hybrid search
- Convenience query methods
- Interactive Plotly visualization

### 1. Initialize Database

```python
from PhotonicsAI.KnowledgeBase.ArangoDB import KnowledgeBaseClient, ArangoDBConfig

# Create client
client = KnowledgeBaseClient()

# Initialize database schema
client.initialize()

# Import YAML data
yaml_dir = "PhotonicsAI/KnowledgeBase/GenerativeOntology/Primitives"
client.import_yaml_data(yaml_dir)
```

### 2. Basic Retrieval

```python
# Direct neighbor retrieval (automatically finds vertex by name)
neighbors = client.get_neighbors("Mach–Zehnder Interferometer", "Architectures")

# Get specific edge types only
functions = client.get_neighbors("Grating Coupler", "Components", 
                                 edge_types=["PERFORMS_FUNCTION"])
properties = client.get_neighbors("Ring Resonator", "Components",
                                  edge_types=["HAS_PROPERTY"])

# Semantic search with longer descriptive queries (works better than keywords)
results = client.semantic_search(
    "optical modulation devices that control light amplitude",
    "Components",
    limit=5,
    threshold=0.2  # Lower threshold for better recall
)

# Short keyword queries also work
results = client.semantic_search("modulator", "Components", limit=5, threshold=0.2)

# Relationship traversal
path = ["HAS_PROPERTY"]
related = client.traverse("Ring Resonator", "Components", path, max_depth=1)

# Hybrid search (combines vector search with graph filters)
filters = {"must_have_edge": "HAS_PROPERTY", "edge_target": "Insertion_Loss"}
results = client.hybrid_search("modulator", "Components", relationship_filters=filters)
```

### 3. Adding New Documents

```python
# Add a new paper/document
doc_key = client.add_document(
    title="Advanced Photonic Modulators",
    source="Journal of Photonics, 2024",
    doc_type="paper",
    metadata={"authors": ["Smith", "Jones"], "year": 2024}
)

# Add architecture extracted from paper
arch_key = client.add_architecture(
    name="Novel Ring Modulator",
    description="A new ring modulator design with improved efficiency",
    source="Journal of Photonics, 2024",
    relationships={
        "performs_function": ["Modulator", "Amplitude_Modulator"],
        "based_on_principle": ["Resonance", "Plasma_Dispersion_Effect"],
        "has_property": ["Insertion_Loss", "Bandwidth", "Q-factor"],
        "uses_component": ["Ring_Resonator", "Directional_Coupler"]
    },
    document_key=doc_key
)
```

## Graph Visualization

There are several ways to visualize the graph database:

### 1. ArangoDB Web Interface (Built-in)

The easiest way is to use ArangoDB's built-in web interface:

1. Start ArangoDB (if using Docker, it's already running)
2. Open your browser and go to: `http://localhost:8529`
3. Log in with your ArangoDB credentials (username: `root`, password: your `ARANGO_PASSWORD`)
4. Select your database (`photonics_kb`)
5. Go to the **Graphs** tab to visualize the graph structure
6. Use the **Graph Viewer** to explore relationships interactively

### 2. Python Visualization (NetworkX/Matplotlib)

Export and visualize using Python:

```python
from PhotonicsAI.KnowledgeBase.ArangoDB import KnowledgeBaseClient

client = KnowledgeBaseClient()
client.connect()

# Get visualizer
viz = client.get_visualizer()

# Print statistics
viz.print_statistics()

# Create matplotlib visualization (requires: pip install networkx matplotlib)
viz.plot_networkx(max_nodes=100, layout='spring', show_labels=True)

# Export to GraphML (for Gephi, Cytoscape, etc.)
viz.export_to_graphml("knowledge_base.graphml", max_nodes=200)

# Export to JSON (for web visualization)
viz.export_to_json("knowledge_base.json", max_nodes=200)
```

### 3. Interactive Plotly Visualization

Create an interactive HTML visualization:

```python
# Requires: pip install plotly networkx
viz = client.get_visualizer()
viz.plot_plotly(max_nodes=100, output_file="graph.html")
# Opens in browser automatically
```

### 4. External Tools

Export the graph and use external visualization tools:

- **Gephi**: Import GraphML file for advanced network analysis
- **Cytoscape**: Import GraphML for biological-style network visualization
- **D3.js**: Use exported JSON with D3.js force-directed graphs
- **yEd**: Import GraphML for hierarchical layouts

### Installation for Visualization

```bash
# For NetworkX/Matplotlib visualization
pip install networkx matplotlib

# For Plotly interactive visualization
pip install plotly networkx

# For GraphML export (NetworkX includes this)
pip install networkx
```

## Database Schema

### Vertex Collections

- **Components**: Basic photonic components (Directional_Coupler, Y_Branch, etc.)
- **Architectures**: Composite structures (MZI, Micro-ring Modulator, etc.)
- **Properties**: Physical properties (Insertion_Loss, Bandwidth, etc.)
- **Design_Functions**: Functional capabilities (Modulator, Filter, etc.)
- **Physical_Principles**: Underlying principles (Plasma_Dispersion_Effect, etc.)
- **Documents**: Source documents/papers (for provenance tracking)

### Edge Collections

- **PERFORMS_FUNCTION**: Component/Architecture → Design_Function
- **BASED_ON_PRINCIPLE**: Component/Architecture → Physical_Principle
- **HAS_PROPERTY**: Component/Architecture → Property
- **USES_COMPONENT**: Architecture → Component (Architecture uses Component)
- **RELATED_TO**: Property → Property (for related properties)
- **EXTRACTED_FROM**: Any entity → Document (provenance tracking)

## API Reference

### KnowledgeBaseClient

Main client class for all operations.

#### Initialization

```python
client = KnowledgeBaseClient(config=ArangoDBConfig())
client.connect()  # Connect to database
client.initialize(reset=False)  # Initialize schema
```

#### Retrieval Methods

- `get_neighbors(vertex_name, collection, edge_types=None, max_depth=1)`: Get direct neighbors of a vertex by name. Automatically resolves vertex by name before querying edges.
- `semantic_search(query_text, collection, limit=10, threshold=0.3)`: Vector semantic search using sentence transformers. Supports both short keywords and longer descriptive phrases. Default threshold is 0.3 (lowered from 0.7 for better recall).
- `traverse(start_name, start_collection, relationship_path, max_depth=3)`: Graph traversal following a specific relationship path pattern.
- `hybrid_search(query_text, collection, relationship_filters=None, limit=10)`: Hybrid search combining vector similarity with graph structure filters.

#### Update Methods

- `add_document(title, source, doc_type="paper", metadata=None)`: Add new document
- `add_architecture(name, description, source, relationships=None, document_key=None)`: Add architecture
- `add_component(name, description, source, relationships=None, document_key=None)`: Add component
- `add_entity(entity_data, collection_name, document_key=None)`: Add generic entity

#### Query Methods

- `find_by_name(name, collection)`: Find vertex by name (searches both by sanitized key and name field)
- `find_components_by_principle(principle_name)`: Find all components/architectures that use a specific principle
- `find_architectures_using_component(component_name)`: Find all architectures that use a specific component
- `get_visualizer()`: Get a GraphVisualizer instance for visualization and statistics

## Usage Examples

### Example 1: Semantic Search with Descriptive Queries

```python
# Longer descriptive queries work better for semantic search
modulators = client.semantic_search(
    "components that perform phase shifting and modulation",
    "Components",
    limit=10,
    threshold=0.2
)

# Short queries also work
modulators = client.semantic_search("modulator", "Components", limit=10, threshold=0.2)

# Filter results by properties
for mod in modulators:
    neighbors = client.get_neighbors(mod["name"], "Components", 
                                    edge_types=["HAS_PROPERTY"])
    properties = [n["name"] for n in neighbors]
    print(f"{mod['name']}: {properties}")
```

### Example 2: Find Components Using a Specific Principle

```python
# Find all components based on Plasma_Dispersion_Effect
components = client.find_components_by_principle("Plasma_Dispersion_Effect")

for comp in components:
    print(f"{comp['name']} - {comp.get('description', '')[:100]}")
```

### Example 3: Traverse Component Hierarchy

```python
# Start from an architecture and traverse to its components
architectures = client.semantic_search("MZI", "Architectures", limit=1)
if architectures:
    mzi = architectures[0]
    
    # Get components used by MZI
    components = client.get_neighbors(mzi["name"], "Architectures",
                                    edge_types=["USES_COMPONENT"])
    
    # Get properties of MZI
    properties = client.get_neighbors(mzi["name"], "Architectures",
                                     edge_types=["HAS_PROPERTY"])
    
    print(f"MZI uses: {[c['name'] for c in components]}")
    print(f"MZI has properties: {[p['name'] for p in properties]}")
```

### Example 4: Incremental Update from Paper

```python
# Add a new paper
paper_key = client.add_document(
    title="Novel Grating Coupler Design",
    source="Photonics Research, 2024",
    metadata={"doi": "10.1234/photonics.2024.001"}
)

# Extract new architecture from paper
new_arch = client.add_architecture(
    name="Optimized Grating Coupler",
    description="A grating coupler with improved coupling efficiency...",
    source="Photonics Research, 2024",
    relationships={
        "performs_function": ["Optical_Input_Output_Coupler"],
        "based_on_principle": ["Bragg_Reflection", "Mode_Overlap_Maximization"],
        "has_property": ["Insertion_Loss", "Directionality", "Bandwidth"],
        "uses_component": []  # No sub-components
    },
    document_key=paper_key
)
```

## Troubleshooting

### Docker Permission Issues

If you get a "permission denied" error when running Docker commands:

```bash
# Option 1: Add your user to the docker group (recommended)
sudo usermod -aG docker $USER
# Then log out and log back in, or run:
newgrp docker

# Option 2: Use sudo (not recommended for regular use)
sudo docker run -d --name arangodb -p 8529:8529 -e ARANGO_ROOT_PASSWORD=my_secure_password arangodb:latest

# Option 3: Check if Docker daemon is running
sudo systemctl status docker
# If not running, start it:
sudo systemctl start docker
```

### Connection Issues

If you can't connect to ArangoDB:

1. Check if ArangoDB is running: `docker ps` or check service status
2. Verify connection settings in environment variables
3. Test connection: `curl http://localhost:8529/_api/version`

### Vector Search Not Working

Vector search uses Python-based cosine similarity calculation (works with any ArangoDB version):
- Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2` by default
- Embeddings are stored in the `embedding` field of each vertex document
- Default threshold is 0.3 (lowered from 0.7 for better recall)
- For better results, use longer descriptive queries rather than single keywords
- Debug output shows: total docs, docs with embeddings, similarity ranges

**Tips for better vector search results:**
- Use descriptive phrases: "optical modulation devices" instead of just "modulator"
- Lower threshold (0.2-0.3) for broader results, higher (0.5-0.7) for precision
- Check debug output to see if embeddings exist and similarity ranges

### Import Errors

If YAML import fails:
- Check YAML file syntax
- Verify file paths are correct
- The system includes a manual parser for non-standard YAML formats
- Debug output shows parsed relationships during import
- Check that relationships are being extracted (look for "Debug: Parsed" messages)

### Retrieval Not Returning Results

If retrieval functions return empty results:
- **Neighbor retrieval**: Verify edges exist using `viz.print_statistics()` or ArangoDB web UI
- **Vector search**: Check debug output to see if embeddings exist and similarity scores
- **Name resolution**: The system automatically resolves vertex names, handling underscores/spaces and case differences
- **Edge queries**: Uses collection API directly (no AQL parsing issues)
- Check that vertex names match exactly (including special characters like en-dashes)

## Performance Tips

1. **Batch Operations**: Import data in batches for better performance
2. **Index Usage**: Ensure proper indexes are created (done automatically on `_from` and `_to` fields)
3. **Embedding Caching**: Embeddings are generated on import, not on query
4. **Query Optimization**: Use specific edge types in `get_neighbors` for faster queries
5. **Vector Search**: Lower thresholds (0.2-0.3) for broader results, but may be slower on large collections
6. **Name Resolution**: Vertex lookup by name is cached during import, but may require database queries for new lookups

## Implementation Details

### Retrieval Architecture

- **Edge Queries**: Uses python-arango collection API (`collection.find()`) instead of AQL for reliability
- **Vertex Lookup**: Automatically resolves vertex names by searching both `_key` and `name` fields
- **Vector Search**: Python-based cosine similarity using NumPy (no ArangoDB vector indexes required)
- **Error Handling**: Comprehensive error handling with debug output for troubleshooting

### Data Import

- **YAML Parsing**: Handles both standard YAML and non-standard formats with manual parser fallback
- **Relationship Extraction**: Automatically extracts relationships from YAML structure
- **Key Sanitization**: Handles special characters (Greek letters, en-dashes, etc.) in entity names
- **Embedding Generation**: Creates embeddings for all entities during import using sentence transformers

## License

Part of the PhIDO project.

