# RDF and Ontology Integration

This module provides tools for integrating the ArangoDB knowledge graph with standard RDF/OWL ontologies.

## Directory Structure

- `ontology/`: Contains the OWL ontology file (`pic_ontology.ttl`).
- `src/`: Contains the Python code for RDF integration and validation.

## Key Components

### 1. OWL Ontology (`ontology/pic_ontology.ttl`)

Defines the semantic schema for Photonic Integrated Circuits (PICs), mapped to standard ontologies:
- **SAREF**: Devices and functions.
- **SOSA/SSN**: Observations and properties.
- **QUDT**: Quantities and units.
- **PROV-O**: Provenance (documents).

### 2. Graph Manager (`src/graph_manager.py`)

Handles the conversion between ArangoDB and RDF:
- **`fetch_subgraph_rdf(component_name)`**: Fetches a component's subgraph and converts it to an RDFLib graph.
- **Reification**: Automatically handles `HAS_PROPERTY` edge attributes (value, unit) by creating `ComponentObservation` nodes (N-ary relation pattern).
- **`validate_subgraph(component_name, ontology_path)`**: Validates the RDF data against the OWL ontology using `pyshacl` (checking for consistency and constraints).
- **`export_to_turtle`**: Exports the RDF graph to a Turtle (.ttl) file.

## Usage

```python
from PhotonicsAI.KnowledgeBase.GenerativeOntology.src.graph_manager import GraphManager

# Initialize
db_config = {
    'url': 'http://localhost:8529',
    'username': 'root',
    'password': 'password',
    'db_name': 'photonics_kb'
}
manager = GraphManager(db_config)

# Export to RDF
manager.export_to_turtle("Mach-Zehnder Interferometer", "output.ttl")

# Validate
report = manager.validate_subgraph("Mach-Zehnder Interferometer", "ontology/pic_ontology.ttl")
if report['conforms']:
    print("Graph is valid!")
else:
    print(report['report_text'])
```

## Dependencies

- `arango-rdf`
- `rdflib`
- `pyshacl`

