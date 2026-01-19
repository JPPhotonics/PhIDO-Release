# RDF Export Strategy for HAS_PROPERTY Edges

In the ArangoDB Photonic Knowledge Graph, the `HAS_PROPERTY` edge connects a Component to a Property and carries attributes like `value` and `unit`. 

Standard RDF triples are binary (Subject -> Predicate -> Object) and cannot directly represent edge attributes. To preserve this data in the RDF export, we use the **N-ary relation pattern** (specifically, **Reification**).

## The Pattern

Instead of a direct link:
`Component -> hasProperty -> Property`

We introduce an intermediate node (an `Observation`):
1. `Component -> hasFeatureOfInterest -> Observation`
2. `Observation -> observedProperty -> Property`
3. `Observation -> hasSimpleResult -> Value`
4. `Observation -> unit -> Unit`

## Implementation in `graph_manager.py`

The `fetch_subgraph_rdf` function implements this logic:

```python
if edge_type == "HAS_PROPERTY":
    # 1. Create a unique Observation URI
    obs_uri = URIRef(PIC + f"Observation_{edge['_key']}")
    
    # 2. Assign Types
    rdf_graph.add((obs_uri, RDF.type, PIC.ComponentObservation))
    rdf_graph.add((obs_uri, RDF.type, SOSA.Observation))
    
    # 3. Link to Component (Source)
    rdf_graph.add((obs_uri, SOSA.hasFeatureOfInterest, source_uri))
    
    # 4. Link to Property (Target)
    rdf_graph.add((obs_uri, SOSA.observedProperty, target_uri))
    
    # 5. Add Value (if present)
    if "value" in edge:
        rdf_graph.add((obs_uri, SOSA.hasSimpleResult, Literal(edge["value"])))
        rdf_graph.add((obs_uri, QUDT.numericValue, Literal(edge["value"])))

    # 6. Add Unit (if present)
    if "unit" in edge:
         rdf_graph.add((obs_uri, QUDT.unit, Literal(edge["unit"])))
```

This ensures that the rich data stored on the graph edge is fully preserved and semantically accessible in the exported ontology.
