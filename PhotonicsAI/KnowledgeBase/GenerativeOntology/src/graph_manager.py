import os
from typing import Dict, Any, List, Optional
from arango import ArangoClient
from arango_rdf import ArangoRDF
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
import pyshacl

# Namespace Definitions
PIC = Namespace("http://www.photonics.ai/ontology/pic#")
SAREF = Namespace("https://saref.etsi.org/core/")
SOSA = Namespace("http://www.w3.org/ns/sosa/")
QUDT = Namespace("http://qudt.org/schema/qudt/")
PROV = Namespace("http://www.w3.org/ns/prov#")
BFO = Namespace("http://purl.obolibrary.org/obo/BFO_")

class GraphManager:
    def __init__(self, db_config: Dict[str, str]):
        """
        Initialize the GraphManager.
        
        Args:
            db_config (Dict[str, str]): Database configuration containing 'url', 'username', 'password', 'db_name'.
        """
        self.client = ArangoClient(hosts=db_config['url'])
        self.db = self.client.db(
            db_config['db_name'],
            username=db_config['username'],
            password=db_config['password']
        )
        self.arango_rdf = ArangoRDF(self.db)
        
        # Define Concept Map for arango-rdf
        self.concept_map = {
            "Components": str(PIC.Component),
            "Architectures": str(PIC.Architecture),
            "Design_Functions": str(PIC.DesignFunction),
            "Physical_Principles": str(PIC.PhysicalPrinciple),
            "Properties": str(PIC.Property),
            "Documents": str(PIC.Document),
            "PERFORMS_FUNCTION": str(PIC.performsFunction),
            "BASED_ON_PRINCIPLE": str(PIC.basedOnPrinciple),
            "USES_COMPONENT": str(PIC.usesComponent),
            "EXTRACTED_FROM": str(PIC.extractedFrom),
            "RELATED_TO": str(PIC.relatedTo)
        }

    def fetch_subgraph_rdf(self, component_name: str) -> Graph:
        """
        Fetch a subgraph for a component and convert it to an RDFLib Graph.
        Handles reification of HAS_PROPERTY edges manually.
        
        Args:
            component_name (str): The name of the component (e.g., 'Mach-Zehnder Interferometer').
            
        Returns:
            rdflib.Graph: The constructed RDF graph.
        """
        rdf_graph = Graph()
        
        # Bind Namespaces
        rdf_graph.bind("pic", PIC)
        rdf_graph.bind("saref", SAREF)
        rdf_graph.bind("sosa", SOSA)
        rdf_graph.bind("qudt", QUDT)
        rdf_graph.bind("prov", PROV)
        
        # AQL to fetch component and its immediate neighbors
        # We need to search in both Components and Architectures
        aql = """
        LET start_node = (
            FOR doc IN Components FILTER doc.name == @name RETURN doc
        )[0] || (
            FOR doc IN Architectures FILTER doc.name == @name RETURN doc
        )[0]
        
        FILTER start_node != null
        
        FOR v, e, p IN 1..1 OUTBOUND start_node GRAPH 'KnowledgeGraph'
            RETURN {
                "source": start_node,
                "edge": e,
                "target": v
            }
        """
        
        cursor = self.db.aql.execute(aql, bind_vars={"name": component_name})
        
        for record in cursor:
            source = record["source"]
            edge = record["edge"]
            target = record["target"]
            
            # Create Source Node URI
            source_uri = URIRef(PIC + self._sanitize(source["_key"]))
            source_type_uri = URIRef(self.concept_map.get(source["_id"].split("/")[0], str(PIC.Component)))
            rdf_graph.add((source_uri, RDF.type, source_type_uri))
            
            # Create Target Node URI
            target_uri = URIRef(PIC + self._sanitize(target["_key"]))
            target_collection = target["_id"].split("/")[0]
            target_type_uri = URIRef(self.concept_map.get(target_collection, str(PIC.Component)))
            rdf_graph.add((target_uri, RDF.type, target_type_uri))

            # Handle Edges
            edge_type = edge["_id"].split("/")[0] # e.g., PERFORMS_FUNCTION/123 -> PERFORMS_FUNCTION
            
            if edge_type == "HAS_PROPERTY":
                # Reification Pattern for HAS_PROPERTY
                # Create Observation Node
                obs_uri = URIRef(PIC + f"Observation_{edge['_key']}")
                rdf_graph.add((obs_uri, RDF.type, PIC.ComponentObservation))
                rdf_graph.add((obs_uri, RDF.type, SOSA.Observation))
                
                # Link Observation to Component (Feature of Interest)
                rdf_graph.add((obs_uri, SOSA.hasFeatureOfInterest, source_uri))
                
                # Link Observation to Property (Observed Property)
                rdf_graph.add((obs_uri, SOSA.observedProperty, target_uri))
                
                # Add attributes (Value and Unit)
                if "value" in edge:
                    # Try to convert to float, otherwise string
                    try:
                        val = float(edge["value"])
                        rdf_graph.add((obs_uri, SOSA.hasSimpleResult, Literal(val, datatype=XSD.double)))
                        rdf_graph.add((obs_uri, QUDT.numericValue, Literal(val, datatype=XSD.double)))
                    except (ValueError, TypeError):
                         rdf_graph.add((obs_uri, SOSA.hasSimpleResult, Literal(edge["value"])))

                if "unit" in edge:
                     rdf_graph.add((obs_uri, QUDT.unit, Literal(edge["unit"])))
            
            elif edge_type in self.concept_map:
                # Standard Object Property
                pred_uri = URIRef(self.concept_map[edge_type])
                rdf_graph.add((source_uri, pred_uri, target_uri))
                
        return rdf_graph

    def validate_subgraph(self, component_name: str, ontology_path: str):
        """
        Validate the subgraph of a component against the OWL ontology.
        
        Args:
            component_name (str): Name of the component.
            ontology_path (str): Path to the .ttl ontology file.
            
        Returns:
            Dict: Validation report (conforms, results_text, results_graph).
        """
        # 1. Fetch Data Graph
        data_graph = self.fetch_subgraph_rdf(component_name)
        
        # 2. Load Ontology Graph
        ontology_graph = Graph()
        ontology_graph.parse(ontology_path, format="turtle")
        
        # 3. Combine Graphs for Validation (Inference is limited in pyshacl, mostly structure)
        # For deeper OWL DL reasoning, we would use owlrl here to expand the graph first
        
        # 4. Run Validation (using pyshacl as a lightweight checker or simple constraint check)
        # Note: OWL 2 DL validation is complex. Here we check if the RDF structure violates
        # any obvious constraints definable in SHACL or basic RDFS logic if inferred.
        # Since pyshacl validates against shapes, and we have an OWL ontology, 
        # we can use the ontology as the SHACL graph if it contains SHACL shapes, 
        # OR we can just check RDFS consistency if we used an RDFS reasoner.
        
        # For this implementation, we will assume the ontology file might contain some
        # implicit constraints we want to check.
        
        # Since standard OWL isn't directly validated by pyshacl without converting to SHACL,
        # We will return the graph serialization for manual inspection/external validation
        # unless we add SHACL shapes to the ontology.
        
        # However, we can use pyshacl's 'advanced' mode to try and validate against RDFS/OWL semantics
        # if the ontology is provided as the shacl_graph (it attempts auto-conversion).
        
        conforms, v_graph, v_text = pyshacl.validate(
            data_graph,
            shacl_graph=ontology_graph,
            ont_graph=ontology_graph,
            inference='rdfs',
            abort_on_first=False,
            meta_shacl=False,
            debug=False
        )
        
        return {
            "conforms": conforms,
            "report_text": v_text,
            "graph_size": len(data_graph)
        }

    def _sanitize(self, key: str) -> str:
        """Sanitize ArangoDB key for URI usage."""
        return key.replace(" ", "_").replace("-", "_")

    def export_to_turtle(self, component_name: str, output_file: str):
        """Export subgraph to Turtle file."""
        g = self.fetch_subgraph_rdf(component_name)
        g.serialize(destination=output_file, format="turtle")
        print(f"Exported subgraph for '{component_name}' to {output_file}")

