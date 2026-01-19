"""Ontology loader utility for PPC Agent (OEMA framework)."""

from __future__ import annotations

from typing import Dict, List, Any, Optional
from rdflib import Graph, RDFS, OWL, RDF, URIRef

class OntologyLoader:
    """Load and expose lightweight schema info from the PIC ontology."""

    def __init__(self, ontology_path: str):
        self.ontology_path = ontology_path
        self.graph = Graph()
        try:
            self.graph.parse(ontology_path, format="turtle")
            print(f"Successfully loaded ontology from {ontology_path}")
        except Exception as e:
            print(f"Error loading ontology from {ontology_path}: {e}")
            # Initialize empty graph to avoid crashes, but warn heavily
            self.graph = Graph()

    def get_ontology_schema(self) -> Dict[str, Any]:
        """
        Return a structured view of classes and constraints for prompt injection
        and lightweight rule checks.

        Structure:
        {
          "classes": {
             "Component": {"uri": "...", "parents": [...], "description": "...", "disjoint_with": [...]},
             ...
          },
          "constraints": {
             "disjoint_pairs": [(A,B), ...],
             "property_domains": {"performsFunction": ["Component", ...]},
             "property_ranges": {"performsFunction": ["DesignFunction", ...]}
          }
        }
        """
        classes = self._extract_classes()
        constraints = self._extract_constraints(classes)
        return {"classes": classes, "constraints": constraints}

    def _extract_classes(self) -> Dict[str, Dict[str, Any]]:
        """Extract classes with parents, descriptions, and disjointness."""
        classes: Dict[str, Dict[str, Any]] = {}

        for cls in self.graph.subjects(RDF.type, OWL.Class):
            if not isinstance(cls, URIRef):
                continue
                
            label = self._label(cls)
            parents = [self._label(p) for p in self.graph.objects(cls, RDFS.subClassOf) if isinstance(p, URIRef)]
            
            comment = ""
            for obj in self.graph.objects(cls, RDFS.comment):
                comment = str(obj)
                break
                
            disjoints = [self._label(d) for d in self.graph.objects(cls, OWL.disjointWith) if isinstance(d, URIRef)]

            classes[label] = {
                "uri": str(cls),
                "parents": [p for p in parents if p],
                "description": str(comment) if comment else "",
                "disjoint_with": [d for d in disjoints if d],
            }

        return classes

    def _extract_constraints(self, classes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Extract simple constraints: disjoint pairs and property domain/range."""
        disjoint_pairs: List[tuple] = []
        for cls, info in classes.items():
            for other in info.get("disjoint_with", []):
                disjoint_pairs.append((cls, other))

        property_domains: Dict[str, List[str]] = {}
        property_ranges: Dict[str, List[str]] = {}

        for prop in self.graph.subjects(RDF.type, OWL.ObjectProperty):
            label = self._label(prop)
            domains = [self._label(d) for d in self.graph.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
            ranges = [self._label(r) for r in self.graph.objects(prop, RDFS.range) if isinstance(r, URIRef)]
            if label:
                if domains:
                    property_domains[label] = [d for d in domains if d]
                if ranges:
                    property_ranges[label] = [r for r in ranges if r]

        # Datatype properties (e.g., numericValue, unit) – capture domains too
        for prop in self.graph.subjects(RDF.type, OWL.DatatypeProperty):
            label = self._label(prop)
            domains = [self._label(d) for d in self.graph.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
            if label and domains:
                property_domains[label] = [d for d in domains if d]

        return {
            "disjoint_pairs": disjoint_pairs,
            "property_domains": property_domains,
            "property_ranges": property_ranges,
        }

    def _label(self, uri) -> str:
        """Derive a readable label from URI (favor fragment)."""
        if uri is None:
            return ""
        uri_str = str(uri)
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        return uri_str.rsplit("/", 1)[-1]
    
    def validate_entity_type(self, entity_type: str) -> bool:
        """Check if an entity type exists in the ontology."""
        classes = self._extract_classes()
        return entity_type in classes or entity_type in ["Component", "Architecture", "Property", "DesignFunction", "PhysicalPrinciple"]

# Convenience function
def get_ontology_schema(ontology_path: str) -> Dict[str, Any]:
    return OntologyLoader(ontology_path).get_ontology_schema()
