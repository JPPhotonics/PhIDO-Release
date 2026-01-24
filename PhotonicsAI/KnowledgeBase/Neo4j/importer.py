"""Import YAML data into Neo4j with vector embeddings."""

import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase, Driver

from .config import Neo4jConfig
from ..ArangoDB.yaml_loader import YAMLLoader # Reuse the existing YAML loader

class Neo4jImporter:
    """Import ontology data into Neo4j with vector embeddings."""
    
    def __init__(self, config: Neo4jConfig, driver: Driver):
        """Initialize importer with config and Neo4j driver."""
        self.config = config
        self.driver = driver
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            print(f"Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding for text."""
        if not self.embedding_model or not text:
            return None
        
        try:
            # Combine all text fields for embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Warning: Failed to generate embedding: {e}")
            return None
    
    def prepare_properties(self, entity: Dict[str, Any], label: str) -> Dict[str, Any]:
        """Prepare node properties with embedding."""
        # Create text for embedding (name + description)
        text_for_embedding = f"{entity.get('name', '')} {entity.get('description', '')}"
        
        # Add equations if present
        equations = entity.get("equations", [])
        if equations:
            text_for_embedding += " " + " ".join(str(eq) for eq in equations)
        
        # Generate embedding
        embedding = self.generate_embedding(text_for_embedding)
        
        # Base properties
        props = {
            "name": entity["name"],
            "source": entity.get("source", ""),
            "description": entity.get("description", ""),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Add collection-specific fields
        if label == "Property":
            props["units"] = entity.get("units", "")
            props["equations"] = entity.get("equations", [])
        elif label in ["Design_Function", "Physical_Principle"]:
            props["equations"] = entity.get("equations", [])
        elif label in ["Component", "Architecture"]:
            props["type"] = entity.get("type", "")
        
        # Add embedding if available
        if embedding:
            props["embedding"] = embedding
        
        return props
    
    def import_yaml_data(self, yaml_dir: Path):
        """Import all YAML data into Neo4j."""
        loader = YAMLLoader(yaml_dir)
        data = loader.load_all_files()
        
        # Track counts
        counts = {"nodes": 0, "edges": 0}
        
        # Import entities by type
        # Maps entity name -> (Neo4j Node ID/Element ID, Label)
        # Note: Neo4j uses integer IDs or element strings, but we can query by name.
        # We'll just track created names for reference.
        created_names = set()

        # Helper to process a category
        def process_category(key_name, label):
            if key_name in data and data[key_name]:
                for entity in data[key_name]:
                    if isinstance(entity, dict):
                        self._import_node(entity, label)
                        created_names.add(entity["name"])
                        counts["nodes"] += 1
        
        # Order matters less in Neo4j MERGE, but good to follow dependency order
        process_category("properties", "Property")
        process_category("design_functions", "Design_Function")
        process_category("physical_principles", "Physical_Principle")
        process_category("components", "Component")
        process_category("architectures", "Architecture")
        
        # Import relationships
        counts["edges"] = self._import_relationships(data)
        
        print(f"Import complete. Created/Updated {counts['nodes']} nodes and {counts['edges']} edges.")
        return counts
    
    def _import_node(self, entity: Dict[str, Any], label: str):
        """Import a single node."""
        props = self.prepare_properties(entity, label)
        
        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n += $props
        RETURN n
        """
        
        try:
            with self.driver.session() as session:
                session.run(query, name=entity["name"], props=props)
                print(f"Merged {label}: {entity['name']}")
        except Exception as e:
            print(f"Error importing {entity.get('name', 'unknown')} as {label}: {e}")

    def _import_relationships(self, data: Dict) -> int:
        """Import all relationships as edges."""
        edge_count = 0
        
        # Process components and architectures
        for entity_type in ["components", "architectures"]:
            if entity_type not in data:
                continue
            
            source_label = "Architecture" if entity_type == "architectures" else "Component"
            
            for entity in data[entity_type]:
                source_name = entity["name"]
                relationships = entity.get("relationships", {})
                
                if not relationships:
                    continue
                
                # PERFORMS_FUNCTION
                for func_name in relationships.get("performs_function", []):
                    if self._create_edge(source_name, source_label, func_name, "Design_Function", "PERFORMS_FUNCTION"):
                        edge_count += 1
                
                # BASED_ON_PRINCIPLE
                for principle_name in relationships.get("based_on_principle", []):
                    if self._create_edge(source_name, source_label, principle_name, "Physical_Principle", "BASED_ON_PRINCIPLE"):
                        edge_count += 1
                
                # HAS_PROPERTY
                for prop_name in relationships.get("has_property", []):
                    if self._create_edge(source_name, source_label, prop_name, "Property", "HAS_PROPERTY"):
                        edge_count += 1
                
                # USES_COMPONENT (Architectures only)
                if entity_type == "architectures":
                    for comp_name in relationships.get("uses_component", []):
                        if self._create_edge(source_name, source_label, comp_name, "Component", "USES_COMPONENT"):
                            edge_count += 1
                            
        return edge_count

    def _create_edge(self, from_name: str, from_label: str, to_name: str, to_label: str, rel_type: str) -> bool:
        """Create a relationship between two nodes by name."""
        # Note: We try various name formats for the target (spaces, underscores) similar to Arango importer
        
        query = f"""
        MATCH (a:{from_label} {{name: $from_name}})
        MATCH (b:{to_label})
        WHERE b.name = $to_name OR b.name = $to_name_underscore OR b.name = $to_name_lower
        MERGE (a)-[r:{rel_type}]->(b)
        RETURN type(r)
        """
        
        to_name_underscore = to_name.replace(" ", "_")
        to_name_lower = to_name.lower() # Approximate check, though Neo4j is case sensitive usually.
        # Ideally we standardized names before, but the query handles some variance.
        
        try:
            with self.driver.session() as session:
                result = session.run(query, 
                                   from_name=from_name, 
                                   to_name=to_name, 
                                   to_name_underscore=to_name_underscore,
                                   to_name_lower=to_name_lower)
                record = result.single()
                if record:
                    print(f"  Created {rel_type}: {from_name} -> {to_name}")
                    return True
                else:
                    print(f"Warning: Could not link {from_name} -> {to_name} ({rel_type}). Target might be missing.")
                    return False
        except Exception as e:
            print(f"Error creating edge {rel_type} from {from_name} to {to_name}: {e}")
            return False

    def add_document(self, document_data: Dict[str, Any]) -> str:
        """Add a new document node."""
        title = document_data.get("title", "unknown")
        
        props = {
            "title": title,
            "source": document_data.get("source", ""),
            "type": document_data.get("type", "paper"),
            "created_at": datetime.utcnow().isoformat(),
        }
        
        # Flatten metadata if present
        meta = document_data.get("metadata", {})
        if meta:
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    props[f"meta_{k}"] = v
        
        query = """
        MERGE (d:Document {title: $title})
        SET d += $props
        RETURN elementId(d) as id
        """
        
        with self.driver.session() as session:
            result = session.run(query, title=title, props=props)
            record = result.single()
            return record["id"] if record else ""

    def _import_entity_direct(self, entity_data: Dict[str, Any], label: str) -> Optional[str]:
        """Directly import an entity dictionary (helper for client.add_entity)."""
        try:
            self._import_node(entity_data, label)
            # Return name as 'key' effectively, or fetch ID
            return entity_data["name"]
        except Exception:
            return None

    def create_edge_by_id(self, edge_type: str, from_id: str, from_coll: str, to_id: str, to_coll: str) -> bool:
        """Create an edge between two nodes using their element IDs."""
        query = f"""
        MATCH (a) WHERE elementId(a) = $from_id
        MATCH (b) WHERE elementId(b) = $to_id
        MERGE (a)-[r:{edge_type}]->(b)
        RETURN type(r)
        """
        # Note: We ignore collections here as IDs are unique globally in Neo4j (mostly), 
        # but we could verify labels if needed.
        
        try:
            with self.driver.session() as session:
                result = session.run(query, from_id=from_id, to_id=to_id)
                record = result.single()
                if record:
                    print(f"  Created edge {edge_type}: {from_id} -> {to_id}")
                    return True
                else:
                    print(f"Warning: Could not link {from_id} -> {to_id} ({edge_type}). Nodes might be missing.")
                    return False
        except Exception as e:
            print(f"Error creating edge {edge_type} ({from_id} -> {to_id}): {e}")
            return False

    def create_extracted_from_edge(self, entity_name: str, label: str, doc_title: str):
        """Link an entity to a document."""
        query = """
        MATCH (n) WHERE labels(n)[0] = $label AND n.name = $entity_name
        MATCH (d:Document {title: $doc_title})
        MERGE (n)-[:EXTRACTED_FROM]->(d)
        """
        # Note: labels(n)[0] is a bit loose, but sufficient given our distinct labels
        # Better: MATCH (n:SpecificLabel {name: ...})
        
        safe_query = f"""
        MATCH (n:{label} {{name: $entity_name}})
        MATCH (d:Document {{title: $doc_title}})
        MERGE (n)-[:EXTRACTED_FROM]->(d)
        """
        
        with self.driver.session() as session:
            session.run(safe_query, entity_name=entity_name, doc_title=doc_title)

