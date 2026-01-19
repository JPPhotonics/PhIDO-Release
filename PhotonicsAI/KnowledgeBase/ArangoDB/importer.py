"""Import YAML data into ArangoDB with vector embeddings."""

import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from arango import ArangoClient
from sentence_transformers import SentenceTransformer

from .config import ArangoDBConfig
from .yaml_loader import YAMLLoader


class DataImporter:
    """Import ontology data into ArangoDB with vector embeddings."""
    
    def __init__(self, config: ArangoDBConfig, db):
        """Initialize importer with config and database connection."""
        self.config = config
        self.db = db
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
    
    def prepare_vertex(self, entity: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """Prepare vertex document with embedding."""
        # Create text for embedding (name + description)
        text_for_embedding = f"{entity.get('name', '')} {entity.get('description', '')}"
        
        # Add equations if present
        equations = entity.get("equations", [])
        if equations:
            text_for_embedding += " " + " ".join(str(eq) for eq in equations)
        
        # Generate embedding
        embedding = self.generate_embedding(text_for_embedding)
        
        # Create vertex document
        vertex = {
            "_key": self._sanitize_key(entity["name"]),
            "name": entity["name"],
            "source": entity.get("source", ""),
            "description": entity.get("description", ""),
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Add collection-specific fields
        if collection_name == "Properties":
            vertex["units"] = entity.get("units", "")
            vertex["equations"] = entity.get("equations", [])
        elif collection_name in ["Design_Functions", "Physical_Principles"]:
            vertex["equations"] = entity.get("equations", [])
        elif collection_name in ["Components", "Architectures"]:
            vertex["type"] = entity.get("type", "")
        
        # Add embedding if available
        if embedding:
            vertex["embedding"] = embedding
        
        return vertex
    
    def _sanitize_key(self, name: str) -> str:
        """Sanitize name to create valid ArangoDB key."""
        # Replace spaces and special chars with underscores
        key = name.replace(" ", "_").replace("-", "_")
        # Replace common special characters
        replacements = {
            "π": "pi",
            "Δ": "Delta",
            "α": "alpha",
            "β": "beta",
            "λ": "lambda",
            "τ": "tau",
            "η": "eta",
            "κ": "kappa",
            "ϕ": "phi",
            "θ": "theta",
            "μ": "mu",
            "ε": "epsilon",
            "σ": "sigma",
            "ω": "omega",
        }
        for char, replacement in replacements.items():
            key = key.replace(char, replacement)
        # Remove any remaining special characters (keep only alphanumeric and underscore)
        key = "".join(c if c.isalnum() or c == "_" else "" for c in key)
        # Ensure key doesn't start with a number
        if key and key[0].isdigit():
            key = "_" + key
        return key
    
    def import_yaml_data(self, yaml_dir: Path, batch_size: int = 100):
        """Import all YAML data into ArangoDB."""
        loader = YAMLLoader(yaml_dir)
        data = loader.load_all_files()
        
        # Import entities
        vertex_map = {}  # Map entity names to their _key and collection
        
        # Import properties
        if "properties" in data and data["properties"]:
            for entity in data["properties"]:
                if isinstance(entity, dict):
                    vertex_map.update(self._import_entity(entity, "Properties"))
        
        # Import design functions
        if "design_functions" in data and data["design_functions"]:
            for entity in data["design_functions"]:
                if isinstance(entity, dict):
                    vertex_map.update(self._import_entity(entity, "Design_Functions"))
        
        # Import physical principles
        if "physical_principles" in data and data["physical_principles"]:
            for entity in data["physical_principles"]:
                if isinstance(entity, dict):
                    vertex_map.update(self._import_entity(entity, "Physical_Principles"))
        
        # Import components
        if "components" in data and data["components"]:
            for entity in data["components"]:
                vertex_map.update(self._import_entity(entity, "Components"))
        
        # Import architectures
        if "architectures" in data and data["architectures"]:
            for entity in data["architectures"]:
                vertex_map.update(self._import_entity(entity, "Architectures"))
        
        # Import relationships
        self._import_relationships(data, vertex_map)
        
        print(f"Import complete. Created {len(vertex_map)} vertices.")
        return vertex_map
    
    def _import_entity(self, entity: Dict[str, Any], collection_name: str) -> Dict[str, tuple]:
        """Import a single entity and return mapping."""
        vertex_map = {}
        
        try:
            vertex = self.prepare_vertex(entity, collection_name)
            collection = self.db.collection(collection_name)
            
            # Check if vertex already exists
            if collection.has(vertex["_key"]):
                # Update existing vertex
                collection.update(vertex)
                print(f"Updated {collection_name}: {vertex['name']}")
            else:
                # Insert new vertex
                collection.insert(vertex)
                print(f"Inserted {collection_name}: {vertex['name']}")
            
            # Add to vertex_map using the entity name as key
            vertex_map[entity["name"]] = (vertex["_key"], collection_name)
        except Exception as e:
            print(f"Error importing {entity.get('name', 'unknown')} to {collection_name}: {e}")
        
        return vertex_map
    
    def _import_relationships(self, data: Dict, vertex_map: Dict[str, tuple]):
        """Import all relationships as edges."""
        edge_count = 0
        
        # Debug: Print vertex_map keys to see what we have
        print(f"Debug: vertex_map has {len(vertex_map)} entries")
        print(f"Debug: Sample keys: {list(vertex_map.keys())[:10]}")
        
        # Process components and architectures
        for entity_type in ["components", "architectures"]:
            if entity_type not in data:
                continue
            
            for entity in data[entity_type]:
                source_key = self._sanitize_key(entity["name"])
                source_collection = "Architectures" if entity_type == "architectures" else "Components"
                
                # Get source info - check vertex_map first, then try database lookup
                if entity["name"] in vertex_map:
                    source_info = vertex_map[entity["name"]]
                else:
                    # Try to find in database (might have been imported in previous run)
                    source_info = self._find_target_vertex(entity["name"], vertex_map, [source_collection])
                    if not source_info:
                        # If still not found, use the sanitized key (entity should exist)
                        try:
                            coll = self.db.collection(source_collection)
                            if coll.has(source_key):
                                source_info = (source_key, source_collection)
                            else:
                                print(f"Warning: Source entity '{entity['name']}' not found, skipping relationships")
                                continue
                        except Exception as e:
                            print(f"Warning: Could not verify source entity '{entity['name']}': {e}")
                            continue
                
                relationships = entity.get("relationships", {})
                if not relationships:
                    continue
                
                print(f"Debug: Processing {entity['name']} with {sum(len(v) for v in relationships.values())} relationships")
                
                # PERFORMS_FUNCTION edges
                for func_name in relationships.get("performs_function", []):
                    target_info = self._find_target_vertex(func_name, vertex_map, ["Design_Functions"])
                    if target_info:
                        self._create_edge("PERFORMS_FUNCTION", source_key, source_collection, 
                                        target_info[0], target_info[1])
                        edge_count += 1
                        print(f"  Created PERFORMS_FUNCTION edge: {entity['name']} -> {func_name}")
                    else:
                        print(f"Warning: Could not find Design_Function '{func_name}' for {entity['name']}")
                
                # BASED_ON_PRINCIPLE edges
                for principle_name in relationships.get("based_on_principle", []):
                    target_info = self._find_target_vertex(principle_name, vertex_map, ["Physical_Principles"])
                    if target_info:
                        self._create_edge("BASED_ON_PRINCIPLE", source_key, source_collection,
                                        target_info[0], target_info[1])
                        edge_count += 1
                    else:
                        print(f"Warning: Could not find Physical_Principle '{principle_name}' for {entity['name']}")
                
                # HAS_PROPERTY edges
                for prop_name in relationships.get("has_property", []):
                    target_info = self._find_target_vertex(prop_name, vertex_map, ["Properties"])
                    if target_info:
                        self._create_edge("HAS_PROPERTY", source_key, source_collection,
                                        target_info[0], target_info[1])
                        edge_count += 1
                    else:
                        print(f"Warning: Could not find Property '{prop_name}' for {entity['name']}")
                
                # USES_COMPONENT edges (only for architectures)
                if entity_type == "architectures":
                    for comp_name in relationships.get("uses_component", []):
                        target_info = self._find_target_vertex(comp_name, vertex_map, ["Components"])
                        if target_info:
                            self._create_edge("USES_COMPONENT", source_key, source_collection,
                                            target_info[0], target_info[1])
                            edge_count += 1
                        else:
                            print(f"Warning: Could not find Component '{comp_name}' for {entity['name']}")
        
        print(f"Created {edge_count} edges.")
    
    def _find_target_vertex(self, name: str, vertex_map: Dict[str, tuple], 
                           collections: List[str]) -> Optional[tuple]:
        """Find target vertex by name, trying various name formats."""
        # Try exact match first
        if name in vertex_map:
            return vertex_map[name]
        
        # Try with underscores instead of spaces
        name_underscore = name.replace(" ", "_")
        if name_underscore in vertex_map:
            return vertex_map[name_underscore]
        
        # Try case-insensitive match
        name_lower = name.lower()
        for key, value in vertex_map.items():
            if key.lower() == name_lower:
                return value
        
        # Try with underscores and case-insensitive
        for key, value in vertex_map.items():
            if key.replace(" ", "_").lower() == name_underscore.lower():
                return value
        
        # Try searching in database by name field
        for collection_name in collections:
            try:
                coll = self.db.collection(collection_name)
                # Search by name field
                query = f"""
                FOR doc IN {collection_name}
                    FILTER doc.name == @name OR doc.name == @name_underscore
                    RETURN doc
                """
                cursor = self.db.aql.execute(
                    query,
                    bind_vars={"name": name, "name_underscore": name_underscore}
                )
                results = list(cursor)
                if results:
                    doc = results[0]
                    return (doc["_key"], collection_name)
            except Exception:
                pass
        
        return None
    
    def _create_edge(self, edge_type: str, from_key: str, from_collection: str,
                    to_key: str, to_collection: str):
        """Create an edge between two vertices."""
        try:
            edge_collection = self.db.collection(edge_type)
            edge_key = f"{from_key}_{to_key}"
            
            edge = {
                "_key": edge_key,
                "_from": f"{from_collection}/{from_key}",
                "_to": f"{to_collection}/{to_key}",
            }
            
            if edge_collection.has(edge_key):
                edge_collection.update(edge)
            else:
                edge_collection.insert(edge)
        except Exception as e:
            print(f"Error creating edge {edge_type} from {from_key} to {to_key}: {e}")
    
    def add_document(self, document_data: Dict[str, Any]) -> str:
        """Add a new document to the Documents collection."""
        doc_collection = self.db.collection("Documents")
        
        doc = {
            "_key": self._sanitize_key(document_data.get("title", "unknown")),
            "title": document_data.get("title", ""),
            "source": document_data.get("source", ""),
            "type": document_data.get("type", "paper"),
            "metadata": document_data.get("metadata", {}),
            "created_at": datetime.utcnow().isoformat(),
        }
        
        if doc_collection.has(doc["_key"]):
            doc_collection.update(doc)
        else:
            doc_collection.insert(doc)
        
        return doc["_key"]
    
    def add_entity_from_document(self, entity_data: Dict[str, Any], 
                                collection_name: str, document_key: str) -> Optional[str]:
        """Add a new entity extracted from a document."""
        # Import the entity
        vertex_map = self._import_entity(entity_data, collection_name)
        
        if entity_data["name"] in vertex_map:
            entity_key = vertex_map[entity_data["name"]][0]
            
            # Create EXTRACTED_FROM edge
            try:
                edge_collection = self.db.collection("EXTRACTED_FROM")
                edge_key = f"{entity_key}_{document_key}"
                edge = {
                    "_key": edge_key,
                    "_from": f"{collection_name}/{entity_key}",
                    "_to": f"Documents/{document_key}",
                }
                if edge_collection.has(edge_key):
                    edge_collection.update(edge)
                else:
                    edge_collection.insert(edge)
            except Exception as e:
                print(f"Error creating EXTRACTED_FROM edge: {e}")
            
            return entity_key
        
        return None

