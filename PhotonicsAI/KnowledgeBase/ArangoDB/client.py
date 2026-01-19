"""Main client class for ArangoDB knowledge base operations."""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from arango.database import StandardDatabase

from .config import ArangoDBConfig
from .setup import create_database_connection, initialize_database
from .importer import DataImporter
from .retrieval import RetrievalEngine


class KnowledgeBaseClient:
    """Main client for interacting with the ArangoDB knowledge base."""
    
    def __init__(self, config: Optional[ArangoDBConfig] = None):
        """Initialize knowledge base client."""
        self.config = config or ArangoDBConfig()
        self.db: Optional[StandardDatabase] = None
        self.importer: Optional[DataImporter] = None
        self.retrieval: Optional[RetrievalEngine] = None
        self._connected = False
    
    @property
    def hosts(self):
        """Get database hosts."""
        return [self.config.connection_url]

    @property
    def username(self):
        """Get database username."""
        return self.config.username

    @property
    def password(self):
        """Get database password."""
        return self.config.password

    @property
    def db_name(self):
        """Get database name."""
        return self.config.database

    def connect(self):
        """Connect to ArangoDB database."""
        if not self._connected:
            self.db = create_database_connection(self.config)
            self.importer = DataImporter(self.config, self.db)
            self.retrieval = RetrievalEngine(self.db, self.config)
            self._connected = True
    
    def initialize(self, reset: bool = False):
        """Initialize database schema."""
        if reset:
            from .setup import reset_database
            reset_database(self.config, confirm=True)
        else:
            if not self._connected:
                self.connect()
            initialize_database(self.config, self.db)
    
    def import_yaml_data(self, yaml_dir: Path):
        """Import YAML ontology files into the database."""
        if not self._connected:
            self.connect()
        
        yaml_dir = Path(yaml_dir)
        return self.importer.import_yaml_data(yaml_dir)
    
    # Retrieval methods
    def get_neighbors(self, vertex_name: str, collection: str,
                     edge_types: Optional[List[str]] = None,
                     max_depth: int = 1) -> List[Dict[str, Any]]:
        """Get direct neighbors of a vertex."""
        if not self._connected:
            self.connect()
        
        # First find the vertex by name to get its actual _key
        vertex = self.find_by_name(vertex_name, collection)
        if not vertex:
            return []
        
        vertex_key = vertex["_key"]
        return self.retrieval.get_direct_neighbors(vertex_key, collection, edge_types, max_depth)
    
    def semantic_search(self, query_text: str, collection: str,
                       limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic vector search."""
        if not self._connected:
            self.connect()
        
        results = self.retrieval.vector_semantic_search(query_text, collection, limit, threshold)
        return [r["vertex"] for r in results]
    
    def traverse(self, start_name: str, start_collection: str,
                relationship_path: List[str], max_depth: int = 3) -> List[Dict[str, Any]]:
        """Traverse relationships following a path pattern."""
        if not self._connected:
            self.connect()
        
        # First find the vertex by name to get its actual _key
        vertex = self.find_by_name(start_name, start_collection)
        if not vertex:
            return []
        
        start_key = vertex["_key"]
        return self.retrieval.traverse_relationships(start_key, start_collection, relationship_path, max_depth)
    
    def hybrid_search(self, query_text: str, collection: str,
                     relationship_filters: Optional[Dict[str, Any]] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and graph queries."""
        if not self._connected:
            self.connect()
        
        return self.retrieval.hybrid_search(query_text, collection, relationship_filters, limit)
    
    # Incremental update methods
    def add_document(self, title: str, source: str, doc_type: str = "paper",
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new document to the knowledge base."""
        if not self._connected:
            self.connect()
        
        document_data = {
            "title": title,
            "source": source,
            "type": doc_type,
            "metadata": metadata or {},
        }
        return self.importer.add_document(document_data)
    
    def add_entity(self, entity_data: Dict[str, Any], collection_name: str,
                  document_key: Optional[str] = None) -> Optional[str]:
        """Add a new entity to the knowledge base."""
        if not self._connected:
            self.connect()
        
        vertex_map = self.importer._import_entity(entity_data, collection_name)
        
        if entity_data["name"] in vertex_map:
            entity_key = vertex_map[entity_data["name"]][0]
            
            # Link to document if provided
            if document_key:
                self.importer._create_edge("EXTRACTED_FROM", entity_key, collection_name,
                                         document_key, "Documents")
            
            return entity_key
        
        return None
    
    def add_architecture(self, name: str, description: str, source: str,
                       relationships: Optional[Dict[str, List[str]]] = None,
                       document_key: Optional[str] = None) -> Optional[str]:
        """Add a new architecture to the knowledge base."""
        entity_data = {
            "name": name,
            "type": "Architecture",
            "source": source,
            "description": description,
            "relationships": relationships or {},
        }
        
        entity_key = self.add_entity(entity_data, "Architectures", document_key)
        
        # Add relationships if provided
        if entity_key and relationships:
            self._add_relationships(entity_key, "Architectures", relationships)
        
        return entity_key
    
    def add_component(self, name: str, description: str, source: str,
                     relationships: Optional[Dict[str, List[str]]] = None,
                     document_key: Optional[str] = None) -> Optional[str]:
        """Add a new component to the knowledge base."""
        entity_data = {
            "name": name,
            "type": "Component",
            "source": source,
            "description": description,
            "relationships": relationships or {},
        }
        
        entity_key = self.add_entity(entity_data, "Components", document_key)
        
        # Add relationships if provided
        if entity_key and relationships:
            self._add_relationships(entity_key, "Components", relationships)
        
        return entity_key
    
    def _add_relationships(self, entity_key: str, collection: str,
                           relationships: Dict[str, List[str]]):
        """Add relationships for an entity."""
        # Get vertex map for target lookups
        vertex_map = {}
        for coll_name in ["Components", "Architectures", "Properties",
                         "Design_Functions", "Physical_Principles"]:
            coll = self.db.collection(coll_name)
            for doc in coll:
                vertex_map[doc["name"]] = (doc["_key"], coll_name)
        
        # Create edges
        for rel_type, target_names in relationships.items():
            edge_type_map = {
                "performs_function": "PERFORMS_FUNCTION",
                "based_on_principle": "BASED_ON_PRINCIPLE",
                "has_property": "HAS_PROPERTY",
                "uses_component": "USES_COMPONENT",
            }
            
            edge_type = edge_type_map.get(rel_type)
            if not edge_type:
                continue
            
            for target_name in target_names:
                if target_name in vertex_map:
                    target_key, target_collection = vertex_map[target_name]
                    self.importer._create_edge(edge_type, entity_key, collection,
                                              target_key, target_collection)
    
    def update_entity_embedding(self, entity_key: str, collection: str):
        """Update embedding for an existing entity."""
        if not self._connected:
            self.connect()
        
        coll = self.db.collection(collection)
        entity = coll.get(entity_key)
        
        if entity:
            # Regenerate embedding
            text = f"{entity.get('name', '')} {entity.get('description', '')}"
            embedding = self.importer.generate_embedding(text)
            
            if embedding:
                entity["embedding"] = embedding
                entity["updated_at"] = datetime.utcnow().isoformat()
                coll.update(entity)
    
    def _sanitize_key(self, name: str) -> str:
        """Sanitize name to create valid ArangoDB key (matches importer logic)."""
        # Use the same logic as importer._sanitize_key
        key = name.replace(" ", "_").replace("-", "_").replace("–", "_")  # Handle en-dash
        # Handle Greek letters and special characters
        replacements = {
            "π": "pi", "Δ": "Delta", "α": "alpha", "β": "beta", "γ": "gamma",
            "δ": "delta", "ε": "epsilon", "θ": "theta", "λ": "lambda",
            "μ": "mu", "σ": "sigma", "ω": "omega",
        }
        for char, replacement in replacements.items():
            key = key.replace(char, replacement)
        # Remove any remaining special characters (keep only alphanumeric and underscore)
        key = "".join(c if c.isalnum() or c == "_" else "" for c in key)
        # Ensure key doesn't start with a number
        if key and key[0].isdigit():
            key = "_" + key
        return key
    
    # Convenience query methods
    def find_by_name(self, name: str, collection: str) -> Optional[Dict[str, Any]]:
        """Find a vertex by name."""
        if not self._connected:
            self.connect()
        
        coll = self.db.collection(collection)
        key = self._sanitize_key(name)
        
        if coll.has(key):
            return coll.get(key)
        
        # Try searching by name field (use backticks for collection name)
        query = f"""
        FOR doc IN `{collection}`
            FILTER doc.name == @name
            RETURN doc
        """
        try:
            cursor = self.db.aql.execute(query, bind_vars={"name": name})
            results = list(cursor)
            return results[0] if results else None
        except Exception:
            return None
    
    def find_components_by_principle(self, principle_name: str) -> List[Dict[str, Any]]:
        """Find all components/architectures using a specific principle."""
        if not self._connected:
            self.connect()
        
        return self.retrieval.find_components_by_principle(principle_name)
    
    def find_architectures_using_component(self, component_name: str) -> List[Dict[str, Any]]:
        """Find all architectures that use a specific component."""
        if not self._connected:
            self.connect()
        
        return self.retrieval.find_architectures_using_component(component_name)
    
    def get_visualizer(self):
        """Get a GraphVisualizer instance for visualization."""
        if not self._connected:
            self.connect()
        
        from .visualization import GraphVisualizer
        return GraphVisualizer(self.db, self.config)

