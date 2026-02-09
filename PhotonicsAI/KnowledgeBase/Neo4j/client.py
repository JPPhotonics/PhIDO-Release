"""Main client class for Neo4j knowledge base operations."""

from pathlib import Path
from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase

from .config import Neo4jConfig
from .importer import Neo4jImporter
from .schema_registry import SchemaRegistry

class Neo4jClient:
    """Main client for interacting with the Neo4j knowledge base."""
    
    def __init__(self, config: Optional[Neo4jConfig] = None):
        """Initialize knowledge base client."""
        self.config = config or Neo4jConfig()
        self.driver = None
        self.importer: Optional[Neo4jImporter] = None
        self._connected = False
    
    def connect(self):
        """Connect to Neo4j database."""
        if not self._connected:
            self.driver = GraphDatabase.driver(
                self.config.uri, 
                auth=(self.config.username, self.config.password)
            )
            self.importer = Neo4jImporter(self.config, self.driver)
            # Compatibility layer: Alias self as retrieval engine
            # kb_grounding_tool expects client.retrieval.vector_semantic_search
            self.retrieval = self 
            self._connected = True
            
            # Verify connection
            try:
                self.driver.verify_connectivity()
            except Exception as e:
                self._connected = False
                raise ConnectionError(f"Failed to connect to Neo4j: {e}")

    # Compatibility alias for kb_grounding_tool
    def vector_semantic_search(self, query_text: str, collection: str,
                             limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Alias for semantic_search to match ArangoDB RetrievalEngine interface."""
        results = self.semantic_search(query_text, collection, limit, threshold)
        # kb_grounding_tool expects format: [{"vertex": {...}, "similarity": 0.9}]
        # semantic_search currently returns: [{"name":..., "score": 0.9, ...}]
        
        formatted = []
        for r in results:
            # Reconstruct vertex dict from flat result
            vertex = r.copy()
            score = vertex.pop("score", 0.0)
            formatted.append({
                "vertex": vertex,
                "similarity": score
            })
        return formatted
    
    def close(self):
        """Close connection."""
        if self.driver:
            self.driver.close()
            self._connected = False

    def initialize(self, reset: bool = False):
        """Initialize database schema and indices."""
        if not self._connected:
            self.connect()
            
        if reset:
            # Dangerous! Wipes the DB.
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                # Also drop indexes?
                # session.run("CALL db.index.vector.queryNodes(...)") - complicates things.
                # For now, just clearing data is often enough for re-init.
        
        self._initialize_vector_indexes()
        self._initialize_constraints()
        # Ensure seed relationship types exist in the schema registry
        self.get_schema_registry().initialize_seed_schema()

    def get_schema_registry(self) -> SchemaRegistry:
        """Return a SchemaRegistry instance backed by this client's driver."""
        if not self._connected:
            self.connect()
        return SchemaRegistry(self.driver)

    def _initialize_vector_indexes(self):
        """Automatically create vector indexes if they don't exist."""
        # For each label that has embeddings, create a vector index
        # We need indexes for Component, Architecture, Property, Design_Function, Physical_Principle
        
        labels_to_index = [
            "Component", "Architecture", "Property", 
            "Design_Function", "Physical_Principle"
        ]
        
        dim = self.config.embedding_dimension
        
        with self.driver.session() as session:
            for label in labels_to_index:
                index_name = f"{label.lower()}_embedding_index"
                
                # Check if exists
                check_query = "SHOW INDEXES WHERE name = $name"
                result = session.run(check_query, name=index_name)
                if result.peek():
                    print(f"Vector index {index_name} already exists.")
                    continue
                
                print(f"Creating vector index: {index_name} for label {label}...")
                
                # Create vector index
                # Note: Syntax varies slightly by Neo4j version (5.x vs 4.x). Assuming 5.x standard.
                create_query = f"""
                CREATE VECTOR INDEX {index_name} IF NOT EXISTS
                FOR (n:{label})
                ON (n.embedding)
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {dim},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
                try:
                    session.run(create_query)
                    print(f"✓ Created {index_name}")
                except Exception as e:
                    print(f"Error creating index {index_name}: {e}")

    def _initialize_constraints(self):
        """Create uniqueness constraints for names."""
        labels = [
            "Component", "Architecture", "Property", 
            "Design_Function", "Physical_Principle", "Document"
        ]
        
        with self.driver.session() as session:
            for label in labels:
                constraint_name = f"{label.lower()}_name_unique"
                query = f"""
                CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                FOR (n:{label}) REQUIRE n.name IS UNIQUE
                """
                # For Document, it's title
                if label == "Document":
                    query = f"""
                    CREATE CONSTRAINT document_title_unique IF NOT EXISTS
                    FOR (n:Document) REQUIRE n.title IS UNIQUE
                    """
                
                try:
                    session.run(query)
                except Exception as e:
                    print(f"Warning: Could not create constraint for {label}: {e}")

    def import_yaml_data(self, yaml_dir: Path):
        """Import YAML ontology files into the database."""
        if not self._connected:
            self.connect()
        
        yaml_dir = Path(yaml_dir)
        return self.importer.import_yaml_data(yaml_dir)

    # Retrieval methods (Mirroring KnowledgeBaseClient)
    
    def find_by_name(self, name: str, collection: str) -> Optional[Dict[str, Any]]:
        """Find a node by name."""
        if not self._connected:
            self.connect()
        
        label = self.config.get_label_for_collection(collection)
        
        # Try searching by exact name first
        query = f"""
        MATCH (n:{label})
        WHERE n.name = $name
        RETURN n
        """
        
        with self.driver.session() as session:
            result = session.run(query, name=name)
            record = result.single()
            if record:
                return self._record_to_dict(record["n"])
        
        # Fallback: Try with hyphens instead of en-dashes (or vice versa)
        # Common issue: "Mach-Zehnder" (hyphen) vs "Mach–Zehnder" (en-dash)
        if "-" in name:
            alt_name = name.replace("-", "–") # Try en-dash
        elif "–" in name:
            alt_name = name.replace("–", "-") # Try hyphen
        else:
            alt_name = None
            
        if alt_name:
            with self.driver.session() as session:
                result = session.run(query, name=alt_name)
                record = result.single()
                if record:
                    return self._record_to_dict(record["n"])

        # Fallback: Try case-insensitive search (slower, but robust)
        # Using db.index.fulltext is better but requires setup.
        # Simple regex filter for now (okay for small lookups, bad for massive DBs)
        query_insensitive = f"""
        MATCH (n:{label})
        WHERE toLower(n.name) = toLower($name)
        RETURN n
        """
        with self.driver.session() as session:
            result = session.run(query_insensitive, name=name)
            record = result.single()
            if record:
                 return self._record_to_dict(record["n"])

        return None

    def _record_to_dict(self, node):
        """Helper to convert Neo4j node to dict with _key."""
        data = dict(node)
        data["_key"] = node.element_id if hasattr(node, 'element_id') else node.id
        return data

    def semantic_search(self, query_text: str, collection: str,
                       limit: int = 10, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Perform semantic vector search."""
        if not self._connected:
            self.connect()
        
        label = self.config.get_label_for_collection(collection)
        index_name = f"{label.lower()}_embedding_index"
        
        # Generate query embedding
        embedding = self.importer.generate_embedding(query_text)
        if not embedding:
            return []
            
        # Cypher for vector search
        # Using db.index.vector.queryNodes
        query = f"""
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        WHERE score >= $threshold
        RETURN node, score
        """
        
        results = []
        with self.driver.session() as session:
            try:
                result = session.run(query, 
                                   index_name=index_name, 
                                   limit=limit, 
                                   embedding=embedding, 
                                   threshold=threshold)
                
                for record in result:
                    node = record["node"]
                    data = dict(node)
                    data["_key"] = node.element_id if hasattr(node, 'element_id') else node.id
                    data["score"] = record["score"]
                    results.append(data)
            except Exception as e:
                print(f"Vector search failed (index {index_name} might be missing): {e}")
                
        return results

    # Write methods
    
    def add_entity(self, entity_data: Dict[str, Any], collection_name: str,
                  document_key: Optional[str] = None) -> Optional[str]:
        """Add a new entity to the knowledge base."""
        if not self._connected:
            self.connect()
            
        label = self.config.get_label_for_collection(collection_name)
        name = self.importer._import_entity_direct(entity_data, label)
        
        if name and document_key:
            # We assume document_key is the title for linking
            self.importer.create_extracted_from_edge(name, label, document_key)
            
        return name

    def add_document(self, title: str, source: str, doc_type: str = "paper",
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a new document."""
        if not self._connected:
            self.connect()
            
        data = {
            "title": title,
            "source": source,
            "type": doc_type,
            "metadata": metadata
        }
        return self.importer.add_document(data)

    def update_node(self, node_data: Dict[str, Any], collection_name: str):
        """Update a node in the database."""
        if not self._connected:
            self.connect()
            
        key = node_data.get("_key")
        if not key:
            raise ValueError("Node data must contain '_key' (elementId) for update.")
            
        # We need to construct a SET clause.
        # We assume properties are flat or can be stored as is.
        # Note: _key, _id, element_id should not be in properties.
        props = {k: v for k, v in node_data.items() if not k.startswith("_") and k != "element_id"}
        
        query = """
        MATCH (n)
        WHERE elementId(n) = $key
        SET n += $props
        RETURN n
        """
        
        with self.driver.session() as session:
            session.run(query, key=key, props=props)
            
    def update_entity_embedding(self, entity_key: str, collection: str):
        """Update embedding for an existing entity."""
        if not self._connected:
            self.connect()
        
        # 1. Fetch node to get text
        query_fetch = "MATCH (n) WHERE elementId(n) = $key RETURN n"
        with self.driver.session() as session:
            result = session.run(query_fetch, key=entity_key)
            record = result.single()
            if not record:
                return
            node = record["n"]
            
            # 2. Generate embedding
            text = f"{node.get('name', '')} {node.get('description', '')}"
            equations = node.get("equations", [])
            if equations:
                text += " " + " ".join(str(eq) for eq in equations)
                
            embedding = self.importer.generate_embedding(text)
            
            # 3. Update
            if embedding:
                query_update = """
                MATCH (n) WHERE elementId(n) = $key 
                SET n.embedding = $embedding, n.updated_at = datetime()
                """
                session.run(query_update, key=entity_key, embedding=embedding)

    def create_edge(
        self,
        edge_type: str,
        from_key: str,
        from_collection: str,
        to_key: str,
        to_collection: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Create an edge between two nodes identified by their keys (Element IDs)."""
        if not self._connected:
            self.connect()
        return self.importer.create_edge_by_id(
            edge_type,
            from_key,
            from_collection,
            to_key,
            to_collection,
            props=props,
        )


