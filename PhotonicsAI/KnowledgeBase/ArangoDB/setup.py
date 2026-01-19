"""Database setup and initialization utilities."""

from arango import ArangoClient
from arango.database import StandardDatabase
from typing import Optional

from .config import ArangoDBConfig


def create_database_connection(config: ArangoDBConfig) -> StandardDatabase:
    """Create and return ArangoDB database connection."""
    client = ArangoClient(hosts=config.connection_url)
    
    # Connect to system database to create user database if needed
    sys_db = client.db("_system", username=config.username, password=config.password)
    
    # Create database if it doesn't exist
    if not sys_db.has_database(config.database):
        sys_db.create_database(config.database)
        print(f"Created database: {config.database}")
    
    # Connect to the database
    db = client.db(config.database, username=config.username, password=config.password)
    return db


def initialize_database(config: ArangoDBConfig, db: Optional[StandardDatabase] = None) -> StandardDatabase:
    """Initialize database with collections and indexes."""
    if db is None:
        db = create_database_connection(config)
    
    # Create vertex collections
    vertex_collections = [
        "Components",
        "Architectures",
        "Properties",
        "Design_Functions",
        "Physical_Principles",
        "Documents",
    ]
    
    for collection_name in vertex_collections:
        if not db.has_collection(collection_name):
            db.create_collection(collection_name)
            print(f"Created collection: {collection_name}")
    
    # Create edge collections
    edge_collections = [
        "PERFORMS_FUNCTION",
        "BASED_ON_PRINCIPLE",
        "HAS_PROPERTY",
        "USES_COMPONENT",
        "RELATED_TO",
        "EXTRACTED_FROM",
    ]
    
    for collection_name in edge_collections:
        if not db.has_collection(collection_name):
            db.create_collection(collection_name, edge=True)
            print(f"Created edge collection: {collection_name}")
    
    # Create indexes
    _create_indexes(db, config)
    
    # Create named graph for graph queries
    graph_name = "KnowledgeGraph"
    if not db.has_graph(graph_name):
        try:
            # Define edge definitions for the graph
            edge_definitions = [
                {
                    "edge_collection": "PERFORMS_FUNCTION",
                    "from_vertex_collections": ["Components", "Architectures"],
                    "to_vertex_collections": ["Design_Functions"]
                },
                {
                    "edge_collection": "BASED_ON_PRINCIPLE",
                    "from_vertex_collections": ["Components", "Architectures"],
                    "to_vertex_collections": ["Physical_Principles"]
                },
                {
                    "edge_collection": "HAS_PROPERTY",
                    "from_vertex_collections": ["Components", "Architectures"],
                    "to_vertex_collections": ["Properties"]
                },
                {
                    "edge_collection": "USES_COMPONENT",
                    "from_vertex_collections": ["Architectures"],
                    "to_vertex_collections": ["Components"]
                },
                {
                    "edge_collection": "RELATED_TO",
                    "from_vertex_collections": ["Properties", "Physical_Principles", "Design_Functions"],
                    "to_vertex_collections": ["Properties", "Physical_Principles", "Design_Functions"]
                },
                {
                    "edge_collection": "EXTRACTED_FROM",
                    "from_vertex_collections": ["Components", "Architectures", "Properties", "Design_Functions", "Physical_Principles"],
                    "to_vertex_collections": ["Documents"]
                }
            ]
            db.create_graph(graph_name, edge_definitions=edge_definitions)
            print(f"Created named graph: {graph_name}")
        except Exception as e:
            print(f"Error creating graph {graph_name}: {e}")
            
    print("Database initialization complete.")
    return db


def _create_indexes(db: StandardDatabase, config: ArangoDBConfig):
    """Create necessary indexes for efficient queries."""
    
    def _has_index(collection, index_name):
        """Check if collection has an index with the given name."""
        indexes = collection.indexes()
        return any(idx.get("name") == index_name for idx in indexes)
    
    # Create indexes on name fields for fast lookups
    for collection_name in ["Components", "Architectures", "Properties", 
                           "Design_Functions", "Physical_Principles"]:
        collection = db.collection(collection_name)
        
        # Index on name
        if not _has_index(collection, "name"):
            try:
                collection.add_index({"type": "persistent", "fields": ["name"], "name": "name"})
            except Exception as e:
                print(f"Note: Could not create name index on {collection_name}: {e}")
        
        # Index on source for filtering
        if not _has_index(collection, "source"):
            try:
                collection.add_index({"type": "persistent", "fields": ["source"], "name": "source"})
            except Exception as e:
                print(f"Note: Could not create source index on {collection_name}: {e}")
    
    # Note: Vector search in ArangoDB requires special setup
    # We'll use Python-based cosine similarity for vector search instead
    # The embedding field is stored but not indexed (we calculate similarity in Python)
    # This avoids the ArangoSearch index error with array fields
    
    # Check for and remove any existing ArangoSearch indexes that might include embedding
    # Also check for ArangoSearch views that might auto-index all fields
    for collection_name in ["Components", "Architectures", "Properties",
                           "Design_Functions", "Physical_Principles"]:
        collection = db.collection(collection_name)
        indexes = collection.indexes()
        
        # Find and drop any inverted/ArangoSearch indexes
        for idx in indexes:
            idx_type = idx.get("type", "")
            if idx_type == "inverted":
                try:
                    collection.delete_index(idx["id"])
                    print(f"Removed existing ArangoSearch index from {collection_name}")
                except Exception as e:
                    print(f"Note: Could not remove index from {collection_name}: {e}")
    
    # Check for ArangoSearch views and update them to exclude embedding field
    try:
        views = db.views()
        for view in views:
            view_name = view["name"]
            # Try to get view properties and exclude embedding if it's an ArangoSearch view
            try:
                view_obj = db.view(view_name)
                props = view_obj.properties()
                if props.get("type") == "arangosearch":
                    # Update view to exclude embedding field
                    links = props.get("links", {})
                    for coll_name in links:
                        if "fields" in links[coll_name]:
                            # Exclude embedding from fields
                            fields = links[coll_name]["fields"]
                            if isinstance(fields, dict) and "embedding" in fields:
                                del fields["embedding"]
                                view_obj.properties(links=links)
                                print(f"Updated ArangoSearch view {view_name} to exclude embedding field")
            except Exception as e:
                # View might not be accessible or modifiable
                pass
    except Exception as e:
        # Views might not be accessible
        pass
    
    print("Note: Using Python-based vector search (embeddings stored but not indexed)")
    
    # Create edge indexes for traversal
    for edge_collection_name in ["PERFORMS_FUNCTION", "BASED_ON_PRINCIPLE", 
                                 "HAS_PROPERTY", "USES_COMPONENT", "RELATED_TO", "EXTRACTED_FROM"]:
        edge_collection = db.collection(edge_collection_name)
        
        # Index on _from and _to for fast traversal
        if not _has_index(edge_collection, "_from"):
            try:
                edge_collection.add_index({"type": "persistent", "fields": ["_from"], "name": "_from"})
            except Exception as e:
                print(f"Note: Could not create _from index on {edge_collection_name}: {e}")
        if not _has_index(edge_collection, "_to"):
            try:
                edge_collection.add_index({"type": "persistent", "fields": ["_to"], "name": "_to"})
            except Exception as e:
                print(f"Note: Could not create _to index on {edge_collection_name}: {e}")


def reset_database(config: ArangoDBConfig, confirm: bool = False):
    """Reset database by dropping all collections (use with caution!)."""
    if not confirm:
        raise ValueError("reset_database requires confirm=True")
    
    db = create_database_connection(config)
    
    # Drop all collections
    collections = db.collections()
    for collection in collections:
        if collection["name"] not in ["_graphs", "_analyzers"]:  # System collections
            db.delete_collection(collection["name"])
            print(f"Dropped collection: {collection['name']}")
    
    # Reinitialize
    initialize_database(config, db)
    print("Database reset complete.")

