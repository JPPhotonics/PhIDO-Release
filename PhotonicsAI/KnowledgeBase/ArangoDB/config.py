"""ArangoDB connection and configuration settings."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArangoDBConfig:
    """Configuration for ArangoDB connection and vector embeddings."""
    
    # ArangoDB connection settings
    host: str = os.getenv("ARANGO_HOST", "localhost")
    port: int = int(os.getenv("ARANGO_PORT", "8529"))
    username: str = os.getenv("ARANGO_USERNAME", "root")
    password: str = os.getenv("ARANGO_PASSWORD", "my_secure_password")
    database: str = os.getenv("ARANGO_DATABASE", "photonics_kb")
    
    # Vector embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dimension: int = 1024  # Dimension for Qwen3-Embedding-0.6B
    
    # Collection names
    collections: dict = None
    
    def __post_init__(self):
        """Initialize default collection names."""
        if self.collections is None:
            self.collections = {
                "components": "Components",
                "architectures": "Architectures",
                "properties": "Properties",
                "design_functions": "Design_Functions",
                "physical_principles": "Physical_Principles",
                "documents": "Documents",
            }
        
        self.edge_collections = {
            "performs_function": "PERFORMS_FUNCTION",
            "based_on_principle": "BASED_ON_PRINCIPLE",
            "has_property": "HAS_PROPERTY",
            "uses_component": "USES_COMPONENT",
            "related_to": "RELATED_TO",
            "extracted_from": "EXTRACTED_FROM",
        }
    
    @property
    def connection_url(self) -> str:
        """Get ArangoDB connection URL."""
        return f"http://{self.host}:{self.port}"

