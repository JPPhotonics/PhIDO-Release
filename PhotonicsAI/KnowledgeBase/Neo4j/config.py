"""Neo4j connection and configuration settings."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection and vector embeddings."""
    
    # Neo4j connection settings
    uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username: str = os.getenv("NEO4J_USER", "neo4j")
    password: str = os.getenv("NEO4J_PASSWORD", "password")
    database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Vector embedding model
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    embedding_dimension: int = 1024  # Dimension for Qwen3-Embedding-0.6B
    
    # Label/Collection mapping (mimicking ArangoDB collections)
    labels: Dict[str, str] = field(default_factory=lambda: {
        "components": "Component",
        "architectures": "Architecture",
        "properties": "Property",
        "design_functions": "Design_Function",
        "physical_principles": "Physical_Principle",
        "documents": "Document",
        "pdk_cells": "PDK_Cell",
    })
    
    # Relationship mapping
    relationships: Dict[str, str] = field(default_factory=lambda: {
        "performs_function": "PERFORMS_FUNCTION",
        "based_on_principle": "BASED_ON_PRINCIPLE",
        "has_property": "HAS_PROPERTY",
        "uses_component": "USES_COMPONENT",
        "related_to": "RELATED_TO",
        "extracted_from": "EXTRACTED_FROM",
        "implements": "IMPLEMENTS",
        "composed_of": "COMPOSED_OF",
        "exhibits": "EXHIBITS",
        "fabricated_with": "FABRICATED_WITH",
        "supersedes": "SUPERSEDES",
    })

    def get_label_for_collection(self, collection_name: str) -> str:
        """Map generic collection names (e.g. 'Components') to Neo4j Labels (e.g. 'Component')."""
        mapping = {
            "Components": "Component",
            "Architectures": "Architecture",
            "Properties": "Property",
            "Design_Functions": "Design_Function",
            "Physical_Principles": "Physical_Principle",
            "Documents": "Document",
            "PDK_Cells": "PDK_Cell",
        }
        return mapping.get(collection_name, collection_name.rstrip('s'))

