"""ArangoDB Knowledge Base module for photonic ontology storage and retrieval."""

from .client import KnowledgeBaseClient
from .config import ArangoDBConfig

__all__ = ["KnowledgeBaseClient", "ArangoDBConfig"]

