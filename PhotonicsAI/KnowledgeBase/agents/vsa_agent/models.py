"""Pydantic models for VSA Agent transaction manifest."""

from typing import List, Optional, Dict, Any, Literal
from enum import Enum
from pydantic import BaseModel, Field

class ProposedNode(BaseModel):
    """A node to be added or updated in the Knowledge Base."""
    
    collection: str = Field(..., description="Target collection (e.g., Components, Architectures)")
    name: str = Field(..., description="Unique name of the entity")
    operation: Literal["CREATE", "UPDATE", "MERGE"] = Field(..., description="Operation type")
    
    # Payload fields
    description: Optional[str] = Field(None, description="Description of the entity")
    evidence_quotes: List[str] = Field(default_factory=list, description="Quotes supporting the facts")
    key_metrics: List[str] = Field(default_factory=list, description="Extracted metrics")
    
    # Architecture-specific fields
    components: List[str] = Field(default_factory=list, description="List of components (for Architectures)")
    connectivity: List[str] = Field(default_factory=list, description="Connectivity details (for Architectures)")
    
    # Metadata
    source_document: Optional[str] = Field(None, description="Document key/ID where this was found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ProposedEdge(BaseModel):
    """An edge to be created between two nodes."""
    
    edge_collection: str = Field(..., description="Edge collection name (e.g., PERFORMS_FUNCTION)")
    from_node: str = Field(..., description="Name of the source node")
    from_collection: str = Field(..., description="Collection of the source node")
    to_node: str = Field(..., description="Name of the target node")
    to_collection: str = Field(..., description="Collection of the target node")
    
    operation: Literal["CREATE", "MERGE"] = Field("CREATE", description="Operation type")
    
    # Context
    description: Optional[str] = Field(None, description="Context for why this relationship exists")
    evidence_quotes: List[str] = Field(default_factory=list, description="Quotes supporting the edge")
    weight: float = Field(1.0, description="Confidence or relevance weight")
    confidence: Optional[float] = Field(None, description="Confidence score (0.0 to 1.0)")
    source_document: Optional[str] = Field(None, description="Document key/ID supporting this edge")
    provenance: Optional[str] = Field(None, description="Originating agent or tool")
    extracted_at: Optional[str] = Field(None, description="Timestamp when edge was extracted")


class VSAUpdatePayload(BaseModel):
    """The Transaction Manifest: final output of the VSA Agent."""
    
    document_key: str = Field(..., description="Key of the processed document")
    nodes: List[ProposedNode] = Field(default_factory=list, description="List of nodes to process")
    edges: List[ProposedEdge] = Field(default_factory=list, description="List of edges to process")
    
    # Validation metadata
    generated_at: str = Field(..., description="ISO timestamp of generation")
    agent_version: str = Field("1.0.0", description="Version of VSA Agent")
    validation_flags: List[str] = Field(default_factory=list, description="Any warnings or flags raised during validation")


# --- Deep Inference Models ---

class EdgeTypeEnum(str, Enum):
    """Allowed edge types for inference."""
    PERFORMS_FUNCTION = "PERFORMS_FUNCTION"
    BASED_ON_PRINCIPLE = "BASED_ON_PRINCIPLE"
    HAS_PROPERTY = "HAS_PROPERTY"
    USES_COMPONENT = "USES_COMPONENT"
    RELATED_TO = "RELATED_TO"
    # EXTRACTED_FROM is handled automatically, not by inference

class InferredEdge(BaseModel):
    """A hypothetical edge inferred from context."""
    source_node: str = Field(..., description="Name of the source entity")
    target_node: str = Field(..., description="Name of the target entity")
    edge_type: str = Field(..., description="Type of relationship (e.g. PERFORMS_FUNCTION, or a novel UPPER_SNAKE_CASE type)")
    confidence_score: float = Field(..., description="Confidence from 0.0 to 1.0")
    inference_reasoning: str = Field(..., description="Explanation of why this edge was inferred")
    # Novel type discovery fields
    is_novel: bool = Field(False, description="True if proposing a new relationship type not in the current schema")
    novel_type_description: Optional[str] = Field(None, description="Description of the proposed novel type (required when is_novel=True)")
    novel_source_types: List[str] = Field(default_factory=list, description="Expected source entity types for the novel type")
    novel_target_types: List[str] = Field(default_factory=list, description="Expected target entity types for the novel type")

class InferredEdgeList(BaseModel):
    """Container for list of inferred edges."""
    edges: List[InferredEdge]
