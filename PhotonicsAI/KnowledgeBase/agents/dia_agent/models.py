"""Pydantic models for DIA Agent."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

from PhotonicsAI.KnowledgeBase.agents.vsa_agent.models import ProposedNode, ProposedEdge

class DIAReport(BaseModel):
    """Report summarizing the integration process."""
    document_key: str = Field(..., description="Source document key")
    nodes_created: int = Field(0, description="Count of nodes created")
    nodes_updated: int = Field(0, description="Count of nodes updated")
    nodes_merged: int = Field(0, description="Count of nodes merged (no change)")
    explicit_edges_created: int = Field(0, description="Count of explicit edges created from manifest")
    inferred_edges_found: int = Field(0, description="Count of implicit global edges discovered and created")
    total_edges_created: int = Field(0, description="Total edges created")
    review_items_queued: int = Field(0, description="Count of nodes/edges queued for review")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    generated_at: str = Field(..., description="Timestamp of report generation")

class SemanticVerificationResult(BaseModel):
    """Result of LLM verification for semantic edges."""
    pair_id: str = Field(..., description="The unique ID of the pair being verified")
    is_related: bool = Field(..., description="True if a valid relationship exists")
    edge_type: str = Field("None", description="The specific edge type (e.g., PERFORMS_FUNCTION) or 'None' if unrelated")
    reasoning: str = Field(..., description="Explanation for the decision")
    confidence: float = Field(..., description="LLM confidence score (0.0 to 1.0)")

class BatchSemanticVerificationResult(BaseModel):
    """Container for batch verification results."""
    results: List[SemanticVerificationResult] = Field(..., description="List of verification results")

