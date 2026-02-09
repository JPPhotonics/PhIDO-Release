"""Pydantic models for Schema Evolution Agent (SEA)."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class ObservationCluster(BaseModel):
    """A cluster of semantically similar RelationshipObservation nodes."""

    cluster_id: str = Field(..., description="Unique identifier for this cluster")
    canonical_name: str = Field(..., description="Most frequent proposed_type label in the cluster")
    merged_description: str = Field(
        "", description="Concatenation of unique descriptions across observations"
    )
    observations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Raw observation dicts from Neo4j"
    )
    distinct_documents: int = Field(0, description="Number of distinct source documents")
    mean_confidence: float = Field(0.0, description="Mean confidence across observations")
    dominant_source_types: List[str] = Field(
        default_factory=list, description="Most frequent source entity types (>60%)"
    )
    dominant_target_types: List[str] = Field(
        default_factory=list, description="Most frequent target entity types (>60%)"
    )


class PromotionCandidate(BaseModel):
    """An ObservationCluster that passed statistical gates and may be promoted."""

    cluster: ObservationCluster
    llm_validated: bool = Field(False, description="True if LLM confirmed distinctness")
    llm_canonical_name: str = Field("", description="Canonical name refined by LLM")
    llm_description: str = Field("", description="Description refined by LLM")
    llm_source_types: List[str] = Field(default_factory=list)
    llm_target_types: List[str] = Field(default_factory=list)
    llm_reason: str = Field("", description="LLM reasoning for acceptance/rejection")


class RecategorizationAction(BaseModel):
    """Record of a single edge recategorization action."""

    edge_from: str = Field(..., description="Source node name")
    edge_to: str = Field(..., description="Target node name")
    old_type: str = Field(..., description="Previous relationship type (typically RELATED_TO)")
    new_type: str = Field(..., description="New relationship type after recategorization")
    confidence: float = Field(..., description="LLM confidence for this recategorization")
    action: str = Field(..., description="'committed' or 'queued_for_review'")


class EvolutionReport(BaseModel):
    """Summary report produced by the SEA after one evolution pass."""

    new_types_promoted: List[str] = Field(
        default_factory=list, description="Names of newly promoted relationship types"
    )
    candidates_pending: List[str] = Field(
        default_factory=list,
        description="Cluster names that did not yet meet the promotion threshold",
    )
    edges_recategorized_auto: int = Field(
        0, description="Number of RELATED_TO edges auto-committed to new types"
    )
    edges_queued_for_review: int = Field(
        0, description="Number of RELATED_TO edges queued for human review"
    )
    clusters_below_threshold: int = Field(
        0, description="Number of clusters that did not pass statistical gates"
    )
    recategorization_actions: List[RecategorizationAction] = Field(
        default_factory=list, description="Detailed log of recategorization actions"
    )
    generated_at: str = Field(..., description="ISO timestamp of report generation")


# ---------------------------------------------------------------------------
# LLM-facing structured output models
# ---------------------------------------------------------------------------

class SchemaValidationResult(BaseModel):
    """Structured LLM output for Phase 3 schema validation."""

    is_distinct: bool = Field(
        ..., description="True if the candidate is genuinely distinct from all existing types"
    )
    reason: str = Field(..., description="Explanation for acceptance or rejection")
    canonical_name: str = Field(
        ..., description="Recommended UPPER_SNAKE_CASE canonical name for the type"
    )
    description: str = Field(
        ..., description="One-sentence description of what this relationship represents"
    )
    source_types: List[str] = Field(
        ..., description="Allowed source entity types (e.g. ['Component', 'Architecture'])"
    )
    target_types: List[str] = Field(
        ..., description="Allowed target entity types (e.g. ['Material'])"
    )


class RecategorizationItem(BaseModel):
    """Per-edge result from the recategorization LLM call."""

    edge_id: str = Field(..., description="The edge identifier passed in the prompt")
    should_retype: bool = Field(
        ..., description="True if this edge should be recategorized to the new type"
    )
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(..., description="Brief explanation")


class RecategorizationVerification(BaseModel):
    """Batch container for recategorization LLM results."""

    results: List[RecategorizationItem] = Field(
        ..., description="One result per edge in the batch"
    )
