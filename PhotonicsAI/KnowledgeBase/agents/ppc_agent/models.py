"""Pydantic models for PPC Agent structured outputs."""

from typing import List, Optional
from pydantic import BaseModel, Field


class RawEntity(BaseModel):
    """Raw entity extracted from text."""
    name: str = Field(..., description="The extracted entity name")
    entity_type: str = Field(..., description="Type: Component, Architecture, Property, Design_Function, or Physical_Principle")
    context: Optional[str] = Field(None, description="Context where the entity was found")


class KBUpdateSuggestion(BaseModel):
    """Suggested KB augmentation for an entity that already exists in the KB."""

    kb_name: str = Field(..., description="The existing KB entity name to update")
    additions: List[str] = Field(
        default_factory=list,
        description="Bullet-point list of new facts from the input document that are NOT already present in the KB entry",
    )
    updated_description: Optional[str] = Field(
        None,
        description="Optional rewritten KB description that merges existing KB description + new additions",
    )
    evidence_quotes: List[str] = Field(
        default_factory=list,
        description="1-5 short verbatim quotes from the input document supporting the additions",
    )
    key_metrics: List[str] = Field(
        default_factory=list,
        description="Optional metric statements extracted from the input document (value/unit) relevant to the additions",
    )


class NormalizedEntity(BaseModel):
    """Entity normalized to knowledge base."""
    raw_name: str = Field(..., description="Original extracted name")
    kb_name: str = Field(..., description="Matched knowledge base entity name")
    similarity: float = Field(..., description="Similarity score (0.0-1.0)")
    entity_type: str = Field(..., description="Entity type")
    collection: str = Field(..., description="KB collection name")
    components: List[str] = Field(
        default_factory=list,
        description="For Architecture entities: component make-up as extracted from the input document; empty for non-architectures",
    )
    connectivity: List[str] = Field(
        default_factory=list,
        description="For Architecture entities: how components are connected (1+ statements); empty for non-architectures",
    )
    exact_match_override: bool = Field(
        False,
        description="True if similarity was forced to 1.0 due to raw_name == kb_name (case-insensitive exact match override)",
    )
    context_pack: Optional[str] = Field(
        None,
        description="Complete evidence pack (mentions, neighbor entities, sections) used for context-aware merging",
    )
    top_candidates: List[dict] = Field(
        default_factory=list,
        description="Top KB matches found during normalization"
    )
    kb_update_suggestion: Optional[KBUpdateSuggestion] = Field(
        None,
        description="If exact_match_override is True, optional suggestion for what new context from the input document should be added to the KB entry",
    )


class NewConcept(BaseModel):
    """Novel entity to be added to knowledge base."""
    name: str = Field(..., description="The new entity name")
    entity_type: str = Field(..., description="Type: Component, Architecture, Property, Design_Function, or Physical_Principle")
    description: Optional[str] = Field(None, description="Description from paper")
    context: Optional[str] = Field(None, description="Context where found")
    evidence_quotes: List[str] = Field(default_factory=list, description="Verbatim supporting quotes from the paper")
    key_metrics: List[str] = Field(default_factory=list, description="Extracted metric statements (value/unit) for this entity")
    related_entities: List[str] = Field(default_factory=list, description="Other entities co-mentioned in the same context")
    components: List[str] = Field(
        default_factory=list,
        description="For Architecture entities: component make-up as extracted from the input document; empty for non-architectures",
    )
    connectivity: List[str] = Field(
        default_factory=list,
        description="For Architecture entities: how components are connected (1+ statements); empty for non-architectures",
    )
    top_candidates: List[dict] = Field(
        default_factory=list,
        description="Top KB matches found during normalization"
    )


class PPCResult(BaseModel):
    """Final PPC Agent output."""
    known_entities: List[NormalizedEntity] = Field(default_factory=list, description="Entities matched to existing KB")
    new_concepts: List[NewConcept] = Field(default_factory=list, description="New entities to add to KB")
    
    def to_json(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "known_entities": [e.model_dump() for e in self.known_entities],
            "new_concepts": [c.model_dump() for c in self.new_concepts]
        }

