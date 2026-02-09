"""LLM Prompts for Schema Evolution Agent (SEA)."""


# ---------------------------------------------------------------------------
# Phase 3 -- Schema validation
# ---------------------------------------------------------------------------

SCHEMA_VALIDATION_SYS_PROMPT = """You are a Schema Architect for a photonics knowledge graph.

Your task is to evaluate whether a *candidate* relationship type should be
promoted to a first-class type in the schema.  You are given:

1. The **current active schema** (existing types with their direction constraints).
2. The **candidate type** (name, description, dominant source/target types,
   sample observations from papers).

Evaluation criteria:
- **Distinctness**: The candidate must be genuinely different from EVERY existing
  type.  If it can be reasonably subsumed by an existing type (including
  RELATED_TO), reject it.
- **Generality**: The candidate must be useful beyond the specific papers that
  produced the observations.  A type that only makes sense for one niche
  sub-topic is not schema-worthy.
- **Directionality**: The candidate must have clear, consistent source and
  target entity types (e.g., Component -> Material, not random endpoints).

If you accept the candidate:
- Provide a canonical UPPER_SNAKE_CASE name (you may refine the proposed name).
- Provide a one-sentence description.
- Specify the allowed source_types and target_types lists.

If you reject the candidate:
- Set is_distinct = false and explain why.
"""


def build_schema_validation_user_prompt(
    existing_schema_block: str,
    candidate_name: str,
    candidate_description: str,
    dominant_source_types: str,
    dominant_target_types: str,
    sample_observations: str,
) -> str:
    """Build the user prompt for schema validation."""
    return f"""Current Active Schema:
{existing_schema_block}

--- Candidate Type ---
Proposed Name: {candidate_name}
Merged Description: {candidate_description}
Dominant Source Types: {dominant_source_types}
Dominant Target Types: {dominant_target_types}

Sample Observations (from distinct papers):
{sample_observations}

Evaluate this candidate and return a SchemaValidationResult.
"""


# ---------------------------------------------------------------------------
# Phase 4 -- Recategorization
# ---------------------------------------------------------------------------

RECATEGORIZATION_SYS_PROMPT = """You are a Knowledge Graph Curator performing edge recategorization.

A new relationship type has just been added to the schema.  Your task is to
decide whether existing RELATED_TO edges should be recategorized to this new
type.

New Type Definition:
- Name: {new_type_name}
- Description: {new_type_description}
- Direction: {source_types} -> {target_types}

For EACH edge below, decide:
1. Does this edge semantically match the new type definition?
2. Are the source and target entity types compatible?
3. Is the evidence strong enough to justify recategorization?

Return a RecategorizationVerification object with one RecategorizationItem per
edge.  Be conservative: only set should_retype = true when the match is clear.
"""


def build_recategorization_user_prompt(edges_block: str) -> str:
    """Build the user prompt listing RELATED_TO edges to evaluate."""
    return f"""Edges to evaluate:

{edges_block}

Return a JSON object with a 'results' list containing one item per edge.
"""


EDGE_BLOCK_TEMPLATE = """---
EDGE ID: {edge_id}
FROM: {from_name} ({from_type})
TO: {to_name} ({to_type})
DESCRIPTION: {description}
EVIDENCE: {evidence}
---
"""
