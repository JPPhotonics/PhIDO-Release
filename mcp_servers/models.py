"""DesignIntent and disambiguation models — structured output contracts for the interpreter agent."""

from typing import Any, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Disambiguation models (Phase 1.75)
# ---------------------------------------------------------------------------

class ClarificationQuestion(BaseModel):
    """A single question the agent wants the user to answer before structuring."""
    question: str = Field(..., description="The question to ask the user")
    context: str = Field(
        ...,
        description="Why this matters — what the agent found in tool results that raised the question",
    )
    options: list[str] = Field(
        default_factory=list,
        description="Suggested answer choices (empty list if open-ended)",
    )
    default: str = Field(
        ...,
        description="What the agent would assume if the user doesn't answer",
    )
    priority: str = Field(
        ...,
        description="'critical' (design blocked without answer) or 'helpful' (improves accuracy but has a reasonable default)",
    )


class ClarificationRequest(BaseModel):
    """Structured set of clarification questions from the agent."""
    questions: list[ClarificationQuestion] = Field(
        default_factory=list,
        description="Questions the agent wants answered",
    )
    ready_to_proceed: bool = Field(
        ...,
        description="True if no critical unknowns — agent can proceed with defaults",
    )


# ---------------------------------------------------------------------------
# Design intent models
# ---------------------------------------------------------------------------

class SpecEntry(BaseModel):
    """A single specification key-value pair."""
    key: str = Field(..., description="Specification name, e.g. 'arm_length', 'bandwidth'")
    value: str = Field(..., description="Specification value with units, e.g. '150µm', '10nm'")

class ComponentIntent(BaseModel):
    """A structured photonic component extracted from user input."""
    id: str = Field(..., description="Instance identifier (C1, C2, ...)")
    description: str = Field(..., description="Natural language description")
    port_config: Optional[str] = Field(None, description="Port config e.g. '2x2', '1x4' - only if stated or confidently inferred")
    specs: list[SpecEntry] = Field(default_factory=list, description="Extracted specifications as key-value pairs")
    role: Optional[str] = Field(None, description="Functional role: modulator, splitter, detector, filter, coupler, etc.")
    # Provenance
    confidence: float = Field(1.0, description="Extraction confidence 0.0-1.0")
    source_span: Optional[str] = Field(None, description="Verbatim text this was extracted from")

class Connection(BaseModel):
    """A hybrid connection: component IDs + natural language description."""
    from_component: str = Field(..., description="Source component id (e.g. C1)")
    to_component: str = Field(..., description="Target component id (e.g. C2)")
    description: str = Field(..., description="How/why they connect, in natural language")
    # Provenance
    confidence: float = Field(1.0, description="Extraction confidence 0.0-1.0")
    source_span: Optional[str] = Field(None, description="Verbatim text this was extracted from")

class DesignIntent(BaseModel):
    """Structured output of the interpreter agent."""
    title: str = Field(..., description="Concise title for the circuit")
    brief_summary: str = Field(..., description="<150 word summary of the input")
    components: list[ComponentIntent] = Field(..., description="Extracted components, one entry per instance")
    connections: list[Connection] = Field(default_factory=list, description="Pairwise connections between components")
    ambiguities: list[str] = Field(default_factory=list, description="Noted ambiguities or assumptions made")
    
    def summary(self) -> dict:
        """Clean view - no provenance."""
        return {
            "title": self.title,
            "brief_summary": self.brief_summary,
            "components": [
                {"id": c.id, "description": c.description,
                 "port_config": c.port_config, "specs": {s.key: s.value for s in c.specs}, "role": c.role}
                for c in self.components
            ],
            "connections": [
                {"from": cn.from_component, "to": cn.to_component,
                 "description": cn.description}
                for cn in self.connections
            ],
            "ambiguities": self.ambiguities,
        }
        
    def full(self) -> dict:
        """Full view - includes provenance."""
        return self.model_dump()
    
    def to_pretemplate(self) -> dict:
        """Backwards-compatible conversion to the legacy pretemplate format."""
        return {
            "title": self.title,
            "brief_summary": self.brief_summary,
            "components_list": [c.description for c in self.components],
            "circuit_instructions": "; ".join(
                f"{cn.from_component} → {cn.to_component}: {cn.description}"
                for cn in self.connections
            ) if self.connections else "",
        }

    def to_dot(self) -> str:
        """Generate a Graphviz DOT string for a preschematic graph.

        Components become nodes (labeled with id, role, port_config).
        Connections become undirected edges. Layout is left-to-right.
        """
        lines = ["graph G {", "    rankdir=LR;", "    node [shape=box, style=rounded];", ""]

        # Nodes
        for c in self.components:
            parts = [c.id]
            if c.role:
                parts.append(c.role.title())
            if c.port_config:
                parts.append(f"({c.port_config})")
            label = "\\n".join(parts)
            # Escape double-quotes inside the label
            label = label.replace('"', '\\"')
            lines.append(f'    {c.id} [label="{label}"];')

        lines.append("")

        # Edges
        component_ids = {c.id for c in self.components}
        for cn in self.connections:
            if cn.from_component in component_ids and cn.to_component in component_ids:
                lines.append(f"    {cn.from_component} -- {cn.to_component};")

        lines.append("}")
        return "\n".join(lines)
