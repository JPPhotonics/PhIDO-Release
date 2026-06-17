"""DesignIntent and disambiguation models — structured output contracts for the interpreter agent."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requirement traceability models
# ---------------------------------------------------------------------------

class UserRequirement(BaseModel):
    """A single design requirement extracted from the user's prompt."""
    id: str = Field(..., description="Sequential identifier (R1, R2, ...)")
    category: str = Field(
        ...,
        description="'functional' (what it does), 'structural' (components/topology), "
                    "'performance' (quantitative targets), or 'constraint' (fixed params)",
    )
    description: str = Field(..., description="Concise requirement statement")
    source_span: str = Field(..., description="Verbatim text from user input")
    priority: str = Field(
        ...,
        description="'explicit' (user stated directly) or 'inferred' (agent deduced from context)",
    )


class RequirementManifest(BaseModel):
    """Structured list of all user requirements for a design session."""
    requirements: list[UserRequirement] = Field(default_factory=list)
    original_prompt: str = Field("", description="Raw user input, preserved verbatim")


class RequirementTrace(BaseModel):
    """Maps a single requirement to the design elements that address it."""
    requirement_id: str = Field(..., description="References UserRequirement.id")
    satisfied_by: list[str] = Field(
        default_factory=list,
        description="Component IDs, spec keys, or 'architecture' that address this",
    )
    satisfaction_type: str = Field(
        ...,
        description="'direct', 'partial', 'implicit', or 'unaddressed'",
    )
    notes: str = Field("", description="How/why this requirement is or isn't met")


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
    component_type: Optional[str] = Field(
        None,
        description="Canonical device type: 'splitter', 'combiner', 'mzm', 'phase_shifter', "
                    "'ring_resonator', 'waveguide', 'coupler', 'crossing', 'detector', 'grating_coupler'",
    )
    sub_type: Optional[str] = Field(
        None,
        description="Sub-type qualifier: 'mmi', 'directional_coupler', 'add_drop', "
                    "'all_pass', '90_degree', 'balanced', 'unbalanced', 'heater', 'pin'",
    )
    pdk_module: Optional[str] = Field(
        None,
        description="PDK module name if already grounded (e.g. 'mzi_2x2_pn_diode'). "
                    "Set by the iterative builder; None for single-shot extraction.",
    )
    # Provenance
    confidence: float = Field(1.0, description="Extraction confidence 0.0-1.0")
    source_span: Optional[str] = Field(None, description="Verbatim text this was extracted from")

class Connection(BaseModel):
    """A hybrid connection: component IDs + natural language description."""
    from_component: str = Field(..., description="Source component id (e.g. C1)")
    to_component: str = Field(..., description="Target component id (e.g. C2)")
    from_port: Optional[str] = Field(None, description="Source port name (e.g. 'o3'). Set by iterative builder.")
    to_port: Optional[str] = Field(None, description="Target port name (e.g. 'o2'). Set by iterative builder.")
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
    architecture_type: Optional[str] = Field(
        None,
        description="Primary architecture: 'mzi', 'splitter_tree', 'benes', 'clements', "
                    "'reck', 'qpsk', 'wdm_demux', 'wdm_mux', 'crossbar', 'spanke', 'ring_filter'",
    )
    n_value: Optional[int] = Field(
        None,
        description="Primary scaling parameter (output count, port size, channel count)",
    )
    requirement_manifest: Optional[RequirementManifest] = Field(
        None, description="Structured requirements extracted from user prompt",
    )
    requirement_traces: list[RequirementTrace] = Field(
        default_factory=list,
        description="Maps each requirement to the design elements that satisfy it",
    )
    unaddressed_requirements: list[str] = Field(
        default_factory=list,
        description="Requirement IDs the interpreter could not address",
    )
    
    def summary(self) -> dict:
        """Clean view - no provenance."""
        result = {
            "title": self.title,
            "brief_summary": self.brief_summary,
            "components": [
                {"id": c.id, "description": c.description,
                 "port_config": c.port_config, "specs": {s.key: s.value for s in c.specs},
                 "role": c.role, "component_type": c.component_type, "sub_type": c.sub_type}
                for c in self.components
            ],
            "connections": [
                {"from": cn.from_component, "to": cn.to_component,
                 "description": cn.description}
                for cn in self.connections
            ],
            "ambiguities": self.ambiguities,
        }
        if self.architecture_type:
            result["architecture_type"] = self.architecture_type
        if self.n_value is not None:
            result["n_value"] = self.n_value
        if self.requirement_traces:
            result["requirement_traces"] = [
                t.model_dump() for t in self.requirement_traces
            ]
        if self.unaddressed_requirements:
            result["unaddressed_requirements"] = self.unaddressed_requirements
        return result
        
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

class SettingEntry(BaseModel):
    """A single GDSFactory setting as a key-value pair."""
    key: str
    value: str

class ComponentMapping(BaseModel):
    """Maps a ComponentIntent to a concrete PDK module."""
    component_id: str          # C1, C2, ... (matches ComponentIntent.id)
    pdk_module: str            # Exact PDK module name, e.g. "mzi_2x2_heater_tin_cband"
    match_quality: str         # "exact", "partial", "poor"
    port_config: str           # Actual port config from PDK, e.g. "2x2"
    resolved_settings: list[SettingEntry]  # GDSFactory default settings
    user_overrides: list[SettingEntry]     # User specs mapped to GDSFactory param keys
    notes: list[str]           # Any warnings or assumptions
    
class ComponentSelectionLLM(BaseModel):
    """LLM-facing subset — does not include the original DesignIntent."""
    mappings: list[ComponentMapping]
    unmapped: list[str]
    ambiguities: list[str]

class ComponentSelection(BaseModel):
    """Complete result of the component selection stage."""
    design_intent: DesignIntent           # Original input (passed through)
    mappings: list[ComponentMapping]      # One per component
    unmapped: list[str]                   # Component IDs with no good match
    ambiguities: list[str]               # Selection-level ambiguities
    
class CircuitNode(BaseModel):
    component: str             # PDK module name
    ports: str                 # e.g. "2x2"
    params: dict               # GDSFactory settings
    footprint: tuple[float, float]  # (dx, dy) in microns
    placement: Optional[dict]  # {"x": float, "y": float, "rotation": float}

class CircuitEdge(BaseModel):
    link: str                  # e.g. "C1,o3: C2,o1"

class CircuitDSL(BaseModel):
    """Fabrication-ready circuit representation."""
    title: str
    description: str
    nodes: dict[str, CircuitNode]
    edges: dict[str, CircuitEdge]
    ports: dict[str, str]      # External circuit ports
    
class ComplianceResult(BaseModel):
    """LLM judgement of whether a single component mapping is functionally faithful."""
    component_id: str
    compliant: bool
    reason: str
    suggested_alternative: Optional[str] = None

class ComplianceVerdict(BaseModel):
    """Batch compliance verdict for all component mappings."""
    results: list[ComplianceResult]

# ---------------------------------------------------------------------------
# Tiered schematic feedback models
# ---------------------------------------------------------------------------

class EdgePatch(BaseModel):
    """A single edge addition, removal, or replacement in the schematic DOT."""
    action: Literal["add", "remove", "replace"]
    src_node: Optional[str] = None
    src_port: Optional[str] = None
    tgt_node: Optional[str] = None
    tgt_port: Optional[str] = None
    # Only used for action=="replace" — identifies the old edge to remove
    old_src_node: Optional[str] = None
    old_src_port: Optional[str] = None
    old_tgt_node: Optional[str] = None
    old_tgt_port: Optional[str] = None

class ComponentSwap(BaseModel):
    """Request to swap a specific component to a different PDK module."""
    component_id: str = Field(..., description="Component ID (e.g. C1)")
    new_module_hint: str = Field(
        ...,
        description="Search term or exact PDK module name for the replacement",
    )

class FeedbackClassification(BaseModel):
    """LLM-produced classification of user schematic feedback."""
    tier: Literal["edge_edit", "component_swap", "architectural"]
    edge_patches: list[EdgePatch] = Field(default_factory=list)
    component_swaps: list[ComponentSwap] = Field(default_factory=list)
    reasoning: str


class PipelineFeedback(BaseModel):
    source_phase: str          # "component_selection" | "schematic_builder"
    target_phase: str          # "interpreter" | "component_selector" | "user" | "local"
    severity: str              # "fundamental" | "major" | "minor" | "needs_user"
    description: str
    affected_components: list[str]
    suggested_action: str
    context: dict              # Evidence (tool results, error details)
    
class StageResult(BaseModel):
    success: bool
    output: Any
    feedback: list[PipelineFeedback] = Field(default_factory=list)
    
    def has_feedback_for(self, target: str) -> bool:
        return any(f.target_phase == target for f in self.feedback)
    
    def get_feedback_for(self, target: str) -> list[PipelineFeedback]:
        return [f for f in self.feedback if f.target_phase == target]
    
