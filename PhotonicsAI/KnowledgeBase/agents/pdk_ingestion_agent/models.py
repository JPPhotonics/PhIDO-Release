"""Pydantic models for PDK ingestion pipeline data structures."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PortInfo(BaseModel):
    """Physical port metadata from GDSFactory instantiation."""

    name: str
    x: float
    y: float
    orientation: float
    width: Optional[float] = None
    layer: Optional[str] = None


class NumericSpec(BaseModel):
    """A parsed numeric specification from a component docstring."""

    field_name: str
    value: float
    unit: str
    raw_text: str


class PDKCellNode(BaseModel):
    """Full schema for a PDK_Cell node in Neo4j."""

    module_name: str = Field(..., description="Python module name (filename without .py)")
    display_name: str = Field(..., description="Human-readable name from docstring metadata")
    description: str = Field(default="", description="Component description")
    pdk_name: str = Field(..., description="PDK identifier (e.g. 'DemoPDK')")
    pdk_version: str = Field(default="1.0.0", description="PDK version string")
    ports: str = Field(default="unknown", description="Port configuration string (e.g. '2x2')")
    port_details: list[PortInfo] = Field(default_factory=list, description="Introspected port positions")
    labels: list[str] = Field(default_factory=list, description="NodeLabels from docstring")
    aka: str = Field(default="", description="Comma-separated aliases")
    technology: str = Field(default="", description="Technology field from docstring")
    parameters: dict = Field(default_factory=dict, description="Args from docstring")
    numeric_specs: list[NumericSpec] = Field(default_factory=list, description="Parsed numeric specs")
    dx_um: Optional[float] = Field(None, description="Footprint width in microns")
    dy_um: Optional[float] = Field(None, description="Footprint height in microns")
    is_flattened: bool = Field(False, description="Whether the cell calls .flatten()")
    is_primitive: bool = Field(False, description="True if component has no DesignLibrary imports")
    has_simulation_model: bool = Field(False, description="True if a get_model function exists")
    source_file: str = Field(default="", description="Path to the source .py file")
    topology_template: Optional[dict] = Field(None, description="ArchitectureTemplate as JSON")


class LayoutInstantiation(BaseModel):
    """An instantiation of a sub-component found via AST analysis."""

    var_name: str
    module_name: str
    call_args: dict = Field(default_factory=dict)


class ExplicitConnection(BaseModel):
    """An explicit port-level connection found via AST analysis."""

    from_var: str
    from_port: str
    to_var: str
    to_port: str


class RoleAssignment(BaseModel):
    """A keyword argument role assignment in a GDSFactory compound call."""

    gf_compound: str = Field(..., description="GDSFactory compound function (e.g. 'mzi2x2_2x2')")
    keyword_roles: dict[str, str] = Field(default_factory=dict, description="role_name -> module_name")


class ExternalPort(BaseModel):
    """An external port mapping found via AST analysis."""

    external_port: str
    instance_var: str
    instance_port: str


class LayoutComposition(BaseModel):
    """Result of AST-based layout composition analysis."""

    module_name: str
    instantiations: list[LayoutInstantiation] = Field(default_factory=list)
    explicit_connections: list[ExplicitConnection] = Field(default_factory=list)
    role_assignments: list[RoleAssignment] = Field(default_factory=list)
    external_ports: list[ExternalPort] = Field(default_factory=list)
    imported_modules: list[str] = Field(default_factory=list, description="DesignLibrary modules imported")
    composition_pattern: str = Field(
        default="unknown",
        description="One of: explicit, delegated, wrapper, primitive",
    )


class ImplementsResolution(BaseModel):
    """LLM output for IMPLEMENTS matching."""

    candidate_name: str
    does_implement: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


class RelationshipResolution(BaseModel):
    """Generic LLM output for function/principle/property relationship matching."""

    candidate_name: str
    is_related: bool
    confidence: float = Field(ge=0.0, le=1.0)
    edge_type: str
    reasoning: str


class BatchRelationshipResolution(BaseModel):
    """Batch output from LLM relationship resolution."""

    results: list[RelationshipResolution]


class BatchImplementsResolution(BaseModel):
    """Batch output from LLM IMPLEMENTS resolution."""

    results: list[ImplementsResolution]


class PDKIngestionReport(BaseModel):
    """Summary report of a PDK ingestion run."""

    pdk_name: str
    pdk_version: str
    cells_discovered: int = 0
    cells_created: int = 0
    cells_updated: int = 0
    edges_created: int = 0
    implements_edges: int = 0
    composed_of_edges: int = 0
    performs_function_edges: int = 0
    fabricated_with_edges: int = 0
    exhibits_edges: int = 0
    enrichments_propagated: int = 0
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
