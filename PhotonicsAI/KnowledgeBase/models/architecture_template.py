"""Structured representation of a photonic architecture's internal topology.

An ArchitectureTemplate captures the *roles* that sub-components play inside a
composite architecture, the port-level connections between those roles, and
optional scaling / variant metadata.  It is the target output of the dedicated
architecture decomposition phase (PPC Phase A.7) and can later be persisted to
the Knowledge Graph as USES_COMPONENT edges with role/port properties.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ComponentRole(BaseModel):
    """A named functional slot inside an architecture.

    Example: an MZI has two splitter roles ("input_splitter", "output_combiner")
    that are both of component_type "directional coupler" but serve different
    purposes.
    """

    role_id: str = Field(
        ...,
        description="Unique role identifier within the template (e.g. 'input_splitter', 'phase_arm_1')",
    )
    component_type: str = Field(
        ...,
        description="Component type that fills this role (e.g. 'directional coupler')",
    )
    port_config: Optional[str] = Field(
        None,
        description="Port configuration hint (e.g. '2x2', '1x2')",
    )
    quantity: int = Field(
        1,
        description="How many instances of this role exist (e.g. 7 splitters in a 1x8 tree)",
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative component types that can fill this role (e.g. ['Y-branch'] for a splitter role)",
    )


class AbstractConnection(BaseModel):
    """A directed port-level link between two component roles."""

    from_role: str = Field(..., description="Source role_id")
    from_port: str = Field(..., description="Source port name (e.g. 'out1', 'cross')")
    to_role: str = Field(..., description="Target role_id")
    to_port: str = Field(..., description="Target port name (e.g. 'in', 'add')")


class ArchitectureTemplate(BaseModel):
    """Complete structural blueprint of an architecture."""

    name: str = Field(..., description="Architecture name")
    topology_class: Optional[str] = Field(
        None,
        description="Topology family (e.g. 'tree', 'cascade', 'mesh', 'ring-bus')",
    )
    component_roles: list[ComponentRole] = Field(
        ...,
        description="Functional roles and the component types that fill them",
    )
    connections: list[AbstractConnection] = Field(
        ...,
        description="Port-level links between roles",
    )
    scaling_rules: Optional[dict] = Field(
        None,
        description="Parameterised scaling (e.g. {'N': 'num_channels', 'stages': 'log2(N)'})",
    )
    variants: list[str] = Field(
        default_factory=list,
        description="Known variant names or alternate configurations",
    )
    source: Optional[str] = Field(
        None,
        description="Provenance tag (e.g. 'paper:<doi>', 'kg_prior', 'pdk')",
    )
    confidence: float = Field(
        1.0,
        description="Extraction confidence (0.0–1.0); lower when inferred from sparse evidence",
    )
