"""Dynamic Schema Registry backed by Neo4j.

Stores all relationship types as SchemaRelationType nodes and all novel
relationship observations as RelationshipObservation nodes.  Provides the
single source of truth consumed by VSA, DIA, SEA, and the review-queue UI.
"""

import datetime
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from neo4j import Driver


class SchemaRegistry:
    """Dynamic schema registry backed by Neo4j SchemaRelationType nodes."""

    # Seed relationship definitions ------------------------------------------
    # These mirror the original hardcoded schema from config.py / dia_agent.py
    _SEED_TYPES: List[Dict[str, Any]] = [
        {
            "name": "PERFORMS_FUNCTION",
            "description": "Links a Component, Architecture, or PDK_Cell to a Design_Function it performs",
            "allowed_source_types": ["Component", "Architecture", "PDK_Cell"],
            "allowed_target_types": ["Design_Function"],
        },
        {
            "name": "BASED_ON_PRINCIPLE",
            "description": "Links a Component, Architecture, or PDK_Cell to a Physical_Principle it is based on",
            "allowed_source_types": ["Component", "Architecture", "PDK_Cell"],
            "allowed_target_types": ["Physical_Principle"],
        },
        {
            "name": "HAS_PROPERTY",
            "description": "Links a Component, Architecture, or PDK_Cell to a Property it possesses",
            "allowed_source_types": ["Component", "Architecture", "PDK_Cell"],
            "allowed_target_types": ["Property"],
        },
        {
            "name": "USES_COMPONENT",
            "description": "Links an Architecture to a Component it uses",
            "allowed_source_types": ["Architecture"],
            "allowed_target_types": ["Component"],
        },
        {
            "name": "RELATED_TO",
            "description": "Generic relationship between any two entities (use sparingly)",
            "allowed_source_types": ["*"],
            "allowed_target_types": ["*"],
        },
        {
            "name": "EXTRACTED_FROM",
            "description": "Provenance link from an entity to the Document it was extracted from",
            "allowed_source_types": ["*"],
            "allowed_target_types": ["Document"],
        },
        {
            "name": "IMPLEMENTS",
            "description": "Links a concrete PDK cell to the abstract component it implements",
            "allowed_source_types": ["PDK_Cell"],
            "allowed_target_types": ["Component"],
        },
        {
            "name": "COMPOSED_OF",
            "description": "Layout composition: a PDK cell contains instances of another PDK cell",
            "allowed_source_types": ["PDK_Cell"],
            "allowed_target_types": ["PDK_Cell"],
        },
        {
            "name": "EXHIBITS",
            "description": "A PDK cell exhibits a measured property with a specific value",
            "allowed_source_types": ["PDK_Cell"],
            "allowed_target_types": ["Property"],
        },
        {
            "name": "FABRICATED_WITH",
            "description": "Links a PDK cell to the physical principle/technology it uses",
            "allowed_source_types": ["PDK_Cell"],
            "allowed_target_types": ["Physical_Principle"],
        },
        {
            "name": "SUPERSEDES",
            "description": "Version history link from current to previous PDK cell snapshot",
            "allowed_source_types": ["PDK_Cell"],
            "allowed_target_types": ["PDK_Cell_History"],
        },
    ]

    _CACHE_TTL_SECONDS = 30.0

    def __init__(self, driver: Driver) -> None:
        self.driver = driver
        self._cache: Optional[List[Dict[str, Any]]] = None
        self._cache_time: float = 0.0

    # ------------------------------------------------------------------
    # Seed initialisation
    # ------------------------------------------------------------------

    def initialize_seed_schema(self) -> None:
        """Idempotent MERGE of seed relationship types into Neo4j."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.driver.session() as session:
            for seed in self._SEED_TYPES:
                session.run(
                    """
                    MERGE (s:SchemaRelationType {name: $name})
                    ON CREATE SET
                        s.description        = $description,
                        s.allowed_source_types = $allowed_source_types,
                        s.allowed_target_types = $allowed_target_types,
                        s.is_seed            = true,
                        s.status             = 'active',
                        s.observation_count  = 0,
                        s.source_documents   = [],
                        s.promoted_at        = null,
                        s.created_at         = $now
                    """,
                    name=seed["name"],
                    description=seed["description"],
                    allowed_source_types=seed["allowed_source_types"],
                    allowed_target_types=seed["allowed_target_types"],
                    now=now,
                )
        self._invalidate_cache()

    # ------------------------------------------------------------------
    # Active type queries (cached)
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._cache = None
        self._cache_time = 0.0

    def get_active_edge_types(self) -> List[Dict[str, Any]]:
        """Return all active SchemaRelationType nodes (TTL-cached)."""
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_time) < self._CACHE_TTL_SECONDS:
            return self._cache

        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:SchemaRelationType {status: 'active'})
                RETURN s
                ORDER BY s.name
                """
            )
            types: List[Dict[str, Any]] = []
            for record in result:
                node = record["s"]
                types.append({
                    "name": node.get("name"),
                    "description": node.get("description", ""),
                    "allowed_source_types": list(node.get("allowed_source_types", [])),
                    "allowed_target_types": list(node.get("allowed_target_types", [])),
                    "is_seed": node.get("is_seed", False),
                    "status": node.get("status", "active"),
                    "observation_count": node.get("observation_count", 0),
                })

        self._cache = types
        self._cache_time = time.monotonic()
        return types

    def get_edge_rules(self) -> Dict[str, Tuple[List[str], List[str]]]:
        """Build edge_rules dict: {type_name: ([source_types], [target_types])}.

        Replaces the hardcoded ``edge_rules`` in DIA and the schema validation
        in VSA.  Wildcard ``*`` entries (RELATED_TO, EXTRACTED_FROM) are
        excluded since they don't constrain direction.
        """
        rules: Dict[str, Tuple[List[str], List[str]]] = {}
        for t in self.get_active_edge_types():
            src = t["allowed_source_types"]
            tgt = t["allowed_target_types"]
            if "*" in src or "*" in tgt:
                continue  # skip wildcards
            rules[t["name"]] = (src, tgt)
        return rules

    def get_edge_constraints(self) -> Dict[str, List[str]]:
        """Build edge_constraints dict: {entity_type: [target_entity_types]}.

        Inverts the active edge rules so that, for each entity type, we know
        which target types are reachable.  Replaces the hardcoded
        ``edge_constraints`` in DIA.
        """
        constraints: Dict[str, set] = {}
        for _name, (sources, targets) in self.get_edge_rules().items():
            for src in sources:
                constraints.setdefault(src, set()).update(targets)
            # Bi-directional potential: targets may also reach sources
            for tgt in targets:
                constraints.setdefault(tgt, set()).update(sources)
        return {k: sorted(v) for k, v in constraints.items()}

    def get_active_type_names(self) -> List[str]:
        """Return sorted list of active relationship type names."""
        return [t["name"] for t in self.get_active_edge_types()]

    def get_core_rel_types(self) -> List[str]:
        """Return relationship type names suitable for graph traversal.

        Excludes EXTRACTED_FROM which is a provenance link, not a domain
        relationship.
        """
        return [
            t["name"]
            for t in self.get_active_edge_types()
            if t["name"] != "EXTRACTED_FROM"
        ]

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def get_prompt_schema_block(self) -> str:
        """Generate the 'Allowed Edge Types and Rules' text block for LLM prompts."""
        lines: List[str] = []
        for t in self.get_active_edge_types():
            src = t["allowed_source_types"]
            tgt = t["allowed_target_types"]
            src_str = "/".join(src) if "*" not in src else "Any"
            tgt_str = "/".join(tgt) if "*" not in tgt else "Any"
            tag = ""
            if not t.get("is_seed", True):
                tag = "  [DISCOVERED]"
            lines.append(f"- {t['name']}: {src_str} -> {tgt_str}{tag}")
            if t.get("description"):
                lines.append(f"  Description: {t['description']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Observation recording
    # ------------------------------------------------------------------

    def record_observation(
        self,
        proposed_type: str,
        description: str,
        from_entity: str,
        from_type: str,
        to_entity: str,
        to_type: str,
        evidence: List[str],
        confidence: float,
        document_key: str,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Create a RelationshipObservation node in Neo4j.

        Parameters
        ----------
        embedding : list[float] | None
            Pre-computed embedding of ``"{proposed_type}: {description}"``.
            If ``None`` the observation is stored without an embedding and the
            SEA will need to generate one at clustering time.

        Returns
        -------
        str
            The UUID of the created observation node.
        """
        obs_id = str(uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        props: Dict[str, Any] = {
            "id": obs_id,
            "proposed_type": proposed_type,
            "description": description,
            "from_entity": from_entity,
            "from_entity_type": from_type,
            "to_entity": to_entity,
            "to_entity_type": to_type,
            "evidence_quotes": evidence,
            "confidence": confidence,
            "source_document": document_key,
            "observed_at": now,
            "cluster_id": None,
        }

        query = """
        CREATE (o:RelationshipObservation)
        SET o = $props
        """
        params: Dict[str, Any] = {"props": props}

        # Embeddings must be set separately because Neo4j list-of-float handling
        # sometimes differs from map-property merging.
        if embedding:
            query = """
            CREATE (o:RelationshipObservation)
            SET o = $props, o.embedding = $embedding
            """
            params["embedding"] = embedding

        try:
            with self.driver.session() as session:
                session.run(query, **params)
        except Exception as e:
            print(f"  [SchemaRegistry] Failed to record observation: {e}")

        return obs_id

    # ------------------------------------------------------------------
    # Promotion
    # ------------------------------------------------------------------

    def promote_candidate(
        self,
        name: str,
        description: str,
        allowed_sources: List[str],
        allowed_targets: List[str],
    ) -> str:
        """Promote a candidate relationship type to active status.

        Creates (or updates) a SchemaRelationType node with status='active'.
        Returns the name of the promoted type.
        """
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self.driver.session() as session:
            session.run(
                """
                MERGE (s:SchemaRelationType {name: $name})
                SET s.description          = $description,
                    s.allowed_source_types  = $allowed_source_types,
                    s.allowed_target_types  = $allowed_target_types,
                    s.is_seed              = false,
                    s.status               = 'active',
                    s.promoted_at          = $now
                ON CREATE SET
                    s.observation_count    = 0,
                    s.source_documents     = [],
                    s.created_at           = $now
                """,
                name=name,
                description=description,
                allowed_source_types=allowed_sources,
                allowed_target_types=allowed_targets,
                now=now,
            )
        self._invalidate_cache()
        return name

    # ------------------------------------------------------------------
    # Candidate helpers
    # ------------------------------------------------------------------

    def get_candidate_type_names(self) -> List[str]:
        """Return names of status='candidate' types (for debugging/reporting)."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (s:SchemaRelationType {status: 'candidate'}) RETURN s.name AS name"
            )
            return [record["name"] for record in result]

    def is_active_type(self, name: str) -> bool:
        """Check if a relationship type name is currently active (case-insensitive)."""
        active = self.get_active_type_names()
        return name.upper() in {n.upper() for n in active}
