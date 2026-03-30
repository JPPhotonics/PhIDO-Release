"""Main PDK Ingestion Agent — orchestrates the full pipeline.

Phases:
1. Parse docstrings + instantiate for ports
2. AST-analyze topology / LLM complete delegated patterns
3. Create PDK_Cell nodes with embeddings
4. Resolve relationships (IMPLEMENTS, PERFORMS_FUNCTION, FABRICATED_WITH, EXHIBITS)
5. Bidirectional enrichment
6. Versioning (snapshot existing nodes before update)
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from .ast_analyzer import analyze_cell_file
from .docstring_parser import introspect_ports, parse_component_file
from .enrichment import (
    enrich_components_from_pdk,
    enrich_pdk_from_components,
    upgrade_architecture_templates,
)
from .models import PDKCellNode, PDKIngestionReport
from .relationship_resolver import (
    complete_topology,
    resolve_exhibits,
    resolve_fabricated_with,
    resolve_implements,
    resolve_performs_function,
)

if TYPE_CHECKING:
    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient


# Confidence thresholds
AUTO_COMMIT_THRESHOLD = 0.8
REVIEW_THRESHOLD = 0.5


class PDKIngestionAgent:
    """Orchestrates PDK component ingestion into the Knowledge Graph."""

    def __init__(
        self,
        kb_client: "Neo4jClient",
        pdk_name: str = "DemoPDK",
        pdk_version: str = "1.0.0",
        pdk_dir: Optional[Path] = None,
    ):
        self.kb_client = kb_client
        self.pdk_name = pdk_name
        self.pdk_version = pdk_version
        self.pdk_dir = pdk_dir or (
            Path(__file__).parent.parent.parent / "DesignLibrary"
        )

    def ingest(self) -> PDKIngestionReport:
        """Run the full ingestion pipeline."""
        report = PDKIngestionReport(pdk_name=self.pdk_name, pdk_version=self.pdk_version)

        if not self.kb_client._connected:
            self.kb_client.connect()

        # Discover component files
        py_files = sorted(
            f for f in self.pdk_dir.glob("*.py")
            if f.name != "__init__.py"
        )
        report.cells_discovered = len(py_files)
        print(f"\nPDK Ingestion: Found {len(py_files)} component files in {self.pdk_dir}")

        # Phase 1: Parse + introspect all cells
        cells: list[PDKCellNode] = []
        for py_file in py_files:
            try:
                cell = parse_component_file(py_file, self.pdk_name, self.pdk_version)
                cells.append(cell)
            except Exception as e:
                msg = f"Failed to parse {py_file.name}: {e}"
                print(f"  [Error] {msg}")
                report.errors.append(msg)

        # Port introspection (batch, with caching)
        print("PDK Ingestion: Introspecting ports...")
        for cell in cells:
            try:
                ports, dx, dy = introspect_ports(cell.module_name)
                cell.port_details = ports
                cell.dx_um = dx
                cell.dy_um = dy
            except Exception as e:
                msg = f"Port introspection failed for {cell.module_name}: {e}"
                print(f"  [Warn] {msg}")
                report.warnings.append(msg)

        # Phase 2: AST topology extraction + LLM completion
        print("PDK Ingestion: Analyzing layout composition...")
        for cell in cells:
            try:
                composition = analyze_cell_file(Path(cell.source_file))

                if composition.composition_pattern in ("explicit", "delegated"):
                    source_code = Path(cell.source_file).read_text(encoding="utf-8")
                    template = complete_topology(source_code, composition, self.pdk_name)
                    if template:
                        cell.topology_template = template

                # Store composition metadata for COMPOSED_OF edges
                cell._composition = composition  # type: ignore[attr-defined]
            except Exception as e:
                msg = f"AST analysis failed for {cell.module_name}: {e}"
                print(f"  [Warn] {msg}")
                report.warnings.append(msg)

        # Phase 3: Create/update PDK_Cell nodes
        print("PDK Ingestion: Creating PDK_Cell nodes...")
        for cell in cells:
            try:
                created = self._upsert_pdk_cell(cell, report)
                if created:
                    report.cells_created += 1
                else:
                    report.cells_updated += 1
            except Exception as e:
                msg = f"Failed to upsert node {cell.module_name}: {e}"
                print(f"  [Error] {msg}")
                report.errors.append(msg)

        # Phase 4: COMPOSED_OF edges
        print("PDK Ingestion: Creating COMPOSED_OF edges...")
        for cell in cells:
            composition = getattr(cell, "_composition", None)
            if composition is None:
                continue
            for mod_name in composition.imported_modules:
                # Find the target PDK_Cell node
                target = next((c for c in cells if c.module_name == mod_name), None)
                if target is None:
                    continue
                try:
                    self._create_edge_by_name(
                        edge_type="COMPOSED_OF",
                        from_module=cell.module_name,
                        to_module=target.module_name,
                        props={"provenance": "ast_analysis"},
                    )
                    report.composed_of_edges += 1
                    report.edges_created += 1
                except Exception as e:
                    report.warnings.append(f"COMPOSED_OF {cell.module_name}->{mod_name}: {e}")

        # Phase 5: Relationship resolution
        print("PDK Ingestion: Resolving IMPLEMENTS edges...")
        for cell in cells:
            try:
                results = resolve_implements(cell, self.kb_client)
                for r in results:
                    if r.does_implement and r.confidence >= REVIEW_THRESHOLD:
                        props = {
                            "confidence": r.confidence,
                            "reasoning": r.reasoning,
                            "resolved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "needs_review": r.confidence < AUTO_COMMIT_THRESHOLD,
                        }
                        self._create_implements_edge(cell.module_name, r.candidate_name, props)
                        report.implements_edges += 1
                        report.edges_created += 1
            except Exception as e:
                report.warnings.append(f"IMPLEMENTS failed for {cell.module_name}: {e}")

        print("PDK Ingestion: Resolving PERFORMS_FUNCTION edges...")
        for cell in cells:
            try:
                results = resolve_performs_function(cell, self.kb_client)
                for r in results:
                    if r.is_related and r.confidence >= REVIEW_THRESHOLD:
                        self._create_typed_edge(
                            cell.module_name, r.candidate_name,
                            "PERFORMS_FUNCTION", "Design_Function", r,
                        )
                        report.performs_function_edges += 1
                        report.edges_created += 1
            except Exception as e:
                report.warnings.append(f"PERFORMS_FUNCTION failed for {cell.module_name}: {e}")

        print("PDK Ingestion: Resolving FABRICATED_WITH edges...")
        for cell in cells:
            try:
                results = resolve_fabricated_with(cell, self.kb_client)
                for r in results:
                    if r.is_related and r.confidence >= REVIEW_THRESHOLD:
                        self._create_typed_edge(
                            cell.module_name, r.candidate_name,
                            "FABRICATED_WITH", "Physical_Principle", r,
                        )
                        report.fabricated_with_edges += 1
                        report.edges_created += 1
            except Exception as e:
                report.warnings.append(f"FABRICATED_WITH failed for {cell.module_name}: {e}")

        print("PDK Ingestion: Resolving EXHIBITS edges...")
        for cell in cells:
            try:
                results = resolve_exhibits(cell, self.kb_client)
                for r in results:
                    if r.is_related and r.confidence >= REVIEW_THRESHOLD:
                        self._create_typed_edge(
                            cell.module_name, r.candidate_name,
                            "EXHIBITS", "Property", r,
                        )
                        report.exhibits_edges += 1
                        report.edges_created += 1
            except Exception as e:
                report.warnings.append(f"EXHIBITS failed for {cell.module_name}: {e}")

        # Phase 6: Bidirectional enrichment
        print("PDK Ingestion: Running bidirectional enrichment...")
        try:
            n_b = enrich_components_from_pdk(self.kb_client)
            n_a = enrich_pdk_from_components(self.kb_client)
            n_t = upgrade_architecture_templates(self.kb_client)
            report.enrichments_propagated = n_b + n_a + n_t
            print(f"  Enrichment: {n_b} PDK→Component, {n_a} Component→PDK, {n_t} template upgrades")
        except Exception as e:
            report.warnings.append(f"Enrichment failed: {e}")

        print(f"\nPDK Ingestion complete: {report.cells_created} created, "
              f"{report.cells_updated} updated, {report.edges_created} edges, "
              f"{len(report.errors)} errors")
        return report

    # ------------------------------------------------------------------
    # Node creation / versioning
    # ------------------------------------------------------------------

    def _upsert_pdk_cell(self, cell: PDKCellNode, report: PDKIngestionReport) -> bool:
        """Create or update a PDK_Cell node. Returns True if newly created."""
        existing = self._find_pdk_cell(cell.module_name)

        node_props = self._cell_to_props(cell)

        if existing:
            # Version snapshot before update
            self._snapshot_to_history(existing)
            # Update existing node
            with self.kb_client.driver.session() as session:
                session.run(
                    """\
                    MATCH (n:PDK_Cell {pdk_name: $pdk_name, module_name: $module_name})
                    SET n += $props, n.updated_at = datetime()
                    """,
                    pdk_name=self.pdk_name,
                    module_name=cell.module_name,
                    props=node_props,
                )
            # Update embedding
            self._update_pdk_cell_embedding(cell)
            return False
        else:
            # Create new node
            with self.kb_client.driver.session() as session:
                session.run(
                    """\
                    CREATE (n:PDK_Cell)
                    SET n = $props, n.created_at = datetime()
                    """,
                    props=node_props,
                )
            self._update_pdk_cell_embedding(cell)
            return True

    def _find_pdk_cell(self, module_name: str) -> Optional[dict]:
        """Find an existing PDK_Cell node by pdk_name + module_name."""
        query = """\
        MATCH (n:PDK_Cell {pdk_name: $pdk_name, module_name: $module_name})
        RETURN n, elementId(n) AS eid
        """
        with self.kb_client.driver.session() as session:
            result = session.run(query, pdk_name=self.pdk_name, module_name=module_name)
            record = result.single()
            if record:
                data = dict(record["n"])
                data["_eid"] = record["eid"]
                return data
        return None

    def _snapshot_to_history(self, existing: dict):
        """Create a PDK_Cell_History node from current state and link via SUPERSEDES."""
        eid = existing.get("_eid")
        if not eid:
            return

        history_props = {k: v for k, v in existing.items() if not k.startswith("_")}
        history_props.pop("embedding", None)

        with self.kb_client.driver.session() as session:
            session.run(
                """\
                MATCH (current:PDK_Cell) WHERE elementId(current) = $eid
                CREATE (h:PDK_Cell_History)
                SET h = $props, h.archived_at = datetime()
                CREATE (current)-[:SUPERSEDES]->(h)
                """,
                eid=eid,
                props=history_props,
            )

    def _cell_to_props(self, cell: PDKCellNode) -> dict:
        """Convert a PDKCellNode to a flat dict for Neo4j storage."""
        props = {
            "module_name": cell.module_name,
            "name": cell.display_name,
            "display_name": cell.display_name,
            "description": cell.description,
            "pdk_name": cell.pdk_name,
            "pdk_version": cell.pdk_version,
            "ports": cell.ports,
            "labels_list": cell.labels,
            "aka": cell.aka,
            "technology": cell.technology,
            "parameters": json.dumps(cell.parameters) if cell.parameters else "{}",
            "dx_um": cell.dx_um,
            "dy_um": cell.dy_um,
            "is_flattened": cell.is_flattened,
            "is_primitive": cell.is_primitive,
            "has_simulation_model": cell.has_simulation_model,
            "source_file": cell.source_file,
        }

        if cell.port_details:
            props["port_details"] = json.dumps(
                [p.model_dump() for p in cell.port_details]
            )

        if cell.numeric_specs:
            props["numeric_specs"] = json.dumps(
                [s.model_dump() for s in cell.numeric_specs]
            )

        if cell.topology_template:
            props["topology_template"] = json.dumps(cell.topology_template)

        # Remove None values (Neo4j doesn't like them)
        return {k: v for k, v in props.items() if v is not None}

    def _update_pdk_cell_embedding(self, cell: PDKCellNode):
        """Generate and store an embedding for a PDK_Cell node."""
        text = f"{cell.display_name} {cell.description}"
        if cell.aka:
            text += f" {cell.aka}"
        if cell.labels:
            text += f" {' '.join(cell.labels)}"

        embedding = self.kb_client.importer.generate_embedding(text)
        if embedding:
            with self.kb_client.driver.session() as session:
                session.run(
                    """\
                    MATCH (n:PDK_Cell {pdk_name: $pdk_name, module_name: $module_name})
                    SET n.embedding = $embedding
                    """,
                    pdk_name=self.pdk_name,
                    module_name=cell.module_name,
                    embedding=embedding,
                )

    # ------------------------------------------------------------------
    # Edge creation helpers
    # ------------------------------------------------------------------

    def _create_edge_by_name(
        self, edge_type: str, from_module: str, to_module: str, props: Optional[dict] = None
    ):
        """Create an edge between two PDK_Cell nodes by module_name."""
        query = f"""\
        MATCH (a:PDK_Cell {{pdk_name: $pdk, module_name: $from_mod}})
        MATCH (b:PDK_Cell {{pdk_name: $pdk, module_name: $to_mod}})
        MERGE (a)-[r:{edge_type}]->(b)
        """
        if props:
            query += " SET r += $props"

        with self.kb_client.driver.session() as session:
            session.run(
                query,
                pdk=self.pdk_name,
                from_mod=from_module,
                to_mod=to_module,
                props=props or {},
            )

    def _create_implements_edge(
        self, from_module: str, to_component_name: str, props: Optional[dict] = None
    ):
        """Create IMPLEMENTS edge from PDK_Cell to Component."""
        query = """\
        MATCH (a:PDK_Cell {pdk_name: $pdk, module_name: $from_mod})
        MATCH (b:Component {name: $to_name})
        MERGE (a)-[r:IMPLEMENTS]->(b)
        SET r += $props
        """
        with self.kb_client.driver.session() as session:
            session.run(
                query,
                pdk=self.pdk_name,
                from_mod=from_module,
                to_name=to_component_name,
                props=props or {},
            )

    def _create_typed_edge(
        self,
        from_module: str,
        to_name: str,
        edge_type: str,
        to_label: str,
        resolution,
    ):
        """Create a typed edge from PDK_Cell to a target entity."""
        props = {
            "confidence": resolution.confidence,
            "reasoning": resolution.reasoning,
            "resolved_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "needs_review": resolution.confidence < AUTO_COMMIT_THRESHOLD,
        }

        query = f"""\
        MATCH (a:PDK_Cell {{pdk_name: $pdk, module_name: $from_mod}})
        MATCH (b:{to_label} {{name: $to_name}})
        MERGE (a)-[r:{edge_type}]->(b)
        SET r += $props
        """
        with self.kb_client.driver.session() as session:
            session.run(
                query,
                pdk=self.pdk_name,
                from_mod=from_module,
                to_name=to_name,
                props=props,
            )
