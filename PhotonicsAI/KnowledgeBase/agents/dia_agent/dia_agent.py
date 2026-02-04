"""Database Integration Agent (DIA)."""

import datetime
import json
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.Photon import llm_api
from PhotonicsAI.KnowledgeBase.agents.vsa_agent.models import VSAUpdatePayload, ProposedNode, ProposedEdge
from .models import DIAReport, SemanticVerificationResult, BatchSemanticVerificationResult
from .prompts import SEMANTIC_VERIFICATION_SYS_PROMPT, SEMANTIC_VERIFICATION_USER_TEMPLATE, PAIR_TEMPLATE

class DIAAgent:
    """
    Database Integration Agent (DIA).
    
    Responsible for executing the VSA Manifest and performing Global Inference
    to discover implicit relationships with existing Knowledge Base entities.
    """

    def __init__(
        self,
        kb_client: Optional[Neo4jClient] = None,
        llm_model: str = "gemini-2.5-pro"
    ):
        """
        Initialize DIA Agent.
        
        Args:
            kb_client: Neo4jClient instance
            llm_model: LLM model to use for verification
        """
        if kb_client is None:
            kb_client = Neo4jClient()
            # Connect happens lazily or explicitly
            
        self.kb_client = kb_client
        self.llm_model = llm_model
        
        # Edge Constraints (Source Type -> [Target Types]) for semantic search filtering
        # Used to narrow down which collections to search for candidates
        self.edge_constraints = {
            "Component": ["Physical_Principle", "Design_Function", "Property"],
            "Architecture": ["Component", "Physical_Principle", "Design_Function", "Property"],
            "Physical_Principle": ["Component", "Architecture"], # Bi-directional potential
            "Design_Function": ["Component", "Architecture"],
            "Property": ["Component", "Architecture"]
        }
        
        # Mapping from Entity Type to Collection Name
        self.type_to_collection = {
            "Component": "Components",
            "Architecture": "Architectures",
            "Property": "Properties",
            "Design_Function": "Design_Functions",
            "Physical_Principle": "Physical_Principles"
        }
        self.label_to_collection = {
            "Component": "Components",
            "Architecture": "Architectures",
            "Property": "Properties",
            "Design_Function": "Design_Functions",
            "Physical_Principle": "Physical_Principles",
            "Document": "Documents",
        }

    def integrate_manifest(self, manifest: VSAUpdatePayload) -> DIAReport:
        """
        Execute the manifest and perform global inference.
        
        Args:
            manifest: VSAUpdatePayload containing nodes and edges to integrate.
            
        Returns:
            DIAReport summarizing the integration.
        """
        print(f"\nDIA: Starting integration for document {manifest.document_key}...")
        
        if not self.kb_client._connected:
            self.kb_client.connect()
            
        report = DIAReport(
            document_key=manifest.document_key,
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )
        
        # 1. Global Inference (Internal Semantic Evidence Mining)
        print("DIA: Running Global Inference (Semantic Evidence Mining)...")
        inferred_edges, queued_inference_reviews = self._perform_global_inference(manifest.nodes)
        report.inferred_edges_found = len(inferred_edges)
        report.review_items_queued += queued_inference_reviews
        print(f"DIA: Found {len(inferred_edges)} inferred global edges.")
        
        # 2. Atomic Integration (The Commit)
        print("DIA: Committing changes to ArangoDB...")
        
        try:
            # Note: ArangoDB transactions across multiple collections can be complex.
            # We will perform operations sequentially with robust error checking.
            # If a strict ACID transaction is required, we would wrap this in a db.transaction() 
            # block, but that requires predefined read/write collections.
            
            # 2a. Nodes
            for node in manifest.nodes:
                try:
                    if not self._has_node_evidence(node):
                        # Skip empty MERGE nodes without evidence
                        if node.operation == "MERGE" and not node.description:
                            print(f"  [Skip] No evidence for MERGE node {node.name}.")
                            continue
                        queued = self._queue_review_item(
                            item_type="node",
                            payload=node,
                            reason="missing_evidence",
                            source_document=manifest.document_key,
                            priority="high"
                        )
                        if queued:
                            report.review_items_queued += 1
                        continue

                    self._commit_node(node, manifest.document_key)
                    if node.operation == "CREATE":
                        report.nodes_created += 1
                    elif node.operation == "UPDATE":
                        report.nodes_updated += 1
                    else:
                        report.nodes_merged += 1
                except Exception as e:
                    error_msg = f"Failed to commit node {node.name}: {str(e)}"
                    print(f"  ! {error_msg}")
                    report.errors.append(error_msg)

            # 2b. Edges (Explicit + Inferred)
            all_edges = manifest.edges + inferred_edges
            for edge in all_edges:
                try:
                    if not self._has_edge_evidence(edge):
                        queued = self._queue_review_item(
                            item_type="edge",
                            payload=edge,
                            reason="missing_evidence",
                            source_document=manifest.document_key,
                            priority="high"
                        )
                        if queued:
                            report.review_items_queued += 1
                        continue

                    self._commit_edge(edge)
                    if edge in manifest.edges:
                        if edge.edge_collection != "EXTRACTED_FROM": # Don't count provenance edges as "created" in user summary usually, or count separately
                            report.explicit_edges_created += 1
                    # Inferred edges are counted separately in report
                except Exception as e:
                    error_msg = f"Failed to commit edge {edge.from_node} -> {edge.to_node}: {str(e)}"
                    print(f"  ! {error_msg}")
                    report.errors.append(error_msg)
            
            report.total_edges_created = report.explicit_edges_created + report.inferred_edges_found

        except Exception as e:
            critical_error = f"Critical failure during integration: {str(e)}"
            print(f"DIA: X {critical_error}")
            report.errors.append(critical_error)

        print(f"DIA: Integration complete. Errors: {len(report.errors)}")
        return report

    def _perform_global_inference(self, new_nodes: List[ProposedNode]) -> Tuple[List[ProposedEdge], int]:
        """
        Mine for implicit relationships between new nodes and existing KB entities.
        Uses batched LLM calls to reduce API overhead.
        """
        inferred_edges = []
        queued_reviews = 0
        extracted_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        # Map collection names back to entity types for logic
        collection_to_type = {v: k for k, v in self.type_to_collection.items()}
        
        # 1. Collect all verification tasks
        verification_tasks = [] # List of dicts with all info needed for prompt and processing
        
        print("DIA: Collecting candidates for batch verification...")
        
        for node in new_nodes:
            # Only infer for content-bearing nodes (CREATE/UPDATE) or those with description
            if not node.description:
                continue
                
            node_type = collection_to_type.get(node.collection)
            if not node_type:
                continue
                
            # Determine target collections
            allowed_targets = self.edge_constraints.get(node_type, [])
            target_collections = [self.type_to_collection.get(t) for t in allowed_targets if t in self.type_to_collection]
            
            if not target_collections:
                continue

            # Candidate Search
            candidates = []
            for target_coll in target_collections:
                try:
                    results = self.kb_client.semantic_search(
                        query_text=node.description,
                        collection=target_coll,
                        threshold=0.4 # High threshold for relevance
                    )
                    candidates.extend([(r, target_coll) for r in results])
                except Exception as e:
                    print(f"  [Warn] Search failed for {target_coll}: {e}")

            # Phase 1: Neighbor expansion from semantic seeds
            seed_nodes = []
            for cand, cand_collection in candidates[:10]:
                if cand.get("name"):
                    seed_nodes.append({
                        "name": cand.get("name"),
                        "collection": cand_collection,
                    })
            neighbor_candidates = self._expand_neighbors(seed_nodes, max_hops=2, limit=50)

            # Phase 2: Personalized PageRank expansion (if GDS available)
            ppr_candidates = self._ppr_candidates(seed_nodes, limit=50)

            # Merge candidates and filter to target collections
            extra_candidates = neighbor_candidates + ppr_candidates
            for cand in extra_candidates:
                cand_collection = cand.get("collection")
                if cand_collection in target_collections:
                    candidates.append((cand, cand_collection))
            
            # Limit candidates per node
            # candidates = candidates[:3] 
            
            # Create tasks
            for candidate, cand_collection in candidates:
                cand_name = candidate.get("name")
                
                # Skip self-match
                if cand_name == node.name:
                    continue
                    
                cand_desc = candidate.get("description", "")
                cand_score = candidate.get("score", 0.0)
                cand_type = collection_to_type.get(cand_collection, "Unknown")
                
                # Prepare evidence quotes
                quotes = "\n".join([f"- {q}" for q in node.evidence_quotes])
                if not quotes:
                    quotes = ""

                task_id = f"{node.name}::{cand_name}"
                
                verification_tasks.append({
                    "pair_id": task_id,
                    "node": node,
                    "node_type": node_type,
                    "cand_name": cand_name,
                    "cand_type": cand_type,
                    "cand_collection": cand_collection,
                    "cand_score": cand_score,
                    "prompt_data": {
                        "pair_id": task_id,
                        "new_entity_name": node.name,
                        "new_entity_type": node_type,
                        "new_entity_description": node.description,
                        "evidence_quotes": quotes,
                        "candidate_name": cand_name,
                        "candidate_type": cand_type,
                        "candidate_description": cand_desc[:500] # Shorter truncation for batching
                    }
                })

        print(f"DIA: Found {len(verification_tasks)} candidate pairs. Processing in batches...")

        # 2. Process in Batches
        BATCH_SIZE = 10 # conservative batch size
        
        # Chunk tasks
        chunks = [verification_tasks[i:i + BATCH_SIZE] for i in range(0, len(verification_tasks), BATCH_SIZE)]
        
        for chunk_idx, chunk in enumerate(chunks):
            # Prepare batch prompt
            pairs_content = ""
            for task in chunk:
                pairs_content += PAIR_TEMPLATE.format(**task["prompt_data"])
            
            prompt = SEMANTIC_VERIFICATION_USER_TEMPLATE.format(pairs_content=pairs_content)
            
            try:
                print(f"  Processing batch {chunk_idx+1}/{len(chunks)} ({len(chunk)} pairs)...")
                batch_result = llm_api.callgoogle_pydantic(
                    prompt=prompt,
                    sys_prompt=SEMANTIC_VERIFICATION_SYS_PROMPT,
                    pydantic_model=BatchSemanticVerificationResult
                )
                
                # Process results
                result_map = {r.pair_id: r for r in batch_result.results}
                
                for task in chunk:
                    res = result_map.get(task["pair_id"])
                    if not res:
                        queued = self._queue_review_item(
                            item_type="edge",
                            payload={
                                "pair_id": task["pair_id"],
                                "new_entity_name": task["node"].name,
                                "candidate_name": task["cand_name"],
                                "candidate_type": task["cand_type"],
                            },
                            reason="missing_verification_result",
                            source_document=task["node"].source_document,
                            priority="medium"
                        )
                        if queued:
                            queued_reviews += 1
                        print(f"    [Warn] No result for {task['pair_id']}")
                        continue
                        
                    node = task["node"]
                    node_type = task["node_type"]
                    cand_name = task["cand_name"]
                    cand_type = task["cand_type"]
                    cand_collection = task["cand_collection"]

                    # Define constraints for edge directions to ensure correct graph topology
                    # Map: Edge Type -> (Allowed Source Types, Allowed Target Types)
                    edge_rules = {
                        "PERFORMS_FUNCTION": (["Component", "Architecture"], ["Design_Function"]),
                        "BASED_ON_PRINCIPLE": (["Component", "Architecture"], ["Physical_Principle"]),
                        "HAS_PROPERTY": (["Component", "Architecture"], ["Property"]),
                        "USES_COMPONENT": (["Architecture"], ["Component"]),
                        # RELATED_TO is generic, usually keep direction as found or ignore
                    }

                    etype = res.edge_type if res.edge_type and res.edge_type != "None" else None
                    from_n, from_c = node.name, node.collection
                    to_n, to_c = cand_name, cand_collection
                    from_type = node_type
                    to_type = cand_type

                    is_direct_valid = False
                    is_reverse_valid = False
                    if etype and etype in edge_rules:
                        allowed_sources, allowed_targets = edge_rules[etype]

                        # Check if current direction matches
                        # Current: New(from) -> Existing(to)
                        is_direct_valid = (from_type in allowed_sources) and (to_type in allowed_targets)

                        # Check if reverse matches
                        # Reverse: Existing(to) -> New(from)
                        is_reverse_valid = (to_type in allowed_sources) and (from_type in allowed_targets)

                        if not is_direct_valid and is_reverse_valid:
                            # Swap!
                            print(f"  [Debug] Swapping edge direction for {etype}: {to_n} -> {from_n}")
                            from_n, from_c, to_n, to_c = to_n, to_c, from_n, from_c

                    evidence_count = len(node.evidence_quotes or [])
                    evidence_score = 1.0 if evidence_count >= 2 else 0.5 if evidence_count == 1 else 0.0
                    similarity_score = float(task.get("cand_score", 0.0) or 0.0)
                    llm_conf = float(getattr(res, "confidence", 0.0) or 0.0)

                    if res.is_related and etype:
                        schema_ok = 1.0 if is_direct_valid or is_reverse_valid else 0.0
                        combined_conf = (
                            0.45 * similarity_score +
                            0.15 * evidence_score +
                            0.15 * schema_ok +
                            0.25 * llm_conf
                        )

                        edge = ProposedEdge(
                            edge_collection=etype,
                            from_node=from_n,
                            from_collection=from_c,
                            to_node=to_n,
                            to_collection=to_c,
                            operation="MERGE",
                            description=f"Global Inference: {res.reasoning}",
                            evidence_quotes=list(node.evidence_quotes or []),
                            weight=combined_conf,
                            confidence=combined_conf,
                            source_document=node.source_document,
                            provenance="DIA",
                            extracted_at=extracted_at
                        )
                        inferred_edges.append(edge)
                        print(f"  [Inference] Found link: {from_n} --[{etype}]--> {to_n}")
                    else:
                        combined_conf = (
                            0.55 * similarity_score +
                            0.15 * evidence_score +
                            0.30 * llm_conf
                        )

                        placeholder_etype = etype or "RELATED_TO"
                        queue_allowed = (
                            placeholder_etype == "RELATED_TO" or (is_direct_valid or is_reverse_valid)
                        )

                        if 0.8 <= combined_conf <= 1.0 and queue_allowed:
                            queued_edge = ProposedEdge(
                                edge_collection=placeholder_etype,
                                from_node=from_n,
                                from_collection=from_c,
                                to_node=to_n,
                                to_collection=to_c,
                                operation="MERGE",
                                description=f"Rejected by semantic verifier: {res.reasoning}",
                                evidence_quotes=list(node.evidence_quotes or []),
                                weight=combined_conf,
                                confidence=combined_conf,
                                source_document=node.source_document,
                                provenance="DIA",
                                extracted_at=extracted_at
                            )
                            queued = self._queue_review_item(
                                item_type="edge",
                                payload=queued_edge,
                                reason="semantic_verification_rejected",
                                source_document=node.source_document,
                                priority="medium"
                            )
                            if queued:
                                queued_reviews += 1

            except Exception as e:
                print(f"  [Error] Batch failed: {e}")

        return inferred_edges, queued_reviews

    def _expand_neighbors(
        self,
        seed_nodes: List[Dict[str, str]],
        max_hops: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Expand neighbors from seed nodes using Cypher."""
        if not seed_nodes:
            return []
        if not hasattr(self.kb_client, "driver") or self.kb_client.driver is None:
            return []

        seed_names = [s["name"] for s in seed_nodes if s.get("name")]
        if not seed_names:
            return []

        query = f"""
        MATCH (seed)
        WHERE seed.name IN $seed_names
        MATCH (seed)-[*1..{max_hops}]-(n)
        RETURN DISTINCT n, labels(n) AS labels
        LIMIT $limit
        """
        results = []
        try:
            with self.kb_client.driver.session() as session:
                records = session.run(query, seed_names=seed_names, limit=limit)
                for record in records:
                    node = record["n"]
                    labels = record["labels"] or []
                    label = labels[0] if labels else None
                    if not label:
                        continue
                    collection = self.label_to_collection.get(label)
                    if not collection:
                        continue
                    results.append({
                        "name": node.get("name"),
                        "description": node.get("description", ""),
                        "score": 0.0,
                        "collection": collection,
                    })
        except Exception as e:
            print(f"  [Warn] Neighbor expansion failed: {e}")
        return results

    def _ppr_candidates(
        self,
        seed_nodes: List[Dict[str, str]],
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Run Personalized PageRank on a local subgraph if GDS is available."""
        if not seed_nodes:
            return []
        if not hasattr(self.kb_client, "driver") or self.kb_client.driver is None:
            return []

        seed_names = [s["name"] for s in seed_nodes if s.get("name")]
        if not seed_names:
            return []

        graph_name = f"ppr_temp_{datetime.datetime.now().timestamp()}"
        results = []
        try:
            with self._gds_session() as session:
                # Project subgraph around seeds (2-hop neighborhood)
                session.run(
                    """
                    CALL gds.graph.project.cypher(
                      $graphName,
                      'MATCH (seed) WHERE seed.name IN $seed_names
                       MATCH (seed)-[*1..2]-(n)
                       RETURN DISTINCT id(n) AS id',
                      'MATCH (seed) WHERE seed.name IN $seed_names
                       MATCH (seed)-[*1..2]-(n)-[r]-(m)
                       WHERE id(n) IS NOT NULL AND id(m) IS NOT NULL
                       RETURN DISTINCT id(n) AS source, id(m) AS target',
                      {parameters: {seed_names: $seed_names}, validateRelationships: false}
                    )
                    """,
                    graphName=graph_name,
                    seed_names=seed_names,
                )

                # Run personalized PageRank seeded on initial nodes
                records = session.run(
                    """
                    CALL gds.pageRank.stream($graphName, {maxIterations: 20, dampingFactor: 0.85})
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId) AS node, score
                    ORDER BY score DESC
                    LIMIT $limit
                    """,
                    graphName=graph_name,
                    limit=limit,
                )

                for record in records:
                    node = record["node"]
                    labels = list(node.labels) if hasattr(node, "labels") else []
                    label = labels[0] if labels else None
                    if not label:
                        continue
                    collection = self.label_to_collection.get(label)
                    if not collection:
                        continue
                    results.append({
                        "name": node.get("name"),
                        "description": node.get("description", ""),
                        "score": record["score"],
                        "collection": collection,
                    })
        except Exception as e:
            print(f"  [Warn] PPR failed or GDS unavailable: {e}")
        finally:
            try:
                with self._gds_session() as session:
                    session.run("CALL gds.graph.drop($graphName)", graphName=graph_name)
            except Exception:
                pass

        return results

    def _gds_session(self):
        """Open a Neo4j session with deprecation notifications disabled."""
        try:
            return self.kb_client.driver.session(
                notifications_disabled_categories=["DEPRECATION"]
            )
        except TypeError:
            # Fallback for older driver versions
            return self.kb_client.driver.session()

    def _has_node_evidence(self, node: ProposedNode) -> bool:
        """Require source_document and evidence for non-empty updates."""
        if not node.source_document:
            return False
        if node.evidence_quotes:
            return True
        if node.metadata and node.metadata.get("context_pack"):
            return True
        return False

    def _has_edge_evidence(self, edge: ProposedEdge) -> bool:
        """Require source_document and rationale for edges."""
        if edge.edge_collection == "EXTRACTED_FROM":
            return True
        if not edge.source_document:
            return False
        if edge.description:
            return True
        if edge.evidence_quotes:
            return True
        return False

    def _serialize_payload(self, payload: Any) -> Dict[str, Any]:
        """Serialize pydantic or plain payloads for review queue."""
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        if hasattr(payload, "dict"):
            return payload.dict()
        if isinstance(payload, dict):
            return payload
        return {"value": str(payload)}

    def _queue_review_item(
        self,
        item_type: str,
        payload: Any,
        reason: str,
        source_document: Optional[str] = None,
        priority: str = "normal"
    ) -> bool:
        """Store a review item in Neo4j if available."""
        if not hasattr(self.kb_client, "driver") or self.kb_client.driver is None:
            print(f"  [ReviewQueue] Skipped (no Neo4j driver): {reason}")
            return False

        item_id = str(uuid4())
        payload_json = json.dumps(self._serialize_payload(payload), default=str)
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        props = {
            "id": item_id,
            "item_type": item_type,
            "reason": reason,
            "priority": priority,
            "status": "pending",
            "created_at": created_at,
            "source_document": source_document or "",
            "payload_json": payload_json
        }

        query = """
        MERGE (r:ReviewItem {id: $id})
        SET r += $props
        """

        try:
            with self.kb_client.driver.session() as session:
                session.run(query, id=item_id, props=props)
                if source_document:
                    session.run(
                        "MATCH (r:ReviewItem {id: $id}) MATCH (d:Document {title: $title}) MERGE (r)-[:REVIEW_OF]->(d)",
                        id=item_id,
                        title=source_document
                    )
            print(f"  [ReviewQueue] Queued {item_type} ({reason})")
            return True
        except Exception as e:
            print(f"  [ReviewQueue] Failed to queue item: {e}")
            return False

    def _commit_node(self, node: ProposedNode, document_key: str):
        """Commit a single node to ArangoDB."""
        # Check existence
        existing = self.kb_client.find_by_name(node.name, node.collection)
        
        entity_data = {
            "name": node.name,
            "description": node.description,
            "source": document_key, # Last source
            # Add other fields like components/connectivity if present
            **node.metadata
        }
        if node.components:
            entity_data["components"] = node.components
        if node.connectivity:
            entity_data["connectivity"] = node.connectivity
            
        if existing:
            # UPDATE or MERGE
            if node.operation == "UPDATE" or (node.operation == "MERGE" and not existing.get("description")):
                # Update logic
                # For now, we overwrite description if UPDATE.
                # ideally we merge, but VSA already did the text merging.
                
                # We use internal importer/db methods or simple update
                # Replaced direct DB access with client method for Neo4j compatibility
                existing["description"] = node.description # Update description
                # Merge metadata/lists? For now simple overwrite of specific fields
                for k, v in entity_data.items():
                    if v: existing[k] = v
                
                existing["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                
                # Check if client has update_node method (Neo4j) or use Arango style
                if hasattr(self.kb_client, 'update_node'):
                    self.kb_client.update_node(existing, node.collection)
                else:
                    # Legacy ArangoDB direct access
                    coll = self.kb_client.db.collection(node.collection)
                    coll.update(existing)
                
                # Regenerate embedding
                self.kb_client.update_entity_embedding(existing["_key"], node.collection)
        else:
            # CREATE
            self.kb_client.add_entity(entity_data, node.collection, document_key)

    def _commit_edge(self, edge: ProposedEdge):
        """Commit a single edge to ArangoDB."""
        # We need _keys for the from/to nodes.
        # Use kb_client to find them.
        
        # 1. Resolve Keys
        # Special handling for Papers collection where we likely have the _key, not name
        if edge.from_collection == "Papers":
            from_v = {"_key": edge.from_node} # Assume node is key
        else:
            from_v = self.kb_client.find_by_name(edge.from_node, edge.from_collection)
            
        if edge.to_collection == "Papers":
            # For EXTRACTED_FROM, to_node is the document key
            to_v = {"_key": edge.to_node} 
        else:
            to_v = self.kb_client.find_by_name(edge.to_node, edge.to_collection)
        
        if not from_v:
            # raise ValueError(f"Source node not found: {edge.from_node}")
            print(f"  [Error] Source node missing for edge: {edge.from_node}")
            return
        if not to_v:
            # raise ValueError(f"Target node not found: {edge.to_node}")
            print(f"  [Error] Target node missing for edge: {edge.to_node}")
            return
            
        from_key = from_v["_key"]
        to_key = to_v["_key"]
        
        # 2. Create Edge
        # Access importer internal method or implement direct logic
        # kb_client.importer._create_edge uses internal logic.
        
        # We can call importer._create_edge directly if accessible, 
        # but let's be safe and use a public method or explicit DB call if client doesn't expose it well.
        # kb_client.importer IS exposed.
        
        try:
            edge_props = {
                "description": edge.description,
                "evidence_quotes": edge.evidence_quotes,
                "weight": edge.weight,
                "confidence": edge.confidence,
                "source_document": edge.source_document,
                "provenance": edge.provenance,
                "extracted_at": edge.extracted_at,
            }
            edge_props = {
                key: value
                for key, value in edge_props.items()
                if value not in (None, "", [])
            }

            # Check for unified create_edge method (Neo4j)
            if hasattr(self.kb_client, 'create_edge'):
                self.kb_client.create_edge(
                    edge.edge_collection,
                    from_key,
                    edge.from_collection,
                    to_key,
                    edge.to_collection,
                    props=edge_props
                )
            else:
                # Use positional arguments as per signature: _create_edge(edge_type, from_key, from_collection, to_key, to_collection)
                self.kb_client.importer._create_edge(
                    edge.edge_collection, # edge_type
                    from_key,             # from_key
                    edge.from_collection, # from_collection
                    to_key,               # to_key
                    edge.to_collection    # to_collection
                )
        except Exception as e:
            # Check if it's a unique constraint violation (which is fine for MERGE)
            if "unique constraint" not in str(e).lower():
                raise e

