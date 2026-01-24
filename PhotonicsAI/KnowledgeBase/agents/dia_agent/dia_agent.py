"""Database Integration Agent (DIA)."""

import datetime
from typing import List, Dict, Any, Optional
from arango.database import StandardDatabase

from PhotonicsAI.KnowledgeBase.ArangoDB.client import KnowledgeBaseClient
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
        kb_client: Optional[KnowledgeBaseClient] = None,
        llm_model: str = "gemini-2.5-pro"
    ):
        """
        Initialize DIA Agent.
        
        Args:
            kb_client: KnowledgeBaseClient instance
            llm_model: LLM model to use for verification
        """
        if kb_client is None:
            kb_client = KnowledgeBaseClient()
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
        inferred_edges = self._perform_global_inference(manifest.nodes)
        report.inferred_edges_found = len(inferred_edges)
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

    def _perform_global_inference(self, new_nodes: List[ProposedNode]) -> List[ProposedEdge]:
        """
        Mine for implicit relationships between new nodes and existing KB entities.
        Uses batched LLM calls to reduce API overhead.
        """
        inferred_edges = []
        
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
            
            # Limit candidates per node
            # candidates = candidates[:3] 
            
            # Create tasks
            for candidate, cand_collection in candidates:
                cand_name = candidate.get("name")
                
                # Skip self-match
                if cand_name == node.name:
                    continue
                    
                cand_desc = candidate.get("description", "")
                cand_type = collection_to_type.get(cand_collection, "Unknown")
                
                # Prepare evidence quotes
                quotes = "\n".join([f"- {q}" for q in node.evidence_quotes])
                if not quotes:
                    quotes = "(No direct quotes, using description): " + node.description

                task_id = f"{node.name}::{cand_name}"
                
                verification_tasks.append({
                    "pair_id": task_id,
                    "node": node,
                    "node_type": node_type,
                    "cand_name": cand_name,
                    "cand_type": cand_type,
                    "cand_collection": cand_collection,
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
        BATCH_SIZE = 5 # conservative batch size
        
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
                        print(f"    [Warn] No result for {task['pair_id']}")
                        continue
                        
                    if res.is_related and res.edge_type and res.edge_type != "None":
                        # Create Edge
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

                        etype = res.edge_type
                        from_n, from_c = node.name, node.collection
                        to_n, to_c = cand_name, cand_collection
                        from_type = node_type
                        to_type = cand_type

                        # Logic to check if we need to swap
                        if etype in edge_rules:
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
                            
                            elif not is_direct_valid and not is_reverse_valid:
                                # Neither direction fits strictly?
                                pass

                        edge = ProposedEdge(
                            edge_collection=etype,
                            from_node=from_n,
                            from_collection=from_c,
                            to_node=to_n,
                            to_collection=to_c,
                            operation="MERGE",
                            description=f"Global Inference: {res.reasoning}",
                            weight=0.8,
                            source_document=node.source_document
                        )
                        inferred_edges.append(edge)
                        print(f"  [Inference] Found link: {from_n} --[{res.edge_type}]--> {to_n}")

            except Exception as e:
                print(f"  [Error] Batch failed: {e}")

        return inferred_edges

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
            # Check for unified create_edge method (Neo4j)
            if hasattr(self.kb_client, 'create_edge'):
                self.kb_client.create_edge(
                    edge.edge_collection,
                    from_key,
                    edge.from_collection,
                    to_key,
                    edge.to_collection
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

