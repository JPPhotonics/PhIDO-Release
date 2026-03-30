"""VSA Agent: Validation and Synthesis."""

import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.Photon import llm_api
from PhotonicsAI.KnowledgeBase.agents.ppc_agent.models import PPCResult, NormalizedEntity, NewConcept

from .models import (
    VSAUpdatePayload,
    ProposedNode,
    ProposedEdge,
    InferredEdge,
    InferredEdgeList,
    EdgeTypeEnum
)
from .prompts import (
    ARCH_INTEGRITY_SYS_PROMPT,
    ArchitectureValidationResult,
    KNOWLEDGE_MERGE_SYS_PROMPT,
    MERGE_USER_PROMPT_TEMPLATE,
    DEEP_INFERENCE_SYS_PROMPT,
    DEEP_INFERENCE_USER_TEMPLATE,
    build_deep_inference_sys_prompt,
)

if TYPE_CHECKING:
    from PhotonicsAI.KnowledgeBase.Neo4j.schema_registry import SchemaRegistry


class VSAAgent:
    """
    Validation and Synthesis Agent (VSA).
    
    Transforms normalized PPC Agent output into a verified Transaction Manifest
    ready for database integration.
    """

    def __init__(
        self,
        kb_client: Optional[Neo4jClient] = None,
        llm_model: str = "gemini-2.5-pro",
        schema_registry: Optional["SchemaRegistry"] = None,
    ):
        """
        Initialize VSA Agent.
        
        Args:
            kb_client: Neo4jClient instance (will create if None)
            llm_model: LLM model to use for validation and merging
            schema_registry: Optional SchemaRegistry for dynamic edge types.
                If provided, edge-type prompts and validation are driven by
                the registry rather than the hardcoded EdgeTypeEnum.
        """
        if kb_client is None:
            kb_client = Neo4jClient()
            # kb_client.connect() # Connect happens lazily or explicitly
            
        self.kb_client = kb_client
        self.llm_model = llm_model
        self.schema_registry = schema_registry
        
        # Collection mapping
        self.type_to_collection = {
            "Component": "Components",
            "Architecture": "Architectures",
            "Property": "Properties",
            "Design_Function": "Design_Functions",
            "Physical_Principle": "Physical_Principles",
            "PDK_Cell": "PDK_Cells",
        }

    def process(self, ppc_result: Dict[str, Any], document_key: str) -> VSAUpdatePayload:
        """
        Execute the 4-phase VSA pipeline.
        
        Args:
            ppc_result: JSON dict from PPC Agent (known_entities, new_concepts)
            document_key: Key of the source document in ArangoDB
            
        Returns:
            VSAUpdatePayload (Transaction Manifest)
        """
        # Parse input if dict
        if isinstance(ppc_result, dict):
            # Safe reconstruction
            known = [NormalizedEntity(**e) for e in ppc_result.get("known_entities", [])]
            new_concepts = [NewConcept(**c) for c in ppc_result.get("new_concepts", [])]
        else:
            # Assume it's already a PPCResult or similar object
            known = ppc_result.known_entities
            new_concepts = ppc_result.new_concepts

        # Containers for results
        final_nodes: List[ProposedNode] = []
        final_edges: List[ProposedEdge] = []
        validation_flags: List[str] = []

        print(f"VSA: Processing {len(known)} known entities and {len(new_concepts)} new concepts...")

        # --- Phase 1: Architecture Integrity Gate ---
        # Filter new concepts (specifically Architectures)
        valid_new_concepts = []
        for concept in new_concepts:
            if concept.entity_type == "Architecture":
                is_valid, reason = self._phase_1_architecture_gate(concept)
                if is_valid:
                    valid_new_concepts.append(concept)
                else:
                    msg = f"Rejected Architecture '{concept.name}': {reason}"
                    print(f"  [Gate] {msg}")
                    validation_flags.append(msg)
            else:
                valid_new_concepts.append(concept)
        
        # --- Phase 2: Relational Synthesis & Edge Mapping ---
        # Generate explicit edges (USES_COMPONENT) and inferred edges
        print("VSA: Synthesizing edges...")
        
        extracted_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # 2a. Explicit edges (from structure)
        explicit_edges = self._phase_2_relational_synthesis(
            known, valid_new_concepts, document_key, extracted_at
        )
        final_edges.extend(explicit_edges)
        
        # 2b. Deep Inference (Implicit edges from context)
        print("VSA: Running Deep Inference for implicit edges...")
        inferred_edges = self._infer_local_edges(
            known, valid_new_concepts, document_key, extracted_at
        )
        final_edges.extend(inferred_edges)
        print(f"  [Inference] Proposed {len(inferred_edges)} implicit relationships.")

        # --- Phase 3: Knowledge Merge (Contextual Enrichment) ---
        # Process known entities with updates
        print("VSA: Merging knowledge...")
        merged_nodes = self._phase_3_knowledge_merge(known, document_key)
        final_nodes.extend(merged_nodes)
        
        # Convert valid new concepts to ProposedNode (CREATE)
        for concept in valid_new_concepts:
            collection = self.type_to_collection.get(concept.entity_type, "Concepts")
            
            # Prepare metadata
            meta = {"related_entities": concept.related_entities}
            if concept.context:
                meta["context_pack"] = concept.context
                
            node = ProposedNode(
                collection=collection,
                name=concept.name,
                operation="CREATE",
                description=concept.description,
                evidence_quotes=concept.evidence_quotes,
                key_metrics=concept.key_metrics,
                components=concept.components,
                connectivity=concept.connectivity,
                source_document=document_key,
                metadata=meta
            )
            final_nodes.append(node)

        # --- Phase 4: Transaction Manifest Generation ---
        payload = self._phase_4_manifest_generation(
            document_key, final_nodes, final_edges, validation_flags
        )
        
        print(f"VSA: Generated manifest with {len(payload.nodes)} nodes and {len(payload.edges)} edges.")
        return payload

    def _phase_1_architecture_gate(self, concept: NewConcept) -> tuple[bool, str]:
        """
        Validate Architecture concepts for specificity.
        Returns (is_valid, reason).
        """
        # Hard check: must have explicit components or connectivity listed
        if not concept.components and not concept.connectivity:
            return False, "Missing components and connectivity details."

        # LLM Check for "vague connectivity"
        # Construct prompt
        details = f"Components: {concept.components}\nConnectivity: {concept.connectivity}\nDescription: {concept.description}"
        prompt = f"Architecture Name: {concept.name}\n\nDetails:\n{details}"
        
        try:
            # Use callgoogle_pydantic for structured validation
            result = llm_api.callgoogle_pydantic(
                prompt=prompt,
                sys_prompt=ARCH_INTEGRITY_SYS_PROMPT,
                pydantic_model=ArchitectureValidationResult
            )
            return result.is_valid, result.reason
        except Exception as e:
            print(f"  [Gate] LLM validation failed for {concept.name}: {e}")
            # Fallback: strict pass if we have data, but log warning
            return True, "LLM check failed; passed based on presence of fields."

    def _phase_2_relational_synthesis(
        self,
        known: List[NormalizedEntity],
        new_concepts: List[NewConcept],
        document_key: str,
        extracted_at: str,
    ) -> List[ProposedEdge]:
        """
        Propose edges based on entity types and evidence.
        """
        edges = []
        
        # Helper to process a list of entities (either NormalizedEntity or NewConcept)
        all_entities = []
        for e in known:
            all_entities.append({
                "name": e.kb_name, # Use KB name for known
                "type": e.entity_type,
                "collection": e.collection,
                "desc": getattr(e, "kb_update_suggestion", None) # Access suggestion if present
            })
        for c in new_concepts:
            all_entities.append({
                "name": c.name,
                "type": c.entity_type,
                "collection": self.type_to_collection.get(c.entity_type, "Concepts"),
                "desc": None
            })

        # Architecture -> Component edges (USES_COMPONENT)
        for entity in known + new_concepts: # Iterate original objects to access fields
            
            # Determine source name/collection
            if isinstance(entity, NormalizedEntity):
                src_name = entity.kb_name
                src_coll = entity.collection
                src_type = entity.entity_type
                comps = entity.components
            else:
                src_name = entity.name
                src_type = entity.entity_type
                src_coll = self.type_to_collection.get(src_type, "Concepts")
                comps = entity.components

            if src_type == "Architecture" and comps:
                for comp_name in comps:
                    edges.append(ProposedEdge(
                        edge_collection="USES_COMPONENT",
                        from_node=src_name,
                        from_collection=src_coll,
                        to_node=comp_name, # This might need resolution if it's a raw extraction
                        to_collection="Components", # Assumption
                        operation="MERGE", # If exists, don't dupe
                        description="Extracted from architecture components list",
                        evidence_quotes=[],
                        source_document=document_key,
                        provenance="VSA",
                        extracted_at=extracted_at
                    ))

        return edges

    def _infer_local_edges(
        self,
        known: List[NormalizedEntity],
        new_concepts: List[NewConcept],
        document_key: str,
        extracted_at: str,
    ) -> List[ProposedEdge]:
        """
        Deep Inference module: Uses LLM to infer implicit relationships from paper context.
        When a SchemaRegistry is available, the prompt includes the dynamic schema and
        the LLM may propose novel relationship types.
        """
        # Step A: Preparation
        # Aggregate unique evidence quotes and entity names
        evidence_quotes = set()
        entity_registry = {} # name -> type mapping
        
        # Registry population helper
        def register(name, etype, quotes):
            entity_registry[name] = etype
            for q in quotes:
                if q: evidence_quotes.add(q.strip())

        for e in known:
            register(e.kb_name, e.entity_type, getattr(e.kb_update_suggestion, "evidence_quotes", []) or [])
        
        for c in new_concepts:
            register(c.name, c.entity_type, c.evidence_quotes)
            
        if not entity_registry:
            return []

        # Step B: LLM Call
        entities_list_str = "\n".join([f"- {name} ({etype})" for name, etype in entity_registry.items()])
        quote_list = list(evidence_quotes)[:20]
        quotes_str = "\n".join([f"\"{q}\"" for q in quote_list]) # Limit to 20 quotes to fit context
        
        prompt = DEEP_INFERENCE_USER_TEMPLATE.format(
            entities_list=entities_list_str,
            evidence_quotes=quotes_str
        )

        # Choose system prompt: dynamic (registry-backed) or legacy (hardcoded)
        if self.schema_registry is not None:
            sys_prompt = build_deep_inference_sys_prompt(
                self.schema_registry.get_prompt_schema_block()
            )
        else:
            sys_prompt = DEEP_INFERENCE_SYS_PROMPT
        
        try:
            result = llm_api.callgoogle_pydantic(
                prompt=prompt,
                sys_prompt=sys_prompt,
                pydantic_model=InferredEdgeList
            )
            inferred_edges = result.edges
        except Exception as e:
            print(f"  [Inference] LLM call failed: {e}")
            return []

        # Build dynamic edge rules from registry (or fall back to hardcoded)
        if self.schema_registry is not None:
            edge_rules = self.schema_registry.get_edge_rules()
            active_type_names = set(self.schema_registry.get_active_type_names())
        else:
            edge_rules = {
                "PERFORMS_FUNCTION": (["Component", "Architecture"], ["Design_Function"]),
                "BASED_ON_PRINCIPLE": (["Component", "Architecture"], ["Physical_Principle"]),
                "HAS_PROPERTY": (["Component", "Architecture"], ["Property"]),
                "USES_COMPONENT": (["Architecture"], ["Component"]),
            }
            active_type_names = {e.value for e in EdgeTypeEnum}

        # Step C: Verification & Conversion
        proposed_edges = []
        for edge in inferred_edges:
            # 1. Entity Existence Check
            src_valid = edge.source_node in entity_registry
            tgt_valid = edge.target_node in entity_registry
            
            if not (src_valid and tgt_valid):
                continue
                
            # 2. Schema Constraint Validation
            src_type = entity_registry[edge.source_node]
            tgt_type = entity_registry[edge.target_node]
            etype = edge.edge_type  # now a plain str

            # 3. Confidence Threshold
            if edge.confidence_score < 0.7:
                continue

            # --- Novel type handling ---
            if edge.is_novel and etype not in active_type_names:
                # Record the observation for potential future promotion
                if self.schema_registry is not None:
                    self.schema_registry.record_observation(
                        proposed_type=etype,
                        description=edge.novel_type_description or edge.inference_reasoning,
                        from_entity=edge.source_node,
                        from_type=src_type,
                        to_entity=edge.target_node,
                        to_type=tgt_type,
                        evidence=quote_list,
                        confidence=edge.confidence_score,
                        document_key=document_key,
                    )
                    print(f"  [Inference] Recorded novel type observation: {etype}")
                # Fall through: create a RELATED_TO edge for graph connectivity
                src_coll = self.type_to_collection.get(src_type, "Concepts")
                tgt_coll = self.type_to_collection.get(tgt_type, "Concepts")
                proposed_edges.append(ProposedEdge(
                    edge_collection="RELATED_TO",
                    from_node=edge.source_node,
                    from_collection=src_coll,
                    to_node=edge.target_node,
                    to_collection=tgt_coll,
                    operation="MERGE",
                    description=f"Novel type '{etype}': {edge.inference_reasoning} (Confidence: {edge.confidence_score})",
                    evidence_quotes=quote_list,
                    weight=edge.confidence_score,
                    confidence=edge.confidence_score,
                    source_document=document_key,
                    provenance="VSA",
                    extracted_at=extracted_at,
                ))
                continue

            # --- Known type validation ---
            is_valid_schema = False
            if etype in edge_rules:
                allowed_src, allowed_tgt = edge_rules[etype]
                is_valid_schema = (src_type in allowed_src) and (tgt_type in allowed_tgt)
            elif etype == "RELATED_TO":
                # RELATED_TO is a wildcard; accept same-type pairs or any pair
                is_valid_schema = True
            
            if not is_valid_schema:
                print(f"  [Inference] Schema violation: {edge.source_node}({src_type}) -[{etype}]-> {edge.target_node}({tgt_type})")
                continue
                
            # Step D: Create ProposedEdge
            src_coll = self.type_to_collection.get(src_type, "Concepts")
            tgt_coll = self.type_to_collection.get(tgt_type, "Concepts")
            
            proposed_edges.append(ProposedEdge(
                edge_collection=etype,
                from_node=edge.source_node,
                from_collection=src_coll,
                to_node=edge.target_node,
                to_collection=tgt_coll,
                operation="MERGE",
                description=f"{edge.inference_reasoning} (Confidence: {edge.confidence_score})",
                evidence_quotes=quote_list,
                weight=edge.confidence_score,
                confidence=edge.confidence_score,
                source_document=document_key,
                provenance="VSA",
                extracted_at=extracted_at
            ))
            
        return proposed_edges

    def _phase_3_knowledge_merge(
        self, 
        known: List[NormalizedEntity], 
        document_key: str
    ) -> List[ProposedNode]:
        """
        Merge updates for known entities using LLM.
        """
        nodes = []
        
        for entity in known:
            # Base operation is MERGE/UPDATE if we have suggestions, else just reference
            # If no update, we might still want to link it to the document (handled via EXTRACTED_FROM)
            # But here we specifically look for content updates.
            
            if not entity.kb_update_suggestion:
                # No content update, just a reference.
                # We won't add a ProposedNode for *content* update, 
                # but we will rely on Phase 4 to link it.
                # However, to be explicit, let's create a MERGE node with no changes
                # so Phase 4 sees it.
                nodes.append(ProposedNode(
                    collection=entity.collection,
                    name=entity.kb_name,
                    operation="MERGE",
                    source_document=document_key,
                    metadata={"context_pack": entity.context_pack} if entity.context_pack else {}
                ))
                continue
                
            suggestion = entity.kb_update_suggestion
            if not suggestion.additions:
                nodes.append(ProposedNode(
                    collection=entity.collection,
                    name=entity.kb_name,
                    operation="MERGE",
                    source_document=document_key,
                    metadata={"context_pack": entity.context_pack} if entity.context_pack else {}
                ))
                continue

            # We have additions. Perform LLM merge.
            # 1. Get current description (simulated or fetched)
            # In a real run, we might fetch the current description from KB if not provided.
            # The suggestion object has `updated_description` potentially pre-filled by PPC,
            # but the prompt says "Use an LLM to merge...".
            # PPC Agent Phase C.5 already did some of this! 
            # "suggest_kb_updates_for_exact_matches" in entity_describer.py produces KBUpdateSuggestion.
            # If `updated_description` is already there, we can use it.
            
            final_desc = suggestion.updated_description
            
            if not final_desc:
                # If PPC didn't generate the full text, we do it here.
                # Fetch current node to get description
                curr_node = self.kb_client.find_by_name(entity.kb_name, entity.collection)
                curr_desc = curr_node.get("description", "") if curr_node else ""
                
                # LLM Call
                prompt = MERGE_USER_PROMPT_TEMPLATE.format(
                    name=entity.kb_name,
                    current_description=curr_desc,
                    additions="\n".join(f"- {a}" for a in suggestion.additions)
                )
                try:
                    # Using raw call_llm as we want text back
                    final_desc = llm_api.call_llm(
                        prompt=prompt,
                        sys_prompt=KNOWLEDGE_MERGE_SYS_PROMPT,
                        llm_api_selection=self.llm_model
                    )
                except Exception as e:
                    print(f"  [Merge] LLM merge failed for {entity.kb_name}: {e}")
                    final_desc = curr_desc + "\n" + "\n".join(suggestion.additions)

            # Create Update Node
            nodes.append(ProposedNode(
                collection=entity.collection,
                name=entity.kb_name,
                operation="UPDATE",
                description=final_desc,
                evidence_quotes=suggestion.evidence_quotes,
                key_metrics=suggestion.key_metrics,
                source_document=document_key,
                metadata={"context_pack": entity.context_pack} if entity.context_pack else {}
            ))
            
        return nodes

    def _phase_4_manifest_generation(
        self,
        document_key: str,
        nodes: List[ProposedNode],
        edges: List[ProposedEdge],
        flags: List[str]
    ) -> VSAUpdatePayload:
        """
        Assemble and validate the final manifest.
        """
        # Add EXTRACTED_FROM edges for all nodes
        doc_edges = []
        for node in nodes:
            doc_edges.append(ProposedEdge(
                edge_collection="EXTRACTED_FROM",
                from_node=node.name,
                from_collection=node.collection,
                to_node=document_key, # Assumes document key is the ID or Name
                to_collection="Documents", # Fixed collection for papers
                operation="MERGE",
                description="Entity extracted from this document",
                source_document=document_key
            ))
        
        # Combine all edges
        all_edges = edges + doc_edges
        
        # Create Payload
        payload = VSAUpdatePayload(
            document_key=document_key,
            nodes=nodes,
            edges=all_edges,
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            validation_flags=flags
        )
        
        return payload
