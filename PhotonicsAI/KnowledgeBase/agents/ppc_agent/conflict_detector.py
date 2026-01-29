"""Phase C: Conflict and novelty identification."""

import json
import re
from typing import List

from PhotonicsAI.Photon import llm_api
from .models import NormalizedEntity, NewConcept, PPCResult
from .ontology_loader import OntologyLoader


class ConflictDetector:
    """Detect conflicts and identify new concepts."""
    
    def __init__(self, llm_model: str = "gemini-2.5-pro", ontology_path: str = "PhotonicsAI/KnowledgeBase/GenerativeOntology/ontology/pic_ontology.ttl"):
        """
        Initialize conflict detector.
        
        Args:
            llm_model: LLM model to use
        """
        self.llm_model = llm_model
        try:
            loader = OntologyLoader(ontology_path)
            self.schema = loader.get_ontology_schema()
        except Exception as e:
            print(f"Warning: Failed to load ontology schema: {e}")
            self.schema = {"classes": {}, "constraints": {}}
    
    def categorize_entities(
        self,
        normalized_entities: List[NormalizedEntity],
        raw_entities_context: List[dict]
    ) -> PPCResult:
        """
        Categorize entities into known facts and new concepts.
        
        Args:
            normalized_entities: List of normalized entities with similarity scores
            raw_entities_context: List of raw entities with context for new concepts
            
        Returns:
            PPCResult with known_entities and new_concepts
        """
        known_entities = []
        new_concepts = []
        
        # Thresholds
        HIGH_SIMILARITY = 0.9  # Known entity (raised from 0.8)
        LOW_SIMILARITY = 0.7   # New concept (raised from 0.4)
        
        # Create mapping from raw name to context/description and describer artifacts
        # context: evidence pack (preferred) or brief extractor context
        context_map = {item["name"]: item.get("context", "") for item in raw_entities_context}
        description_map = {item["name"]: item.get("description", "") for item in raw_entities_context}
        evidence_quotes_map = {item["name"]: item.get("evidence_quotes", []) for item in raw_entities_context}
        key_metrics_map = {item["name"]: item.get("key_metrics", []) for item in raw_entities_context}
        related_entities_map = {item["name"]: item.get("related_entities", []) for item in raw_entities_context}
        components_map = {item["name"]: item.get("components", []) for item in raw_entities_context}
        connectivity_map = {item["name"]: item.get("connectivity", []) for item in raw_entities_context}
        
        ambiguous_entities = []
        
        for norm_entity in normalized_entities:
            # Exact match boost: if names match case-insensitively, force high confidence
            if norm_entity.kb_name and norm_entity.raw_name.lower().strip() == norm_entity.kb_name.lower().strip():
                print(f"    Debug: Exact match override for '{norm_entity.raw_name}' -> '{norm_entity.kb_name}'")
                norm_entity.similarity = 1.0
                norm_entity.exact_match_override = True
            
            similarity = norm_entity.similarity

            # Logic validation flags (ontology / physics)
            logic_flags = self._logic_validate(norm_entity, context_map, key_metrics_map)
            if logic_flags:
                # Treat as hallucination/new concept, bypassing similarity
                new_concepts.append(NewConcept(
                    name=norm_entity.raw_name,
                    entity_type=norm_entity.entity_type,
                    description=f"Flagged: {', '.join(logic_flags)}",
                    context=context_map.get(norm_entity.raw_name, ""),
                    evidence_quotes=[],
                    key_metrics=[],
                    related_entities=[],
                    components=components_map.get(norm_entity.raw_name, []) or [],
                    connectivity=connectivity_map.get(norm_entity.raw_name, []) or [],
                    top_candidates=norm_entity.top_candidates,
                ))
                continue
            
            if similarity >= HIGH_SIMILARITY:
                # Known entity - successfully normalized
                norm_entity.context_pack = context_map.get(norm_entity.raw_name, "")
                known_entities.append(norm_entity)
            
            elif similarity < LOW_SIMILARITY:
                # New concept - no good match
                new_concepts.append(NewConcept(
                    name=norm_entity.raw_name,
                    entity_type=norm_entity.entity_type,
                    description=description_map.get(norm_entity.raw_name, "") or context_map.get(norm_entity.raw_name, ""),
                    context=context_map.get(norm_entity.raw_name, ""),
                    evidence_quotes=evidence_quotes_map.get(norm_entity.raw_name, []) or [],
                    key_metrics=key_metrics_map.get(norm_entity.raw_name, []) or [],
                    related_entities=related_entities_map.get(norm_entity.raw_name, []) or [],
                    components=components_map.get(norm_entity.raw_name, []) or [],
                    connectivity=connectivity_map.get(norm_entity.raw_name, []) or [],
                    top_candidates=norm_entity.top_candidates,
                ))
            
            else:
                # Ambiguous - similarity in 0.4-0.8 range
                ambiguous_entities.append(norm_entity)
        
        # Use LLM to resolve ambiguous cases
        if ambiguous_entities:
            resolved = self._resolve_ambiguous_entities(
                ambiguous_entities,
                context_map,
                description_map,
                evidence_quotes_map,
                key_metrics_map,
                related_entities_map,
                components_map,
                connectivity_map,
            )
            known_entities.extend(resolved["known"])
            new_concepts.extend(resolved["new"])
        
        # Deduplicate known_entities based on KB name to avoid repetitive listing
        unique_known_entities = []
        seen_kb_names = set()
        for entity in known_entities:
            if entity.kb_name not in seen_kb_names:
                seen_kb_names.add(entity.kb_name)
                unique_known_entities.append(entity)

        # Deduplicate new_concepts based on case-insensitive name
        unique_new_concepts = []
        seen_new_names = set()
        for concept in new_concepts:
            name_lower = concept.name.lower().strip()
            if name_lower not in seen_new_names:
                seen_new_names.add(name_lower)
                unique_new_concepts.append(concept)
        
        return PPCResult(
            known_entities=unique_known_entities,
            new_concepts=unique_new_concepts
        )
    
    def _resolve_ambiguous_entities(
        self,
        ambiguous: List[NormalizedEntity],
        context_map: dict,
        description_map: dict,
        evidence_quotes_map: dict,
        key_metrics_map: dict,
        related_entities_map: dict,
        components_map: dict,
        connectivity_map: dict,
    ) -> dict:
        """
        Use LLM to resolve ambiguous entities (similarity 0.4-0.8).
        
        Returns:
            Dict with 'known' and 'new' lists
        """
        sys_prompt = """You are an assistant to a photonic engineer.
Your task is to determine if an extracted entity from a paper is:
1. A known entity that matches an existing knowledge base entry (even if similarity is moderate)
2. A new concept that should be added to the knowledge base

Consider:
- If the entity name is very similar to the KB match (>0.6 similarity), it's likely a known entity
- If the entity has unique characteristics or is described as "novel", "new", "proposed", it's likely new
- If the context suggests it's a variation or different implementation, it might be new
- If it's just a different name for the same thing, it's known
"""
        
        ambiguous_list = []
        for entity in ambiguous:
            ambiguous_list.append({
                "raw_name": entity.raw_name,
                "kb_match": entity.kb_name,
                "similarity": entity.similarity,
                "entity_type": entity.entity_type,
                "context": context_map.get(entity.raw_name, "")
            })
        
        prompt = f"""For each of these ambiguous entities (similarity 0.4-0.8), determine if it's a known entity or new concept:

{json.dumps(ambiguous_list, indent=2)}

For each entity, respond with:
- "known" if it matches the KB entry (even if similarity is moderate)
- "new" if it's a novel concept to add
If any physics/ontology violation is evident, choose "new".

Return JSON in format:
{{
  "entities": [
    {{"raw_name": "...", "decision": "known"}},
    {{"raw_name": "...", "decision": "new"}}
  ]
}}
"""
        
        try:
            response = llm_api.call_llm(prompt, sys_prompt, self.llm_model)
            
            # Parse response
            json_str = self._extract_json_from_text(response)
            if json_str:
                data = json.loads(json_str)
                decisions = {item["raw_name"]: item["decision"] for item in data.get("entities", [])}
                
                known = []
                new = []
                
                for entity in ambiguous:
                    decision = decisions.get(entity.raw_name, "new")  # Default to new if unclear
                    
                    if decision == "known":
                        entity.context_pack = context_map.get(entity.raw_name, "")
                        known.append(entity)
                    else:
                        new.append(NewConcept(
                            name=entity.raw_name,
                            entity_type=entity.entity_type,
                            description=description_map.get(entity.raw_name, "") or context_map.get(entity.raw_name, ""),
                            context=context_map.get(entity.raw_name, ""),
                            evidence_quotes=evidence_quotes_map.get(entity.raw_name, []) or [],
                            key_metrics=key_metrics_map.get(entity.raw_name, []) or [],
                            related_entities=related_entities_map.get(entity.raw_name, []) or [],
                            components=components_map.get(entity.raw_name, []) or [],
                            connectivity=connectivity_map.get(entity.raw_name, []) or [],
                            top_candidates=entity.top_candidates,
                        ))
                
                return {"known": known, "new": new}
        
        except Exception as e:
            print(f"Error resolving ambiguous entities: {e}")
            # Default: treat all ambiguous as new concepts (safer)
            return {
                "known": [],
                "new": [
                    NewConcept(
                        name=e.raw_name,
                        entity_type=e.entity_type,
                        description=description_map.get(e.raw_name, "") or context_map.get(e.raw_name, ""),
                        context=context_map.get(e.raw_name, ""),
                        evidence_quotes=evidence_quotes_map.get(e.raw_name, []) or [],
                        key_metrics=key_metrics_map.get(e.raw_name, []) or [],
                        related_entities=related_entities_map.get(e.raw_name, []) or [],
                        components=components_map.get(e.raw_name, []) or [],
                        connectivity=connectivity_map.get(e.raw_name, []) or [],
                        top_candidates=e.top_candidates,
                    )
                    for e in ambiguous
                ]
            }
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other formatting."""
        import json
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            text = '\n'.join(lines)
        
        # Try to find JSON object
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        return text

    def _logic_validate(self, norm_entity: NormalizedEntity, context_map: dict, key_metrics_map: dict) -> List[str]:
        """Lightweight ontology-based checks; returns list of violation flags."""
        flags = []
        constraints = self.schema.get("constraints", {})
        if not isinstance(constraints, dict):
            constraints = {}
        
        # Removed "NotInOntology" check as it's not appropriate for A-Box instances vs T-Box schema.

        # Example physics/ontology sanity checks
        name_lower = norm_entity.raw_name.lower()
        ctx_lower = context_map.get(norm_entity.raw_name, "").lower()

        # If a property suggests gain on passive device (very lightweight heuristic)
        if norm_entity.entity_type in {"Component", "Architecture"} and "gain" in name_lower:
            flags.append("GainOnComponent")
        if norm_entity.entity_type == "Property" and "gain" in name_lower and "passive" in ctx_lower:
            flags.append("PassiveDeviceGainConflict")

        # Minimal numeric conflict check within extracted key metrics
        key_metrics = key_metrics_map.get(norm_entity.raw_name, []) or []
        metric_conflicts = self._detect_metric_conflicts(key_metrics)
        flags.extend(metric_conflicts)

        # Disjoint class sanity: if kb_name belongs to class disjoint with a hinted context term
        disjoint_pairs = constraints.get("disjoint_pairs", [])
        for a, b in disjoint_pairs:
            if a.lower() in ctx_lower and norm_entity.kb_name == b:
                flags.append(f"DisjointWithContext:{a}-{b}")
            if b.lower() in ctx_lower and norm_entity.kb_name == a:
                flags.append(f"DisjointWithContext:{b}-{a}")

        return flags

    def _detect_metric_conflicts(self, key_metrics: List[str]) -> List[str]:
        """Detect conflicting numeric values for the same metric name."""
        if not key_metrics:
            return []

        metric_values = {}
        pattern = re.compile(r"(?P<metric>[A-Za-z][A-Za-z0-9_ \-\/]+?)\s*[:=]\s*(?P<value>[-+]?\d+(?:\.\d+)?)")

        for metric in key_metrics:
            if not metric:
                continue
            match = pattern.search(metric)
            if not match:
                continue
            metric_name = " ".join(match.group("metric").lower().strip().split())
            value = float(match.group("value"))
            metric_values.setdefault(metric_name, set()).add(round(value, 6))

        conflicts = []
        for metric_name, values in metric_values.items():
            if len(values) > 1:
                conflicts.append(f"MetricConflict:{metric_name}")

        return conflicts
