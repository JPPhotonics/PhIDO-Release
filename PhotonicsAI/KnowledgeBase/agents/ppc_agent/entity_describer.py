"""Entity Describer: builds richer evidence context and generates detailed descriptions.

This step sits after raw entity extraction. It:
  1) Builds a "context pack" per entity from the filtered text:
     - relevant section blocks (from [Section i: ...] markers)
     - multiple mention windows
     - numeric/unit-bearing sentences (especially for Property entities)
     - neighbor entities co-mentioned nearby
  2) Uses the LLM to produce an evidence-backed description with quotes and metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from PhotonicsAI.Photon import llm_api
from .models import RawEntity, NormalizedEntity, KBUpdateSuggestion


class KBUpdateSuggestionItem(BaseModel):
    """One KB update suggestion tied to a specific known entity."""

    raw_name: str = Field(..., description="Raw name as extracted from the paper")
    kb_name: str = Field(..., description="Matched KB entry name")
    additions: List[str] = Field(default_factory=list, description="New info from paper to add to KB (do not repeat existing KB text)")
    updated_description: Optional[str] = Field(None, description="Optional rewritten description merging KB + new info")
    evidence_quotes: List[str] = Field(default_factory=list, description="Verbatim quotes supporting additions")
    key_metrics: List[str] = Field(default_factory=list, description="Metric statements relevant to additions")


class KBUpdateSuggestionList(BaseModel):
    updates: List[KBUpdateSuggestionItem]


class EntityDescription(BaseModel):
    """Structured description of an extracted entity grounded in paper evidence."""

    name: str = Field(..., description="Entity name (must match input name exactly)")
    entity_type: str = Field(..., description="One of: Component, Architecture, Property, Design_Function, Physical_Principle")
    description: str = Field(..., description="2-6 sentence technical description grounded in provided evidence")
    evidence_quotes: List[str] = Field(default_factory=list, description="1-5 short verbatim quotes supporting key claims")
    key_metrics: List[str] = Field(default_factory=list, description="Optional list of extracted metric statements, e.g. 'Insertion loss: 4.5 dB'")
    related_entities: List[str] = Field(default_factory=list, description="Other entities mentioned in same context (subset of provided neighbor list)")
    components: List[str] = Field(
        default_factory=list,
        description=(
            "For Architecture entities ONLY: list of component entities that make up the architecture "
            "(preferably from the provided candidate list, but others are allowed if evidence supports them). "
            "Empty for non-architectures."
        ),
    )
    connectivity: List[str] = Field(
        default_factory=list,
        description=(
            "For Architecture entities ONLY: 1+ statements describing how the components are connected "
            "(e.g., 'Laser -> MZI -> photodetector', 'Coupler feeds ring resonator then output waveguide'). "
            "Empty for non-architectures."
        ),
    )


class EntityDescriptionList(BaseModel):
    entities: List[EntityDescription]


class ArchitectureClassification(BaseModel):
    """Classification of a figure caption as describing a photonic architecture."""
    is_architecture: bool = Field(..., description="True if the caption describes a photonic chip layout, circuit schematic, or device architecture")
    confidence: float = Field(..., description="Confidence score 0.0-1.0")
    reasoning: str = Field(..., description="Brief reason for classification")

class ArchitectureClassificationList(BaseModel):
    classifications: List[ArchitectureClassification]

class VLMArchitectureExtraction(BaseModel):
    """Data extracted from a schematic image by VLM."""
    components: List[str] = Field(..., description="List of specific photonic components visible in the layout")
    connectivity: List[str] = Field(..., description="Directional connectivity graph (e.g., 'Waveguide -> Ring Resonator')")
    properties: List[str] = Field(default_factory=list, description="Any properties or metrics visible in the figure (e.g., 'heater', 'high-speed')")

@dataclass
class ContextPack:
    """Evidence context pack for a single entity."""

    section_blocks: List[str]
    mention_windows: List[str]
    numeric_sentences: List[str]
    neighbor_entities: List[str]

    def render(self, max_chars: int = 12000) -> str:
        """Render to a bounded string for prompting."""
        parts: List[str] = []
        if self.section_blocks:
            parts.append("SECTION BLOCKS (full blocks where entity appears):")
            for b in self.section_blocks:
                parts.append(b.strip())
                parts.append("")
        if self.mention_windows:
            parts.append("MENTION WINDOWS (local windows around mentions):")
            for w in self.mention_windows:
                parts.append(w.strip())
                parts.append("")
        if self.numeric_sentences:
            parts.append("NUMERIC/UNIT SENTENCES (metrics/values near entity):")
            for s in self.numeric_sentences:
                parts.append(s.strip())
            parts.append("")
        if self.neighbor_entities:
            parts.append("NEIGHBOR ENTITIES (co-mentioned nearby):")
            parts.append(", ".join(self.neighbor_entities))
            parts.append("")

        text = "\n".join(parts).strip()
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n[TRUNCATED]"
        return text


class EntityDescriber:
    """Builds richer context packs and asks the LLM to generate detailed descriptions."""

    def __init__(self, llm_model: str = "gemini-2.5-pro"):
        self.llm_model = llm_model

    def enrich_architectures_with_vlm(
        self, 
        raw_entities: List[RawEntity], 
        caption_map: Dict[str, str], 
        images_path: Optional[str] = None
    ) -> List[RawEntity]:
        """
        Use VLM to extract architecture details from figures and enrich/create entities.
        
        Args:
            raw_entities: Current list of extracted entities
            caption_map: Mapping of filename -> caption text
            images_path: Root path where images are stored (optional, defaults to CWD if None)
            
        Returns:
            Updated list of RawEntity objects (including new ones from images)
        """
        import os
        from pathlib import Path
        
        print(f"VLM: Analyzing {len(caption_map)} figure captions for architecture candidates...")
        
        # 1. Classify captions to find architectures
        arch_candidates = self._classify_captions_as_architecture(caption_map)
        print(f"VLM: Found {len(arch_candidates)} architecture-relevant figures.")
        
        if not arch_candidates:
            return raw_entities
            
        # Create a map of existing architectures for fuzzy matching
        existing_archs = {e.name.lower(): e for e in raw_entities if e.entity_type == "Architecture"}
        existing_components = {e.name.lower() for e in raw_entities if e.entity_type == "Component"}
        
        new_raw_entities = list(raw_entities)
        
        for filename, caption in arch_candidates.items():
            # Resolve image path
            img_path = Path(filename)
            if not img_path.exists():
                if images_path:
                    img_path = Path(images_path) / filename
                if not img_path.exists():
                    # Try CWD
                    img_path = Path(os.getcwd()) / filename
            
            if not img_path.exists():
                print(f"VLM: Skipping {filename} - file not found.")
                continue
                
            print(f"VLM: Processing {filename}...")
            
            # 2. Extract data using VLM
            vlm_data = self._extract_architecture_from_image(img_path, caption)
            if not vlm_data:
                continue
                
            # 3. Match to existing or create new
            # Simple heuristic: Check if any existing architecture name is in the caption
            matched_entity = None
            for name, entity in existing_archs.items():
                if name in caption.lower():
                    matched_entity = entity
                    break
            
            if matched_entity:
                print(f"VLM: Augmenting existing architecture '{matched_entity.name}'")
                # We can't easily modify the RawEntity in-place safely without complex logic later,
                # but we can add a "vlm_context" field if we modified the model.
                # For now, we will append a special "VLM Augmented" context note to the entity
                # that the Describer will pick up later.
                # BETTER APPROACH: We are *before* the main description phase.
                # We can inject these components/connectivity into the entity's context field
                # so the text-based describer sees them as high-confidence evidence.
                
                vlm_context = (
                    f"\n[VLM Analysis of {filename}]\n"
                    f"Components visible: {', '.join(vlm_data.components)}\n"
                    f"Connectivity graph: {'; '.join(vlm_data.connectivity)}\n"
                )
                if matched_entity.context:
                    matched_entity.context += vlm_context
                else:
                    matched_entity.context = vlm_context
                    
            else:
                # Create NEW architecture entity
                # Name it derived from caption (e.g. "MZI Modulator (Fig 3)")
                # We need a robust way to name it. For now, use a generic name + Figure ref
                # or ask LLM to name it. Let's use a simple heuristic.
                # Try to extract a noun phrase from the caption start.
                short_name = caption.split('.')[0]
                if len(short_name) > 50:
                    short_name = short_name[:47] + "..."
                new_name = f"{short_name} ({filename})"
                
                print(f"VLM: Discovered NEW architecture '{new_name}'")
                
                vlm_context = (
                    f"Derived from figure analysis of {filename}.\n"
                    f"Caption: {caption}\n"
                    f"Components visible: {', '.join(vlm_data.components)}\n"
                    f"Connectivity graph: {'; '.join(vlm_data.connectivity)}\n"
                )
                
                new_arch = RawEntity(
                    name=new_name,
                    entity_type="Architecture",
                    context=vlm_context
                )
                new_raw_entities.append(new_arch)
            
            # 4. Component Integrity: Add any new components found by VLM
            for comp_name in vlm_data.components:
                if comp_name.lower() not in existing_components:
                    print(f"VLM: Adding new component '{comp_name}' found in figure.")
                    new_comp = RawEntity(
                        name=comp_name,
                        entity_type="Component",
                        context=f"Identified in figure {filename} as part of architecture."
                    )
                    new_raw_entities.append(new_comp)
                    existing_components.add(comp_name.lower())
                    
        return new_raw_entities

    def _classify_captions_as_architecture(self, caption_map: Dict[str, str]) -> Dict[str, str]:
        """Return subset of caption_map that describes architectures."""
        if not caption_map:
            return {}
            
        captions_list = [{"filename": k, "caption": v} for k, v in caption_map.items()]
        
        sys_prompt = """You are a technical assistant. Classify if each figure caption describes a photonic chip layout, circuit schematic, or device architecture.
        
        Return True for: "Fig 1. Schematic of MZI", "Fig 2. Chip layout", "Fig 3. Proposed architecture".
        Return False for: "Fig 4. BER measurements", "Fig 5. Simulation results", "Fig 6. Eye diagram".
        """
        
        prompt = f"""Classify these captions:
        {captions_list}
        
        Return JSON:
        {{
            "classifications": [
                {{"is_architecture": true, "confidence": 0.9, "reasoning": "..."}},
                ...
            ]
        }}
        Order must match input list.
        """
        
        try:
            result = llm_api.callgoogle_pydantic(prompt, sys_prompt, ArchitectureClassificationList)
            relevant = {}
            if result and result.classifications:
                for idx, cls in enumerate(result.classifications):
                    if idx < len(captions_list) and cls.is_architecture and cls.confidence > 0.7:
                        item = captions_list[idx]
                        relevant[item["filename"]] = item["caption"]
            return relevant
        except Exception as e:
            print(f"Warning: Caption classification failed: {e}")
            return {}

    def _extract_architecture_from_image(self, image_path, caption: str) -> Optional[VLMArchitectureExtraction]:
        """Run VLM on a single image to extract architecture details."""
        sys_prompt = """You are an expert Photonic Integrated Circuit (PIC) layout parser. 
        Your task is to analyze the waveguide layout and component structure in the image."""
        
        prompt = f"""Analyze the waveguide layout in this image. 
        Caption: "{caption}"
        
        Extract:
        1. A list of specific photonic components (e.g., "MZM", "Ring Resonator", "Grating Coupler").
        2. A directional connectivity graph: [Source Component] -> [Target Component].
           - Be specific about direction if visible (light path).
           - Connect components logically based on the schematic.
        3. Any associated properties (e.g., 'thermo-optic', 'high-speed') visible or mentioned.
        
        Return valid JSON:
        {{
          "components": ["..."],
          "connectivity": ["Comp A -> Comp B", ...],
          "properties": ["..."]
        }}
        """
        
        try:
            # Use call_google_vlm which returns text (JSON string)
            response_text = llm_api.call_google_vlm(prompt, image_path, sys_prompt)
            
            # Parse the JSON response manually since call_google_vlm doesn't support pydantic yet
            # We can reuse the JSON extraction helper from llm_api if accessible, or simple one here
            import json
            
            # Strip code fences
            cleaned_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_text)
            
            return VLMArchitectureExtraction(
                components=data.get("components", []),
                connectivity=data.get("connectivity", []),
                properties=data.get("properties", [])
            )
        except Exception as e:
            print(f"Warning: VLM extraction failed for {image_path}: {e}")
            return None

    def describe_entities(
        self,
        filtered_text: str,
        raw_entities: List[RawEntity],
        *,
        max_mentions: int = 10,
        mention_window_sentences: int = 3,
        max_section_blocks: int = 4,
    ) -> Dict[str, Dict]:
        """
        Returns a mapping raw_entity.name -> {
            "description": str,
            "evidence_quotes": [str],
            "key_metrics": [str],
            "related_entities": [str],
            "context_pack": str
        }
        """
        if not raw_entities:
            return {}

        # Build context packs first (no LLM)
        packs: Dict[str, ContextPack] = {}
        entity_names = [e.name for e in raw_entities]
        sections = self._split_into_sections(filtered_text)
        sentences = self._split_into_sentences(filtered_text)

        for ent in raw_entities:
            packs[ent.name] = self._build_context_pack(
                ent,
                raw_entities=raw_entities,
                sections=sections,
                sentences=sentences,
                max_mentions=max_mentions,
                mention_window_sentences=mention_window_sentences,
                max_section_blocks=max_section_blocks,
            )

        # Ask LLM to generate descriptions grounded in the packs
        sys_prompt = """You are an assistant to a photonic engineer.
Your task: write grounded, technical descriptions of extracted entities from a paper.

Rules:
- Use ONLY the provided evidence context. Do not invent missing details.
- Prefer concrete phrasing (what it is, what it does, how it is used, and measured metrics if present).
- Include 1-5 short verbatim evidence quotes that support your main claims.
- If you cannot find enough evidence, say so explicitly in the description.

Architecture-specific rules (IMPORTANT):
- If entity_type is "Architecture", you MUST:
  - Provide `components`: the component make-up of the architecture (preferably from the provided candidate_components list, but you may name others if the evidence clearly supports them).
  - Provide `connectivity`: detailed statements CLEARLY describing how components connect to one another to form the architecture (e.g., "The laser output couples into the MZI input arm," NOT just "The components are connected").
- Connectivity Consistency Rule:
  - The `connectivity` description must ONLY reference components listed in `components`.
  - ALL components listed in `components` must be mentioned in the `connectivity` description.
- If you cannot identify both components AND detailed connectivity set:
  - `components` = []
  - `connectivity` = []
  (A downstream dedicated decomposition phase will attempt recovery.)

Output must be valid JSON matching the requested schema."""

        # Build a compact prompt for all entities in one call to reduce cost
        all_component_names = [e.name for e in raw_entities if e.entity_type == "Component"]
        prompt_items = []
        for ent in raw_entities:
            pack_text = packs[ent.name].render()
            neighbors = packs[ent.name].neighbor_entities

            if ent.entity_type == "Architecture":
                candidates = sorted(set(neighbors + all_component_names))
            else:
                candidates = neighbors

            prompt_items.append(
                {
                    "name": ent.name,
                    "entity_type": ent.entity_type,
                    "neighbor_entities": candidates,
                    "evidence_context": pack_text,
                }
            )

        prompt = f"""Generate detailed descriptions for each extracted entity below.

Input entities (with evidence packs):
{prompt_items}

Return JSON with this shape:
{{
  "entities": [
    {{
      "name": "...",
      "entity_type": "...",
      "description": "...",
      "evidence_quotes": ["..."],
      "key_metrics": ["..."],
      "related_entities": ["..."],
      "components": ["..."],
      "connectivity": ["..."]
    }}
  ]
}}

Important:
- The 'name' MUST match the input name exactly.
- 'related_entities' must be a subset of the provided neighbor_entities for that entity.
- For entity_type == "Architecture": `components` should preferably come from the provided neighbor_entities but may include other names if the evidence clearly supports them. `connectivity` must be non-empty if evidence supports it.
"""

        out: Dict[str, Dict] = {}
        try:
            result = llm_api.callgoogle_pydantic(prompt, sys_prompt, EntityDescriptionList)
            entities = result.entities if result and hasattr(result, "entities") else []
        except Exception as e:
            print(f"Warning: Entity describer LLM call failed: {e}")
            entities = []

        # Index by name; always include at least context_pack fallback
        paper_entity_names_lower = {e.name.lower() for e in raw_entities}
        generated_by_name: Dict[str, EntityDescription] = {e.name: e for e in entities}
        for ent in raw_entities:
            pack = packs[ent.name]
            pack_text = pack.render()
            if ent.name in generated_by_name:
                d = generated_by_name[ent.name]
                # Ensure architecture related_entities includes its components (if provided)
                related = list(dict.fromkeys((d.related_entities or []) + (d.components or [])))
                is_arch = ent.entity_type == "Architecture"
                arch_components = d.components or []
                arch_connectivity = d.connectivity or []
                arch_valid = (not is_arch) or (len(arch_components) > 0 and len(arch_connectivity) > 0)

                # Post-hoc validation: flag components not found in the paper's entity list
                component_warnings: List[str] = []
                if is_arch and arch_components:
                    for comp in arch_components:
                        if comp.lower() not in paper_entity_names_lower:
                            component_warnings.append(
                                f"Component '{comp}' not found in paper entity list"
                            )

                out[ent.name] = {
                    "description": d.description,
                    "evidence_quotes": d.evidence_quotes,
                    "key_metrics": d.key_metrics,
                    "related_entities": related,
                    "context_pack": pack_text,
                    "components": arch_components,
                    "connectivity": arch_connectivity,
                    "arch_valid": arch_valid,
                    "component_warnings": component_warnings,
                }
            else:
                out[ent.name] = {
                    "description": "",
                    "evidence_quotes": [],
                    "key_metrics": [],
                    "related_entities": [],
                    "context_pack": pack_text,
                    "components": [],
                    "connectivity": [],
                    "arch_valid": ent.entity_type != "Architecture",
                    "component_warnings": [],
                }

        return out

    def suggest_kb_updates_for_exact_matches(
        self,
        *,
        known_entities: List[NormalizedEntity],
        raw_entities_context: List[dict],
        kb_client,
        max_chars_per_entity: int = 12000,
    ) -> Dict[str, KBUpdateSuggestion]:
        """
        For known entities that were marked as exact match overrides, suggest what additional context
        from the input document should be added to the existing KB entry.

        Returns mapping: raw_name -> KBUpdateSuggestion
        """
        if not known_entities:
            return {}

        # Build raw context maps (same structure PPCAgent passes to conflict detector)
        context_map = {item["name"]: item.get("context", "") for item in raw_entities_context}
        description_map = {item["name"]: item.get("description", "") for item in raw_entities_context}
        evidence_quotes_map = {item["name"]: item.get("evidence_quotes", []) for item in raw_entities_context}
        key_metrics_map = {item["name"]: item.get("key_metrics", []) for item in raw_entities_context}

        # Build prompt items with BOTH paper evidence and existing KB entry
        prompt_items = []
        for ent in known_entities:
            raw_name = ent.raw_name
            kb_name = ent.kb_name
            collection = ent.collection

            kb_entry = None
            try:
                # This returns the vertex doc (name, description, etc.) or None
                kb_entry = kb_client.find_by_name(kb_name, collection) if kb_client else None
            except Exception:
                kb_entry = None

            # Keep KB entry compact (avoid dumping embeddings or very large fields)
            kb_entry_compact = None
            if isinstance(kb_entry, dict):
                kb_entry_compact = {
                    "name": kb_entry.get("name", ""),
                    "description": kb_entry.get("description", ""),
                    "source": kb_entry.get("source", ""),
                    "type": kb_entry.get("type", ""),
                    "units": kb_entry.get("units", ""),
                    "equations": kb_entry.get("equations", []),
                }

            paper_pack = {
                "raw_name": raw_name,
                "entity_type": ent.entity_type,
                "paper_context_pack": (context_map.get(raw_name, "") or "")[:max_chars_per_entity],
                "paper_description": (description_map.get(raw_name, "") or "")[:2000],
                "paper_evidence_quotes": evidence_quotes_map.get(raw_name, [])[:8],
                "paper_key_metrics": key_metrics_map.get(raw_name, [])[:20],
                "kb_entry": kb_entry_compact,
            }
            prompt_items.append(paper_pack)

        sys_prompt = """You are an assistant to a photonic engineer maintaining a knowledge base (KB).
You will be given:
1) A KB entry (name, description, etc.) representing what the KB already knows.
2) Evidence from a new paper about the same entity (context pack + quotes + metrics).

Task:
- Identify ONLY the additional information present in the paper that is NOT already captured in the KB entry.
- Do NOT restate what is already in the KB description.
- Use ONLY the provided paper evidence; do not invent details.
- IMPORTANT: Do NOT propose adding paper-specific specifications/performance results to the KB.
  - Exclude any additions that are primarily numeric measurements, benchmark results, or "this paper demonstrates X dB / Y GHz / Z nm" type statements.
  - Do not add single-paper experimental outcomes, one-off device performance values, or dataset-specific numbers.
  - You may still include generalizable, non-numeric insights (mechanisms, causal relationships, design guidance, common uses).
- Prefer generalizable KB-friendly facts:
  - physical principle → what it enables / where it applies
  - component/architecture → design patterns, functional role, qualitative trade-offs (without numbers), configuration/structure
  - property → definition/meaning, what influences it, how it is typically improved, how it relates to other concepts (without quoting specific measured values)
- If the only "new" information in the paper is specifications/measurements, return no additions.

Output valid JSON matching the requested schema."""

        prompt = f"""For each item below, propose KB update suggestions (delta vs existing KB).

Items:
{prompt_items}

Return JSON:
{{
  "updates": [
    {{
      "raw_name": "...",
      "kb_name": "...",
      "additions": ["..."],
      "updated_description": "...",
      "evidence_quotes": ["..."],
      "key_metrics": ["..."]
    }}
  ]
}}

Rules:
- 'raw_name' must match an input raw_name exactly.
- 'kb_name' must match the input kb_entry.name (or provided kb_name) exactly.
- 'additions' must be only NEW info (delta) AND must be generalizable (no paper-specific specs/measurements).
- If there is nothing new, return empty additions and empty/omitted updated_description.
"""

        out: Dict[str, KBUpdateSuggestion] = {}
        try:
            result = llm_api.callgoogle_pydantic(prompt, sys_prompt, KBUpdateSuggestionList)
            updates = result.updates if result and hasattr(result, "updates") else []
        except Exception as e:
            print(f"Warning: KB update suggestion LLM call failed: {e}")
            updates = []

        # Map raw_name -> KBUpdateSuggestion (only for entities we asked about)
        asked_raw = {e.raw_name for e in known_entities}
        for u in updates:
            if u.raw_name not in asked_raw:
                continue
            out[u.raw_name] = KBUpdateSuggestion(
                kb_name=u.kb_name,
                additions=u.additions or [],
                updated_description=u.updated_description,
                evidence_quotes=u.evidence_quotes or [],
                key_metrics=u.key_metrics or [],
            )

        return out

    def _build_context_pack(
        self,
        ent: RawEntity,
        *,
        raw_entities: List[RawEntity],
        sections: List[Tuple[str, str]],
        sentences: List[str],
        max_mentions: int,
        mention_window_sentences: int,
        max_section_blocks: int,
    ) -> ContextPack:
        name = ent.name
        name_re = re.compile(re.escape(name), re.IGNORECASE)

        # Section blocks: include full section blocks that contain the entity name
        section_blocks: List[str] = []
        for header, block in sections:
            if name_re.search(block):
                section_blocks.append(f"[{header}]\n{block}".strip())
            if len(section_blocks) >= max_section_blocks:
                break

        # Mention windows: grab multiple windows around mentions from sentence list
        mention_windows: List[str] = []
        mention_indices = [i for i, s in enumerate(sentences) if name_re.search(s)]
        for idx in mention_indices[:max_mentions]:
            start = max(0, idx - mention_window_sentences)
            end = min(len(sentences), idx + mention_window_sentences + 1)
            window = " ".join(s.strip() for s in sentences[start:end] if s.strip())
            if window and window not in mention_windows:
                mention_windows.append(window)

        # Numeric/unit sentences: any sentence that contains the entity name AND a number/unit,
        # plus for Property entities: also collect sentences with numbers that appear near the property in text.
        numeric_sentences: List[str] = []
        unit_re = re.compile(r"(\d|\bdb\b|\bghz\b|\bnm\b|\bum\b|\bμm\b|\bv\b|\bw\b|\bma\b|\bcm\b|\bmm\b|\bps\b|\bns\b)", re.IGNORECASE)
        for s in sentences:
            if name_re.search(s) and unit_re.search(s):
                numeric_sentences.append(s.strip())
        # If it's a Property and there are no direct numeric sentences, fall back to any numeric sentence that mentions
        # a common abbreviation in parentheses, e.g. "Insertion Loss (IL)".
        if ent.entity_type == "Property" and not numeric_sentences:
            abbrev_match = re.search(r"\(([A-Za-z][A-Za-z0-9\-]{1,8})\)", name)
            if abbrev_match:
                abbr = abbrev_match.group(1)
                abbr_re = re.compile(rf"\\b{re.escape(abbr)}\\b", re.IGNORECASE)
                for s in sentences:
                    if abbr_re.search(s) and unit_re.search(s):
                        numeric_sentences.append(s.strip())

        # Neighbor entities: co-mentioned in same sentence windows
        neighbor_entities: List[str] = []
        other_names = [e.name for e in raw_entities if e.name != name]
        other_res = [(n, re.compile(re.escape(n), re.IGNORECASE)) for n in other_names]
        neighbor_set = set()
        for w in mention_windows[: min(5, len(mention_windows))]:
            for n, nre in other_res:
                if nre.search(w):
                    neighbor_set.add(n)
        neighbor_entities = sorted(neighbor_set)

        return ContextPack(
            section_blocks=section_blocks,
            mention_windows=mention_windows,
            numeric_sentences=numeric_sentences[:25],
            neighbor_entities=neighbor_entities,
        )

    def _split_into_sections(self, text: str) -> List[Tuple[str, str]]:
        """Split combined filtered text into section blocks using [Section i: ...] markers."""
        # Format from LLMDocumentPreprocessor: "[Section i: Type]" then optional "Reason:" line, then text.
        marker_re = re.compile(r"^\[Section\s+\d+:\s*(.+?)\]\s*$", re.MULTILINE)
        matches = list(marker_re.finditer(text))
        if not matches:
            return [("FullText", text)]

        blocks: List[Tuple[str, str]] = []
        for idx, m in enumerate(matches):
            header = m.group(1).strip()
            start = m.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            blocks.append((header, block))
        return blocks

    def _split_into_sentences(self, text: str) -> List[str]:
        """Very lightweight sentence splitter (good enough for context windows)."""
        # Normalize whitespace a bit
        cleaned = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
        if not cleaned:
            return []
        # Split on punctuation boundaries. Keep it simple and robust.
        parts = re.split(r"(?<=[\.\?\!])\s+", cleaned)
        return [p.strip() for p in parts if p and p.strip()]


