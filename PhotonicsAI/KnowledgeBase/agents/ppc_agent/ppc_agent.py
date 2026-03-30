"""Main PPC Agent orchestrator for paper ingestion."""

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient

from .mcp_document_annotator import MCPDocumentAnnotator
from .llm_document_preprocessor import LLMDocumentPreprocessor
from .entity_extractor import EntityExtractor
from .entity_describer import EntityDescriber
from .normalizer import Normalizer
from .conflict_detector import ConflictDetector
from .models import PPCResult, RawEntity
from .acronym_agent import AcronymAgent


class PPCAgent:
    """Pre-processing & Context Agent for ingesting AMF photonics papers."""
    
    def __init__(
        self,
        kb_client: Optional[Neo4jClient] = None,
        llm_model: str = "gemini-2.5-pro"
    ):
        """
        Initialize PPC Agent.
        
        Args:
            kb_client: Neo4jClient instance (will create if None)
            llm_model: LLM model to use for extraction
        """
        # Initialize KB client
        if kb_client is None:
            kb_client = Neo4jClient()
            kb_client.connect()
        
        self.kb_client = kb_client
        
        # Initialize components
        # MCP document annotator uses subprocess-based client (no function parameter needed)
        # DISABLED: self.document_annotator = MCPDocumentAnnotator()
        # Using LLM-based preprocessor instead
        self.llm_preprocessor = LLMDocumentPreprocessor(llm_model=llm_model)
        self.entity_extractor = EntityExtractor(llm_model=llm_model)
        self.entity_describer = EntityDescriber(llm_model=llm_model)
        self.normalizer = Normalizer(kb_client=kb_client, llm_model=llm_model)
        self.conflict_detector = ConflictDetector(llm_model=llm_model)
    
    
    def process_paper(
        self,
        pdf_path: Optional[Path] = None,
        text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a paper and extract/normalize entities.
        
        Args:
            pdf_path: Path to PDF file (if processing from PDF)
            text: Raw text (if processing from text directly)
            
        Returns:
            JSON-serializable dict with known_entities and new_concepts
        """
        # Stage 0: Preprocess document to identify relevant sections
        if pdf_path:
            print(f"Preprocessing PDF with LLM-based section identification: {pdf_path}")
            try:
                # Use LLM-based preprocessor to identify relevant sections
                filtered_text = self.llm_preprocessor.preprocess_pdf(pdf_path)
                print(f"✓ LLM preprocessing successful")
                print(f"Extracted relevant text length: {len(filtered_text)} characters")
            except Exception as e:
                print(f"ℹ LLM preprocessing failed ({e}). Using fallback: direct PDF text extraction.")
                # Fallback: try to read PDF directly if LLM preprocessing fails
                try:
                    from .pdf_processor import PDFProcessor
                    pdf_processor = PDFProcessor()
                    pdf_data = pdf_processor.extract_text(pdf_path)
                    filtered_text = pdf_data["text"]
                    print(f"Fallback: Extracted text length: {len(filtered_text)} characters")
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Both LLM preprocessing and fallback PDF extraction failed. "
                        f"LLM error: {e}. Fallback error: {fallback_error}"
                    )
            
            # DISABLED: MCP Document Annotator path (kept for future use)
            # Uncomment to use MCP annotation instead of LLM preprocessing:
            # try:
            #     annotation_result = self.document_annotator.annotate_pdf(pdf_path)
            #     filtered_text = annotation_result["annotated_text"]
            #     print(f"✓ MCP annotation successful")
            # except Exception as e:
            #     print(f"ℹ MCP annotation failed ({e}). Using LLM preprocessing.")
            #     filtered_text = self.llm_preprocessor.preprocess_pdf(pdf_path)
        elif text:
            # For direct text input, preprocess with LLM to identify relevant sections
            print(f"Preprocessing provided text with LLM-based section identification...")
            try:
                filtered_text = self.llm_preprocessor.preprocess_text(text)
                print(f"✓ LLM preprocessing successful")
                print(f"Extracted relevant text length: {len(filtered_text)} characters")
            except Exception as e:
                print(f"ℹ LLM preprocessing failed ({e}). Using raw text as-is.")
                filtered_text = text
                print(f"Using raw text, length: {len(filtered_text)} characters")
        else:
            raise ValueError("Either pdf_path or text must be provided")
        
        if not filtered_text.strip():
            print("Warning: Extracted text is empty. Cannot proceed.")
            return PPCResult().to_json()
        
        # Phase A: Raw entity extraction
        print("Phase A: Extracting raw entities...")
        raw_entities = self.entity_extractor.extract_entities(filtered_text)
        print(f"Extracted {len(raw_entities)} raw entities")

        # Visual Processing (VLM) for Architectures
        # If we processed a PDF, we can use the figures to enrich architecture extraction
        # if pdf_path:
        #     try:
        #         print("Phase A.2: Visual Architecture Extraction...")
        #         # 1. Extract captions from the markdown text
        #         # We need the markdown text (from LLM preprocessor internal state or we can re-extract/pass it if needed)
        #         # Currently preprocess_pdf returns just the filtered text, which might not be the full markdown.
        #         # However, LLMDocumentPreprocessor.preprocess_pdf returns text "combined" from relevant sections.
        #         # To get captions, we might need to parse the filtered text itself if it preserved markdown image links.
        #         # Or better, let's call extract_figure_captions on the filtered_text directly.
        #         captions = self.llm_preprocessor.extract_figure_captions(filtered_text)
                
        #         if captions:
        #             print(f"Extracted {len(captions)} figure captions.")
        #             # 2. Enrich architectures using VLM
        #             raw_entities = self.entity_describer.enrich_architectures_with_vlm(
        #                 raw_entities=raw_entities,
        #                 caption_map=captions,
        #                 images_path=str(pdf_path.parent) # Assume images are in same dir as PDF
        #             )
        #         else:
        #             print("No figure captions found in filtered text.")
        #     except Exception as e:
        #         print(f"Warning: Visual architecture extraction failed: {e}")

        # Rule: entity names should not include acronyms.
        # Expand acronym tokens inside entity names (e.g., "quad TIA" -> "quad transimpedance amplifier")
        # and persist any new acronym mappings to acronym_mappings.json.
        try:
            acronym_agent = AcronymAgent(llm_model=self.entity_extractor.llm_model)  # type: ignore[attr-defined]
            rewritten: List[RawEntity] = []
            seen = set()
            rewrites = 0
            for e in raw_entities:
                expanded, new_maps = acronym_agent.expand_acronyms_in_phrase(e.name, context=e.context or "")
                name2 = expanded.strip()
                if name2 and name2 != e.name:
                    rewrites += 1
                # De-dupe after expansion by (name, entity_type)
                key = (name2, e.entity_type)
                if key in seen:
                    continue
                seen.add(key)
                rewritten.append(RawEntity(name=name2, entity_type=e.entity_type, context=e.context))
            if rewrites:
                print(f"Phase A.1: Expanded acronyms inside {rewrites} entity names (mappings saved to acronym_mappings.json)")
            raw_entities = rewritten
        except Exception as e:
            print(f"Warning: Acronym expansion step failed: {e}")
        
        if not raw_entities:
            print("Warning: No entities extracted. Returning empty result.")
            return PPCResult().to_json()
        
        # Phase A.5: Entity descriptions (richer evidence + LLM summary)
        print("Phase A.5: Generating detailed entity descriptions...")
        described = self.entity_describer.describe_entities(filtered_text, raw_entities)
        print(f"Generated descriptions for {len(described)} entities")

        # Enforce Architecture quality:
        # - Architecture descriptions must include component make-up + connectivity, otherwise discard the architecture.
        # - Components that make up architectures must also exist as Component entities.
        try:
            # Track which raw entities are components by name
            component_names = {e.name for e in raw_entities if e.entity_type == "Component"}

            filtered_entities: List[RawEntity] = []
            discarded_arch = 0
            added_components = 0

            for e in raw_entities:
                if e.entity_type != "Architecture":
                    filtered_entities.append(e)
                    continue

                d = described.get(e.name, {}) if isinstance(described, dict) else {}
                arch_valid = bool(d.get("arch_valid", False))
                if not arch_valid:
                    discarded_arch += 1
                    continue

                # Ensure listed architecture components exist as Component entities
                arch_components = d.get("components", []) or []
                for c in arch_components:
                    if c and c not in component_names:
                        filtered_entities.append(RawEntity(name=c, entity_type="Component", context=f"Subcomponent of architecture '{e.name}'"))
                        component_names.add(c)
                        added_components += 1

                filtered_entities.append(e)

            if discarded_arch or added_components:
                print(f"Phase A.6: Architecture validation discarded {discarded_arch} architectures; added {added_components} missing components")
            raw_entities = filtered_entities
        except Exception as e:
            print(f"Warning: Architecture validation step failed: {e}")

        # Phase A.7: Dedicated Architecture Decomposition
        # Re-attempt structural decomposition for all Architecture entities using a
        # specialised prompt that scans the full paper for evidence.
        try:
            arch_entities = [e for e in raw_entities if e.entity_type == "Architecture"]
            if arch_entities:
                from .architecture_decomposer import decompose_architectures

                comp_entities = [e for e in raw_entities if e.entity_type == "Component"]
                templates = decompose_architectures(
                    arch_entities,
                    comp_entities,
                    filtered_text,
                    described,
                    self.kb_client,
                )
                upgraded = 0
                for name, tmpl in templates.items():
                    if name in described:
                        described[name]["architecture_template"] = tmpl.model_dump()
                        described[name]["components"] = [
                            r.component_type for r in tmpl.component_roles
                        ]
                        described[name]["connectivity"] = [
                            f"{c.from_role}.{c.from_port} -> {c.to_role}.{c.to_port}"
                            for c in tmpl.connections
                        ]
                        described[name]["arch_valid"] = True
                        upgraded += 1
                if upgraded:
                    print(f"Phase A.7: Decomposed {upgraded}/{len(arch_entities)} architectures into structured templates")
        except Exception as e:
            print(f"Warning: Architecture decomposition step (A.7) failed: {e}")

        # Phase B: Normalization
        print("Phase B: Normalizing entities against KB...")
        # Convert describer object to dict if it isn't already (it might be EntityDescriptionList)
        descriptions_map = described
        if hasattr(described, "entities"):
             # If it's a Pydantic list wrapper, map it: name -> dict
             descriptions_map = {d.name: d.dict() for d in described.entities}
        elif isinstance(described, list):
             descriptions_map = {d["name"]: d for d in described}
             
        normalized_entities = self.normalizer.normalize_entities(raw_entities, descriptions=descriptions_map)
        print(f"Normalized {len(normalized_entities)} entities")

        # Prepare context/description for conflict detection
        # - context: evidence pack (sections + mention windows + numeric sentences + neighbors)
        # - description: LLM-generated technical summary
        raw_entities_context = []
        for e in raw_entities:
            d = described.get(e.name, {})
            raw_entities_context.append(
                {
                    "name": e.name,
                    "entity_type": e.entity_type,
                    "context": d.get("context_pack") or (e.context or ""),
                    "description": d.get("description") or "",
                    "evidence_quotes": d.get("evidence_quotes") or [],
                    "key_metrics": d.get("key_metrics") or [],
                    "related_entities": d.get("related_entities") or [],
                    # Architecture structure (if available)
                    "components": d.get("components") or [],
                    "connectivity": d.get("connectivity") or [],
                }
            )
        
        # Phase C: Conflict detection
        print("Phase C: Detecting conflicts and new concepts...")
        result = self.conflict_detector.categorize_entities(
            normalized_entities,
            raw_entities_context
        )
        
        # Optional enrichment: for known entities confirmed via exact-name override (similarity forced to 1.0),
        # suggest what additional context from this document should be added to the existing KB entry.
        try:
            exact_known = [e for e in result.known_entities if getattr(e, "exact_match_override", False)]
            if exact_known:
                print(f"Phase C.5: Suggesting KB updates for {len(exact_known)} exact-match known entities...")
                suggestions = self.entity_describer.suggest_kb_updates_for_exact_matches(
                    known_entities=exact_known,
                    raw_entities_context=raw_entities_context,
                    kb_client=self.kb_client,
                )
                for e in result.known_entities:
                    if getattr(e, "exact_match_override", False) and e.raw_name in suggestions:
                        e.kb_update_suggestion = suggestions[e.raw_name]
        except Exception as e:
            print(f"Warning: KB update suggestion step failed: {e}")

        # Final Sanity Check: Filter known entities with missing update suggestions or descriptions
        # If a known entity doesn't have a valid suggestion or description update, it shouldn't pollute the output.
        filtered_known = []
        discarded_known = 0
        for e in result.known_entities:
            has_suggestion = bool(e.kb_update_suggestion)
            has_desc = bool(e.kb_update_suggestion and e.kb_update_suggestion.updated_description)
            
            # Keep if it has a suggestion AND that suggestion has a description (or at least additions)
            # User request: "null updated_description or a null kb_update_suggestion" -> remove
            if has_suggestion and e.kb_update_suggestion.updated_description:
                filtered_known.append(e)
            else:
                discarded_known += 1
        
        if discarded_known > 0:
            print(f"Phase C.6: Filtered {discarded_known} known entities missing update suggestions/descriptions.")
        result.known_entities = filtered_known

        print(f"Final Results: {len(result.known_entities)} known entities, {len(result.new_concepts)} new concepts")

        # Attach architecture structure fields to final outputs (known/new) if present in describer outputs.
        try:
            described_map = described if isinstance(described, dict) else {}
            for e in result.known_entities:
                if getattr(e, "entity_type", "") == "Architecture":
                    d = described_map.get(e.raw_name, {}) or {}
                    e.components = d.get("components") or []
                    e.connectivity = d.get("connectivity") or []
            for c in result.new_concepts:
                if getattr(c, "entity_type", "") == "Architecture":
                    d = described_map.get(c.name, {}) or {}
                    c.components = d.get("components") or []
                    c.connectivity = d.get("connectivity") or []
        except Exception as e:
            print(f"Warning: Failed to attach architecture structure to outputs: {e}")
        
        # Phase D: Return structured JSON
        return result.to_json()
    
    def process_paper_to_json(
        self,
        pdf_path: Optional[Path] = None,
        text: Optional[str] = None
    ) -> str:
        """
        Process paper and return JSON string.
        
        Args:
            pdf_path: Path to PDF file
            text: Raw text
            
        Returns:
            JSON string
        """
        result = self.process_paper(pdf_path=pdf_path, text=text)
        return json.dumps(result, indent=2)
    
    def process_paper_to_result(
        self,
        pdf_path: Optional[Path] = None,
        text: Optional[str] = None
    ) -> PPCResult:
        """
        Process paper and return PPCResult object.
        
        Args:
            pdf_path: Path to PDF file
            text: Raw text
            
        Returns:
            PPCResult object
        """
        result_dict = self.process_paper(pdf_path=pdf_path, text=text)
        # Reconstruct PPCResult from dict
        from .models import NormalizedEntity, NewConcept
        known = [NormalizedEntity(**e) for e in result_dict.get("known_entities", [])]
        new = [NewConcept(**c) for c in result_dict.get("new_concepts", [])]
        return PPCResult(known_entities=known, new_concepts=new)

