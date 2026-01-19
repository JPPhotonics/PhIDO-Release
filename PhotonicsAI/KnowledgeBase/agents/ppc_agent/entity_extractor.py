"""Phase A: Raw entity extraction from filtered text."""

import json
from typing import List, Dict, Any

from pydantic import BaseModel

from PhotonicsAI.Photon import llm_api
from .models import RawEntity
from .ontology_loader import OntologyLoader

class RawEntityList(BaseModel):
    """Pydantic model for raw entity extraction output."""
    entities: List[RawEntity]


class EntityExtractor:
    """Extract raw entities from filtered text using LLM."""
    
    def __init__(self, llm_model: str = "gemini-2.5-pro", ontology_path: str = "PhotonicsAI/KnowledgeBase/GenerativeOntology/ontology/pic_ontology.ttl"):
        """
        Initialize entity extractor.
        
        Args:
            llm_model: LLM model to use for extraction
            ontology_path: Path to OWL ontology file
        """
        self.llm_model = llm_model
        try:
            loader = OntologyLoader(ontology_path)
            self.schema = loader.get_ontology_schema()
        except Exception as e:
            print(f"Warning: Failed to load ontology schema: {e}")
            self.schema = {"classes": {}, "constraints": {}}
        self.allowed_high_level = {
            "Component",
            "Architecture",
            "Property",
            "Design_Function",
            "Physical_Principle"
        }
        self.ontology_classes = set(self.schema.get("classes", {}).keys())
    
    def extract_entities(self, filtered_text: str) -> List[RawEntity]:
        """
        Extract raw entities from filtered text.
        
        Args:
            filtered_text: Filtered text chunk from Stage 0
            
        Returns:
            List of RawEntity objects
        """
        # Format class hierarchy for prompt
        class_info = ""
        if self.schema and self.schema.get("classes"):
            class_info = "\nOntology Class Hierarchy (use these specific classes where possible):\n"
            for cls, details in self.schema["classes"].items():
                parents = ", ".join(details.get("parents", []))
                comment = details.get("description", "")
                class_info += f"- {cls} (Subclass of: {parents}): {comment}\n"
        
        sys_prompt = f"""You are an assistant to a photonic engineer.
Your task is to extract technical entities from the provided text chunk, which has already been filtered for relevance.

{class_info}

From the following highly relevant text chunk, list all candidates for:
1. Components: Individual photonic components (e.g., MZM, Grating_Coupler, Phase_Shifter)
2. Architectures: Composite photonic structures composed of components (e.g., Optical_Transceiver, LiDAR_System)
3. Properties: Physical properties (e.g., Insertion_Loss, Bandwidth, Q_Factor)
4. Design_Functions: Functional capabilities (e.g., Modulation, Filtering, Switching)
5. Physical_Principles: Underlying physical principles (e.g., Plasma_Dispersion, Thermo_Optic_Effect)

Exclusion Rules:
- DO NOT extract generic terms like "Photonic Chip", "Photonic Circuit", "PIC", "Device", "System", "Structure", "Platform", "Technology".

For each entity:
- Extract the exact name as it appears in the text.
- **Granularity Rule**: If the text mentions a specific class from the Ontology Hierarchy (e.g., 'Mach-Zehnder Interferometer'), classify it as the specific class (e.g., 'MZI') rather than a generic parent class (e.g., 'Modulator').
- Classify the entity type using Zero-Shot Chain-of-Thought reasoning.
- Include brief context if helpful.
- The entity_type MUST be one of: Component, Architecture, Property, Design_Function, Physical_Principle.
- Prefer class labels that exist in the ontology when possible: {", ".join(sorted(list(self.ontology_classes)))}.

Return a JSON list of entities with 'name', 'entity_type', and optional 'context' fields.
The 'entity_type' MUST correspond to one of the 5 high-level categories: Component, Architecture, Property, Design_Function, Physical_Principle.
"""
        
        prompt = f"""Extract all technical entities from this text chunk:

{filtered_text}

Return a JSON list of entities. Each entity should have:
- name: The exact entity name from the text
- entity_type: One of Component, Architecture, Property, Design_Function, Physical_Principle
- context: (optional) Brief context where found

Example format:
[
  {{"name": "Mach-Zehnder Interferometer", "entity_type": "Architecture", "context": "proposed modulator design"}},
  {{"name": "Insertion Loss", "entity_type": "Property", "context": "measured at 1.5 dB"}},
  {{"name": "Plasma Dispersion Effect", "entity_type": "Physical_Principle", "context": "used for phase modulation"}}
]
"""
        
        try:
            # Use Pydantic model for structured output
            result = llm_api.callgoogle_pydantic(prompt, sys_prompt, RawEntityList)
            
            if result and hasattr(result, 'entities'):
                return self._post_validate(result.entities)
            else:
                # Fallback: try to parse JSON from LLM response
                return self._post_validate(self._parse_json_fallback(prompt, sys_prompt))
        
        except Exception as e:
            print(f"Error in entity extraction: {e}")
            # Fallback to non-structured extraction
            return self._post_validate(self._parse_json_fallback(prompt, sys_prompt))
    
    def _parse_json_fallback(self, prompt: str, sys_prompt: str) -> List[RawEntity]:
        """Fallback method using regular LLM call and JSON parsing."""
        try:
            response = llm_api.call_llm(prompt, sys_prompt, self.llm_model)
            
            # Try to extract JSON from response
            json_str = self._extract_json_from_text(response)
            
            if json_str:
                data = json.loads(json_str)
                if isinstance(data, list):
                    entities = []
                    for item in data:
                        if isinstance(item, dict) and 'name' in item and 'entity_type' in item:
                            entities.append(RawEntity(
                                name=item['name'],
                                entity_type=item['entity_type'],
                                context=item.get('context')
                            ))
                    return entities
            
            return []
        
        except Exception as e:
            print(f"Error in fallback extraction: {e}")
            return []

    def _post_validate(self, entities: List[RawEntity]) -> List[RawEntity]:
        """Validate entity types and filter to ontology-aware categories."""
        validated = []
        for ent in entities:
            if ent.entity_type not in self.allowed_high_level:
                continue
            # Optionally nudge name toward ontology labels if an exact match exists
            if self.ontology_classes and ent.name in self.ontology_classes:
                validated.append(ent)
            else:
                validated.append(ent)
        return validated
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from text that may contain markdown or other formatting."""
        # Remove markdown code blocks
        text = text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].startswith('```'):
                lines = lines[:-1]
            text = '\n'.join(lines)
        
        # Try to find JSON array
        start_idx = text.find('[')
        end_idx = text.rfind(']')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx:end_idx + 1]
        
        return text

