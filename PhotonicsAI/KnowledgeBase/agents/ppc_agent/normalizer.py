"""Phase B: Context retrieval and normalization using Vector Search."""

import json
from typing import List, Optional, Dict

from .models import RawEntity, NormalizedEntity
from .kb_grounding_tool import create_kb_grounding_tool
from .acronym_agent import AcronymAgent


class Normalizer:
    """Normalize entities against knowledge base using Vector Search."""
    
    def __init__(self, kb_client, llm_model: str = "gemini-2.5-pro"):
        """
        Initialize normalizer.
        
        Args:
            kb_client: Neo4jClient instance
            llm_model: LLM model to use (default: gemini-2.5-pro)
        """
        self.kb_client = kb_client
        self.llm_model = llm_model
        self.kb_tool = create_kb_grounding_tool(kb_client)
        self.acronym_agent = AcronymAgent(llm_model=llm_model)
    
    def normalize_entities(self, raw_entities: List[RawEntity], descriptions: Optional[Dict[str, Dict]] = None) -> List[NormalizedEntity]:
        """
        Normalize raw entities against knowledge base using Vector Search.
        
        Args:
            raw_entities: List of raw extracted entities
            descriptions: Optional map of entity_name -> description dict (from EntityDescriber)
            
        Returns:
            List of normalized entities with KB matches.
        """
        normalized = []
        descriptions = descriptions or {}
        
        for entity in raw_entities:
            # Map entity type to collection name
            collection = self._entity_type_to_collection(entity.entity_type)
            if not collection:
                continue
            
            # Resolve acronyms if applicable (e.g. "MZI" -> "Mach-Zehnder Interferometer")
            canonical_name = self._canonicalize_name(entity.name)
            query_name = self.acronym_agent.resolve(canonical_name, context=entity.context or "")
            if query_name != entity.name:
                print(f"    Debug: Resolved acronym '{entity.name}' -> '{query_name}'")
            
            # Construct search query
            # If we have a rich description, append it to the name for better semantic matching
            query_text = query_name
            desc_data = descriptions.get(entity.name, {})
            description = desc_data.get("description", "")
            
            if description:
                # Limit description length to avoid diluting the name too much, 
                # though Qwen can handle long context.
                # A simple concatenation "Name. Description" works well.
                query_text = f"{query_name}. {description}"
            
            # Vector Search
            try:
                vector_results = self._vector_search(query_text, collection)
                
                if vector_results:
                    # Get best match
                    best_match = vector_results[0]
                    normalized.append(NormalizedEntity(
                        raw_name=entity.name,
                        kb_name=best_match["name"],
                        similarity=best_match.get("similarity", 0.0),
                        entity_type=entity.entity_type,
                        collection=collection,
                        top_candidates=vector_results  # Store all candidates
                    ))
                else:
                    # No match found - will be handled in conflict detection
                    normalized.append(NormalizedEntity(
                        raw_name=entity.name,
                        kb_name="",
                        similarity=0.0,
                        entity_type=entity.entity_type,
                        collection=collection,
                        top_candidates=[]
                    ))
            
            except Exception as e:
                print(f"Error normalizing {entity.name}: {e}")
                # Continue with next entity
                normalized.append(NormalizedEntity(
                    raw_name=entity.name,
                    kb_name="",
                    similarity=0.0,
                    entity_type=entity.entity_type,
                    collection=collection,
                    top_candidates=[]
                ))
        
        return normalized

    def _vector_search(self, query: str, collection: str, limit: int = 10) -> List[Dict]:
        """Perform semantic vector search using the KB tool."""
        try:
            tool_result = self.kb_tool.invoke({
                "entity_name": query,
                "collection": collection,
                "threshold": 0.3
            })
            result_data = json.loads(tool_result)
            matches = result_data.get("matches", [])
            print(f"    Debug: Vector search '{query}' in {collection} found {len(matches)} matches")
            if matches:
                top = matches[0]
                print(f"      Top vector match: {top.get('name')} (sim: {top.get('similarity', 0.0):.3f})")
            return matches
        except Exception as e:
            print(f"    Debug: Vector search error: {e}")
            return []
    
    def _entity_type_to_collection(self, entity_type: str) -> Optional[str]:
        """Map entity type to KB collection name."""
        mapping = {
            "Component": "Components",
            "Architecture": "Architectures",
            "Property": "Properties",
            "Design_Function": "Design_Functions",
            "Physical_Principle": "Physical_Principles"
        }
        return mapping.get(entity_type)

    def _canonicalize_name(self, name: str) -> str:
        """Lightweight canonicalization for surface-form variants."""
        if not name:
            return name
        # Normalize whitespace and dash variants for more stable matching.
        normalized = " ".join(name.strip().split())
        normalized = normalized.replace("–", "-").replace("—", "-")
        return normalized
