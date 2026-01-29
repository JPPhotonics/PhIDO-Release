"""LangChain tool wrapper for KB semantic search."""

import json
from typing import Optional

from langchain_core.tools import StructuredTool, tool

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient


def create_kb_grounding_tool(kb_client: Neo4jClient) -> StructuredTool:
    """
    Create KB_Grounding_Tool as a LangChain StructuredTool.
    
    Args:
        kb_client: Neo4jClient instance
        
    Returns:
        LangChain StructuredTool
    """
    def kb_grounding_search(
        entity_name: str,
        collection: str,
        threshold: float = 0.5
    ) -> str:
        """
        Perform semantic search on a specific collection to find the closest match 
        for an entity name extracted from a paper. Used for:
        1. Normalization (e.g., mapping 'High Loss' to 'Insertion_Loss').
        2. Conflict Detection (if similarity is low, it's a NEW concept).
        
        Args:
            entity_name: The raw extracted term from the paper
            collection: Target collection name (Components, Architectures, Properties, 
                      Design_Functions, Physical_Principles)
            threshold: Similarity threshold (default 0.5)
        
        Returns:
            JSON string containing top matches and similarity scores
        """
        # Validate collection name
        valid_collections = [
            "Components", "Architectures", "Properties",
            "Design_Functions", "Physical_Principles"
        ]
        if collection not in valid_collections:
            return json.dumps({
                "error": f"Invalid collection. Must be one of: {valid_collections}",
                "matches": []
            })
        
        try:
            # Perform semantic search
            results = kb_client.semantic_search(
                query_text=entity_name,
                collection=collection,
                limit=5,
                threshold=threshold
            )
            
            # Format results with similarity scores
            # Note: semantic_search returns vertices, but we need similarity scores
            # We'll need to call the retrieval engine directly for scores
            matches = []
            
            # Get similarity scores by calling retrieval engine directly
            if kb_client.retrieval:
                search_results = kb_client.retrieval.vector_semantic_search(
                    query_text=entity_name,
                    collection=collection,
                    limit=5,
                    threshold=threshold
                )
                matches = [
                    {
                        "name": r["vertex"]["name"],
                        "similarity": r.get("similarity", 0.0),
                        "description": r["vertex"].get("description", "")
                    }
                    for r in search_results
                ]
            else:
                # Fallback: just return entity names without scores
                matches = [
                    {
                        "name": r["name"],
                        "similarity": 0.0,
                        "description": r.get("description", "")
                    }
                    for r in results
                ]
            
            return json.dumps({
                "query": entity_name,
                "collection": collection,
                "matches": matches,
                "count": len(matches)
            })
        
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "matches": []
            })
    
    # Create LangChain tool
    tool = StructuredTool.from_function(
        func=kb_grounding_search,
        name="KB_Grounding_Tool",
        description=(
            "Performs semantic search on the knowledge base to find the closest match "
            "for an entity name. Used for normalization (mapping extracted terms to "
            "existing KB entities) and conflict detection (identifying new concepts). "
            "Returns JSON with top matches and similarity scores."
        ),
        return_direct=False
    )
    
    return tool


# Alternative: Simple function-based tool decorator
@tool
def kb_grounding_tool(
    entity_name: str,
    collection: str,
    threshold: float = 0.5,
    kb_client: Optional[Neo4jClient] = None
) -> str:
    """
    Perform semantic search on a specific collection to find the closest match 
    for an entity name extracted from a paper. Used for:
    1. Normalization (e.g., mapping 'High Loss' to 'Insertion_Loss').
    2. Conflict Detection (if similarity is low, it's a NEW concept).
    
    Input MUST be the raw extracted term and the target collection name.
    Returns a JSON string containing top matches and similarity scores.
    """
    if kb_client is None:
        return json.dumps({
            "error": "Neo4jClient not provided",
            "matches": []
        })
    
    # Validate collection name
    valid_collections = [
        "Components", "Architectures", "Properties",
        "Design_Functions", "Physical_Principles"
    ]
    if collection not in valid_collections:
        return json.dumps({
            "error": f"Invalid collection. Must be one of: {valid_collections}",
            "matches": []
        })
    
    try:
        # Get similarity scores by calling retrieval engine directly
        matches = []
        if kb_client.retrieval:
            search_results = kb_client.retrieval.vector_semantic_search(
                query_text=entity_name,
                collection=collection,
                limit=5,
                threshold=threshold
            )
            matches = [
                {
                    "name": r["vertex"]["name"],
                    "similarity": r.get("similarity", 0.0),
                    "description": r["vertex"].get("description", "")
                }
                for r in search_results
            ]
        
        return json.dumps({
            "query": entity_name,
            "collection": collection,
            "matches": matches,
            "count": len(matches)
        })
    
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "matches": []
        })

