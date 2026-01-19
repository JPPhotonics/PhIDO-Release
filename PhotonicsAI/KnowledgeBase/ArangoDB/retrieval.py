"""Retrieval functions for querying the knowledge base."""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from arango.database import StandardDatabase
from sentence_transformers import SentenceTransformer

from .config import ArangoDBConfig


class RetrievalEngine:
    """Engine for various retrieval operations."""
    
    def __init__(self, db: StandardDatabase, config: ArangoDBConfig):
        """Initialize retrieval engine."""
        self.db = db
        self.config = config
        self.embedding_model = None
        self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load embedding model for query encoding."""
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
    
    def get_direct_neighbors(self, vertex_key: str, collection: str,
                            edge_types: Optional[List[str]] = None,
                            max_depth: int = 1) -> List[Dict[str, Any]]:
        """
        Retrieve direct neighbors of a vertex.
        
        Args:
            vertex_key: The _key of the vertex
            collection: Collection name containing the vertex
            edge_types: List of edge collection names to traverse (None = all)
            max_depth: Maximum traversal depth (1 = direct neighbors only)
        
        Returns:
            List of neighbor vertex documents
        """
        if max_depth < 1:
            max_depth = 1
        
        neighbors = []
        visited = set()
        
        # Get all edge collections if not specified
        if edge_types is None:
            edge_types = ["PERFORMS_FUNCTION", "BASED_ON_PRINCIPLE", "HAS_PROPERTY",
                         "USES_COMPONENT", "RELATED_TO", "EXTRACTED_FROM"]
        
        vertex_id = f"{collection}/{vertex_key}"
        
        # Query outgoing edges
        for edge_type in edge_types:
            # Check if collection exists
            if not self.db.has_collection(edge_type):
                continue
            
            # Find edges from this vertex
            # Use python-arango API to query edges directly, then fetch target vertices
            try:
                edge_coll = self.db.collection(edge_type)
                # Query edges where _from matches our vertex
                edges = edge_coll.find({"_from": vertex_id})
                
                for edge in edges:
                    to_vertex_id = edge.get("_to")
                    if to_vertex_id:
                        # Extract collection and key from _to
                        to_collection, to_key = to_vertex_id.split("/", 1)
                        to_coll = self.db.collection(to_collection)
                        vertex = to_coll.get(to_key)
                        
                        if vertex and vertex.get("_id") not in visited:
                            visited.add(vertex.get("_id"))
                            neighbors.append(vertex)
            except Exception as e:
                # Print error for debugging but continue
                print(f"Debug: Error querying {edge_type} for {vertex_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # For bidirectional edges, also check incoming
            if edge_type in ["RELATED_TO"]:
                try:
                    edge_coll = self.db.collection(edge_type)
                    edges = edge_coll.find({"_to": vertex_id})
                    
                    for edge in edges:
                        from_vertex_id = edge.get("_from")
                        if from_vertex_id:
                            from_collection, from_key = from_vertex_id.split("/", 1)
                            from_coll = self.db.collection(from_collection)
                            vertex = from_coll.get(from_key)
                            
                            if vertex and vertex.get("_id") not in visited:
                                visited.add(vertex.get("_id"))
                                neighbors.append(vertex)
                except Exception as e:
                    print(f"Debug: Error querying {edge_type} (incoming) for {vertex_id}: {e}")
                    continue
        
        return neighbors
    
    def vector_semantic_search(self, query_text: str, collection: str,
                               limit: int = 10, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Semantic search using vector embeddings.
        
        Args:
            query_text: Text query to search for
            collection: Collection name to search in
            limit: Maximum number of results
            threshold: Minimum cosine similarity threshold
        
        Returns:
            List of matching documents with similarity scores
        """
        if not self.embedding_model:
            raise ValueError("Embedding model not loaded")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True)
        query_vector = query_embedding.tolist()
        
        # Search using AQL with cosine similarity
        # Note: This is a simplified approach. ArangoDB 3.10+ has native vector search
        # Use fallback method since AQL vector similarity is complex
        return self._fallback_vector_search(query_vector, collection, limit, threshold)
        
        try:
            cursor = self.db.aql.execute(
                query,
                bind_vars={
                    "query_vector": query_vector,
                    "threshold": threshold,
                    "limit": limit
                }
            )
            results = []
            for item in cursor:
                results.append({
                    "vertex": item["doc"],
                    "similarity": item["similarity"]
                })
            return results
        except Exception as e:
            # Fallback to simpler cosine similarity calculation
            return self._fallback_vector_search(query_vector, collection, limit, threshold)
    
    def _fallback_vector_search(self, query_vector, collection: str,
                               limit: int, threshold: float) -> List[Dict[str, Any]]:
        """Fallback vector search using Python cosine similarity."""
        if not self.db.has_collection(collection):
            print(f"Debug: Collection {collection} does not exist")
            return []
        
        coll = self.db.collection(collection)
        query_vector_np = np.array(query_vector)
        
        results = []
        total_docs = 0
        docs_with_embeddings = 0
        docs_without_embeddings = 0
        
        try:
            # Iterate through all documents in the collection
            for doc in coll:
                total_docs += 1
                
                # Check if document has embedding
                if "embedding" not in doc or not doc["embedding"]:
                    docs_without_embeddings += 1
                    continue
                
                docs_with_embeddings += 1
                doc_vector = np.array(doc["embedding"])
                
                # Check dimension match
                if len(doc_vector) != len(query_vector_np):
                    print(f"Debug: Dimension mismatch for {doc.get('name', 'unknown')}: doc={len(doc_vector)}, query={len(query_vector_np)}")
                    continue
                
                # Calculate cosine similarity
                dot_product = np.dot(query_vector_np, doc_vector)
                norm_query = np.linalg.norm(query_vector_np)
                norm_doc = np.linalg.norm(doc_vector)
                
                if norm_query == 0 or norm_doc == 0:
                    continue
                
                similarity = dot_product / (norm_query * norm_doc)
                
                # Add all results, we'll sort and filter by threshold later
                results.append({
                    "vertex": doc,
                    "similarity": float(similarity)
                })
        except Exception as e:
            print(f"Debug: Error in vector search for {collection}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
        # Debug output
        if total_docs > 0:
            print(f"Debug: Vector search in {collection}: {total_docs} total docs, {docs_with_embeddings} with embeddings, {docs_without_embeddings} without")
            if results:
                print(f"Debug: Found {len(results)} results before threshold filter (threshold={threshold})")
                print(f"Debug: Similarity range: {min(r['similarity'] for r in results):.3f} to {max(r['similarity'] for r in results):.3f}")
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Filter by threshold and limit
        filtered_results = [r for r in results if r["similarity"] >= threshold]
        
        if filtered_results:
            print(f"Debug: After threshold filter: {len(filtered_results)} results")
        else:
            print(f"Debug: No results above threshold {threshold}. Top 3 similarities: {[r['similarity'] for r in results[:3]]}")
        
        return filtered_results[:limit]
    
    def traverse_relationships(self, start_key: str, start_collection: str,
                              relationship_path: List[str], max_depth: int = 3) -> List[Dict[str, Any]]:
        """
        Traverse relationships following a specific path pattern.
        
        Args:
            start_key: Starting vertex _key
            start_collection: Starting vertex collection
            relationship_path: List of edge types to traverse (e.g., ["HAS_PROPERTY", "RELATED_TO"])
            max_depth: Maximum traversal depth
        
        Returns:
            List of vertices reached by traversal
        """
        if not relationship_path:
            return []
        
        start_id = f"{start_collection}/{start_key}"
        visited = set()
        results = []
        
        def traverse(current_id: str, path: List[str], depth: int):
            if depth > max_depth or not path:
                return
            
            if current_id in visited:
                return
            visited.add(current_id)
            
            edge_type = path[0]
            remaining_path = path[1:] if len(path) > 1 else []
            
            # Query outgoing edges
            # Check if collection exists
            if not self.db.has_collection(edge_type):
                return
            
            try:
                edge_coll = self.db.collection(edge_type)
                edges = edge_coll.find({"_from": current_id})
                
                for edge in edges:
                    to_vertex_id = edge.get("_to")
                    if to_vertex_id:
                        to_collection, to_key = to_vertex_id.split("/", 1)
                        to_coll = self.db.collection(to_collection)
                        vertex = to_coll.get(to_key)
                        
                        if vertex and vertex.get("_id") not in visited:
                            vertex_id = vertex.get("_id")
                            results.append(vertex)
                            visited.add(vertex_id)
                            
                            # Continue traversal if path continues
                            if remaining_path:
                                traverse(vertex_id, remaining_path, depth + 1)
                            elif depth < max_depth:
                                # If path exhausted but depth allows, continue with same edge type
                                traverse(vertex_id, [edge_type], depth + 1)
            except Exception as e:
                print(f"Error in traversal: {e}")
                import traceback
                traceback.print_exc()
        
        traverse(start_id, relationship_path, 0)
        return results
    
    def hybrid_search(self, query_text: str, collection: str,
                     relationship_filters: Optional[Dict[str, Any]] = None,
                     limit: int = 10) -> List[Dict[str, Any]]:
        """
        Hybrid search combining vector similarity with graph structure.
        
        Args:
            query_text: Text query for semantic search
            collection: Collection to search in
            relationship_filters: Dict with relationship constraints
                Example: {"must_have_edge": "HAS_PROPERTY", "edge_target": "Insertion_Loss"}
            limit: Maximum number of results
        
        Returns:
            List of matching vertices with scores
        """
        # First, get vector search results
        vector_results = self.vector_semantic_search(query_text, collection, limit=limit * 2, threshold=0.5)
        
        if not relationship_filters:
            return [r["vertex"] for r in vector_results[:limit]]
        
        # Apply relationship filters
        filtered_results = []
        for result in vector_results:
            vertex = result["vertex"]
            vertex_id = vertex["_id"]
            
            # Check relationship constraints
            matches = True
            
            if "must_have_edge" in relationship_filters:
                edge_type = relationship_filters["must_have_edge"]
                target = relationship_filters.get("edge_target")
                
                # Check if collection exists
                if not self.db.has_collection(edge_type):
                    matches = False
                else:
                    # Check if vertex has the required edge
                    try:
                        edge_coll = self.db.collection(edge_type)
                        edges = list(edge_coll.find({"_from": vertex_id}))
                    except Exception:
                        edges = []
                
                if not edges:
                    matches = False
                elif target:
                    # Check if edge connects to specific target
                    target_found = False
                    for edge in edges:
                        target_vertex = self.db.collection(edge["_to"].split("/")[0]).get(edge["_to"].split("/")[1])
                        if target_vertex and target_vertex.get("name") == target:
                            target_found = True
                            break
                    if not target_found:
                        matches = False
            
            if matches:
                filtered_results.append({
                    "vertex": vertex,
                    "similarity": result["similarity"]
                })
        
        # Sort by similarity and limit
        filtered_results.sort(key=lambda x: x["similarity"], reverse=True)
        return [r["vertex"] for r in filtered_results[:limit]]
    
    def find_components_by_principle(self, principle_name: str) -> List[Dict[str, Any]]:
        """Find all components/architectures that use a specific principle."""
        # First, find the principle by name (not just by key)
        principle_coll = self.db.collection("Physical_Principles")
        principle_doc = None
        
        # Try to find by name field
        query = """
        FOR doc IN `Physical_Principles`
            FILTER doc.name == @principle_name
            RETURN doc
        """
        try:
            cursor = self.db.aql.execute(query, bind_vars={"principle_name": principle_name})
            results = list(cursor)
            if results:
                principle_doc = results[0]
        except Exception as e:
            print(f"Debug: Error finding principle '{principle_name}': {e}")
            return []
        
        if not principle_doc:
            return []
        
        principle_id = principle_doc["_id"]
        
        # Find all vertices connected via BASED_ON_PRINCIPLE
        if not self.db.has_collection("BASED_ON_PRINCIPLE"):
            return []
        
        try:
            edge_coll = self.db.collection("BASED_ON_PRINCIPLE")
            edges = edge_coll.find({"_to": principle_id})
            
            results = []
            for edge in edges:
                from_vertex_id = edge.get("_from")
                if from_vertex_id:
                    from_collection, from_key = from_vertex_id.split("/", 1)
                    from_coll = self.db.collection(from_collection)
                    vertex = from_coll.get(from_key)
                    if vertex:
                        results.append(vertex)
            
            return results
        except Exception as e:
            print(f"Debug: Error finding components by principle: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def find_architectures_using_component(self, component_name: str) -> List[Dict[str, Any]]:
        """Find all architectures that use a specific component."""
        # First, find the component by name (not just by key)
        component_coll = self.db.collection("Components")
        component_doc = None
        
        # Try to find by name field
        query = """
        FOR doc IN `Components`
            FILTER doc.name == @component_name
            RETURN doc
        """
        try:
            cursor = self.db.aql.execute(query, bind_vars={"component_name": component_name})
            results = list(cursor)
            if results:
                component_doc = results[0]
        except Exception as e:
            print(f"Debug: Error finding component '{component_name}': {e}")
            return []
        
        if not component_doc:
            return []
        
        component_id = component_doc["_id"]
        
        # Find architectures via USES_COMPONENT edge
        if not self.db.has_collection("USES_COMPONENT"):
            return []
        
        try:
            edge_coll = self.db.collection("USES_COMPONENT")
            edges = edge_coll.find({"_to": component_id})
            
            results = []
            for edge in edges:
                from_vertex_id = edge.get("_from")
                if from_vertex_id:
                    from_collection, from_key = from_vertex_id.split("/", 1)
                    from_coll = self.db.collection(from_collection)
                    vertex = from_coll.get(from_key)
                    if vertex:
                        results.append(vertex)
            
            return results
        except Exception as e:
            print(f"Debug: Error finding architectures using component: {e}")
            import traceback
            traceback.print_exc()
            return []

