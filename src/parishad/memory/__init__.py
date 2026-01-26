"""
Parishad Memory System.
Provides vector storage and retrieval capabilities.
"""

from typing import Any, List, Dict, Optional

class VectorStore:
    """
    VectorStore implementation using ChromaDB.
    """
    
    def __init__(self, collection_name: str = "parishad_memory", persist_dir: str = "./.parishad_data/vector_store"):
        import chromadb
        import os
        
        # Ensure directory exists
        os.makedirs(persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add(self, documents: List[str], metadata: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None) -> None:
        """Add documents to the store."""
        if not documents:
            return
            
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in documents]
            
        # Ensure metadata is provided for all docs
        if metadata is None:
            metadata = [{} for _ in documents]
            
        self.collection.add(
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the store.
        Returns list of dicts with 'content', 'metadata', 'distance'.
        """
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            
            # Chroma returns lists of lists (one per query)
            if not results['documents'] or not results['documents'][0]:
                return []
                
            formatted_results = []
            
            # Zip the first (and only) query result lists
            doc_list = results['documents'][0]
            meta_list = results['metadatas'][0]
            
            # Handle distances if available (cosine distance usually)
            dist_list = results['distances'][0] if results['distances'] else [0.0] * len(doc_list)
            
            for doc, meta, dist in zip(doc_list, meta_list, dist_list):
                formatted_results.append({
                    'content': doc,
                    'metadata': meta,
                    'distance': dist
                })
                
            return formatted_results
            
        except Exception as e:
            # Fallback or empty on error
            print(f"Vector search error: {e}")
            return []
