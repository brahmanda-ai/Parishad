"""
Retrieval Tool (Memory Access).
Allows agents to query the Vector Store (RDMA).
"""

from typing import Any, Dict

from .base import BaseTool, ToolResult
from ..memory import VectorStore

class RetrievalTool(BaseTool):
    """
    Tool for searching the Agent's Long-Term Memory (Vector Store).
    """
    
    name = "memory_retrieval"
    description = "Search the council's long-term memory (codebase, docs) for relevant context."
    
    def __init__(self, collection_name: str = "parishad_memory"):
        super().__init__()
        # Initialize store
        # Note: We share the persist_dir convention
        self.store = VectorStore(collection_name=collection_name)

    @property
    def schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g. 'How does the ShellTool work?')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }

    def run(self, query: str, limit: int = 5) -> ToolResult:
        """Execute retrieval."""
        try:
            results = self.store.query(query, n_results=limit)
            
            if not results:
                return ToolResult(success=True, data="No relevant memories found.", metadata={"count": 0})
            
            # Format results into a readable string
            formatted = []
            for i, res in enumerate(results):
                content = res['content']
                meta = res['metadata']
                dist = res['distance']
                source = meta.get('source', 'unknown')
                formatted.append(f"[{i+1}] (Source: {source}, Dist: {dist:.3f})\n{content}\n")
                
            return ToolResult(
                success=True,
                data="\n".join(formatted),
                metadata={"count": len(results)}
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Retrieval failed: {str(e)}"
            )
