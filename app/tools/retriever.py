"""Retriever tool for searching the knowledge base."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.documents import Document
from langsmith import traceable
from app.rag.store import get_vector_store
from app.core.state import Citation
import json


class RetrieverInput(BaseModel):
    """Input schema for the retriever tool."""
    query: str = Field(..., description="The search query")
    top_k: Optional[int] = Field(default=5, description="Number of results to return", ge=1, le=20)
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


class RetrieverOutput(BaseModel):
    """Output schema for the retriever tool."""
    contexts: List[Dict[str, Any]] = Field(..., description="Retrieved contexts")
    query: str = Field(..., description="The original query")
    total_results: int = Field(..., description="Total number of results")


class RetrieverTool(BaseTool):
    """Tool for retrieving documents from the knowledge base."""
    
    name: str = "retriever"
    description: str = "Search the knowledge base for relevant information"
    args_schema: type[BaseModel] = RetrieverInput
    return_direct: bool = False
    
    @traceable
    def _run(self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute retrieval from the knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            Dictionary with retrieved contexts
        """
        try:
            # Get vector store
            vector_store = get_vector_store()
            
            # Perform similarity search with scores
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                filter=filter
            )
            
            # Format results
            contexts = []
            for doc, score in results:
                context = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "score": float(score),
                    "metadata": doc.metadata
                }
                
                # Add specific metadata fields if available
                if "filename" in doc.metadata:
                    context["filename"] = doc.metadata["filename"]
                if "chunk_id" in doc.metadata:
                    context["chunk_id"] = doc.metadata["chunk_id"]
                
                contexts.append(context)
            
            return {
                "contexts": contexts,
                "query": query,
                "total_results": len(contexts)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "contexts": [],
                "query": query,
                "total_results": 0
            }
    
    async def _arun(self, query: str, top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async version of retriever (calls sync version)."""
        return self._run(query, top_k, filter)


def format_contexts_for_prompt(contexts: List[Dict[str, Any]], max_length: int = 3000) -> str:
    """
    Format retrieved contexts for inclusion in a prompt.
    
    Args:
        contexts: List of context dictionaries
        max_length: Maximum total length of formatted contexts
        
    Returns:
        Formatted string of contexts
    """
    formatted = []
    total_length = 0
    
    for i, ctx in enumerate(contexts, 1):
        source = ctx.get("source", "unknown")
        content = ctx.get("content", "")
        score = ctx.get("score", 0.0)
        
        # Format single context
        formatted_ctx = f"[Source {i}: {source} (relevance: {score:.2f})]\n{content}\n"
        
        # Check if adding this would exceed max length
        if total_length + len(formatted_ctx) > max_length and formatted:
            break
        
        formatted.append(formatted_ctx)
        total_length += len(formatted_ctx)
    
    return "\n".join(formatted)


def extract_citations_from_contexts(contexts: List[Dict[str, Any]]) -> List[Citation]:
    """
    Extract citations from retrieved contexts.
    
    Args:
        contexts: List of context dictionaries
        
    Returns:
        List of Citation objects
    """
    citations = []
    seen_sources = set()
    
    for i, ctx in enumerate(contexts, 1):
        source = ctx.get("source", "unknown")
        
        # Skip if we've already cited this source
        if source in seen_sources:
            continue
        
        seen_sources.add(source)
        
        citation = Citation(
            marker=f"[#{i}]",
            title=ctx.get("filename", source),
            url=source,  # In local KB, this is the file path
            date=None,  # Could extract from metadata if available
            snippet=ctx.get("content", "")[:200]  # First 200 chars as snippet
        )
        
        citations.append(citation)
    
    return citations


# Create singleton instance
retriever_tool = RetrieverTool()