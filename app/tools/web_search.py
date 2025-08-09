"""Web search tool for current information retrieval."""

from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from datetime import datetime
import hashlib
import requests
import json
from app.core.config import settings


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(..., description="The search query")
    top_k: Optional[int] = Field(default=5, description="Number of results to return", ge=1, le=10)
    recent_only: Optional[bool] = Field(default=False, description="Only return recent results")


class WebSearchOutput(BaseModel):
    """Output schema for web search tool."""
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    query: str = Field(..., description="The original query")
    total_results: int = Field(..., description="Total number of results")


class WebSearchTool(BaseTool):
    """Tool for searching the web for current information."""
    
    name: str = "web_search"
    description: str = "Search the web for current information"
    args_schema: type[BaseModel] = WebSearchInput
    return_direct: bool = False
    
    def __init__(self):
        super().__init__()
        self._seen_urls: Set[str] = set()
        self._rate_limit_count = 0
        self._max_requests_per_session = 50
    
    def _serpapi_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using SerpAPI."""
        if not settings.search_api_key:
            return self._mock_search(query, top_k)
        
        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": settings.search_api_key,
            "num": top_k,
            "engine": "google"
        }
        
        try:
            response = requests.get(url, params=params, timeout=settings.timeout_seconds)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("organic_results", [])[:top_k]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "published_at": item.get("date", None)
                })
            
            return results
            
        except Exception as e:
            print(f"SerpAPI error: {e}")
            return self._mock_search(query, top_k)
    
    def _mock_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock search results for testing without API key."""
        mock_results = [
            {
                "title": f"Understanding {query}: A Comprehensive Guide",
                "url": f"https://example.com/guide-{hashlib.md5(query.encode()).hexdigest()[:8]}",
                "snippet": f"This comprehensive guide covers everything you need to know about {query}, including recent developments and best practices...",
                "published_at": "2024-01-15"
            },
            {
                "title": f"Latest Updates on {query}",
                "url": f"https://news.example.com/updates-{hashlib.md5(query.encode()).hexdigest()[:8]}",
                "snippet": f"Recent developments in {query} have shown significant progress. Experts say that the implications are far-reaching...",
                "published_at": "2024-02-20"
            },
            {
                "title": f"{query}: Research and Analysis",
                "url": f"https://research.example.com/analysis-{hashlib.md5(query.encode()).hexdigest()[:8]}",
                "snippet": f"Our research team has conducted an in-depth analysis of {query}. The findings suggest several key trends...",
                "published_at": "2024-01-30"
            }
        ]
        
        return mock_results[:top_k]
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL."""
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            if url and url not in self._seen_urls:
                self._seen_urls.add(url)
                unique_results.append(result)
        
        return unique_results
    
    def _filter_recent(self, results: List[Dict[str, Any]], days: int = 90) -> List[Dict[str, Any]]:
        """Filter results to only include recent ones."""
        recent_results = []
        cutoff_date = datetime.now()
        
        for result in results:
            published = result.get("published_at")
            if published:
                try:
                    # Try to parse the date
                    pub_date = datetime.strptime(published, "%Y-%m-%d")
                    if (cutoff_date - pub_date).days <= days:
                        recent_results.append(result)
                except:
                    # If can't parse, include it anyway
                    recent_results.append(result)
            else:
                # No date, include it
                recent_results.append(result)
        
        return recent_results
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        recent_only: bool = False
    ) -> Dict[str, Any]:
        """
        Execute web search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            recent_only: Only return recent results
            
        Returns:
            Dictionary with search results
        """
        # Check rate limit
        self._rate_limit_count += 1
        if self._rate_limit_count > self._max_requests_per_session:
            return {
                "error": "Rate limit exceeded for this session",
                "results": [],
                "query": query,
                "total_results": 0
            }
        
        try:
            # Perform search based on configured provider
            if settings.search_api == "serpapi":
                results = self._serpapi_search(query, top_k * 2)  # Get extra for filtering
            else:
                results = self._mock_search(query, top_k * 2)
            
            # Deduplicate
            results = self._deduplicate_results(results)
            
            # Filter for recent if requested
            if recent_only:
                results = self._filter_recent(results)
            
            # Limit to top_k
            results = results[:top_k]
            
            return {
                "results": results,
                "query": query,
                "total_results": len(results)
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "results": [],
                "query": query,
                "total_results": 0
            }
    
    async def _arun(
        self,
        query: str,
        top_k: int = 5,
        recent_only: bool = False
    ) -> Dict[str, Any]:
        """Async version of web search (calls sync version)."""
        return self._run(query, top_k, recent_only)


def extract_citations_from_search(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract citations from search results.
    
    Args:
        results: List of search result dictionaries
        
    Returns:
        List of citation dictionaries
    """
    citations = []
    
    for i, result in enumerate(results, 1):
        citation = {
            "marker": f"[#{i}]",
            "title": result.get("title", "Untitled"),
            "url": result.get("url", ""),
            "date": result.get("published_at"),
            "snippet": result.get("snippet", "")
        }
        citations.append(citation)
    
    return citations


# Create singleton instance
web_search_tool = WebSearchTool()