"""Web search tool for current information retrieval."""

from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langsmith import traceable
from datetime import datetime
import hashlib
import requests
import json
from bs4 import BeautifulSoup
import urllib.parse
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
    
    @traceable(name="WebSearch.serpapi_search")
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
    
    @traceable(name="WebSearch.duckduckgo_search")
    def _duckduckgo_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo HTML version (free, no API key needed)."""
        try:
            # DuckDuckGo HTML search URL
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            # Headers to look like a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make request
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            # Find all results (including ads which we'll filter)
            all_results = soup.find_all('div', class_=['result', 'result__body'])
            
            for result_div in all_results:
                try:
                    # Skip ads - they have y.js redirects
                    title_elem = result_div.find('a', class_='result__a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Skip if it's an ad (DuckDuckGo ads go through y.js)
                    if 'duckduckgo.com/y.js' in url:
                        continue
                    
                    # Extract snippet
                    snippet_elem = result_div.find('a', class_='result__snippet')
                    if not snippet_elem:
                        # Try alternative snippet location
                        snippet_elem = result_div.find('span', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # Clean up URL if needed
                    if url.startswith('//'):
                        url = 'https:' + url
                    elif url.startswith('/'):
                        # DuckDuckGo redirect URL - extract actual URL
                        if 'uddg=' in url:
                            import re
                            match = re.search(r'uddg=([^&]+)', url)
                            if match:
                                url = urllib.parse.unquote(match.group(1))
                    
                    # Skip if URL is still not valid
                    if not url.startswith('http'):
                        continue
                    
                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": snippet[:200] if snippet else "No description available",
                        "published_at": None,  # DuckDuckGo doesn't provide dates
                        "source": "DuckDuckGo"
                    })
                    
                    # Stop when we have enough non-ad results
                    if len(results) >= top_k:
                        break
                    
                except Exception as e:
                    continue
            
            # If we got results, return them
            if results:
                print(f"DuckDuckGo found {len(results)} results for '{query}'")
                return results
            
            # Fallback if parsing fails
            print(f"DuckDuckGo parsing failed, using fallback")
            return self._fallback_search(query, top_k)
            
        except Exception as e:
            print(f"DuckDuckGo search error: {e}")
            # Fallback to basic search
            return self._fallback_search(query, top_k)
    
    def _fallback_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback search when DuckDuckGo fails."""
        terms = query.replace(' ', '+')
        query_encoded = urllib.parse.quote(query)
        
        # Return search engine links as fallback
        return [
            {
                "title": f"{query} - DuckDuckGo Search",
                "url": f"https://duckduckgo.com/?q={query_encoded}",
                "snippet": f"Search results for {query} on DuckDuckGo. Click to see current results.",
                "published_at": None,
                "source": "DuckDuckGo"
            },
            {
                "title": f"{query} - Google Search", 
                "url": f"https://www.google.com/search?q={terms}",
                "snippet": f"Google search results for {query}. Contains the most comprehensive results.",
                "published_at": None,
                "source": "Google"
            },
            {
                "title": f"{query} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/Special:Search?search={query_encoded}",
                "snippet": f"Wikipedia articles related to {query}. Authoritative encyclopedia content.",
                "published_at": None,
                "source": "Wikipedia"
            }
        ][:top_k]

    def _mock_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Mock search results when no API key is configured.
        
        Note: These link to real search engines but are not actual search results.
        Configure SEARCH_API_KEY in .env for real web search with working links.
        """
        terms = query.replace(' ', '+')
        query_encoded = query.replace(' ', '_')
        
        mock_results = [
            {
                "title": f"{query} - Wikipedia",
                "url": f"https://en.wikipedia.org/wiki/{query_encoded}",
                "snippet": f"Wikipedia article covering {query} with comprehensive background, definitions, and references to authoritative sources.",
                "published_at": "2024-01-15"
            },
            {
                "title": f"{query} - Google Scholar Search Results", 
                "url": f"https://scholar.google.com/scholar?q={terms}",
                "snippet": f"Academic research papers and scholarly articles related to {query}. Contains peer-reviewed research and technical discussions.",
                "published_at": "2024-02-20"
            },
            {
                "title": f"{query} - ArXiv Research Papers",
                "url": f"https://arxiv.org/search/?query={terms}",
                "snippet": f"Recent research papers and preprints about {query}. Cutting-edge research findings from the academic community.",
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
    
    @traceable(name="WebSearch._run")
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
            if settings.search_api == "serpapi" and settings.search_api_key:
                results = self._serpapi_search(query, top_k * 2)  # Get extra for filtering
            else:
                # Use DuckDuckGo for free web search (no API key needed!)
                results = self._duckduckgo_search(query, top_k * 2)
            
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