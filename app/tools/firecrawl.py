"""Firecrawl tool for robust web content extraction."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
import requests
from app.core.config import settings


class FirecrawlInput(BaseModel):
    """Input schema for Firecrawl tool."""
    url: str = Field(..., description="The URL to extract content from")
    mode: Optional[str] = Field(default="article", description="Extraction mode: 'article' or 'full'")


class FirecrawlOutput(BaseModel):
    """Output schema for Firecrawl tool."""
    text: str = Field(..., description="Extracted text content")
    html: Optional[str] = Field(None, description="HTML content if available")
    links: List[str] = Field(default_factory=list, description="Links found in the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FirecrawlTool(BaseTool):
    """Tool for extracting content from web pages using Firecrawl."""
    
    name: str = "firecrawl"
    description: str = "Extract clean content from web pages"
    args_schema: type[BaseModel] = FirecrawlInput
    return_direct: bool = False
    
    def _mock_extraction(self, url: str, mode: str = "article") -> Dict[str, Any]:
        """Mock extraction for testing without Firecrawl API."""
        return {
            "text": f"Mock extracted content from {url}\n\nThis is a simulated extraction. In production, this would contain the actual content from the webpage.",
            "html": None,
            "links": ["https://example.com/link1", "https://example.com/link2"],
            "metadata": {
                "title": f"Page from {url}",
                "description": "Mock page description",
                "mode": mode
            }
        }
    
    def _firecrawl_extract(self, url: str, mode: str = "article") -> Dict[str, Any]:
        """Extract content using Firecrawl API."""
        if not settings.firecrawl_api_key or not settings.firecrawl_base_url:
            return self._mock_extraction(url, mode)
        
        endpoint = f"{settings.firecrawl_base_url}/scrape"
        headers = {
            "Authorization": f"Bearer {settings.firecrawl_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "url": url,
            "mode": mode,
            "formats": ["markdown", "html", "links"]
        }
        
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=settings.timeout_seconds
            )
            response.raise_for_status()
            data = response.json()
            
            return {
                "text": data.get("markdown", ""),
                "html": data.get("html"),
                "links": data.get("links", []),
                "metadata": data.get("metadata", {})
            }
            
        except Exception as e:
            print(f"Firecrawl API error: {e}")
            return self._mock_extraction(url, mode)
    
    def _run(self, url: str, mode: str = "article") -> Dict[str, Any]:
        """
        Extract content from a web page.
        
        Args:
            url: The URL to extract from
            mode: Extraction mode ('article' or 'full')
            
        Returns:
            Dictionary with extracted content
        """
        try:
            # Validate URL
            if not url.startswith(("http://", "https://")):
                return {
                    "error": "Invalid URL format",
                    "text": "",
                    "html": None,
                    "links": [],
                    "metadata": {}
                }
            
            # Check if Firecrawl is configured
            if settings.firecrawl_api_key and settings.firecrawl_base_url:
                result = self._firecrawl_extract(url, mode)
            else:
                result = self._mock_extraction(url, mode)
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "text": "",
                "html": None,
                "links": [],
                "metadata": {}
            }
    
    async def _arun(self, url: str, mode: str = "article") -> Dict[str, Any]:
        """Async version of Firecrawl (calls sync version)."""
        return self._run(url, mode)


# Create singleton instance
firecrawl_tool = FirecrawlTool()