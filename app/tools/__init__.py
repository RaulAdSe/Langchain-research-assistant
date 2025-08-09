"""Tools for the research assistant."""

from app.tools.web_search import web_search_tool, WebSearchTool
from app.tools.retriever import retriever_tool, RetrieverTool
from app.tools.firecrawl import firecrawl_tool, FirecrawlTool

# Tool registry
AVAILABLE_TOOLS = {
    "web_search": web_search_tool,
    "retriever": retriever_tool,
    "firecrawl": firecrawl_tool
}

__all__ = [
    "web_search_tool",
    "retriever_tool",
    "firecrawl_tool",
    "WebSearchTool",
    "RetrieverTool",
    "FirecrawlTool",
    "AVAILABLE_TOOLS"
]