"""Tools for the research assistant."""

from app.tools.web_search import web_search_tool, WebSearchTool
from app.tools.retriever import retriever_tool, RetrieverTool
# Note: Firecrawl tool available but not loaded by default
# from app.tools.firecrawl import firecrawl_tool, FirecrawlTool

# Tool registry - DuckDuckGo web search + local knowledge base
AVAILABLE_TOOLS = {
    "web_search": web_search_tool,  # Free DuckDuckGo search
    "retriever": retriever_tool     # Local ChromaDB knowledge base
}

__all__ = [
    "web_search_tool",
    "retriever_tool",
    "WebSearchTool",
    "RetrieverTool",
    "AVAILABLE_TOOLS"
]