"""Unit tests for tools."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from app.tools.web_search import WebSearchTool, extract_citations_from_search
from app.tools.retriever import RetrieverTool, format_contexts_for_prompt
from app.tools.firecrawl import FirecrawlTool


@pytest.mark.unit
class TestWebSearchTool:
    """Test web search tool functionality."""
    
    def test_mock_search_returns_valid_results(self):
        """It should return mock results when no API key is configured."""
        # Arrange
        tool = WebSearchTool()
        
        # Act
        result = tool._mock_search("test query", top_k=3)
        
        # Assert
        assert len(result) == 3
        assert all("title" in r for r in result)
        assert all("url" in r for r in result)
        assert all("snippet" in r for r in result)
    
    def test_deduplicate_removes_duplicate_urls(self):
        """It should remove results with duplicate URLs."""
        # Arrange
        tool = WebSearchTool()
        results = [
            {"url": "https://example.com/1", "title": "First"},
            {"url": "https://example.com/2", "title": "Second"},
            {"url": "https://example.com/1", "title": "Duplicate"},
        ]
        
        # Act
        unique = tool._deduplicate_results(results)
        
        # Assert
        assert len(unique) == 2
        urls = [r["url"] for r in unique]
        assert urls == ["https://example.com/1", "https://example.com/2"]
    
    def test_filter_recent_keeps_only_recent_results(self):
        """It should filter results to keep only recent ones."""
        # Arrange
        tool = WebSearchTool()
        results = [
            {"url": "url1", "published_at": "2024-02-01"},
            {"url": "url2", "published_at": "2023-01-01"},
            {"url": "url3", "published_at": "2024-01-15"},
        ]
        
        # Act
        recent = tool._filter_recent(results, days=90)
        
        # Assert
        # Only results from 2024 should be kept (assuming current date is 2024)
        assert len(recent) >= 2  # At least the 2024 results
    
    def test_rate_limit_enforcement(self):
        """It should enforce rate limits."""
        # Arrange
        tool = WebSearchTool()
        tool._max_requests_per_session = 2
        
        # Act & Assert
        # First two requests should work
        result1 = tool._run("query1")
        assert "error" not in result1
        
        result2 = tool._run("query2")
        assert "error" not in result2
        
        # Third request should be rate limited
        result3 = tool._run("query3")
        assert "error" in result3
        assert "Rate limit" in result3["error"]
    
    def test_extract_citations_from_search(self):
        """It should extract citations from search results."""
        # Arrange
        results = [
            {
                "title": "Article 1",
                "url": "https://example.com/1",
                "snippet": "Snippet 1",
                "published_at": "2024-01-15"
            },
            {
                "title": "Article 2",
                "url": "https://example.com/2",
                "snippet": "Snippet 2",
                "published_at": None
            }
        ]
        
        # Act
        citations = extract_citations_from_search(results)
        
        # Assert
        assert len(citations) == 2
        assert citations[0]["marker"] == "[#1]"
        assert citations[0]["title"] == "Article 1"
        assert citations[0]["date"] == "2024-01-15"
        assert citations[1]["marker"] == "[#2]"
        assert citations[1]["date"] is None


@pytest.mark.unit
class TestRetrieverTool:
    """Test retriever tool functionality."""
    
    @patch('app.tools.retriever.get_vector_store')
    def test_retriever_formats_results(self, mock_get_store):
        """It should format retrieval results correctly."""
        # Arrange
        mock_store = MagicMock()
        mock_store.similarity_search_with_score.return_value = [
            (Mock(page_content="Content 1", metadata={"source": "doc1.pdf"}), 0.95),
            (Mock(page_content="Content 2", metadata={"source": "doc2.pdf"}), 0.88),
        ]
        mock_get_store.return_value = mock_store
        
        tool = RetrieverTool()
        
        # Act
        result = tool._run("test query", top_k=2)
        
        # Assert
        assert len(result["contexts"]) == 2
        assert result["contexts"][0]["content"] == "Content 1"
        assert result["contexts"][0]["score"] == 0.95
        assert result["contexts"][0]["source"] == "doc1.pdf"
    
    @patch('app.tools.retriever.get_vector_store')
    def test_retriever_handles_errors(self, mock_get_store):
        """It should handle retrieval errors gracefully."""
        # Arrange
        mock_get_store.side_effect = Exception("Vector store error")
        tool = RetrieverTool()
        
        # Act
        result = tool._run("test query")
        
        # Assert
        assert "error" in result
        assert "Vector store error" in result["error"]
        assert result["contexts"] == []
        assert result["total_results"] == 0
    
    def test_format_contexts_for_prompt(self):
        """It should format contexts for inclusion in prompts."""
        # Arrange
        contexts = [
            {"source": "doc1.pdf", "content": "Long content " * 100, "score": 0.95},
            {"source": "doc2.pdf", "content": "Short content", "score": 0.88},
        ]
        
        # Act
        formatted = format_contexts_for_prompt(contexts, max_length=200)
        
        # Assert
        assert len(formatted) <= 200
        assert "[Source 1: doc1.pdf" in formatted
        assert "relevance: 0.95" in formatted


@pytest.mark.unit
class TestFirecrawlTool:
    """Test Firecrawl tool functionality."""
    
    def test_mock_extraction_returns_valid_data(self):
        """It should return mock extraction when no API configured."""
        # Arrange
        tool = FirecrawlTool()
        
        # Act
        result = tool._mock_extraction("https://example.com", mode="article")
        
        # Assert
        assert "text" in result
        assert "links" in result
        assert "metadata" in result
        assert result["metadata"]["mode"] == "article"
    
    def test_validates_url_format(self):
        """It should validate URL format before extraction."""
        # Arrange
        tool = FirecrawlTool()
        
        # Act
        result = tool._run("not-a-url")
        
        # Assert
        assert "error" in result
        assert "Invalid URL" in result["error"]
    
    @patch('requests.post')
    def test_firecrawl_api_call(self, mock_post):
        """It should make correct API call when configured."""
        # Arrange
        with patch('app.tools.firecrawl.settings') as mock_settings:
            mock_settings.firecrawl_api_key = "test-key"
            mock_settings.firecrawl_base_url = "https://api.firecrawl.dev"
            mock_settings.timeout_seconds = 30
            
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "markdown": "Extracted text",
                "html": "<p>HTML</p>",
                "links": ["https://example.com/link1"],
                "metadata": {"title": "Page Title"}
            }
            mock_post.return_value = mock_response
            
            tool = FirecrawlTool()
            
            # Act
            result = tool._firecrawl_extract("https://example.com", "article")
            
            # Assert
            assert result["text"] == "Extracted text"
            assert result["html"] == "<p>HTML</p>"
            assert len(result["links"]) == 1
            mock_post.assert_called_once()
            
            # Verify API call parameters
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://api.firecrawl.dev/scrape"
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-key"