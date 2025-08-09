"""Shared fixtures and configuration for tests."""

import pytest
import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
from unittest.mock import Mock, MagicMock
from datetime import datetime
import shutil

# Set test environment
os.environ["TESTING"] = "true"
os.environ["LANGSMITH_TRACING"] = "false"  # Disable tracing during tests


@pytest.fixture
def mock_llm():
    """Mock LLM for testing without API calls."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content='{"test": "response"}')
    return mock


@pytest.fixture
def sample_state():
    """Sample pipeline state for testing."""
    return {
        "question": "What is the capital of France?",
        "context": None,
        "plan": "Search for information about France's capital",
        "tool_sequence": ["retriever", "web_search"],
        "key_terms": ["France", "capital", "Paris"],
        "findings": [],
        "citations": [],
        "draft": "",
        "confidence": 0.0
    }


@pytest.fixture
def sample_findings():
    """Sample research findings."""
    return [
        {
            "claim": "Paris is the capital of France",
            "evidence": "Paris has been the capital of France since 987 AD",
            "source": {
                "title": "France Overview",
                "url": "https://example.com/france",
                "date": "2024-01-15",
                "snippet": "Paris, the capital city..."
            },
            "confidence": 0.9
        }
    ]


@pytest.fixture
def sample_citations():
    """Sample citations."""
    return [
        {
            "marker": "[#1]",
            "title": "France Overview",
            "url": "https://example.com/france",
            "date": "2024-01-15"
        }
    ]


@pytest.fixture
def temp_vector_store(tmp_path):
    """Temporary vector store for testing."""
    store_path = tmp_path / "test_chroma"
    store_path.mkdir(exist_ok=True)
    
    # Update config to use temp directory
    original_dir = os.environ.get("CHROMA_PERSIST_DIRECTORY")
    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(store_path)
    
    yield store_path
    
    # Cleanup
    if original_dir:
        os.environ["CHROMA_PERSIST_DIRECTORY"] = original_dir
    else:
        os.environ.pop("CHROMA_PERSIST_DIRECTORY", None)
    
    # Remove temp directory
    if store_path.exists():
        shutil.rmtree(store_path)


@pytest.fixture
def sample_documents():
    """Sample documents for ingestion testing."""
    return [
        {
            "content": "Paris is the capital and largest city of France.",
            "metadata": {"source": "test_doc_1.txt", "page": 1}
        },
        {
            "content": "The Eiffel Tower is located in Paris.",
            "metadata": {"source": "test_doc_2.txt", "page": 1}
        }
    ]


@pytest.fixture
def mock_web_search_results():
    """Mock web search results."""
    return {
        "results": [
            {
                "title": "Paris - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Paris",
                "snippet": "Paris is the capital and most populous city of France",
                "published_at": "2024-02-20"
            },
            {
                "title": "Facts about Paris",
                "url": "https://example.com/paris-facts",
                "snippet": "Interesting facts about Paris, the City of Light",
                "published_at": "2024-01-30"
            }
        ],
        "query": "Paris capital France",
        "total_results": 2
    }


@pytest.fixture
def mock_retriever_results():
    """Mock retriever results."""
    return {
        "contexts": [
            {
                "content": "Paris has been the capital of France since 987 AD",
                "source": "history_doc.pdf",
                "score": 0.92,
                "metadata": {"chunk_id": 1, "filename": "history_doc.pdf"}
            },
            {
                "content": "The population of Paris is approximately 2.2 million",
                "source": "demographics.pdf",
                "score": 0.85,
                "metadata": {"chunk_id": 5, "filename": "demographics.pdf"}
            }
        ],
        "query": "Paris capital France",
        "total_results": 2
    }


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    mock = MagicMock()
    mock.provider = "openai"
    mock.model_name = "gpt-4o-mini"
    mock.openai_api_key = "test-key"
    mock.chunk_size = 500
    mock.chunk_overlap = 50
    mock.chroma_persist_directory = Path("./test_chroma")
    mock.chroma_collection_name = "test_collection"
    mock.timeout_seconds = 30
    mock.search_api = "serpapi"
    mock.search_api_key = None  # Will use mock search
    return mock


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset any global state
    from app.rag import store
    if hasattr(store, '_store_manager'):
        store._store_manager = None
    
    yield
    
    # Cleanup after test
    from app.rag import store
    if hasattr(store, '_store_manager'):
        store._store_manager = None


@pytest.fixture
def mock_time():
    """Mock datetime for deterministic tests."""
    fixed_time = datetime(2024, 1, 15, 12, 0, 0)
    mock = MagicMock()
    mock.utcnow.return_value = fixed_time
    mock.now.return_value = fixed_time
    return mock


# Test markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow"
    )