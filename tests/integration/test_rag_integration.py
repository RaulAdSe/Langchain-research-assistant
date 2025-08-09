"""Integration tests for RAG components."""

import pytest
from pathlib import Path
import tempfile
import shutil
from langchain_core.documents import Document
from app.rag.ingest import DocumentIngester
from app.rag.store import VectorStoreManager, get_vector_store


@pytest.mark.integration
class TestRAGIntegration:
    """Test RAG ingestion and retrieval integration."""
    
    def test_ingest_and_retrieve_documents(self, temp_vector_store):
        """It should ingest documents and retrieve them successfully."""
        # Arrange
        store = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="test_collection"
        )
        
        documents = [
            Document(
                page_content="Paris is the capital of France.",
                metadata={"source": "doc1.txt"}
            ),
            Document(
                page_content="The Eiffel Tower is in Paris.",
                metadata={"source": "doc2.txt"}
            )
        ]
        
        # Act - Ingest
        doc_ids = store.add_documents(documents)
        
        # Assert - Documents added
        assert len(doc_ids) == 2
        
        # Act - Retrieve
        results = store.similarity_search("Paris France", k=2)
        
        # Assert - Documents retrieved
        assert len(results) == 2
        assert any("capital" in doc.page_content for doc in results)
        assert any("Eiffel" in doc.page_content for doc in results)
    
    def test_document_ingester_with_real_files(self, tmp_path):
        """It should ingest real files from disk."""
        # Arrange
        test_dir = tmp_path / "test_docs"
        test_dir.mkdir()
        
        # Create test files
        (test_dir / "doc1.txt").write_text("Machine learning is a subset of AI.")
        (test_dir / "doc2.md").write_text("# Neural Networks\n\nDeep learning uses neural networks.")
        
        ingester = DocumentIngester(chunk_size=100, chunk_overlap=20)
        
        # Act
        documents = ingester.load_directory(test_dir)
        chunks = ingester.chunk_documents(documents)
        
        # Assert
        assert len(documents) == 2
        assert len(chunks) >= 2  # May be more due to chunking
        assert any("Machine learning" in chunk.page_content for chunk in chunks)
        assert any("Neural Networks" in chunk.page_content for chunk in chunks)
    
    def test_vector_store_persistence(self, temp_vector_store):
        """It should persist and reload vector store."""
        # Arrange - First store instance
        store1 = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="persist_test"
        )
        
        doc = Document(
            page_content="Persistent content",
            metadata={"source": "test.txt"}
        )
        
        # Act - Add document with first instance
        store1.add_documents([doc])
        
        # Create new instance with same directory
        store2 = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="persist_test"
        )
        
        # Act - Search with second instance
        results = store2.similarity_search("persistent", k=1)
        
        # Assert - Document persisted
        assert len(results) == 1
        assert "Persistent content" in results[0].page_content
    
    def test_similarity_search_with_scores(self, temp_vector_store):
        """It should return similarity scores with results."""
        # Arrange
        store = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="score_test"
        )
        
        documents = [
            Document(page_content="Exact match query", metadata={"id": "1"}),
            Document(page_content="Partial match", metadata={"id": "2"}),
            Document(page_content="Unrelated content", metadata={"id": "3"})
        ]
        
        store.add_documents(documents)
        
        # Act
        results = store.similarity_search_with_score("Exact match query", k=3)
        
        # Assert
        assert len(results) == 3
        # First result should have best score (lowest distance)
        assert results[0][0].page_content == "Exact match query"
        assert results[0][1] < results[1][1]  # Better score than second
        assert results[1][1] < results[2][1]  # Better score than third
    
    def test_collection_stats(self, temp_vector_store):
        """It should return accurate collection statistics."""
        # Arrange
        store = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="stats_test"
        )
        
        documents = [
            Document(page_content=f"Document {i}", metadata={"id": str(i)})
            for i in range(5)
        ]
        
        # Act
        store.add_documents(documents)
        stats = store.get_collection_stats()
        
        # Assert
        assert stats["collection_name"] == "stats_test"
        assert stats["document_count"] == 5
        assert str(temp_vector_store) in stats["persist_directory"]
    
    def test_reset_collection(self, temp_vector_store):
        """It should reset collection completely."""
        # Arrange
        store = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="reset_test"
        )
        
        documents = [
            Document(page_content="To be deleted", metadata={"id": "1"})
        ]
        store.add_documents(documents)
        
        # Act
        store.reset()
        
        # Re-initialize after reset
        store = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="reset_test"
        )
        
        # Try to search
        results = store.similarity_search("deleted", k=10)
        
        # Assert - No documents found
        assert len(results) == 0
    
    def test_metadata_filtering(self, temp_vector_store):
        """It should filter results by metadata."""
        # Arrange
        store = VectorStoreManager(
            persist_directory=temp_vector_store,
            collection_name="filter_test"
        )
        
        documents = [
            Document(
                page_content="Python programming",
                metadata={"language": "python", "level": "beginner"}
            ),
            Document(
                page_content="Advanced Python",
                metadata={"language": "python", "level": "advanced"}
            ),
            Document(
                page_content="JavaScript basics",
                metadata={"language": "javascript", "level": "beginner"}
            )
        ]
        
        store.add_documents(documents)
        
        # Act - Filter by language
        python_results = store.similarity_search(
            "programming",
            k=10,
            filter={"language": "python"}
        )
        
        # Assert
        assert len(python_results) == 2
        assert all(
            doc.metadata.get("language") == "python"
            for doc in python_results
        )