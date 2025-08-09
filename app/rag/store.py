"""Vector store management for RAG retrieval."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from app.core.config import settings
from app.core.llm import get_embeddings_model


class VectorStoreManager:
    """Manages the vector store for document retrieval."""
    
    def __init__(self, persist_directory: Optional[Path] = None, collection_name: Optional[str] = None):
        """
        Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.collection_name = collection_name or settings.chroma_collection_name
        self._vectorstore = None
        self._client = None
        
        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def client(self) -> chromadb.Client:
        """Get or create ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._client
    
    @property
    def vectorstore(self) -> VectorStore:
        """Get or create the vector store."""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=get_embeddings_model(),
                persist_directory=str(self.persist_directory)
            )
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Add metadata if missing
        for doc in documents:
            if "source" not in doc.metadata:
                doc.metadata["source"] = "unknown"
        
        ids = self.vectorstore.add_documents(documents)
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of similar documents
        """
        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            **kwargs: Additional search parameters
            
        Returns:
            List of (document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._vectorstore = None
        except Exception as e:
            print(f"Error deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "error": str(e)
            }
    
    def reset(self) -> None:
        """Reset the vector store by deleting and recreating the collection."""
        self.delete_collection()
        self._vectorstore = None
        self._client = None


# Global instance
_store_manager = None

def get_vector_store() -> VectorStoreManager:
    """Get the global vector store manager instance."""
    global _store_manager
    if _store_manager is None:
        _store_manager = VectorStoreManager()
    return _store_manager


def create_retriever(top_k: int = 5, **kwargs):
    """
    Create a retriever from the vector store.
    
    Args:
        top_k: Number of documents to retrieve
        **kwargs: Additional retriever parameters
        
    Returns:
        A configured retriever
    """
    store = get_vector_store()
    return store.vectorstore.as_retriever(
        search_kwargs={"k": top_k},
        **kwargs
    )