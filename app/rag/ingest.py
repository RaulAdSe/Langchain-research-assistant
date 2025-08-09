"""Document ingestion pipeline for RAG."""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    DirectoryLoader
)
from app.core.config import settings
from app.rag.store import get_vector_store
import hashlib
import json


class DocumentIngester:
    """Handles document loading, chunking, and ingestion into vector store."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the document ingester.
        
        Args:
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.vector_store = get_vector_store()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document based on file extension.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of loaded documents
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        
        try:
            if extension == ".pdf":
                loader = PDFLoader(str(file_path))
            elif extension == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif extension == ".html" or extension == ".htm":
                loader = UnstructuredHTMLLoader(str(file_path))
            elif extension in [".txt", ".text"]:
                loader = TextLoader(str(file_path))
            else:
                # Try to load as text
                loader = TextLoader(str(file_path))
            
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    "source": str(file_path),
                    "filename": file_path.name,
                    "extension": extension
                })
            
            return documents
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def load_directory(
        self,
        directory_path: Path,
        glob_pattern: str = "**/*",
        recursive: bool = True
    ) -> List[Document]:
        """
        Load all documents from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file selection
            recursive: Whether to search recursively
            
        Returns:
            List of loaded documents
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        all_documents = []
        
        # Define supported extensions
        supported_extensions = [".pdf", ".md", ".html", ".htm", ".txt", ".text"]
        
        # Find all matching files
        if recursive:
            pattern = directory_path.glob(glob_pattern)
        else:
            pattern = directory_path.glob(f"*{glob_pattern}")
        
        for file_path in pattern:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                docs = self.load_document(file_path)
                all_documents.extend(docs)
        
        return all_documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            # Create a content hash for deduplication
            content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
            chunk.metadata["content_hash"] = content_hash
        
        return chunks
    
    def deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        Remove duplicate chunks based on content hash.
        
        Args:
            chunks: List of chunks
            
        Returns:
            List of unique chunks
        """
        seen_hashes = set()
        unique_chunks = []
        
        for chunk in chunks:
            content_hash = chunk.metadata.get("content_hash")
            if content_hash and content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def ingest_documents(
        self,
        documents: List[Document],
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest documents into the vector store.
        
        Args:
            documents: List of documents to ingest
            deduplicate: Whether to deduplicate chunks
            
        Returns:
            Ingestion statistics
        """
        if not documents:
            return {"status": "error", "message": "No documents to ingest"}
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        original_count = len(chunks)
        
        # Deduplicate if requested
        if deduplicate:
            chunks = self.deduplicate_chunks(chunks)
        
        # Add to vector store
        ids = self.vector_store.add_documents(chunks)
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "chunks_created": original_count,
            "chunks_ingested": len(chunks),
            "duplicates_removed": original_count - len(chunks),
            "document_ids": ids[:10]  # Return first 10 IDs as sample
        }
    
    def ingest_file(
        self,
        file_path: Path,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a single file.
        
        Args:
            file_path: Path to the file
            deduplicate: Whether to deduplicate chunks
            
        Returns:
            Ingestion statistics
        """
        documents = self.load_document(file_path)
        return self.ingest_documents(documents, deduplicate)
    
    def ingest_directory(
        self,
        directory_path: Path,
        glob_pattern: str = "**/*",
        recursive: bool = True,
        deduplicate: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest all documents from a directory.
        
        Args:
            directory_path: Path to the directory
            glob_pattern: Glob pattern for file selection
            recursive: Whether to search recursively
            deduplicate: Whether to deduplicate chunks
            
        Returns:
            Ingestion statistics
        """
        documents = self.load_directory(directory_path, glob_pattern, recursive)
        return self.ingest_documents(documents, deduplicate)


def ingest_sample_data():
    """Ingest sample documents for testing."""
    sample_dir = Path("data/sample_docs")
    
    # Create sample documents if they don't exist
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_files = {
        "langchain_overview.md": """# LangChain Overview

LangChain is a framework for developing applications powered by language models. It enables applications that are:
- Context-aware: connect a language model to sources of context
- Reasoning: rely on a language model to reason about how to answer

## Key Components

1. **LangChain Libraries**: Python and JavaScript libraries with interfaces and integrations
2. **LangChain Templates**: Architectures for common tasks
3. **LangSmith**: Developer platform for debugging, testing, and monitoring

## Use Cases

- Question answering over documents
- Chatbots and conversational AI
- Agents that can take actions
- Structured data extraction
""",
        "rag_best_practices.md": """# RAG Best Practices

Retrieval-Augmented Generation (RAG) combines retrieval and generation for better AI responses.

## Chunking Strategies

1. **Fixed-size chunks**: Simple but may break context
2. **Semantic chunks**: Preserve meaning but more complex
3. **Hybrid approach**: Balance between simplicity and quality

## Embedding Models

- OpenAI text-embedding-3-large: High quality, good for most use cases
- Sentence transformers: Open source alternatives
- Custom fine-tuned models: For domain-specific applications

## Retrieval Optimization

- Use metadata filtering to improve relevance
- Implement reranking for better precision
- Consider hybrid search (keyword + semantic)
"""
    }
    
    # Write sample files
    for filename, content in sample_files.items():
        file_path = sample_dir / filename
        if not file_path.exists():
            file_path.write_text(content)
    
    # Ingest the samples
    ingester = DocumentIngester()
    stats = ingester.ingest_directory(sample_dir)
    
    print("Sample data ingestion complete:")
    print(json.dumps(stats, indent=2))
    
    return stats


if __name__ == "__main__":
    # CLI interface for ingestion
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.rag.ingest <path> [--chunk <size>] [--overlap <size>]")
        print("Or: python -m app.rag.ingest --sample")
        sys.exit(1)
    
    if sys.argv[1] == "--sample":
        ingest_sample_data()
    else:
        path = Path(sys.argv[1])
        chunk_size = settings.chunk_size
        chunk_overlap = settings.chunk_overlap
        
        # Parse optional arguments
        for i in range(2, len(sys.argv)):
            if sys.argv[i] == "--chunk" and i + 1 < len(sys.argv):
                chunk_size = int(sys.argv[i + 1])
            elif sys.argv[i] == "--overlap" and i + 1 < len(sys.argv):
                chunk_overlap = int(sys.argv[i + 1])
        
        ingester = DocumentIngester(chunk_size, chunk_overlap)
        
        if path.is_file():
            stats = ingester.ingest_file(path)
        elif path.is_dir():
            stats = ingester.ingest_directory(path)
        else:
            print(f"Error: {path} is neither a file nor a directory")
            sys.exit(1)
        
        print(json.dumps(stats, indent=2))