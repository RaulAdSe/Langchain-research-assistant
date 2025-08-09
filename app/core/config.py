"""Configuration management for the multi-agent research assistant."""

import os
from typing import Literal, Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

Provider = Literal["anthropic", "openai", "other"]
SearchProvider = Literal["serpapi", "bing", "searx"]
EmbeddingsProvider = Literal["openai", "huggingface", "other"]


class Settings(BaseSettings):
    """Application settings from environment variables."""
    
    # Observability - LangSmith
    langsmith_tracing: bool = Field(default=True, env="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com", env="LANGSMITH_ENDPOINT")
    langsmith_api_key: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="multiagent-research", env="LANGSMITH_PROJECT")
    
    # Legacy LangChain env vars (for compatibility)
    langchain_tracing_v2: bool = Field(default=True, env="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="multiagent-research", env="LANGCHAIN_PROJECT")
    langchain_api_key: Optional[str] = Field(default=None, env="LANGCHAIN_API_KEY")
    
    # LLM Configuration
    provider: Provider = Field(default="anthropic", env="PROVIDER")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    model_name: str = Field(default="claude-3-5-sonnet-20241022", env="MODEL_NAME")
    
    # Embeddings
    embeddings_provider: EmbeddingsProvider = Field(default="openai", env="EMBEDDINGS_PROVIDER")
    embeddings_model: str = Field(default="text-embedding-3-large", env="EMBEDDINGS_MODEL")
    openai_api_key_embeddings: Optional[str] = Field(default=None, env="OPENAI_API_KEY_EMBEDDINGS")
    
    # Web Search
    search_api: SearchProvider = Field(default="serpapi", env="SEARCH_API")
    search_api_key: Optional[str] = Field(default=None, env="SEARCH_API_KEY")
    
    # Firecrawl (optional)
    firecrawl_base_url: Optional[str] = Field(default=None, env="FIRECRAWL_BASE_URL")
    firecrawl_api_key: Optional[str] = Field(default=None, env="FIRECRAWL_API_KEY")
    
    # ChromaDB
    chroma_persist_directory: Path = Field(default=Path("./chroma_db"), env="CHROMA_PERSIST_DIRECTORY")
    chroma_collection_name: str = Field(default="research_docs", env="CHROMA_COLLECTION_NAME")
    
    # Application settings
    max_retries: int = 3
    timeout_seconds: int = 30
    chunk_size: int = 800
    chunk_overlap: int = 120
    retriever_top_k: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def validate_provider_keys(self) -> None:
        """Validate that required API keys are present for the selected provider."""
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when PROVIDER=anthropic")
        elif self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when PROVIDER=openai")
        
        if self.embeddings_provider == "openai":
            if not (self.openai_api_key_embeddings or self.openai_api_key):
                raise ValueError("OPENAI_API_KEY or OPENAI_API_KEY_EMBEDDINGS required for OpenAI embeddings")
    
    def get_embeddings_api_key(self) -> Optional[str]:
        """Get the appropriate API key for embeddings."""
        if self.embeddings_provider == "openai":
            return self.openai_api_key_embeddings or self.openai_api_key
        return None


# Global settings instance
settings = Settings()

# Validate on import
try:
    settings.validate_provider_keys()
except ValueError as e:
    print(f"Warning: {e}")
    print("Please configure your .env file with the required API keys.")