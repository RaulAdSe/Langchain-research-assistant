"""Provider-agnostic LLM adapter for seamless model switching."""

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from app.core.config import settings, Provider


def get_chat_model(
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> BaseChatModel:
    """
    Get a LangChain-compatible chat model instance.
    
    Args:
        provider: The LLM provider to use (defaults to settings.provider)
        model: The model name (defaults to settings.model_name)
        temperature: Model temperature for response generation
        **kwargs: Additional provider-specific parameters
    
    Returns:
        A configured BaseChatModel instance
    
    Raises:
        ValueError: If the provider is not supported or API keys are missing
    """
    provider = provider or settings.provider
    model = model or settings.model_name
    
    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
        
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs
        )
    
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            temperature=temperature,
            max_tokens=kwargs.get("max_tokens", 4096),
            **kwargs
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def get_embeddings_model():
    """
    Get an embeddings model based on configuration.
    
    Returns:
        An embeddings model instance
    
    Raises:
        ValueError: If the embeddings provider is not supported
    """
    if settings.embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = settings.get_embeddings_api_key()
        if not api_key:
            raise ValueError("OpenAI API key required for OpenAI embeddings")
        
        return OpenAIEmbeddings(
            model=settings.embeddings_model,
            api_key=api_key
        )
    
    elif settings.embeddings_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.embeddings_model
        )
    
    else:
        raise ValueError(f"Unsupported embeddings provider: {settings.embeddings_provider}")


# Convenience function for quick access
def chat_model(**kwargs) -> BaseChatModel:
    """Convenience function to get the default chat model."""
    return get_chat_model(**kwargs)