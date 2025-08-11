"""Provider-agnostic LLM adapter for seamless model switching."""

from typing import Optional, Dict, Any
from langchain_core.language_models import BaseChatModel
from app.core.config import settings, Provider


def get_chat_model(
    provider: Optional[Provider] = None,
    model: Optional[str] = None,
    agent_type: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs
) -> BaseChatModel:
    """
    Get a LangChain-compatible chat model instance.
    
    Args:
        provider: The LLM provider to use (defaults to settings.provider)
        model: The model name (defaults to settings.model_name)
        agent_type: Agent type for model selection (orchestrator, researcher, critic, synthesizer)
        temperature: Model temperature for response generation
        **kwargs: Additional provider-specific parameters
    
    Returns:
        A configured BaseChatModel instance
    
    Raises:
        ValueError: If the provider is not supported or API keys are missing
    """
    provider = provider or settings.provider
    
    # Select agent-specific model if configured
    if agent_type and not model:
        agent_models = {
            "orchestrator": settings.orchestrator_model,
            "researcher": settings.researcher_model,
            "critic": settings.critic_model,
            "synthesizer": settings.synthesizer_model
        }
        model = agent_models.get(agent_type) or settings.model_name
    else:
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
        # Handle gpt-5-nano parameter differences
        if model == "gpt-5-nano":
            # Remove max_tokens from kwargs and use max_completion_tokens instead
            kwargs.pop("max_tokens", None)
            return ChatOpenAI(
                model=model,
                api_key=settings.openai_api_key,
                temperature=temperature,
                max_completion_tokens=kwargs.get("max_completion_tokens", 4096),
                **kwargs
            )
        else:
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
def chat_model(agent_type: Optional[str] = None, **kwargs) -> BaseChatModel:
    """Convenience function to get the default chat model with low temperature for consistency."""
    # Set low temperature by default to reduce stochasticity
    kwargs.setdefault("temperature", 0.1)
    return get_chat_model(agent_type=agent_type, **kwargs)