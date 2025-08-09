"""Core modules for configuration, LLM, and state management."""

from app.core.config import settings
from app.core.llm import chat_model, get_chat_model, get_embeddings_model
from app.core.state import (
    PipelineState,
    Citation,
    Finding,
    ResearchRequest,
    ResearchResponse,
    init_state,
    update_state
)

__all__ = [
    "settings",
    "chat_model",
    "get_chat_model",
    "get_embeddings_model",
    "PipelineState",
    "Citation",
    "Finding",
    "ResearchRequest",
    "ResearchResponse",
    "init_state",
    "update_state"
]