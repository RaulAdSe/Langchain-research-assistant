"""Typed state management for the multi-agent pipeline."""

from typing import List, TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class Citation(TypedDict):
    """A source citation with metadata."""
    marker: str  # e.g., "[#1]"
    title: str
    url: str
    date: Optional[str]
    snippet: Optional[str]  # Relevant excerpt


class Finding(TypedDict):
    """A research finding with supporting evidence."""
    claim: str
    evidence: str
    source: Citation
    confidence: float  # 0.0 to 1.0


class ToolCall(TypedDict):
    """Record of a tool invocation."""
    tool_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    timestamp: str
    duration_ms: Optional[int]


class CritiqueIssue(TypedDict):
    """An issue identified by the critic."""
    issue_type: str  # e.g., "missing_evidence", "outdated_source", "ambiguous_claim"
    description: str
    severity: str  # "critical", "major", "minor"
    suggested_fix: Optional[str]


class PipelineState(TypedDict, total=False):
    """
    Shared state that flows through the multi-agent pipeline.
    Using total=False allows partial updates at each step.
    """
    # Input
    question: str
    context: Optional[str]  # Additional context from user
    
    # Planning phase (Orchestrator)
    plan: str
    tool_sequence: List[str]
    key_terms: List[str]
    search_strategy: Optional[str]
    
    # Research phase (Researcher)
    findings: List[Finding]
    citations: List[Citation]
    tool_calls: List[ToolCall]
    draft: str
    gaps: List[str]  # Information gaps identified
    
    # Critique phase (Critic)
    critique: Dict[str, Any]
    issues: List[CritiqueIssue]
    required_fixes: List[str]
    quality_score: float  # 0.0 to 1.0
    
    # Synthesis phase (Synthesizer)
    final: str
    summary: str
    key_points: List[str]
    caveats: List[str]
    confidence: float  # Overall confidence 0.0 to 1.0
    
    # Metadata
    trace_id: Optional[str]
    start_time: Optional[str]
    end_time: Optional[str]
    total_tokens: Optional[int]
    error: Optional[str]


class ResearchRequest(BaseModel):
    """Input model for research requests."""
    question: str = Field(..., min_length=1, max_length=1000)
    context: Optional[str] = Field(None, max_length=2000)
    max_sources: int = Field(default=5, ge=1, le=20)
    require_recent: bool = Field(default=False)
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None


class ResearchResponse(BaseModel):
    """Output model for research responses."""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    caveats: Optional[List[str]] = None
    trace_url: Optional[str] = None
    duration_seconds: Optional[float] = None


class AgentOutput(BaseModel):
    """Base output format for individual agents."""
    agent_name: str
    success: bool
    output: Dict[str, Any]
    error: Optional[str] = None
    duration_ms: Optional[int] = None


# Helper functions for state management
def init_state(question: str, context: Optional[str] = None) -> PipelineState:
    """Initialize pipeline state with a question."""
    return PipelineState(
        question=question,
        context=context,
        findings=[],
        citations=[],
        tool_calls=[],
        issues=[],
        gaps=[],
        key_points=[],
        caveats=[],
        start_time=datetime.utcnow().isoformat(),
        confidence=0.0
    )


def update_state(state: PipelineState, **updates) -> PipelineState:
    """Update pipeline state with new values."""
    new_state = state.copy()
    new_state.update(updates)
    return new_state


def extract_citations(state: PipelineState) -> List[Citation]:
    """Extract unique citations from state."""
    seen_urls = set()
    unique_citations = []
    
    for citation in state.get("citations", []):
        if citation["url"] not in seen_urls:
            seen_urls.add(citation["url"])
            unique_citations.append(citation)
    
    return unique_citations