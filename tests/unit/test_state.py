"""Unit tests for state management."""

import pytest
from app.core.state import (
    init_state,
    update_state,
    extract_citations,
    PipelineState,
    ResearchRequest,
    ResearchResponse,
    Citation,
    Finding
)


@pytest.mark.unit
class TestStateManagement:
    """Test state management functions."""
    
    def test_init_state_creates_valid_state(self):
        """It should initialize state with question and default values."""
        # Arrange
        question = "What is machine learning?"
        context = "Focus on supervised learning"
        
        # Act
        state = init_state(question, context)
        
        # Assert
        assert state["question"] == question
        assert state["context"] == context
        assert state["findings"] == []
        assert state["citations"] == []
        assert state["confidence"] == 0.0
        assert "start_time" in state
    
    def test_update_state_preserves_existing_values(self):
        """It should update only specified fields while preserving others."""
        # Arrange
        initial_state = {
            "question": "Original question",
            "confidence": 0.5,
            "findings": ["finding1"]
        }
        
        # Act
        updated = update_state(initial_state, confidence=0.8, draft="New draft")
        
        # Assert
        assert updated["question"] == "Original question"
        assert updated["confidence"] == 0.8
        assert updated["draft"] == "New draft"
        assert updated["findings"] == ["finding1"]
    
    def test_extract_citations_removes_duplicates(self):
        """It should extract unique citations based on URL."""
        # Arrange
        state = {
            "citations": [
                {"url": "https://example.com/1", "title": "Title 1"},
                {"url": "https://example.com/2", "title": "Title 2"},
                {"url": "https://example.com/1", "title": "Duplicate"},
            ]
        }
        
        # Act
        unique_citations = extract_citations(state)
        
        # Assert
        assert len(unique_citations) == 2
        urls = [c["url"] for c in unique_citations]
        assert "https://example.com/1" in urls
        assert "https://example.com/2" in urls


@pytest.mark.unit
class TestResearchModels:
    """Test Pydantic models for requests and responses."""
    
    def test_research_request_validates_question_length(self):
        """It should validate question length constraints."""
        # Valid question
        request = ResearchRequest(question="Valid question")
        assert request.question == "Valid question"
        
        # Too short
        with pytest.raises(ValueError):
            ResearchRequest(question="")
        
        # Too long
        with pytest.raises(ValueError):
            ResearchRequest(question="x" * 1001)
    
    def test_research_request_defaults(self):
        """It should apply default values correctly."""
        # Arrange & Act
        request = ResearchRequest(question="Test question")
        
        # Assert
        assert request.max_sources == 5
        assert request.require_recent == False
        assert request.allowed_domains is None
        assert request.blocked_domains is None
    
    def test_research_response_confidence_validation(self):
        """It should validate confidence is between 0 and 1."""
        # Valid confidence
        response = ResearchResponse(
            answer="Answer",
            citations=[],
            confidence=0.75
        )
        assert response.confidence == 0.75
        
        # Invalid confidence
        with pytest.raises(ValueError):
            ResearchResponse(
                answer="Answer",
                citations=[],
                confidence=1.5
            )


@pytest.mark.unit
class TestTypedDictStructures:
    """Test TypedDict structures."""
    
    def test_citation_structure(self):
        """It should create valid Citation structure."""
        citation: Citation = {
            "marker": "[#1]",
            "title": "Test Article",
            "url": "https://example.com",
            "date": "2024-01-15",
            "snippet": "Test snippet"
        }
        
        assert citation["marker"] == "[#1]"
        assert citation["title"] == "Test Article"
        assert citation["url"] == "https://example.com"
    
    def test_finding_structure(self):
        """It should create valid Finding structure."""
        finding: Finding = {
            "claim": "Test claim",
            "evidence": "Supporting evidence",
            "source": {
                "marker": "[#1]",
                "title": "Source",
                "url": "https://example.com",
                "date": None,
                "snippet": None
            },
            "confidence": 0.85
        }
        
        assert finding["claim"] == "Test claim"
        assert finding["confidence"] == 0.85
        assert finding["source"]["url"] == "https://example.com"
    
    def test_pipeline_state_partial_update(self):
        """It should allow partial updates with total=False."""
        state: PipelineState = {
            "question": "Test question"
        }
        
        # Should not require all fields
        state["confidence"] = 0.9
        state["findings"] = []
        
        assert state["question"] == "Test question"
        assert state["confidence"] == 0.9