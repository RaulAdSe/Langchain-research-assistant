"""Unit tests for agent chains."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import json
from app.chains.orchestrator import OrchestratorChain
from app.chains.researcher import ResearcherChain
from app.chains.critic import CriticChain
from app.chains.synthesizer import SynthesizerChain


@pytest.mark.unit
class TestOrchestratorChain:
    """Test orchestrator chain functionality."""
    
    @patch('app.chains.orchestrator.chat_model')
    def test_orchestrator_generates_valid_plan(self, mock_chat_model, sample_state):
        """It should generate a valid research plan."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "plan": "Search for information about France",
            "tool_sequence": ["retriever", "web_search"],
            "key_terms": ["France", "capital", "Paris"],
            "search_strategy": "Start with knowledge base, then web",
            "validation_criteria": "Verify Paris is current capital"
        }))
        mock_chat_model.return_value = mock_llm
        
        orchestrator = OrchestratorChain()
        
        # Act
        result = orchestrator.plan(sample_state)
        
        # Assert
        assert result["plan"] == "Search for information about France"
        assert result["tool_sequence"] == ["retriever", "web_search"]
        assert "Paris" in result["key_terms"]
        assert "validation_criteria" in result
    
    @patch('app.chains.orchestrator.chat_model')
    def test_orchestrator_handles_errors_gracefully(self, mock_chat_model, sample_state):
        """It should handle LLM errors with default plan."""
        # Arrange
        mock_chat_model.side_effect = Exception("LLM error")
        orchestrator = OrchestratorChain()
        
        # Act
        result = orchestrator.plan(sample_state)
        
        # Assert
        assert "error" in result
        assert "Orchestrator error" in result["error"]
        assert result["plan"] == "Default plan: Search knowledge base and web for relevant information"
        assert result["tool_sequence"] == ["retriever", "web_search"]
    
    def test_orchestrator_uses_default_prompt_if_file_missing(self):
        """It should use default prompt when file is not found."""
        # Arrange
        with patch('pathlib.Path.exists', return_value=False):
            orchestrator = OrchestratorChain()
        
        # Assert
        assert "Orchestrator" in orchestrator.system_prompt
        assert "OUTPUT SCHEMA" in orchestrator.system_prompt


@pytest.mark.unit
class TestResearcherChain:
    """Test researcher chain functionality."""
    
    @patch('app.chains.researcher.AVAILABLE_TOOLS')
    def test_researcher_executes_tools_in_sequence(self, mock_tools, sample_state):
        """It should execute tools according to the plan."""
        # Arrange
        mock_retriever = MagicMock()
        mock_retriever._run.return_value = {
            "contexts": [
                {"content": "Paris is the capital", "source": "doc.pdf", "score": 0.9}
            ]
        }
        
        mock_web_search = MagicMock()
        mock_web_search._run.return_value = {
            "results": [
                {"title": "Paris Info", "url": "https://example.com", "snippet": "Paris facts"}
            ]
        }
        
        mock_tools.__getitem__.side_effect = {
            "retriever": mock_retriever,
            "web_search": mock_web_search
        }.get
        
        researcher = ResearcherChain()
        sample_state["tool_sequence"] = ["retriever", "web_search"]
        
        # Act
        result = researcher.research(sample_state)
        
        # Assert
        assert len(result["tool_calls"]) == 2
        assert result["tool_calls"][0]["tool_name"] == "retriever"
        assert result["tool_calls"][1]["tool_name"] == "web_search"
    
    def test_researcher_compiles_findings_from_tools(self):
        """It should compile findings from tool results."""
        # Arrange
        researcher = ResearcherChain()
        tool_results = [
            {
                "tool_name": "retriever",
                "output": {
                    "contexts": [
                        {
                            "content": "Paris is the capital of France",
                            "source": "doc.pdf",
                            "score": 0.95,
                            "filename": "doc.pdf"
                        }
                    ]
                }
            },
            {
                "tool_name": "web_search",
                "output": {
                    "results": [
                        {
                            "title": "Paris Facts",
                            "url": "https://example.com",
                            "snippet": "Interesting facts about Paris",
                            "published_at": "2024-01-15"
                        }
                    ]
                }
            }
        ]
        
        # Act
        compiled = researcher._compile_findings(tool_results)
        
        # Assert
        assert len(compiled["findings"]) >= 2
        assert len(compiled["citations"]) >= 2
        assert "[#1]" in compiled["citations"][0]["marker"]
        assert compiled["draft"] != ""
    
    @patch('app.chains.researcher.AVAILABLE_TOOLS')
    def test_researcher_handles_tool_errors(self, mock_tools, sample_state):
        """It should handle tool execution errors."""
        # Arrange
        mock_tool = MagicMock()
        mock_tool._run.side_effect = Exception("Tool error")
        mock_tools.__getitem__.return_value = mock_tool
        
        researcher = ResearcherChain()
        sample_state["tool_sequence"] = ["retriever"]
        
        # Act
        result = researcher.research(sample_state)
        
        # Assert
        assert result["tool_calls"][0]["output"]["error"] == "Tool error"


@pytest.mark.unit
class TestCriticChain:
    """Test critic chain functionality."""
    
    @patch('app.chains.critic.chat_model')
    def test_critic_identifies_issues(self, mock_chat_model, sample_state):
        """It should identify issues in research findings."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "issues": [
                {
                    "issue_type": "missing_evidence",
                    "description": "No recent sources",
                    "severity": "major",
                    "suggested_fix": "Add sources from 2024"
                }
            ],
            "required_fixes": ["Add recent sources"],
            "quality_score": 0.6,
            "strengths": ["Good coverage"],
            "missing_perspectives": ["Historical context"]
        }))
        mock_chat_model.return_value = mock_llm
        
        critic = CriticChain()
        sample_state["findings"] = [{"claim": "Test"}]
        sample_state["draft"] = "Test draft"
        
        # Act
        result = critic.critique(sample_state)
        
        # Assert
        assert len(result["issues"]) == 1
        assert result["issues"][0]["issue_type"] == "missing_evidence"
        assert result["quality_score"] == 0.6
        assert "Add recent sources" in result["required_fixes"]
    
    @patch('app.chains.critic.chat_model')
    def test_critic_approves_good_research(self, mock_chat_model, sample_state):
        """It should approve high-quality research."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "issues": [],
            "required_fixes": [],
            "quality_score": 0.9,
            "strengths": ["Comprehensive", "Well-cited"],
            "missing_perspectives": []
        }))
        mock_chat_model.return_value = mock_llm
        
        critic = CriticChain()
        
        # Act
        result = critic.critique(sample_state)
        
        # Assert
        assert result["quality_score"] == 0.9
        assert len(result["issues"]) == 0
        assert len(result["required_fixes"]) == 0


@pytest.mark.unit
class TestSynthesizerChain:
    """Test synthesizer chain functionality."""
    
    @patch('app.chains.synthesizer.chat_model')
    def test_synthesizer_creates_final_answer(self, mock_chat_model, sample_state):
        """It should create a well-formatted final answer."""
        # Arrange
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=json.dumps({
            "final": "Paris is the capital of France [#1]",
            "summary": "Paris has been France's capital since 987 AD",
            "key_points": ["Capital since 987 AD", "Population 2.2 million"],
            "caveats": ["Data from 2024"],
            "citations": [
                {"marker": "[#1]", "url": "https://example.com", "title": "France Facts", "date": "2024"}
            ],
            "confidence": 0.85,
            "metadata": {
                "sources_used": 3,
                "primary_sources": 2,
                "answer_completeness": "complete"
            }
        }))
        mock_chat_model.return_value = mock_llm
        
        synthesizer = SynthesizerChain()
        sample_state["findings"] = [{"claim": "Test"}]
        sample_state["critique"] = {"quality_score": 0.8}
        
        # Act
        result = synthesizer.synthesize(sample_state)
        
        # Assert
        assert "Paris" in result["final"]
        assert result["confidence"] == 0.85
        assert len(result["key_points"]) == 2
        assert len(result["caveats"]) == 1
        assert "**Summary**" in result["final"]
        assert "**Sources**" in result["final"]
    
    def test_synthesizer_formats_markdown_correctly(self):
        """It should format the final answer in proper markdown."""
        # Arrange
        synthesizer = SynthesizerChain()
        result_data = {
            "summary": "Test summary",
            "key_points": ["Point 1", "Point 2"],
            "caveats": ["Caveat 1"],
            "citations": [
                {"marker": "[#1]", "title": "Source", "url": "https://example.com", "date": "2024"}
            ]
        }
        
        # Act
        formatted = synthesizer._format_final_answer(result_data, {})
        
        # Assert
        assert "**Summary**" in formatted
        assert "Test summary" in formatted
        assert "**Key Points**" in formatted
        assert "- Point 1" in formatted
        assert "**Caveats and Limitations**" in formatted
        assert "**Sources**" in formatted
        assert "[#1] [Source](https://example.com) - 2024" in formatted