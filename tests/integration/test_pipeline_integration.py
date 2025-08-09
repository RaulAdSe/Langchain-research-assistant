"""Integration tests for the complete pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from app.pipeline import ResearchPipeline
from app.core.state import ResearchRequest


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @patch('app.chains.orchestrator.chat_model')
    @patch('app.chains.synthesizer.chat_model')
    def test_complete_pipeline_flow(self, mock_synthesizer_llm, mock_orchestrator_llm):
        """It should run complete pipeline from question to answer."""
        # Arrange
        # Mock orchestrator response
        mock_orchestrator_llm.return_value.invoke.return_value = MagicMock(content='{"plan": "Test plan", "tool_sequence": ["retriever"], "key_terms": ["test"]}')
        
        # Mock synthesizer response
        mock_synthesizer_llm.return_value.invoke.return_value = MagicMock(content='{"final": "Test answer", "summary": "Test summary", "key_points": ["Point 1"], "caveats": [], "citations": [], "confidence": 0.8}')
        
        pipeline = ResearchPipeline()
        request = ResearchRequest(question="What is machine learning?")
        
        # Act
        response = pipeline.run(request)
        
        # Assert
        assert response.answer is not None
        assert response.confidence > 0
        assert response.duration_seconds is not None
    
    @patch('app.tools.AVAILABLE_TOOLS')
    def test_pipeline_with_mock_tools(self, mock_tools):
        """It should handle tool execution in pipeline."""
        # Arrange
        mock_retriever = MagicMock()
        mock_retriever._run.return_value = {
            "contexts": [
                {
                    "content": "Machine learning is AI subset",
                    "source": "ml_guide.pdf",
                    "score": 0.95
                }
            ],
            "total_results": 1
        }
        
        mock_tools.__getitem__.return_value = mock_retriever
        mock_tools.get.return_value = mock_retriever
        
        with patch('app.chains.orchestrator.chat_model') as mock_orch_llm:
            with patch('app.chains.synthesizer.chat_model') as mock_synth_llm:
                mock_orch_llm.return_value.invoke.return_value = MagicMock(
                    content='{"plan": "Search ML info", "tool_sequence": ["retriever"], "key_terms": ["machine learning"]}'
                )
                
                mock_synth_llm.return_value.invoke.return_value = MagicMock(
                    content='{"final": "ML is AI subset", "confidence": 0.9, "citations": []}'
                )
                
                pipeline = ResearchPipeline()
                request = ResearchRequest(question="What is machine learning?")
                
                # Act
                response = pipeline.run(request)
                
                # Assert
                assert "ML" in response.answer
                assert response.confidence == 0.9
    
    def test_pipeline_error_handling(self):
        """It should handle errors gracefully."""
        # Arrange
        with patch('app.chains.orchestrator.OrchestratorChain.plan') as mock_plan:
            mock_plan.side_effect = Exception("Planning failed")
            
            pipeline = ResearchPipeline()
            request = ResearchRequest(question="Error test")
            
            # Act
            response = pipeline.run(request)
            
            # Assert
            assert "error occurred" in response.answer.lower()
            assert response.confidence == 0.0
    
    @patch('app.chains.orchestrator.chat_model')
    @patch('app.chains.critic.chat_model')
    @patch('app.chains.synthesizer.chat_model')
    def test_pipeline_with_critic_iteration(self, mock_synth, mock_critic, mock_orch):
        """It should iterate based on critic feedback."""
        # Arrange
        mock_orch.return_value.invoke.return_value = MagicMock(
            content='{"plan": "Test plan", "tool_sequence": ["retriever"], "key_terms": ["test"]}'
        )
        
        # First critique - low quality, needs fixes
        mock_critic.return_value.invoke.side_effect = [
            MagicMock(content='{"issues": [{"issue_type": "missing", "severity": "major"}], "required_fixes": ["Add more sources"], "quality_score": 0.5}'),
            MagicMock(content='{"issues": [], "required_fixes": [], "quality_score": 0.8}')  # Second iteration passes
        ]
        
        mock_synth.return_value.invoke.return_value = MagicMock(
            content='{"final": "Improved answer", "confidence": 0.85, "citations": []}'
        )
        
        pipeline = ResearchPipeline(max_iterations=2)
        request = ResearchRequest(question="Test iteration")
        
        # Act
        response = pipeline.run(request)
        
        # Assert
        # Should have called critic twice (iteration)
        assert mock_critic.return_value.invoke.call_count == 2
        assert "Improved answer" in response.answer
    
    @patch('app.tools.AVAILABLE_TOOLS')
    def test_pipeline_tool_error_handling(self, mock_tools):
        """It should handle tool execution errors."""
        # Arrange
        mock_tool = MagicMock()
        mock_tool._run.side_effect = Exception("Tool error")
        mock_tools.__getitem__.return_value = mock_tool
        
        with patch('app.chains.orchestrator.chat_model') as mock_orch:
            with patch('app.chains.synthesizer.chat_model') as mock_synth:
                mock_orch.return_value.invoke.return_value = MagicMock(
                    content='{"plan": "Test", "tool_sequence": ["retriever"], "key_terms": ["test"]}'
                )
                
                mock_synth.return_value.invoke.return_value = MagicMock(
                    content='{"final": "Partial answer despite errors", "confidence": 0.4, "citations": []}'
                )
                
                pipeline = ResearchPipeline()
                request = ResearchRequest(question="Error test")
                
                # Act
                response = pipeline.run(request)
                
                # Assert
                assert response.answer is not None
                assert response.confidence <= 0.5  # Lower confidence due to errors
    
    def test_research_convenience_function(self):
        """It should provide convenient research function."""
        from app.pipeline import research
        
        with patch('app.pipeline.default_pipeline.run') as mock_run:
            mock_response = MagicMock()
            mock_response.answer = "Test answer"
            mock_run.return_value = mock_response
            
            # Act
            result = research("Test question", context="Test context")
            
            # Assert
            assert result.answer == "Test answer"
            mock_run.assert_called_once()
            
            # Check request was built correctly
            request_arg = mock_run.call_args[0][0]
            assert request_arg.question == "Test question"
            assert request_arg.context == "Test context"