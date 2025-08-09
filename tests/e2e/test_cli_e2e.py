"""End-to-end tests for CLI functionality."""

import pytest
import subprocess
import json
from pathlib import Path
from unittest.mock import patch


@pytest.mark.e2e
@pytest.mark.slow
class TestCLIEndToEnd:
    """Test CLI commands end-to-end."""
    
    def test_cli_help_command(self):
        """It should display help information."""
        # Act
        result = subprocess.run(
            ["python", "-m", "app.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Assert
        assert result.returncode == 0
        assert "Multi-agent research assistant CLI" in result.stdout
        assert "ask" in result.stdout
        assert "ingest" in result.stdout
    
    def test_cli_stats_command(self):
        """It should show knowledge base statistics."""
        # Act
        result = subprocess.run(
            ["python", "-m", "app.cli", "stats"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Assert
        # Command should complete without error (even if collection is empty)
        assert result.returncode == 0
        assert "Knowledge Base Statistics" in result.stdout
    
    @patch('app.pipeline.default_pipeline.run')
    def test_cli_ask_command_with_mock(self, mock_pipeline_run):
        """It should execute ask command with mocked pipeline."""
        # Arrange
        from app.core.state import ResearchResponse
        
        mock_response = ResearchResponse(
            answer="Paris is the capital of France.",
            citations=[],
            confidence=0.9,
            duration_seconds=2.5
        )
        mock_pipeline_run.return_value = mock_response
        
        # Act
        result = subprocess.run(
            ["python", "-m", "app.cli", "ask", "What is the capital of France?"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Assert
        assert result.returncode == 0
        assert "Paris" in result.stdout
        assert "90%" in result.stdout or "0.9" in result.stdout  # Confidence display
    
    def test_cli_ask_with_json_output(self):
        """It should output JSON format when requested."""
        with patch('app.pipeline.default_pipeline.run') as mock_run:
            from app.core.state import ResearchResponse
            
            mock_response = ResearchResponse(
                answer="Test answer",
                citations=[],
                confidence=0.8
            )
            mock_run.return_value = mock_response
            
            # Act
            result = subprocess.run(
                ["python", "-m", "app.cli", "ask", "Test question", "--format", "json"],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # Assert
            assert result.returncode == 0
            # Should be valid JSON
            json_output = json.loads(result.stdout)
            assert json_output["answer"] == "Test answer"
            assert json_output["confidence"] == 0.8
    
    def test_cli_ingest_sample_command(self):
        """It should execute sample ingestion command."""
        with patch('app.rag.ingest.ingest_sample_data') as mock_ingest:
            mock_ingest.return_value = {
                "status": "success",
                "documents_processed": 2,
                "chunks_ingested": 5
            }
            
            # Act
            result = subprocess.run(
                ["python", "-m", "app.cli", "sample"],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # Assert
            assert result.returncode == 0
            assert "Sample data ingested successfully" in result.stdout
            assert "Documents: 2" in result.stdout
            assert "Chunks: 5" in result.stdout
    
    def test_cli_ingest_file_command(self, tmp_path):
        """It should ingest a single file."""
        # Arrange
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document for ingestion.")
        
        with patch('app.rag.ingest.DocumentIngester.ingest_file') as mock_ingest:
            mock_ingest.return_value = {
                "status": "success",
                "documents_processed": 1,
                "chunks_created": 1,
                "chunks_ingested": 1,
                "duplicates_removed": 0
            }
            
            # Act
            result = subprocess.run(
                ["python", "-m", "app.cli", "ingest", str(test_file)],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # Assert
            assert result.returncode == 0
            assert "Ingestion successful" in result.stdout
            assert "Documents processed: 1" in result.stdout
    
    def test_cli_ingest_nonexistent_file(self):
        """It should handle non-existent files gracefully."""
        # Act
        result = subprocess.run(
            ["python", "-m", "app.cli", "ingest", "/nonexistent/file.txt"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        # Assert
        assert result.returncode == 1
        assert "does not exist" in result.stderr or "does not exist" in result.stdout
    
    def test_cli_ask_error_handling(self):
        """It should handle errors in ask command."""
        with patch('app.pipeline.default_pipeline.run') as mock_run:
            mock_run.side_effect = Exception("Test error")
            
            # Act
            result = subprocess.run(
                ["python", "-m", "app.cli", "ask", "Test question"],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            # Assert
            assert result.returncode == 1
            assert "Error:" in result.stderr or "Error:" in result.stdout
    
    def test_cli_reset_command_with_confirmation(self):
        """It should handle reset command with confirmation."""
        with patch('typer.confirm', return_value=False):
            # Act
            result = subprocess.run(
                ["python", "-m", "app.cli", "reset"],
                capture_output=True,
                text=True,
                input="n\n",
                cwd=Path.cwd()
            )
            
            # Assert
            assert result.returncode == 0
            assert "Reset cancelled" in result.stdout
    
    def test_cli_ask_with_context(self):
        """It should pass context parameter correctly."""
        with patch('app.pipeline.default_pipeline.run') as mock_run:
            from app.core.state import ResearchResponse
            
            mock_response = ResearchResponse(
                answer="Answer with context",
                citations=[],
                confidence=0.7
            )
            mock_run.return_value = mock_response
            
            # Act
            result = subprocess.run([
                "python", "-m", "app.cli", "ask",
                "Test question",
                "--context", "Additional context info"
            ], capture_output=True, text=True, cwd=Path.cwd())
            
            # Assert
            assert result.returncode == 0
            
            # Verify context was passed to pipeline
            request_arg = mock_run.call_args[0][0]
            assert request_arg.context == "Additional context info"