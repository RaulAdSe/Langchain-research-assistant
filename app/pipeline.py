"""Main pipeline orchestrating all agents."""

from typing import Dict, Any, Optional
from datetime import datetime
from app.core.state import PipelineState, init_state, ResearchRequest, ResearchResponse
from app.chains import orchestrator, researcher, critic, synthesizer
import traceback


class ResearchPipeline:
    """Orchestrates the multi-agent research workflow."""
    
    def __init__(self, max_iterations: int = 1):
        """
        Initialize the research pipeline.
        
        Args:
            max_iterations: Maximum number of research-critique iterations
        """
        self.max_iterations = max_iterations
        self.orchestrator = orchestrator
        self.researcher = researcher
        self.critic = critic
        self.synthesizer = synthesizer
    
    def run(self, request: ResearchRequest) -> ResearchResponse:
        """
        Run the complete research pipeline.
        
        Args:
            request: Research request with question and parameters
            
        Returns:
            Research response with answer and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Initialize state
            state = init_state(request.question, request.context)
            
            # Phase 1: Orchestrator plans the research
            print("📋 Planning research strategy...")
            state = self.orchestrator.plan(state)
            
            if state.get("error"):
                raise Exception(f"Planning failed: {state['error']}")
            
            # Phase 2: Researcher executes the plan
            print("🔍 Conducting research...")
            state = self.researcher.research(state)
            
            if state.get("error"):
                print(f"Warning: Research error - {state['error']}")
            
            # Phase 3: Critic reviews (with potential iteration)
            for iteration in range(self.max_iterations):
                print(f"🔎 Reviewing findings (iteration {iteration + 1})...")
                state = self.critic.critique(state)
                
                quality_score = state.get("quality_score", 0)
                required_fixes = state.get("required_fixes", [])
                
                # If quality is good enough or no fixes required, break
                if quality_score >= 0.7 or not required_fixes:
                    break
                
                # If fixes are required and we have iterations left, re-research
                if iteration < self.max_iterations - 1:
                    print(f"♻️ Addressing {len(required_fixes)} issues...")
                    # Update search strategy based on critique
                    state["key_terms"].extend(required_fixes[:2])  # Add fix keywords
                    state = self.researcher.research(state)
            
            # Phase 4: Synthesizer produces final answer
            print("✍️ Synthesizing final answer...")
            state = self.synthesizer.synthesize(state)
            
            # Calculate duration
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Build response
            response = ResearchResponse(
                answer=state.get("final", "No answer generated"),
                citations=state.get("citations", []),
                confidence=state.get("confidence", 0.5),
                summary=state.get("summary"),
                key_points=state.get("key_points"),
                caveats=state.get("caveats"),
                trace_url=self._get_trace_url(state),
                duration_seconds=duration
            )
            
            print(f"✅ Research complete (confidence: {response.confidence:.1%})")
            return response
            
        except Exception as e:
            # Handle errors gracefully
            print(f"❌ Pipeline error: {str(e)}")
            traceback.print_exc()
            
            return ResearchResponse(
                answer=f"An error occurred during research: {str(e)}",
                citations=[],
                confidence=0.0,
                summary="Error occurred",
                duration_seconds=(datetime.utcnow() - start_time).total_seconds()
            )
    
    def _get_trace_url(self, state: PipelineState) -> Optional[str]:
        """Get LangSmith trace URL if available."""
        # This would integrate with LangSmith to get the actual trace URL
        # For now, return a placeholder
        trace_id = state.get("trace_id")
        if trace_id:
            return f"https://smith.langchain.com/public/{trace_id}/r"
        return None
    
    async def arun(self, request: ResearchRequest) -> ResearchResponse:
        """Async version of run."""
        # For now, just call sync version
        # In production, would use async versions of all chains
        return self.run(request)


# Create default pipeline instance
default_pipeline = ResearchPipeline()


def research(question: str, context: Optional[str] = None, **kwargs) -> ResearchResponse:
    """
    Convenience function to run research.
    
    Args:
        question: The research question
        context: Optional additional context
        **kwargs: Additional parameters for ResearchRequest
        
    Returns:
        Research response with answer and metadata
    """
    request = ResearchRequest(
        question=question,
        context=context,
        **kwargs
    )
    
    return default_pipeline.run(request)