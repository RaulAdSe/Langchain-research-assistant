"""Iterative research pipeline with quality-driven feedback loops and streaming."""

from typing import Dict, Any, List, AsyncIterator, Optional, Callable
from datetime import datetime, timezone
from langsmith import traceable
from app.core.state import PipelineState, update_state
from app.chains.orchestrator import orchestrator
from app.chains.researcher import researcher
from app.chains.critic import critic
from app.chains.synthesizer import synthesizer


class IterativeResearchPipeline:
    """
    Iterative pipeline where critic provides feedback to orchestrator for improvement.
    Quality only grows through cumulative research - never decreases.
    """
    
    def __init__(self, quality_threshold: float = 0.7, max_iterations: int = 3):
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.orchestrator = orchestrator
        self.researcher = researcher
        self.critic = critic
        self.synthesizer = synthesizer
    
    @traceable(name="IterativePipeline")
    async def run(
        self,
        question: str,
        context: str = "",
        verbose: bool = False
    ) -> PipelineState:
        """Run iterative pipeline with quality-driven improvements (non-streaming)."""
        async for final_state in self.astream(question, context, verbose):
            if final_state.get("type") == "pipeline_complete":
                return final_state["state"]
        # Fallback if no complete event
        return {}
    
    @traceable(name="IterativePipeline.Stream")
    async def astream(
        self,
        question: str,
        context: str = "",
        verbose: bool = False,
        stream_handler: Optional[Callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """Run iterative pipeline with quality-driven improvements."""
        
        # Initialize state
        state: PipelineState = {
            "question": question,
            "context": context,
            "iteration_count": 0,
            "iteration_history": [],
            "start_time": datetime.now(timezone.utc).isoformat(),
            "quality_threshold": self.quality_threshold,
            "max_iterations": self.max_iterations
        }
        
        # Emit pipeline start event
        yield {
            "type": "pipeline_start",
            "question": question,
            "quality_threshold": self.quality_threshold,
            "max_iterations": self.max_iterations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        for iteration in range(1, self.max_iterations + 1):
            # Emit iteration start
            yield {
                "type": "iteration_start", 
                "iteration": iteration,
                "max_iterations": self.max_iterations,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Update iteration count
            state["iteration_count"] = iteration
            
            # Phase 1: Orchestrator creates research plan
            yield {"type": "phase_start", "phase": "orchestrator", "description": "Planning research strategy"}
            state = self.orchestrator.plan(state)
            
            if state.get("error"):
                yield {"type": "error", "phase": "orchestrator", "error": state['error']}
                break
            
            # Emit orchestrator decision
            orchestrator_reasoning = state.get("orchestrator_reasoning", "")
            if orchestrator_reasoning:
                yield {
                    "type": "orchestrator_decision",
                    "reasoning": orchestrator_reasoning,
                    "next_action": state.get("next_action", "research")
                }
            
            # Phase 2: Execute Orchestrator's decision
            next_action = state.get("next_action", "research")
            
            if next_action == "synthesis_only":
                # Skip research, go directly to synthesis
                yield {
                    "type": "phase_start", 
                    "phase": "synthesizer", 
                    "description": "Rewriting answer with existing findings",
                    "findings_count": len(state.get("findings", []))
                }
                synthesis_instructions = state.get("synthesis_instructions", "")
                if synthesis_instructions:
                    yield {"type": "synthesis_instructions", "instructions": synthesis_instructions}
            else:
                # Standard path: research first
                yield {"type": "phase_start", "phase": "researcher", "description": "Gathering additional information"}
                state = self.researcher.research(state)
                
                if state.get("error"):
                    yield {"type": "error", "phase": "researcher", "error": state['error']}
                    break
                
                findings_count = len(state.get("findings", []))
                yield {"type": "research_complete", "findings_count": findings_count}
                yield {"type": "phase_start", "phase": "synthesizer", "description": "Creating comprehensive answer"}
            
            # Phase 3: Synthesizer creates/rewrites answer
            state = self.synthesizer.synthesize(state)
            
            if state.get("error"):
                yield {"type": "error", "phase": "synthesizer", "error": state['error']}
                break
            
            # Phase 4: Critic evaluates the complete synthesized answer
            yield {"type": "phase_start", "phase": "critic", "description": "Evaluating answer quality"}
            state = self.critic.critique(state)
            
            quality_score = state.get("quality_score", 0.0)
            subjective_quality = state.get("subjective_quality", quality_score)
            objective_quality = state.get("objective_quality", 0.0)
            
            # Emit quality assessment
            yield {
                "type": "quality_assessment",
                "quality_score": quality_score,
                "subjective_quality": subjective_quality,
                "objective_quality": objective_quality,
                "iteration": iteration
            }
            
            # Track iteration history
            iteration_entry = {
                "iteration": iteration,
                "quality_score": quality_score,
                "findings_count": findings_count,
                "improvements": state.get("required_fixes", []),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            state.setdefault("iteration_history", []).append(iteration_entry)
            
            # Check if quality threshold is met - STOP if reached
            if quality_score >= self.quality_threshold:
                yield {
                    "type": "threshold_reached",
                    "quality_score": quality_score,
                    "threshold": self.quality_threshold,
                    "iteration": iteration
                }
                break
            
            # If not the last iteration, prepare improvement feedback
            if iteration < self.max_iterations:
                yield {
                    "type": "quality_below_threshold",
                    "quality_score": quality_score,
                    "threshold": self.quality_threshold,
                    "iteration": iteration
                }
                
                # Create targeted feedback for next iteration
                feedback = self._create_improvement_feedback(state, quality_score, verbose)
                state["critic_feedback"] = feedback
                
                yield {
                    "type": "feedback_prepared",
                    "feedback_preview": feedback[:200] + "..." if len(feedback) > 200 else feedback,
                    "next_iteration": iteration + 1
                }
            else:
                yield {
                    "type": "max_iterations_reached",
                    "final_quality": quality_score,
                    "max_iterations": self.max_iterations,
                    "threshold": self.quality_threshold
                }
        
        # Final state is the result of the last iteration
        final_state = state
        
        # Add iteration metadata
        final_state["total_iterations"] = iteration
        final_state["final_quality"] = quality_score
        final_state["end_time"] = datetime.now(timezone.utc).isoformat()
        
        # Emit pipeline completion
        yield {
            "type": "pipeline_complete",
            "state": final_state,
            "total_iterations": iteration,
            "final_quality": quality_score,
            "final_confidence": final_state.get('confidence', 0.0),
            "timestamp": final_state["end_time"]
        }
    
    def _create_improvement_feedback(
        self, 
        state: Dict[str, Any], 
        current_quality: float, 
        verbose: bool
    ) -> str:
        """Create precise, actionable feedback for improving the synthesized answer."""
        required_fixes = state.get("required_fixes", [])
        issues = state.get("issues", [])
        missing_perspectives = state.get("missing_perspectives", [])
        findings = state.get("findings", [])
        final_answer = state.get("final", "")
        
        # Analyze the synthesized answer for improvement opportunities
        answer_improvements = []
        search_recommendations = []
        content_gaps = []
        
        # Analyze final answer content
        if final_answer:
            # Check answer length and depth
            if len(final_answer) < 500:
                answer_improvements.append("Answer lacks depth - needs more comprehensive coverage")
                search_recommendations.append("Find detailed explanations and comprehensive coverage")
            
            # Check for specific data/statistics
            has_numbers = any(char.isdigit() for char in final_answer)
            if not has_numbers:
                answer_improvements.append("Missing quantitative data and statistics")
                search_recommendations.append("Search for specific numbers, percentages, and statistical data")
            
            # Check for current information
            import re
            years = re.findall(r'\b20[12][0-9]\b', final_answer)
            recent_years = [y for y in years if int(y) >= 2020]
            if not recent_years:
                answer_improvements.append("Lacks recent/current information")
                search_recommendations.append("Add '2023 OR 2024' to search terms for current data")
        
        # Identify weak findings that need replacement
        low_confidence_findings = [f for f in findings if f.get("confidence", 0.5) < 0.7]
        if low_confidence_findings:
            content_gaps.append(f"Replace {len(low_confidence_findings)} low-confidence findings with stronger evidence")
            search_recommendations.append("Target authoritative sources (.edu, .gov, major research institutions)")
        
        # Check for authoritative sources
        has_edu_gov = any(".edu" in str(f.get("source", {}).get("url", "")) or 
                          ".gov" in str(f.get("source", {}).get("url", "")) 
                          for f in findings)
        if not has_edu_gov:
            content_gaps.append("Missing academic or government sources for credibility")
            search_recommendations.append("Include 'site:edu OR site:gov' in searches")
        
        # Generate targeted search recommendations based on question type
        question = state.get("question", "")
        if "what" in question.lower():
            search_recommendations.append("Search for comprehensive definitions and explanations")
        if "how" in question.lower():
            search_recommendations.append("Find step-by-step processes and methodologies")
        if "why" in question.lower():
            search_recommendations.append("Look for causal explanations and reasoning")
        if "benefit" in question.lower() or "advantage" in question.lower():
            search_recommendations.append("Find specific examples and case studies")
        
        # Use critic's specific feedback
        critic_issues = []
        for fix in required_fixes[:3]:
            critic_issues.append(fix)
        for issue in issues[:3]:
            critic_issues.append(str(issue))
        
        feedback = f"""
ANSWER QUALITY IMPROVEMENT PLAN

Current Answer Quality: {current_quality:.2f} → Target: {self.quality_threshold:.2f}

FINAL ANSWER WEAKNESSES:
{chr(10).join(f"• {improvement}" for improvement in answer_improvements[:3])}

CONTENT GAPS TO FILL:
{chr(10).join(f"• {gap}" for gap in content_gaps[:3])}

CRITIC'S SPECIFIC ISSUES:
{chr(10).join(f"• {issue}" for issue in critic_issues[:3])}

MISSING PERSPECTIVES:
{chr(10).join(f"• {perspective}" for perspective in missing_perspectives[:3])}

TARGETED RESEARCH STRATEGY:
{chr(10).join(f"• {rec}" for rec in search_recommendations[:4])}

NEXT ITERATION GOALS:
1. Address each weakness in the final answer
2. Fill content gaps with high-quality research
3. Add more authoritative sources and recent data
4. Improve answer comprehensiveness and depth
5. Build upon current findings - don't start over

TARGET: Each iteration must genuinely improve the final answer quality.
"""
        
        # Feedback creation complete - no verbose prints needed in streaming
        
        return feedback


# Convenience function
async def run_iterative_research(
    question: str,
    context: str = "",
    quality_threshold: float = 0.7,
    max_iterations: int = 3,
    verbose: bool = False
) -> PipelineState:
    """Run iterative research with quality-driven improvements."""
    pipeline = IterativeResearchPipeline(quality_threshold, max_iterations)
    return await pipeline.run(question, context, verbose)