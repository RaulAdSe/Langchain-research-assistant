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
            
            # Check for content novelty to prevent meaningless iterations
            content_novelty = self._assess_content_novelty(state, iteration)
            if content_novelty["is_stagnant"]:
                yield {
                    "type": "content_stagnation",
                    "reason": content_novelty["reason"],
                    "iteration": iteration,
                    "recommendation": content_novelty["recommendation"]
                }
            
            # Track iteration history
            final_answer = state.get("final", "")
            iteration_entry = {
                "iteration": iteration,
                "quality_score": quality_score,
                "findings_count": findings_count,
                "improvements": state.get("required_fixes", []),
                "final_answer_length": len(final_answer.split()) if final_answer else 0,
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
            content_gaps.append(f"Strengthen {len(low_confidence_findings)} findings with better evidence")
            search_recommendations.append("Search for more specific data and expert analysis")
        
        # Check for authoritative source types (based on what we can actually provide)
        authoritative_sources = ["Nature", "IEEE", "Science", "EPA", "IEA", "IRENA", "Academic", "Government"]
        has_authoritative = any(
            any(auth in str(f.get("source", "")) for auth in authoritative_sources)
            for f in findings
        )
        if not has_authoritative:
            content_gaps.append("Lacks authoritative institutional sources for credibility")
            search_recommendations.append("Search for institutional and research organization data")
        
        # Generate content-focused improvements based on question type and current answer
        question = state.get("question", "")
        if "what" in question.lower():
            search_recommendations.append("Find more detailed definitions and comprehensive explanations")
        if "how" in question.lower():
            search_recommendations.append("Search for step-by-step processes and practical methods")
        if "why" in question.lower():
            search_recommendations.append("Look for causal relationships and underlying mechanisms")
        if "benefit" in question.lower() or "advantage" in question.lower():
            search_recommendations.append("Find quantitative data and specific real-world examples")
        if "compare" in question.lower() or "vs" in question.lower() or "versus" in question.lower():
            search_recommendations.append("Search for side-by-side comparisons and contrasting data")
        
        # Content quality improvements
        if final_answer and len(final_answer.split()) < 200:
            answer_improvements.append("Answer needs more depth and detail")
            search_recommendations.append("Find additional context and background information")
        
        # Check for balanced perspective
        if final_answer and not any(word in final_answer.lower() for word in ["however", "although", "but", "despite", "challenge", "limitation"]):
            answer_improvements.append("Lacks balanced perspective - needs challenges/limitations")
            search_recommendations.append("Search for potential drawbacks, challenges, and counterarguments")
        
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
    
    def _assess_content_novelty(self, state: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        """Assess if the current iteration added meaningful new content."""
        iteration_history = state.get("iteration_history", [])
        current_findings = state.get("findings", [])
        current_answer = state.get("final", "")
        
        # Skip assessment for first iteration
        if iteration <= 1:
            return {"is_stagnant": False, "reason": "First iteration"}
        
        # Check if findings count has increased meaningfully
        previous_iteration = iteration_history[-1] if iteration_history else {}
        previous_findings_count = previous_iteration.get("findings_count", 0)
        current_findings_count = len(current_findings)
        
        # Check for content stagnation indicators
        stagnation_reasons = []
        
        # 1. No new findings added
        if current_findings_count <= previous_findings_count:
            stagnation_reasons.append("No new findings discovered")
        
        # 2. Very similar final answer length (within 5%)
        if iteration >= 2:
            previous_answers = [h.get("final_answer_length", 0) for h in iteration_history[-2:]]
            current_length = len(current_answer.split())
            
            if previous_answers:
                avg_previous_length = sum(previous_answers) / len(previous_answers)
                if avg_previous_length > 0:
                    length_change = abs(current_length - avg_previous_length) / avg_previous_length
                    if length_change < 0.05:  # Less than 5% change
                        stagnation_reasons.append("Answer length unchanged")
        
        # 3. Quality hasn't improved for 2 iterations
        if len(iteration_history) >= 2:
            recent_qualities = [h.get("quality_score", 0) for h in iteration_history[-2:]]
            current_quality = state.get("quality_score", 0)
            
            if recent_qualities and all(q > 0 for q in recent_qualities):
                if current_quality <= max(recent_qualities):
                    stagnation_reasons.append("Quality not improving")
        
        # 4. Identical findings content (check similarity)
        if iteration >= 2 and current_findings:
            # Simple content similarity check
            current_claims = set()
            for finding in current_findings:
                claim = finding.get("claim", "").lower()
                if claim:
                    # Extract key words from claim
                    words = set(claim.split())
                    current_claims.update(words)
            
            # Check if we're seeing very similar content
            if len(current_claims) < 20:  # Very limited vocabulary suggests repetition
                stagnation_reasons.append("Limited content diversity")
        
        # Determine if stagnant
        is_stagnant = len(stagnation_reasons) >= 2  # Multiple indicators
        
        # Generate recommendations
        recommendations = []
        if "No new findings discovered" in stagnation_reasons:
            recommendations.append("Try different search terms or approaches")
        if "Quality not improving" in stagnation_reasons:
            recommendations.append("Focus on synthesis improvements rather than more research")
        if "Limited content diversity" in stagnation_reasons:
            recommendations.append("Explore different aspects or perspectives of the topic")
        
        return {
            "is_stagnant": is_stagnant,
            "reason": "; ".join(stagnation_reasons) if stagnation_reasons else "Content progressing normally",
            "recommendation": "; ".join(recommendations) if recommendations else "Continue iterating",
            "indicators_count": len(stagnation_reasons)
        }


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