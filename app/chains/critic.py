"""Critic agent for reviewing research findings."""

from pathlib import Path
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langsmith import traceable
from app.core.llm import chat_model
from app.core.state import PipelineState, update_state, CritiqueIssue
import json
import re


class CriticChain:
    """Reviews research findings for quality and completeness."""
    
    def __init__(self):
        """Initialize the critic chain."""
        # Load prompt from file
        prompt_path = Path("prompts/critic.claude")
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text()
        else:
            self.system_prompt = self._get_default_prompt()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Question: {question}
Final Answer: {final_answer}
Findings: {findings}
Citations: {citations}

Review the final synthesized answer for accuracy, completeness, and quality.""")
        ])
        
        # Create output parser
        self.output_parser = JsonOutputParser()
        
        # Create the chain
        self.chain = (
            self.prompt
            | chat_model()
            | self.output_parser
        )
    
    def _get_default_prompt(self) -> str:
        """Get default prompt if file not found."""
        return """You are the Critic. Review research findings for accuracy and completeness.

CHECKS
- Claims supported by evidence?
- Missing recent developments?
- Ambiguous statements?
- Source quality?
- Logical consistency?

OUTPUT SCHEMA (JSON)
{
  "issues": [
    {
      "issue_type": "missing_evidence" | "outdated_source" | "ambiguous_claim" | "bias" | "contradiction",
      "description": "Issue description",
      "severity": "critical" | "major" | "minor",
      "suggested_fix": "How to fix"
    }
  ],
  "required_fixes": ["Fix 1", "Fix 2"],
  "quality_score": 0.0-1.0,
  "strengths": ["Strength 1"],
  "missing_perspectives": ["Perspective 1"],
  "fact_check_notes": ["Note 1"]
}

RULES
- Be specific and constructive
- Reference citations ([#1])
- Quality: >0.8 excellent, 0.6-0.8 good, <0.6 needs work
- Max 5 critical, 10 total issues"""
    
    @traceable(name="Critic.critique")
    def critique(self, state: PipelineState) -> PipelineState:
        """
        Critique the research findings.
        
        Args:
            state: Current pipeline state with findings
            
        Returns:
            Updated state with critique
        """
        try:
            # Extract relevant information
            question = state.get("question", "")
            final_answer = state.get("final", "")
            findings = state.get("findings", [])
            citations = state.get("citations", [])
            
            # Format findings for prompt
            findings_str = json.dumps(findings, indent=2) if findings else "No findings"
            citations_str = json.dumps(citations, indent=2) if citations else "No citations"
            
            # Generate critique
            result = self.chain.invoke({
                "question": question,
                "final_answer": final_answer or "No final answer generated",
                "findings": findings_str,
                "citations": citations_str
            })
            
            # Process issues into typed format
            issues = []
            for issue_dict in result.get("issues", []):
                issue = CritiqueIssue(
                    issue_type=issue_dict.get("issue_type", "unknown"),
                    description=issue_dict.get("description", ""),
                    severity=issue_dict.get("severity", "minor"),
                    suggested_fix=issue_dict.get("suggested_fix")
                )
                issues.append(issue)
            
            # Calculate reliable quality score combining objective and subjective measures
            quality_score = result.get("quality_score", None)
            if quality_score is None or quality_score == 0.0:
                # Auto-calculate based on issues found
                critical_issues = sum(1 for i in issues if i.severity == "critical")
                major_issues = sum(1 for i in issues if i.severity == "major")
                minor_issues = sum(1 for i in issues if i.severity == "minor")
                
                # Start with perfect score and deduct
                quality_score = 1.0
                quality_score -= critical_issues * 0.3  # Critical issues heavily impact score
                quality_score -= major_issues * 0.15    # Major issues moderately impact
                quality_score -= minor_issues * 0.05    # Minor issues slightly impact
                quality_score = max(0.1, min(1.0, quality_score))  # Clamp between 0.1 and 1.0
            
            # Add objective quality components to reduce stochasticity
            objective_quality = self._calculate_objective_quality(state)
            # Blend subjective LLM score (70%) with objective measures (30%)
            final_quality_score = quality_score * 0.7 + objective_quality * 0.3
            
            # Update state with critique
            updated_state = update_state(
                state,
                critique=result,
                issues=issues,
                required_fixes=result.get("required_fixes", []),
                quality_score=final_quality_score,
                subjective_quality=quality_score,
                objective_quality=objective_quality
            )
            
            # Add additional critique metadata if present
            if "strengths" in result:
                updated_state["strengths"] = result["strengths"]
            if "missing_perspectives" in result:
                updated_state["missing_perspectives"] = result["missing_perspectives"]
            if "fact_check_notes" in result:
                updated_state["fact_check_notes"] = result["fact_check_notes"]
            
            return updated_state
            
        except Exception as e:
            # On error, return state with minimal critique
            return update_state(
                state,
                error=f"Critic error: {str(e)}",
                critique={"error": str(e)},
                issues=[],
                required_fixes=[],
                quality_score=0.5
            )
    
    async def acritique(self, state: PipelineState) -> PipelineState:
        """Async version of critique."""
        return self.critique(state)
    
    def _calculate_objective_quality(self, state: Dict[str, Any]) -> float:
        """Calculate objective quality metrics to reduce LLM stochasticity."""
        final_answer = state.get("final", "")
        findings = state.get("findings", [])
        citations = state.get("citations", [])
        
        # Initialize score
        score = 0.0
        max_score = 0.0
        
        # 1. Answer comprehensiveness (0.4 weight)
        answer_length = len(final_answer)
        if answer_length >= 1500:  # Comprehensive
            score += 0.4
        elif answer_length >= 1000:  # Good
            score += 0.3
        elif answer_length >= 500:  # Adequate
            score += 0.2
        else:  # Too short
            score += 0.1
        max_score += 0.4
        
        # 2. Source quality and quantity (0.3 weight)
        citation_count = len(citations)
        if citation_count >= 5:
            score += 0.3
        elif citation_count >= 3:
            score += 0.2
        elif citation_count >= 1:
            score += 0.1
        max_score += 0.3
        
        # 3. Evidence quality (0.2 weight)
        if findings:
            avg_confidence = sum(f.get("confidence", 0.5) for f in findings) / len(findings)
            score += avg_confidence * 0.2
        max_score += 0.2
        
        # 4. Structure indicators (0.1 weight)
        has_headers = bool(re.search(r'#{1,3}\s+\w+', final_answer))
        has_sections = "**" in final_answer or len(re.findall(r'\n\n', final_answer)) >= 3
        if has_headers and has_sections:
            score += 0.1
        elif has_headers or has_sections:
            score += 0.05
        max_score += 0.1
        
        # Normalize to 0-1 range
        return min(1.0, score / max_score if max_score > 0 else 0.0)


# Create singleton instance
critic = CriticChain()