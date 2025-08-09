"""Critic agent for reviewing research findings."""

from pathlib import Path
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.core.llm import chat_model
from app.core.state import PipelineState, update_state, CritiqueIssue
import json


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
Findings: {findings}
Draft: {draft}
Citations: {citations}

Review the research for accuracy, completeness, and potential issues.""")
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
            findings = state.get("findings", [])
            draft = state.get("draft", "")
            citations = state.get("citations", [])
            
            # Format findings for prompt
            findings_str = json.dumps(findings, indent=2) if findings else "No findings"
            citations_str = json.dumps(citations, indent=2) if citations else "No citations"
            
            # Generate critique
            result = self.chain.invoke({
                "question": question,
                "findings": findings_str,
                "draft": draft,
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
            
            # Update state with critique
            updated_state = update_state(
                state,
                critique=result,
                issues=issues,
                required_fixes=result.get("required_fixes", []),
                quality_score=result.get("quality_score", 0.5)
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


# Create singleton instance
critic = CriticChain()