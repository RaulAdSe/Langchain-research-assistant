"""Synthesizer agent for producing final polished answers."""

from pathlib import Path
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.core.llm import chat_model
from app.core.state import PipelineState, update_state
import json


class SynthesizerChain:
    """Produces final, well-structured answers incorporating critic feedback."""
    
    def __init__(self):
        """Initialize the synthesizer chain."""
        # Load prompt from file
        prompt_path = Path("prompts/synthesizer.claude")
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text()
        else:
            self.system_prompt = self._get_default_prompt()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Question: {question}
Findings: {findings}
Critique: {critique}
Draft: {draft}
Required Fixes: {required_fixes}

Produce a comprehensive, well-structured final answer incorporating all feedback.""")
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
        return """You are the Synthesizer. Produce a polished answer incorporating critic feedback.

RESPONSE STRUCTURE
- Executive Summary (3-5 sentences)
- Key Points (bulleted)
- Detailed Analysis (if needed)
- Caveats and Limitations
- Sources with citations

OUTPUT SCHEMA (JSON)
{
  "final": "Complete markdown answer with [#1] citations",
  "summary": "3-5 sentence summary",
  "key_points": ["Point 1", "Point 2"],
  "caveats": ["Limitation 1"],
  "citations": [
    {"marker": "[#1]", "url": "URL", "title": "Title", "date": "Date"}
  ],
  "confidence": 0.0-1.0,
  "metadata": {
    "sources_used": 5,
    "primary_sources": 3,
    "answer_completeness": "complete" | "partial" | "conditional"
  }
}

RULES
- Incorporate ALL required fixes
- No new claims beyond findings
- 200-500 words typically
- End with Sources section"""
    
    def _format_final_answer(self, result: Dict[str, Any], state: PipelineState) -> str:
        """Format the final answer in markdown."""
        parts = []
        
        # Add summary if present
        if "summary" in result:
            parts.append(f"**Summary**\n{result['summary']}\n")
        
        # Add key points
        if "key_points" in result and result["key_points"]:
            parts.append("**Key Points**")
            for point in result["key_points"]:
                parts.append(f"- {point}")
            parts.append("")
        
        # Add main content (if different from summary)
        if "final" in result and result["final"] != result.get("summary", ""):
            parts.append("**Details**")
            parts.append(result["final"])
            parts.append("")
        
        # Add caveats
        if "caveats" in result and result["caveats"]:
            parts.append("**Caveats and Limitations**")
            for caveat in result["caveats"]:
                parts.append(f"- {caveat}")
            parts.append("")
        
        # Add sources
        if "citations" in result and result["citations"]:
            parts.append("**Sources**")
            for citation in result["citations"]:
                marker = citation.get("marker", "")
                title = citation.get("title", "Untitled")
                url = citation.get("url", "")
                date = citation.get("date", "")
                
                if date:
                    parts.append(f"{marker} [{title}]({url}) - {date}")
                else:
                    parts.append(f"{marker} [{title}]({url})")
        
        return "\n".join(parts)
    
    def synthesize(self, state: PipelineState) -> PipelineState:
        """
        Synthesize the final answer.
        
        Args:
            state: Current pipeline state with findings and critique
            
        Returns:
            Updated state with final answer
        """
        try:
            # Extract relevant information
            question = state.get("question", "")
            findings = state.get("findings", [])
            critique = state.get("critique", {})
            draft = state.get("draft", "")
            required_fixes = state.get("required_fixes", [])
            
            # Format inputs
            findings_str = json.dumps(findings, indent=2) if findings else "No findings"
            critique_str = json.dumps(critique, indent=2) if critique else "No critique"
            fixes_str = json.dumps(required_fixes) if required_fixes else "[]"
            
            # Generate final answer
            result = self.chain.invoke({
                "question": question,
                "findings": findings_str,
                "critique": critique_str,
                "draft": draft,
                "required_fixes": fixes_str
            })
            
            # Format the final answer
            final_answer = self._format_final_answer(result, state)
            
            # Update state with final answer
            updated_state = update_state(
                state,
                final=final_answer,
                summary=result.get("summary", ""),
                key_points=result.get("key_points", []),
                caveats=result.get("caveats", []),
                confidence=result.get("confidence", 0.7),
                citations=result.get("citations", state.get("citations", []))
            )
            
            # Add metadata if present
            if "metadata" in result:
                updated_state["answer_metadata"] = result["metadata"]
            
            # Mark as complete
            updated_state["end_time"] = datetime.utcnow().isoformat()
            
            return updated_state
            
        except Exception as e:
            # On error, use the draft as final answer
            return update_state(
                state,
                error=f"Synthesizer error: {str(e)}",
                final=state.get("draft", "Unable to generate final answer"),
                summary="Error occurred during synthesis",
                confidence=0.3
            )
    
    async def asynthesize(self, state: PipelineState) -> PipelineState:
        """Async version of synthesize."""
        return self.synthesize(state)


# Import datetime for timestamp
from datetime import datetime

# Create singleton instance
synthesizer = SynthesizerChain()