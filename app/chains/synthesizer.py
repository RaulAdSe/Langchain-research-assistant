"""Synthesizer agent for producing final polished answers."""

from pathlib import Path
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import re
from langsmith import traceable
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
        
        # Create the chain without output parser (we'll handle JSON parsing manually)
        self.chain = (
            self.prompt
            | chat_model(agent_type="synthesizer")
        )
    
    def _parse_json_output(self, raw_output) -> Dict[str, Any]:
        """Parse JSON output with robust error handling."""
        try:
            # Extract content from AIMessage or string
            if hasattr(raw_output, 'content'):
                content = raw_output.content
            elif hasattr(raw_output, 'text'):
                content = raw_output.text
            else:
                content = str(raw_output)
            
            # Handle empty content
            if not content or content.strip() == "":
                print("Warning: Empty content received from LLM")
                raise json.JSONDecodeError("Empty content", "", 0)
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw output (first 1000 chars): {content[:1000]}")
            print(f"Raw output type: {type(raw_output)}")
            
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL | re.MULTILINE)
            if json_match:
                try:
                    json_content = json_match.group(1).strip()
                    return json.loads(json_content)
                except json.JSONDecodeError as e2:
                    print(f"Failed to parse extracted JSON: {e2}")
                    print(f"Extracted content: {json_match.group(1)[:200]}...")
            
            # Try to find JSON between first { and last }
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    json_content = content[start_idx:end_idx+1]
                    return json.loads(json_content)
                except json.JSONDecodeError as e3:
                    print(f"Failed to parse extracted bracket JSON: {e3}")
            
            # Try to find JSON object in the content
            json_match = re.search(r'\{[^{}]*"final"[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    # Fix common JSON issues
                    json_str = self._fix_json_string(json_str)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Fallback: create a basic structure from the content
            return {
                "final": content,
                "summary": "Error parsing structured output",
                "key_points": [],
                "caveats": [],
                "citations": [],
                "confidence": 0.5,
                "metadata": {"sources_used": 0, "primary_sources": 0, "answer_completeness": "partial"}
            }
    
    def _fix_json_string(self, json_str: str) -> str:
        """Fix common JSON formatting issues."""
        # Replace unescaped quotes in string values
        # This is a simplified fix - in production you'd want more robust parsing
        lines = json_str.split('\n')
        fixed_lines = []
        
        for line in lines:
            # If this line contains a field with string value, escape quotes within the value
            if ':' in line and '"' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key_part = parts[0]
                    value_part = parts[1].strip()
                    
                    # If value starts and ends with quotes, fix internal quotes
                    if value_part.startswith('"') and value_part.endswith('"') and len(value_part) > 2:
                        inner_content = value_part[1:-1]
                        # Escape any unescaped quotes
                        inner_content = inner_content.replace('\\"', '###ESCAPED###')
                        inner_content = inner_content.replace('"', '\\"')
                        inner_content = inner_content.replace('###ESCAPED###', '\\"')
                        value_part = f'"{inner_content}"'
                        line = f"{key_part}: {value_part}"
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
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
        
        # Add sources (only if not already present in final text)
        final_text = result.get("final", "")
        # Check if sources are already at the end of the final text
        has_sources_at_end = final_text and ("[#" in final_text[-200:] or "Sources" in final_text[-200:])
        
        if "citations" in result and result["citations"] and not has_sources_at_end:
            parts.append("**Sources**")
            
            # Check if we have mock sources and add disclaimer
            has_mock_sources = any(
                "wikipedia.org" in citation.get("url", "") or 
                "scholar.google.com" in citation.get("url", "") or
                "arxiv.org" in citation.get("url", "") or
                citation.get("url", "").startswith("local://")
                for citation in result["citations"]
            )
            
            if has_mock_sources:
                parts.append("*Note: Some links may be generic search URLs since no web search API is configured.*")
                parts.append("")
            
            for citation in result["citations"]:
                marker = citation.get("marker", "")
                title = citation.get("title", "Untitled")
                url = citation.get("url", "")
                date = citation.get("date", "")
                source_type = citation.get("source_type", "")
                
                # Format different source types (without dates)
                if source_type == "knowledge_base":
                    parts.append(f"{marker} {title} (Local Knowledge Base)")
                else:
                    parts.append(f"{marker} [{title}]({url})")
        
        return "\n".join(parts)
    
    @traceable(name="Synthesizer.synthesize") 
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
            raw_output = self.chain.invoke({
                "question": question,
                "findings": findings_str,
                "critique": critique_str,
                "draft": draft,
                "required_fixes": fixes_str
            })
            
            # Parse the JSON output with robust error handling
            result = self._parse_json_output(raw_output)
            
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