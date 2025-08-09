"""Researcher agent for executing research plans with tools."""

from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_structured_chat_agent, AgentExecutor
from langsmith import traceable
from app.core.llm import chat_model
from app.core.state import PipelineState, update_state, Finding, Citation
from app.tools import AVAILABLE_TOOLS
import json


class ResearcherChain:
    """Executes research plans using available tools."""
    
    def __init__(self):
        """Initialize the researcher chain."""
        # Load prompt from file
        prompt_path = Path("prompts/researcher.claude")
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text()
        else:
            self.system_prompt = self._get_default_prompt()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Question: {question}
Plan: {plan}
Key Terms: {key_terms}
Tool Sequence: {tool_sequence}

Execute the research plan using the available tools and compile findings with citations.""")
        ])
        
        # Create output parser
        self.output_parser = JsonOutputParser()
        
        # Initialize tools
        self.tools = list(AVAILABLE_TOOLS.values())
    
    def _get_default_prompt(self) -> str:
        """Get default prompt if file not found."""
        return """You are the Researcher. Execute the plan using available tools and compile findings with citations.

PROCESS
1. Run tools systematically
2. Extract relevant quotes with sources
3. Build findings with evidence
4. Create draft with [#] markers
5. Identify gaps

OUTPUT SCHEMA (JSON)
{
  "findings": [
    {
      "claim": "Factual claim",
      "evidence": "Supporting quote",
      "source": {
        "title": "Source title",
        "url": "URL",
        "date": "Date if available",
        "snippet": "Excerpt"
      },
      "confidence": 0.0-1.0
    }
  ],
  "draft": "Answer with [#1] citations",
  "citations": [
    {"marker": "[#1]", "url": "URL", "title": "Title", "date": "Date"}
  ],
  "gaps": ["Gap 1"],
  "next_queries": ["Follow-up query"]
}

RULES
- Provide exact quotes
- Prefer primary sources
- Mark low-confidence items
- 100-300 word draft"""
    
    def _execute_tools(self, state: PipelineState) -> Dict[str, Any]:
        """Execute tools based on the plan."""
        tool_results = []
        tool_sequence = state.get("tool_sequence", ["retriever"])
        key_terms = state.get("key_terms", [])
        question = state.get("question", "")
        
        # Build search query from key terms and question
        search_query = " ".join(key_terms) if key_terms else question
        
        for tool_name in tool_sequence:
            if tool_name in AVAILABLE_TOOLS:
                tool = AVAILABLE_TOOLS[tool_name]
                
                try:
                    # Execute tool with appropriate parameters
                    start_time = datetime.now()
                    
                    if tool_name == "retriever":
                        result = tool._run(query=search_query, top_k=5)
                    elif tool_name == "web_search":
                        result = tool._run(query=search_query, top_k=5)
                    elif tool_name == "firecrawl":
                        # For firecrawl, we'd need URLs from previous searches
                        # Skip if no URLs available
                        continue
                    else:
                        result = tool._run(search_query)
                    
                    duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
                    
                    tool_results.append({
                        "tool_name": tool_name,
                        "input": {"query": search_query},
                        "output": result,
                        "timestamp": datetime.now().isoformat(),
                        "duration_ms": duration_ms
                    })
                    
                except Exception as e:
                    tool_results.append({
                        "tool_name": tool_name,
                        "input": {"query": search_query},
                        "output": {"error": str(e)},
                        "timestamp": datetime.now().isoformat(),
                        "duration_ms": 0
                    })
        
        return {"tool_results": tool_results}
    
    def _compile_findings(self, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile findings from tool results."""
        findings = []
        citations = []
        citation_counter = 1
        
        for tool_result in tool_results:
            tool_name = tool_result["tool_name"]
            output = tool_result.get("output", {})
            
            if "error" in output:
                continue
            
            if tool_name == "retriever":
                contexts = output.get("contexts", [])
                for ctx in contexts[:3]:  # Top 3 contexts
                    finding = {
                        "claim": f"Information from knowledge base",
                        "evidence": ctx.get("content", "")[:200],
                        "source": {
                            "title": ctx.get("filename", "Knowledge Base"),
                            "url": ctx.get("source", ""),
                            "date": None,
                            "snippet": ctx.get("content", "")[:100]
                        },
                        "confidence": min(ctx.get("score", 0.5) + 0.3, 1.0)
                    }
                    findings.append(finding)
                    
                    citation = {
                        "marker": f"[#{citation_counter}]",
                        "url": ctx.get("source", ""),
                        "title": ctx.get("filename", "Knowledge Base"),
                        "date": None
                    }
                    citations.append(citation)
                    citation_counter += 1
            
            elif tool_name == "web_search":
                results = output.get("results", [])
                for result in results[:3]:  # Top 3 results
                    finding = {
                        "claim": result.get("title", "Web search result"),
                        "evidence": result.get("snippet", ""),
                        "source": {
                            "title": result.get("title", ""),
                            "url": result.get("url", ""),
                            "date": result.get("published_at"),
                            "snippet": result.get("snippet", "")
                        },
                        "confidence": 0.7  # Default confidence for web results
                    }
                    findings.append(finding)
                    
                    citation = {
                        "marker": f"[#{citation_counter}]",
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "date": result.get("published_at")
                    }
                    citations.append(citation)
                    citation_counter += 1
        
        # Create a simple draft
        draft_parts = []
        for i, finding in enumerate(findings[:5], 1):
            draft_parts.append(f"{finding['evidence'][:100]}... [{i}]")
        
        draft = " ".join(draft_parts) if draft_parts else "No relevant information found."
        
        return {
            "findings": findings,
            "citations": citations,
            "draft": draft,
            "gaps": ["More specific information needed"] if not findings else [],
            "next_queries": []
        }
    
    @traceable(name="Researcher.research")
    def research(self, state: PipelineState) -> PipelineState:
        """
        Execute research based on the plan.
        
        Args:
            state: Current pipeline state with plan
            
        Returns:
            Updated state with findings
        """
        try:
            # Execute tools
            tool_execution = self._execute_tools(state)
            tool_results = tool_execution.get("tool_results", [])
            
            # Compile findings from tool results
            compiled = self._compile_findings(tool_results)
            
            # Update state
            updated_state = update_state(
                state,
                findings=compiled.get("findings", []),
                citations=compiled.get("citations", []),
                draft=compiled.get("draft", ""),
                gaps=compiled.get("gaps", []),
                tool_calls=tool_results
            )
            
            return updated_state
            
        except Exception as e:
            return update_state(
                state,
                error=f"Researcher error: {str(e)}",
                findings=[],
                citations=[],
                draft="Error occurred during research",
                gaps=["Unable to complete research"]
            )
    
    async def aresearch(self, state: PipelineState) -> PipelineState:
        """Async version of research."""
        return self.research(state)


# Create singleton instance
researcher = ResearcherChain()