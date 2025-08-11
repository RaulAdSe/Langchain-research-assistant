"""Orchestrator agent for planning research strategies."""

from pathlib import Path
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable
from app.core.llm import chat_model
from app.core.state import PipelineState, update_state
import json


class OrchestratorChain:
    """Plans research strategies based on user questions."""
    
    def __init__(self):
        """Initialize the orchestrator chain."""
        # Load prompt from file
        prompt_path = Path("prompts/orchestrator.claude")
        if prompt_path.exists():
            self.system_prompt = prompt_path.read_text()
        else:
            self.system_prompt = self._get_default_prompt()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Question: {question}\nContext: {context}")
        ])
        
        # Create output parser
        self.output_parser = JsonOutputParser()
        
        # Create the chain with orchestrator-specific model
        self.chain = (
            self.prompt
            | chat_model(agent_type="orchestrator")
            | self.output_parser
        )
    
    def _get_default_prompt(self) -> str:
        """Get default prompt if file not found."""
        return """You are the Orchestrator, a planning agent. Given a user question, output a minimal, actionable research plan.

OBJECTIVES
- Pick the right tools: WebSearch (current info via DuckDuckGo), Retriever (local knowledge base)
- Break work into 2-5 steps max
- Note what evidence would falsify early assumptions
- Prefer high-quality, citable sources

OUTPUT SCHEMA (JSON)
{
  "plan": "Clear research strategy",
  "tool_sequence": ["web_search" | "retriever"],
  "key_terms": ["term1", "term2"],
  "search_strategy": "Explanation of approach",
  "validation_criteria": "What would confirm/refute findings"
}

RULES
- Don't fabricate sources or dates
- If unanswerable, say so in the plan
- Keep plan under 200 words"""
    
    @traceable(name="Orchestrator.plan")
    def plan(self, state: PipelineState) -> PipelineState:
        """
        Generate a research plan for the given question.
        
        Args:
            state: Current pipeline state
            
        Returns:
            Updated state with plan
        """
        try:
            # Extract question and context
            question = state.get("question", "")
            context = state.get("context", "")
            
            # Generate plan
            result = self.chain.invoke({
                "question": question,
                "context": context or "No additional context provided"
            })
            
            # Update state with plan details
            updated_state = update_state(
                state,
                plan=result.get("plan", ""),
                tool_sequence=result.get("tool_sequence", ["retriever", "web_search"]),
                key_terms=result.get("key_terms", []),
                search_strategy=result.get("search_strategy", "")
            )
            
            # Add validation criteria to state if present
            if "validation_criteria" in result:
                updated_state["validation_criteria"] = result["validation_criteria"]
            
            return updated_state
            
        except Exception as e:
            # Extract meaningful key terms from the question for better search
            question = state.get("question", "")
            # Remove common words and extract meaningful terms
            stop_words = {"what", "is", "are", "the", "a", "an", "and", "or", "but", "to", "of", "for", "in", "on", "at", "by", "with"}
            key_terms = [word.lower().strip("?.,!:;") for word in question.split() if word.lower() not in stop_words and len(word) > 2]
            
            # On error, return state with error and default plan
            print(f"Orchestrator error: {str(e)}")
            return update_state(
                state,
                error=f"Orchestrator error: {str(e)}",
                plan="Default plan: Search knowledge base and web for relevant information",
                tool_sequence=["retriever", "web_search"],
                key_terms=key_terms[:5]  # Take first 5 meaningful terms
            )
    
    async def aplan(self, state: PipelineState) -> PipelineState:
        """Async version of plan."""
        try:
            # Extract question and context
            question = state.get("question", "")
            context = state.get("context", "")
            
            # Generate plan using async invoke
            result = await self.chain.ainvoke({
                "question": question,
                "context": context or "No additional context provided"
            })
            
            # Update state with plan details
            updated_state = update_state(
                state,
                plan=result.get("plan", ""),
                tool_sequence=result.get("tool_sequence", ["retriever", "web_search"]),
                key_terms=result.get("key_terms", []),
                search_strategy=result.get("search_strategy", "")
            )
            
            # Add validation criteria to state if present
            if "validation_criteria" in result:
                updated_state["validation_criteria"] = result["validation_criteria"]
            
            return updated_state
            
        except Exception as e:
            # Extract meaningful key terms from the question for better search
            question = state.get("question", "")
            # Remove common words and extract meaningful terms
            stop_words = {"what", "is", "are", "the", "a", "an", "and", "or", "but", "to", "of", "for", "in", "on", "at", "by", "with"}
            key_terms = [word.lower().strip("?.,!:;") for word in question.split() if word.lower() not in stop_words and len(word) > 2]
            
            # On error, return state with error and default plan
            print(f"Orchestrator error: {str(e)}")
            return update_state(
                state,
                error=f"Orchestrator error: {str(e)}",
                plan="Default plan: Search knowledge base and web for relevant information",
                tool_sequence=["retriever", "web_search"],
                key_terms=key_terms[:5]  # Take first 5 meaningful terms
            )


# Create singleton instance
orchestrator = OrchestratorChain()