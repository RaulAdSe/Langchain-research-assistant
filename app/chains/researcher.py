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
import re


class ResearcherChain:
    """Executes research plans using available tools."""
    
    def _normalize_similarity_score(self, chromadb_score: float) -> float:
        """
        Normalize ChromaDB similarity score to 0.0-1.0 range.
        
        ChromaDB uses different distance metrics (L2, cosine, etc.) where:
        - Lower scores = more similar for L2 distance
        - Higher scores = more similar for cosine similarity
        
        Args:
            chromadb_score: Raw similarity score from ChromaDB
            
        Returns:
            Normalized relevance score between 0.0 and 1.0
        """
        # ChromaDB typically returns L2 distances where 0 = identical, higher = less similar
        # Convert to similarity score where 1 = most relevant, 0 = least relevant
        
        if chromadb_score < 0:
            # Negative scores shouldn't happen, but handle gracefully
            return 0.0
        elif chromadb_score > 2.0:
            # Very high L2 distance = very dissimilar
            return 0.0
        else:
            # Convert L2 distance to similarity: closer to 0 = more similar = higher score
            # Use exponential decay to map distances to [0, 1]
            import math
            similarity = math.exp(-chromadb_score)
            return min(1.0, similarity)
    
    def _assess_retriever_relevance(self, question: str, retriever_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess if retriever results are relevant using ChromaDB's embedding similarity scores.
        
        Args:
            question: The research question
            retriever_result: Results from retriever tool
            
        Returns:
            Dictionary with relevance assessment and filtered results
        """
        if 'error' in retriever_result:
            return {
                'relevant': False,
                'max_similarity': 0.0,
                'filtered_results': [],
                'assessment': f'Retriever error: {retriever_result["error"]}'
            }
        
        contexts = retriever_result.get('contexts', [])
        if not contexts:
            return {
                'relevant': False,
                'max_similarity': 0.0,
                'filtered_results': [],
                'assessment': 'No documents found in knowledge base'
            }
        
        # Use ChromaDB's similarity scores directly (already computed with embeddings)
        scored_contexts = []
        for context in contexts:
            chromadb_score = context.get('score', 0.0)
            # Normalize ChromaDB score to 0-1 relevance score
            relevance_score = self._normalize_similarity_score(chromadb_score)
            
            scored_contexts.append({
                **context,
                'relevance_score': relevance_score,
                'chromadb_score': chromadb_score
            })
        
        # Find max relevance
        max_similarity = max(ctx['relevance_score'] for ctx in scored_contexts) if scored_contexts else 0.0
        
        # Set similarity threshold for relevance (0.4 for embedding similarity)
        similarity_threshold = 0.4
        relevant_contexts = [ctx for ctx in scored_contexts if ctx['relevance_score'] >= similarity_threshold]
        
        is_relevant = max_similarity >= similarity_threshold
        
        # Create detailed assessment
        best_match = max(scored_contexts, key=lambda x: x['relevance_score']) if scored_contexts else None
        assessment_parts = [
            f"Best similarity: {max_similarity:.3f}",
            f"ChromaDB score: {best_match['chromadb_score']:.3f}" if best_match else "No results",
            f"Threshold: {similarity_threshold}",
            "✅ RELEVANT" if is_relevant else "❌ NOT RELEVANT"
        ]
        assessment = ", ".join(assessment_parts)
        
        return {
            'relevant': is_relevant,
            'max_similarity': max_similarity,
            'filtered_results': relevant_contexts,
            'assessment': assessment,
            'total_docs_checked': len(contexts),
            'relevant_docs_found': len(relevant_contexts),
            'threshold': similarity_threshold
        }
    
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
                        # Always query retriever but assess relevance
                        raw_result = tool._run(query=search_query, top_k=5)
                        relevance_assessment = self._assess_retriever_relevance(question, raw_result)
                        
                        # Log relevance assessment
                        print(f"📊 Retriever Relevance Assessment: {relevance_assessment['assessment']}")
                        print(f"   Documents checked: {relevance_assessment['total_docs_checked']}, Relevant: {relevance_assessment['relevant_docs_found']}")
                        
                        if relevance_assessment['relevant']:
                            # Use filtered relevant results
                            result = {
                                **raw_result,
                                'contexts': relevance_assessment['filtered_results'],
                                'relevance_filtered': True,
                                'max_similarity': relevance_assessment['max_similarity']
                            }
                            print(f"✅ Using {len(relevance_assessment['filtered_results'])} relevant local documents")
                            print(f"   Best similarity: {relevance_assessment['max_similarity']:.3f}")
                        else:
                            # Skip retriever results - not relevant
                            print(f"❌ Skipping retriever - local knowledge not relevant to: {search_query}")
                            continue  # Skip this tool
                            
                    elif tool_name == "web_search":
                        result = tool._run(query=search_query, top_k=5)
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
                    
                    # Create better citation for knowledge base documents
                    filename = ctx.get("filename", "Knowledge Base Document")
                    citation = {
                        "marker": f"[#{citation_counter}]",
                        "url": f"local://knowledge_base/{filename}",
                        "title": f"{filename} (Knowledge Base)",
                        "date": None,
                        "source_type": "knowledge_base"
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