"""Streaming-enabled research pipeline with real-time updates."""

import asyncio
from typing import AsyncIterator, Dict, Any, Optional, Callable
from datetime import datetime
import json
from langchain_core.runnables.config import RunnableConfig
from langchain_core.callbacks import AsyncCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langsmith import traceable

from app.core.state import PipelineState, update_state
from app.core.config import settings
from app.chains.orchestrator import orchestrator
from app.chains.researcher import researcher
from app.chains.critic import critic
from app.chains.synthesizer import synthesizer
from app.tools.retriever import retriever_tool
from app.tools.web_search import web_search_tool


class StreamingCallback(AsyncCallbackHandler):
    """Custom callback to stream progress updates."""
    
    def __init__(self, stream_handler: Optional[Callable] = None):
        self.stream_handler = stream_handler or print
        
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain starts."""
        chain_name = serialized.get("name", "Unknown")
        await self._emit({
            "type": "chain_start",
            "chain": chain_name,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when a chain ends."""
        await self._emit({
            "type": "chain_end",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts."""
        tool_name = serialized.get("name", "Unknown")
        await self._emit({
            "type": "tool_start",
            "tool": tool_name,
            "input": input_str[:100],  # First 100 chars
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool ends."""
        await self._emit({
            "type": "tool_end",
            "output_preview": output[:100] if output else "",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs) -> None:
        """Called when an LLM starts."""
        model = serialized.get("kwargs", {}).get("model_name", "Unknown")
        await self._emit({
            "type": "llm_start",
            "model": model,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Stream tokens as they're generated."""
        await self._emit({
            "type": "token",
            "content": token
        })
    
    async def _emit(self, event: Dict[str, Any]):
        """Emit event to stream handler."""
        if asyncio.iscoroutinefunction(self.stream_handler):
            await self.stream_handler(event)
        else:
            self.stream_handler(event)


class StreamingResearchPipeline:
    """Research pipeline with streaming support."""
    
    def __init__(self):
        self.orchestrator = orchestrator
        self.researcher = researcher
        self.critic = critic
        self.synthesizer = synthesizer
    
    @traceable(name="StreamingPipeline")
    async def astream(
        self,
        question: str,
        context: Optional[str] = None,
        fast_mode: bool = False,
        stream_handler: Optional[Callable] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream research pipeline execution with real-time updates.
        
        Yields:
            Dict containing event type and data
        """
        # Initialize callback
        callback = StreamingCallback(stream_handler)
        config = RunnableConfig(callbacks=[callback])
        
        # Initialize state
        state: PipelineState = {
            "question": question,
            "context": context or "",
            "plan": "",
            "tool_sequence": [],
            "key_terms": [],
            "findings": [],
            "critique": {},
            "required_fixes": [],
            "draft": "",
            "final": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
            "start_time": datetime.utcnow().isoformat()
        }
        
        try:
            # Phase 1: Orchestrator plans
            yield {
                "type": "phase_start",
                "phase": "orchestrator",
                "description": "Planning research strategy",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stream orchestrator output
            async for chunk in self._stream_agent(
                self.orchestrator.plan,
                state,
                config,
                "Orchestrator"
            ):
                yield chunk
            
            # Get orchestrator result
            state = await self.orchestrator.aplan(state)
            
            yield {
                "type": "phase_complete",
                "phase": "orchestrator",
                "plan": state.get("plan", ""),
                "tools": state.get("tool_sequence", []),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Phase 2: Researcher executes
            yield {
                "type": "phase_start",
                "phase": "researcher",
                "description": "Conducting research with tools",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Stream researcher execution
            async for chunk in self._stream_agent(
                self.researcher.research,
                state,
                config,
                "Researcher"
            ):
                yield chunk
            
            # For researcher, use sync version since it doesn't have async yet
            state = self.researcher.research(state)
            
            yield {
                "type": "phase_complete",
                "phase": "researcher",
                "findings_count": len(state.get("findings", [])),
                "draft_length": len(state.get("draft", "")),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Phase 3: Critic reviews (unless fast mode)
            if not fast_mode:
                yield {
                    "type": "phase_start",
                    "phase": "critic",
                    "description": "Reviewing answer quality",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                async for chunk in self._stream_agent(
                    self.critic.critique,
                    state,
                    config,
                    "Critic"
                ):
                    yield chunk
                
                # For critic, use sync version
                state = self.critic.critique(state)
                
                yield {
                    "type": "phase_complete",
                    "phase": "critic",
                    "quality_score": state.get("critique", {}).get("score", 0),
                    "fixes_required": len(state.get("required_fixes", [])),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                yield {
                    "type": "phase_skip",
                    "phase": "critic",
                    "reason": "fast_mode",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # Phase 4: Synthesizer creates final answer
            yield {
                "type": "phase_start",
                "phase": "synthesizer",
                "description": "Creating final answer",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            async for chunk in self._stream_agent(
                self.synthesizer.synthesize,
                state,
                config,
                "Synthesizer"
            ):
                yield chunk
            
            # For synthesizer, use sync version  
            state = self.synthesizer.synthesize(state)
            
            yield {
                "type": "phase_complete",
                "phase": "synthesizer",
                "confidence": state.get("confidence", 0),
                "answer_length": len(state.get("final", "")),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Final result
            state["end_time"] = datetime.utcnow().isoformat()
            
            yield {
                "type": "pipeline_complete",
                "final_answer": state.get("final", ""),
                "confidence": state.get("confidence", 0),
                "citations": state.get("citations", []),
                "timestamp": state["end_time"]
            }
            
        except Exception as e:
            yield {
                "type": "error",
                "error": str(e),
                "phase": "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
            raise
    
    async def _stream_agent(
        self,
        agent_func: Callable,
        state: PipelineState,
        config: RunnableConfig,
        agent_name: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream updates from a specific agent."""
        yield {
            "type": "agent_thinking",
            "agent": agent_name,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _run_with_config(
        self,
        func: Callable,
        state: PipelineState,
        config: RunnableConfig
    ) -> PipelineState:
        """Run function with config (handles async/sync)."""
        if asyncio.iscoroutinefunction(func):
            return await func(state)
        else:
            return func(state)
    
    async def astream_events(
        self,
        question: str,
        context: Optional[str] = None,
        fast_mode: bool = False
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Alternative streaming using astream_events for LCEL chains.
        This provides more granular token-by-token streaming.
        """
        state: PipelineState = {
            "question": question,
            "context": context or "",
            "plan": "",
            "tool_sequence": [],
            "key_terms": [],
            "findings": [],
            "critique": {},
            "required_fixes": [],
            "draft": "",
            "final": "",
            "citations": [],
            "confidence": 0.0,
            "error": None,
            "start_time": datetime.utcnow().isoformat()
        }
        
        # Stream orchestrator
        if hasattr(self.orchestrator.chain, 'astream_events'):
            async for event in self.orchestrator.chain.astream_events(
                {"question": question, "context": context or ""},
                version="v2"
            ):
                if event["event"] == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        yield {
                            "type": "token",
                            "agent": "orchestrator",
                            "content": token
                        }
                elif event["event"] == "on_chain_end":
                    yield {
                        "type": "agent_complete",
                        "agent": "orchestrator"
                    }
        
        # Continue with other agents...
        # (Similar pattern for researcher, critic, synthesizer)


# Convenience function
async def stream_research(
    question: str,
    context: Optional[str] = None,
    fast_mode: bool = False,
    stream_handler: Optional[Callable] = None
) -> AsyncIterator[Dict[str, Any]]:
    """Stream research with real-time updates."""
    pipeline = StreamingResearchPipeline()
    async for event in pipeline.astream(question, context, fast_mode, stream_handler):
        yield event