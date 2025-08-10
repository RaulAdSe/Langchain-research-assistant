#!/usr/bin/env python3
"""Test streaming implementation."""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, List

from app.streaming_pipeline import stream_research


class StreamingTester:
    """Test and benchmark streaming functionality."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.phase_times = {}
        self.token_count = 0
        self.tool_calls = []
    
    async def collect_event(self, event: Dict[str, Any]):
        """Collect and analyze streaming events."""
        self.events.append(event)
        event_type = event.get("type")
        
        # Track phase timing
        if event_type == "phase_start":
            phase = event.get("phase")
            self.phase_times[phase] = {"start": time.time()}
            print(f"📍 {phase.upper()}: {event.get('description')}")
        
        elif event_type == "phase_complete":
            phase = event.get("phase")
            if phase in self.phase_times:
                self.phase_times[phase]["end"] = time.time()
                duration = self.phase_times[phase]["end"] - self.phase_times[phase]["start"]
                self.phase_times[phase]["duration"] = duration
                print(f"✅ {phase.upper()} completed in {duration:.1f}s")
                
                # Print phase-specific metrics
                if phase == "researcher":
                    print(f"   - Findings: {event.get('findings_count', 0)}")
                elif phase == "synthesizer":
                    print(f"   - Confidence: {event.get('confidence', 0):.0%}")
        
        elif event_type == "phase_skip":
            phase = event.get("phase")
            print(f"⏭️  {phase.upper()} skipped: {event.get('reason')}")
        
        elif event_type == "tool_start":
            tool = event.get("tool")
            self.tool_calls.append(tool)
            print(f"   🔧 Tool: {tool}")
        
        elif event_type == "token":
            self.token_count += 1
            # Print dots to show token streaming
            if self.token_count % 10 == 0:
                print(".", end="", flush=True)
        
        elif event_type == "agent_thinking":
            agent = event.get("agent")
            print(f"   💭 {agent} thinking...")
        
        elif event_type == "pipeline_complete":
            print(f"\n🎉 Pipeline complete!")
            print(f"   - Answer length: {event.get('answer_length', 0)} chars")
            print(f"   - Final confidence: {event.get('confidence', 0):.0%}")
        
        elif event_type == "error":
            print(f"\n❌ Error: {event.get('error')}")
    
    async def test_streaming(self, question: str, fast_mode: bool = False):
        """Run a streaming test."""
        print(f"\n{'='*60}")
        print(f"🧪 Testing Streaming Pipeline")
        print(f"{'='*60}")
        print(f"Question: {question}")
        print(f"Fast mode: {fast_mode}")
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        try:
            # Stream with event collection
            async for event in stream_research(
                question=question,
                fast_mode=fast_mode,
                stream_handler=self.collect_event
            ):
                pass  # Events handled by stream_handler
            
            # Print newline after token dots
            if self.token_count > 0:
                print()
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            return False
        
        total_time = time.time() - start_time
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Streaming Test Summary")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Events received: {len(self.events)}")
        print(f"Tokens streamed: {self.token_count}")
        print(f"Tools used: {', '.join(self.tool_calls) if self.tool_calls else 'None'}")
        
        print(f"\n⏱️  Phase Timing:")
        for phase, times in self.phase_times.items():
            if "duration" in times:
                print(f"  - {phase}: {times['duration']:.1f}s")
        
        # Verify we got key events
        event_types = {e.get("type") for e in self.events}
        required_events = {"phase_start", "phase_complete", "pipeline_complete"}
        missing = required_events - event_types
        
        if missing:
            print(f"\n⚠️  Missing event types: {missing}")
            return False
        
        print(f"\n✅ Test passed!")
        return True


async def test_token_streaming():
    """Test that tokens are being streamed properly."""
    print("\n" + "="*60)
    print("🧪 Testing Token Streaming")
    print("="*60)
    
    token_buffer = []
    token_agents = set()
    
    async def capture_tokens(event):
        if event.get("type") == "token":
            token_buffer.append(event.get("content", ""))
            if event.get("agent"):
                token_agents.add(event.get("agent"))
    
    async for event in stream_research(
        question="Explain streaming in 2 sentences",
        fast_mode=True,
        stream_handler=capture_tokens
    ):
        if event.get("type") == "pipeline_complete":
            break
    
    print(f"Tokens captured: {len(token_buffer)}")
    print(f"Agents that streamed: {token_agents}")
    print(f"Sample tokens: {' '.join(token_buffer[:20])[:100]}...")
    
    return len(token_buffer) > 0


async def test_error_handling():
    """Test error handling in streaming."""
    print("\n" + "="*60)
    print("🧪 Testing Error Handling")
    print("="*60)
    
    error_caught = False
    
    try:
        async for event in stream_research(
            question="",  # Empty question should cause error
            fast_mode=True
        ):
            if event.get("type") == "error":
                error_caught = True
                print(f"✅ Error properly streamed: {event.get('error')}")
                break
    except Exception as e:
        error_caught = True
        print(f"✅ Exception properly raised: {e}")
    
    return error_caught


async def compare_streaming_vs_regular():
    """Compare streaming vs regular pipeline performance."""
    print("\n" + "="*60)
    print("🧪 Comparing Streaming vs Regular Pipeline")
    print("="*60)
    
    question = "What is the capital of France?"
    
    # Test streaming
    print("\n📡 Streaming Pipeline:")
    stream_start = time.time()
    event_count = 0
    first_event_time = None
    
    async for event in stream_research(question=question, fast_mode=True):
        if first_event_time is None:
            first_event_time = time.time() - stream_start
        event_count += 1
        if event.get("type") == "pipeline_complete":
            break
    
    stream_time = time.time() - stream_start
    
    print(f"  - Total time: {stream_time:.2f}s")
    print(f"  - Time to first event: {first_event_time:.3f}s")
    print(f"  - Total events: {event_count}")
    
    # Test regular (import if available)
    try:
        from app.pipeline import research
        print("\n⚡ Regular Pipeline:")
        regular_start = time.time()
        result = research(question=question, fast_mode=True)
        regular_time = time.time() - regular_start
        
        print(f"  - Total time: {regular_time:.2f}s")
        print(f"  - No intermediate feedback")
        
        print(f"\n📊 Comparison:")
        print(f"  - Streaming overhead: {stream_time - regular_time:.2f}s")
        print(f"  - User gets feedback {first_event_time:.3f}s after start")
        
    except ImportError:
        print("\n  Regular pipeline not available for comparison")
    
    return True


async def main():
    """Run all streaming tests."""
    print("\n" + "🚀 " * 20)
    print("STREAMING IMPLEMENTATION TEST SUITE")
    print("🚀 " * 20)
    
    # Test 1: Basic streaming
    tester = StreamingTester()
    test1 = await tester.test_streaming(
        "What are the benefits of streaming in LLM applications?",
        fast_mode=True
    )
    
    # Test 2: Token streaming
    test2 = await test_token_streaming()
    
    # Test 3: Error handling
    test3 = await test_error_handling()
    
    # Test 4: Performance comparison
    test4 = await compare_streaming_vs_regular()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST RESULTS")
    print("="*60)
    print(f"✅ Basic streaming: {'PASSED' if test1 else 'FAILED'}")
    print(f"✅ Token streaming: {'PASSED' if test2 else 'FAILED'}")
    print(f"✅ Error handling: {'PASSED' if test3 else 'FAILED'}")
    print(f"✅ Performance test: {'PASSED' if test4 else 'FAILED'}")
    
    all_passed = all([test1, test2, test3, test4])
    print(f"\n{'🎉 ALL TESTS PASSED!' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)