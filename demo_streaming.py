#!/usr/bin/env python3
"""Demo streaming capabilities of the research assistant."""

import asyncio
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Add project root to path
sys.path.append('.')

from app.streaming_pipeline import stream_research

console = Console()


async def demo_streaming():
    """Demonstrate streaming with real-time updates."""
    
    # Demo questions
    questions = [
        "What are the benefits of streaming in LLM applications?",
        "How does LangChain handle streaming for multi-agent systems?",
        "What is Server-Sent Events (SSE) and how is it used for streaming?"
    ]
    
    console.print(Panel(
        "[bold cyan]🚀 Research Assistant Streaming Demo[/bold cyan]\n\n"
        "This demo shows real-time streaming updates during research.\n"
        "Watch as each agent phase provides live feedback!",
        title="Welcome",
        border_style="cyan"
    ))
    
    for i, question in enumerate(questions, 1):
        console.print(f"\n[bold]Demo {i}/3:[/bold] {question}\n")
        
        start_time = datetime.now()
        phases_completed = []
        current_phase = None
        tokens = ""
        
        # Stream the research
        async for event in stream_research(question=question, fast_mode=True):
            event_type = event.get("type")
            
            if event_type == "phase_start":
                phase = event.get("phase")
                current_phase = phase
                console.print(f"  [blue]▶ {phase.upper()}:[/blue] {event.get('description')}")
            
            elif event_type == "phase_complete":
                phase = event.get("phase")
                phases_completed.append(phase)
                
                # Show phase-specific metrics
                if phase == "orchestrator":
                    tools = event.get("tools", [])
                    console.print(f"    [green]✓[/green] Plan ready, tools: {', '.join(tools[:3])}")
                elif phase == "researcher":
                    findings = event.get("findings_count", 0)
                    console.print(f"    [green]✓[/green] Found {findings} findings")
                elif phase == "synthesizer":
                    conf = event.get("confidence", 0)
                    console.print(f"    [green]✓[/green] Confidence: {conf:.0%}")
            
            elif event_type == "phase_skip":
                phase = event.get("phase")
                console.print(f"  [yellow]⏭ {phase.upper()}:[/yellow] Skipped ({event.get('reason')})")
            
            elif event_type == "tool_start":
                tool = event.get("tool")
                console.print(f"    [dim]🔧 Using {tool}...[/dim]")
            
            elif event_type == "token":
                # Collect tokens but don't print each one
                tokens += event.get("content", "")
            
            elif event_type == "agent_thinking":
                agent = event.get("agent")
                console.print(f"    [dim]💭 {agent} thinking...[/dim]")
            
            elif event_type == "pipeline_complete":
                elapsed = (datetime.now() - start_time).total_seconds()
                answer = event.get("final_answer", "")
                confidence = event.get("confidence", 0)
                
                console.print(f"\n  [green bold]✅ Complete in {elapsed:.1f}s[/green bold]")
                console.print(f"  [cyan]Confidence: {confidence:.0%}[/cyan]")
                
                # Show a preview of the answer
                preview = answer[:200] + "..." if len(answer) > 200 else answer
                console.print(Panel(
                    preview,
                    title="Answer Preview",
                    border_style="green"
                ))
            
            elif event_type == "error":
                console.print(f"  [red]❌ Error: {event.get('error')}[/red]")
        
        # Brief pause between demos
        if i < len(questions):
            console.print("\n[dim]Next demo starting in 2 seconds...[/dim]")
            await asyncio.sleep(2)
    
    console.print(Panel(
        "[bold green]✨ Demo Complete![/bold green]\n\n"
        "The streaming pipeline provides:\n"
        "• Real-time phase updates\n"
        "• Tool execution visibility\n"
        "• Token streaming (when enabled)\n"
        "• Progress metrics\n"
        "• Error handling\n\n"
        "[cyan]Try it yourself:[/cyan]\n"
        "  python -m app.cli_streaming research 'Your question here'",
        title="Summary",
        border_style="green"
    ))


async def compare_with_without_streaming():
    """Compare user experience with and without streaming."""
    
    question = "What is quantum computing?"
    
    console.print(Panel(
        "[bold]Comparison: With vs Without Streaming[/bold]\n\n"
        "Same question, different user experience",
        title="UX Comparison",
        border_style="yellow"
    ))
    
    # Without streaming (simulated)
    console.print("\n[bold]1. Without Streaming:[/bold]")
    console.print("  ⏳ Processing...", end="")
    await asyncio.sleep(5)  # Simulate processing
    console.print(" Done!")
    console.print("  [dim]User waits 5+ seconds with no feedback[/dim]")
    
    # With streaming
    console.print("\n[bold]2. With Streaming:[/bold]")
    event_count = 0
    first_feedback_time = None
    start = datetime.now()
    
    async for event in stream_research(question=question, fast_mode=True):
        if first_feedback_time is None:
            first_feedback_time = (datetime.now() - start).total_seconds()
        
        event_count += 1
        event_type = event.get("type")
        
        # Show key events
        if event_type == "phase_start":
            phase = event.get("phase")
            console.print(f"  📍 {phase}: {event.get('description')}")
        elif event_type == "pipeline_complete":
            total_time = (datetime.now() - start).total_seconds()
            console.print(f"  ✅ Complete!")
            break
    
    console.print(f"\n[green]Results:[/green]")
    console.print(f"  • First feedback in: {first_feedback_time:.3f}s")
    console.print(f"  • Total events streamed: {event_count}")
    console.print(f"  • Total time: {total_time:.1f}s")
    console.print(f"  • [bold]User engaged throughout with live updates![/bold]")


async def main():
    """Run all demos."""
    
    demos = [
        ("Basic Streaming", demo_streaming),
        ("UX Comparison", compare_with_without_streaming)
    ]
    
    console.print(Panel(
        "[bold cyan]🎭 LangChain Streaming Demos[/bold cyan]\n\n"
        "Choose a demo to run:",
        title="Demo Selection",
        border_style="cyan"
    ))
    
    for i, (name, _) in enumerate(demos, 1):
        console.print(f"  {i}. {name}")
    console.print(f"  {len(demos)+1}. Run all demos")
    console.print(f"  0. Exit")
    
    choice = console.input("\n[cyan]Enter your choice:[/cyan] ")
    
    if choice == "0":
        return
    elif choice == str(len(demos)+1):
        for name, demo_func in demos:
            console.print(f"\n[bold]Running: {name}[/bold]")
            await demo_func()
            await asyncio.sleep(2)
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                name, demo_func = demos[idx]
                console.print(f"\n[bold]Running: {name}[/bold]")
                await demo_func()
            else:
                console.print("[red]Invalid choice[/red]")
        except (ValueError, IndexError):
            console.print("[red]Invalid choice[/red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user[/yellow]")