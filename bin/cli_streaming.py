#!/usr/bin/env python3
"""CLI with real-time streaming using Rich."""

import asyncio
from typing import Optional
import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.syntax import Syntax
from rich.markdown import Markdown
from datetime import datetime
import json

from app.streaming_pipeline import stream_research


app = typer.Typer(help="Research Assistant with Live Streaming")
console = Console()


class StreamingDisplay:
    """Manage Rich display for streaming updates."""
    
    def __init__(self):
        self.layout = Layout()
        self.phases = {
            "orchestrator": {"status": "⏳", "details": ""},
            "researcher": {"status": "⏳", "details": ""},
            "critic": {"status": "⏳", "details": ""},
            "synthesizer": {"status": "⏳", "details": ""}
        }
        self.current_phase = None
        self.tokens = {}
        self.tools_used = []
        self.final_answer = ""
        self.confidence = 0.0
        self.error = None
        self.start_time = datetime.now()
        
        # Setup layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=8),
            Layout(name="activity", size=10),
            Layout(name="output")
        )
    
    def update_header(self, question: str):
        """Update header with question."""
        elapsed = (datetime.now() - self.start_time).seconds
        header = Panel(
            f"[bold cyan]Research Question:[/bold cyan] {question}\n"
            f"[dim]Elapsed: {elapsed}s[/dim]",
            title="🔬 Research Assistant",
            border_style="cyan"
        )
        self.layout["header"].update(header)
    
    def update_progress(self):
        """Update progress panel showing phases."""
        table = Table(title="Pipeline Progress", show_header=True, header_style="bold magenta")
        table.add_column("Phase", style="cyan", width=15)
        table.add_column("Status", width=10)
        table.add_column("Details", width=50)
        
        for phase, info in self.phases.items():
            style = "green" if info["status"] == "✅" else "yellow" if info["status"] == "🔄" else "dim"
            table.add_row(
                phase.capitalize(),
                info["status"],
                info["details"],
                style=style
            )
        
        self.layout["progress"].update(Panel(table, border_style="blue"))
    
    def update_activity(self, activity: str):
        """Update current activity panel."""
        if self.tools_used:
            tools_text = "\n".join(f"  • {tool}" for tool in self.tools_used[-5:])
            content = f"[bold]Current:[/bold] {activity}\n\n[bold]Recent Tools:[/bold]\n{tools_text}"
        else:
            content = f"[bold]Current:[/bold] {activity}"
        
        self.layout["activity"].update(
            Panel(content, title="🎯 Activity", border_style="yellow")
        )
    
    def update_output(self):
        """Update output panel with tokens or final answer."""
        if self.final_answer:
            # Show final answer
            content = Markdown(self.final_answer)
            confidence_bar = "█" * int(self.confidence * 20) + "░" * (20 - int(self.confidence * 20))
            title = f"📊 Final Answer [green]({self.confidence:.0%} {confidence_bar})[/green]"
            self.layout["output"].update(Panel(content, title=title, border_style="green"))
        elif self.error:
            # Show error
            self.layout["output"].update(
                Panel(f"[red]Error: {self.error}[/red]", title="❌ Error", border_style="red")
            )
        else:
            # Show streaming tokens
            if self.tokens:
                agent = self.current_phase or "system"
                content = self.tokens.get(agent, "Waiting for response...")
                # Limit display to last 500 chars for readability
                if len(content) > 500:
                    content = "..." + content[-500:]
            else:
                content = "[dim]Waiting for response...[/dim]"
            
            self.layout["output"].update(
                Panel(content, title="💭 Agent Output", border_style="dim")
            )
    
    def handle_event(self, event: dict):
        """Process streaming event and update display."""
        event_type = event.get("type")
        
        if event_type == "phase_start":
            phase = event.get("phase")
            self.current_phase = phase
            self.phases[phase]["status"] = "🔄"
            self.phases[phase]["details"] = event.get("description", "Processing...")
            self.update_activity(f"Starting {phase}")
        
        elif event_type == "phase_complete":
            phase = event.get("phase")
            self.phases[phase]["status"] = "✅"
            
            # Add phase-specific details
            if phase == "orchestrator":
                tools = event.get("tools", [])
                self.phases[phase]["details"] = f"Plan ready, tools: {', '.join(tools[:3])}"
            elif phase == "researcher":
                findings = event.get("findings_count", 0)
                self.phases[phase]["details"] = f"Found {findings} findings"
            elif phase == "critic":
                score = event.get("quality_score", 0)
                self.phases[phase]["details"] = f"Quality score: {score:.2f}/1.0"
            elif phase == "synthesizer":
                conf = event.get("confidence", 0)
                self.phases[phase]["details"] = f"Confidence: {conf:.0%}"
        
        elif event_type == "phase_skip":
            phase = event.get("phase")
            self.phases[phase]["status"] = "⏭️"
            self.phases[phase]["details"] = f"Skipped ({event.get('reason', '')})"
        
        elif event_type == "tool_start":
            tool = event.get("tool", "Unknown")
            self.tools_used.append(f"{tool}: {event.get('input', '')[:50]}")
            self.update_activity(f"Using tool: {tool}")
        
        elif event_type == "token":
            agent = event.get("agent", self.current_phase or "system")
            content = event.get("content", "")
            if agent not in self.tokens:
                self.tokens[agent] = ""
            self.tokens[agent] += content
        
        elif event_type == "agent_thinking":
            agent = event.get("agent")
            self.update_activity(f"{agent} is thinking...")
        
        elif event_type == "pipeline_complete":
            self.final_answer = event.get("final_answer", "")
            self.confidence = event.get("confidence", 0.0)
            self.update_activity("Research complete!")
        
        elif event_type == "error":
            self.error = event.get("error", "Unknown error")
    
    def render(self):
        """Get the current layout for rendering."""
        self.update_progress()
        self.update_output()
        return self.layout


async def stream_with_display(
    question: str,
    context: Optional[str] = None,
    fast_mode: bool = False
):
    """Stream research with Rich display."""
    display = StreamingDisplay()
    display.update_header(question)
    
    with Live(display.render(), console=console, refresh_per_second=4) as live:
        try:
            async for event in stream_research(
                question=question,
                context=context,
                fast_mode=fast_mode
            ):
                display.handle_event(event)
                live.update(display.render())
                
                # Small delay for visual effect
                await asyncio.sleep(0.05)
            
            # Keep display for 2 seconds after completion
            await asyncio.sleep(2)
            
        except Exception as e:
            display.error = str(e)
            live.update(display.render())
            await asyncio.sleep(2)
            raise
    
    # Print final answer in a nice format after Live display
    if display.final_answer:
        console.print("\n")
        console.print(Panel(
            Markdown(display.final_answer),
            title=f"✨ Final Answer (Confidence: {display.confidence:.0%})",
            border_style="green",
            padding=(1, 2)
        ))


@app.command()
def research(
    question: str = typer.Argument(..., help="Research question"),
    context: Optional[str] = typer.Option(None, help="Additional context"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Fast mode (skip critic)"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON")
):
    """
    Research a question with live streaming updates.
    
    Example:
        python -m app.cli_streaming research "What is quantum computing?"
    """
    if json_output:
        # Collect all events and output as JSON
        events = []
        
        async def collect_events():
            async for event in stream_research(question, context, fast):
                events.append(event)
        
        asyncio.run(collect_events())
        console.print_json(data=events)
    else:
        # Run with Rich display
        asyncio.run(stream_with_display(question, context, fast))


@app.command()
def demo():
    """Run a demo with a sample question."""
    question = "What are the latest developments in AI streaming interfaces?"
    console.print(Panel(
        f"[bold cyan]Demo Question:[/bold cyan] {question}",
        title="🎭 Demo Mode",
        border_style="cyan"
    ))
    console.print()
    
    asyncio.run(stream_with_display(question, fast_mode=True))


if __name__ == "__main__":
    app()