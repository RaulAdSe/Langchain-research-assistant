#!/usr/bin/env python3
"""Main CLI for research assistant with streaming."""

import asyncio
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from datetime import datetime

from app.streaming_pipeline import stream_research

app = typer.Typer(help="Research Assistant CLI with Real-time Updates")
console = Console()


async def run_research(
    question: str,
    context: Optional[str] = None,
    fast_mode: bool = False,
    verbose: bool = False
):
    """Run research with live terminal updates."""
    
    # Start with a header
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {question}",
        title="🔬 Research Assistant",
        border_style="cyan"
    ))
    
    # Track progress
    current_phase = None
    phases_complete = []
    tools_used = []
    findings_count = 0
    confidence = 0.0
    final_answer = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Starting research...", total=None)
        
        async for event in stream_research(
            question=question,
            context=context,
            fast_mode=fast_mode
        ):
            event_type = event.get("type")
            
            if event_type == "phase_start":
                phase = event.get("phase")
                current_phase = phase
                desc = event.get("description", "Processing...")
                progress.update(task, description=f"[yellow]{phase.upper()}:[/yellow] {desc}")
                
            elif event_type == "phase_complete":
                phase = event.get("phase")
                phases_complete.append(phase)
                
                # Show phase results
                if phase == "orchestrator":
                    tools = event.get("tools", [])
                    console.print(f"  [green]✓[/green] Plan ready, will use: {', '.join(tools)}")
                    if verbose:
                        state_output = event.get("state_output", {})
                        if state_output:
                            plan = state_output.get('plan', '')
                            if plan:
                                console.print(f"    [dim]Plan:[/dim] {plan}")
                            key_terms = state_output.get('key_terms', [])
                            if key_terms:
                                console.print(f"    [dim]Key terms:[/dim] {', '.join(key_terms)}")
                        
                elif phase == "researcher":
                    findings_count = event.get("findings_count", 0)
                    console.print(f"  [green]✓[/green] Found {findings_count} findings")
                    if verbose:
                        state_output = event.get("state_output", {})
                        if state_output:
                            draft_preview = state_output.get('draft_preview', '')
                            if draft_preview:
                                console.print(f"    [dim]Draft preview:[/dim] {draft_preview}")
                            findings = state_output.get('findings', [])
                            if findings:
                                console.print(f"    [dim]Key findings:[/dim]")
                                for finding in findings:
                                    console.print(f"      • {finding}")
                            citations_count = state_output.get('citations_count', 0)
                            if citations_count:
                                console.print(f"    [dim]Citations found:[/dim] {citations_count}")
                        
                elif phase == "critic":
                    score = event.get("quality_score", 0)
                    console.print(f"  [green]✓[/green] Quality score: {score:.1f}/10")
                    if verbose:
                        state_output = event.get("state_output", {})
                        if state_output:
                            issues_count = state_output.get('issues_found', 0)
                            critical_count = state_output.get('critical_issues', 0)
                            if issues_count:
                                console.print(f"    [dim]Issues found:[/dim] {issues_count} ({critical_count} critical)")
                            fixes = state_output.get('required_fixes', [])
                            if fixes:
                                console.print(f"    [dim]Required fixes:[/dim] {', '.join(fixes)}")
                            strengths = state_output.get('strengths', [])
                            if strengths:
                                console.print(f"    [dim]Strengths:[/dim] {', '.join(strengths)}")
                        
                elif phase == "synthesizer":
                    confidence = event.get("confidence", 0)
                    console.print(f"  [green]✓[/green] Final answer ready ({confidence:.0%} confidence)")
                    progress.update(task, description=f"[green]Finalizing answer...[/green] ({confidence:.0%} confidence)")
                    if verbose:
                        state_output = event.get("state_output", {})
                        if state_output:
                            final_preview = state_output.get('final_preview', '')
                            if final_preview:
                                console.print(f"    [dim]Answer preview:[/dim] {final_preview}")
                            sections = state_output.get('sections_count', 0)
                            citations = state_output.get('citations_count', 0)
                            console.print(f"    [dim]Structure:[/dim] {sections} sections, {citations} citations")
                    
            elif event_type == "phase_skip":
                phase = event.get("phase")
                if verbose:
                    console.print(f"  [yellow]⏭[/yellow]  {phase.upper()} skipped ({event.get('reason')})")
                    
            elif event_type == "tool_start":
                tool = event.get("tool")
                tools_used.append(tool)
                if verbose:
                    progress.update(task, description=f"[blue]Using {tool}...[/blue]")
                    tool_input = event.get("input", "")
                    if tool_input:
                        console.print(f"    [dim]Tool input:[/dim] {tool_input}")
                    
            elif event_type == "agent_thinking":
                agent = event.get("agent")
                if verbose:
                    console.print(f"  [dim]🧠 {agent} processing...[/dim]")
                progress.update(task, description=f"[dim]{agent} thinking...[/dim]")
                
            elif event_type == "tool_end":
                if verbose:
                    output_preview = event.get("output_preview", "")
                    if output_preview:
                        console.print(f"    [dim]Tool output:[/dim] {output_preview}")
                        
            elif event_type == "pipeline_complete":
                final_answer = event.get("final_answer", "")
                confidence = event.get("confidence", 0)
                progress.update(task, description="[bold green]Research complete![/bold green]")
                
            elif event_type == "error":
                error_msg = event.get("error", "Unknown error")
                console.print(f"[red]❌ Error: {error_msg}[/red]")
                progress.stop()
                return
    
    # Display final answer
    if final_answer:
        console.print("\n")
        console.print(Panel(
            Markdown(final_answer),
            title=f"✨ Answer [green]({confidence:.0%} confidence)[/green]",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Show summary stats if verbose
        if verbose:
            console.print(f"\n[dim]📊 Stats: {len(phases_complete)} phases, {findings_count} findings, {len(tools_used)} tool calls[/dim]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your research question"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Fast mode (skip quality review)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress")
):
    """
    Ask a research question and get real-time updates.
    
    Examples:
        research ask "What is quantum computing?"
        research ask "Compare GPT-4 and Claude" --fast
        research ask "Latest AI developments" --verbose
    """
    asyncio.run(run_research(question, context, fast, verbose))


@app.command()
def chat():
    """
    Interactive chat mode with streaming responses.
    """
    console.print(Panel(
        "[bold cyan]Research Assistant - Interactive Mode[/bold cyan]\n"
        "Type your questions, or 'exit' to quit.\n"
        "Use /fast for quick answers, /verbose for details.",
        border_style="cyan"
    ))
    
    while True:
        try:
            # Get user input
            question = console.input("\n[bold cyan]You:[/bold cyan] ")
            
            # Check for commands
            if question.lower() in ['exit', 'quit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not question.strip():
                continue
            
            # Parse flags
            fast_mode = '/fast' in question
            verbose = '/verbose' in question
            question = question.replace('/fast', '').replace('/verbose', '').strip()
            
            # Run research
            console.print()
            asyncio.run(run_research(question, fast_mode=fast_mode, verbose=verbose))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def batch(
    file: str = typer.Argument(..., help="File with questions (one per line)"),
    fast: bool = typer.Option(False, "--fast", "-f", help="Fast mode for all questions"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save answers to file")
):
    """
    Process multiple questions from a file.
    
    Example:
        research batch questions.txt --fast --output answers.md
    """
    try:
        with open(file, 'r') as f:
            questions = [q.strip() for q in f.readlines() if q.strip()]
    except FileNotFoundError:
        console.print(f"[red]File not found: {file}[/red]")
        return
    
    console.print(f"[cyan]Processing {len(questions)} questions...[/cyan]\n")
    
    results = []
    for i, question in enumerate(questions, 1):
        console.print(f"[bold]Question {i}/{len(questions)}:[/bold] {question}")
        
        # Collect answer
        answer_text = []
        
        async def collect_answer():
            async for event in stream_research(question=question, fast_mode=fast):
                if event.get("type") == "pipeline_complete":
                    answer_text.append(event.get("final_answer", ""))
        
        asyncio.run(collect_answer())
        
        if answer_text:
            results.append(f"## Q: {question}\n\n{answer_text[0]}\n\n---\n")
            console.print("[green]✓ Complete[/green]\n")
        else:
            results.append(f"## Q: {question}\n\nError: No answer generated\n\n---\n")
            console.print("[red]✗ Failed[/red]\n")
    
    # Save results if output specified
    if output and results:
        with open(output, 'w') as f:
            f.write('\n'.join(results))
        console.print(f"[green]Results saved to {output}[/green]")


@app.command()
def watch(
    question: str = typer.Argument(..., help="Research question"),
    interval: int = typer.Option(60, "--interval", "-i", help="Refresh interval in seconds")
):
    """
    Continuously monitor a topic with periodic updates.
    
    Example:
        research watch "Latest developments in AI" --interval 300
    """
    console.print(Panel(
        f"[bold cyan]Monitoring:[/bold cyan] {question}\n"
        f"[dim]Refreshing every {interval} seconds. Press Ctrl+C to stop.[/dim]",
        border_style="cyan"
    ))
    
    iteration = 0
    while True:
        try:
            iteration += 1
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            console.print(f"\n[bold]Update #{iteration}[/bold] at {timestamp}")
            
            # Add timestamp context to get latest info
            context = f"Please focus on the most recent information as of {datetime.now().strftime('%Y-%m-%d')}"
            
            asyncio.run(run_research(
                question=question,
                context=context,
                fast_mode=True,
                verbose=False
            ))
            
            # Wait for next iteration
            console.print(f"\n[dim]Next update in {interval} seconds...[/dim]")
            asyncio.sleep(interval)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped.[/yellow]")
            break


if __name__ == "__main__":
    # If no command provided, show help
    if len(sys.argv) == 1:
        app(["--help"])
    else:
        app()