#!/usr/bin/env python3
"""CLI for testing the iterative research pipeline."""

import asyncio
import sys
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.markdown import Markdown
from datetime import datetime

sys.path.append('.')
from app.iterative_pipeline import IterativeResearchPipeline

app = typer.Typer(help="Iterative Research Assistant with Quality-Driven Refinement")
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your research question"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    threshold: float = typer.Option(0.7, "--threshold", "-t", help="Quality threshold (0.0-1.0)"),
    max_iter: int = typer.Option(3, "--max-iter", "-m", help="Maximum iterations"),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v/-q", help="Show detailed progress")
):
    """
    Ask a research question with iterative quality refinement.
    
    The system will:
    1. Generate initial research
    2. Assess quality with critic
    3. Iterate if quality < threshold or critical issues found
    4. Maximum iterations for refinement
    
    Examples:
        iterative_research ask "What is quantum computing?"
        iterative_research ask "Compare Python vs JavaScript" --threshold 0.7 --max-iter 2
    """
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {question}\n"
        f"[dim]Quality threshold: {threshold}, Max iterations: {max_iter}[/dim]",
        title="🔬 Iterative Research Assistant",
        border_style="cyan"
    ))
    
    asyncio.run(run_research(question, context, threshold, max_iter, verbose))


async def run_research(
    question: str, 
    context: Optional[str], 
    threshold: float, 
    max_iter: int, 
    verbose: bool
):
    """Run iterative research with streaming updates and display results."""
    
    # Create pipeline
    pipeline = IterativeResearchPipeline(quality_threshold=threshold, max_iterations=max_iter)
    
    # Track progress
    current_iteration = 0
    current_phase = None
    quality_scores = []
    final_state = None
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Starting iterative research...", total=None)
        
        async for event in pipeline.astream(question, context or "", verbose):
            event_type = event.get("type")
            
            if event_type == "pipeline_start":
                progress.update(task, description="[cyan]Pipeline started[/cyan]")
                if verbose:
                    console.print(f"  [dim]Quality threshold: {threshold}, Max iterations: {max_iter}[/dim]")
                    
            elif event_type == "iteration_start":
                current_iteration = event.get("iteration", 0)
                progress.update(task, description=f"[yellow]Iteration {current_iteration}/{max_iter}[/yellow]")
                if verbose:
                    console.print(f"\n[bold]--- Iteration {current_iteration} ---[/bold]")
                    
            elif event_type == "phase_start":
                current_phase = event.get("phase")
                desc = event.get("description", "Processing...")
                progress.update(task, description=f"[blue]{current_phase.upper()}:[/blue] {desc}")
                if verbose:
                    console.print(f"  [cyan]▶[/cyan] {current_phase.upper()}: {desc}")
                    
            elif event_type == "orchestrator_decision":
                reasoning = event.get("reasoning", "")
                next_action = event.get("next_action", "research")
                if verbose:
                    console.print(f"    [dim]Decision:[/dim] {next_action} - {reasoning[:100]}...")
                    
            elif event_type == "research_complete":
                findings_count = event.get("findings_count", 0)
                console.print(f"  [green]✓[/green] Research: {findings_count} findings gathered")
                
            elif event_type == "synthesis_instructions":
                instructions = event.get("instructions", "")
                if verbose:
                    console.print(f"    [dim]Synthesis focus:[/dim] {instructions[:100]}...")
                    
            elif event_type == "quality_assessment":
                quality_score = event.get("quality_score", 0.0)
                subjective = event.get("subjective_quality", quality_score)
                objective = event.get("objective_quality", 0.0)
                iteration = event.get("iteration", 0)
                
                quality_scores.append(quality_score)
                console.print(f"  [green]✓[/green] Quality: {quality_score:.2f} (subj: {subjective:.2f}, obj: {objective:.2f})")
                
                if verbose:
                    console.print(f"    [dim]Quality breakdown:[/dim] Subjective {subjective:.2f} + Objective {objective:.2f}")
                    
            elif event_type == "threshold_reached":
                quality = event.get("quality_score", 0.0)
                threshold_val = event.get("threshold", 0.0)
                console.print(f"  [🎯] [green]Threshold achieved![/green] {quality:.2f} >= {threshold_val:.2f}")
                progress.update(task, description="[green]Quality threshold achieved![/green]")
                
            elif event_type == "quality_below_threshold":
                quality = event.get("quality_score", 0.0)
                threshold_val = event.get("threshold", 0.0)
                if verbose:
                    console.print(f"  [yellow]⚠[/yellow] Quality {quality:.2f} < threshold {threshold_val:.2f}")
                    
            elif event_type == "feedback_prepared":
                feedback_preview = event.get("feedback_preview", "")
                next_iter = event.get("next_iteration", 0)
                if verbose:
                    console.print(f"  [blue]🔄[/blue] Feedback prepared for iteration {next_iter}")
                    console.print(f"    [dim]Preview:[/dim] {feedback_preview}")
                    
            elif event_type == "content_stagnation":
                reason = event.get("reason", "")
                recommendation = event.get("recommendation", "")
                console.print(f"  [orange1]⚠[/orange1] Content stagnation detected: {reason}")
                if verbose and recommendation:
                    console.print(f"    [dim]Recommendation:[/dim] {recommendation}")
                
            elif event_type == "max_iterations_reached":
                final_quality = event.get("final_quality", 0.0)
                max_iters = event.get("max_iterations", 0)
                console.print(f"  [yellow]⚠[/yellow] Max iterations ({max_iters}) reached. Final quality: {final_quality:.2f}")
                
            elif event_type == "pipeline_complete":
                final_state = event.get("state", {})
                total_iters = event.get("total_iterations", 0)
                final_quality = event.get("final_quality", 0.0)
                confidence = event.get("final_confidence", 0.0)
                
                progress.update(task, description="[bold green]Pipeline complete![/bold green]")
                console.print(f"\n[bold green]✓ Research complete![/bold green]")
                console.print(f"  Total iterations: {total_iters}")
                console.print(f"  Final quality: {final_quality:.2f}")
                console.print(f"  Confidence: {confidence:.0%}")
                
            elif event_type == "error":
                error = event.get("error", "Unknown error")
                phase = event.get("phase", "unknown")
                console.print(f"[red]❌ Error in {phase}:[/red] {error}")
                return
    
    # Display final results
    if final_state:
        display_results(final_state, quality_scores)


def display_results(state: dict, quality_scores: list):
    """Display research results with quality progression."""
    # Quality progression visualization
    if len(quality_scores) > 1:
        console.print(f"\n[bold]Quality progression:[/bold]")
        for i, score in enumerate(quality_scores, 1):
            trend = ""
            if i > 1:
                diff = score - quality_scores[i-2]
                if diff > 0:
                    trend = f" [green](+{diff:.2f})[/green]"
                elif diff < 0:
                    trend = f" [red]({diff:.2f})[/red]"
                else:
                    trend = " [dim](=)[/dim]"
            console.print(f"  Iteration {i}: {score:.2f}{trend}")
    
    # Iteration history details
    iteration_history = state.get("iteration_history", [])
    if iteration_history and len(iteration_history) > 1:
        console.print(f"\n[bold]Research progression:[/bold]")
        for entry in iteration_history:
            iteration = entry.get("iteration", 0)
            findings_count = entry.get("findings_count", 0)
            improvements = entry.get("improvements", [])
            console.print(f"  Iteration {iteration}: {findings_count} findings, {len(improvements)} improvements")
    
    # Final answer
    final_answer = state.get("final", "")
    if final_answer:
        console.print("\n[bold]Final Answer:[/bold]")
        # Use markdown for better formatting
        console.print(Panel(
            Markdown(final_answer),
            border_style="green",
            title=f"✨ Research Result ({state.get('confidence', 0.0):.0%} confidence)"
        ))
    
    # Statistics
    total_iterations = state.get("total_iterations", 1)
    citations_count = len(state.get("citations", []))
    findings_count = len(state.get("findings", []))
    
    console.print(f"\n[dim]📊 Final stats: {total_iterations} iterations, {findings_count} findings, {citations_count} sources[/dim]")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        app(["--help"])
    else:
        app()