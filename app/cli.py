"""CLI interface for the research assistant."""

import typer
from typing import Optional
from pathlib import Path
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from app.pipeline import research
from app.core.state import ResearchRequest
from app.rag.ingest import DocumentIngester, ingest_sample_data
from app.rag.store import get_vector_store

app = typer.Typer(help="Multi-agent research assistant CLI")
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="The research question to answer"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Additional context"),
    max_sources: int = typer.Option(5, "--sources", "-s", help="Maximum number of sources"),
    output_format: str = typer.Option("markdown", "--format", "-f", help="Output format: markdown or json"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress"),
    fast: bool = typer.Option(False, "--fast", help="Fast mode: skip critic review for faster responses")
):
    """Ask a research question and get a comprehensive answer."""
    
    # Show question
    console.print(Panel(f"[bold blue]Question:[/bold blue] {question}", expand=False))
    
    if context:
        console.print(f"[dim]Context: {context}[/dim]")
    
    # Run research with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Researching...", total=None)
        
        try:
            response = research(
                question=question,
                context=context,
                max_sources=max_sources,
                fast_mode=fast
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    # Display results
    if output_format == "json":
        console.print_json(response.model_dump_json(indent=2))
    else:
        # Display markdown answer
        console.print()
        console.print(Panel(Markdown(response.answer), title="📝 Answer", expand=False))
        
        # Display confidence
        confidence_color = "green" if response.confidence >= 0.7 else "yellow" if response.confidence >= 0.5 else "red"
        console.print(f"\n[{confidence_color}]Confidence: {response.confidence:.1%}[/{confidence_color}]")
        
        # Display key points if available
        if response.key_points:
            console.print("\n[bold]Key Points:[/bold]")
            for point in response.key_points:
                console.print(f"  • {point}")
        
        # Display caveats if available
        if response.caveats:
            console.print("\n[bold yellow]Caveats:[/bold yellow]")
            for caveat in response.caveats:
                console.print(f"  ⚠️  {caveat}")
        
        # Display trace URL if available
        if response.trace_url and verbose:
            console.print(f"\n[dim]Trace: {response.trace_url}[/dim]")
        
        # Display duration
        if response.duration_seconds:
            console.print(f"\n[dim]Completed in {response.duration_seconds:.1f} seconds[/dim]")


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to file or directory to ingest"),
    chunk_size: int = typer.Option(800, "--chunk", help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(120, "--overlap", help="Chunk overlap in characters"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Recursively ingest directories")
):
    """Ingest documents into the knowledge base."""
    
    if not path.exists():
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Ingesting:[/bold] {path}")
    
    ingester = DocumentIngester(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    try:
        if path.is_file():
            stats = ingester.ingest_file(path)
        elif path.is_dir():
            stats = ingester.ingest_directory(path, recursive=recursive)
        else:
            console.print(f"[red]Error: '{path}' is neither a file nor directory[/red]")
            raise typer.Exit(1)
        
        # Display statistics
        if stats["status"] == "success":
            console.print("\n[green]✅ Ingestion successful![/green]")
            console.print(f"  Documents processed: {stats['documents_processed']}")
            console.print(f"  Chunks created: {stats['chunks_created']}")
            console.print(f"  Chunks ingested: {stats['chunks_ingested']}")
            if stats.get('duplicates_removed', 0) > 0:
                console.print(f"  Duplicates removed: {stats['duplicates_removed']}")
        else:
            console.print(f"\n[red]❌ Ingestion failed: {stats.get('message', 'Unknown error')}[/red]")
            
    except Exception as e:
        console.print(f"[red]Error during ingestion: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def sample():
    """Ingest sample documents for testing."""
    console.print("[bold]Ingesting sample documents...[/bold]")
    
    try:
        stats = ingest_sample_data()
        
        if stats["status"] == "success":
            console.print("\n[green]✅ Sample data ingested successfully![/green]")
            console.print(f"  Documents: {stats['documents_processed']}")
            console.print(f"  Chunks: {stats['chunks_ingested']}")
        else:
            console.print(f"\n[red]Failed to ingest sample data[/red]")
            
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def stats():
    """Show knowledge base statistics."""
    store = get_vector_store()
    stats = store.get_collection_stats()
    
    console.print("[bold]Knowledge Base Statistics[/bold]")
    console.print(f"  Collection: {stats.get('collection_name', 'N/A')}")
    console.print(f"  Documents: {stats.get('document_count', 0)}")
    console.print(f"  Location: {stats.get('persist_directory', 'N/A')}")
    
    if "error" in stats:
        console.print(f"  [red]Error: {stats['error']}[/red]")


@app.command()
def reset():
    """Reset the knowledge base (delete all documents)."""
    if typer.confirm("Are you sure you want to delete all documents from the knowledge base?"):
        store = get_vector_store()
        store.reset()
        console.print("[green]✅ Knowledge base reset successfully[/green]")
    else:
        console.print("[yellow]Reset cancelled[/yellow]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()