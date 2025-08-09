#!/usr/bin/env python3
"""
Performance benchmark script for multi-agent research assistant.
Tests different OpenAI models for speed and quality comparison.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.pipeline import ResearchPipeline
from app.core.state import ResearchRequest


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model: str
    question: str
    mode: str  # "normal" or "fast"
    
    # Timing metrics
    planning_time: float
    research_time: float
    critic_time: Optional[float]
    synthesis_time: float
    total_time: float
    
    # Quality metrics
    confidence: float
    answer_length: int
    citation_count: int
    key_points_count: int
    caveats_count: int
    
    # Status
    success: bool
    error: Optional[str] = None


class ModelBenchmark:
    """Benchmark runner for multiple models."""
    
    # OpenAI models to test - your specified models
    MODELS_TO_TEST = [
        "gpt-5-mini",       # GPT-5 mini - needs max_completion_tokens
        "gpt-5-nano",       # GPT-5 nano - needs max_completion_tokens  
        "gpt-4.1",          # GPT-4.1
        "gpt-4.1-mini",     # GPT-4.1 mini
        "gpt-4.1-nano",     # GPT-4.1 nano
        "o3-mini",          # O3 mini (if available)
    ]
    
    # Test questions - mix of complexity levels
    TEST_QUESTIONS = [
        "What is machine learning?",
        "What are the benefits of multi-agent research systems?",
        "Explain the EU AI Act and its key provisions",
        "How does ChromaDB work for vector storage?",
        "What is the difference between RAG and fine-tuning?",
    ]
    
    def __init__(self, output_dir: str = "benchmark_results"):
        """Initialize benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
        # Store original MODEL_NAME to restore later
        self.original_model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    def run_single_test(
        self, 
        model: str, 
        question: str, 
        fast_mode: bool = False
    ) -> BenchmarkResult:
        """Run a single benchmark test."""
        print(f"Testing {model} {'(fast)' if fast_mode else '(normal)'}: {question[:50]}...")
        
        # Set model in environment
        os.environ["MODEL_NAME"] = model
        
        # Create pipeline
        pipeline = ResearchPipeline(fast_mode=fast_mode)
        request = ResearchRequest(question=question)
        
        # Track timing for each phase
        phase_times = {}
        start_time = time.time()
        
        try:
            # Monkey patch to capture phase timing
            original_plan = pipeline.orchestrator.plan
            original_research = pipeline.researcher.research  
            original_critique = pipeline.critic.critique
            original_synthesize = pipeline.synthesizer.synthesize
            
            def timed_plan(state):
                phase_start = time.time()
                result = original_plan(state)
                phase_times['planning'] = time.time() - phase_start
                return result
            
            def timed_research(state):
                phase_start = time.time()
                result = original_research(state)
                phase_times['research'] = time.time() - phase_start
                return result
            
            def timed_critique(state):
                phase_start = time.time()
                result = original_critique(state)
                phase_times['critic'] = time.time() - phase_start
                return result
            
            def timed_synthesize(state):
                phase_start = time.time()
                result = original_synthesize(state)
                phase_times['synthesis'] = time.time() - phase_start
                return result
            
            # Apply timing wrappers
            pipeline.orchestrator.plan = timed_plan
            pipeline.researcher.research = timed_research
            pipeline.critic.critique = timed_critique
            pipeline.synthesizer.synthesize = timed_synthesize
            
            # Run the pipeline
            response = pipeline.run(request)
            total_time = time.time() - start_time
            
            # Extract quality metrics
            answer_length = len(response.answer) if response.answer else 0
            citation_count = len(response.citations) if response.citations else 0
            key_points_count = len(response.key_points) if response.key_points else 0
            caveats_count = len(response.caveats) if response.caveats else 0
            
            return BenchmarkResult(
                model=model,
                question=question,
                mode="fast" if fast_mode else "normal",
                planning_time=phase_times.get('planning', 0),
                research_time=phase_times.get('research', 0),
                critic_time=phase_times.get('critic') if not fast_mode else None,
                synthesis_time=phase_times.get('synthesis', 0),
                total_time=total_time,
                confidence=response.confidence,
                answer_length=answer_length,
                citation_count=citation_count,
                key_points_count=key_points_count,
                caveats_count=caveats_count,
                success=True
            )
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"  ❌ Error: {str(e)}")
            
            return BenchmarkResult(
                model=model,
                question=question,
                mode="fast" if fast_mode else "normal",
                planning_time=0,
                research_time=0,
                critic_time=None,
                synthesis_time=0,
                total_time=total_time,
                confidence=0,
                answer_length=0,
                citation_count=0,
                key_points_count=0,
                caveats_count=0,
                success=False,
                error=str(e)
            )
    
    def run_full_benchmark(
        self, 
        models: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        test_both_modes: bool = True
    ):
        """Run complete benchmark across all models and questions."""
        models = models or self.MODELS_TO_TEST
        questions = questions or self.TEST_QUESTIONS
        
        total_tests = len(models) * len(questions) * (2 if test_both_modes else 1)
        current_test = 0
        
        print(f"🚀 Starting benchmark: {len(models)} models × {len(questions)} questions × {'2 modes' if test_both_modes else '1 mode'}")
        print(f"Total tests: {total_tests}")
        print("=" * 80)
        
        for model in models:
            print(f"\n📊 Testing model: {model}")
            
            for question in questions:
                # Test normal mode
                current_test += 1
                print(f"[{current_test}/{total_tests}]", end=" ")
                result = self.run_single_test(model, question, fast_mode=False)
                self.results.append(result)
                
                if result.success:
                    print(f"  ✅ Normal: {result.total_time:.1f}s (confidence: {result.confidence:.1%})")
                else:
                    print(f"  ❌ Normal: Failed")
                
                # Test fast mode if enabled
                if test_both_modes:
                    current_test += 1
                    print(f"[{current_test}/{total_tests}]", end=" ")
                    result = self.run_single_test(model, question, fast_mode=True)
                    self.results.append(result)
                    
                    if result.success:
                        print(f"  ⚡ Fast: {result.total_time:.1f}s (confidence: {result.confidence:.1%})")
                    else:
                        print(f"  ❌ Fast: Failed")
        
        # Restore original model
        os.environ["MODEL_NAME"] = self.original_model
    
    def generate_reports(self):
        """Generate comprehensive benchmark reports."""
        if not self.results:
            print("No results to report!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save raw results as JSON
        json_file = self.output_dir / f"benchmark_raw_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        print(f"📁 Raw results saved: {json_file}")
        
        # 2. Create DataFrame for analysis
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        # Filter successful results only for performance analysis
        success_df = df[df['success'] == True].copy()
        
        if success_df.empty:
            print("❌ No successful results to analyze!")
            return
        
        # 3. Performance summary by model
        summary = success_df.groupby(['model', 'mode']).agg({
            'total_time': ['mean', 'std', 'min', 'max'],
            'planning_time': 'mean',
            'research_time': 'mean',
            'synthesis_time': 'mean',
            'confidence': 'mean',
            'answer_length': 'mean',
            'citation_count': 'mean',
            'key_points_count': 'mean'
        }).round(2)
        
        # Save detailed CSV
        csv_file = self.output_dir / f"benchmark_detailed_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"📁 Detailed results saved: {csv_file}")
        
        # 4. Generate markdown report
        self._generate_markdown_report(success_df, timestamp)
        
        # 5. Print summary table
        self._print_summary_table(success_df)
    
    def _generate_markdown_report(self, df: pd.DataFrame, timestamp: str):
        """Generate a markdown report."""
        md_file = self.output_dir / f"benchmark_report_{timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write(f"# Model Performance Benchmark Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Performance by model table
            f.write("## Performance Summary by Model\n\n")
            
            summary = df.groupby(['model', 'mode']).agg({
                'total_time': 'mean',
                'planning_time': 'mean', 
                'research_time': 'mean',
                'synthesis_time': 'mean',
                'confidence': 'mean',
                'answer_length': 'mean',
                'citation_count': 'mean'
            }).round(2)
            
            f.write("| Model | Mode | Total Time (s) | Planning (s) | Research (s) | Synthesis (s) | Confidence | Avg Length | Citations |\n")
            f.write("|-------|------|----------------|--------------|--------------|---------------|------------|------------|-----------|\n")
            
            for (model, mode), row in summary.iterrows():
                f.write(f"| {model} | {mode} | {row['total_time']:.1f} | {row['planning_time']:.1f} | {row['research_time']:.1f} | {row['synthesis_time']:.1f} | {row['confidence']:.1%} | {int(row['answer_length'])} | {int(row['citation_count'])} |\n")
            
            # Speed ranking
            f.write("\n## Speed Ranking (Fastest to Slowest)\n\n")
            speed_ranking = df.groupby('model')['total_time'].mean().sort_values()
            
            for i, (model, avg_time) in enumerate(speed_ranking.items(), 1):
                f.write(f"{i}. **{model}**: {avg_time:.1f}s average\n")
            
            # Quality ranking
            f.write("\n## Quality Ranking (by Confidence Score)\n\n")
            quality_ranking = df.groupby('model')['confidence'].mean().sort_values(ascending=False)
            
            for i, (model, avg_conf) in enumerate(quality_ranking.items(), 1):
                f.write(f"{i}. **{model}**: {avg_conf:.1%} average confidence\n")
        
        print(f"📁 Markdown report saved: {md_file}")
    
    def _print_summary_table(self, df: pd.DataFrame):
        """Print a summary table to console."""
        print("\n" + "="*80)
        print("📊 BENCHMARK SUMMARY")
        print("="*80)
        
        # Average performance by model
        summary = df.groupby('model').agg({
            'total_time': 'mean',
            'confidence': 'mean',
            'answer_length': 'mean',
            'citation_count': 'mean'
        }).round(2)
        
        # Sort by speed (total_time)
        summary = summary.sort_values('total_time')
        
        print(f"\n{'Model':<15} {'Avg Time':<10} {'Confidence':<12} {'Avg Length':<12} {'Citations':<10}")
        print("-" * 70)
        
        for model, row in summary.iterrows():
            print(f"{model:<15} {row['total_time']:.1f}s{'':<5} {row['confidence']:.1%}{'':>6} {int(row['answer_length']):<12} {int(row['citation_count']):<10}")
        
        # Best performers
        fastest = summary.index[0]
        highest_confidence = summary.sort_values('confidence', ascending=False).index[0]
        
        print(f"\n🏆 WINNERS:")
        print(f"⚡ Fastest: {fastest} ({summary.loc[fastest, 'total_time']:.1f}s)")
        print(f"🎯 Highest Confidence: {highest_confidence} ({summary.loc[highest_confidence, 'confidence']:.1%})")


def main():
    """Main benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark OpenAI models for research assistant performance")
    parser.add_argument("--models", nargs="+", help="Models to test", default=None)
    parser.add_argument("--questions", nargs="+", help="Questions to test", default=None)
    parser.add_argument("--fast-only", action="store_true", help="Test fast mode only")
    parser.add_argument("--normal-only", action="store_true", help="Test normal mode only")
    parser.add_argument("--output", "-o", default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Determine which modes to test
    test_both = not (args.fast_only or args.normal_only)
    
    # Create benchmark runner
    benchmark = ModelBenchmark(output_dir=args.output)
    
    try:
        # Run benchmark
        benchmark.run_full_benchmark(
            models=args.models,
            questions=args.questions, 
            test_both_modes=test_both
        )
        
        # Generate reports
        benchmark.generate_reports()
        
    except KeyboardInterrupt:
        print("\n\n⏸️  Benchmark interrupted by user")
        if benchmark.results:
            print("Generating reports for completed tests...")
            benchmark.generate_reports()
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()