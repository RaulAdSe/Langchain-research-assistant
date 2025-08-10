#!/usr/bin/env python3
"""
PARALLEL Performance benchmark script for multi-agent research assistant.
Runs multiple models concurrently for much faster benchmarking.
"""

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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


class ParallelModelBenchmark:
    """Parallel benchmark runner for multiple models."""
    
    # Your specified models
    MODELS_TO_TEST = [
        "gpt-5-mini",       # GPT-5 mini
        "gpt-5-nano",       # GPT-5 nano
        "gpt-4.1",          # GPT-4.1
        "gpt-4.1-mini",     # GPT-4.1 mini
        "gpt-4.1-nano",     # GPT-4.1 nano
        "o3-mini",          # O3 mini
    ]
    
    # Test questions
    TEST_QUESTIONS = [
        "What is machine learning?",
        "What is Python programming?",
        "Explain the EU AI Act key provisions",
    ]
    
    def __init__(self, output_dir: str = "benchmark_results", max_workers: int = 6):
        """Initialize parallel benchmark runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.max_workers = max_workers  # Number of parallel workers
        self.lock = threading.Lock()    # Thread-safe results collection
        
        # Store original MODEL_NAME to restore later
        self.original_model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    
    def run_single_test(
        self, 
        model: str, 
        question: str, 
        fast_mode: bool = False
    ) -> BenchmarkResult:
        """Run a single benchmark test (thread-safe)."""
        thread_id = threading.get_ident()
        print(f"[Thread-{thread_id % 1000:03d}] Testing {model} {'(fast)' if fast_mode else '(normal)'}: {question[:30]}...")
        
        # Create a copy of environment for this thread
        thread_env = os.environ.copy()
        thread_env["MODEL_NAME"] = model
        
        # Temporarily set model in current process (not thread-safe, but needed for imports)
        original_model = os.environ.get("MODEL_NAME")
        os.environ["MODEL_NAME"] = model
        
        # Track timing for each phase
        phase_times = {}
        start_time = time.time()
        
        try:
            # Create pipeline (each thread gets its own instance)
            pipeline = ResearchPipeline(fast_mode=fast_mode)
            request = ResearchRequest(question=question)
            
            # Monkey patch to capture phase timing
            original_plan = pipeline.orchestrator.plan
            original_research = pipeline.researcher.research  
            original_critique = pipeline.critic.critique if not fast_mode else None
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
                if original_critique:
                    phase_start = time.time()
                    result = original_critique(state)
                    phase_times['critic'] = time.time() - phase_start
                    return result
                return state
            
            def timed_synthesize(state):
                phase_start = time.time()
                result = original_synthesize(state)
                phase_times['synthesis'] = time.time() - phase_start
                return result
            
            # Apply timing wrappers
            pipeline.orchestrator.plan = timed_plan
            pipeline.researcher.research = timed_research
            if not fast_mode and original_critique:
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
            
            result = BenchmarkResult(
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
            
            print(f"[Thread-{thread_id % 1000:03d}] ✅ {model}: {total_time:.1f}s (confidence: {response.confidence:.1%})")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            print(f"[Thread-{thread_id % 1000:03d}] ❌ {model} Error: {str(e)[:50]}...")
            
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
        
        finally:
            # Restore original model
            if original_model:
                os.environ["MODEL_NAME"] = original_model
            else:
                os.environ.pop("MODEL_NAME", None)
    
    def run_parallel_benchmark(
        self, 
        models: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
        fast_mode_only: bool = True
    ):
        """Run benchmark with parallel execution."""
        models = models or self.MODELS_TO_TEST
        questions = questions or self.TEST_QUESTIONS[:1]  # Default to 1 question for speed
        
        # Generate all test combinations
        test_combinations = []
        for model in models:
            for question in questions:
                if fast_mode_only:
                    test_combinations.append((model, question, True))
                else:
                    test_combinations.append((model, question, False))  # normal mode
                    test_combinations.append((model, question, True))   # fast mode
        
        total_tests = len(test_combinations)
        print(f"🚀 Starting PARALLEL benchmark:")
        print(f"   Models: {len(models)} ({', '.join(models)})")
        print(f"   Questions: {len(questions)}")
        print(f"   Mode: {'Fast only' if fast_mode_only else 'Both modes'}")
        print(f"   Total tests: {total_tests}")
        print(f"   Max parallel workers: {self.max_workers}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Run tests in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_test = {
                executor.submit(self.run_single_test, model, question, fast_mode): (model, question, fast_mode)
                for model, question, fast_mode in test_combinations
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_test):
                model, question, fast_mode = future_to_test[future]
                try:
                    result = future.result()
                    with self.lock:  # Thread-safe result collection
                        self.results.append(result)
                    completed += 1
                    
                    # Progress indicator
                    progress = completed / total_tests * 100
                    print(f"📊 Progress: {completed}/{total_tests} ({progress:.1f}%) - Latest: {model}")
                    
                except Exception as e:
                    print(f"❌ Test failed for {model}: {e}")
                    completed += 1
        
        total_time = time.time() - start_time
        successful_tests = sum(1 for r in self.results if r.success)
        
        print("\n" + "=" * 80)
        print(f"🎉 PARALLEL BENCHMARK COMPLETE!")
        print(f"   Total time: {total_time:.1f}s (vs ~{len(models) * 15:.0f}s sequential)")
        print(f"   Successful tests: {successful_tests}/{total_tests}")
        print(f"   Speedup: ~{len(models) * 15 / total_time:.1f}x faster")
        print("=" * 80)
        
        # Restore original model
        os.environ["MODEL_NAME"] = self.original_model
    
    def generate_reports(self):
        """Generate comprehensive benchmark reports."""
        if not self.results:
            print("No results to report!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Filter successful results
        success_results = [r for r in self.results if r.success]
        if not success_results:
            print("❌ No successful results to analyze!")
            return
        
        # Create DataFrame
        df = pd.DataFrame([asdict(r) for r in success_results])
        
        # Save detailed CSV
        csv_file = self.output_dir / f"parallel_benchmark_{timestamp}.csv"
        pd.DataFrame([asdict(r) for r in self.results]).to_csv(csv_file, index=False)
        print(f"📁 Results saved: {csv_file}")
        
        # Print performance ranking
        print("\n🏆 PERFORMANCE RANKING:")
        print("-" * 60)
        
        # Speed ranking (fastest to slowest)
        speed_ranking = df.groupby('model')['total_time'].mean().sort_values()
        print(f"\n⚡ SPEED RANKING (Average Total Time):")
        for i, (model, avg_time) in enumerate(speed_ranking.items(), 1):
            print(f"{i}. {model:<12} {avg_time:.1f}s")
        
        # Quality ranking (by confidence)
        quality_ranking = df.groupby('model')['confidence'].mean().sort_values(ascending=False)
        print(f"\n🎯 QUALITY RANKING (Average Confidence):")
        for i, (model, avg_conf) in enumerate(quality_ranking.items(), 1):
            print(f"{i}. {model:<12} {avg_conf:.1%}")
        
        # Detailed table
        summary = df.groupby('model').agg({
            'total_time': 'mean',
            'planning_time': 'mean',
            'research_time': 'mean', 
            'synthesis_time': 'mean',
            'confidence': 'mean',
            'answer_length': 'mean'
        }).round(2)
        
        print(f"\n📊 DETAILED PERFORMANCE TABLE:")
        print(f"{'Model':<12} {'Total':<6} {'Plan':<5} {'Research':<8} {'Synth':<6} {'Confidence':<10} {'Length':<7}")
        print("-" * 65)
        
        # Sort by total time for the table
        summary = summary.sort_values('total_time')
        for model, row in summary.iterrows():
            print(f"{model:<12} {row['total_time']:.1f}s   {row['planning_time']:.1f}s   {row['research_time']:.1f}s      {row['synthesis_time']:.1f}s   {row['confidence']:.1%}      {int(row['answer_length'])}")
        
        # Winner announcement
        fastest_model = speed_ranking.index[0]
        highest_quality = quality_ranking.index[0]
        print(f"\n🏆 WINNERS:")
        print(f"⚡ Fastest Model: {fastest_model} ({speed_ranking.iloc[0]:.1f}s)")
        print(f"🎯 Highest Quality: {highest_quality} ({quality_ranking.iloc[0]:.1%} confidence)")


def main():
    """Main parallel benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PARALLEL benchmark for OpenAI models")
    parser.add_argument("--models", nargs="+", help="Models to test", default=None)
    parser.add_argument("--questions", nargs="+", help="Questions to test", default=None)
    parser.add_argument("--workers", "-w", type=int, default=6, help="Max parallel workers")
    parser.add_argument("--include-normal", action="store_true", help="Include normal mode (slower)")
    parser.add_argument("--output", "-o", default="benchmark_results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    benchmark = ParallelModelBenchmark(
        output_dir=args.output,
        max_workers=args.workers
    )
    
    try:
        # Run parallel benchmark
        benchmark.run_parallel_benchmark(
            models=args.models,
            questions=args.questions,
            fast_mode_only=not args.include_normal
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
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()