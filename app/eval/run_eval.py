"""Evaluation runner for the research assistant."""

import json
import csv
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
import re
from dataclasses import dataclass

from app.pipeline import default_pipeline
from app.core.state import ResearchRequest
import requests


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    answer: str
    confidence: float
    citations: List[Dict]
    duration_seconds: float
    
    # Evaluation metrics
    faithfulness_score: float
    answerability_score: float
    citation_coverage_score: float
    completeness_score: float
    coherence_score: float
    currency_score: float
    
    # Metadata
    difficulty: str
    category: str
    error: Optional[str] = None


class ResearchAssistantEvaluator:
    """Evaluates the research assistant performance."""
    
    def __init__(self, dataset_path: str = "app/eval/datasets.jsonl"):
        """Initialize evaluator with dataset."""
        self.dataset_path = Path(dataset_path)
        self.results: List[EvaluationResult] = []
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load evaluation dataset from JSONL file."""
        dataset = []
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.dataset_path}")
        
        with open(self.dataset_path, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        
        return dataset
    
    def evaluate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """
        Evaluate faithfulness by checking if claims are supported by contexts.
        
        Simplified implementation using keyword overlap.
        In production, would use more sophisticated NLP techniques.
        """
        if not contexts or not answer:
            return 0.0
        
        # Split answer into sentences (simple approach)
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        supported_count = 0
        
        for sentence in sentences:
            # Check if sentence has support in any context
            sentence_words = set(sentence.lower().split())
            
            for context in contexts:
                context_words = set(context.lower().split())
                # If significant overlap, consider supported
                overlap = len(sentence_words.intersection(context_words))
                if overlap >= min(3, len(sentence_words) * 0.3):
                    supported_count += 1
                    break
        
        return supported_count / len(sentences) if sentences else 0.0
    
    def evaluate_citation_coverage(self, answer: str, citations: List[Dict]) -> float:
        """Evaluate citation coverage and quality."""
        if not citations:
            return 0.0
        
        # Count citation markers in text
        citation_markers = re.findall(r'\[#\d+\]', answer)
        marker_coverage = len(set(citation_markers)) / len(citations) if citations else 0
        
        # Check citation link validity (simplified)
        valid_citations = 0
        for citation in citations:
            url = citation.get('url', '')
            if url and (url.startswith('http') or Path(url).exists()):
                valid_citations += 1
        
        link_validity = valid_citations / len(citations) if citations else 0
        
        # Combined score
        return (marker_coverage * 0.6 + link_validity * 0.4)
    
    def evaluate_answerability(self, question: str, answer: str, expected: Dict) -> float:
        """Evaluate if the system handled answerability correctly."""
        expected_type = expected.get('answer_type', 'normal')
        
        if expected_type == 'refusal':
            # Should refuse to answer
            refusal_indicators = ['cannot', 'unable', 'impossible', 'unclear', 'insufficient']
            has_refusal = any(indicator in answer.lower() for indicator in refusal_indicators)
            return 1.0 if has_refusal else 0.0
        
        elif expected_type == 'error':
            # Should handle error gracefully
            return 1.0 if len(answer.strip()) == 0 or 'error' in answer.lower() else 0.0
        
        else:
            # Should provide substantive answer
            return 1.0 if len(answer.strip()) > 50 else 0.0
    
    def evaluate_completeness(self, question: str, answer: str, expected: Dict) -> float:
        """Evaluate answer completeness based on expected content."""
        expected_content = expected.get('answer_contains', [])
        if not expected_content:
            return 0.8  # Default score when no specific expectations
        
        found_count = 0
        answer_lower = answer.lower()
        
        for expected_term in expected_content:
            if expected_term.lower() in answer_lower:
                found_count += 1
        
        return found_count / len(expected_content)
    
    def evaluate_coherence(self, answer: str) -> float:
        """Evaluate answer coherence and structure."""
        if not answer.strip():
            return 0.0
        
        score = 0.0
        
        # Check for structure indicators
        if '**' in answer or '#' in answer:  # Headers/bold
            score += 0.3
        
        if any(marker in answer for marker in ['•', '-', '1.', '2.']):  # Lists
            score += 0.2
        
        # Check length appropriateness
        word_count = len(answer.split())
        if 50 <= word_count <= 500:  # Reasonable length
            score += 0.3
        elif word_count < 50:
            score += 0.1
        
        # Check for citations
        if '[#' in answer:
            score += 0.2
        
        return min(score, 1.0)
    
    def evaluate_currency(self, question: str, answer: str, citations: List[Dict], metadata: Dict) -> float:
        """Evaluate currency of information."""
        if not metadata.get('requires_recent', False):
            return 1.0  # Not applicable
        
        current_year = datetime.now().year
        recent_citations = 0
        
        for citation in citations:
            date_str = citation.get('date', '')
            if date_str:
                try:
                    # Simple year extraction
                    if str(current_year) in date_str or str(current_year - 1) in date_str:
                        recent_citations += 1
                except:
                    continue
        
        if citations:
            return recent_citations / len(citations)
        else:
            # Check if answer mentions current year
            return 0.5 if str(current_year) in answer else 0.2
    
    async def evaluate_single_question(self, item: Dict[str, Any], question_id: str) -> EvaluationResult:
        """Evaluate a single question."""
        question_data = item['input']
        expected = item['expected']
        metadata = item.get('metadata', {})
        
        question = question_data['question']
        
        try:
            # Run research pipeline
            request = ResearchRequest(
                question=question,
                max_sources=5
            )
            
            start_time = datetime.now()
            response = default_pipeline.run(request)
            duration = (datetime.now() - start_time).total_seconds()
            
            # Extract contexts for faithfulness evaluation
            # This is simplified - in practice would extract from tool results
            contexts = []
            for citation in response.citations:
                if 'snippet' in citation:
                    contexts.append(citation['snippet'])
            
            # Calculate evaluation metrics
            faithfulness = self.evaluate_faithfulness(response.answer, contexts)
            answerability = self.evaluate_answerability(question, response.answer, expected)
            citation_coverage = self.evaluate_citation_coverage(response.answer, response.citations)
            completeness = self.evaluate_completeness(question, response.answer, expected)
            coherence = self.evaluate_coherence(response.answer)
            currency = self.evaluate_currency(question, response.answer, response.citations, metadata)
            
            result = EvaluationResult(
                question_id=question_id,
                question=question,
                answer=response.answer,
                confidence=response.confidence,
                citations=response.citations,
                duration_seconds=duration,
                faithfulness_score=faithfulness,
                answerability_score=answerability,
                citation_coverage_score=citation_coverage,
                completeness_score=completeness,
                coherence_score=coherence,
                currency_score=currency,
                difficulty=metadata.get('difficulty', 'unknown'),
                category=metadata.get('category', 'unknown')
            )
            
            return result
            
        except Exception as e:
            return EvaluationResult(
                question_id=question_id,
                question=question,
                answer="",
                confidence=0.0,
                citations=[],
                duration_seconds=0.0,
                faithfulness_score=0.0,
                answerability_score=0.0,
                citation_coverage_score=0.0,
                completeness_score=0.0,
                coherence_score=0.0,
                currency_score=0.0,
                difficulty=metadata.get('difficulty', 'unknown'),
                category=metadata.get('category', 'unknown'),
                error=str(e)
            )
    
    async def run_evaluation(self, max_questions: Optional[int] = None) -> List[EvaluationResult]:
        """Run evaluation on the full dataset."""
        dataset = self.load_dataset()
        
        if max_questions:
            dataset = dataset[:max_questions]
        
        print(f"Running evaluation on {len(dataset)} questions...")
        
        results = []
        for i, item in enumerate(dataset):
            print(f"Evaluating question {i+1}/{len(dataset)}: {item['input']['question'][:60]}...")
            
            result = await self.evaluate_single_question(item, f"q_{i+1}")
            results.append(result)
            
            # Print progress
            if result.error:
                print(f"  ❌ Error: {result.error}")
            else:
                print(f"  ✅ Completed (confidence: {result.confidence:.2f})")
        
        self.results = results
        return results
    
    def calculate_aggregate_metrics(self) -> Dict[str, float]:
        """Calculate aggregate metrics across all results."""
        if not self.results:
            return {}
        
        valid_results = [r for r in self.results if r.error is None]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        metrics = {
            'total_questions': len(self.results),
            'successful_questions': len(valid_results),
            'error_rate': (len(self.results) - len(valid_results)) / len(self.results),
            'avg_faithfulness': sum(r.faithfulness_score for r in valid_results) / len(valid_results),
            'avg_answerability': sum(r.answerability_score for r in valid_results) / len(valid_results),
            'avg_citation_coverage': sum(r.citation_coverage_score for r in valid_results) / len(valid_results),
            'avg_completeness': sum(r.completeness_score for r in valid_results) / len(valid_results),
            'avg_coherence': sum(r.coherence_score for r in valid_results) / len(valid_results),
            'avg_currency': sum(r.currency_score for r in valid_results) / len(valid_results),
            'avg_confidence': sum(r.confidence for r in valid_results) / len(valid_results),
            'avg_duration': sum(r.duration_seconds for r in valid_results) / len(valid_results)
        }
        
        # Calculate overall score
        score_components = [
            metrics['avg_faithfulness'],
            metrics['avg_answerability'],
            metrics['avg_citation_coverage'],
            metrics['avg_completeness'],
            metrics['avg_coherence'],
            metrics['avg_currency']
        ]
        metrics['overall_score'] = sum(score_components) / len(score_components)
        
        return metrics
    
    def save_results(self, output_dir: str = "eval_results"):
        """Save evaluation results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        detailed_results = []
        for result in self.results:
            detailed_results.append({
                'question_id': result.question_id,
                'question': result.question,
                'answer': result.answer,
                'confidence': result.confidence,
                'citations_count': len(result.citations),
                'duration_seconds': result.duration_seconds,
                'faithfulness_score': result.faithfulness_score,
                'answerability_score': result.answerability_score,
                'citation_coverage_score': result.citation_coverage_score,
                'completeness_score': result.completeness_score,
                'coherence_score': result.coherence_score,
                'currency_score': result.currency_score,
                'difficulty': result.difficulty,
                'category': result.category,
                'error': result.error
            })
        
        with open(output_path / f"detailed_results_{timestamp}.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)
        
        # Save summary metrics
        metrics = self.calculate_aggregate_metrics()
        with open(output_path / f"summary_metrics_{timestamp}.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save CSV for analysis
        with open(output_path / f"results_{timestamp}.csv", 'w', newline='') as f:
            if self.results:
                fieldnames = ['question_id', 'difficulty', 'category', 'confidence', 
                             'faithfulness_score', 'answerability_score', 'citation_coverage_score',
                             'completeness_score', 'coherence_score', 'currency_score',
                             'duration_seconds', 'error']
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    writer.writerow({
                        'question_id': result.question_id,
                        'difficulty': result.difficulty,
                        'category': result.category,
                        'confidence': result.confidence,
                        'faithfulness_score': result.faithfulness_score,
                        'answerability_score': result.answerability_score,
                        'citation_coverage_score': result.citation_coverage_score,
                        'completeness_score': result.completeness_score,
                        'coherence_score': result.coherence_score,
                        'currency_score': result.currency_score,
                        'duration_seconds': result.duration_seconds,
                        'error': result.error or ''
                    })
        
        print(f"Results saved to {output_path}/")
        return output_path


async def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate research assistant")
    parser.add_argument("--dataset", default="app/eval/datasets.jsonl", help="Dataset file path")
    parser.add_argument("--max-questions", type=int, help="Maximum number of questions to evaluate")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory")
    
    args = parser.parse_args()
    
    evaluator = ResearchAssistantEvaluator(args.dataset)
    
    # Run evaluation
    await evaluator.run_evaluation(args.max_questions)
    
    # Calculate and display metrics
    metrics = evaluator.calculate_aggregate_metrics()
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25}: {value:.3f}")
        else:
            print(f"{key:25}: {value}")
    
    # Save results
    output_path = evaluator.save_results(args.output_dir)
    
    print(f"\nDetailed results saved to: {output_path}")
    
    # Return overall score for CI/CD
    overall_score = metrics.get('overall_score', 0)
    if overall_score < 0.7:
        print(f"\n⚠️  WARNING: Overall score ({overall_score:.3f}) is below threshold (0.7)")
        return 1
    else:
        print(f"\n✅ PASS: Overall score ({overall_score:.3f}) meets threshold")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())