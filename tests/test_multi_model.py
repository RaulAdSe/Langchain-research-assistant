#!/usr/bin/env python3
"""
Test script to compare single-model vs multi-model performance.
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_configuration(config_name, env_vars):
    """Test a specific model configuration."""
    print(f"\n🧪 Testing {config_name}:")
    print("=" * 50)
    
    # Set environment variables
    original_vars = {}
    for key, value in env_vars.items():
        original_vars[key] = os.environ.get(key)
        if value:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
    
    try:
        # Import after setting env vars
        from app.pipeline import research
        
        start_time = time.time()
        
        # Test question
        response = research(
            question="What are the benefits of multi-model architectures?",
            fast_mode=True  # Skip critic for faster testing
        )
        
        total_time = time.time() - start_time
        
        print(f"⏱️  Total time: {total_time:.1f}s")
        print(f"🎯 Confidence: {response.confidence:.1%}")
        print(f"📝 Answer length: {len(response.answer)} chars")
        
        return {
            "config": config_name,
            "total_time": total_time,
            "confidence": response.confidence,
            "answer_length": len(response.answer)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
        
    finally:
        # Restore original environment
        for key, value in original_vars.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]


def main():
    """Run multi-model performance comparison."""
    print("🚀 Multi-Model Architecture Performance Test")
    print("Testing single model vs optimized multi-model setup")
    
    results = []
    
    # Test 1: Single model (current)
    single_model = {
        "ORCHESTRATOR_MODEL": None,
        "RESEARCHER_MODEL": None, 
        "CRITIC_MODEL": None,
        "SYNTHESIZER_MODEL": None,
        "MODEL_NAME": "gpt-4.1-nano"
    }
    
    result = test_configuration("Single Model (gpt-4.1-nano)", single_model)
    if result:
        results.append(result)
    
    # Test 2: Optimized multi-model
    multi_model = {
        "ORCHESTRATOR_MODEL": "gpt-5-nano",
        "RESEARCHER_MODEL": "gpt-4.1-nano",
        "CRITIC_MODEL": "gpt-4.1-mini", 
        "SYNTHESIZER_MODEL": "gpt-4.1",
        "MODEL_NAME": "gpt-4.1-nano"  # fallback
    }
    
    result = test_configuration("Multi-Model (Optimized)", multi_model)
    if result:
        results.append(result)
    
    # Test 3: Premium multi-model
    premium_model = {
        "ORCHESTRATOR_MODEL": "gpt-5-nano",
        "RESEARCHER_MODEL": "gpt-4.1-nano",
        "CRITIC_MODEL": "gpt-4.1",
        "SYNTHESIZER_MODEL": "gpt-5-mini",
        "MODEL_NAME": "gpt-4.1-nano"
    }
    
    result = test_configuration("Premium Multi-Model", premium_model) 
    if result:
        results.append(result)
    
    # Results summary
    if results:
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("=" * 60)
        print(f"{'Configuration':<25} {'Time':<8} {'Quality':<10} {'Length':<8}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['config']:<25} {r['total_time']:.1f}s    {r['confidence']:.1%}     {r['answer_length']:>6}")
        
        # Find best performers
        if len(results) > 1:
            fastest = min(results, key=lambda x: x['total_time'])
            highest_quality = max(results, key=lambda x: x['confidence'])
            
            print(f"\n🏆 WINNERS:")
            print(f"⚡ Fastest: {fastest['config']} ({fastest['total_time']:.1f}s)")
            print(f"🎯 Highest Quality: {highest_quality['config']} ({highest_quality['confidence']:.1%})")


if __name__ == "__main__":
    main()