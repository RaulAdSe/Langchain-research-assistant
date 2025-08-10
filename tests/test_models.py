#!/usr/bin/env python3
"""
Quick script to test which model names are valid with OpenAI API.
"""

import os
import sys
from pathlib import Path
from openai import OpenAI

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def test_model(client: OpenAI, model_name: str) -> tuple[bool, str]:
    """Test if a model is available and working."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return True, "✅ Available"
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "Invalid model" in error_msg:
            return False, "❌ Model not found"
        elif "max_tokens" in error_msg and "max_completion_tokens" in error_msg:
            return True, "⚠️  Needs max_completion_tokens parameter"
        else:
            return False, f"❌ Error: {error_msg[:50]}..."

def main():
    """Test model availability."""
    # Models you want to test
    models_to_test = [
        "gpt-5-mini", 
        "gpt-5-nano",
        "gpt-4.1",
        "gpt-4.1-mini", 
        "gpt-4.1-nano",
        "o3-mini",
    ]
    
    client = OpenAI()
    
    print("🔍 Testing model availability...")
    print("=" * 50)
    
    available_models = []
    special_param_models = []
    
    for model in models_to_test:
        is_available, status = test_model(client, model)
        print(f"{model:<20} {status}")
        
        if is_available:
            if "max_completion_tokens" in status:
                special_param_models.append(model)
            else:
                available_models.append(model)
    
    print("\n📊 SUMMARY:")
    print(f"✅ Available models: {len(available_models)}")
    for model in available_models:
        print(f"   - {model}")
    
    if special_param_models:
        print(f"⚠️  Special parameter models: {len(special_param_models)}")
        for model in special_param_models:
            print(f"   - {model} (use max_completion_tokens)")
    
    print(f"\n🚀 Ready to benchmark {len(available_models + special_param_models)} models!")

if __name__ == "__main__":
    main()