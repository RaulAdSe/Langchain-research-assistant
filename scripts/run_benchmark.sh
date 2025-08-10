#!/bin/bash
# Quick benchmark runner that loads environment variables

cd "$(dirname "$0")"
source .venv/bin/activate

# Load environment variables from .env
set -a  # automatically export all variables
source .env
set +a  # stop automatically exporting

# Run benchmark with your specified models
python benchmark_models.py \
    --models gpt-5-mini gpt-5-nano gpt-4.1 gpt-4.1-mini gpt-4.1-nano o3-mini \
    --questions "What is machine learning?" "What is Python?" \
    --fast-only \
    "$@"

echo ""
echo "🎉 Benchmark completed! Check the benchmark_results/ directory for reports."