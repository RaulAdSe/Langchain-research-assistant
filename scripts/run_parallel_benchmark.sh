#!/bin/bash
# PARALLEL benchmark runner - much faster!

cd "$(dirname "$0")"
source .venv/bin/activate

# Load environment variables from .env
set -a  # automatically export all variables
source .env
set +a  # stop automatically exporting

echo "🚀 Running PARALLEL benchmark with your 6 models..."
echo "This will be ~6x faster than sequential benchmarking!"
echo ""

# Run parallel benchmark with your models
python benchmark_parallel.py \
    --models gpt-5-mini gpt-5-nano gpt-4.1 gpt-4.1-mini gpt-4.1-nano o3-mini \
    --questions "What is machine learning?" \
    --workers 6 \
    "$@"

echo ""
echo "🎉 Parallel benchmark completed! Check the benchmark_results/ directory for reports."