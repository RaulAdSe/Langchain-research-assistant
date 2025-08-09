#!/bin/bash

# Multi-Agent Research Assistant Quick Start Script

echo "🚀 Starting Multi-Agent Research Assistant Setup..."

# Check if Python 3.11+ is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.11 or higher."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before proceeding."
    echo "   Required: OPENAI_API_KEY and LANGSMITH_API_KEY"
fi

# Create data directory if it doesn't exist
mkdir -p data/sample_docs

echo ""
echo "✅ Setup complete! Next steps:"
echo ""
echo "1. Edit .env file with your API keys:"
echo "   nano .env"
echo ""
echo "2. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "3. Add sample data:"
echo "   python3 -m app.cli sample"
echo ""
echo "4. Ask your first question:"
echo "   python3 -m app.cli ask 'What is machine learning?'"
echo ""
echo "5. Start the API server:"
echo "   uvicorn app.api:app --reload"
echo ""
echo "📖 For full documentation, see README.md"
echo "🛟 For quick setup guide, see SETUP.md"