# Quick Setup Guide

## Prerequisites

- Python 3.11 or higher
- OpenAI API key (configured in .env)
- Git

## Installation Steps

1. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Test the installation:**
   ```bash
   python3 -m app.cli stats
   ```

5. **Add sample data:**
   ```bash
   python3 -m app.cli sample
   ```

6. **Ask your first question:**
   ```bash
   python3 -m app.cli ask "What is machine learning?"
   ```

## Development Setup

1. **Install development dependencies (with virtual environment activated):**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

3. **Start the API server:**
   ```bash
   uvicorn app.api:app --reload
   ```

Visit http://localhost:8000/docs for API documentation.

## Configuration

### Required Environment Variables

- `OPENAI_API_KEY` - Your OpenAI API key
- `LANGSMITH_API_KEY` - Your LangSmith API key (optional)
- `LANGSMITH_PROJECT` - Your LangSmith project name

### Optional Environment Variables

- `SEARCH_API_KEY` - SerpAPI key for web search
- `FIRECRAWL_API_KEY` - Firecrawl API key for content extraction

See `.env.example` for the complete configuration template.

## Usage Examples

### CLI Commands

```bash
# Ask research questions
python3 -m app.cli ask "What are the latest developments in AI?"

# Add context
python3 -m app.cli ask "Explain quantum computing" --context "Focus on practical applications"

# Ingest documents
python3 -m app.cli ingest /path/to/documents

# Check knowledge base status
python3 -m app.cli stats

# Reset knowledge base
python3 -m app.cli reset
```

### API Usage

```bash
# Start server
uvicorn app.api:app --reload

# Ask question via API
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'

# Ingest content
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"content": "Your document content here"}'
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Run `pip install -r requirements.txt`
2. **API Key Error**: Check your `.env` file configuration
3. **Empty Knowledge Base**: Run `python3 -m app.cli sample` to add sample data
4. **Port Already in Use**: Change port with `uvicorn app.api:app --port 8001`

### Support

- Check the full README.md for detailed documentation
- Review the test suite for usage examples
- Open an issue on GitHub for bugs or questions