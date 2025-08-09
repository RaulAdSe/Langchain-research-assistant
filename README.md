# Multi-Agent Research Assistant

A comprehensive research assistant powered by multiple AI agents using LangChain and LangSmith. This system orchestrates specialized agents to plan research strategies, execute searches across knowledge bases and the web, critique findings, and synthesize well-structured answers with proper citations.

## 🌟 Features

- **Multi-Agent Architecture**: Orchestrator, Researcher, Critic, and Synthesizer agents working together
- **Provider-Agnostic LLM Support**: Switch between OpenAI, Anthropic, and other providers via configuration
- **Comprehensive RAG Pipeline**: Document ingestion, chunking, embedding, and retrieval
- **Multiple Search Tools**: Knowledge base search, web search, and optional Firecrawl integration
- **Quality Assurance**: Built-in critique and review system for answer quality
- **Full Observability**: Complete LangSmith tracing and monitoring
- **CLI and REST API**: Multiple interfaces for different use cases
- **Comprehensive Evaluation**: Automated testing framework with quality metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key or Anthropic API key
- LangSmith account (optional but recommended)

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd multiagent-research-assistant
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env file with your API keys
```

3. **Ingest sample documents:**
```bash
python3 -m app.cli sample
```

4. **Test the system:**
```bash
python3 -m app.cli ask "What is machine learning?"
```

### Using the CLI

```bash
# Ask research questions
python -m app.cli ask "What are the latest developments in quantum computing?"

# Add context for more focused research
python -m app.cli ask "Explain neural networks" --context "Focus on deep learning applications"

# Ingest your own documents
python -m app.cli ingest /path/to/your/documents

# Check knowledge base status
python -m app.cli stats

# Get help
python -m app.cli --help
```

### Using the API

1. **Start the server:**
```bash
uvicorn app.api:app --reload
# Server starts at http://localhost:8000
```

2. **Ask questions via HTTP:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "max_sources": 5
  }'
```

3. **View API documentation:**
Visit http://localhost:8000/docs for interactive Swagger documentation.

## 🏗️ Architecture

### Agent Pipeline

```
User Question → Orchestrator (plans research)
                      ↓
                 Researcher (executes with tools)
                      ↓
                 Critic (reviews quality)
                      ↓
               Synthesizer (produces final answer)
```

### Components

- **Core**: Configuration, LLM adapters, state management
- **Chains**: Individual agent implementations
- **Tools**: Web search, knowledge base retrieval, content extraction
- **RAG**: Document ingestion and vector storage
- **Evaluation**: Quality metrics and testing framework

## 🛠️ Development

### Setup Development Environment

```bash
make dev-setup
```

### Running Tests

```bash
# All tests
make test

# By category
make test-unit
make test-integration
make test-e2e

# With coverage
make test-coverage

# Fast tests only (excludes slow E2E tests)
make test-fast
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Type checking
make type-check

# All quality checks
make check-all
```

### Evaluation

```bash
# Run evaluation on test dataset
make eval

# Quick evaluation (5 questions)
make eval-fast

# Full evaluation with detailed output
make eval-full
```

## 📊 Evaluation Framework

The system includes a comprehensive evaluation framework that tests:

- **Faithfulness**: Claims supported by evidence
- **Answerability**: Appropriate handling of answerable vs unanswerable questions
- **Citation Coverage**: Quality and completeness of source citations
- **Completeness**: Thoroughness in addressing the question
- **Coherence**: Logical structure and flow
- **Currency**: Use of up-to-date information when relevant

### Evaluation Dataset

The evaluation dataset (`app/eval/datasets.jsonl`) includes:
- 15 test questions across different difficulty levels
- Geography, science, technology, and policy questions
- Edge cases for refusal and error handling
- Expected outcomes and validation criteria

### Running Evaluations

```bash
# Full evaluation
python -m app.eval.run_eval

# Evaluate specific subset
python -m app.eval.run_eval --max-questions 5

# Custom dataset
python -m app.eval.run_eval --dataset custom_questions.jsonl
```

Results are saved as JSON, CSV, and summary metrics for analysis.

## ⚙️ Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# LLM Provider
PROVIDER=openai  # or anthropic
OPENAI_API_KEY=your-key
MODEL_NAME=gpt-4o-mini

# LangSmith Tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
LANGSMITH_PROJECT=your-project

# Embeddings
EMBEDDINGS_PROVIDER=openai
EMBEDDINGS_MODEL=text-embedding-3-small

# Optional: Web Search
SEARCH_API=serpapi
SEARCH_API_KEY=your-key

# Optional: Firecrawl
FIRECRAWL_API_KEY=your-key
```

### Provider Configuration

The system supports multiple LLM providers through a unified interface:

```python
# Automatic provider selection based on config
from app.core.llm import chat_model

llm = chat_model()  # Uses configured provider

# Override provider
llm = chat_model(provider="anthropic", model="claude-3-sonnet")
```

## 📁 Project Structure

```
├── app/                    # Main application code
│   ├── chains/            # Agent implementations
│   ├── core/              # Configuration and state management
│   ├── rag/               # Document ingestion and retrieval
│   ├── tools/             # Search and extraction tools
│   ├── eval/              # Evaluation framework
│   ├── api.py             # FastAPI REST interface
│   └── cli.py             # Command-line interface
├── prompts/               # Agent prompts
├── tests/                 # Test suite
│   ├── unit/             # Unit tests
│   ├── integration/      # Integration tests
│   └── e2e/              # End-to-end tests
├── docs/                 # Documentation
└── data/                 # Sample data and documents
```

## 🔧 Customization

### Adding New Tools

1. Create tool class inheriting from `BaseTool`
2. Register in `app/tools/__init__.py`
3. Update orchestrator logic to use the tool

### Custom Prompts

Edit prompt files in the `prompts/` directory:
- `orchestrator.claude` - Research planning
- `researcher.claude` - Information gathering
- `critic.claude` - Quality review
- `synthesizer.claude` - Final answer generation

### Evaluation Metrics

Add custom evaluation metrics in `app/eval/run_eval.py`:

```python
def evaluate_custom_metric(self, answer: str, expected: Dict) -> float:
    # Your custom evaluation logic
    return score
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run quality checks: `make check-all`
5. Submit a pull request

### Development Workflow

```bash
# Setup
make dev-setup

# Make changes
# ... edit code ...

# Test changes
make test
make check-all

# Run evaluation
make eval-fast

# Commit
git commit -m "Your changes"
```

## 📝 License

MIT License - see LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**No API Key Error:**
```bash
make env-check  # Verify configuration
```

**Empty Knowledge Base:**
```bash
make ingest-sample  # Add sample documents
make stats          # Check status
```

**LangSmith Tracing Issues:**
- Verify `LANGSMITH_API_KEY` is set
- Check project name matches your LangSmith project
- Set `LANGSMITH_TRACING=false` to disable if needed

**Slow Performance:**
- Reduce `max_sources` parameter
- Use smaller embedding models
- Enable caching for repeated queries

### Getting Help

1. Check the [documentation](docs/)
2. Review [example usage](examples/)
3. Open an [issue](https://github.com/yourusername/multiagent-research-assistant/issues)

## 🎯 Roadmap

- [ ] Additional LLM providers (Cohere, Mistral)
- [ ] Advanced reranking for better retrieval
- [ ] Multi-modal document support (images, tables)
- [ ] Real-time collaborative research
- [ ] Custom agent workflows
- [ ] Performance optimization and caching

---

**Built with ❤️ using LangChain, LangSmith, and modern Python practices.**