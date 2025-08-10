# 📁 Project Structure

Clean, organized structure with everything in its proper place:

```
langchain-research-assistant/
├── 📂 app/                     # Core application code
│   ├── 📂 chains/             # Agent implementations
│   │   ├── orchestrator.py
│   │   ├── researcher.py
│   │   ├── critic.py
│   │   └── synthesizer.py
│   ├── 📂 core/               # Core functionality
│   │   ├── config.py          # Configuration management
│   │   ├── llm.py            # LLM provider abstraction
│   │   └── state.py          # Pipeline state management
│   ├── 📂 tools/              # Research tools
│   │   ├── retriever.py       # RAG retrieval
│   │   ├── web_search.py      # Web search
│   │   └── firecrawl.py       # Web scraping
│   ├── 📂 rag/                # RAG implementation
│   │   ├── ingest.py          # Document ingestion
│   │   └── store.py           # Vector store management
│   ├── 📂 eval/               # Evaluation framework
│   │   ├── datasets.jsonl     # Test datasets
│   │   ├── rubric.md          # Evaluation criteria
│   │   └── run_eval.py        # Evaluation runner
│   ├── api.py                 # FastAPI REST endpoints
│   ├── cli.py                 # Original CLI interface
│   ├── pipeline.py            # Main research pipeline
│   └── streaming_pipeline.py  # Streaming implementation
├── 📂 bin/                     # Executable scripts
│   ├── ask                    # Quick CLI wrapper
│   ├── research.py            # Main CLI with streaming
│   └── cli_streaming.py       # Advanced CLI features
├── 📂 scripts/                # Shell scripts
│   ├── run_benchmark.sh       # Benchmark runner
│   ├── run_parallel_benchmark.sh  # Parallel benchmarks
│   └── start.sh               # Development server
├── 📂 tests/                  # All test files
│   ├── 📂 unit/               # Unit tests
│   ├── 📂 integration/        # Integration tests
│   ├── 📂 e2e/                # End-to-end tests
│   ├── test_streaming.py      # Streaming tests
│   ├── test_multi_model.py    # Multi-model tests
│   ├── test_models.py         # Model tests
│   ├── benchmark_models.py    # Model benchmarks
│   └── benchmark_parallel.py  # Parallel benchmarks
├── 📂 data/                   # Sample documents
│   └── 📂 sample_docs/
├── 📂 prompts/                # Agent prompt templates
├── 📂 docs/                   # Documentation
├── 📂 benchmark_results/      # Benchmark outputs
├── 📂 eval_results/           # Evaluation outputs
├── 📂 chroma_db/              # Vector database
├── 📂 venv/                   # Python virtual environment
├── CLI_GUIDE.md               # CLI usage guide
├── README.md                  # Main documentation
├── SETUP.md                   # Setup instructions
├── pyproject.toml             # Python project config
├── requirements.txt           # Dependencies
└── .env                       # Environment variables
```

## Usage Paths

### Command Line Interface
- `./bin/ask "question"` - Quick research
- `./bin/research.py` - Full CLI with all features

### Testing
- `python -m pytest tests/` - Run all tests
- `./scripts/run_benchmark.sh` - Performance benchmarks

### Development
- `./scripts/start.sh` - Start development server
- `python -m app.api` - REST API server

## Key Benefits of This Structure

1. **Clean Separation** - Code, tests, scripts, and docs are properly separated
2. **Intuitive Paths** - Everything is where you'd expect it
3. **Easy Navigation** - No more hunting for files in the root directory
4. **Scalable** - New components fit naturally into existing structure
5. **Professional** - Follows Python project best practices

## What Was Moved

- ❌ **Removed**: All frontend/web files (SSE endpoints, demo pages, etc.)
- 📁 **tests/**: All test and benchmark files
- 📁 **bin/**: Executable CLI tools  
- 📁 **scripts/**: Shell scripts and utilities
- 🧹 **Root**: Cleaned up temporary and demo files

The project is now focused purely on **terminal/CLI usage** with a clean, organized structure!