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

## 🏗️ System Architecture

### Multi-Agent Research Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          🖥️  TERMINAL INTERFACE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ./bin/ask "question"     │  ./bin/research.py chat  │  Interactive Mode    │
└─────────────┬───────────────────────────┬───────────────────────────┬───────┘
              │                           │                           │
              ▼                           ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       📡 STREAMING PIPELINE                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Event Stream   │    │  Progress Track │    │  Token Stream   │         │
│  │  Generator      │    │  & Callbacks    │    │  Handler        │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        🤖 MULTI-AGENT CHAIN                               │
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌────────┐ │
│  │ 🧭 ORCHESTR │─────▶│ 🔍 RESEARCH │─────▶│ 📝 CRITIC   │─────▶│ ✨ SYNT│ │
│  │ ATOR        │      │ ER          │      │             │      │ HESIZER│ │
│  │             │      │             │      │ (Optional)  │      │        │ │
│  │ • Planning  │      │ • Tools     │      │ • Review    │      │ • Final│ │
│  │ • Strategy  │      │ • Research  │      │ • Quality   │      │ • Polish│ │
│  │ • Tool Seq  │      │ • Data      │      │ • Fixes     │      │ • Format│ │
│  └─────────────┘      └─────────────┘      └─────────────┘      └────────┘ │
│                                │                                            │
│                                ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────────┤
│  │                      🛠️ RESEARCH TOOLS                                  │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │  │ 🌐 Web      │   │ 📚 RAG      │   │ 🕷️ Web      │   │ 💾 Vector   │ │
│  │  │ Search      │   │ Retrieval   │   │ Scraping    │   │ Store       │ │
│  │  │             │   │             │   │             │   │             │ │
│  │  │ • SERP API  │   │ • ChromaDB  │   │ • Firecrawl │   │ • Embeddings│ │
│  │  │ • Real-time │   │ • Semantic  │   │ • Content   │   │ • Search    │ │
│  │  │ • Fresh     │   │ • Context   │   │ • Extraction│   │ • Similarity│ │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘ │
│  └─────────────────────────────────────────────────────────────────────────┘
└─────────────┬───────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ⚙️ CORE INFRASTRUCTURE                              │
│                                                                             │
│  ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────┐ │
│  │ 🧠 LLM PROVIDER     │    │ 📊 STATE MGMT       │    │ ⚙️ CONFIG       │ │
│  │                     │    │                     │    │                 │ │
│  │ • OpenAI/Anthropic  │    │ • Pipeline State    │    │ • Environment   │ │
│  │ • Model Routing     │    │ • Event Tracking    │    │ • API Keys      │ │
│  │ • Agent-Specific    │    │ • Progress Metrics  │    │ • Model Selection│ │
│  │   Models            │    │ • Error Handling    │    │ • Multi-Agent   │ │
│  │                     │    │                     │    │   Optimization  │ │
│  │ gpt-5-nano ─────────┼────┼─ Orchestrator      │    │                 │ │
│  │ gpt-5-mini ─────────┼────┼─ Researcher        │    │                 │ │
│  │ gpt-5 ──────────────┼────┼─ Synthesizer       │    │                 │ │
│  └─────────────────────┘    └─────────────────────┘    └─────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          📊 OBSERVABILITY                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ 🔍 LangSmith    │    │ 📈 Benchmarks   │    │ 🧪 Testing Framework   │ │
│  │ Tracing         │    │                 │    │                         │ │
│  │                 │    │ • Model Perf    │    │ • Unit Tests            │ │
│  │ • Agent Steps   │    │ • Latency       │    │ • Integration Tests     │ │
│  │ • Tool Calls    │    │ • Quality       │    │ • E2E Tests             │ │
│  │ • Token Usage   │    │ • Cost          │    │ • Streaming Tests       │ │
│  │ • Error Debug   │    │ • Parallel      │    │ • Benchmark Suite       │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Architecture

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ User    │───▶│ CLI     │───▶│ Stream  │───▶│ Multi   │───▶│ Result  │
│ Question│    │ Parser  │    │ Pipeline│    │ Agent   │    │ Output  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                      │              │             │
                      ▼              ▼             ▼
               ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
               │ Rich UI     │ │ Event       │ │ LangSmith   │
               │ Progress    │ │ Callbacks   │ │ Tracing     │
               │ Formatting  │ │ Streaming   │ │ Monitoring  │
               └─────────────┘ └─────────────┘ └─────────────┘
```

### Agent Interaction Flow

```
🧭 ORCHESTRATOR ──┐
                  │ Plan: ["web_search", "retriever"]
                  │ Strategy: "Latest tech trends"  
                  ▼
🔍 RESEARCHER ────┐
    │             │ Findings: [6 sources found]
    │             │ Draft: "Technical analysis..."
    ├─ 🌐 Web ────┤
    ├─ 📚 RAG ────┤ Tools execute in parallel
    └─ 🕷️ Crawl ──┤
                  ▼
📝 CRITIC ────────┐ (Optional in --fast mode)
                  │ Score: 8.5/10
                  │ Fixes: ["Add more examples"]
                  ▼
✨ SYNTHESIZER ───┐
                  │ Final Answer: Markdown formatted
                  │ Confidence: 85%
                  │ Citations: [#1] [#2] [#3]
                  ▼
🖥️ Terminal Output with Rich formatting
```

### Performance Optimizations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ⚡ SPEED OPTIMIZATIONS                         │
│                                                                         │
│  Fast Mode (--fast)     │  Multi-Model Setup    │  Streaming Updates   │
│  ┌─────────────────┐   │  ┌─────────────────┐   │  ┌─────────────────┐ │
│  │ Skip Critic     │   │  │ gpt-5-nano      │   │  │ Real-time       │ │
│  │ 30% time saving │   │  │ ├─ Orchestrator │   │  │ Progress        │ │
│  │ Trade quality   │   │  │ gpt-5-mini      │   │  │ ├─ Phase status  │ │
│  │ for speed       │   │  │ ├─ Researcher   │   │  │ ├─ Tool calls    │ │
│  └─────────────────┘   │  │ gpt-5           │   │  │ ├─ Confidence    │ │
│                        │  │ └─ Synthesizer  │   │  │ └─ Token stream  │ │
│                        │  └─────────────────┘   │  └─────────────────┘ │
│                        │                        │                      │
│  5-10s response time   │  Optimized per agent   │  0.1s first feedback │
└─────────────────────────────────────────────────────────────────────────┘
```

This architecture provides a robust, scalable, and user-friendly research assistant focused entirely on terminal/CLI usage with real-time streaming capabilities.