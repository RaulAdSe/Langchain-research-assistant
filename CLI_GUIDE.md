# 🖥️ Terminal/CLI Usage Guide

## Quick Start

The research assistant is designed for **terminal-first** usage with real-time streaming updates. No more blank screens during processing!

### Simple One-liner

```bash
./ask "Your question here"
```

That's it! You'll see live progress as the AI researches your question.

## Installation

```bash
# Clone the repo
git clone https://github.com/RaulAdSe/Langchain-research-assistant.git
cd Langchain-research-assistant

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .

# Make scripts executable
chmod +x research.py ask
```

## Main Commands

### 1. Ask a Question (Most Common)

```bash
# Basic usage
./research.py ask "What is quantum computing?"

# Fast mode (skip quality review, ~30% faster)
./research.py ask "Explain Docker" --fast

# Verbose mode (see all details)
./research.py ask "Compare Python vs JavaScript" --verbose

# With additional context
./research.py ask "Analyze this" --context "Focus on performance aspects"

# Combine flags
./research.py ask "What is AI?" --fast --verbose
```

### 2. Interactive Chat Mode

```bash
./research.py chat
```

Then just type your questions naturally:
```
You: What is machine learning?
[Real-time streaming answer...]

You: How does it differ from deep learning?
[Real-time streaming answer...]

You: /fast explain neural networks
[Quick answer without quality review...]

You: exit
```

### 3. Batch Processing

Process multiple questions from a file:

```bash
# Create a questions file
cat > questions.txt << EOF
What is Docker?
How does Kubernetes work?
Explain microservices architecture
EOF

# Process all questions
./research.py batch questions.txt --fast --output answers.md
```

### 4. Monitor a Topic

Get periodic updates on a topic:

```bash
# Check for updates every 5 minutes
./research.py watch "Latest AI developments" --interval 300
```

## What You'll See

### Standard Output
```
╭─────────────────────────── 🔬 Research Assistant ────────────────────────────╮
│ Question: What is quantum computing?                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
⠋ ORCHESTRATOR: Planning research strategy...
⠸ RESEARCHER: Using web_search...
⠼ SYNTHESIZER: Creating final answer...
✓ Research complete!

╭───────────────────────── ✨ Answer (85% confidence) ─────────────────────────╮
│                                                                              │
│ [Your detailed answer appears here with markdown formatting]                 │
│                                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯
```

### Verbose Output
With `--verbose`, you'll also see:
- ✓ Plan ready, will use: web_search, retriever
- ✓ Found 6 findings  
- ✓ Quality score: 8.5/10
- 📊 Stats: 4 phases, 6 findings, 3 tool calls

## Response Times

| Mode | Typical Time | Use Case |
|------|--------------|----------|
| Fast | 5-10s | Quick questions, general info |
| Standard | 10-20s | Detailed research, quality matters |
| Complex | 20-40s | Multi-source, comprehensive analysis |

## Keyboard Shortcuts

- `Ctrl+C` - Cancel current research
- `Ctrl+D` - Exit chat mode
- `↑/↓` - Navigate command history

## Environment Variables

Create a `.env` file with your API keys:

```bash
# Required
OPENAI_API_KEY=your-key-here

# Optional optimizations
MODEL_NAME=gpt-5-mini           # Default model
ORCHESTRATOR_MODEL=gpt-5-nano   # Fast planning
RESEARCHER_MODEL=gpt-5-mini     # Balanced research
SYNTHESIZER_MODEL=gpt-5         # Best quality answers

# For tracing (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key
```

## Terminal Features

### Color Coding
- 🟡 **Yellow** - Active phase
- 🟢 **Green** - Completed successfully  
- 🔵 **Blue** - Tool execution
- ⚫ **Dim** - Background thinking
- 🔴 **Red** - Errors

### Progress Indicators
- `⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏` - Animated spinner during processing
- Percentage for confidence scores
- Phase completion checkmarks

## Tips & Tricks

### 1. Alias for Quick Access
Add to your `~/.bashrc` or `~/.zshrc`:
```bash
alias ask="/path/to/Langchain-research-assistant/ask"
```

Then from anywhere:
```bash
ask "What is the meaning of life?"
```

### 2. Pipe Output
```bash
# Save answer to file
./ask "Explain quantum computing" > quantum.md

# Search in answer
./ask "List Python frameworks" | grep Django

# Copy to clipboard (macOS)
./ask "What is AI?" | pbcopy
```

### 3. Fast Lookups
For quick definitions or simple questions:
```bash
./ask "Define API" --fast  # ~5 seconds
```

### 4. Research Mode
For comprehensive research:
```bash
./ask "Analyze the impact of AI on healthcare" --verbose
```

## Troubleshooting

### No streaming visible?
- Ensure your terminal supports ANSI colors
- Try with `--verbose` flag
- Check if virtual environment is activated

### Slow responses?
- Use `--fast` mode for quicker answers
- Check your internet connection
- Verify API key is valid

### Error messages?
```bash
# Check configuration
python -c "from app.core.config import settings; print(settings.model_name)"

# Test with simple question
./ask "Hello" --fast --verbose
```

## Advanced Usage

### Custom Models per Agent
Edit `.env` to optimize performance:
```
ORCHESTRATOR_MODEL=gpt-5-nano   # Fast planning
RESEARCHER_MODEL=gpt-5-mini     # Balanced
CRITIC_MODEL=gpt-5-mini         # Quality check  
SYNTHESIZER_MODEL=gpt-5         # Best output
```

### JSON Output (for scripts)
```bash
./research.py ask "What is AI?" --fast --json | jq '.final_answer'
```

## Examples

```bash
# Quick definition
./ask "What is REST API?" --fast

# Detailed comparison  
./ask "Compare PostgreSQL vs MongoDB" --verbose

# Latest information
./ask "Latest GPT models from OpenAI"

# Technical explanation
./ask "How does Docker networking work?"

# Multiple related questions
./research.py chat
> explain kubernetes
> what are pods?
> how do services work?
> exit
```

## Performance

With streaming, you get:
- **First feedback**: ~0.1 seconds
- **Phase updates**: Real-time
- **Total overhead**: ~5% vs non-streaming
- **User experience**: 100x better!

---

**Pro tip**: The terminal interface is the fastest, most efficient way to use this research assistant. No web browser needed, no UI delays - just pure, streaming intelligence at your fingertips! 🚀