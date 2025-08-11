# 📚 Citation System & Web Search

## 🎉 Real Web Search Now Available!

Your research assistant now has **FREE real web search** using DuckDuckGo:

### 🦆 **DuckDuckGo Integration (Active)**
- **No API key needed** - Works out of the box!
- **Real search results** with actual URLs
- **Free forever** - No rate limits or costs
- Automatically filters out ads for quality results

### 🏠 **Only Local Knowledge Base**
- Real citations come from ChromaDB (your local documents)
- These show as `(Local Knowledge Base)` in results
- Limited to documents you've ingested

## ✅ How to Get Real Citations

### Option 1: Add SerpAPI (Recommended)
Get real Google search results with actual working URLs:

1. **Sign up at [SerpAPI](https://serpapi.com)**
   - Free tier: 100 searches/month
   - Paid plans for more searches

2. **Get your API key** from dashboard

3. **Add to `.env`**:
   ```bash
   SEARCH_API_KEY=your_serpapi_key_here
   ```

4. **Restart and test**:
   ```bash
   ./bin/ask "What is quantum computing?" --fast
   ```

You'll now get **real citations** with working URLs from actual web pages!

### Option 2: Ingest More Documents
Add more sources to your knowledge base:

```bash
# Add documents to data/sample_docs/
cp your_documents.pdf data/sample_docs/

# Ingest into ChromaDB
python -m app.rag.ingest data/sample_docs/
```

## Current Citation Types

### 🦆 **Web Search (Real - DuckDuckGo)**
```
[#1] Docker Engine | Docker Docs
[#2] Containerization using Docker - GeeksforGeeks
[#3] Docker Explained - Medium
```

**These are REAL articles with working URLs from DuckDuckGo search!**

### 🌐 **Web Search (Enhanced - With SerpAPI Key)**
```
[#1] Python Programming Language - python.org - 2024-02-15
[#2] Python Tutorial for Beginners - realpython.com - 2024-01-10
[#3] Advanced Python Features - towardsdatascience.com - 2024-01-25
```

**With SerpAPI: Includes dates, more results, and Google's ranking**

### 🏠 **Knowledge Base (Local Documents)**
```
[#4] langchain_overview.md (Local Knowledge Base)
[#5] rag_best_practices.md (Local Knowledge Base)
```

**Your own ingested documents**

## Citation Quality Comparison

| Source Type | URL Quality | Content Quality | Cost |
|-------------|-------------|-----------------|------|
| **Mock Web** | ❌ Generic | ⚠️ Limited | Free |
| **Real Web** | ✅ Specific | ✅ Current | ~$0.005/search |
| **Local KB** | ⚠️ Local | ✅ Authoritative | Free |

## Testing Citations

### Test with Mock (Current)
```bash
./bin/ask "What is Docker?" --verbose
```

**Expected**: Generic Wikipedia/Scholar links

### Test with Real API
```bash
# After adding SEARCH_API_KEY
./bin/ask "What is Docker?" --verbose  
```

**Expected**: Real Docker documentation, tutorials, articles

## Advanced Configuration

### Custom Search Parameters
Edit `app/tools/web_search.py` to customize:

```python
# Search parameters
params = {
    "q": query,
    "api_key": settings.search_api_key,
    "num": top_k,           # Number of results
    "engine": "google",     # Search engine
    "hl": "en",            # Language
    "gl": "us",            # Country
    "safe": "active"       # Safe search
}
```

### Multiple Search APIs
The system supports different APIs via `SEARCH_API` in `.env`:

```bash
# Current (SerpAPI)
SEARCH_API=serpapi
SEARCH_API_KEY=your_serpapi_key

# Future: Could add Bing, DuckDuckGo, etc.
```

## Troubleshooting Citations

### ❌ "Sources" section empty
- Check if tools are actually running: use `--verbose`
- Verify API key is correct
- Check API quota/billing

### ❌ Citations still generic
- Restart CLI after changing `.env`
- Check API key format (SerpAPI keys start with specific prefix)
- Test API key directly at serpapi.com

### ❌ "Local Knowledge Base" only
- Add documents to `data/sample_docs/`
- Run ingestion: `python -m app.rag.ingest`
- Configure web search API for external sources

## Cost Analysis

### SerpAPI Pricing (Example)
- **Free**: 100 searches/month
- **Starter**: $50/month → 5,000 searches  
- **Professional**: $125/month → 15,000 searches

### Per-Question Cost
- Average question uses 2-3 searches
- Cost: ~$0.01-0.03 per question with real web search
- **vs Free but generic citations**

## Recommendation

For **development/personal use**: Start with free tier (100 searches)
For **production**: Upgrade based on usage

The difference in citation quality is substantial - real working URLs vs generic search pages make the research much more valuable!

---

**Quick Start**: Just add your SerpAPI key to `.env` and restart. That's it! 🚀