"""FastAPI REST API for the research assistant."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime
from pathlib import Path
import traceback

from app.pipeline import default_pipeline, research
from app.core.state import ResearchRequest, ResearchResponse
from app.rag.ingest import DocumentIngester, ingest_sample_data
from app.rag.store import get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Research Assistant API",
    description="A comprehensive research assistant powered by multiple AI agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AskRequest(BaseModel):
    """Request model for ask endpoint."""
    question: str
    context: Optional[str] = None
    max_sources: int = 5
    require_recent: bool = False
    allowed_domains: Optional[List[str]] = None
    blocked_domains: Optional[List[str]] = None


class AskResponse(BaseModel):
    """Response model for ask endpoint."""
    answer: str
    citations: List[Dict[str, Any]]
    confidence: float
    summary: Optional[str] = None
    key_points: Optional[List[str]] = None
    caveats: Optional[List[str]] = None
    trace_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    timestamp: str


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    file_path: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_size: int = 800
    chunk_overlap: int = 120


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    status: str
    message: str
    documents_processed: int
    chunks_created: int
    chunks_ingested: int


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    collection_name: str
    document_count: int
    persist_directory: str
    status: str
    error: Optional[str] = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "research-assistant-api"
    }


# Main research endpoint
@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Ask a research question and get a comprehensive answer.
    
    This endpoint orchestrates multiple AI agents to:
    1. Plan the research strategy
    2. Execute research using tools (knowledge base, web search)
    3. Critique the findings
    4. Synthesize a final, well-structured answer
    """
    try:
        logger.info(f"Received question: {request.question[:100]}...")
        
        # Convert to internal request format
        research_request = ResearchRequest(
            question=request.question,
            context=request.context,
            max_sources=request.max_sources,
            require_recent=request.require_recent,
            allowed_domains=request.allowed_domains,
            blocked_domains=request.blocked_domains
        )
        
        # Run research pipeline
        start_time = datetime.utcnow()
        response = default_pipeline.run(research_request)
        
        # Format response
        api_response = AskResponse(
            answer=response.answer,
            citations=response.citations,
            confidence=response.confidence,
            summary=response.summary,
            key_points=response.key_points,
            caveats=response.caveats,
            trace_url=response.trace_url,
            duration_seconds=response.duration_seconds,
            timestamp=start_time.isoformat()
        )
        
        logger.info(f"Research completed in {response.duration_seconds:.2f}s with confidence {response.confidence:.2%}")
        return api_response
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Research pipeline error: {str(e)}"
        )


# Knowledge base ingestion endpoint
@app.post("/ingest", response_model=IngestResponse)
async def ingest_content(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest content into the knowledge base.
    
    Can ingest either a file path or direct content.
    Processing happens in the background for large files.
    """
    try:
        if not request.file_path and not request.content:
            raise HTTPException(
                status_code=400,
                detail="Either file_path or content must be provided"
            )
        
        ingester = DocumentIngester(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        if request.file_path:
            # Validate file exists
            file_path = Path(request.file_path)
            if not file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"File not found: {request.file_path}"
                )
            
            # Ingest file
            stats = ingester.ingest_file(file_path)
        else:
            # Ingest direct content
            from langchain_core.documents import Document
            doc = Document(
                page_content=request.content,
                metadata=request.metadata or {}
            )
            stats = ingester.ingest_documents([doc])
        
        if stats["status"] == "success":
            return IngestResponse(
                status="success",
                message="Content ingested successfully",
                documents_processed=stats.get("documents_processed", 1),
                chunks_created=stats.get("chunks_created", 0),
                chunks_ingested=stats.get("chunks_ingested", 0)
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=stats.get("message", "Ingestion failed")
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ingest endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion error: {str(e)}"
        )


# Sample data ingestion endpoint
@app.post("/ingest/sample")
async def ingest_sample():
    """Ingest sample documents for testing and demonstration."""
    try:
        stats = ingest_sample_data()
        
        if stats["status"] == "success":
            return {
                "status": "success",
                "message": "Sample data ingested successfully",
                "documents_processed": stats.get("documents_processed", 0),
                "chunks_ingested": stats.get("chunks_ingested", 0)
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to ingest sample data"
            )
            
    except Exception as e:
        logger.error(f"Error ingesting sample data: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Sample ingestion error: {str(e)}"
        )


# Knowledge base statistics endpoint
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    try:
        store = get_vector_store()
        stats = store.get_collection_stats()
        
        return StatsResponse(
            collection_name=stats.get("collection_name", "unknown"),
            document_count=stats.get("document_count", 0),
            persist_directory=stats.get("persist_directory", ""),
            status="success" if "error" not in stats else "error",
            error=stats.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        return StatsResponse(
            collection_name="unknown",
            document_count=0,
            persist_directory="",
            status="error",
            error=str(e)
        )


# Reset knowledge base endpoint
@app.delete("/reset")
async def reset_knowledge_base():
    """Reset the knowledge base (delete all documents)."""
    try:
        store = get_vector_store()
        store.reset()
        
        return {
            "status": "success",
            "message": "Knowledge base reset successfully"
        }
        
    except Exception as e:
        logger.error(f"Error resetting knowledge base: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Reset error: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": [
                "/health",
                "/ask",
                "/ingest",
                "/ingest/sample",
                "/stats",
                "/reset",
                "/docs"
            ]
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Multi-Agent Research Assistant API")
    logger.info("API documentation available at /docs")
    
    # Validate configuration
    try:
        from app.core.config import settings
        settings.validate_provider_keys()
        logger.info(f"Configuration validated - Provider: {settings.provider}")
    except Exception as e:
        logger.warning(f"Configuration warning: {e}")


# Main entry point
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )