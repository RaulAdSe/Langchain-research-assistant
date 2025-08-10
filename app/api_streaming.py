"""FastAPI streaming endpoints with Server-Sent Events."""

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, AsyncIterator
import json
import asyncio
from datetime import datetime

from app.streaming_pipeline import stream_research
from app.models import ResearchRequest


app = FastAPI(title="Research Assistant Streaming API")

# Add CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamRequest(BaseModel):
    """Request model for streaming research."""
    question: str = Field(..., description="Research question")
    context: Optional[str] = Field(None, description="Additional context")
    fast_mode: bool = Field(False, description="Skip critic review for faster results")


async def format_sse(data: dict) -> str:
    """Format data as Server-Sent Event."""
    # SSE format: "data: {json}\n\n"
    return f"data: {json.dumps(data)}\n\n"


async def event_generator(request: StreamRequest) -> AsyncIterator[str]:
    """Generate SSE events from research pipeline."""
    try:
        # Send initial connection event
        yield await format_sse({
            "type": "connection",
            "status": "connected",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Stream research events
        async for event in stream_research(
            question=request.question,
            context=request.context,
            fast_mode=request.fast_mode
        ):
            yield await format_sse(event)
            
            # Small delay to prevent overwhelming client
            await asyncio.sleep(0.01)
        
        # Send completion event
        yield await format_sse({
            "type": "stream_end",
            "status": "completed",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        # Send error event
        yield await format_sse({
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })


@app.post("/stream")
async def stream_research_endpoint(request: StreamRequest):
    """
    Stream research process with Server-Sent Events.
    
    Returns real-time updates including:
    - Phase transitions (orchestrator, researcher, critic, synthesizer)
    - Tool executions
    - Token streaming (when available)
    - Progress indicators
    - Final results
    """
    return StreamingResponse(
        event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        }
    )


@app.get("/stream/demo")
async def demo_page():
    """Return a simple HTML page for testing streaming."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Research Assistant Streaming Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            #question { width: 500px; padding: 10px; }
            #submit { padding: 10px 20px; margin-left: 10px; }
            #events { 
                margin-top: 20px; 
                padding: 10px; 
                border: 1px solid #ccc; 
                height: 400px; 
                overflow-y: auto;
                background: #f5f5f5;
            }
            .event { 
                margin: 5px 0; 
                padding: 5px; 
                background: white;
                border-radius: 3px;
            }
            .phase { color: blue; font-weight: bold; }
            .tool { color: green; }
            .token { color: #333; display: inline; }
            .error { color: red; }
            #answer {
                margin-top: 20px;
                padding: 15px;
                border: 2px solid #4CAF50;
                background: #f9f9f9;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <h1>Research Assistant Streaming Demo</h1>
        <div>
            <input type="text" id="question" placeholder="Enter your research question..." />
            <button id="submit">Research</button>
            <label><input type="checkbox" id="fastMode"> Fast Mode</label>
        </div>
        
        <div id="events"></div>
        <div id="answer" style="display:none;"></div>
        
        <script>
            let eventSource = null;
            let tokens = {};
            
            document.getElementById('submit').onclick = function() {
                const question = document.getElementById('question').value;
                const fastMode = document.getElementById('fastMode').checked;
                
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                // Clear previous results
                document.getElementById('events').innerHTML = '';
                document.getElementById('answer').style.display = 'none';
                tokens = {};
                
                // Close previous connection if exists
                if (eventSource) {
                    eventSource.close();
                }
                
                // Start streaming
                fetch('/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        question: question,
                        fast_mode: fastMode
                    })
                }).then(response => {
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    
                    function processText(text) {
                        buffer += text;
                        const lines = buffer.split('\\n');
                        buffer = lines.pop() || '';
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                try {
                                    const data = JSON.parse(line.slice(6));
                                    handleEvent(data);
                                } catch (e) {
                                    console.error('Parse error:', e);
                                }
                            }
                        }
                    }
                    
                    function read() {
                        reader.read().then(({done, value}) => {
                            if (done) return;
                            processText(decoder.decode(value));
                            read();
                        });
                    }
                    
                    read();
                });
            };
            
            function handleEvent(data) {
                const events = document.getElementById('events');
                const div = document.createElement('div');
                div.className = 'event';
                
                switch(data.type) {
                    case 'phase_start':
                        div.className += ' phase';
                        div.textContent = `[${data.phase.toUpperCase()}] ${data.description}`;
                        break;
                    
                    case 'tool_start':
                        div.className += ' tool';
                        div.textContent = `🔧 Tool: ${data.tool} - ${data.input}`;
                        break;
                    
                    case 'token':
                        // Aggregate tokens by agent
                        const agent = data.agent || 'default';
                        if (!tokens[agent]) tokens[agent] = '';
                        tokens[agent] += data.content;
                        
                        // Update token display
                        let tokenDiv = document.getElementById('tokens-' + agent);
                        if (!tokenDiv) {
                            tokenDiv = document.createElement('div');
                            tokenDiv.id = 'tokens-' + agent;
                            tokenDiv.className = 'token';
                            events.appendChild(tokenDiv);
                        }
                        tokenDiv.textContent = `[${agent}]: ${tokens[agent].slice(-200)}...`;
                        return; // Don't add new div
                    
                    case 'phase_complete':
                        div.textContent = `✅ ${data.phase} complete`;
                        if (data.confidence) {
                            div.textContent += ` (confidence: ${(data.confidence * 100).toFixed(0)}%)`;
                        }
                        break;
                    
                    case 'pipeline_complete':
                        const answer = document.getElementById('answer');
                        answer.innerHTML = `
                            <h2>Final Answer</h2>
                            <p><strong>Confidence:</strong> ${(data.confidence * 100).toFixed(0)}%</p>
                            <div>${data.final_answer}</div>
                        `;
                        answer.style.display = 'block';
                        div.className += ' phase';
                        div.textContent = '🎉 Research complete!';
                        break;
                    
                    case 'error':
                        div.className += ' error';
                        div.textContent = `❌ Error: ${data.error}`;
                        break;
                    
                    default:
                        div.textContent = JSON.stringify(data);
                }
                
                events.appendChild(div);
                events.scrollTop = events.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)