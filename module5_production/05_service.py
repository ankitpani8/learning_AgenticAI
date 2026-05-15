"""FastAPI service exposing the agent over HTTP, with streaming support."""
import importlib
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

_agent = importlib.import_module("04_agent_RAG")
_cache = importlib.import_module("01_cache")


# Per-IP rate limiter. In-memory storage is fine for single-process dev;
# production behind multiple workers needs a shared store (Redis).
limiter = Limiter(key_func=get_remote_address, default_limits=[])
RATE_LIMIT = "5/minute"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks. Production systems use this to verify
    dependencies (DB, vector store, model providers) before accepting traffic."""
    print("[service] starting up")
    # Warm the embedding model
    _cache.stats()
    print(f"[service] rate limit: {RATE_LIMIT} per IP on /ask, /ask/stream, /cache/clear")
    print("[service] ready")
    yield
    print("[service] shutting down")


app = FastAPI(title="Agentic RAG Service", version="0.5.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
async def health():
    return {"status": "ok", "cache": _cache.stats()}


@app.post("/ask", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT)
async def ask(request: Request, req: QueryRequest):
    """Non-streaming endpoint. Returns the full answer when done."""
    try:
        answer = await _agent.answer_async(req.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
@limiter.limit(RATE_LIMIT)
async def ask_stream(request: Request, req: QueryRequest):
    """Streaming endpoint via Server-Sent Events."""
    async def event_generator():
        try:
            async for chunk in _agent.stream_answer_async(req.query):
                yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}
        except Exception as e:
            yield {"event": "error", "data": str(e)}

    return EventSourceResponse(event_generator())

@app.post("/cache/clear")
@limiter.limit(RATE_LIMIT)
async def clear_cache(request: Request):
    _cache.clear()
    return {"status": "cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("05_service:app", host="127.0.0.1", port=8000, reload=False)