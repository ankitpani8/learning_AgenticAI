"""Async RAG agent reusing Module 4's knowledge base. Production patterns:
async throughout, retries on LLM calls, telemetry, cache integration."""
import sys
import importlib
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "module4_memory_rag"))

from lib.providers import select_all_models
import importlib
_retry = importlib.import_module("02_retry")
_telemetry = importlib.import_module("03_telemetry")
_cache = importlib.import_module("01_cache")
get_cached = _cache.get_cached
cache_store = _cache.store
# Reuse Module 4's indexer
_indexer = importlib.import_module("01_indexer")
retrieve = _indexer.retrieve

load_dotenv(Path(__file__).parent.parent / ".env")

SELECTIONS = select_all_models(roles=["heavy"])
llm = SELECTIONS["heavy"].to_langchain(temperature=0.3)

TOKEN_BUDGET = 5000  # hard per-request cap: input + output tokens combined


class TokenBudgetExceeded(Exception):
    """Raised when a request's token usage exceeds TOKEN_BUDGET."""


def _estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


SYSTEM_PROMPT = """You are a helpful assistant answering questions using the
provided context. Be concise. If the context doesn't contain the answer,
say so plainly — do not invent facts."""


async def answer_async(query: str) -> str:
    """Non-streaming async answer. Used when streaming isn't needed."""
    with _telemetry.telemetry(query) as t:
        # Cache check
        if cached := get_cached(query):
            response, hit_type = cached
            t.cache_hit = hit_type
            return response

        # RAG retrieval
        chunks = retrieve(query, k=3, score_threshold=0.7)
        t.rag_chunks = len(chunks)

        context = ""
        if chunks:
            context = "\n\n".join(
                f"From {c['meta']['doc_id']}:\n{c['text']}" for c in chunks
            )

        # Build messages
        sys_content = SYSTEM_PROMPT
        if context:
            sys_content += f"\n\n## Context\n{context}"
        else:
            sys_content += "\n\nNo context retrieved. If the query needs facts, say you don't know."

        messages = [SystemMessage(content=sys_content), HumanMessage(content=query)]

        # LLM call with retry
        response = await _retry.call_with_retry(lambda: llm.ainvoke(messages))

        # Telemetry
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            u = response.usage_metadata
            t.add_llm_usage(u.get("input_tokens", 0), u.get("output_tokens", 0))

        total_tokens = t.input_tokens + t.output_tokens
        if total_tokens > TOKEN_BUDGET:
            raise TokenBudgetExceeded(
                f"Request used {total_tokens} tokens, exceeding budget of {TOKEN_BUDGET}."
            )

        # Store in cache for next time
        cache_store(query, response.content)
        return response.content


async def stream_answer_async(query: str) -> AsyncIterator[str]:
    """Streaming variant. Yields tokens as they're generated."""
    with _telemetry.telemetry(query) as t:
        if cached := get_cached(query):
            response, hit_type = cached
            t.cache_hit = hit_type
            # Stream the cached response in chunks for consistent UX
            for chunk in [response[i:i+50] for i in range(0, len(response), 50)]:
                yield chunk
            return

        chunks = retrieve(query, k=3, score_threshold=0.7)
        t.rag_chunks = len(chunks)

        context = ""
        if chunks:
            context = "\n\n".join(
                f"From {c['meta']['doc_id']}:\n{c['text']}" for c in chunks
            )

        sys_content = SYSTEM_PROMPT
        if context:
            sys_content += f"\n\n## Context\n{context}"

        messages = [SystemMessage(content=sys_content), HumanMessage(content=query)]

        # Pre-flight: abort before any LLM call if input alone exceeds budget.
        estimated_input = _estimate_tokens(sys_content + query)
        if estimated_input > TOKEN_BUDGET:
            raise TokenBudgetExceeded(
                f"Estimated input ({estimated_input} tokens) already exceeds "
                f"budget of {TOKEN_BUDGET} tokens."
            )

        # Stream tokens. Note: no retry wrapper here because mid-stream
        # retries are complex (would re-emit partial output). Production
        # systems usually accept stream failures and let the client retry.
        full_response = ""
        async for chunk in llm.astream(messages):
            token = chunk.content
            if token:
                full_response += token
                estimated_total = estimated_input + _estimate_tokens(full_response)
                if estimated_total > TOKEN_BUDGET:
                    t.add_llm_usage(estimated_input, _estimate_tokens(full_response))
                    raise TokenBudgetExceeded(
                        f"Token budget of {TOKEN_BUDGET} exceeded mid-stream "
                        f"(~{estimated_total} tokens used so far)."
                    )
                yield token

        # Cache only the complete response
        cache_store(query, full_response)
        t.add_llm_usage(estimated_input, _estimate_tokens(full_response))