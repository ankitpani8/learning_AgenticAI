# Module 5 — Production Architecture

A FastAPI service wrapping Module 4's RAG agent with the patterns that turn
lab-grade code into something that could run on a server: async throughout,
streaming responses, two-layer caching, retry policies with failure-type
classification, telemetry per request, and rate limiting.

## The shift this module makes

Previous modules built agents that work. This module builds a *service* —
the deployment shell around an agent. The distinction matters: an agent is
logic (graph, tools, prompts); a service is everything around it (HTTP
endpoints, async I/O, retries, caching, observability, rate limits).
Production agents are agents plus services.

## Core insight: agents are I/O-bound

Almost everything an agent does is wait — for the LLM API, for tool calls,
for the vector store. CPU is idle most of the time. That fact dictates the
entire toolkit:

- **Async** — one process handles many concurrent waits
- **Streaming** — tokens appear as generated, not after completion
- **Caching** — skip the wait when the answer is already known
- **Retries with backoff** — wait *intelligently* when something fails
- **Rate limiting** — protect against one user torching the budget

## What's in the service

- Async LangGraph + RAG agent with full `async`/`await` stack
- FastAPI endpoints: `/ask` (full response), `/ask/stream` (SSE), `/health`,
  `/cache/clear` (admin)
- Two-layer cache: exact-match (in-process dict) + semantic (ChromaDB)
- Tenacity retry policy distinguishing transient, rate-limit, and permanent failures
- Per-request telemetry (request ID, duration, tokens, cache hit, errors)
  emitted as JSON lines
- Per-IP rate limiting via slowapi (5 req/min on `/ask`)
- Token-budget cap aborting mid-run if exceeded (Challenge 1)
- Cache hit-rate counters visible on `/health` (Challenge 4)

## Files

```
module5_production/
├── 01_cache.py             -- two-layer cache + hit/miss counters
├── 02_retry.py             -- tenacity retry with exception classification
├── 03_telemetry.py         -- per-request structured logging
├── 04_agent.py             -- async RAG agent, token budget, cache integration
├── 05_service.py           -- FastAPI app with all endpoints + rate limiter
├── 06_load_test.py         -- concurrent load harness with cache clearing
└── README.md
```

## Findings

### Async is necessary but not sufficient
First load test attempt: 8 concurrent requests against an Ollama-backed
service. Result: 6/8 timed out, throughput identical to sequential.
Ollama's single Python process serializes requests internally regardless
of how many `Send`/`asyncio.gather` calls fire at it.

Second load test against Gemini Flash-Lite (which has real backend
parallelism) showed the architecture working as designed:
- Sequential 5 requests: 10.69s wall-clock (sum of latencies)
- Concurrent 5 requests: 2.35s wall-clock (slowest tail only)
- Throughput improvement: 4.5x for the same work

**The async architecture is necessary, but the backend has to support
concurrency for the architecture to pay off.** Self-hosted single-instance
model servers won't benefit from async dispatch; hosted APIs and
multi-instance setups will.

### Cache contamination invalidates load tests
Initial run produced misleading numbers because the same query mix was
reused across sequential and concurrent phases — the second phase
measured the cache, not the system. The cleaner approach: unique queries
per phase, plus an admin `/cache/clear` endpoint called between phases.
This is also how production engineers benchmark properly — never measure
caches as if they were the system.

### Provider fallback catches configuration errors, not just runtime ones
Replacing the Gemini API key with garbage caused the startup health check
to fail and the agent to bind to the next provider in the chain (Ollama)
automatically. The fallback chain handles auth misconfigurations the same
way it handles transient outages — fail fast at startup, switch providers,
proceed.

### p50 vs p95 matters more than the mean
The first sequential run showed p50 = 1.99s but max = 37.64s — a huge
tail caused by cold-start serialization on Ollama. Production SLOs are
written against p95 and p99 because **the tail is what your worst-treated
users actually feel.** 1 in 20 users hitting a 24-second response is not
negligible at scale.

### Rate limits are a production reality, not an edge case
Free-tier Gemini caps at 10 RPM on Flash, 15 RPM on Flash-Lite. Load
testing repeatedly burned the quota and forced 60-second cooldowns
between runs. Production agent services pay for higher RPM or run
multi-provider fallbacks for exactly this reason. The retry +
fallback infrastructure built earlier in the curriculum is what makes
this manageable when it happens at runtime.

### Cache hit ratio is the cheapest production win
After the rate-limiter test (5 successful + 2 blocked requests, all with
the same query): hit rate = 0.8. Four LLM calls saved out of five. At
production scale, an FAQ-style agent with 30-60% hit rate is a 30-60%
cost reduction at zero quality cost. Caching is the single highest-ROI
optimization in production agent systems.

## Run it

```bash
cd module5_production

# Start the service (foreground; Ctrl+C to stop)
python 05_service.py

# In another terminal:

# Health + cache stats
curl http://localhost:8000/health

# Non-streaming question
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query": "How should I manage Python dependencies?"}'

# Streaming question (Server-Sent Events)
curl -X POST http://localhost:8000/ask/stream -H "Content-Type: application/json" \
  -d '{"query": "What is the ReAct loop?"}' --no-buffer

# Load test (cleans cache between phases)
python 06_load_test.py

# Interactive docs
open http://localhost:8000/docs
```

Requires the service to use a provider with real backend parallelism
(Gemini, Anthropic) for the load test to show concurrency gains.
Local Ollama works for correctness testing but serializes at the backend.

## What we deliberately didn't do

- **Real Redis / message queues.** In-process dicts demonstrate the
  caching interface; Redis is a one-line config change.
- **Distributed tracing** (OpenTelemetry, Datadog). Module 6 covers
  observability properly; we log to stdout as JSON for now.
- **Connection pooling, request batching.** Useful at higher scale than
  this module exercises.

## What's next

Module 6 introduces **observability and evaluation** — tracing every
request, building evaluation datasets, LLM-as-judge patterns, and
measuring RAG quality (faithfulness, relevance, retrieval recall).