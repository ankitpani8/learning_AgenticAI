"""Concurrent load test to demonstrate async throughput.

Uses the service's /cache/clear endpoint to start each phase cold."""
import asyncio
import time
import httpx
import statistics

URL = "http://localhost:8000/ask"
CLEAR_URL = "http://localhost:8000/cache/clear"

UNIQUE_QUERIES_A = [
    "Tell me about Python virtual environments",
    "What is dependency pinning",
    "Explain type hints in Python",
    "What is exception handling",
    "Describe the ReAct loop briefly",
    "What are multi-agent topologies",
    "Explain token economics in agents",
    "Why use small models for critics",
]

UNIQUE_QUERIES_B = [
    "How do I lock package versions in Python",
    "When should I add type annotations",
    "What's a good Python testing approach",
    "Tell me about HTTP error handling",
    "Describe how agents iterate on tasks",
    "Compare hierarchical and router agent patterns",
    "How do agent costs grow per turn",
    "What's the role of an evaluator in multi-agent",
]


async def one_request(client: httpx.AsyncClient, query: str) -> tuple[float, int]:
    start = time.time()
    response = await client.post(URL, json={"query": query}, timeout=120)
    return (time.time() - start), response.status_code


async def clear_cache(client: httpx.AsyncClient) -> None:
    await client.post(CLEAR_URL, timeout=10)


async def run_concurrent(client: httpx.AsyncClient, queries: list[str]) -> None:
    start = time.time()
    results = await asyncio.gather(*[one_request(client, q) for q in queries])
    total = time.time() - start
    _report("concurrent", queries, results, total)


async def run_sequential(client: httpx.AsyncClient, queries: list[str]) -> None:
    start = time.time()
    results = []
    for q in queries:
        results.append(await one_request(client, q))
    total = time.time() - start
    _report("sequential", queries, results, total)


def _report(label: str, queries, results, total: float) -> None:
    latencies = [r[0] for r in results]
    success = sum(1 for r in results if r[1] == 200)
    print(f"\n=== {len(queries)} {label} requests ===")
    print(f"Total wall-clock: {total:.2f}s")
    print(f"Success rate:     {success}/{len(queries)}")
    print(f"Latency p50:      {statistics.median(latencies):.2f}s")
    if len(latencies) >= 20:
        print(f"Latency p95:      {statistics.quantiles(latencies, n=20)[18]:.2f}s")
    print(f"Latency max:      {max(latencies):.2f}s")
    print(f"Throughput:       {len(queries) / total:.1f} req/s")


async def main():
    async with httpx.AsyncClient() as client:
        print("[setup] clearing cache for sequential run")
        await clear_cache(client)
        await run_sequential(client, UNIQUE_QUERIES_A[:5])
        
        print("\n[pause] waiting 60s for rate limit window to clear")
        await asyncio.sleep(60)
        
        print("\n[setup] clearing cache for concurrent run")
        await clear_cache(client)
        await run_concurrent(client, UNIQUE_QUERIES_B[:5])


if __name__ == "__main__":
    asyncio.run(main())