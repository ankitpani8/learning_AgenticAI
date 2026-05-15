"""Two-layer cache for agent responses.

Layer 1: exact-match (hash of query) -- O(1), zero LLM cost
Layer 2: semantic (embedded query similarity) -- one embedding call, no LLM

In production, layer 1 is Redis and layer 2 is the same vector DB you use
for RAG. For the lab we use in-process dicts and ChromaDB.
"""
import hashlib
import sys
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent.parent))

CHROMA_DIR = Path(__file__).parent / "cache_db"
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
SEMANTIC_THRESHOLD = 0.15  # cosine distance; tighter than RAG since we want near-exact

# Layer 1: in-memory exact-match cache. dict[hash, (response, timestamp)]
_exact_cache: dict[str, tuple[str, float]] = {}
EXACT_TTL_SECONDS = 3600  # entries expire after an hour


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _semantic_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name="response_cache",
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def store(query: str, response: str) -> None:
    """Write to both cache layers."""
    key = _hash(query)
    _exact_cache[key] = (response, time.time())

    coll = _semantic_collection()
    coll.add(
        ids=[f"cache_{key[:16]}_{time.time()}"],
        documents=[response],
        metadatas=[{"query": query, "timestamp": time.time()}],
    )

_stats = {"hits_exact": 0, "hits_semantic": 0, "misses": 0}

def get_cached(query: str) -> tuple[str, str] | None:
    key = _hash(query)
    if entry := _exact_cache.get(key):
        response, ts = entry
        if time.time() - ts < EXACT_TTL_SECONDS:
            _stats["hits_exact"] += 1
            return response, "exact"
        del _exact_cache[key]
    coll = _semantic_collection()
    if coll.count() == 0:
        _stats["misses"] += 1
        return None
    result = coll.query(query_texts=[query], n_results=1)
    if result["distances"] and result["distances"][0]:
        dist = result["distances"][0][0]
        if dist <= SEMANTIC_THRESHOLD:
            _stats["hits_semantic"] += 1
            return result["documents"][0][0], f"semantic(d={dist:.3f})"
    _stats["misses"] += 1
    return None


def stats() -> dict:
    total = _stats["hits_exact"] + _stats["hits_semantic"] + _stats["misses"]
    hit_rate = (_stats["hits_exact"] + _stats["hits_semantic"]) / total if total else 0
    return {
        "exact_size": len(_exact_cache),
        "semantic_size": _semantic_collection().count(),
        **_stats,
        "hit_rate": round(hit_rate, 3),
    }

def clear() -> None:
    global _exact_cache
    _exact_cache.clear()
    coll = _semantic_collection()
    if coll.count() > 0:
        ids = coll.get()["ids"]
        coll.delete(ids=ids)



