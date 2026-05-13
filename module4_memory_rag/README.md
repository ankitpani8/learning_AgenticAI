# Module 4 — Memory and RAG

A personal assistant agent with three memory layers and a knowledge base that
incrementally reindexes itself. Demonstrates how production agents resolve
the gap between finite context windows and unbounded knowledge.

## The core problem

Three forces collide in real agents:
1. Models have finite context windows
2. Real knowledge bases are unboundedly large
3. Users expect persistent memory across sessions

This module builds two complementary tools that resolve the collision:
**memory** (what the agent remembers about the user/conversation) and
**RAG** (what it can look up from external knowledge).

## Memory layers

Real agents stack multiple memory layers, each with a different scope:

- **Conversation (short-term)** — current messages, lives in LangGraph state
- **Session (medium-term)** — survives turns within one session, via the
  `MemorySaver` checkpointer
- **Semantic facts (long-term)** — durable user facts (name, diet, location)
  in a SQLite key-value store
- **Episodic (long-term)** — conversation summaries embedded in ChromaDB,
  retrievable by similarity

The agent decides per-turn what to load. Conflating these is the most
common mistake in this space — facts go in key-value storage, episodes go
in vector storage, they have different write/retrieve semantics.

## RAG with content-hash incremental reindexing

The knowledge base is a folder of markdown files. The indexer keeps the
vector store in sync using SHA-256 hashes:

- New file → embed and add
- File content changed → delete old chunks, re-embed
- File deleted → drop chunks from store
- File unchanged → skip embedding entirely (the expensive step)

This is the second of three indexing strategies (full reindex / hash-based
incremental / event-driven). It's the middle option — significantly faster
than full reindex, less infrastructure than event-driven.

## Files

```
module4_memory_rag/
├── knowledge_base/              -- markdown corpus for RAG
├── 01_indexer.py                -- content-hash incremental indexer + retrieval
├── 02_memory_stores.py          -- semantic (SQLite) + episodic (ChromaDB) memory
├── 03_agent_langgraph.py        -- the three-layer memory agent
├── 04_test_episodic.py          -- 3-session test: write in session 1, recall in 3
├── 05_test_rag.py               -- RAG retrieval + factual refusal test
├── 06_inspect_memory.py         -- audit what the agent currently remembers
└── README.md
```

## Findings

### Memory systems are easy to write and hard to operate
The first 80% of memory functionality takes a day. The last 20% —
correctness, freshness, auditability, conflict resolution — is what
production teams spend months on. Every layer (extraction, validation,
storage, retrieval, expiration) has its own failure mode, and the failures
compound across time.

### Content-only hashing misses indexing-config changes
Hashing file content alone is incorrect — if you change `CHUNK_SIZE` or
swap embedding models, hashes still match but the stored chunks are stale.
Production hashes the **(content + chunk_size + embed_model)** tuple, so
any axis of indexing change correctly marks chunks for reindexing. For
this module we ship with content-only hashing as a teaching simplification.

### RAG and episodic memory have different lifecycle semantics
RAG sources are external (files), mutable, and need full lifecycle:
detect, add, update, delete. Episodes are internal (agent-created),
immutable, and either kept or evicted by policy. Same vector store, very
different write rules.

### LLM-based extraction needs deterministic validation
We saw an LLM extractor classify *"I love pineapple on pizza"* as a diet
preference and *"Belem, Sintra"* as a location. The fix isn't a better
prompt — small models reliably make these substitutions regardless of
prompting. The architectural fix is: **smartness suggests, dumbness
validates.** The LLM proposes extracted facts; deterministic code (regex,
allowlists, type checks) approves them before they reach storage.

### Hallucination by substitution applies at every layer
We met this pattern in Module 3 (manager dispatching on NASA when asked
about Mars) and again here in two places:
- LLM extractor turning *"I love pineapple on pizza"* into a diet
- Summarizer attributing assistant-spoken claims to the user

Small models substitute rather than refuse, at every layer they're used.

### The factual-refusal pattern
Hard refusal when context is empty is the single highest-ROI hallucination
mitigation. Don't let the LLM near a factual question that has no
retrieved context — classify first, refuse cleanly. This is the cheapest
guardrail in the entire production playbook.

### Memory rot compounds silently
One bad write becomes many bad retrievals. Episode summaries quote each
other transitively, and over time the agent's memory diverges from reality.
**User-visible memory inspection** (our `06_inspect_memory.py`) is essential
production UX — users need to see and correct what the system thinks it
knows. ChatGPT, Claude, and Letta all expose memory because they have to.

## What we deliberately didn't do

- **Production vector DBs (Pinecone, Weaviate, Qdrant).** ChromaDB is fine
  for learning; production choices are an ops decision, not an architectural
  one.
- **Advanced retrieval** (hybrid search, reranking, query rewriting) —
  Module 6 revisits these with evaluation tooling.
- **Recency-weighted retrieval and conflict resolution** — surfaced in
  findings, deferred to Module 6.
- **Multimodal RAG** — PDFs, images, audio. Same architectural pattern
  with a different embedding model; out of scope here.

## Run it

```bash
cd module4_memory_rag

# Index the knowledge base (idempotent — skips unchanged files)
python 01_indexer.py

# Test episodic recall across 3 sessions
python 04_test_episodic.py

# Test RAG retrieval and factual refusal
python 05_test_rag.py

# Audit what the agent remembers
python 06_inspect_memory.py
```

Memory persists across runs in `chroma_db/` and `semantic_memory.db`.
Wipe them for a clean state:
```powershell
Remove-Item semantic_memory.db -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force chroma_db -ErrorAction SilentlyContinue
```

## What's next

Module 5 introduces **production architecture** — async, streaming,
caching, retry policies, and orchestration patterns that turn lab-grade
agents into systems that can handle load.