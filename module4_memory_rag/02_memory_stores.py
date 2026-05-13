"""Long-term memory: semantic facts (SQLite) and episodic memories (ChromaDB).

Semantic memory = key-value facts about the user, retrieved by key.
    "user_diet" -> "vegetarian"
    "user_name" -> "Ankit"

Episodic memory = embedded summaries of past conversations, retrieved by
similarity. Lets the agent answer "remind me what we discussed about X?"
"""
import sqlite3
import time
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

MODULE_DIR = Path(__file__).parent
SQLITE_PATH = MODULE_DIR / "semantic_memory.db"
CHROMA_DIR = MODULE_DIR / "chroma_db"
EPISODIC_COLLECTION = "episodic_memory"

EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# --- Semantic memory (SQLite key-value) -----------------------------------

def _semantic_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
    """)
    return conn


def set_fact(key: str, value: str) -> None:
    with _semantic_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO facts (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )


def get_fact(key: str) -> str | None:
    with _semantic_conn() as conn:
        row = conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def all_facts() -> dict[str, str]:
    with _semantic_conn() as conn:
        rows = conn.execute("SELECT key, value FROM facts").fetchall()
    return {k: v for k, v in rows}


# --- Episodic memory (ChromaDB) -------------------------------------------

def _episodic_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=EPISODIC_COLLECTION,
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def record_episode(summary: str, session_id: str) -> None:
    """Store a summary of a conversation, retrievable by similarity later."""
    coll = _episodic_collection()
    ts = time.time()
    coll.add(
        ids=[f"ep_{ts}"],
        documents=[summary],
        metadatas=[{"session_id": session_id, "timestamp": ts}],
    )


def retrieve_episodes(query: str, k: int = 2, threshold: float = 0.9) -> list[dict]:
    coll = _episodic_collection()
    if coll.count() == 0:
        return []
    result = coll.query(query_texts=[query], n_results=min(k, coll.count()))
    episodes = []

    for ep_id, doc, meta, dist in zip(
        result["ids"][0], result["documents"][0],
        result["metadatas"][0], result["distances"][0],
    ):
        if dist > threshold:
            continue
        episodes.append({"summary": doc, "meta": meta, "distance": dist})
    return episodes

# --- Inspection (auditability) --------------------------------------------

def inspect_memory(verbose: bool = True) -> dict:
    """Print and return everything in long-term memory.

    Production assistants expose this to the user as 'what do you remember
    about me?' UX. Auditability is essential when memory rot accumulates --
    users need to see and correct what the system thinks it knows.
    """
    facts = all_facts()

    coll = _episodic_collection()
    if coll.count() > 0:
        ep_data = coll.get(include=["documents", "metadatas"])
        episodes = [
            {
                "id": eid,
                "summary": doc,
                "session_id": meta.get("session_id"),
                "timestamp": meta.get("timestamp"),
            }
            for eid, doc, meta in zip(
                ep_data["ids"], ep_data["documents"], ep_data["metadatas"]
            )
        ]
        episodes.sort(key=lambda e: e["timestamp"] or 0, reverse=True)
    else:
        episodes = []

    if verbose:
        print("\n" + "=" * 60)
        print("MEMORY INSPECTION")
        print("=" * 60)
        print(f"\nSemantic facts ({len(facts)}):")
        for k, v in facts.items():
            print(f"  {k}: {v}")
        print(f"\nEpisodic memories ({len(episodes)}):")
        for ep in episodes:
            import datetime
            ts = datetime.datetime.fromtimestamp(ep["timestamp"]).strftime("%Y-%m-%d %H:%M")
            print(f"  [{ts}] (session={ep['session_id']})")
            print(f"    {ep['summary']}")
        print("=" * 60)

    return {"facts": facts, "episodes": episodes}


def forget_fact(key: str) -> bool:
    """Delete a single fact. Returns True if it existed."""
    with _semantic_conn() as conn:
        cur = conn.execute("DELETE FROM facts WHERE key=?", (key,))
        return cur.rowcount > 0


def forget_episode(episode_id: str) -> None:
    """Delete a single episodic memory by ID."""
    _episodic_collection().delete(ids=[episode_id])