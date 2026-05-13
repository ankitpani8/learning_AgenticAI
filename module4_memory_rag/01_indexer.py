"""Content-hash incremental indexer for the knowledge base.

For each file in the source folder:
  - Compute SHA-256 of contents
  - If unchanged since last index, skip
  - If changed or new, delete old chunks and reindex
  - If a previously-indexed file no longer exists, delete its chunks

This skips the expensive embedding step for unchanged files while keeping
the vector store in sync with the source folder.
"""
import hashlib
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import os
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# --- Configuration --------------------------------------------------------

CHROMA_DIR = Path(__file__).parent / "chroma_db"
KB_DIR = Path(__file__).parent / "knowledge_base"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 400      # chars per chunk -- illustrative (but naive if we want good retrieval quality)
# CHUNK_SIZE = 50          
CHUNK_OVERLAP = 80

# Local embedding model -- runs on CPU, no API key needed
EMBED_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


# --- Chunking -------------------------------------------------------------

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks by character count.

    Naive but transparent. Real systems chunk by structure (paragraphs,
    sections, sentences) -- Module 6 revisits this.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start = end - overlap if end < len(text) else end
    return chunks


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# --- Indexer --------------------------------------------------------------

def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBED_FN,
        metadata={"hnsw:space": "cosine"},
    )


def reindex_knowledge_base(verbose: bool = True) -> dict:
    """Sync the vector store with the source folder using content hashes."""
    coll = get_collection()
    stats = {"new": 0, "changed": 0, "unchanged": 0, "deleted": 0}

    # 1. Build map of currently-indexed docs and their stored hashes.
    existing = coll.get(include=["metadatas"])
    indexed_docs = {}  # doc_id -> stored_hash
    for chunk_id, meta in zip(existing["ids"], existing["metadatas"]):
        doc_id = meta["doc_id"]
        indexed_docs.setdefault(doc_id, meta["content_hash"])

    # 2. Walk source folder, decide action per file.
    source_doc_ids = set()
    for md_file in sorted(KB_DIR.glob("*.md")):
        doc_id = md_file.name
        source_doc_ids.add(doc_id)
        content = md_file.read_text(encoding="utf-8")
        new_hash = sha256(content)

        if doc_id not in indexed_docs:
            action = "new"
        elif indexed_docs[doc_id] != new_hash:
            action = "changed"
        else:
            action = "unchanged"

        if action == "unchanged":
            stats["unchanged"] += 1
            if verbose:
                print(f"  [skip]    {doc_id} (unchanged)")
            continue

        # Delete old chunks for this doc (if any)
        if action == "changed":
            old_ids = [cid for cid, m in zip(existing["ids"], existing["metadatas"])
                       if m["doc_id"] == doc_id]
            if old_ids:
                coll.delete(ids=old_ids)

        # Chunk + add
        chunks = chunk_text(content)
        chunk_ids = [f"{doc_id}::chunk_{i}" for i in range(len(chunks))]
        coll.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=[{"doc_id": doc_id, "content_hash": new_hash, "chunk_index": i}
                       for i in range(len(chunks))],
        )
        stats[action] += 1
        if verbose:
            print(f"  [{action:<7}] {doc_id} ({len(chunks)} chunks)")

    # 3. Delete chunks for docs that no longer exist in source.
    for doc_id in indexed_docs:
        if doc_id not in source_doc_ids:
            old_ids = [cid for cid, m in zip(existing["ids"], existing["metadatas"])
                       if m["doc_id"] == doc_id]
            if old_ids:
                coll.delete(ids=old_ids)
            stats["deleted"] += 1
            if verbose:
                print(f"  [delete]  {doc_id} (removed from source)")

    if verbose:
        print(f"\n  summary: {stats}")
    return stats


# --- Retrieval ------------------------------------------------------------

def retrieve(query: str, k: int = 3, score_threshold: float = 0.5) -> list[dict]:
    """Top-K retrieval with a relevance cutoff.

    Chroma returns cosine distance in [0, 2]; lower is more similar.
    score_threshold filters out chunks with distance > threshold.
    """
    coll = get_collection()
    result = coll.query(query_texts=[query], n_results=k)

    chunks = []
    for chunk_id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        if dist > score_threshold:
            continue
        chunks.append({"id": chunk_id, "text": doc, "meta": meta, "distance": dist})
    return chunks


if __name__ == "__main__":
    print("=== Indexing knowledge base ===")
    reindex_knowledge_base()

    print("\n=== Sample retrievals ===")
    for q in ["how do I manage Python dependencies?",
              "what are agent patterns?",
              "what's the weather like in Bengaluru?"]:
        print(f"\nQ: {q}")
        for c in retrieve(q, k=2):
            print(f"  [{c['distance']:.3f}] {c['meta']['doc_id']}: {c['text'][:80]}...")