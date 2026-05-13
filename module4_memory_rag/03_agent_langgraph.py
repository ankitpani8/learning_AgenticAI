"""Memory-augmented assistant with three memory layers + RAG.

Layers:
  1. Conversation memory   -- LangGraph MemorySaver (per session)
  2. Semantic memory       -- SQLite key-value facts (cross-session)
  3. Episodic memory       -- ChromaDB embedded summaries (cross-session)
  4. RAG                   -- ChromaDB over the knowledge base

The agent decides each turn what to retrieve. Routing is keyword-based for
clarity; production systems use an LLM-based router or always-retrieve
patterns with reranking.
"""
import sys
import re
from pathlib import Path
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.providers import select_all_models
import importlib
_indexer = importlib.import_module("01_indexer")
_memory  = importlib.import_module("02_memory_stores")

retrieve = _indexer.retrieve
reindex_knowledge_base = _indexer.reindex_knowledge_base
set_fact = _memory.set_fact
all_facts = _memory.all_facts
record_episode = _memory.record_episode
retrieve_episodes = _memory.retrieve_episodes

load_dotenv(Path(__file__).parent.parent / ".env")

SELECTIONS = select_all_models(roles=["heavy", "light"])
llm        = SELECTIONS["heavy"].to_langchain(temperature=0.3)
summarizer = SELECTIONS["light"].to_langchain(temperature=0)


# --- State ----------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    rag_context: str          # populated by RAG node when relevant
    memory_context: str       # populated by memory node
    session_id: str


# --- Memory extraction (run at end of conversation) -----------------------

import json

EXTRACTION_PROMPT = """Extract durable facts about the user from this message.

Save a fact ONLY if the user is making a definite, explicit claim about themselves.

Do NOT extract:
  - Opinions about food they like/dislike (not durable facts)
  - Travel destinations or plans (these are episodic, not facts about the user)
  - Hypotheticals, jokes, or ambiguous statements
  - Anything the user might change their mind about within a week

Output ONLY a JSON object. Use these keys:
  - user_name: explicit self-introduction with a name (e.g., "my name is X")
  - user_diet: only if they explicitly identify as vegetarian/vegan/pescatarian
  - user_location: explicit statement of where they LIVE (not where they're traveling)

If nothing qualifies, output exactly: {}

Message: {message}
"""

ALLOWED_DIETS = {"vegetarian", "vegan", "pescatarian"}

def extract_facts_from_message(text: str) -> dict[str, str]:
    """LLM extraction with deterministic validation downstream."""
    prompt = EXTRACTION_PROMPT.replace("{message}", text)
    response = summarizer.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    # Deterministic validation: the LLM suggests, this code approves.
    validated = {}
    if name := parsed.get("user_name"):
        if isinstance(name, str) and name.isalpha() and len(name) <= 30:
            validated["user_name"] = name.strip()
    if diet := parsed.get("user_diet"):
        if isinstance(diet, str) and diet.lower() in ALLOWED_DIETS:
            validated["user_diet"] = diet.lower()
    if loc := parsed.get("user_location"):
        if isinstance(loc, str) and "," not in loc and len(loc) <= 40:
            validated["user_location"] = loc.strip()

    return validated

# --- Nodes ----------------------------------------------------------------

def load_memory_node(state: AgentState) -> dict:
    """Pull all semantic facts and any relevant episodic memories."""
    facts = all_facts()
    episodes = retrieve_episodes(state["user_query"], k=2)

    blocks = []
    if facts:
        blocks.append("Known facts about the user:\n" +
                      "\n".join(f"- {k}: {v}" for k, v in facts.items()))
    if episodes:
        blocks.append("Relevant past conversations:\n" +
                      "\n".join(f"- {ep['summary']}" for ep in episodes))

    context = "\n\n".join(blocks) if blocks else ""
    print(f"\n[memory] facts={len(facts)} episodes={len(episodes)}")
    return {"memory_context": context}


def rag_node(state: AgentState) -> dict:
    """Retrieve from knowledge base. Empty string if nothing relevant."""
    chunks = retrieve(state["user_query"], k=3, score_threshold=0.7)
    if not chunks:
        print("[rag] no relevant chunks")
        return {"rag_context": ""}

    print(f"[rag] retrieved {len(chunks)} chunks")
    blocks = [f"From {c['meta']['doc_id']} (relevance={1 - c['distance']:.2f}):\n{c['text']}"
              for c in chunks]
    return {"rag_context": "\n\n".join(blocks)}

def respond_node(state: AgentState) -> dict:
    """Generate the response, but refuse cleanly if asked a factual question with no context."""
    has_context = bool(state["memory_context"] or state["rag_context"])
    query = state["user_query"]

    # When we have no memory or RAG context, check if this is a factual lookup.
    # If yes, refuse cleanly rather than letting the model hallucinate.
    if not has_context:
        classification = summarizer.invoke([HumanMessage(content=(
            "Is the following message a factual question asking for specific information "
            "(facts, numbers, dates, definitions, looked-up knowledge)? "
            "Respond with ONLY 'FACTUAL' or 'CONVERSATIONAL'.\n\n"
            f"Message: {query}"
        ))]).content.strip().upper()

        if "FACTUAL" in classification:
            print("[refusal] factual question with no context")
            refusal = AIMessage(content=(
                "I don't have information about that in my knowledge base or memory. "
                "If you'd like, you can share more context, or I can help with something else."
            ))
            # Still extract facts in case the message had any
            for k, v in extract_facts_from_message(query).items():
                set_fact(k, v)
                print(f"[memory] saved fact: {k}={v}")
            return {"messages": [HumanMessage(content=query), refusal]}

    # Normal generation path
    system_parts = [
        "You are a helpful personal assistant. Be concise.",
        "If the user tells you a fact about themselves, acknowledge it briefly.",
        "Use the context below only when relevant; do not paste it verbatim.",
    ]
    if state["memory_context"]:
        system_parts.append(f"\n## Memory\n{state['memory_context']}")
    if state["rag_context"]:
        system_parts.append(f"\n## Knowledge base\n{state['rag_context']}")

    history = state["messages"] + [HumanMessage(content=query)]
    response = llm.invoke([SystemMessage(content="\n".join(system_parts))] + history)

    for k, v in extract_facts_from_message(query).items():
        set_fact(k, v)
        print(f"[memory] saved fact: {k}={v}")

    return {"messages": [HumanMessage(content=query), response]}


# --- Graph ----------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_memory", load_memory_node)
    g.add_node("rag", rag_node)
    g.add_node("respond", respond_node)
    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "rag")
    g.add_edge("rag", "respond")
    g.add_edge("respond", END)
    return g.compile(checkpointer=MemorySaver())


# --- Conversation runner --------------------------------------------------

IMPORTANCE_PROMPT = """Should this conversation be saved as long-term memory?

Save (YES) if the user shared:
  - Specific plans, dates, or decisions
  - Personal facts or preferences not in routine Q&A
  - Goals, commitments, or things they want remembered

Do NOT save (NO) if it's:
  - Routine technical Q&A
  - General assistance without personal context
  - The assistant explaining things without user-specific information

Respond with only YES or NO.

Transcript:
{transcript}
"""

def end_session(messages, session_id):
    if len(messages) < 2:
        return

    transcript = "\n".join(
        f"{'user' if isinstance(m, HumanMessage) else 'assistant'}: {m.content}"
        for m in messages
    )

    decision = summarizer.invoke([
        HumanMessage(content=IMPORTANCE_PROMPT.format(transcript=transcript))
    ]).content.upper()

    if "YES" not in decision:
        print("[episodic] skipped (low importance)")
        return

    summary_prompt = [
        SystemMessage(content=(
            "Summarize this conversation in 1-2 sentences. Focus on facts the "
            "user shared, plans they made, or topics discussed. Be specific."
        )),
        HumanMessage(content=transcript),
    ]
    summary = summarizer.invoke(summary_prompt).content.strip()
    record_episode(summary, session_id)
    print(f"[episodic] saved: {summary}")


def run_session(session_id: str, queries: list[str]):
    """Run a sequence of queries within one session."""
    app = build_graph()
    config = {"configurable": {"thread_id": session_id}}
    print(f"\n{'=' * 60}\nSESSION: {session_id}\n{'=' * 60}")
    final_messages = []

    for q in queries:
        print(f"\n>>> USER: {q}")
        result = app.invoke({
            "messages": [],
            "user_query": q,
            "rag_context": "",
            "memory_context": "",
            "session_id": session_id,
        }, config=config)
        final_messages = result["messages"]
        ai_msg = result["messages"][-1]
        print(f"<<< ASSISTANT: {ai_msg.content}")

    end_session(final_messages, session_id)


if __name__ == "__main__":
    # Ensure knowledge base is indexed
    print("=== Ensuring KB is indexed ===")
    reindex_knowledge_base(verbose=False)