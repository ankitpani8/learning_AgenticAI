"""Four RAGAS-equivalent metrics implemented manually.

faithfulness          -- is the answer grounded in the retrieved context?
answer_relevance      -- does the answer address the question?
context_precision     -- are the retrieved chunks actually relevant?
context_recall        -- were the chunks that *should* be retrieved actually retrieved?

The first three use LLM-as-judge. The fourth uses ground-truth labels.

LLM-as-judge prompts intentionally:
  - Ask for structured output (JSON with score + reasoning)
  - Use a different model than the one that generated the answer (we use 'light'
    here while the agent uses 'heavy') -- mitigates self-preference bias
  - Use rubrics, not vibes ("score 0-1 where 1 = every claim is supported")
"""
import json
import sys
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.providers import select_all_models

SELECTIONS = select_all_models(roles=["light"])
judge = SELECTIONS["light"].to_langchain(temperature=0)


def _parse_json(raw: str) -> dict | None:
    """LLMs return JSON inside code fences sometimes. Strip and parse."""
    raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# --- 1. Faithfulness ------------------------------------------------------

FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to its context.

An answer is FAITHFUL if every factual claim in it is supported by the context.
An answer is UNFAITHFUL if it contains claims that go beyond, contradict, or
invent details not present in the context.

If the answer is a refusal ("I don't know"), score it 1.0 -- refusing when you
don't have grounding is the correct behavior.

Question: {question}

Context:
{context}

Answer to evaluate:
{answer}

Output JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


def faithfulness(question: str, answer: str, context: str) -> dict:
    if not context.strip():
        # Refusal-against-empty-context is correct behavior
        is_refusal = any(p in answer.lower() for p in
                         ["don't have", "do not have", "cannot find", "no information"])
        return {"score": 1.0 if is_refusal else 0.0,
                "reasoning": "refusal-correct" if is_refusal else "no context but answered"}

    prompt = FAITHFULNESS_PROMPT.format(
        question=question, answer=answer, context=context
    )
    response = judge.invoke([HumanMessage(content=prompt)])
    parsed = _parse_json(response.content) or {"score": 0.0, "reasoning": "parse failed"}
    return parsed


# --- 2. Answer relevance --------------------------------------------------

RELEVANCE_PROMPT = """You are evaluating whether an answer addresses the user's question.

An answer is RELEVANT if it directly addresses what was asked.
An answer is IRRELEVANT if it goes off-topic, answers a different question, or
has only unrelated information.

A refusal IS relevant if the question is genuinely outside the system's knowledge --
saying "I don't know" to an out-of-scope question is the correct relevant answer.

Question: {question}

Answer to evaluate:
{answer}

Output JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}"""


def answer_relevance(question: str, answer: str) -> dict:
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    response = judge.invoke([HumanMessage(content=prompt)])
    parsed = _parse_json(response.content) or {"score": 0.0, "reasoning": "parse failed"}
    return parsed


# --- 3. Context precision (LLM-judged, no labels needed) -------------------

PRECISION_PROMPT = """You are evaluating whether a retrieved chunk is relevant to a question.

A chunk is RELEVANT if it contains information that helps answer the question.
A chunk is IRRELEVANT if it's about a different topic, even if the chunk is from
the same general subject area.

Question: {question}

Chunk:
{chunk}

Output JSON only:
{{"relevant": <true|false>, "reasoning": "<one sentence>"}}"""


def context_precision(question: str, retrieved_chunks: list[dict]) -> dict:
    """Of the retrieved chunks, what fraction are actually relevant?"""
    if not retrieved_chunks:
        # No chunks retrieved -- precision is undefined; treat as 1.0 if the
        # question is OOK (handled by the agent's refusal path), else 0.0.
        # We can't tell here, so report N/A.
        return {"score": None, "reasoning": "no chunks retrieved",
                "details": {"relevant": 0, "total": 0}}

    relevant_count = 0
    judgments = []
    for chunk in retrieved_chunks:
        prompt = PRECISION_PROMPT.format(question=question, chunk=chunk["text"])
        response = judge.invoke([HumanMessage(content=prompt)])
        parsed = _parse_json(response.content) or {"relevant": False, "reasoning": "parse failed"}
        judgments.append(parsed)
        if parsed.get("relevant"):
            relevant_count += 1

    score = relevant_count / len(retrieved_chunks)
    return {
        "score": score,
        "reasoning": f"{relevant_count}/{len(retrieved_chunks)} chunks relevant",
        "details": {"relevant": relevant_count, "total": len(retrieved_chunks),
                    "judgments": judgments},
    }


# --- 4. Context recall (uses ground-truth labels) -------------------------

def context_recall(retrieved_chunks: list[dict], relevant_doc_ids: list[str]) -> dict:
    """Of the doc_ids that SHOULD have been retrieved, what fraction were?

    Unlike the other three, this is deterministic -- no LLM judge needed.
    It compares retrieved doc_ids against the labeled ground truth.
    """
    if not relevant_doc_ids:
        # Out-of-knowledge case: nothing should be retrieved.
        # If nothing was retrieved, recall is 1.0 (correctly found nothing).
        # If something was retrieved, recall is undefined -- we report 0.0 to
        # penalize the false-positive case (this is a judgment call).
        if not retrieved_chunks:
            return {"score": 1.0, "reasoning": "correctly retrieved nothing"}
        return {"score": 0.0,
                "reasoning": f"should have retrieved nothing, got {len(retrieved_chunks)} chunks"}

    retrieved_doc_ids = {c["meta"]["doc_id"] for c in retrieved_chunks}
    relevant_set = set(relevant_doc_ids)
    found = relevant_set & retrieved_doc_ids
    score = len(found) / len(relevant_set)
    return {
        "score": score,
        "reasoning": f"retrieved {len(found)}/{len(relevant_set)} expected docs",
        "details": {"expected": list(relevant_set), "retrieved": list(retrieved_doc_ids),
                    "found": list(found)},
    }

# --- 5. Citation check (deterministic, no LLM judge) ----------------------

def cites_source(answer: str, available_docs: list[str]) -> dict:
    """Does the answer mention any source document by name?

    Deterministic metrics like this are always cheaper and more reliable
    than LLM-judge metrics. Use them whenever the property is checkable.
    """
    answer_lower = answer.lower()
    matched = [d for d in available_docs if d.replace(".md", "").replace("_", " ") in answer_lower]
    return {
        "score": 1.0 if matched else 0.0,
        "reasoning": f"matched docs: {matched}" if matched else "no source mention",
    }