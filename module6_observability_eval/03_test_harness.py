"""Eval harness: run agent over dataset, compute four metrics, report.

Reuses Module 4's retrieval and Module 5's async agent. Each run becomes a
trace in LangSmith (auto, via env vars). The metrics scores are aggregated
across the dataset for the final report."""
import asyncio
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "module4_memory_rag"))
sys.path.insert(0, str(Path(__file__).parent.parent / "module5_production"))

# Reuse Module 4's RAG indexerand Module 5's async RAG agent
_indexer = importlib.import_module("01_indexer")
_agent = importlib.import_module("04_agent_RAG")
_eval = importlib.import_module("01_eval_dataset")
_metrics = importlib.import_module("02_metrics")


def retrieve_for_eval(question: str, k: int = 3) -> tuple[list[dict], str]:
    """Wrap retrieval, also return the context string used for the answer."""
    chunks = _indexer.retrieve(question, k=k, score_threshold=0.7)
    if not chunks:
        return [], ""
    context = "\n\n".join(
        f"From {c['meta']['doc_id']}:\n{c['text']}" for c in chunks
    )
    return chunks, context


async def evaluate_one(case: dict, with_recall: bool = False) -> dict:
    """Run agent, score one case across all applicable metrics."""
    question = case["question"]
    print(f"  evaluating: {question[:60]}")

    # Retrieve + answer using the production async agent
    # We need the retrieved chunks separately for precision/recall, so we
    # retrieve here and let the agent regenerate from context.
    chunks, context = retrieve_for_eval(question)
    answer = await _agent.answer_async(question)

    result = {
        "question": question,
        "answer": answer[:500],  # truncate for report readability
        "retrieved_doc_ids": [c["meta"]["doc_id"] for c in chunks],
        "metrics": {
            "faithfulness": _metrics.faithfulness(question, answer, context),
            "answer_relevance": _metrics.answer_relevance(question, answer),
            "context_precision": _metrics.context_precision(question, chunks),
        },
    }
    if with_recall and "relevant_doc_ids" in case:
        result["metrics"]["context_recall"] = _metrics.context_recall(
            chunks, case["relevant_doc_ids"]
        )
    return result


async def run_eval(
    cases_unlabeled: list[dict],
    cases_labeled: list[dict],
    out_path: str = "eval_report.json",
) -> dict:
    print("=" * 60)
    print(f"EVAL RUN: {len(cases_unlabeled)} unlabeled + {len(cases_labeled)} labeled")
    print("=" * 60)

    # Run unlabeled cases (faithfulness, relevance, precision)
    print("\n[1/2] Running unlabeled cases...")
    unlabeled_results = []
    for case in cases_unlabeled:
        result = await evaluate_one(case, with_recall=False)
        unlabeled_results.append(result)

    # Run labeled cases (all four metrics including recall)
    print(f"\n[2/2] Running labeled cases...")
    labeled_results = []
    for case in cases_labeled:
        result = await evaluate_one(case, with_recall=True)
        labeled_results.append(result)

    # Aggregate
    def mean_score(results, metric_name):
        scores = [
            r["metrics"][metric_name]["score"]
            for r in results
            if metric_name in r["metrics"]
            and r["metrics"][metric_name]["score"] is not None
        ]
        return statistics.mean(scores) if scores else None

    all_results = unlabeled_results + labeled_results
    summary = {
        "n_cases": len(all_results),
        "mean_faithfulness": mean_score(all_results, "faithfulness"),
        "mean_answer_relevance": mean_score(all_results, "answer_relevance"),
        "mean_context_precision": mean_score(all_results, "context_precision"),
        "mean_context_recall_labeled_only": mean_score(labeled_results, "context_recall"),
        "timestamp": time.time(),
    }

    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:40s} {v:.3f}")
        else:
            print(f"  {k:40s} {v}")

    # Write full report
    report = {"summary": summary,
              "unlabeled": unlabeled_results,
              "labeled": labeled_results}
    Path(out_path).write_text(json.dumps(report, indent=2))
    print(f"\n[report] written to {out_path}")
    return report


if __name__ == "__main__":
    _indexer.reindex_knowledge_base(verbose=False)
    # asyncio.run(run_eval(_eval.CASES_UNLABELED, _eval.CASES_LABELED))
    asyncio.run(run_eval(_eval.CASES_UNLABELED[:5], _eval.CASES_LABELED[:3]))