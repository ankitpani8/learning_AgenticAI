"""Run RAGAS on the same eval set, compare to our manual implementation.

If our numbers are roughly aligned, we understood the metrics correctly. If
they diverge significantly, examine WHY -- usually a rubric difference or
a prompt-engineering choice. Either way, you'll know what RAGAS does
internally and how to debug its scores when they look wrong.
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.providers import select_all_models

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness as ragas_faithfulness,
    answer_relevancy as ragas_relevance,
    context_precision as ragas_precision,
    context_recall as ragas_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from datasets import Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "module4_memory_rag"))
sys.path.insert(0, str(Path(__file__).parent.parent / "module5_production"))
_indexer = importlib.import_module("01_indexer")
_agent = importlib.import_module("04_agent")
_eval = importlib.import_module("eval_dataset")


# RAGAS needs LangChain-wrapped LLM and embeddings
SELECTIONS = select_all_models(roles=["light"])
ragas_llm = LangchainLLMWrapper(SELECTIONS["light"].to_langchain(temperature=0))


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


ragas_embeddings = LangchainEmbeddingsWrapper(LocalEmbeddings())


async def build_ragas_dataset(cases):
    _indexer.reindex_knowledge_base(verbose=False)
    records = []
    for case in cases:
        question = case["question"]
        chunks = _indexer.retrieve(question, k=3, score_threshold=0.7)
        contexts = [c["text"] for c in chunks]
        answer = await _agent.answer_async(question)
        records.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": case.get("relevant_doc_ids", [""])[0] or "no information available",
        })
    return Dataset.from_list(records)


async def main():
    import asyncio
    # Use a small subset to keep token costs sane
    cases = _eval.CASES_LABELED[:5]
    print("Building RAGAS dataset...")
    ds = await build_ragas_dataset(cases)

    print("\nRunning RAGAS evaluation (this calls the judge LLM many times)...")
    result = evaluate(
        ds,
        metrics=[ragas_faithfulness, ragas_relevance, ragas_precision, ragas_recall],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    print("\n=== RAGAS SCORES ===")
    print(result)
    print("\nCompare against our manual eval (harness.py output) on the same cases.")
    print("Differences are usually due to: judge model, prompt rubric phrasing,")
    print("and how each library handles edge cases (empty context, refusals).")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())