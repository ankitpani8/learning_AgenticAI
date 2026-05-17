"""Regression detection: degrade the agent's prompt, watch metrics drop.

This is the entire point of having an eval harness -- catching quality
regressions when you make a change. Without the harness, this kind of
silent quality drop is invisible until users complain.
"""
import asyncio
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "module5_production"))
_agent_module = importlib.import_module("04_agent")
_harness = importlib.import_module("harness")
_eval = importlib.import_module("eval_dataset")


async def main():
    # Baseline run with normal prompt
    print("\n### BASELINE ###")
    original_prompt = _agent_module.SYSTEM_PROMPT
    baseline = await _harness.run_eval(
        _eval.CASES_UNLABELED[:10], _eval.CASES_LABELED[:5],
        out_path="baseline_report.json"
    )

    # Degraded run with intentionally bad prompt
    print("\n\n### DEGRADED ###")
    _agent_module.SYSTEM_PROMPT = (
        "You are an assistant. Answer the question with confidence. Use your "
        "general knowledge freely. Do not say 'I don't know' -- always provide "
        "an informative answer even if the context doesn't directly cover it."
    )
    degraded = await _harness.run_eval(
        _eval.CASES_UNLABELED[:10], _eval.CASES_LABELED[:5],
        out_path="degraded_report.json"
    )

    # Compare
    print("\n\n### REGRESSION REPORT ###")
    print(f"{'metric':<40s} {'baseline':>10s} {'degraded':>10s} {'change':>10s}")
    for metric in ["mean_faithfulness", "mean_answer_relevance",
                   "mean_context_precision", "mean_context_recall_labeled_only"]:
        b = baseline["summary"][metric] or 0
        d = degraded["summary"][metric] or 0
        change = d - b
        flag = "⚠ REGRESSION" if change < -0.05 else "ok"
        print(f"{metric:<40s} {b:>10.3f} {d:>10.3f} {change:>+10.3f}  {flag}")

    # Restore prompt
    _agent_module.SYSTEM_PROMPT = original_prompt


if __name__ == "__main__":
    asyncio.run(main())