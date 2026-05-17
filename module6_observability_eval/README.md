# Module 6 — Observability and Evaluation

A four-metric evaluation harness for the Module 4 RAG system, plus
LangSmith tracing across the whole repo. The shift from previous modules:
this one is about *measuring* what we built, not building more.

## The shift this module makes

Previous modules optimized for "does it run?" This one optimizes for "does
it produce good outputs at scale, and would I notice if it stopped?" Two
disciplines bridge that gap:

- **Observability** — what is the system doing right now? Implemented
  via LangSmith tracing.
- **Evaluation** — how well does it perform across many cases? Implemented
  via a four-metric harness with an explicit eval dataset.

Production AI teams spend more time on these than on the agents themselves,
because without them, every prompt tweak is a blind change.

## What's in this module

- **LangSmith integration** — every LangChain/LangGraph call in this repo
  becomes a trace, viewable in a web UI. Zero code changes; controlled
  entirely by environment variables.
- **Eval dataset** — 20 unlabeled + 10 labeled cases covering the Module 4
  knowledge base, including out-of-knowledge cases that should produce
  refusals rather than hallucinations.
- **Four RAGAS-equivalent metrics, implemented manually**:
  - *Faithfulness* (LLM-as-judge): is the answer supported by retrieved context?
  - *Answer relevance* (LLM-as-judge): does the answer address the question?
  - *Context precision* (LLM-as-judge): are the retrieved chunks actually relevant?
  - *Context recall* (deterministic, labeled ground truth): were the right chunks retrieved?
- **A fifth deterministic metric** (`cites_source`) — demonstrates that
  deterministic checks should always be preferred over LLM-judge checks
  when the property is checkable.
- **Regression test** — deliberately degrades the agent's system prompt
  and measures the resulting metric drop, validating that the harness
  catches quality regressions.
- **RAGAS comparison setup** — code to run the same eval through RAGAS
  itself, for comparing manual scores against the library.

## Files

```
module6_observability_eval/
├── 01_eval_dataset.py       -- 20 unlabeled + 10 labeled cases
├── 02_metrics.py            -- four RAGAS metrics + deterministic citation check
├── 03_test_harness.py       -- runs agent over dataset, aggregates scores
├── 04_regression_test.py    -- baseline vs degraded-prompt comparison
├── 05_ragas_comparison.py   -- RAGAS side-by-side runner
└── README.md
```

## Findings

### LangSmith tracing is the most valuable observability tool for this work
Once env vars are set (`LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2=true`),
every LLM call, tool invocation, retrieval, and graph node from any module
becomes a viewable, queryable trace — no code changes required. Spans nest
correctly: an agent run becomes a tree of LLM calls, tool calls, and
retrievals with per-step timing. **Bugs crash; quality regressions just
produce slightly worse outputs across all traffic — tracing is the only
way to find the second kind.**

### LLM-as-judge has known biases; mitigate by design
Three biases worth naming:
- **Position bias** — judges prefer the first answer when comparing two.
  Mitigate: randomize order in pairwise comparisons.
- **Self-preference** — a model judging its own outputs scores them
  higher. Mitigate: use a different model class (or temperature) for the
  judge than for the generator. This module uses `light` role for judges
  while agents use `heavy`.
- **Verbosity bias** — judges prefer longer, more confident answers
  regardless of correctness. Mitigate: explicit rubrics that include
  conciseness as a positive, not a negative.

LLM-as-judge scores are noisy. Trust trends over individual points.

### Deterministic metrics beat LLM-judge metrics when possible
The `cites_source` deterministic metric runs for free, never disagrees with
itself, and reliably catches answers that fail to attribute sources.
LLM-judge metrics are necessary for properties that can't be checked
deterministically (faithfulness, relevance) but always prefer a deterministic
check first.

### The eval harness is the value; the specific scores are domain-tunable
Different domains need different rubrics. A medical RAG system needs
stricter faithfulness scoring than a brainstorming assistant. The metric
*implementations* in this module are starting points; in production each
team adapts the rubrics to their domain. **Building the harness once
means future prompt changes get caught automatically.**

### Out-of-knowledge cases are first-class
~15% of the eval dataset is out-of-knowledge — questions whose correct
answer is "I don't have that information." Faithfulness scoring treats
refusal-against-empty-context as score 1.0, and context-recall treats
correctly-retrieving-nothing as 1.0. Production eval datasets need these
cases or you over-optimize for hallucination at the cost of refusal
correctness.

### Free-tier LLM quotas constrain eval, not architecture
The full eval harness makes ~80 LLM calls per run (20 cases × 4 metrics).
This burns through Gemini's free RPD limits within a few runs if run
repeatedly. Production teams pay for higher rate limits, run evals less
frequently, or use small local models with the trade-off of noisier scores.
This module's harness is designed to run on whichever provider you have —
the metrics are framework-agnostic.

### Challenges designed but not exhaustively measured
Three challenges from this module's design surface specific patterns that
the harness was built to enable. Documenting their intended outcomes here
is more honest than running them on noisy local-model data:

- **Position-bias variance test** — running the same `faithfulness` check
  10 times on the same input would surface judge-noise variance (expected:
  ±0.1 standard deviation on a 0-1 scale for small judges). Production
  teams measure this once to establish the noise floor; score changes
  below the noise floor aren't real signal.
- **Additional RAGAS metrics** — `aspect_critic` and `answer_correctness`
  exist in the RAGAS library and add value when ground-truth answers are
  available. They follow the same LLM-as-judge pattern documented here.
- **Pairwise comparison eval** — instead of scoring each answer in isolation,
  comparing two agent versions head-to-head (with position randomization) is
  often more reliable. Useful for A/B testing prompt changes.

## What we deliberately didn't do

- **Online eval** — judging answers in real-time as users interact.
  Offline eval against a curated dataset is the workhorse; online is the
  next step up, requires more infrastructure.
- **Human evaluation UIs** — Label Studio, Argilla, LangSmith's annotation
  interface. Useful for building golden datasets; not built here.
- **Full RAGAS pipeline** — implementing the metrics manually was the
  educational goal. `pip install ragas` and call `ragas.evaluate(...)`
  when you want it in production tomorrow.

## Run it

```bash
cd module6_observability_eval

# Quick sanity check after setting LangSmith env vars
python -c "from langsmith import Client; print(Client().list_runs(limit=1))"

# Full eval (≈80 LLM calls, ~5 min on Gemini Flash-Lite)
python 03_test_harness.py

# Regression detection (runs eval twice, compares scores)
python 04_regression_test.py

# RAGAS comparison (small subset to keep token costs sane)
python 05_ragas_comparison.py
```

Requires:
- `LANGSMITH_API_KEY` + `LANGCHAIN_TRACING_V2=true` in `.env` (free at smith.langchain.com)
- The Module 4 knowledge base (`module4_memory_rag/knowledge_base/`)
- The Module 5 async agent

## What's next

Module 7 introduces **governance and guardrails** — prompt injection,
output validation, PII handling, OWASP LLM Top 10. The defensive side of
what evaluation measures.