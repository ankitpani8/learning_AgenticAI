"""Eval dataset for the Module 4 knowledge base.

Two parts:
  - cases_unlabeled: just (question, expected_topic) - covers faithfulness,
    relevance, and context-precision (which is LLM-judged)
  - cases_labeled: (question, expected_chunk_doc_ids) - covers context-recall,
    which needs ground-truth on what should be retrieved

The split is deliberate: labeling is expensive. We label only the cases where
context-recall matters and lean on LLM-judges for the rest. This is exactly
how production eval datasets are built -- start small, label the highest-value
subset, judge-grade the rest.
"""

# 20 unlabeled cases covering the knowledge base
CASES_UNLABELED = [
    {"question": "How should I manage Python dependencies?",
     "expected_topic": "python_best_practices"},
    {"question": "Why use virtual environments?",
     "expected_topic": "python_best_practices"},
    {"question": "What does 'pinning dependencies' mean?",
     "expected_topic": "python_best_practices"},
    {"question": "Should I catch bare exceptions in Python?",
     "expected_topic": "python_best_practices"},
    {"question": "Why use type hints?",
     "expected_topic": "python_best_practices"},

    {"question": "What is the ReAct loop?",
     "expected_topic": "agentic_ai_patterns"},
    {"question": "What multi-agent patterns exist?",
     "expected_topic": "agentic_ai_patterns"},
    {"question": "How does token cost scale across agent turns?",
     "expected_topic": "agentic_ai_patterns"},
    {"question": "What model should I use for a critic agent?",
     "expected_topic": "agentic_ai_patterns"},
    {"question": "What are the five canonical multi-agent topologies?",
     "expected_topic": "agentic_ai_patterns"},

    {"question": "What's the rate limit for Gemini Flash on the free tier?",
     "expected_topic": "llm_provider_notes"},
    {"question": "Does a claude.ai subscription include API access?",
     "expected_topic": "llm_provider_notes"},
    {"question": "Why doesn't Ollama parallelize requests?",
     "expected_topic": "llm_provider_notes"},
    {"question": "When does Gemini's daily quota reset?",
     "expected_topic": "llm_provider_notes"},
    {"question": "Which provider has the strongest tool-calling reliability?",
     "expected_topic": "llm_provider_notes"},

    # Cross-document queries (harder)
    {"question": "I'm new to Python and want to build agents. Where should I start?",
     "expected_topic": "mixed"},
    {"question": "Are there cost considerations when designing a multi-agent system?",
     "expected_topic": "agentic_ai_patterns"},

    # Out-of-knowledge cases (agent should refuse, not hallucinate)
    {"question": "What's the current GDP of France?",
     "expected_topic": "none"},
    {"question": "How does Kubernetes handle pod failures?",
     "expected_topic": "none"},
    {"question": "Who won the 2024 US presidential election?",
     "expected_topic": "none"},
]

# 10 cases with ground-truth labels on which doc_ids SHOULD be retrieved.
# This is the labeled subset for context-recall scoring.
CASES_LABELED = [
    {"question": "How should I manage Python dependencies?",
     "relevant_doc_ids": ["python_best_practices.md"]},
    {"question": "Why use virtual environments?",
     "relevant_doc_ids": ["python_best_practices.md"]},
    {"question": "What is the ReAct loop?",
     "relevant_doc_ids": ["agentic_ai_patterns.md"]},
    {"question": "What multi-agent patterns exist?",
     "relevant_doc_ids": ["agentic_ai_patterns.md"]},
    {"question": "What's the rate limit for Gemini Flash on the free tier?",
     "relevant_doc_ids": ["llm_provider_notes.md"]},
    {"question": "Does a claude.ai subscription include API access?",
     "relevant_doc_ids": ["llm_provider_notes.md"]},
    {"question": "Why use small models for critics?",
     "relevant_doc_ids": ["agentic_ai_patterns.md"]},
    # Cross-doc cases: multiple docs are relevant
    {"question": "I'm new to Python and want to build agents. Where should I start?",
     "relevant_doc_ids": ["python_best_practices.md", "agentic_ai_patterns.md"]},
    {"question": "What costs should I plan for in an agent system?",
     "relevant_doc_ids": ["agentic_ai_patterns.md", "llm_provider_notes.md"]},
    # Out-of-knowledge: nothing should be retrieved
    {"question": "What's the current GDP of France?",
     "relevant_doc_ids": []},
]