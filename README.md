# Learning Agentic AI — From Scratch to Production

> A hands-on, module-by-module journey building production-grade AI agents.
> Each module pairs a focused tutorial with a working implementation, progressing
> from raw API calls to deployed multi-agent systems with full observability and governance.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Progress](https://img.shields.io/badge/status-in%20progress-orange.svg)]()

---

## Why This Repo

Most agent tutorials wrap LangChain around an OpenAI call and call it production.
This repo goes the other way: build the loop yourself first, then layer in frameworks,
memory, evaluation, governance, and deployment — each module solving a problem the
previous module exposed.

**Target audience:** data scientists and engineers who want to add agentic AI to their
toolkit with real depth, not buzzword-level familiarity.

---

## Curriculum Roadmap

| Module | Focus | Status |
|--------|-------|--------|
| **1. Foundations** | ReAct loop, tool calling, multi-provider fallback | ✅ Complete |
| **2. Frameworks** | LangGraph state machines, multi-agent orchestration | 🚧 Next |
| **3. Memory & RAG** | Persistent state, vector retrieval, context engineering | ⏳ Planned |
| **4. Production Architecture** | Async, streaming, caching, orchestration patterns | ⏳ Planned |
| **5. Observability & Eval** | Tracing, eval datasets, LLM-as-judge | ⏳ Planned |
| **6. Governance & Guardrails** | Prompt injection, output validation, OWASP LLM Top 10 | ⏳ Planned |
| **7. Deployment & Capstone** | Docker, K8s, end-to-end production agent | ⏳ Planned |

---

## Module 1 — Foundations

**Goal:** Build a tool-calling agent from scratch, no frameworks, and learn what every
framework abstracts away.

### What's inside

- ReAct loop (Reason → Act → Observe) implemented in plain Python
- Three tools: `calculator`, `fetch_url`, `read_file`
- OpenAI-compatible client targeting Gemini, Ollama, and any other compatible provider
- Multi-provider fallback chain with exponential backoff
- Per-turn and per-run token accounting
- `MAX_TURNS` circuit breaker for runaway loops
- Tools that return errors as strings instead of raising exceptions

### Key concepts demonstrated

- The two-layer tool pattern (schema vs implementation)
- `stop_reason` as agent control flow
- Parallel vs sequential tool calls — and the non-determinism that makes them tricky
- Provider rate limits, quota errors, and graceful degradation
- Model self-imposed refusals (small models over-refuse repetitive requests)
- Why hosted APIs usually beat self-hosted models for low-end hardware

### Run it yourself

```bash
git clone https://github.com/ankitpani8/learning_AgenticAI.git
cd learning_AgenticAI
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
cp .env.example .env                # then add your GEMINI_API_KEY
python module1_foundations/agent_Gemini.py
```

Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com).

### Files
module1_foundations/
├── agent_Gemini.py    # Main agent loop with multi-provider fallback
├── tools.py           # Tool implementations and schemas
└── README.md          # Module-specific notes and findings

---

## Tech Stack

- **Python 3.11+**
- **OpenAI SDK** (used as the OpenAI-compatible client for any provider)
- **Google Gemini API** — primary LLM provider (free tier)
- **httpx** — async-ready HTTP client
- **python-dotenv** — environment variable management

Modules 2+ will add: LangGraph, LangSmith/Langfuse, FastAPI, Docker, and others.

---

## Repo Structure
learning_AgenticAI/
├── .env.example              # Template for required environment variables
├── .gitignore
├── LICENSE
├── README.md                 # You are here
├── requirements.txt          # Pinned dependencies
└── module1_foundations/      # One folder per module
├── agent_Gemini.py
├── tools.py
└── README.md

---

## Notes for Visitors

This is an active learning project. Each module is tagged on GitHub
(e.g., `v0.1.0-module1`) — browse the [Releases](../../releases) page to see
milestone-by-milestone progress with summaries.

I'm documenting findings publicly because most production lessons in agentic AI
aren't in the docs — they're in the failure modes. This repo captures both.

---

## Connect

- LinkedIn: [@ankitpani](https://www.linkedin.com/in/ankitpani/)
- GitHub: [@ankitpani8](https://github.com/ankitpani8)

---

## License

MIT — see [LICENSE](LICENSE).



