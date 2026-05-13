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
| **2. LangGraph** | State machines, validation nodes, checkpointing | ✅ Complete |
| **3. Multi-Agent** | CrewAI, LangGraph multi-agent, role-based selection | ✅ Complete |
| **4. Memory & RAG** | Persistent state, vector retrieval, context engineering | 🚧 Ongoing |
| **5. Production Architecture** | Async, streaming, caching, orchestration patterns | ⏳ Planned |
| **6. Observability & Eval** | Tracing, eval datasets, LLM-as-judge | ⏳ Planned |
| **7. Governance & Guardrails** | Prompt injection, output validation, OWASP LLM Top 10 | ⏳ Planned |
| **8. Deployment & Capstone** | Docker, K8s, end-to-end production agent | ⏳ Planned |

---

## Tech Stack

- **[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)**
- **OpenAI SDK** (used as the OpenAI-compatible client for any provider)
- **Google Gemini API** — primary LLM provider (free tier)
- **Ollama qwen2.5:1.5b** — Local Small LM (free)
- **httpx** — async-ready HTTP client
- **python-dotenv** — environment variable management

Get a free-tier Gemini API key at [aistudio.google.com](https://aistudio.google.com).
Download Ollama at: [ollama.com](https://ollama.com/download).
Anthropic is included in the provider chain at lower priority. To use Claude, set ANTHROPIC_API_KEY (~/.env) and either reorder ROLE_PREFERENCES to prefer it or run with Gemini disabled (~/lib/providers.py).

## Repo Structure
learning_AgenticAI/
├── .env.example              # Template for required environment variables
├── .gitignore
├── LICENSE
├── README.md                 # You are here
├── requirements.txt          # Pinned dependencies
└── module1_foundations/      # Foundations - building an agent from scratch in Python - without frameworks
    ├── agent_Claude.py
    ├── agent_Gemini_and_Ollama.py
    ├── tools.py
    └── README.md
└── module2_langgraph/        # Module 2: LangGraph framework
    ├── agent.py
    ├── state.py
    ├── tools.py
    ├── test_checkpoint.py
    ├── example.txt
    ├── graph.mmd
    └── README.md
└── module3_multiagent/           # Module 3: Multi-agent frameworks, using :anggraph (mainly) and CrewAI
    ├── crew.py
    ├── crew_in_langgraph.py
    ├── tools.py
    ├── graph.mmd
    └── README.md

---

### Run it yourself

```bash
git clone https://github.com/ankitpani8/learning_AgenticAI.git
cd learning_AgenticAI
py -3.11 -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
cp .env.example .env                # then add your GEMINI_API_KEY
python module1_foundations/agent_Gemini.py #as an example. Run any files by going to the location as per the repo structure

```
---

## Architecture: Role-Based Model Selection

Every agent in this repo requests models by **role** (`heavy`, `light`, `critic`)
rather than by name. A startup health-check protocol pings each provider in the
role's preference chain and binds the first one that responds. This:

- Fails fast on quota/auth/network issues before agents run
- Decouples agent code from provider choice (policy is in `lib/providers.py`)
- Lets the critic role prefer a local Ollama model — demonstrating that critics
  shouldn't cost more than what they're critiquing (a key multi-agent pattern)

See [`lib/providers.py`](lib/providers.py) for the implementation.

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

## Module 2 — LangGraph

**Goal:** Replace Module 1's hand-written ReAct loop with a state machine, and
use the new structure to add capabilities the loop couldn't easily support.

### What's inside

- Full LangGraph agent: typed state, three nodes (`llm`, `validate`, `tools`),
  conditional routing
- Validation node that rejects unsafe tool arguments and routes back to the LLM
- `MemorySaver` checkpointer for state persistence across invocations
- Multi-provider LLM fallback (Gemini Flash-Lite → Flash) encapsulated in one node
- Mermaid graph diagram exported directly from the compiled graph

### Key concepts demonstrated

- Control flow as graph topology, not embedded conditionals
- Reducers (`add_messages` append vs default replace)
- The `tool_call_id` ↔ `ToolMessage` contract
- Composability: how graph structure localizes future changes
- Defense in depth: LLM safety training + orchestration-layer guardrails
- Silent topology failures and why they motivate observability (Module 5)

### Why this matters
A `while` loop with `if/elif` works for one agent. Once you have multiple
specialized agents, parallel subagents, retry strategies that vary by error
type, or human-in-the-loop pauses — you need first-class control flow.
LangGraph (or something like it) is what every production agent system
converges on. Module 2 is where that transition happens in this repo.

See [`module2_langgraph/README.md`](module2_langgraph/README.md) for the
architecture diagram and detailed findings.

### What's new vs Module 1

- Agent expressed as a graph of nodes and edges, not a loop
- Typed state schema with reducers (append vs replace semantics)
- Validation node that inspects tool arguments and routes back to the LLM
  when they fail rules — pattern foundation for Module 6 guardrails
- Checkpointer (in-memory; SQLite-ready) for state persistence and resumption
- Mermaid diagram exported directly from the compiled graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
        __start__([<p>__start__</p>]):::first
        llm(llm)
        validate(validate)
        tools(tools)
        summarizer(summarizer)
        __end__([<p>__end__</p>]):::last
        __start__ --> llm;
        llm -. &nbsp;end&nbsp; .-> __end__;
        llm -.-> summarizer;
        llm -.-> validate;
        tools --> llm;
        validate -. &nbsp;end&nbsp; .-> __end__;
        validate -.-> llm;
        validate -.-> summarizer;
        validate -.-> tools;
        summarizer --> __end__;
        classDef default fill:#f2f0ff,line-height:1.2
        classDef first fill-opacity:0
        classDef last fill:#bfb6fc
```
## Architecture: Role-Based Model Selection

Every agent in this repo requests models by **role** (`heavy`, `light`,
`critic`) rather than by name. A startup health-check protocol pings each
provider in the role's preference chain and binds the first one that
responds. This:

- Fails fast on quota/auth/network issues before agents run
- Decouples agent code from provider choice (policy is in `lib/providers.py`)
- Lets the critic role prefer a local Ollama model — demonstrating that
  critics shouldn't cost more than what they're critiquing

Supported providers: Gemini (primary), Ollama (local), Anthropic (optional
backup). See [`lib/providers.py`](lib/providers.py).

## Module 3 — Multi-Agent Systems

**Goal:** Implement four multi-agent topologies as separate scripts and
develop opinions on when each pattern earns its keep.

### What's inside
- Sequential pipeline with critic loop, in both CrewAI and LangGraph
- Hierarchical dispatch with parallel workers (`Send` API) and synthesis
- Router + specialist experts with scoped tools per specialist
- Side-by-side framework comparison (CrewAI vs LangGraph)
- Empirical measurement of the multi-agent token tax

### Key concepts demonstrated
- Topology and role policy as orthogonal design dimensions
- The `Send` API for dynamic parallel dispatch in LangGraph
- Capability scoping as a security pattern (not just a cost pattern)
- Hallucination by substitution as a failure mode of small manager models
- Why hierarchical loses on small N and only wins on large N

See [`module3_multiagent/README.md`](module3_multiagent/README.md) for
diagrams and findings.

## Module 4 — Memory and RAG

**Goal:** Build a personal assistant with three memory layers and a knowledge
base that incrementally reindexes itself, while encountering and naming the
failure modes of long-running memory systems.

### What's inside
- Content-hash incremental RAG indexing over a markdown corpus
- Three memory layers: conversation (LangGraph), semantic facts (SQLite),
  episodic summaries (ChromaDB)
- LLM-based fact extraction with deterministic validation downstream
- Factual-refusal guardrail when no context is available
- User-visible memory inspection for auditability

### Key concepts demonstrated
- Memory vs RAG (different lifecycle semantics, different storage)
- The smartness/dumbness split: LLM extracts, deterministic code validates
- Hallucination by substitution at every layer (extractor, summarizer)
- Memory rot and why memory inspection is essential UX
- The factual-refusal pattern as the cheapest hallucination guardrail

See [`module4_memory_rag/README.md`](module4_memory_rag/README.md) for
the architecture and findings.

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



