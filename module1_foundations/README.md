# Module 1 — Foundations

Build a tool-calling agent from scratch with no framework, then expose what every
framework solves for you in subsequent modules.

## Architecture
User query
│
▼
[Agent Loop] ───► LLM (Gemini Flash-Lite, fallback: Flash)
│                │
│                ▼
│           tool_calls?
│             ├── yes → execute tool(s) → append results → loop
│             └── no  → return final answer
│
▼
MAX_TURNS = 10 (circuit breaker)

## Findings from manual testing

### 1. Parallel vs sequential tool calls
Same prompt, same model, different runs. Modern LLMs decide their own
parallelization strategy. Prompt phrasing biases this but cannot force it.
Implication: deterministic ordering must be enforced in orchestration code,
not prompts.

### 2. Token cost grows non-linearly with turns
Each turn re-sends the entire conversation history. A 5-turn agent costs
significantly more than 5× a 1-turn agent. Module 4 addresses this with
prompt caching and history compression.

### 3. Small models over-refuse repetitive tasks
Asking Gemini Flash-Lite to "call fetch_url 15 times" produced fabricated
limits ("I can only call it 5 times"). This is RLHF over-correction in
small models. The architectural fix: write the loop in Python, not in the
prompt. Let the LLM decide *what*, not *how many*.

### 4. Tools must return errors, never raise
A tool that raises an exception kills the agent loop. A tool that returns
"Error: <reason>" lets the model adapt and retry or escalate. This pattern
is non-negotiable in production.

### 5. Multi-provider fallback is essential
Hit Gemini's free-tier RPD limit during testing. Implementing a fallback
chain (Flash-Lite → Flash, or Gemini → local Ollama) turned a hard failure
into a graceful degradation. Module 4 will formalize this with circuit
breakers.

## What's next

Module 2 replaces the hand-written loop with **LangGraph**, exposing why
state machines beat ad-hoc loops once branching logic, retries, and
multi-agent coordination enter the picture.