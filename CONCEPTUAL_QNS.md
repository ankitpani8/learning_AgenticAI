# Learning Log

A running list of questions I asked while building this repo, with one-line
takeaways. The questions are the thinking that produced the architecture —
preserved deliberately for the next person (or future me) to learn from.

## Module 1 — Foundations

- **What's actually different between an "agent", a "workflow", and a "chatbot"?**
  An agent has a goal, uses tools, and iterates based on observations. Drop
  any of those three and it's something else.

- **Why does each agent turn cost more than the previous one?**
  Every turn re-sends the entire conversation history. Token cost grows
  super-linearly with turn count, which is why prompt caching, history
  compression, and turn caps matter in production.

- **Why did the same query produce different turn counts on different runs?**
  Modern LLMs decide their own parallelization strategy. Same prompt, same
  model, can sequentialize one run and parallelize the next. You can bias
  this with prompting; you cannot force it.

- **Why did the model refuse to call my tool 15 times in a row?**
  Small RLHF-tuned models have over-corrected safety behaviors that produce
  fabricated limits ("I can only call this 5 times"). The fix isn't to argue
  with the model — it's to put deterministic loops in code, not in prompts.

## Module 2 — LangGraph

- **What does LangGraph actually add over a hand-written `while` loop?**
  Control flow as inspectable graph structure, typed state with reducers,
  conditional routing as first-class, native checkpointing for resumption,
  and native diagram rendering. The `while` loop works until you need any
  of those.

- **Why did my broken graph (missing edge) compile and run successfully?**
  LangGraph treats nodes with no outgoing edges as terminal. The graph
  produced output; the output was just garbage. Topology bugs fail silently
  as quality regressions, not as exceptions. Strongest argument for
  observability.

- **Why couldn't I trigger my URL validator by sending a bad URL?**
  Modern instruction-tuned models silently clean up tool args before
  emitting them. Validators must be tested directly with hand-crafted bad
  inputs, not through the LLM.

## Module 3 — Multi-Agent

- **How is Pattern A different from a LangChain Chain?**
  A Chain composes small operations (prompt → LLM call → parser).
  Pattern A composes whole agents (each with role, tools, possibly its own
  ReAct loop). You can implement Pattern A using a Chain as wiring; the
  chain is the rope, the pattern is the climb.

- **Are role labels (heavy/light/critic) tied to specific patterns?**
  No. Topology and role policy are orthogonal. Every pattern has nodes;
  every node has a role. Same role taxonomy applies in any framework.

- **Why prefer a small/local model for the critic role?**
  Critics are frequent callers and produce structured output. Burning a big
  model on bounded structured tasks is the most common form of multi-agent
  token waste.

- **What does "hierarchical" actually mean architecturally?**
  It means adding a manager LLM above the workers, not rearranging existing
  workers into a tree. The manager decides dispatch dynamically at runtime;
  workers report up to it. Workers don't talk to each other.

- **Why was hierarchical slower than sequential in my test?**
  Parallel dispatch only pays off when workers can actually run concurrently.
  Single-instance Ollama serializes requests internally regardless of how
  many `Send` calls fire. Hosted APIs give real concurrency; local
  single-instance setups don't.

- **For 3 companies, hierarchical cost 2.5× the tokens for no speedup. Why use it?**
  It doesn't pay off on small N. It starts winning around 6–10 parallel
  subtasks and dominates at 20+. Pattern choice depends on scale.

- **Why did Qwen 1.5b dispatch a researcher on "NASA" when I asked about Mars?**
  Hallucination by substitution. Small models don't filter ill-fitting
  inputs — they replace them with plausible nearby entities. The manager is
  the single most leverage-rich role in hierarchical systems; always use
  your best model there.

- **Why did "How much do refunds cost?" trigger the billing specialist to look up an invoice?**
  Specialists assume the user's intent matches their happy path. Router was
  correct; specialist was overeager. Specialist prompts need explicit
  conservatism instructions about when to invoke tools vs answer directly.

- **Why did a bug report route to ESCALATION instead of ENGINEERING?**
  Router category overlap. Whenever you add a category, you create new
  boundary cases with existing ones. Few-shot examples in the router prompt
  are the fix.