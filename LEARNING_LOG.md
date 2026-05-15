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

## Module 4 — Memory and RAG

- **When do I reach for RAG vs fine-tuning vs long context?**
  RAG when knowledge changes often, is large, or has access controls.
  Fine-tuning when you need a specific style/format or are at volume where
  prompt cost matters. Long context for one-off tasks where the document
  fits and you don't need it again.

- **Why use a separate embedding model from the chat model?**
  Embedding and generation are different architectures (encoder vs decoder),
  different sizes (20M–7B vs 1B–500B params), and different cost profiles.
  Chat models can't produce good embeddings; using them for embedding wastes
  10-100x the cost for worse quality.

- **Why is content-only hashing insufficient for incremental indexing?**
  Hashes capture source data but miss indexing-config changes. Change
  CHUNK_SIZE or swap embedding models, content hash stays the same, but
  stored chunks are stale. Production hashes the (content + chunk_size +
  embed_model) tuple.

- **What's different between RAG indexing and episodic memory storage?**
  RAG sources are external, mutable, lifecycle-managed (detect/add/update/
  delete). Episodes are internal, immutable, append-only with eviction
  policies. Same vector store, very different write semantics.

- **Why did the LLM extractor save "I love pineapple on pizza" as a diet?**
  Hallucination by substitution. Small models, given an extraction task
  with a fixed schema, will force an ill-fitting input into a schema-shaped
  answer rather than return empty. Mitigation: deterministic validation
  downstream of LLM extraction. Smartness suggests, dumbness validates.

- **Why did the episodic summarizer say "Alimentaria in Belem, Brazil"?**
  The summarizer is a 1.5B model trying to compress dialogue. It lost
  speaker attribution (user vs assistant) and hallucinated geography.
  Production summarizers use stronger models and structured templates that
  force "USER: ... ASSISTANT: ..." attribution.

- **Why is hard refusal when context is empty better than "answer carefully"?**
  Small models especially have weak refusal habits — they'd rather guess
  than say "I don't know." The factual-refusal pattern (classify the query,
  refuse if factual + no context) is deterministic and costs one cheap LLM
  call. It's the highest-ROI hallucination guardrail in production.

- **Why does memory rot compound across sessions?**
  One bad write becomes many bad retrievals. Bad episodes get summarized
  into more bad episodes. The fix isn't better prompts — it's user-visible
  memory inspection. Every production assistant (ChatGPT, Claude, Letta)
  exposes memory so users can audit and correct.

- **What's the role abstraction's payoff when all roles map to the same model?**
  Even when "heavy" and "light" both bind to qwen2.5:1.5b (because that's
  what fits in 7GB RAM), the indirection costs nothing at runtime and lets
  you change policy in one place when you move to a better machine.

## Module 5 — Production Architecture

- **Why is async worth the complexity premium?**
  Almost everything an agent does is wait (LLM API, tools, vector store).
  Async lets one process hold many concurrent waits. The single-request
  latency is unchanged; throughput goes up 10-1000x. Complexity premium
  is just keywords; the throughput gain is enormous.

- **Why was my first load test slower concurrently than sequentially?**
  Cache contamination across runs. The second phase measured the cache,
  not the system. Fix: unique queries per phase + admin /cache/clear
  endpoint between phases.

- **Why was concurrent slower than sequential even after fixing the cache?**
  Backend serialization. Ollama is a single Python process holding one
  model in RAM; "parallel" requests queue inside Ollama. Async at the
  Python layer can't fix a single-instance backend. Same lesson from
  Module 3 about hierarchical dispatch on Ollama, restated at a different
  layer.

- **What's the difference between p50, p95, p99?**
  Percentiles over response times. p50 is the median (typical user
  experience). p95 is 1-in-20. p99 is 1-in-100, the tail. Production
  SLOs target p95/p99 because the tail is what your worst-treated users
  actually feel. Mean latency is misleading — one bad outlier drags
  it up without changing user experience for most.

- **Why three caching layers (exact, semantic, provider)?**
  Different traffic patterns get caught by different layers. Exact handles
  retries and repeated identical queries. Semantic handles paraphrases.
  Provider-level prompt caching handles stable prefixes (system prompts,
  tool definitions). All three compose; production systems often use all
  three.

- **What's the right way to handle different failure types in retry?**
  Classify before retrying. Transient (5xx, timeout) -> exponential
  backoff with jitter. Rate limit (429) -> wait the suggested delay.
  Bad input (400, schema) -> fail fast, don't retry. Auth (401, 403) ->
  fail fast, alert. Generic retry-everything loops waste latency budget
  and amplify outages.

- **Why does cache hit ratio matter so much in production?**
  Caches don't just save tokens — they save tail latency, smooth out
  rate-limit pressure, and absorb traffic spikes. 30-60% hit rate on an
  FAQ-style agent is realistic and is a 30-60% cost reduction at zero
  quality cost. Single highest-ROI production optimization.

- **Why is the provider fallback chain a defense against more than runtime errors?**
  Adding a garbage API key caused the startup health check to fail; the
  chain bound to the next provider automatically. Fallback handles auth
  misconfigurations, network blips, quota exhaustion, and rate limits
  with the same mechanism.

- **What's the agent / service distinction?**
  An agent is the logic — graph, tools, prompts, state. A service is the
  deployment shell — HTTP endpoints, auth, rate limiting, observability,
  configuration. Production agents are agents plus services. Modules 1-4
  built agents. Module 5 built the service shell around them.