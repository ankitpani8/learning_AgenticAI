# Agentic AI Patterns

## ReAct loop
The Reason-Act loop is the foundation of every agent. The LLM produces a
thought, picks a tool, runs it, observes the result, and repeats until done.

## Multi-agent topologies
Five canonical patterns: sequential pipeline, hierarchical with manager,
router with experts, evaluator-optimizer (critic loop), and debate. Pattern
choice depends on parallelizability and specialization needs.

## Token economics
Each agent turn re-sends the full conversation history, so cost grows
super-linearly with turn count. Production systems use prompt caching,
history compression, and turn caps to control this.

## Critic models
Critics should use small or local models. They produce bounded structured
output and burning a big model on this is the most common form of
multi-agent token waste.