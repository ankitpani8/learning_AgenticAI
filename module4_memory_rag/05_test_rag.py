import importlib
_agent = importlib.import_module("03_agent_langgraph")
run_session = _agent.run_session
reindex_knowledge_base = _agent.reindex_knowledge_base

reindex_knowledge_base(verbose=False)

run_session("rag_test", [
    "How should I manage Python dependencies?",
    "What's the difference between hierarchical and router-experts multi-agent patterns?",
    "What's the current GDP of France?",  # nothing in KB
])