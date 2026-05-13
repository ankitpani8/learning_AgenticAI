"""Three-conversation test:
  Session 1: tell the agent something specific
  Session 2: unrelated conversation
  Session 3: refer back to session 1's topic -- agent should retrieve it
"""
import importlib
_agent = importlib.import_module("03_agent_langgraph")
run_session = _agent.run_session
reindex_knowledge_base = _agent.reindex_knowledge_base

reindex_knowledge_base(verbose=False)

run_session("session_1", [
    "Hi, my name is Ankit. I'm planning a trip to Lisbon in March.",
    "I want to visit Belem and Sintra specifically.",
    "I love pineapple on pizza.",
])

run_session("session_2", [
    "What's a good Python testing framework?",
    "Thanks. Also, I'm vegetarian, so any restaurant suggestions should keep that in mind.",
    "Actually I hate pineapple on pizza",
])

run_session("session_3", [
    # "Remind me what we were planning for March?",
    "Do I like pineapple on pizza?",
    # "What are my upcoming travel plans?",
])