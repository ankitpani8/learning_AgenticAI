"""Demonstrate state persistence and resumption via the checkpointer."""
from langchain_core.messages import HumanMessage, SystemMessage
from agent import build_graph, SYSTEM_PROMPT

app = build_graph()
config = {"configurable": {"thread_id": "session-A"}}

# Turn 1: ask a question
print(">>> First invocation")
app.invoke({
    "messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Calculate 100 * 25.")
    ],
    "turn_count": 0,
    "last_validation": "",
}, config=config)

# Turn 2: continue the SAME conversation (same thread_id) without resending history
print("\n>>> Second invocation, same thread")
result = app.invoke({
    "messages": [HumanMessage(content="Now multiply that result by 2.")],
}, config=config)

print("\n=== ANSWER TO FOLLOW-UP ===")
print(result["messages"][-1].content)

# Inspect the persisted state
print("\n=== STATE HISTORY (most recent first) ===")
for i, snapshot in enumerate(app.get_state_history(config)):
    print(f"  step {i}: turn_count={snapshot.values.get('turn_count')}, "
          f"messages={len(snapshot.values.get('messages', []))}")
    if i >= 4:
        print("  ...")
        break