"""Agent state schema."""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # Conversation history. add_messages tells LangGraph to APPEND
    # new messages rather than overwrite the list.
    messages: Annotated[list, add_messages]

    # Circuit breaker counter — overwritten on each update.
    turn_count: int

    # Whether the most recent tool args passed validation.
    last_validation: str  # "ok", "rejected", "max_retries", or "" (initial)

    # Counter for validation retries to prevent infinite loops.
    validation_retry_count: int