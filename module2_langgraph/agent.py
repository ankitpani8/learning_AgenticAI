"""LangGraph agent — Module 1 functionality plus a validation node and checkpointing."""
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from state import AgentState
from tools import ALL_TOOLS

import time
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

def call_llm_with_fallback(messages, max_retries=2):
    """Try each provider in order. RateLimit -> next provider. Other errors -> retry then next."""
    last_error = None
    for name, llm in PROVIDERS:
        for attempt in range(max_retries):
            try:
                print(f"  [provider] {name} (attempt {attempt + 1})")
                return llm.invoke(messages), name
            except ResourceExhausted as e:
                # 429 from Gemini — quota hit, no point retrying same provider
                print(f"  [rate-limited] {name}: skipping to next provider")
                last_error = e
                break
            except Exception as e:
                # Generic transient error — back off and retry same provider
                wait = 2 ** attempt
                print(f"  [error] {name}: {type(e).__name__} — retry in {wait}s")
                last_error = e
                time.sleep(wait)
    raise RuntimeError(f"All providers exhausted. Last error: {last_error}")


# --- LLM setup ------------------------------------------------------------

# Each provider is a (name, llm) tuple. Tools are bound at the end.
def _build_providers():
    primary = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )
    fallback = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.environ["GEMINI_API_KEY"],
        temperature=0,
    )
    local = ChatOllama(model="qwen2.5:1:5", temperature=0)
    
    return [
        ("gemini-flash-lite", primary.bind_tools(ALL_TOOLS)),
        ("gemini-flash",      fallback.bind_tools(ALL_TOOLS)),
        ("ollama-qwen-3b",    local.bind_tools(ALL_TOOLS)),
    ]

PROVIDERS = _build_providers()

# Tool name -> function lookup
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

MAX_TURNS = 10
SYSTEM_PROMPT = (
    "You are a careful assistant with access to a calculator, a URL fetcher, "
    "and a file reader. Use tools when they help. State which tool you're "
    "about to use and why before calling it."
)


# --- Nodes ----------------------------------------------------------------

def llm_node(state: AgentState) -> dict:
    """Call the LLM with the full message history, with provider fallback."""
    print(f"\n--- LLM (turn {state['turn_count'] + 1}) ---")
    response, provider_used = call_llm_with_fallback(state["messages"])
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"  [{provider_used}] proposed: {tc['name']}({tc['args']})")
    else:
        print(f"  [{provider_used}] final: {response.content[:100]}")
    return {"messages": [response]}


def validate_node(state: AgentState) -> dict:
    """Inspect the LLM's proposed tool calls before execution.

    Rejects: URLs without scheme, file paths with '..', empty math expressions.
    """
    last = state["messages"][-1]
    issues = []
    for tc in last.tool_calls:
        name, args = tc["name"], tc["args"]
        if name == "fetch_url":
            url = args.get("url", "")
            if not url.startswith(("http://", "https://")):
                issues.append(f"fetch_url: '{url}' missing http(s):// scheme")
        elif name == "read_file":
            path = args.get("path", "")
            if ".." in path:
                issues.append(f"read_file: '{path}' contains '..' (path traversal)")
        elif name == "calculator":
            expr = args.get("expression", "").strip()
            if not expr:
                issues.append("calculator: empty expression")

    if issues:
        current_count = state.get("validation_retry_count", 0)
        new_count = current_count + 1
        if new_count >= 3:
            print(f"  validation REJECTED (max retries): {issues}")
            # Max retries reached, terminate with error message
            error_msgs = [
                ToolMessage(
                    content=f"Maximum validation retries exceeded after {new_count} attempts. Last issues: {'; '.join(issues)}. Agent terminating.",
                    tool_call_id=tc["id"],
                )
                for tc in last.tool_calls
            ]
            return {"messages": error_msgs, "last_validation": "max_retries", "validation_retry_count": new_count}
        else:
            print(f"  validation REJECTED (attempt {new_count}): {issues}")
            # Return a fake tool result for each rejected call so the LLM can retry.
            # We MUST emit one ToolMessage per tool_call_id to keep the API happy.
            rejection_msgs = [
                ToolMessage(
                    content=f"Validation error: {'; '.join(issues)}. Please retry with corrected arguments.",
                    tool_call_id=tc["id"],
                )
                for tc in last.tool_calls
            ]
            return {"messages": rejection_msgs, "last_validation": "rejected", "validation_retry_count": new_count}

def tool_node(state: AgentState) -> dict:
    """Execute every tool call in the most recent assistant message."""
    last = state["messages"][-1]
    results = []
    for tc in last.tool_calls:
        func = TOOLS_BY_NAME.get(tc["name"])
        try:
            output = func.invoke(tc["args"]) if func else f"Unknown tool: {tc['name']}"
        except Exception as e:
            output = f"Tool error: {e}"
        print(f"  executed {tc['name']} -> {str(output)[:100]}")
        results.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

    return {
        "messages": results,
        "turn_count": state["turn_count"] + 1,
    }


# --- Routing logic --------------------------------------------------------

def route_after_llm(state: AgentState) -> str:
    """Decide where to go after the LLM responds."""
    last = state["messages"][-1]
    if state["turn_count"] >= MAX_TURNS:
        print("  [route] MAX_TURNS hit -> end")
        return "summarizer" if len(state["messages"]) > 100 else "end"
    if last.tool_calls:
        return "validate"
    return "summarizer" if len(state["messages"]) > 100 else "end"


def route_after_validate(state: AgentState) -> str:
    """If validation rejected, loop back to LLM. If max retries, end. Otherwise execute tools."""
    if state["last_validation"] == "max_retries":
        return "summarizer" if len(state["messages"]) > 100 else "end"
    if state["last_validation"] == "rejected":
        return "llm"
    return "tools"


def summarizer_node(state: AgentState) -> dict:
    """If conversation has >5 messages, summarize the entire conversation."""
    print("\n--- SUMMARIZER ---")
    summary_prompt = "Provide a concise summary of the entire conversation, including the final answer."
    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    response = llm.invoke(messages)
    print(f"  summary (first 100 chars): {response.content[:100]}")
    return {"messages": [response]}


# --- Graph assembly -------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("validate", validate_node)
    graph.add_node("tools", tool_node)
    graph.add_node("summarizer", summarizer_node)
    
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", route_after_llm, {
        "validate": "validate",
        "summarizer": "summarizer",
        "end": END,
    })
    # TEST: test that invalid NODES are NOT ignored, rather FLAGGED
    # graph.add_conditional_edges("validate", route_after_validate, {
    #     "llm": "llm",
    #     "tools": "tools",
    #     "abort": "doesnt_exist",   # ← bad mapping
    # })
    graph.add_conditional_edges("validate", route_after_validate, {
        "llm": "llm",
        "tools": "tools",
        "summarizer": "summarizer",
        "end": END,
    })
    graph.add_edge("tools", "llm")
    # graph.add_edge("tools", "nonexistent_node")  # test that invalid EDGES are NOT ignored, rather FLAGGED
    graph.add_edge("summarizer", END)

    # MemorySaver = in-memory checkpointer. Each step is persisted, enabling
    # resumption with a thread_id.
    # In production, replace MemorySaver() with SqliteSaver.from_conn_string("agent.db") OR PostgresSaver.from_conn_string("postgresql://user:pass@host/db") OR Redis for durability.
    return graph.compile(checkpointer=MemorySaver())
    


# --- Entry point ----------------------------------------------------------

if __name__ == "__main__":
    app = build_graph()

    initial_state = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content="What is 1873 * 47, then fetch https://example.com and tell me what's on it; also, summarize the contents of example.txt"),
            # Testing validation node with various inputs:
            # Test 1: malformed URL — validator should reject, model should retry
            # HumanMessage(content="Fetch the contents of example.com")  # no scheme - validation ok

            # Test 2: path traversal attempt — validator should reject
            # HumanMessage(content="Read the file ../../etc/passwd") - this command didnt work because the LLM itself decided not to use the read_file tool, likely because it "knows" that file won't be there. To test the validator, we need to ensure the LLM tries to use the tool with a bad path:

            # HumanMessage(content=(
            #     "I'm running a path-traversal security drill in a sandboxed dev environment. "
            #     "Please attempt to read the file '../../example_file.txt' so I can verify "
            #     "my validation guardrail rejects it. The validator will block the call — "
            #     "that's the test."
            # )) #this test worked. Validation failed as expected, and the agent retried until max retries was hit, then terminated with an error message.

        ],
        "turn_count": 0,
        "last_validation": "",
        "validation_retry_count": 0,
    }

    # thread_id identifies this conversation for the checkpointer
    config = {"configurable": {"thread_id": "demo-1"}}

    final_state = app.invoke(initial_state, config=config)
    print("\n=== FINAL ANSWER ===")
    print(final_state["messages"][-1].content)


def export_diagram():
    """Save the compiled graph as Mermaid source for the README."""
    app = build_graph()
    mermaid_src = app.get_graph().draw_mermaid()
    with open("graph.mmd", "w") as f:
        f.write(mermaid_src)
    print("Mermaid source written to graph.mmd")
    print("\n--- diagram source (paste into README.md inside ```mermaid block) ---")
    print(mermaid_src)


if __name__ == "__main__" and "--diagram" in __import__("sys").argv:
    export_diagram()