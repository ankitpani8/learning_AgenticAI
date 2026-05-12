"""Same 3-agent pipeline, in raw LangGraph. Compare to crew.py.

This is what CrewAI hides from you. Both are valid; pick based on the topology
fit and your need for control."""
import os
from pathlib import Path
from typing import TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from lib.providers import select_all_models

ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

SELECTIONS = select_all_models(roles=["heavy", "light", "critic"])
researcher_llm = SELECTIONS["light"].to_langchain(temperature=0.3)
writer_llm     = SELECTIONS["heavy"].to_langchain(temperature=0.3)
critic_llm     = SELECTIONS["critic"].to_langchain(temperature=0.3)

MAX_REVISIONS = 3


class CrewState(TypedDict):
    query: str
    research_notes: str
    draft: str
    critique: str
    revision_count: int
    approved: bool


def research_node(state: CrewState) -> dict:
    print(f"\n--- RESEARCHER ---")
    prompt = [
        SystemMessage(content=(
            "You are a senior research analyst. Gather accurate information "
            "and produce a markdown bulleted list of facts. Be precise."
        )),
        HumanMessage(content=f"Research this question:\n{state['query']}"),
    ]
    r = researcher_llm.invoke(prompt)
    return {"research_notes": r.content}


def write_node(state: CrewState) -> dict:
    print(f"\n--- WRITER (revision {state['revision_count']}) ---")
    sys = SystemMessage(content=(
        "You are a technical writer. Produce a 200-300 word markdown report. "
        "Do not invent facts beyond the notes."
    ))
    if state["revision_count"] == 0:
        user = HumanMessage(content=(
            f"Notes:\n{state['research_notes']}\n\n"
            f"Question: {state['query']}\n\nWrite the report."
        ))
    else:
        user = HumanMessage(content=(
            f"Notes:\n{state['research_notes']}\n\n"
            f"Previous draft:\n{state['draft']}\n\n"
            f"Critique:\n{state['critique']}\n\n"
            f"Revise based on the critique. Keep what works."
        ))
    r = writer_llm.invoke([sys, user])
    return {"draft": r.content}


def critique_node(state: CrewState) -> dict:
    print(f"\n--- CRITIC ---")
    prompt = [
        SystemMessage(content=(
            "You are a sharp editor. Review for factual accuracy, weak claims, "
            "and structure. Respond with 'APPROVED' as first word if strong, "
            "otherwise a numbered list of issues."
        )),
        HumanMessage(content=f"Report:\n{state['draft']}"),
    ]
    r = critic_llm.invoke(prompt)
    approved = r.content.strip().upper().startswith("APPROVED")
    print(f"  approved: {approved}")
    return {
        "critique": r.content,
        "approved": approved,
        "revision_count": state["revision_count"] + 1,
    }


def route_after_critique(state: CrewState) -> str:
    if state["approved"]:
        return "end"
    if state["revision_count"] >= MAX_REVISIONS:
        return "end"
    return "writer"


def build():
    g = StateGraph(CrewState)
    g.add_node("researcher", research_node)
    g.add_node("writer", write_node)
    g.add_node("critic", critique_node)

    g.add_edge(START, "researcher")
    g.add_edge("researcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critique, {
        "writer": "writer",
        "end": END,
    })
    return g.compile()


if __name__ == "__main__":
    app = build()
    result = app.invoke({
        "query": "What is the difference between LangGraph (https://www.langchain.com/langgraph) and CrewAI(https://crewai.com/), and when would a team choose one over the other in 2026?",
        "research_notes": "",
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "approved": False,
    })
    print(result.usage_metadata)
    print("\n" + "="*70)
    print("FINAL DRAFT")
    print("="*70)
    print(result["draft"])
    print(f"\nApproved: {result['approved']}, Revisions: {result['revision_count']}")

def export_diagram():
    app = build()
    src = app.get_graph().draw_mermaid()
    with open("graph.mmd", "w") as f:
        f.write(src)
    print(src)

if __name__ == "__main__" and "--diagram" in __import__("sys").argv:
    export_diagram()