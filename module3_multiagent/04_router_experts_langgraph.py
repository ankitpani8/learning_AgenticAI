"""Router + specialist experts for customer support triage.

Router classifies the query into one of four categories and dispatches to a
single specialist. Specialists have scoped tools -- billing can issue refunds,
docs cannot. The router itself has no tools and never touches user data.

This is the canonical pattern for security-driven multi-agent design: minimum
tool surface per role, no agent has more capability than its job requires.
"""
import sys
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.providers import select_all_models
from langgraph.graph import StateGraph, START, END

load_dotenv(Path(__file__).parent.parent / ".env")

SELECTIONS = select_all_models(roles=["light", "heavy"])
router_llm     = SELECTIONS["light"].to_langchain(temperature=0)
specialist_llm = SELECTIONS["light"].to_langchain(temperature=0.3)
hard_llm       = SELECTIONS["heavy"].to_langchain(temperature=0.3)


# --- Mock tools (each scoped to one specialist) ---------------------------

def lookup_invoice(invoice_id: str) -> str:
    """Billing-only. Look up an invoice."""
    return f"Invoice {invoice_id}: $49.00, paid on 2026-04-15, plan=Pro."

def issue_refund(invoice_id: str, amount: float) -> str:
    """Billing-only. Issue a refund."""
    return f"Refund of ${amount} issued for invoice {invoice_id}. Confirmation: RF-9921."

def create_ticket(summary: str, severity: str) -> str:
    """Engineering-only. File a bug ticket."""
    return f"Ticket TICKET-4521 created: '{summary}' (severity={severity})."

def search_docs(query: str) -> str:
    """Docs-only. Search product documentation."""
    return f"[docs] Top result for '{query}': 'You can configure this in Settings > Account.'"


# --- State schema ---------------------------------------------------------

class SupportState(TypedDict):
    user_query: str
    category: str                # router's classification
    response: str                # specialist's answer


# --- Token usage tracking -------------------------------------------------

_token_usage: dict[str, dict] = {}


def _extract_tokens(response) -> dict[str, int]:
    """Extract input/output/total counts from a LangChain AI message."""
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        m = response.usage_metadata
        i = m.get("input_tokens", 0)
        o = m.get("output_tokens", 0)
        return {"input": i, "output": o, "total": m.get("total_tokens", i + o)}
    rm = getattr(response, "response_metadata", {}) or {}
    um = rm.get("usage_metadata", {})
    if um:
        i, o = um.get("prompt_token_count", 0), um.get("candidates_token_count", 0)
        return {"input": i, "output": o, "total": um.get("total_token_count", i + o)}
    if "prompt_eval_count" in rm:
        i, o = rm.get("prompt_eval_count", 0), rm.get("eval_count", 0)
        return {"input": i, "output": o, "total": i + o}
    return {"input": 0, "output": 0, "total": 0}


def _record_usage(response, provider: str) -> None:
    t = _extract_tokens(response)
    entry = _token_usage.setdefault(provider, {"input": 0, "output": 0, "total": 0})
    entry["input"] += t["input"]
    entry["output"] += t["output"]
    entry["total"] += t["total"]


# --- Router node ----------------------------------------------------------

ROUTER_SYSTEM = """You are a support query classifier. Read the user's query
and respond with EXACTLY ONE WORD from this list:

  BILLING       -- invoices, refunds, charges, subscription costs
  ENGINEERING   -- bugs, crashes, broken features, errors
  DOCS          -- how-to questions, product features, configuration help
  ESCALATION    -- explicit requests to speak with a human, urgent/sensitive complaints, legal threats. complaints about prior support 
                interactions, or threats of churn. NOT routine bug reports.
  FALLBACK      -- anything else, unclear queries, greetings

Respond with only the single word. No punctuation, no explanation."""


def router_node(state: SupportState) -> dict:
    print(f"\n[router] query: {state['user_query']!r}")
    response = router_llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=state["user_query"]),
    ])
    _record_usage(response, SELECTIONS["light"].provider)
    raw = response.content.strip().upper()

    # Defensive parse -- model might return "BILLING." or "Billing question"
    valid = {"BILLING", "ENGINEERING", "DOCS", "FALLBACK", "ESCALATION"}
    category = next((v for v in valid if v in raw), "FALLBACK")
    print(f"[router] -> {category}")
    return {"category": category}


# --- Specialist nodes -----------------------------------------------------

def billing_node(state: SupportState) -> dict:
    print("  [billing specialist]")
    # In real systems, the LLM would tool-call. We simulate the decision +
    # tool result inline for clarity.
    sys_prompt = (
        "You are a billing specialist. You have access to invoice lookup and "
        "refund tools. Be helpful but verify before issuing refunds."
    )
    # Simulate a typical billing flow
    invoice_info = lookup_invoice("INV-1234")
    response = specialist_llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=(
            f"User query: {state['user_query']}\n\n"
            f"I looked up their invoice: {invoice_info}\n\n"
            f"Write a helpful response."
        )),
    ])
    _record_usage(response, SELECTIONS["light"].provider)
    return {"response": response.content}


def engineering_node(state: SupportState) -> dict:
    print("  [engineering specialist]")
    sys_prompt = (
        "You are an engineering support specialist. File a ticket for genuine "
        "bugs. Ask clarifying questions only when essential."
    )
    ticket = create_ticket(state["user_query"][:80], severity="medium")
    response = specialist_llm.invoke([
        SystemMessage(content=sys_prompt),
        HumanMessage(content=(
            f"User reported: {state['user_query']}\n\n"
            f"I filed: {ticket}\n\n"
            f"Acknowledge to the user and set expectations."
        )),
    ])
    _record_usage(response, SELECTIONS["light"].provider)
    return {"response": response.content}


def docs_node(state: SupportState) -> dict:
    print("  [docs specialist]")
    docs_result = search_docs(state["user_query"])
    response = specialist_llm.invoke([
        SystemMessage(content="You answer how-to questions using the docs result."),
        HumanMessage(content=(
            f"User asked: {state['user_query']}\n\n"
            f"Docs returned: {docs_result}\n\n"
            f"Answer the user's question."
        )),
    ])
    _record_usage(response, SELECTIONS["light"].provider)
    return {"response": response.content}


def fallback_node(state: SupportState) -> dict:
    print("  [fallback]")
    return {"response": (
        "I'm not sure how to help with that. Could you tell me whether this "
        "is about billing, a technical issue, or how to use a feature?"
    )}


def escalation_node(state: SupportState) -> dict:
    print("  [escalation]")
    print(f"  [escalation] recorded request: {state['user_query']!r}")
    return {"response": (
        "Your request has been escalated to a human support agent. "
        "Someone will reach out to you within 1 business day."
    )}


# --- Routing logic --------------------------------------------------------

def route_by_category(state: SupportState) -> str:
    return {
        "BILLING": "billing",
        "ENGINEERING": "engineering",
        "DOCS": "docs",
        "FALLBACK": "fallback",
        "ESCALATION": "escalation",
    }[state["category"]]


# --- Graph ----------------------------------------------------------------

def build_graph():
    g = StateGraph(SupportState)
    g.add_node("router", router_node)
    g.add_node("billing", billing_node)
    g.add_node("engineering", engineering_node)
    g.add_node("docs", docs_node)
    g.add_node("fallback", fallback_node)
    g.add_node("escalation", escalation_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_by_category, {
        "billing": "billing",
        "engineering": "engineering",
        "docs": "docs",
        "fallback": "fallback",
        "escalation": "escalation",
    })
    for node in ["billing", "engineering", "docs", "fallback", "escalation"]:
        g.add_edge(node, END)

    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    queries = [
        "I was charged twice for my subscription this month, please refund one.",
        "The export button crashes the app every time I click it on Chrome.",
        "How do I change my workspace name?",
        "Hi, is anyone there?",
        "I want to speak to a human agent right now, this is completely unacceptable.",
        # "How much do refunds cost?",
    ]
    for q in queries:
        print("\n" + "=" * 60)
        result = app.invoke({"user_query": q, "category": "", "response": ""})
        print(f"FINAL: {result['response'][:200]}")

    print("\n" + "=" * 60)
    print("TOKEN USAGE SUMMARY")
    print("=" * 60)
    grand = {"input": 0, "output": 0, "total": 0}
    for provider, counts in _token_usage.items():
        print(f"  {provider:12s}  input={counts['input']:>6}  output={counts['output']:>6}  total={counts['total']:>6}")
        for k in grand:
            grand[k] += counts[k]
    print(f"  {'GRAND TOTAL':12s}  input={grand['input']:>6}  output={grand['output']:>6}  total={grand['total']:>6}")
    print("=" * 60)


def export_diagram():
    src = build_graph().get_graph().draw_mermaid()
    with open("04_router_experts_graph.mmd", "w") as f:
        f.write(src)
    print(src)


if __name__ == "__main__" and "--diagram" in sys.argv:
    export_diagram()