"""Hierarchical multi-agent system with dynamic parallel dispatch.

Business case: Competitive Intelligence Brief
--------------------------------------------
Given a list of companies, produce a structured brief covering each company's
flagship product, recent funding, and one strategic risk.

Why hierarchical fits this problem:
  - The research subtasks are independent (Anthropic research doesn't need
    OpenAI research to complete first).
  - The number of subtasks is determined by the user's request, not coded.
  - The dispatch logic (which company gets which kind of research) is an
    LLM judgment call, not deterministic code.

System shape:
    Manager           -- decides which companies to dispatch (heavy role)
      |
      +-> Researcher  -- one parallel invocation per company (light role)
      +-> Researcher
      +-> Researcher
      |
    Synthesizer       -- merges N research results into one brief (heavy role)

The Send API is what makes parallel dispatch concrete -- the manager returns a
list of Send(node_name, payload) objects, and LangGraph runs them concurrently.
"""
import operator
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Make lib/ importable when running this file directly
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.providers import select_all_models

load_dotenv(Path(__file__).parent.parent / ".env")


# --- Model selection (startup-time) ---------------------------------------
SELECTIONS = select_all_models(roles=["heavy", "light"])
manager_llm     = SELECTIONS["heavy"].to_langchain(temperature=0.2)
researcher_llm  = SELECTIONS["light"].to_langchain(temperature=0.3)
synthesizer_llm = SELECTIONS["heavy"].to_langchain(temperature=0.2)

# Cost-control caps
MAX_COMPANIES_PER_BRIEF = 8     # manager can't blow up dispatch counts
MAX_FETCHES_PER_RESEARCHER = 2  # workers can't fetch the entire web


# --- State schema ---------------------------------------------------------

class BriefState(TypedDict):
    """Shared state for the hierarchical run.

    Note `research_results` uses operator.add as its reducer -- this is how
    parallel workers' outputs accumulate into one list rather than overwriting
    each other.
    """
    user_request: str                                    # the original ask
    companies: list[str]                                 # manager's dispatch plan
    research_results: Annotated[list[dict], operator.add]
    final_brief: str                                     # synthesizer's output


class ResearchSubtask(TypedDict):
    """The payload sent to each parallel researcher invocation."""
    company: str
    focus: str  # what to research about this specific company


# --- Manager node ---------------------------------------------------------
# The manager's only job is decomposition: parse the user's request into a
# list of (company, focus) subtasks. It does NOT do research itself.

MANAGER_SYSTEM = """You are a research coordinator. Your job is to parse a user's
competitive intelligence request into a structured dispatch plan.

Output STRICTLY the following format, one company per line:
    COMPANY: <name> | FOCUS: <what to research about this company>

Do not add commentary, intros, or outros. Do not number the lines.
If the user names more than {max_n} companies, include only the first {max_n}."""


def manager_node(state: BriefState) -> dict:
    """Manager decomposes the user request into per-company subtasks."""
    print("\n=== MANAGER ===")
    print(f"  request: {state['user_request']}")

    prompt = [
        SystemMessage(content=MANAGER_SYSTEM.format(max_n=MAX_COMPANIES_PER_BRIEF)),
        HumanMessage(content=state["user_request"]),
    ]
    response = manager_llm.invoke(prompt)
    if response.usage_metadata:
        u = response.usage_metadata
        print(f"tokens: in={u['input_tokens']} out={u['output_tokens']}")
    plan_text = response.content

    # Parse the manager's structured output.
    companies = []
    for line in plan_text.splitlines():
        line = line.strip()
        if not line.startswith("COMPANY:"):
            continue
        try:
            left, right = line.split("|", 1)
            company = left.replace("COMPANY:", "").strip()
            focus = right.replace("FOCUS:", "").strip()
            if company and focus:
                companies.append({"company": company, "focus": focus})
        except ValueError:
            continue  # malformed line, skip

    if not companies:
        # Manager produced no parseable dispatch -- fail loudly rather than
        # silently doing nothing.
        raise ValueError(f"Manager produced no valid dispatch plan. Got:\n{plan_text}")

    if len(companies) > MAX_COMPANIES_PER_BRIEF:
        print(f"  WARNING: manager returned {len(companies)} companies; truncating to {MAX_COMPANIES_PER_BRIEF}")
        companies = companies[:MAX_COMPANIES_PER_BRIEF]

    print(f"  dispatch plan: {len(companies)} companies")
    for c in companies:
        print(f"    - {c['company']}: {c['focus']}")

    return {"companies": companies}


# --- Dispatch routing (the Send API in action) ----------------------------

def dispatch_researchers(state: BriefState) -> list[Send]:
    """Convert the manager's plan into N concurrent Send invocations.

    This is the heart of hierarchical dispatch. Returning a list of Send
    objects from a conditional edge tells LangGraph: 'run all of these
    in parallel; each gets its own payload as state.'
    """
    print(f"\n--- DISPATCHING {len(state['companies'])} RESEARCHERS IN PARALLEL ---")
    return [
        Send("researcher", subtask)
        for subtask in state["companies"]
    ]


# --- Researcher node (the parallelizable worker) --------------------------

RESEARCHER_SYSTEM = """You are a research analyst. Given one company and a
research focus, produce structured findings.

Use your background knowledge -- do not invent fictional details. If you do not
know something, say so explicitly rather than hallucinating.

Output STRICTLY in this format:
    PRODUCT: <flagship product or service in one sentence>
    FUNDING: <most recent funding or revenue point you are confident about>
    RISK: <one specific strategic risk facing this company>

No extra commentary."""


def researcher_node(subtask: ResearchSubtask) -> dict:
    """One company's worth of research. Multiple instances run in parallel.

    Note: this node receives the SUBTASK payload directly (from Send), not
    the full BriefState. That's how Send works -- each parallel invocation
    has its own isolated input.
    """
    company = subtask["company"]
    focus = subtask["focus"]
    print(f"  [worker] researching: {company}")

    prompt = [
        SystemMessage(content=RESEARCHER_SYSTEM),
        HumanMessage(content=f"Company: {company}\nResearch focus: {focus}"),
    ]
    response = researcher_llm.invoke(prompt)

    if response.usage_metadata:
        u = response.usage_metadata
        print(f"    [{company}] tokens: in={u['input_tokens']} out={u['output_tokens']}")

    return {
        "research_results": [
            {
                "company": company,
                "focus": focus,
                "findings": response.content.strip(),
            }
        ]
    }


# --- Synthesizer node -----------------------------------------------------

SYNTHESIZER_SYSTEM = """You are a senior analyst preparing a competitive
intelligence brief. Given research notes on multiple companies, produce a
clean comparative brief.

Format:
# Competitive Intelligence Brief

## Summary
<2-3 sentence overview of the competitive landscape>

## Company-by-Company

### <Company name>
- **Product:** ...
- **Funding/Revenue:** ...
- **Strategic risk:** ...

(repeat for each company)

## Comparative Observations
<2-3 bullet points comparing the companies>

Do not invent details. If the underlying research said 'unknown' or similar,
say so in the brief."""


def synthesizer_node(state: BriefState) -> dict:
    """Merge N parallel research results into one structured brief."""
    print(f"\n=== SYNTHESIZER ({len(state['research_results'])} inputs) ===")

    research_blob = "\n\n".join(
        f"### {r['company']}\n{r['findings']}"
        for r in state["research_results"]
    )

    prompt = [
        SystemMessage(content=SYNTHESIZER_SYSTEM),
        HumanMessage(content=(
            f"Original user request: {state['user_request']}\n\n"
            f"Research notes from {len(state['research_results'])} parallel "
            f"workers:\n\n{research_blob}"
        )),
    ]
    response = synthesizer_llm.invoke(prompt)

    if response.usage_metadata:
        u = response.usage_metadata
        print(f"tokens: in={u['input_tokens']} out={u['output_tokens']}")

    return {"final_brief": response.content}


# --- Graph assembly -------------------------------------------------------

def build_graph():
    g = StateGraph(BriefState)
    g.add_node("manager", manager_node)
    g.add_node("researcher", researcher_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "manager")
    # Conditional edge that returns Send objects -> parallel dispatch
    g.add_conditional_edges("manager", dispatch_researchers, ["researcher"])
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)

    return g.compile()


# --- Entry point ----------------------------------------------------------

# --- Serial Subtasks vs Parallel Subtasks comparison version (for observing parallelism vs serial time gain) ------

def run_serial_baseline(user_request: str) -> tuple[float, list[dict]]:
    """Same researchers, same synthesizer, but loop serially. Diagnostic only."""
    print("\n--- SERIAL BASELINE ---")
    # Reuse the manager to get a comparable dispatch plan
    plan_state = manager_node({"user_request": user_request, "companies": [],
                                "research_results": [], "final_brief": ""})

    start = time.time()
    results = []
    for subtask in plan_state["companies"]:
        out = researcher_node(subtask)
        results.extend(out["research_results"])
    elapsed = time.time() - start

    print(f"  Serial wall-clock for research phase: {elapsed:.2f}s")
    return elapsed, results

if __name__ == "__main__" and "--compare" in sys.argv:
    # Run serial (for loop), then hierarchical, compare wall-clock
    request = "Prepare a competitive intelligence brief on Anthropic, OpenAI, and Mistral."

    print("\n[1/2] Serial baseline")
    seq_time, _ = run_serial_baseline(request)

    print("\n[2/2] Hierarchical with parallel dispatch")
    app = build_graph()
    start = time.time()
    app.invoke({"user_request": request, "companies": [],
                "research_results": [], "final_brief": ""})
    parallel_time = time.time() - start

    print(f"\n--- COMPARISON ---")
    print(f"Serial:   {seq_time:.2f}s (research phase only)")
    print(f"Hierarchical: {parallel_time:.2f}s (full pipeline incl. manager + synthesis)")
    print(f"Speedup on research: {seq_time / (parallel_time / 1.4):.1f}x (rough)")

def export_diagram():
    app = build_graph()
    src = app.get_graph().draw_mermaid()
    with open("03_hierarchical_graph.mmd", "w") as f:
        f.write(src)
    print(src)


if __name__ == "__main__" and "--diagram" in sys.argv:
    export_diagram()


if __name__ == "__main__":
    app = build_graph()

    user_request = (
        "Prepare a competitive intelligence brief on Anthropic, OpenAI, and "
        "Mistral. For each, summarize their flagship product, recent funding "
        "or revenue, and one strategic risk."
    )

    # Testing parsing of random stuff like Mars, which should be ignored (Test result: Mars wasn't ignored, Ollama ran the subtask for "NASA") rather than causing parsing failure:
    # user_request = (
    # "Give me a brief on the leading AI companies and also the company that "
    # "makes the iPhone, plus some random stuff about Mars."
    # )

    print("\n" + "=" * 70)
    print("HIERARCHICAL CRUSH RUN")
    print("=" * 70)

    start_time = time.time()
    final_state = app.invoke({
        "user_request": user_request,
        "companies": [],
        "research_results": [],
        "final_brief": "",
    })
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("FINAL BRIEF")
    print("=" * 70)
    print(final_state["final_brief"])

    print("\n" + "=" * 70)
    print(f"Wall-clock: {elapsed:.2f}s")
    print(f"Companies researched: {len(final_state['research_results'])}")
    print("=" * 70)
