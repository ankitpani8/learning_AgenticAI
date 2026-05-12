"""Three-agent research crew: Researcher -> Writer -> Critic (With / Without looping back to Writer)

SEQUENTIAL + EVALUATOR-OPTIMIZER: A straightforward pipeline with an inline critic step. The critic's feedback flows into the writer's task as context — this is CrewAI's idiomatic way of doing critic loops without explicit graph wiring.
Sequential pipeline with an inline critic step. The critic's
feedback flows into the writer's task as context — this is CrewAI's idiomatic
way of doing critic loops without explicit graph wiring.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from tools import fetch_url, calculator
from lib.providers import select_all_models

# --- Env ------------------------------------------------------------------
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)

# --- LLM ------------------------------------------------------------------
# CrewAI uses LiteLLM under the hood. Gemini's LiteLLM model string is "gemini/<name>".
# Heavy for writing, light for research, critic for critique (local Ollama if available).
SELECTIONS = select_all_models(roles=["heavy", "light", "critic"])
heavy_llm  = SELECTIONS["heavy"].to_crewai(temperature=0.3)
light_llm  = SELECTIONS["light"].to_crewai(temperature=0.3)
critic_llm = SELECTIONS["critic"].to_crewai(temperature=0.3)

# --- Agents ---------------------------------------------------------------
# Note the role/goal/backstory pattern. CrewAI uses these to construct the
# system prompt for each agent. Be specific — this is where specialization
# actually happens.

researcher = Agent(
    role="Senior Research Analyst",
    goal=(
        "Gather accurate, well-sourced information to answer the user's question. "
        "Use the web tool to fetch primary sources. Quote URLs you used."
    ),
    backstory=(
        "You are a research analyst who values precision over speed. You never "
        "speculate beyond what the sources say. If sources disagree, you flag "
        "the disagreement rather than picking a side."
    ),
    tools=[fetch_url, calculator],
    llm=light_llm,    # cheap model for fetching and summarizing
    verbose=True,
    max_iter=4,                # cap tool-calling iterations per task
    allow_delegation=False,
)

writer = Agent(
    role="Technical Writer",
    goal=(
        "Convert raw research notes into a clear, structured short report. "
        "Maintain factual accuracy. Use markdown formatting."
    ),
    backstory=(
        "You write for technically literate readers who want substance over filler. "
        "You prefer concrete claims with sources to abstract generalities. You do "
        "not invent details that the research notes do not contain."
    ),
    llm=heavy_llm,          # better writing → use the bigger model
    verbose=True,
    allow_delegation=False,
)

critic = Agent(
    role="Editorial Critic",
    goal=(
        "Review the writer's draft for factual accuracy, missing context, weak claims, "
        "and structural issues. Be specific. Quote the problematic line."
    ),
    backstory=(
        "You are a sharp-eyed editor. Your job is to find what's wrong, not to "
        "praise what's right. You produce numbered lists of concrete issues."
    ),
    llm=critic_llm,     # critique is structured → cheap model is fine
    verbose=True,
    allow_delegation=False,
)


# --- Tasks ----------------------------------------------------------------
# Each task is bound to one agent. `context=[other_task]` makes that task's
# output flow into this task's prompt. This is CrewAI's handoff mechanism.

def build_tasks(query: str):
    research_task = Task(
        description=(
            f"Research the following question and produce structured notes:\n\n"
            f"QUESTION: {query}\n\n"
            f"Output a bulleted list of facts, each with a source URL. "
            f"Include any numbers, dates, or definitions needed to answer fully. "
            f"Aim for 8-15 bullet points."
        ),
        expected_output="Markdown bulleted research notes with inline source URLs.",
        agent=researcher,
    )

    write_task = Task(
        description=(
            "Using the research notes from the previous task, write a 200-300 word "
            "report that answers the original question clearly. Use markdown headers "
            "if it helps structure. Do not invent facts beyond the notes."
        ),
        expected_output="A 200-300 word markdown report.",
        agent=writer,
        context=[research_task],   # research notes flow in here
    )

    critique_task = Task(
        description=(
            "Review the report from the previous task. Produce a numbered list of "
            "specific issues: factual problems, weak or unsupported claims, missing "
            "context, structural issues. If the report is genuinely strong, say so "
            "and explain why — do not invent issues."
        ),
        expected_output="Numbered list of issues, OR a short approval statement.",
        agent=critic,
        context=[write_task],
    )

    return [research_task, write_task, critique_task]


# --- Run ------------------------------------------------------------------

def run_crew(query: str):
    tasks = build_tasks(query)
    crew = Crew(
        agents=[researcher, writer, critic],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
    result = crew.kickoff()
    return result, tasks

def run_crew_with_revision(query: str, max_revisions: int = 2):
    """Sequential pipeline + manual critic loop.

    The critic's verdict is parsed; if it contains substantive issues, we
    construct a revision task that gives the writer the original notes plus
    the critique, and re-run.
    """
    research_task = Task(
        description=f"Research this question and produce structured notes:\n\n{query}",
        expected_output="Markdown bulleted research notes with inline source URLs.",
        agent=researcher,
    )

    initial_write_task = Task(
        description=(
            "Using the research notes, write a 200-300 word markdown report "
            "answering the question. Do not invent facts."
        ),
        expected_output="A 200-300 word markdown report.",
        agent=writer,
        context=[research_task],
    )

    # Phase 1: research + initial draft
    Crew(
        agents=[researcher, writer],
        tasks=[research_task, initial_write_task],
        process=Process.sequential,
        verbose=False,
    ).kickoff()

    current_draft = initial_write_task.output.raw
    research_notes = research_task.output.raw

    # Phase 2: critic loop
    for revision in range(max_revisions):
        critique_task = Task(
            description=(
                f"Review this report for factual accuracy, weak claims, and "
                f"structural issues. If the report is strong, respond with "
                f"'APPROVED' as the first word.\n\n"
                f"REPORT:\n{current_draft}"
            ),
            expected_output="Either 'APPROVED' followed by reasoning, OR a numbered list of issues.",
            agent=critic,
        )
        Crew(agents=[critic], tasks=[critique_task],
             process=Process.sequential, verbose=False).kickoff()

        critique = critique_task.output.raw
        print(f"\n--- Critique (revision {revision}) ---\n{critique[:500]}\n")

        if critique.strip().upper().startswith("APPROVED"):
            print(f"  [critic] APPROVED after {revision} revision(s)")
            return current_draft

        # Revise
        revise_task = Task(
            description=(
                f"Revise this report based on the critique below. Keep what works; "
                f"fix what the critic flagged. Do not invent facts beyond the notes.\n\n"
                f"NOTES:\n{research_notes}\n\n"
                f"PREVIOUS DRAFT:\n{current_draft}\n\n"
                f"CRITIQUE:\n{critique}"
            ),
            expected_output="A revised 200-300 word markdown report.",
            agent=writer,
        )
        Crew(agents=[writer], tasks=[revise_task],
             process=Process.sequential, verbose=False).kickoff()
        current_draft = revise_task.output.raw

    print(f"  [critic] Hit max_revisions={max_revisions}, returning latest draft")
    return current_draft


if __name__ == "__main__":
    query = (
        "What is the difference between LangGraph and CrewAI, and when would a "
        "team choose one over the other in 2026?"
    )
    print(f"\n{'='*70}\nQUERY: {query}\n{'='*70}\n")

    result, tasks = run_crew(query)

    print("\n" + "="*70)
    print("FINAL RESULT (last task output)")
    print("="*70)
    print(result)

    print("\n" + "="*70)
    print("INDIVIDUAL TASK OUTPUTS")
    print("="*70)
    for t in tasks:
        print(f"\n--- {t.agent.role} ---")
        print(t.output.raw[:800])
        print("...")
    print("\n" + "="*70)
    print("WITH CRITIC LOOP")
    print("="*70)
    final = run_crew_with_revision(
        "What is the difference between LangGraph and CrewAI?",
        max_revisions=2,
    )
    print("\nFINAL APPROVED DRAFT:\n")
    print(final)


