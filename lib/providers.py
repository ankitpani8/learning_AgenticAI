"""Role-based startup model selection with health-check protocol.

Best-practice pattern for production agents:
  1. Code requests a model by ROLE, never by name. Roles encode intent
     ("heavy" = strong synthesis, "critic" = bounded structured output, etc.).
  2. Each role has an ordered preference chain of (model, provider) options.
  3. At startup, we ping each option in order. The first one that returns a
     valid response wins -- that's the bound model for the rest of the run.
  4. If no option responds, we fail fast at startup rather than crashing
     halfway through a multi-agent pipeline.

Supported providers (configured below):
  - gemini    -- requires GEMINI_API_KEY
  - ollama    -- requires a running Ollama service on localhost:11434
  - anthropic -- requires ANTHROPIC_API_KEY plus paid API credit (note:
                 a claude.ai subscription does NOT include API credits;
                 see console.anthropic.com -> Billing)

Why this matters:
  - Decouples agent logic from provider choice. Add a new provider tomorrow,
    update the preference chain, no agent code changes.
  - Catches quota/auth/network issues before burning tokens.
  - Makes model-routing decisions explicit and reviewable. Critics use cheap
    local models because that's encoded in policy, not buried in agent code.

Production note:
  This module covers startup selection. Production systems typically layer a
  per-call retry policy on top (exponential backoff, transient-error handling)
  for mid-run resilience. Module 5 covers that pattern.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH)


# ---------------------------------------------------------------------------
# Role policy — the one place that knows which models fit which jobs.
# ---------------------------------------------------------------------------

ROLE_PREFERENCES: dict[str, List[Tuple[str, str]]] = {
    # Strong synthesis / writing tasks.
    "heavy": [
        ("gemini-2.5-flash-lite", "gemini"),
        ("gemini-2.5-flash", "gemini"),
        ("qwen2.5:1.5b", "ollama"),
        # Documented option: Claude Sonnet for strong synthesis. Requires
        # paid API credit at console.anthropic.com (claude.ai subscription
        # does not include API access). Reorder this entry to make Claude
        # primary for the heavy role.
        ("claude-sonnet-4-6", "anthropic"),
    ],
    
    # Bounded reasoning: research, summarization, classification.
    "light": [
        ("qwen2.5:1.5b", "ollama"),
        ("gemini-2.5-flash-lite", "gemini"),
        ("gemini-2.5-flash", "gemini"),
        ("claude-haiku-4-5", "anthropic"),
    ],
    # Critic / evaluator: structured output (find issues, return list).
    # Local-first by design — critics are frequent callers and a small local
    # model demonstrates the multi-agent token-tax mitigation properly.
    "critic": [
        ("qwen2.5:1.5b", "ollama"),
        ("gemini-2.5-flash-lite", "gemini"),
        ("gemini-2.5-flash", "gemini"),
        ("claude-haiku-4-5", "anthropic"),
    ],
}


# ---------------------------------------------------------------------------
# Provider construction — opaque to callers.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSelection:
    """The result of running the selection protocol for one role."""
    role: str
    name: str
    provider: str

    def to_langchain(self, temperature: float = 0.3) -> BaseChatModel:
        """Build a LangChain ChatModel for the selected provider."""
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.name,
                google_api_key=os.environ["GEMINI_API_KEY"],
                temperature=temperature,
            )
        if self.provider == "ollama":
            return ChatOllama(model=self.name, temperature=temperature)
        if self.provider == "anthropic":
            # Lazy-imported so that users without langchain-anthropic installed
            # (or without an Anthropic API key) can still use the rest of the chain.
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.name,
                api_key=os.environ["ANTHROPIC_API_KEY"],
                temperature=temperature,
            )
        raise ValueError(f"Unknown provider: {self.provider}")

    def to_crewai(self, temperature: float = 0.3):
        """Build a CrewAI LLM for the selected provider. Lazy-imported so that
        modules using only LangChain don't need CrewAI installed."""
        from crewai import LLM
        if self.provider == "gemini":
            return LLM(
                model=f"gemini/{self.name}",
                api_key=os.environ["GEMINI_API_KEY"],
                temperature=temperature,
            )
        if self.provider == "ollama":
            return LLM(
                model=f"ollama/{self.name}",
                base_url="http://localhost:11434",
                temperature=temperature,
            )
        if self.provider == "anthropic":
            return LLM(
                model=f"anthropic/{self.name}",
                api_key=os.environ["ANTHROPIC_API_KEY"],
                temperature=temperature,
            )
        raise ValueError(f"Unknown provider: {self.provider}")


# ---------------------------------------------------------------------------
# The protocol itself.
# ---------------------------------------------------------------------------

def _health_check(model: BaseChatModel, label: str) -> bool:
    """Send a trivial 'hi' and confirm we get a non-empty response."""
    try:
        response = model.invoke([HumanMessage(content="hi")])
        ok = bool(getattr(response, "content", "").strip())
        print(f"  [health] {label}: {'OK' if ok else 'EMPTY'}")
        return ok
    except Exception as e:
        print(f"  [health] {label}: FAIL ({type(e).__name__}: {str(e)[:100]})")
        return False


def select_model_for_role(role: str) -> ModelSelection:
    """Try each option in the role's preference chain. Return the first that responds."""
    if role not in ROLE_PREFERENCES:
        raise ValueError(f"Unknown role '{role}'. Known: {list(ROLE_PREFERENCES)}")

    print(f"\n[selection] role={role}")
    for name, provider in ROLE_PREFERENCES[role]:
        candidate = ModelSelection(role=role, name=name, provider=provider)
        try:
            model = candidate.to_langchain(temperature=0)
        except KeyError as e:
            print(f"  [build]  {provider}:{name}: SKIP (missing env var {e})")
            continue
        except ImportError as e:
            print(f"  [build]  {provider}:{name}: SKIP (missing package: {e.name})")
            continue
        except Exception as e:
            print(f"  [build]  {provider}:{name}: FAIL ({type(e).__name__}: {str(e)[:80]})")
            continue
        if _health_check(model, f"{provider}:{name}"):
            print(f"  -> selected {provider}:{name}")
            return candidate

    raise RuntimeError(
        f"No working provider for role={role}. Tried: {ROLE_PREFERENCES[role]}. "
        f"Check API keys, quota, and Ollama service status."
    )


def select_all_models(roles: List[str]) -> dict[str, ModelSelection]:
    """Run the protocol for multiple roles. Call once at startup."""
    print("=" * 60)
    print("MODEL SELECTION PROTOCOL")
    print("=" * 60)
    selections = {role: select_model_for_role(role) for role in roles}
    print("=" * 60)
    print("BOUND MODELS:")
    for role, sel in selections.items():
        print(f"  {role:8s} -> {sel.provider}:{sel.name}")
    print("=" * 60)
    return selections