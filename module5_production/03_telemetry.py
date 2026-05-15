"""Lightweight telemetry: per-request token and latency tracking.

Production analog: structured logs to Datadog/Honeycomb/CloudWatch, with
each request as a trace. We log to stdout as JSON lines -- pipeable to
anywhere later.
"""
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict


@dataclass
class RequestTelemetry:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_query: str = ""
    started_at: float = field(default_factory=time.time)
    duration_ms: float = 0.0
    cache_hit: str | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    rag_chunks: int = 0
    error: str | None = None

    def add_llm_usage(self, input_t: int, output_t: int) -> None:
        self.llm_calls += 1
        self.input_tokens += input_t
        self.output_tokens += output_t

    def finalize(self) -> None:
        self.duration_ms = (time.time() - self.started_at) * 1000

    def emit(self) -> None:
        # Stable JSON output for log aggregators
        print("[telemetry] " + json.dumps(asdict(self)))


@contextmanager
def telemetry(query: str):
    """Use in a `with` block to ensure finalize/emit always runs."""
    t = RequestTelemetry(user_query=query[:200])
    try:
        yield t
    except Exception as e:
        t.error = f"{type(e).__name__}: {str(e)[:200]}"
        raise
    finally:
        t.finalize()
        t.emit()