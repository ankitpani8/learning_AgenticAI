"""Retry policies for LLM calls. Distinguishes failure types.

Transient errors -> retry with exponential backoff + jitter
Rate limits      -> retry after suggested delay
Bad input        -> fail fast (don't retry)
Auth errors      -> fail fast and alert
"""
import asyncio
import random
from tenacity import (
    AsyncRetrying, retry_if_exception_type, stop_after_attempt,
    wait_exponential_jitter, before_sleep_log,
)
import logging

log = logging.getLogger("agent.retry")


# We define which exceptions are "retryable transient" vs "permanent".
# In real code these come from the provider SDKs; we use broad types here
# because LangChain wraps provider exceptions inconsistently.

class TransientError(Exception):
    """Network blip, 5xx, timeout. Safe to retry."""


class RateLimitError(Exception):
    """429. Retry after a delay."""


class PermanentError(Exception):
    """4xx (except 429), schema violation. Do NOT retry."""


def classify_exception(exc: Exception) -> type[Exception]:
    """Map any exception to one of our three retry categories.

    In production this inspects status codes from the provider SDK. Here we
    pattern-match on the exception string -- crude but transparent.
    """
    msg = str(exc).lower()
    if "429" in msg or "rate" in msg or "quota" in msg:
        return RateLimitError
    if "401" in msg or "403" in msg or "invalid api key" in msg:
        return PermanentError
    if "400" in msg or "schema" in msg or "validation" in msg:
        return PermanentError
    return TransientError


async def call_with_retry(coro_factory, max_attempts: int = 3):
    """Run an async callable with retry. coro_factory must be a no-arg async
    function (not an awaited coroutine) so we can call it fresh each attempt.

    Usage:
        result = await call_with_retry(lambda: llm.ainvoke(messages))
    """
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(initial=1, max=10, jitter=2),
        retry=retry_if_exception_type((TransientError, RateLimitError)),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    ):
        with attempt:
            try:
                return await coro_factory()
            except Exception as e:
                classified = classify_exception(e)
                if classified == PermanentError:
                    log.error(f"Permanent error, not retrying: {e}")
                    raise  # bypass retry loop
                # Re-raise as the classified type so tenacity sees it
                raise classified(str(e)) from e