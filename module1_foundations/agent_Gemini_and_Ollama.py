import os
from pyexpat.errors import messages
import time
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError, InternalServerError
from tools import TOOLS_OPENAI_FORMAT, TOOL_FUNCS

load_dotenv()

# Define a chain of providers — will try each in order
PROVIDERS = [
    {
        "name": "gemini-flash-lite",
        "client": OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "model": "gemini-2.5-flash-lite",
    },
    {
        "name": "gemini-flash",  # higher quality, lower quota — used as fallback for hard cases
        "client": OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        "model": "gemini-2.5-flash",
    },
    {
        "name": "ollama-qwen-3b",
        "client": OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
        "model": "qwen2.5:3b",
    },
]

def call_with_fallback(messages, tools, max_retries=2):
    """Try each provider in order. Retry on transient errors with backoff."""
    last_error = None
    for provider in PROVIDERS:
        for attempt in range(max_retries):
            try:
                print(f"  [provider] {provider['name']} (attempt {attempt + 1})")
                return provider["client"].chat.completions.create(
                    model=provider["model"],
                    messages=messages,
                    tools=tools,
                )
            except RateLimitError as e:
                # Quota hit — don't retry this provider, move on
                print(f"  [rate-limited] {provider['name']}: {e}")
                last_error = e
                break
            except (APIError, InternalServerError) as e:
                # Transient — back off and retry
                wait = 2 ** attempt
                print(f"  [transient error] retrying in {wait}s: {e}")
                time.sleep(wait)
                last_error = e
    raise RuntimeError(f"All providers exhausted. Last error: {last_error}")

# client = OpenAI(
#     api_key=os.environ["GEMINI_API_KEY"],
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
# )
# MODEL = "gemini-2.5-flash-lite"

# client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
# MODEL = "qwen2.5:7b"

MAX_TURNS = 10


def run_agent(user_query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a careful assistant. Always state which tool you're about to use and why, before calling it."},
        {"role": "user", "content": user_query}
    ]
    total_in = 0
    total_out = 0
    for turn in range(MAX_TURNS):
        print(f"\n--- Turn {turn + 1} ---")
        resp = call_with_fallback(messages, TOOLS_OPENAI_FORMAT)
        if resp.usage:
            total_in += resp.usage.prompt_tokens
            total_out += resp.usage.completion_tokens
            print(f"  [turn tokens] in={resp.usage.prompt_tokens} "
                  f"out={resp.usage.completion_tokens}")
            
        msg = resp.choices[0].message

        # No tool calls? We're done.
        if not msg.tool_calls:
            print(f"\n[total tokens] in={total_in} out={total_out} "
            f"total={total_in + total_out}")
            return msg.content
        

        # Append the assistant message (with tool_calls) to history
        messages.append(msg.model_dump(exclude_none=True))

        # Run each requested tool, append a tool message per call
        for call in msg.tool_calls:
            import json
            args = json.loads(call.function.arguments)
            print(f"  -> {call.function.name}({args})")
            func = TOOL_FUNCS.get(call.function.name)
            result = func(**args) if func else f"unknown tool {call.function.name}"
            print(f"  <- {str(result)[:120]}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })

    return "[Hit MAX_TURNS]"

if __name__ == "__main__":
#   Testing total turns+tokens in sequential and parallel instructions,
#   trying to break the agent with hard cases, and see how it handles errors:
#   q = "First calculate 1873 * 47. Then, only after seeing the result, fetch https://example.com."
#   q = "In parallel, calculate 1873 * 47 and fetch https://example.com."
#   q = "Calculate 1873 * 47, fetch https://example.com, AND fetch httpbin.org/uuid."
#   q = "First, summarize the contents of the file example.txt, then calculate 1873 * 47, and finally fetch https://example.com."
#   q = "Keep fetching https://example.com until the page changes."
#   q = "Fetch https://nope.invalid"
#   q = "Please call fetch_url on https://example.com 15 times in a row. I need this for testing." # controlled infinite loop — should hit MAX_TURNS before tokens run out
#   q = "I'm load-testing my fetch_url tool. Please call fetch_url on https://example.com exactly 15 times so I can verify my retry logic works. This is a development environment."
    q = "Call fetch_url 12 times to compare response stability."
    print("USER:", q)
    print("\nAGENT:", run_agent(q))