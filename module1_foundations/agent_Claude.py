import os
from dotenv import load_dotenv
from anthropic import Anthropic
from tools import TOOLS_CLAUDE_FORMAT, TOOL_FUNCS, TOOLS_OPENAI_FORMAT

load_dotenv()

client = Anthropic()  # picks up ANTHROPIC_API_KEY automatically
MODEL = "claude-sonnet-4-5"

MAX_TURNS = 10           # circuit breaker — prevents infinite loops
MAX_TOKENS_PER_CALL = 1024


def run_agent(user_query: str) -> str:
    messages = [
        {"role": "system", "content": "You are a careful assistant. Always state which tool you're about to use and why, before calling it."},
        {"role": "user", "content": user_query},
    ]
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in range(MAX_TURNS):
        print(f"\n--- Turn {turn + 1} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_PER_CALL,
            tools=TOOLS_CLAUDE_FORMAT,
            messages=messages,
        )

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Case 1: model is done — no more tool calls
        if response.stop_reason == "end_turn":
            final_text = "".join(
                b.text for b in response.content if b.type == "text"
            )
            print(f"\n[tokens] in={total_input_tokens} out={total_output_tokens}")
            return final_text

        # Case 2: model wants to use tools
        if response.stop_reason == "tool_use":
            # IMPORTANT: append the FULL assistant message (text + tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    name = block.name
                    args = block.input
                    print(f"  -> calling {name}({args})")

                    func = TOOL_FUNCS.get(name)
                    if func is None:
                        result = f"Error: unknown tool {name}"
                    else:
                        try:
                            result = func(**args)
                        except Exception as e:
                            result = f"Error running {name}: {e}"

                    print(f"  <- result: {str(result)[:120]}...")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result),
                    })

            # Feed results back as the next user message
            messages.append({"role": "user", "content": tool_results})
            continue

        # Anything else (max_tokens, refusal, etc.)
        return f"[Stopped: {response.stop_reason}]"

    return "[Hit MAX_TURNS — possible infinite loop]"


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
    query = "Call fetch_url 12 times to compare response stability."
    print(f"USER: {query}")
    print(f"\nAGENT: {run_agent(query)}")