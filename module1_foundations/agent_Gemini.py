# agent.py — Gemini version
import os
from dotenv import load_dotenv
from openai import OpenAI
from tools import TOOLS_OPENAI_FORMAT, TOOL_FUNCS  # OpenAI-format schemas

load_dotenv()

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"
MAX_TURNS = 10


def run_agent(user_query: str) -> str:
    messages = [{"role": "user", "content": user_query}]

    for turn in range(MAX_TURNS):
        print(f"\n--- Turn {turn + 1} ---")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_OPENAI_FORMAT,
        )
        msg = resp.choices[0].message

        # No tool calls? We're done.
        if not msg.tool_calls:
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
    q = "What is 1873 * 47, then fetch https://example.com and tell me what's on it."
    print("USER:", q)
    print("\nAGENT:", run_agent(q))