import httpx

# --- Tool implementations (the actual Python code) ---

def calculator(expression: str) -> str:
    """Evaluate a math expression. Restricted for safety."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only digits and + - * / ( ) . allowed"
    try:
        # eval is fine here ONLY because we whitelisted characters above.
        # Never use raw eval on LLM output in production.
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def fetch_url(url: str) -> str:
    """Fetch a URL and return the first 2000 chars of text."""
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        r.raise_for_status()
        return r.text[:2000]
    except Exception as e:
        return f"Error fetching {url}: {e}"


# --- Tool schemas (what the model sees) ---

TOOLS_CLAUDE_FORMAT = [
    {
        "name": "calculator",
        "description": "Evaluate a basic math expression. Use for any arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression like '23 * (4 + 7)'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch the contents of a web page. Returns first 2000 chars.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL including https://"}
            },
            "required": ["url"],
        },
    },
]

TOOLS_OPENAI_FORMAT = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a basic math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a web page; returns first 2000 chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
]

# Dispatch table — maps tool name to the function that runs it
TOOL_FUNCS = {
    "calculator": calculator,
    "fetch_url": fetch_url,
}

