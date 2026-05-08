"""Tools for the LangGraph agent. Same logic as Module 1, LangChain-decorated."""
from pathlib import Path
import httpx
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression. Only digits, + - * / ( ) . and spaces allowed."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only digits and + - * / ( ) . allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


@tool
def fetch_url(url: str) -> str:
    """Fetch a web page and return the first 2000 characters of HTML."""
    try:
        r = httpx.get(url, timeout=10, follow_redirects=True)
        r.raise_for_status()
        return r.text[:2000]
    except Exception as e:
        return f"Error fetching {url}: {e}"


@tool
def read_file(path: str) -> str:
    """Read a local text file and return its first 5000 characters."""
    try:
        return Path(path).read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error reading {path}: {e}"


ALL_TOOLS = [calculator, fetch_url, read_file]