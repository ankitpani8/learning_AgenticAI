"""Tools for the research crew. Same logic as Module 2.1, CrewAI-format"""
from pathlib import Path
import httpx
from crewai.tools import tool


@tool("Web Page Fetcher")
def fetch_url(url: str) -> str:
    """Fetch a web page and return the first 4000 characters of its text content.
    Useful for gathering information from a specific URL. Input must be a full
    URL starting with http:// or https://."""
    if not url.startswith(("http://", "https://")):
        return f"Error: URL '{url}' must start with http:// or https://"
    try:
        r = httpx.get(url, timeout=15, follow_redirects=True,
                      headers={"User-Agent": "research-crew/0.1"})
        r.raise_for_status()
        return r.text[:4000]
    except Exception as e:
        return f"Error fetching {url}: {e}"


@tool("Calculator")
def calculator(expression: str) -> str:
    """Evaluate a basic math expression containing only digits, + - * / ( ) . and spaces.
    Use this for any arithmetic — do not compute mentally."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only digits and + - * / ( ) . allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"
    
@tool("File Reader")
def read_file(path: str) -> str:
    """Read a local text file and return its first 5000 characters."""
    try:
        return Path(path).read_text(encoding="utf-8")[:5000]
    except Exception as e:
        return f"Error reading {path}: {e}"