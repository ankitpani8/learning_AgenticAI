# Python Best Practices

## Virtual environments
Always use a virtual environment for Python projects. `python -m venv .venv`
creates one in the current directory. Activate it before installing packages
so dependencies don't pollute the system Python.

## Type hints
Type hints improve code clarity and catch bugs early. Use them on all
function signatures. `from typing import Optional, List` covers most cases.

## Dependency pinning
Pin dependencies in `requirements.txt` using `pip freeze`. This ensures
reproducible builds across machines.

## Error handling
Don't catch bare `Exception`. Catch specific exceptions. Bare exceptions
hide bugs and make debugging painful.