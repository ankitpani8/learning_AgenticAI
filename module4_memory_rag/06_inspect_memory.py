"""Show everything the agent currently remembers. Production analog: a
settings page in a real assistant where users review and edit memory."""
import importlib
_memory = importlib.import_module("02_memory_stores")
_memory.inspect_memory()