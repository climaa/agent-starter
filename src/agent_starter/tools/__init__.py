"""Tool registry. Adding a tool = drop one module in this package that calls `register()`.

Modules are discovered on first `get_tools()`; files starting with `_` are skipped.
"""

import importlib
import pkgutil

from langchain_core.tools import BaseTool

_REGISTRY: dict[str, BaseTool] = {}
_discovered = False


def register(tool: BaseTool) -> BaseTool:
    if tool.name in _REGISTRY:
        raise ValueError(f"duplicate tool name: {tool.name}")
    _REGISTRY[tool.name] = tool
    return tool


def _discover() -> None:
    global _discovered
    if _discovered:
        return
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")
    _discovered = True


def get_tools() -> list[BaseTool]:
    _discover()
    return list(_REGISTRY.values())


def get_tool(name: str) -> BaseTool:
    _discover()
    if name not in _REGISTRY:
        raise KeyError(f"unknown tool: {name}")
    return _REGISTRY[name]
