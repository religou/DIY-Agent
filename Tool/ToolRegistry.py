from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable

from Tool.Tool import Tool

logger = logging.getLogger(__name__)
_MISSING = object()  # Sentinel: distinguish between "no default value" and "default value is None"


@dataclass(frozen=True)
class FunctionEntry:
    description: str
    func: Callable[..., Any]


class ToolRegistry:
    """A registry for managing and storing tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, FunctionEntry] = {}

    def register_tool(self, tool: Tool) -> None:
        """Register a new tool in the registry."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        logger.info("Tool '%s' registered successfully.", tool.name)

    def register_function(self, name: str, description: str,
                          func: Callable[..., Any]) -> None:
        """Register a new function in the registry."""
        if name in self._functions:
            raise ValueError(f"Function with name '{name}' is already registered.")
        self._functions[name] = FunctionEntry(description, func)
        logger.info("Function '%s' registered successfully.", name)

    def get_tool(self, name: str) -> Tool:
        """Retrieve a registered tool by its name."""
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Tool '{name}' is not registered.") from None

    def execute(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered tool or function by its name."""
        if name in self._tools:
            return self._tools[name].run(*args, **kwargs)  # 按 Tool 实际 API 改
        if name in self._functions:
            return self._functions[name].func(*args, **kwargs)
        raise KeyError(f"'{name}' is not registered.")

    def get_tools_description(self) -> str:
        """Get a textual description of all registered tools and functions."""
        lines = [f"{t.name}: {t.description}" for t in self._tools.values()]
        lines += [f"{n}: {e.description}" for n, e in self._functions.items()]
        return "\n".join(lines) if lines else "No tools registered."

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """Return a list of schemas for all registered tools and functions, corresponding to OpenAI's tools field."""
        schemas = [tool.to_openai_schema() for tool in self._tools.values()]
        schemas += [self._function_schema(n, e) for n, e in self._functions.items()]
        return schemas

    @staticmethod
    def _function_schema(name: str, entry: FunctionEntry) -> dict[str, Any]:
        """Automatically generate a parameter schema from the function signature."""
        type_map = {str: "string", int: "integer", float: "number",
                    bool: "boolean", list: "array", dict: "object"}
        properties: dict[str, Any] = {}
        required: list[str] = []
        for pname, p in inspect.signature(entry.func).parameters.items():
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            prop: dict[str, Any] = {"type": type_map.get(p.annotation, "string")}
            if p.default is inspect.Parameter.empty:
                required.append(pname)
            else:
                prop["description"] = f"(Default: {p.default})"
            properties[pname] = prop
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": entry.description,
                "parameters": {"type": "object",
                               "properties": properties,
                               "required": required},
            },
        }