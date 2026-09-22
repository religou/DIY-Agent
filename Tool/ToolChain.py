from typing import Any, Dict, List
from Tool.ToolRegistry import ToolRegistry

class ToolChain:
    """A chain of tools that can be executed in sequence."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None) -> None:
        """Add a step to the tool chain."""
        self.steps.append({
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key
        })

    def execute(self, registry: ToolRegistry, initial_input: str, context: Dict[str, Any] = None) -> str:
        """Execute the tool chain in sequence using the provided registry and initial input."""
        context = context or {}
        context["input"] = initial_input
        print(f"Executing tool chain '{self.name}' with initial input: {initial_input}")

        for i, step in enumerate(self.steps):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]

            try:
                tool_input = input_template.format(**context)
            except KeyError as e:
                raise ValueError(f"Missing key in context for input template: {e}") from e

            print(f"Step {i+1}: Executing tool '{tool_name}' with input: {tool_input}")

            result = registry.execute(tool_name, tool_input)
            if output_key:
                context[output_key] = result

        final_result = context[self.steps[-1]["output_key"]] if self.steps and self.steps[-1]["output_key"] else context["input"]
        return final_result


class ToolChainManager:
    """Manager for handling multiple tool chains."""
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.tool_chains: Dict[str, ToolChain] = {}

    def register_tool_chain(self, tool_chain: ToolChain) -> None:
        """Register a new tool chain."""
        self.tool_chains[tool_chain.name] = tool_chain
        print(f"Registered tool chain '{tool_chain.name}'")

    def execute_tool_chain(self, name: str, initial_input: str, context: Dict[str, Any] = None) -> str:
        """Execute a registered tool chain by name."""
        if name not in self.tool_chains:
            raise ValueError(f"Tool chain '{name}' is not registered.")
        return self.tool_chains[name].execute(self.registry, initial_input, context)

    def list_tool_chains(self) -> List[str]:
        """List the names of all registered tool chains."""
        return list(self.tool_chains.keys())

