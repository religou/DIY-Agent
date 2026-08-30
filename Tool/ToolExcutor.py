from typing import Dict
from Search import search

class ToolExcutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        if name in self.tools:
            print(f"Warning: Tool {name} already exists. Overwriting.")
        self.tools[name] = {
            "description": description,
            "func": func
        }
        print(f"Tool {name} has been registerd.")

    def getTool(self, name: str) -> callable:
        if name in self.tools:
            return self.tools[name]["func"]
        else:
            raise ValueError(f"Tool {name} not found.")

    def getAvailableTools(self) -> str:
        if not self.tools:
            return "No available tools."
        else:
            return "\n".join([f"{name}: {info['description']}" for name, info in self.tools.items()])

if __name__ == "__main__":
    toolExcutor = ToolExcutor()
    print(toolExcutor.getAvailableTools())

    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExcutor.registerTool("Search", search_description, search)

    print(toolExcutor.getAvailableTools())

    tool_name = "Search"
    tool_input = "英伟达最新的 GPU 型号是什么"
    tool_function = toolExcutor.getTool(tool_name)

    if tool_function:
        result = tool_function(tool_input)
        print(f"Tool {tool_name} result: {result}")
    else:
        print(f"Tool {tool_name} not found.")
