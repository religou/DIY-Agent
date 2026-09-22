
from abc import ABC, abstractmethod
from typing import Dict, List
from ToolParameter import ToolParameter

class Tool(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, parameters: Dict[str, any]) -> str:
        pass

    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """
        Returns a list of ToolParameter objects that define the parameters required by this tool.
        """
        pass