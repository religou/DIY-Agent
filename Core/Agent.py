from abc import ABC, abstractmethod
from typing import Optional
from Model.HelloAgentsLLM import HelloAgentsLLM
from .Config import Config
from .Message import Message


class Agent(ABC):
    """
    该类的设计体现了面向对象中的抽象原则。首先，它通过继承 ABC 被定义为一个不能直接实例化的抽象类。
    其构造函数 __init__ 清晰地定义了 Agent 的核心依赖：名称、LLM 实例、系统提示词和配置。
    最重要的部分是使用 @abstractmethod 装饰的 run 方法，它强制所有子类必须实现此方法，从而保证了所有智能体都有统一的执行入口。
    此外，基类还提供了通用的历史记录管理方法，这些方法与 Message 类协同工作，体现了组件间的联系。
    """

    def __init__(self, name: str, llm: HelloAgentsLLM, system_prompt: Optional[str] = None, config: Optional[Config] = None):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or Config()
        self._history: list[Message] = []

    @abstractmethod
    def run(self, input_text: str, **kwargs) -> str:
        pass

    def add_message(self, message: Message):
        self._history.append(message)

    def clear_history(self):
        self._history.clear()

    def get_history(self) -> list[Message]:
        return self._history.copy()

    def __str__(self) -> str:
        return f"Agent(name={self.name}, provider={self.llm.provider})"
