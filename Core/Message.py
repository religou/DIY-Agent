from pydantic import BaseModel
from typing import Literal, Optional, Dict
from datetime import datetime

MessageRole = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = None
    metadata: Optional[Dict[str, any]] = None

    def __init__(self, role: MessageRole, content: str, **kwargs):
        super().__init__(
            role = role,
            content = content,
            timestamp  = kwargs.get("timestamp", datetime.now()),
            metadata = kwargs.get("metadata", {})
        )

    def to_dict(self) -> Dict[str, any]:
        return {
            "role": self.role,
            "content": self.content
        }

    def __str__(self) -> str:
        return f"[{self.role}] {self.content}"