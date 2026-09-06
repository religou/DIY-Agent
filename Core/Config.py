import os
from pydantic import BaseModel
from typing import Dict, Any

class Config(BaseModel):
    default_mode: str = "Qwen3-8B"
    default_provider: str = "siliconflow"
    temperature: float = 0.7
    max_tokens: int = None


    debug: bool = False
    log_level: str = "INFO"

    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug = os.getenv("DEBUG", "false").lower() == "true",
            log_level = os.getenv("LOG_LEVEL", "info"),
            temperature = float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens = int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,
        )

    def to_dict(self) -> Dict[str, any]:
        return self.model_dump()