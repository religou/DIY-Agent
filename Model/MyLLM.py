import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
from typing import Optional
from openai import OpenAI
from Model.HelloAgentsLLM import HelloAgentsLLM
from dotenv import load_dotenv

load_dotenv()

class MyLLM(HelloAgentsLLM):
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None, provider: Optional[str] = "auto", **kwargs):
        if provider.lower() == "modelscope":
            print("正在使用自定义的 ModelScope Provider")
            self.provider = "modelscope"
            self.api_key = api_key or os.getenv("MODELSCOPE_API_KEY")
            self.base_url = base_url or "https://api-inference.modelscope.cn/v1/"

            if not self.api_key:
                raise ValueError("ModelScope API Key not found. Please set ModelScope_API_KEY")

            self.model = model or os.getenv("LLM_MODEL_ID")
            self.temperature = kwargs.get("temperature", 0.7)
            self.max_token = kwargs.get("max_token")
            self.timeout = kwargs.get("timeout", 60)

            self.client = OpenAI(api_key = self.api_key, base_url = self.base_url, timeout = self.timeout)
            
        else:
            super().__init__(model = model, apiKey = api_key, baseURL = base_url)

if __name__ == "__main__":
    llm = MyLLM(provider = "ModelScope")

    message = [{"role": "user", "content": "介绍你自己"}]

    response = llm.think(messages = message)

    for chunk in response:
        pass