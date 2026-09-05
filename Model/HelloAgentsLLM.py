import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

# 加载 .env 文件中的环境变量，并给到后续的 os.getenv() 调用
load_dotenv()

class HelloAgentsLLM:
    def __init__(self, baseURL: str = None, model: str = None, apiKey: str = None, timeout: int = None):
        self.baseURL = baseURL or os.getenv("LLM_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL_ID")
        self.apiKey = apiKey or os.getenv("LLM_API_KEY")
        self.timeout = timeout or int(os.getenv("LLM_TIMEOUT", "60"))
        if not all([self.model, self.apiKey, self.baseURL]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        # 将 OpenAI 实例化，相当于创建了一个大脑
        self.client = OpenAI(api_key = self.apiKey, base_url = self.baseURL, timeout = self.timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float  = 0) -> Optional[str]:
        print(f"🧠 正在调用 {self.model} 模型...")

        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                temperature = temperature,
                stream = True
            )

            print(f"{self.model} 模型响应成功")

            collected_content = []

            # 这里之所以用 chunk，是因为上面的 stream 设置为了 True，也就是流式输出
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end = "", flush = True)
                collected_content.append(content)
                
            return "".join(collected_content)
        except Exception as e:
            print(f"调用模型时出错: {e}")
            return None

    # 自动检查 LLM Provider
    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:

        if os.getenv("SILICONFLOW_API_KEY"): return "siliconflow"
        if os.getenv("MODELSCOPE_API_KEY"): return "modelscope"
        if os.getenv("OPENAI_API_KEY"): return "openai"
        if os.getenv("ZHIPU_API_KEY"): return "zhipu"

        actual_api_key = api_key or os.getenv("LLM_API_KEY")
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")

        if actual_api_key:
            if actual_api_key.startswith("sk-"): return "siliconflow"
            if actual_api_key.startswith("ms-"): return "modelscope"

        if actual_base_url:
            base_url_lower = actual_base_url.lower()
            if "siliconflow.cn" in base_url_lower: return "siliconflow"
            if "modelscope.cn" in base_url_lower: return "modelscope"
            if "bigmodel.cn" in base_url_lower: return "zhipu"
            if "localhost" in base_url_lower:
                if ":11434" in base_url_lower:
                    return "ollma"
                if ":8000" in base_url_lower:
                    return "vllm"
                return "local"
        return "auto"

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple:
        if self.provider == "openai":
            resolved_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api.openai.com/v1"
            return resolved_api_key, resolved_base_url

        elif self.provider == "modelscope":
            resolved_api_key = api_key or os.getenv("MODELSCOPE_API_KEY") or os.getenv("LLM_API_KEY")
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or "https://api-inference.modelscope.cn/v1/"
            return resolved_api_key, resolved_base_url


if __name__ == "__main__":
    try:
        llm = HelloAgentsLLM()
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "撰写递归排序的算法"}
        ]
        response = llm.think(messages=messages)
        if response:
            print("\n\n--- 完整模型响应 ---")
            print(response)
        
    except ValueError as e:
        print(e)