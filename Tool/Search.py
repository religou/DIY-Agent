import os
from serpapi import SerpApiClient
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv()

def search(query: str) -> str:
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SEARCH_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn"
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])

        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]

        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return  results["knowledge_graph"]["description"]

        if "organic_results" in results and results["organic_results"]:
            # 限制结果数量为 3 个
            organic_results = results["organic_results"][:3]
            return "\n".join([
                f"{i+1}. [{result['title']}]({result['link']}) - {result['snippet']}"
                for i, result in enumerate(organic_results)
            ])

        return "未找到相关结果。"

    except Exception as e:
        return f"搜索时出错: {str(e)}"