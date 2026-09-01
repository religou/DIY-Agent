import ast
import os
import sys

# 允许脚本既可以通过 `python Agent/PlanAndSolveAgent.py` 直接运行，
# 也可以通过 `python -m Agent.PlanAndSolveAgent` 以模块方式运行。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Model.HelloAgentsLLM import HelloAgentsLLM

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""
class Planner:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def plan(self, question: str):
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        message = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages = message) or ""

        try:
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []

EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""

class Excutor:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client

    def excutor(self, question: str, plan: list[str]) -> str:
        history = ""

        for i, step in enumerate(plan):
            print(f"正在执行{i+1}/{len(plan)}\n: {step}")

            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history,
                current_step=step
            )

            message = [{"role": "user", "content": prompt}]

            response_text = self.llm_client.think(messages=message)

            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            
            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        final_answer = response_text        
        return final_answer

class PlanAndSolveAgent:
    def __init__(self, llm_client: HelloAgentsLLM):
        self.llm_client = llm_client
        self.Planner = Planner(self.llm_client)
        self.Excutor = Excutor(self.llm_client) 

    def run(self, question: str):
        plan = self.Planner.plan(question)
        if not plan:
            return "无法生成可行的执行计划。"

        final_answer = self.Excutor.excutor(question, plan)
        print(f"🎉 最终答案: {final_answer}")
        return final_answer

if __name__ == "__main__":
    llm_client = HelloAgentsLLM()
    agent = PlanAndSolveAgent(llm_client)
    question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    agent.run(question)
