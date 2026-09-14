from typing import Optional, List
from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry, Config, Message


MY_REACT_PROMPT = """你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤:

Thought: 分析当前问题，思考需要什么信息或采取什么行动。
Action: 选择一个行动，格式必须是以下之一:
- `{{tool_name}}[{{tool_input}}]` - 调用指定工具
- `Finish[最终答案]` - 当你有足够信息给出最终答案时

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循:工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动:
"""


class MyReActAgent(ReActAgent):
    def __init__(
        self, 
        name: str, 
        llm: HelloAgentsLLM, 
        tool_registry: ToolRegistry,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_steps: int = 5,
        custom_prompt: Optional[str] = None,
        ):
        super().__init__(name=name, llm=llm, system_prompt=system_prompt, config=config)
        self.tool_registry = tool_registry
        self.max_steps = max_steps
        self.current_history: List[str] = []
        self.prompt_template = custom_prompt if custom_prompt is not None else MY_REACT_PROMPT
        print(f"✅ {name} 初始化完成，最大步数: {max_steps}")

    def run(self, input_text: str, **kwargs) -> str:

        self.current_history = []
        current_step = 0

        print(f"\n🚀 {self.name} 开始处理问题: {input_text}")

        # 工具集在单次问答内不变，只需生成一次描述，避免每步重复计算。
        tool_desc = self.tool_registry.get_tools_description()

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 🔄 当前步骤: 第 {current_step} 步 ---")

            history_str = "\n".join(self.current_history)
            prompt = self.prompt_template.format(tools=tool_desc, question=input_text, history=history_str)
            print(f"\n📝 当前生成的提示:\n{prompt}")

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm.invoke(messages, **kwargs)

            # 无有效响应时无法继续解析，直接结束本轮执行。
            if not response_text:
                print("错误:LLM未能返回有效响应，流程终止。")
                break

            thought, action = self._parse_output(response_text)

            # 模型输出必须至少包含可解析的 Action，否则终止避免空转。
            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break

            print(f"💭 思考: {thought}")

            if action.startswith("Finish["):
                final_answer = self._parse_action_input(action) or "（未能解析出有效的最终答案）"
                self.add_message(Message(role="user", content=input_text))
                self.add_message(Message(role="assistant", content=final_answer))
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            if not tool_name:
                observation = f"错误:无法解析的Action格式 '{action}'。"
            else:
                print(f"🎬 行动: {tool_name}[{tool_input}]")
                try:
                    observation = self.tool_registry.execute_tool(tool_name, tool_input)
                except Exception as error:
                    # 将工具异常转为观察结果，让模型据此调整下一步动作。
                    observation = f"错误:工具 '{tool_name}' 执行失败: {error}"

            # 记录完整的 Thought/Action/Observation 轨迹供下一步推理使用。
            self.current_history.append(f"Thought: {thought}")
            self.current_history.append(f"Action: {action}")
            self.current_history.append(f"Observation: {observation}")

        # 达到最大步数
        final_answer = "抱歉，我无法在限定步数内完成这个任务。"
        self.add_message(Message(role="user", content=input_text))
        self.add_message(Message(role="assistant", content=final_answer))
        return final_answer
