from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts#从 prompt_loader.py 里导入 load_system_prompts()。这个函数会读取 main_prompt.txt。
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import (AgentRuntimeContext, log_before_model, monitor_tool,
                                    report_prompt_switch)
from memory import MemoryExtractor, MemoryManager, create_memory_manager
from utils.config_handler import memory_conf
"""
streamlit run app.py
        ↓
用户在网页输入问题
        ↓
app.py 调用 ReactAgent.execute_stream(prompt)
        ↓
react_agent.py 创建的 Agent 接收问题
        ↓
读取 main_prompt.txt，判断用户意图
        ↓
根据意图选择工具：

        ├── 普通知识问题
        │       ↓
        │   rag_summarize()，调用 agent_tools.py 里的 rag_summarize(query)
        │       ↓
        │   rag_service.py，rag_summarize 再调用 rag_service.py
        │       ↓
        │   vector_store.py 查 Chroma 知识库，rag_service.py 再调用 vector_store.py 提供的 retriever 去查 Chroma 向量库
        │       ↓
        │   大模型总结资料
        │       ↓
        │   返回回答

        ├── 天气/环境问题
        │       ↓
        │   get_user_location()
        │       ↓
        │   get_weather(city)
        │       ↓
        │   大模型生成建议
        │       ↓
        │   返回回答

        └── 使用报告问题
                ↓
            get_user_id()
                ↓
            get_current_month()
                ↓
            fill_context_for_report()
                ↓
            middleware.py 把 report=True
                ↓
            切换成 report_prompt.txt
                ↓
            fetch_external_data(user_id, month)
                ↓
            大模型生成报告
                ↓
            返回回答
"""
#ReactAgent初始化时创建了一个 LangChain Agent：
class ReactAgent:
    def __init__(
            self,
            memory_manager: MemoryManager | None = None,
            memory_extractor: MemoryExtractor | None = None,
    ):
        self.memory_manager = memory_manager or create_memory_manager(memory_conf)
        self.memory_extractor = memory_extractor
        if self.memory_manager.enabled and self.memory_extractor is None:
            self.memory_extractor = MemoryExtractor(chat_model)
        self.agent = create_agent(
            model=chat_model,#表示用哪个大模型
            system_prompt=load_system_prompts(),#表示加载主提示词，也就是告诉模型“你是扫地机器人客服”。
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
            context_schema=AgentRuntimeContext,
        )

    def execute_stream(self, query: str, user_id: str):
        """召回当前用户长期记忆，然后执行原有 ReAct Agent。"""
        recall_result = self.memory_manager.recall(
            user_id=user_id,
            query=query,
            include_reflection=True,
        )
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        runtime_context: AgentRuntimeContext = {
            "report": False,
            "memory_context": recall_result.to_prompt_context(),
        }
        #把用户问题交给Agent让Agent开始思考允许Agent调工具，最后流式返回结果
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context=runtime_context):
            latest_message = chunk["messages"][-1]
            # 前端只显示最终 AI 回答：隐藏工具消息、工具调用前文本和内部过程。
            if (
                    isinstance(latest_message, AIMessage)
                    and not latest_message.tool_calls
                    and isinstance(latest_message.content, str)
                    and latest_message.content.strip()
            ):
                yield latest_message.content.strip() + "\n"

        # 仅在 Agent 正常完成本轮回复后提取；不向 Hindsight 保存原始聊天。
        if self.memory_manager.enabled and self.memory_extractor is not None:
            memories = self.memory_extractor.extract(query)
            self.memory_manager.retain_many(user_id, memories)


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告", user_id="1001"):
        print(chunk, end="", flush=True)
