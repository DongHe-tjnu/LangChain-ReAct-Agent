from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompts#从 prompt_loader.py 里导入 load_system_prompts()。这个函数会读取 main_prompt.txt。
from agent.tools.agent_tools import (rag_summarize, get_weather, get_user_location, get_user_id,
                                     get_current_month, fetch_external_data, fill_context_for_report)
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
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
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,#表示用哪个大模型
            system_prompt=load_system_prompts(),#表示加载主提示词，也就是告诉模型“你是扫地机器人客服”。
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {
            "messages": [
                {"role": "user", "content": query},
            ]
        }
        #把用户问题交给Agent让Agent开始思考允许Agent调工具，最后流式返回结果
        # 第三个参数context就是上下文runtime中的信息，就是我们做提示词切换的标记
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"


if __name__ == '__main__':
    agent = ReactAgent()

    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
