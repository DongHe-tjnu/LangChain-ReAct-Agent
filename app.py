import time

import streamlit as st
from agent.react_agent import ReactAgent
#app.py 本身不负责智能回答，它只是页面入口。真正干活的是ReactAgent
# 用户输入提示词prompt = st.chat_input()后
#   ↓
# app.py调用res_stream = st.session_state["agent"].execute_stream(prompt)
#   ↓
# ReactAgent.execute_stream(prompt)
# 标题
"""
启动网页
  ↓
初始化 ReactAgent
  ↓
等待用户输入
  ↓
用户输入问题
  ↓
调用 ReactAgent.execute_stream(prompt)
  ↓
接收 Agent 回答
  ↓
显示到网页


1. app.py
   用户网页输入

2. react_agent.py
   创建 Agent，接收用户问题

3. factory.py
   提供 chat_model

4. prompt_loader.py
   加载 main_prompt

5. agent_tools.py
   提供工具

6. 如果是普通知识问题：
   agent_tools.rag_summarize()
      ↓
   rag_service.RagSummarizeService.rag_summarize()
      ↓
   vector_store.get_retriever()
      ↓
   Chroma 查知识库
      ↓
   rag_service 调模型总结

7. 如果是天气问题：
   get_user_location()
      ↓
   get_weather(city)
      ↓
   模型生成建议

8. 如果是报告问题：
   get_user_id()
      ↓
   get_current_month()
      ↓
   fill_context_for_report()
      ↓
   middleware.py 设置 report=True
      ↓
   report_prompt_switch() 切换 report_prompt
      ↓
   fetch_external_data()
      ↓
   模型生成报告
"""
st.title("智扫通机器人智能客服")
st.divider()

# Demo 项目尚未接入登录系统，因此由界面提供稳定用户 ID。
# 生产环境应替换为认证系统中的真实用户唯一标识。
memory_user_id = st.sidebar.text_input(
    "Memory 用户 ID",
    value="1001",
    help="不同用户 ID 使用不同的 Hindsight Memory Bank。",
).strip()

if "agent" not in st.session_state:
    st.session_state["agent"] = ReactAgent()
# 如果页面里还没有聊天记录，就创建一个空列表。后面用户说的话、机器人回答的话都会存到这里。
if "message" not in st.session_state:
    st.session_state["message"] = []
#role可能是：user、assistant，把历史消息重新显示到页面上
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 用户输入提示词
prompt = st.chat_input()

if prompt:
    if not memory_user_id:
        st.warning("请输入 Memory 用户 ID 后再发送消息。")
        st.stop()

    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    response_messages = []#创建列表，用来接收agent流式返回的内容
    with st.spinner("智能客服思考中..."):
        res_stream = st.session_state["agent"].execute_stream(prompt, user_id=memory_user_id)
        #流式输出，定义一个函数用来捕获 agent返回的流式内容
        def capture(generator, cache_list):     # 捕获

            for chunk in generator:#agent每返回一小段内容，就加到response_messages中
                cache_list.append(chunk)

                for char in chunk:#每个字慢慢输出
                    time.sleep(0.01)
                    yield char
        #把agent的流式回答显示到网页上
        st.chat_message("assistant").write_stream(capture(res_stream, response_messages))
        #把机器人最后一段回复保存起来
        st.session_state["message"].append({"role": "assistant", "content": response_messages[-1]})
        st.rerun()
