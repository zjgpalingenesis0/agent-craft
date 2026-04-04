"""
一个流式输出运行器，用于实时显示 AI Agent 的思考和执行过程
让用户看到：
  1. AI 正在说什么（逐字输出）
  2. AI 什么时候调用工具
  3. 工具调用完成后 AI 继续思考
"""

import asyncio
from langchain_core.messages import HumanMessage


async def run_agent_with_streaming(app, query: str):
    """
    通用流式运行器，负责将 LangGraph 的运行过程可视化输出到控制台
    :param app: 编译好的 LangGraph 应用 (workflow.compile())
    :param query: 用户输入的问题
    """
    print(f'\n用户:{query}\n')
    print("🤖 AI:", end="", flush=True)

    # 构造输入消息
    inputs = {"messages": [HumanMessage(content=query)]}

    # 核心:监听v2版本的事件流(相比v1更全面)
    """
    - async for = 异步迭代器，一边接收事件一边处理（不等全部结束）
    - astream_events = LangGraph 的流式事件 API
    - 相当于：实时监听 AI 的大脑活动，而不是等 AI 完全想完才显示
    """
    async for event in app.astream_events(inputs, version="v2"):
        kind = event["event"]

        # 1.监听LLM的流式吐字(嘴在动)
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            # 过滤掉空的chunk(有时工具调用会产生空内容)
            if chunk.content:
                print(chunk.content, end="", flush=True)

        # 2.监听工具开始调用(手在动)
        elif kind == "on_tool_start":
            tool_name = event["name"]
            # 不打印内部包装，只打印自定义的工具
            if not tool_name.startswith("_"):
                print(f"\n\n🔨 正在调用工具: {tool_name} ...")

        # 3.监听工具调用结束(拿到结果)
        elif kind == "on_tool_end":
            tool_name = event["name"]
            if not tool_name.startswith("_"):
                print(f"✅ 调用完成，继续思考...\n")
                print("🤖 AI: ", end="", flush=True)

    # 获取最终状态（流式输出可能遗漏最后的 LLM 响应）
    final_state = await app.ainvoke(inputs)
    final_message = final_state["messages"][-1]

    # 补充显示最终回答（如果流式输出没捕获到）
    from langchain_core.messages import AIMessage
    if isinstance(final_message, AIMessage) and final_message.content:
        print(final_message.content)

    print("\n\n😊 输出结束!")