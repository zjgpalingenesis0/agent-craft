"""
流式输出运行器 V3 - 使用 app.stream() 获取真正的流式输出

与 v1 版本的区别：
- v1 使用 astream_events（监听事件流，但会遗漏第二次 LLM 调用）
- v3 使用 astream（直接监听节点输出，捕获所有内容）

让用户看到：
  1. AI 的决策过程（是否调用工具）
  2. 工具调用状态
  3. 最终回答（带打字机效果）
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage


async def run_agent_with_streaming(app, query: str):
    """
    流式运行器 - 实时显示 AI Agent 的完整思考过程

    :param app: 编译好的 LangGraph 应用 (workflow.compile())
    :param query: 用户输入的问题
    """
    print(f'\n用户:{query}\n')

    # 构造输入消息
    inputs = {"messages": [HumanMessage(content=query)]}

    # === 核心逻辑：使用 astream 而不是 astream_events ===
    # astream 会按顺序返回每个节点的完整输出
    async for event in app.astream(inputs, stream_mode="updates"):

        # event 格式: {节点名: {输出数据}}
        for node_name, node_output in event.items():

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 节点1: Agent（LLM 思考节点）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            if node_name == "agent":
                messages = node_output.get("messages", [])

                for msg in messages:
                    if isinstance(msg, AIMessage):

                        # 情况1: LLM 决定调用工具
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            # tool_calls 可能是对象或字典，兼容处理
                            tool_names = []
                            for tc in msg.tool_calls:
                                if isinstance(tc, dict):
                                    tool_names.append(tc.get('name', 'unknown'))
                                else:
                                    tool_names.append(getattr(tc, 'name', 'unknown'))
                            print(f"🤖 AI: 我需要调用工具来查询信息: {', '.join(tool_names)}\n")

                        # 情况2: LLM 生成最终回答
                        elif hasattr(msg, 'content') and msg.content:
                            print("🤖 AI: ", end="", flush=True)

                            # 打字机效果：逐字输出
                            for char in msg.content:
                                print(char, end="", flush=True)
                                # 可选：添加延迟让打字效果更明显
                                # await asyncio.sleep(0.005)

                            print()  # 换行

            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 节点2: Tools（工具执行节点）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            elif node_name == "tools":
                messages = node_output.get("messages", [])

                for msg in messages:
                    # ToolMessage 有 name 属性表示工具名称
                    if hasattr(msg, 'name'):
                        print(f"✅ 工具 [{msg.name}] 执行完成\n")

    print("\n😊 输出结束!")