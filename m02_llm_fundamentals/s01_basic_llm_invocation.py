from config import OPENAI_API_KEY, OPENAI_BASE_URL
from openai import OpenAI

# client = OpenAI(
#     api_key=OPENAI_API_KEY,
#     base_url="https://api.deepseek.com")
#
# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"}, # 提示词角色
#         {"role": "user", "content": "Hello"}, # 用户输入的对话
#     ],
#     stream=False # 非流式输出, 只会等语句全部生成才返回
# )
#
# print(response.choices[0].message.content)

# 初始化llm实例
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)
response = client.chat.completions.create(
    model="qwen3.5-plus",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好，苹果用英文怎么说"},
    ],
    stream=False  # 流式输出
)
# print(response.choices[0].message.content)

# 流式输出的话，输出分块看一下
# for chunk in response:
#     if (chunk.choices[0].delta.content is not None):
#         print(chunk.choices[0].delta.content + "  part", end="", flush=True)