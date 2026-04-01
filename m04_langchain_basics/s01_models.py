from config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain_openai import ChatOpenAI

# 初始化模型
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     api_key=OPENAI_API_KEY,
#     base_url="https://api.deepseek.com"
# )
llm = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# 调用模型
response = llm.invoke('你好喵')
print(response.content)
