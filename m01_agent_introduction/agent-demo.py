# 让ai说一句话
from config import OPENAI_API_KEY
from langchain_openai import ChatOpenAI

# 配置deepseek
# llm = ChatOpenAI(
#     model="deepseek-chat",
#     api_key=OPENAI_API_KEY,
#     # 注:在.env里把OPENAI_API_KEY改成你自己的api-key即可
#     base_url="https://api.deepseek.com"
# )
# 配置千问
llm = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=OPENAI_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用模型
response = llm.invoke("你好呀,当前你用的是什么大模型呢")
print(response.content)