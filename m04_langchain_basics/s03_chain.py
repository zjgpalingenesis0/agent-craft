from typing import List

from pydantic import Field
from sympy import re

from config import OPENAI_API_KEY, OPENAI_BASE_URL
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, BaseOutputParser


class PetPhraseCounterParser(BaseOutputParser[dict]):
    """统计各种语气词的出现次数"""
    # 类变量
    keywords: list[str] = Field(default_factory=lambda: ['喵', '哦', '呢', '吧', '呀'])

    def parse(self, text: str) -> dict:
        clean_text = text.strip()

        # 统计每个关键词
        counts = {
            keyword: clean_text.count(keyword)
            for keyword in self.keywords
        }

        return {
            "text": clean_text,
            "total_counts": sum(counts.values()),
            "details": counts
        }

    @property
    def _type(self) -> str:
        return "pet_phrase_counter_parser"

# 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你非常可爱，说话末尾会带个喵"),
    ("human", "{input}")  # {input}:占位符
])

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

# 定义解析器，把LLM返回的AIMessage转成字符串
# parser = StrOutputParser()
parser = PetPhraseCounterParser(keywords=['喵', '呀'])
# 组成Chain
# 链式调用  输入数据 → Prompt → LLM → Parser → 输出字符串
chain = prompt | llm | parser
# chain = prompt | llm
# 最终调用
result = chain.invoke({"input": "你好呀"})
print(result)
