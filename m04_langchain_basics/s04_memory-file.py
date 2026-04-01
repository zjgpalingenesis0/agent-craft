from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory, FileChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from config import OPENAI_API_KEY, OPENAI_BASE_URL
from pathlib import Path



prompt = ChatPromptTemplate.from_messages([
    ("system", "你非常可爱，说话末尾会带个喵"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")  # {input}:占位符
])

llm = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

parser = StrOutputParser()

chain = prompt | llm | parser

# 创建 histories 目录
HISTORIES_DIR = Path("histories")
HISTORIES_DIR.mkdir(exist_ok=True)

# 存储所有会话历史(可用数据库替换)
# 此处用文件
store = {}

def get_session_history(session_id:str):
    """根据session_id获取该用户的聊天历史"""
    if session_id not in store:
        # 定义文件路径：histories目录下，以session_id为文件名
        path = HISTORIES_DIR / f"{session_id}.json"
        store[session_id] = FileChatMessageHistory(
            file_path=str(path),
            encoding="utf-8",
            ensure_ascii=False
        )
    return store[session_id]


# 包装成带记忆的Runnable
runnable_with_memory = RunnableWithMessageHistory(
    runnable=chain,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

session_id = 'user_123'
while 1:
    user_input = input("\n你:")
    if user_input=="quit":
        print('拜拜喵!')
        break
    response = runnable_with_memory.invoke(
        {"input":user_input},
        config={"configurable":{"session_id":session_id}}
    )
    print(f'AI:{response}')
