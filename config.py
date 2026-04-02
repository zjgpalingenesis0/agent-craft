import os
from dotenv import load_dotenv

# 1. 加载.env文件
load_dotenv()

# 2. 解决 UUID v7 警告
try:
    from langsmith import uuid7
    import uuid
    uuid.uuid4 = uuid7 # 全局替换
except ImportError:
    pass # 如果没装 langsmith，省略


# 3. 解决TensorFlow & Numpy 兼容性告警
def silence_framework_warnings():
    import os
    import warnings
    import logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    warnings.filterwarnings('ignore')
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

silence_framework_warnings()




# 4. 获取API_KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
AMAP_MAPS_API_KEY = os.getenv("AMAP_MAPS_API_KEY")
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("× 请在.env中设置OPENAI_API_KEY")
if not LANGCHAIN_API_KEY:
    raise ValueError("× 请在.env中设置LANGCHAIN_API_KEY")
if not AMAP_MAPS_API_KEY:
    raise ValueError("× 请在.env中设置AMAP_MAPS_API_KEY")
if not CHATGPT_API_KEY:
    raise ValueError("× 请在.env中设置CHATGPT_API_KEY")

# 新增：获取访问地址
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
if not OPENAI_BASE_URL:
    raise ValueError("× 请在.env中设置OPENAI_BASE_URL")
DASHSCOPE_EMBEDDING_API_KEY = os.getenv("DASHSCOPE_EMBEDDING_API_KEY")
if not DASHSCOPE_EMBEDDING_API_KEY:
    raise ValueError("× 请在.env中设置DASHSCOPE_EMBEDDING_API_KEY")
# 5. MySQL 数据库配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "chat_db")
MYSQL_TABLE_NAME = os.getenv("MYSQL_TABLE_NAME", "chat_history")

# 构建 MySQL 连接字符串
MYSQL_CONNECTION_STRING = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
)